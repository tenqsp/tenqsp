# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""What the library made of a model, in plain language: ``m.explain()`` and
``m.diagnostics()``.

``explain()`` reads the model description (no solve) and says, term by term, what each part
means and which numerical scheme carries it:

- transport with the ``erlang`` scheme is a chain of transit compartments: how many, the rate
  from one to the next (speed / cell width), the mean transit time and the CV; ``sharp``
  keeps fronts and costs more solver work;
- transport plus diffusion on one axis is a Fokker–Planck equation: the single-cell process
  it describes;
- a transition acts on the cells themselves (switching) or redistributes a production
  (daughters carrying a change, at the parent's rate);
- boundaries, starting values (and what starts at 0), doses, extinction groups, and every
  numerical safeguard (there are no hidden clamps or caps);
- a gain that reads another place's value: if it is material moving between places, a flux
  checks its balance.

``diagnostics(sol=None)`` gives accuracy guidance: per transported axis the cell
Péclet number (transport against diffusion), the time to cross one cell, and the Erlang CV;
with a solution also undershoot below ``atol`` followed by regrowth, negative
values, solver retries and how well the ledger closed. It returns a
``Report``.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import re

import numpy as np
import sympy as sp
from sympy.printing.str import StrPrinter

from .canonical import _same
from .expr import SHAPES, FormulaError
from .expr import parse as _parse_formula
from .ir import ModelIR
from .lower import tables
from .validate import Issue, Report, validate
from .writer import _p

__all__ = ["explain", "diagnostics", "diffusion_slope"]

_AUTO_INTEGRAL = re.compile(r"int_[A-Za-z0-9_]+")


def _label(name: str, where) -> str:
    if not where:
        return name
    return f"{name}[" + ", ".join(f"{a}={lab}" for a, lab in where) + "]"


def _num(v: float) -> str:
    return f"{v:.3g}"


def _text(e: sp.Expr) -> str:
    """A SymPy expression as formula text (``^`` for powers)."""
    e = sp.sympify(e)
    e = e.xreplace({f: sp.Float(float(f"{float(f):.6g}")) for f in e.atoms(sp.Float)})
    return sp.sstr(e, full_prec=False).replace("**", "^")


def parse(text: str, names) -> sp.Expr:
    """A formula; one that cannot be read (the model has errors, listed by explain()) reads
    as an unknown symbol, so the rest can still be explained."""
    try:
        return _parse_formula(text, names)
    except (FormulaError, SyntaxError, TypeError, ValueError):
        return sp.Symbol("unreadable")


class _Model:
    """The IR with the lookups explanations need."""

    def __init__(self, ir: ModelIR):
        self.ir = ir
        self.symbols = dict(tables(ir).symbols)
        self.axes = {a.name: a for a in ir.axes}
        self.fields = {f.name: f for f in ir.fields}
        self.comps = {c.name: c for c in ir.compartments}
        self.defs = {d.name: d.expr for d in ir.definitions}
        self.ints = {i.name: i for i in ir.integrals}
        self.scalars = {
            sp.Symbol(p.name, real=True): sp.Float(float(p.value))
            for p in ir.params
            if p.over is None and p.value is not None
        }

    def expr(self, text: str, _seen: frozenset = frozenset()) -> sp.Expr:
        """``text`` with named formulas written out (integrals stay names)."""
        e = parse(text, self.symbols)
        subs = {}
        for s in e.free_symbols:
            if isinstance(s, sp.Symbol) and s.name in self.defs and s.name not in _seen:
                subs[s] = self.expr(self.defs[s.name], _seen | {s.name})
        return e.xreplace(subs) if subs else e

    def reads(self, text: str, _seen: frozenset = frozenset()) -> set[str]:
        """Fields and compartments a formula reads, through named formulas and totals."""
        out: set[str] = set()
        for s in parse(text, self.symbols).free_symbols:
            n = getattr(s, "name", str(s))
            if n in self.fields or n in self.comps:
                out.add(n)
            elif n in self.defs and n not in _seen:
                out |= self.reads(self.defs[n], _seen | {n})
            elif n in self.ints and n not in _seen:
                out |= self.reads(self.ints[n].integrand, _seen | {n})
        for b in parse(text, self.symbols).atoms(sp.Indexed):
            n = b.base.name
            if n in self.defs and n not in _seen:
                out |= self.reads(self.defs[n], _seen | {n})
        return out

    def varies_along(self, text: str, axis: str, _seen: frozenset = frozenset()) -> bool:
        """Whether a formula can take different values in different cells of ``axis``."""
        e = parse(text, self.symbols)
        for s in e.free_symbols:
            n = getattr(s, "name", str(s))
            if n == axis:
                return True
            if n in self.fields and axis in self.fields[n].axes:
                return True
            if n in self.defs and n not in _seen:
                if self.varies_along(self.defs[n], axis, _seen | {n}):
                    return True
            if n in self.ints and n not in _seen:
                i = self.ints[n]
                if axis not in i.over and self.varies_along(i.integrand, axis, _seen | {n}):
                    return True
        if any(i.base.name == axis or axis in self._index_axes(i) for i in e.atoms(sp.Indexed)):
            return True
        return False

    def _index_axes(self, ind: sp.Indexed) -> set[str]:
        return {str(x) for x in ind.indices}

    def changes(self, text: str) -> list[str]:
        """What a formula's value changes with, besides parameters: time and model state."""
        e = self.expr(text)
        names = {getattr(s, "name", str(s)) for s in e.free_symbols}
        out = []
        if "t" in names:
            out.append("time")
        state = sorted(self.reads(text))
        if state:
            out.append("the state (" + ", ".join(state) + ")")
        return out

    def values(self, e: sp.Expr) -> list[float] | None:
        """Every value ``e`` takes over the entries of the parameter tables it uses (a few
        tables at most), if it has no other unknowns."""
        tables = {p.name: p for p in self.ir.params if p.over is not None and p.value is not None}
        ind = sorted(e.atoms(sp.Indexed), key=str)
        if not ind or any(i.base.name not in tables for i in ind):
            return None
        bases = sorted({i.base.name for i in ind})
        sizes = [len(tables[b].value) for b in bases]
        if math.prod(sizes) > 1000:
            return None
        out = []
        for combo in itertools.product(*(tables[b].value for b in bases)):
            pick = dict(zip(bases, combo, strict=True))
            v = self.value(e.xreplace({i: sp.Float(pick[i.base.name]) for i in ind}))
            if v is None:
                return None
            out.append(v)
        return out

    def value(self, e: sp.Expr) -> float | None:
        """A number, if ``e`` has only scalar parameters."""
        v = e.xreplace(self.scalars)
        try:
            return float(v) if v.is_number else None
        except (TypeError, ValueError):
            return None


def _jumps(e: sp.Expr) -> bool:
    """Whether a formula has a conditional or a non-smooth function (abs, max, min, floor,
    ceil, rem), whose derivative SymPy would give without the jump's delta."""
    rough = (sp.Piecewise, sp.Abs, sp.Max, sp.Min, sp.floor, sp.ceiling, sp.Heaviside)
    if any(e.has(c) for c in rough):
        return True
    return any(type(f).__name__ == "Rem" for f in e.atoms(sp.Function))


def _jumps_along(M: _Model, e: sp.Expr, axis: str) -> bool:
    """Whether a conditional or non-smooth function in ``e`` changes along ``axis`` (one in
    time or a parameter only does not make the slope along the axis jump)."""
    for sub in sp.preorder_traversal(e):
        if _jumps(sub) and not any(_jumps(a) for a in sub.args):  # the innermost rough part
            for x in sub.free_symbols:
                n = getattr(x, "name", str(x))
                if n == axis or (n in M.fields and axis in M.fields[n].axes):
                    return True
                if n in M.ints and M.varies_along(n, axis):
                    return True
    return False


class _ExactPrinter(StrPrinter):
    """Formula text that reads back to the same numbers (floats in full, pi as a number)."""

    def _print_Float(self, expr):
        return repr(float(expr))

    def _print_Pi(self, expr):
        return repr(math.pi)

    def _print_Exp1(self, expr):
        return "exp(1)"


def diffusion_slope(ir: ModelIR, D: str, axis: str) -> str:
    """``dD/d(axis)`` as formula text, for the Fokker–Planck drift correction. Refused (with
    the reason) when D varies along the axis through model state, or SymPy's result cannot
    be written back as a formula."""
    from .expr import Grad, grads_used

    M = _Model(ir)
    if axis not in M.symbols:
        raise ValueError(f"fokker_planck: no axis named '{axis}'.")
    try:
        _parse_formula(D, M.symbols)
    except (FormulaError, SyntaxError, TypeError, ValueError) as err:
        raise ValueError(f"fokker_planck: cannot read D = {D}: {err}") from None
    e = M.expr(D)
    for s in e.free_symbols:
        n = getattr(s, "name", str(s))
        if (n in M.fields and axis in M.fields[n].axes) or (
            n in M.ints and M.varies_along(n, axis)
        ):
            raise ValueError(
                f"fokker_planck: D = {D} varies along '{axis}' through '{n}'. Write the two "
                f"terms yourself: m.advect(F, axis={axis}, velocity=drift - (slope of D)) with "
                f"the slope from grad({n}, {axis}), and m.diffuse(F, axis={axis}, D=D)."
            )
    if grads_used(e) or e.atoms(Grad):
        raise ValueError(f"fokker_planck: D = {D} uses grad; write the two terms yourself.")
    if _jumps_along(M, e, axis):
        raise ValueError(
            f"fokker_planck: D = {D} jumps or has a kink along '{axis}' (a conditional or "
            "a non-smooth function); its slope is not a formula there. Give a smooth D (e.g. "
            "with sigmoid), or write the two terms yourself."
        )
    slope = sp.simplify(sp.diff(e, M.symbols[axis]))
    if slope.is_zero:
        return "0"
    text = _ExactPrinter().doprint(slope).replace("**", "^")
    try:
        ok = _same(M.expr(text), slope)
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        raise ValueError(
            f"fokker_planck: the slope of D = {D} along '{axis}' ({slope}) cannot be written "
            "as a formula; give the transport and diffusion terms yourself (m.advect with "
            "velocity drift - dD/dz, and m.diffuse)."
        )
    return text


# ------------------------------------------------------------------------------ explain


def explain(ir: ModelIR) -> str:
    """The model in plain language: every term with its scheme, boundaries, starting values,
    doses, extinction groups and numerical safeguards."""
    M = _Model(ir)
    report = validate(ir, raise_on_error=False)
    out: list[str] = []
    out += _size(ir)
    if report.errors:
        out.append("")
        out.append("ERRORS (the model cannot be solved until they are fixed):")
        out += [f"  - {i.where}: {i.message}" for i in report.errors]
    out.append("")
    out.append("Equations")
    for f in ir.fields:
        out += _field(M, f.name)
    for c in ir.compartments:
        out += _compartment(M, c.name)
    if ir.transfers:
        out.append("")
        out.append("Fluxes between places")
        out += _transfers(M)
    shared, lone = _shared_terms(M)
    if shared:
        out.append("")
        out.append("Terms shared between equations (the same term, matched by TenQSP)")
        out += shared
    if lone:
        out.append("")
        out.append("Terms that read another place but appear in one equation only")
        out += lone
    out.append("")
    out.append("Boundaries")
    out += _boundaries(M)
    out.append("")
    out.append("Starting values")
    out += _starts(M)
    if ir.doses:
        out.append("")
        out.append("Doses")
        field_axes = {f.name: f.axes for f in ir.fields}
        for d in ir.doses:
            amounts = d.amounts[0] if len(set(d.amounts)) == 1 else ", ".join(d.amounts)
            times = ", ".join(f"{t:g}" for t in d.times)
            free = [a for a in field_axes.get(d.target, ()) if a not in dict(d.where)]
            if d.shape is not None:
                how = f" (each amount spread over {', '.join(free)} as {d.shape})"
            elif free:
                how = f" (a density over {', '.join(free)} added to the field)"
            else:
                how = ""
            out.append(f"  {_label(d.target, d.where)}: {amounts} at t = {times}{how}")
        out.append(
            "  Each dose is added at once (a bolus) and the solver restarts there with a "
            "fresh step size."
        )
    if ir.lags or ir.bioavailabilities:
        out.append("")
        out.append("Lags and bioavailability (every dose into the place: the model's own and a")
        out.append("regimen's, boluses and infusions)")
        lags = {x.target: x.value for x in ir.lags}
        fracs = {x.target: x.value for x in ir.bioavailabilities}
        for target in dict.fromkeys([*lags, *fracs]):
            parts = []
            if target in lags:
                parts.append(f"arrive {lags[target]} later")
            if target in fracs:
                parts.append(
                    f"{fracs[target]} of each arrives (the rest is booked as not absorbed)"
                )
            out.append(f"  doses into {target}: " + ", and ".join(parts) + ".")
        if ir.lags:
            out.append(
                "  A lagged infusion is shifted whole (start and stop); a dose arriving after the "
                "end of the run is not given."
            )
    if ir.extinctions:
        out.append("")
        out.append("Extinction groups")
        out += _extinctions(ir)
    out.append("")
    out.append("Numerical safeguards (the complete list)")
    out += _safeguards(ir, report)
    others = [
        i
        for i in report.issues
        if i.level == "warning"
        or (
            i.level == "note"
            and not re.search(r"(lower|upper) end:|no initial condition given", i.message)
        )
    ]
    if others:
        out.append("")
        out.append("Checks")
        out += [f"  [{i.level}] {i.where}: {i.message}" for i in others]
    return "\n".join(out) + "\n"


def _size(ir: ModelIR) -> list[str]:
    counts = ir.state_count()
    sizes = {
        a.name: (len(a.faces) - 1 if a.type == "continuous" else len(a.labels)) for a in ir.axes
    }
    total = sum(counts.values())
    lines = [f"Model '{ir.name}': {total:,} state variables"]
    for f in ir.fields:
        prod = " x ".join(f"{sizes[a]} ({a})" for a in f.axes)
        lines.append(f"  field {f.name}: {counts[f.name]:,} = {prod}")
    if ir.compartments:
        names = ", ".join(c.name for c in ir.compartments)
        lines.append(f"  compartments: {len(ir.compartments)} ({names})")
    declared = [p.name for p in ir.params if p.value is None]
    if declared:
        lines += _wrap(
            f"Parameters without a value in the model (given when it is solved or simulated; "
            f"numbers that depend on them are not shown below): {', '.join(declared)}.",
            "  ",
        )
    if total > 100_000:
        lines.append(
            "  WARNING: above the 100,000-state target (ADR 0003): consider fewer cells, a "
            "separate field for structures that exist only at some sites, or a "
            "reachable-genotype axis."
        )
    return lines


def _sign_at_typical(e: sp.Expr) -> float:
    """Sign of a coefficient with every name at 0.5 (parameters are mostly positive): which
    way round to show a shared term."""
    stand = {x: sp.Float(0.5) for x in (*e.atoms(sp.Indexed), *e.free_symbols)}
    try:
        return float(np.sign(complex(e.xreplace(stand)).real))
    except (TypeError, ValueError):
        return 1.0


def _times(r: sp.Expr) -> str:
    """A place's share of a shared term: +1, -1, or a factor (a conversion, e.g. b phage
    released per lysed cell)."""
    if r == 1:
        return "+1"
    if r == -1:
        return "-1"
    return f"x ({_text(r)})"


def _signed(expr: str) -> str:
    """A term with its sign in front: ``+ k*B``, ``- infect`` (not ``+ -infect``)."""
    e = expr.strip()
    if e.startswith("-") and not e.startswith("--"):
        return "- " + e[1:].strip()
    return "+ " + e


def _wrap(text: str, indent: str, width: int = 92) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width, initial_indent=indent, subsequent_indent=indent + "  ")


def _field(M: _Model, name: str) -> list[str]:
    ir = M.ir
    f = M.fields[name]
    out = [f"  {name} (over {', '.join(f.axes)})" + (f", weights {f.weights}" if f.weights else "")]
    terms = [t for t in ir.terms if t.target == name]
    for t in terms:
        part = _label(name, t.where)
        if t.kind == "reaction":
            out += _wrap(f"{part}: {_signed(t.expr)} — a gain/loss per cell, as written.", "    ")
        elif t.kind == "hazard":
            out += _wrap(
                f"{part}: - {_p(t.expr)}*{name} — a loss at the per-capita rate {t.expr} "
                "(its own ledger line).",
                "    ",
            )
        elif t.kind == "switch":
            out += _wrap(
                f"{part}: trans_{_axes_text(t.axis)}({t.transitions}, {name}) — the cells "
                f"themselves change category (first-order switching: rate x density of the "
                f"origin); moves {_moves(M, t.transitions)}. What leaves a category is "
                "derived, so nothing is created or lost.",
                "    ",
            )
        elif t.kind == "mutate":
            out += _wrap(
                f"{part}: trans_{_axes_text(t.axis)}({t.transitions}, {t.expr}) — redistributes "
                f"the production {t.expr} among the categories at the parent's rate (daughters "
                f"carrying a change); moves {_moves(M, t.transitions)}. The production itself is "
                "the reaction term; this only moves the daughters.",
                "    ",
            )
    for ax in f.axes:
        adv = [t for t in terms if t.kind == "advection" and t.axis == ax]
        dif = [t for t in terms if t.kind == "diffusion" and t.axis == ax]
        if len(adv) > 1 and len({t.scheme for t in adv}) == 1:  # they add up to one velocity
            total = " + ".join(_p(t.expr) for t in adv)
            out += _wrap(f"{len(adv)} transport terms along {ax}; together:", "    ")
            adv = [dataclasses.replace(adv[0], expr=_text(sp.simplify(M.expr(total))))]
        if len(adv) > 1:
            out += _wrap(
                f"{len(adv)} transport terms along {ax} with different schemes ("
                + ", ".join(f"velocity {t.expr} ({t.scheme})" for t in adv)
                + "): each moves at its own velocity; a transit chain describes neither alone.",
                "    ",
            )
        else:
            for t in adv:
                out += _wrap(_transport(M, name, t), "    ")
        for t in dif:
            out += _wrap(
                f"diffusion along {ax}, D = {t.expr}: second-order central differences in "
                "flux form; where D varies, the face value is the harmonic mean of the two "
                "cells (a cell with D = 0 blocks the flux).",
                "    ",
            )
        if adv and dif:
            out += _wrap(_fokker_planck(M, ax, adv, dif), "    ")
    if not terms:
        out.append("    no terms: the field stays as it starts (doses aside).")
    return out


def _axes_text(axis) -> str:
    return axis if isinstance(axis, str) else "{" + ",".join(axis) + "}"


def _moves(M: _Model, name: str) -> str:
    tr = next(x for x in M.ir.transitions if x.name == name)

    def cat(c) -> str:
        return "(" + ", ".join(c) + ")" if isinstance(c, tuple) else str(c)

    return ", ".join(f"{cat(a)} -> {cat(b)} at {r}" for a, b, r in tr.rates)


def _transport(M: _Model, name: str, t) -> str:
    ax = M.axes[t.axis]
    faces = np.asarray(ax.faces, dtype=float)
    widths = np.diff(faces)
    n = len(widths)
    uniform = np.allclose(widths, widths[0], rtol=1e-12, atol=0)
    v = t.expr.strip()
    head = f"transport along {t.axis}, velocity {v} ({t.scheme})"
    if t.scheme == "sharp":
        return (
            f"{head}: MUSCL with the van Leer limiter: a straight line in each cell, its slope "
            "limited so no new peaks or negative values appear. Fronts keep their shape and "
            "refining the grid converges to transport without spread. It costs more solver "
            "work than erlang (about 10x on the toy phage model): the solver follows the front "
            "from cell to cell."
        )
    along = M.varies_along(v, t.axis)
    changes = M.changes(v)
    if along:
        return (
            f"{head}: first-order upwind: each face takes the value from the side the flow "
            f"comes from, so values stay positive. The velocity varies along {t.axis} (it may "
            "change sign), so this is not a fixed transit chain: each cell passes to its "
            "neighbour at velocity / cell width. Upwinding adds a numerical spread of about "
            f"|v| x width / 2 to any diffusion (see diagnostics())."
        )
    width = _nice(widths[0]) if uniform else None
    vel = parse(v, M.symbols)  # written with the modeller's names (ktr, not 4/mtt) ...
    full = M.expr(v)  # ... and with named formulas written out, for the numbers
    sign = M.value(full)
    if sign is None and (vals := M.values(full)):  # a velocity per category
        if min(vals) < 0 < max(vals) or 0 in vals:
            return (
                f"{head}: first-order upwind; the velocity differs in sign between categories "
                f"(from {_num(min(vals))} to {_num(max(vals))}), so each category moves its own "
                "way at velocity / cell width."
            )
        sign = min(vals) if max(vals) < 0 else max(vals)
    if sign is not None and sign == 0:
        return f"{head}: the velocity is 0, so nothing moves along {t.axis}."
    towards = ""
    if sign is not None and sign < 0:  # runs towards the lower end at speed |v|
        vel, full = -vel, -full
        towards = f", running towards the lower end of {t.axis} (the velocity is negative)"
    lines = [head + f": first-order upwind on {n} cells{towards}"]
    if uniform and changes:
        rate = sp.simplify(vel / width)
        lines.append(
            f", a chain of {n} compartments passing to the next at rate |velocity|/width = "
            f"{_text(rate)}, which changes with {' and '.join(changes)}. The mean transit "
            f"time ({_text(_nice(faces[-1] - faces[0]))}/|velocity|) and CV 1/sqrt({n}) of a "
            "chain hold only while the velocity stays the same."
        )
        changes = []
    elif uniform:
        rate = sp.simplify(vel / width)
        length = _nice(faces[-1] - faces[0])
        mtt = sp.simplify(length / vel)
        rate_n, mtt_n = M.value(full / width), M.value(length / full)
        rate_txt = _text(rate) + (
            f" (= {_num(rate_n)})" if rate_n is not None and not rate.is_number else ""
        )
        mtt_txt = _text(mtt) + (
            f" (= {_num(mtt_n)})" if mtt_n is not None and not mtt.is_number else ""
        )
        cells = (
            "is one transit compartment (density = amount)"
            if width == 1
            else (
                f"of width {_text(width)} is one transit compartment "
                f"(amount = density x {_text(width)})"
            )
        )
        exit_side = "lower" if towards else "upper"
        leaves = any(
            b.target == name and b.axis == t.axis and b.side == exit_side and b.kind == "outflow"
            for b in M.ir.boundaries
        )
        if not leaves:
            lines.append(
                f", a chain of {n} transit compartments (each cell {cells}) passing to the next "
                f"at rate |velocity|/width = {rate_txt}; but the {exit_side} end of {t.axis} is "
                "closed, so nothing leaves the chain: material collects in the last cell (add "
                f"'at {t.axis}={exit_side}: outflow' under the equation of {name} for a transit "
                "chain with an exit)."
            )
        else:
            lines.append(
                f", which is exactly a chain of {n} transit compartments: each cell {cells}, "
                f"passing to the next at rate |velocity|/width = {rate_txt}. Mean transit time "
                f"{mtt_txt}, CV 1/sqrt({n}) = {1 / math.sqrt(n):.3g}."
            )
    else:
        lines.append(
            f" of unequal widths ({_num(widths.min())} to {_num(widths.max())}): a chain of "
            f"{n} compartments, each passing to the next at velocity / its own width."
        )
    if changes:
        lines.append(f" The rate changes with {' and '.join(changes)}.")
    lines.append(
        " The spread of this chain is the model's variability, so refining the grid changes "
        "the model; it is not a convergence parameter (use sharp for transport without spread)."
    )
    return "".join(lines)


def _nice(x: float) -> sp.Expr:
    """A width or length as a short fraction when it is one (1/3), else a float."""
    r = sp.Rational(float(x)).limit_denominator(1000)
    if abs(float(r) - float(x)) <= 1e-12 * max(abs(float(x)), 1e-300):
        return r
    return sp.Float(float(f"{float(x):.6g}"))


def _fokker_planck(M: _Model, ax: str, adv, dif) -> str:
    v = " + ".join(t.expr for t in adv)
    D = " + ".join(t.expr for t in dif)
    D_text = D.replace("**", "^")
    if _jumps_along(M, M.expr(D), ax):
        return (
            f"Transport and diffusion along {ax}: D = {D_text} jumps or has a kink along {ax}, "
            "so this is not the density of a single SDE across the jump (an SDE's density "
            "jumps where D does; this equation keeps the value continuous)."
        )
    try:
        dD = sp.diff(M.expr(D), M.symbols[ax])
        state = M.reads(D) and any(ax in M.fields[r].axes for r in M.reads(D) if r in M.fields)
        slope = "the slope of D along " + ax if state else _text(sp.simplify(dD))
    except Exception:  # noqa: BLE001 (a formula SymPy cannot differentiate)
        slope = "the slope of D along " + ax
    if slope == "0":
        drift = v
    elif slope.startswith("the slope"):
        drift = f"{_p(v)} + {slope}"
    else:
        total = sp.simplify(M.expr(v) + dD)
        drift = _text(total) if not _jumps(total) else f"{_p(v)} + {_p(slope)}"
    return (
        f"Together, transport and diffusion along {ax} are a Fokker–Planck equation: single "
        f"cells follow d{ax} = ({drift}) dt + sqrt(2 D) dW, with D = {D_text} (Itô). "
        "tq.Model.fokker_planck writes this from the drift and D."
    )


def _compartment(M: _Model, name: str) -> list[str]:
    c = M.comps[name]
    rates = [t.expr for t in M.ir.terms if t.target == name and t.kind == "rate"]
    head = f"  {name} (a compartment" + (f", volume {c.volume})" if c.volume else ")")
    if not rates:
        moved = any(name in (tr.source, tr.to) for tr in M.ir.transfers) or any(
            b.to == name for b in M.ir.boundaries
        )
        if moved:
            return [
                head,
                "    no rate of its own: changed only by the fluxes and boundaries below.",
            ]
        return [head, "    no rate: stays as it starts (doses aside)."]
    return [head] + _wrap(f"d_t({name}) = {' + '.join(rates)} — as written.", "    ")


def _transfers(M: _Model) -> list[str]:
    out = []
    for tr in M.ir.transfers:
        src = _label(tr.source, tr.source_where)
        if tr.to is None:
            size = (
                M.fields[tr.source].weights if tr.source in M.fields else M.comps[tr.source].volume
            )
            what = "a clearance (volume per time)" if size else "a rate per time"
            out += _wrap(
                f"{src} -> out of the model, {tr.rate} x {tr.source}: {what}; booked as a loss.",
                "  ",
            )
            continue
        dst = _label(tr.to, tr.to_where)
        out += _wrap(
            f"{src} -> {dst}: amount {tr.rate} x {tr.source} per time leaves {src} and arrives "
            f"in {dst} (divided by each side's volume or weights); the ledger books it on both "
            "sides, so it balances by construction.",
            "  ",
        )
    return out


def _shared_terms(M: _Model) -> tuple[list[str], list[str]]:
    """Terms written in more than one equation or boundary condition (``k*P*B`` lost by B
    and by P, gained by I at age 0), matched by their state-dependent part; and, for every
    place a term is proportional to (a participant, as P in ``k*P*B``), whether that place's
    own equation has the term: a forgotten term (phage not used up by adsorption) shows
    there. Flux statements are checked on their own."""
    from .expr import end_key

    states = set(M.fields) | set(M.comps)
    ends = {k: v[0] for k, v in tables(M.ir).ends.items()}
    live = states | set(ends) | {"t"}
    coords = {a.name for a in M.ir.axes if a.type == "continuous"}

    def inline(e: sp.Expr, seen: frozenset = frozenset()) -> sp.Expr:
        subs = {}
        for x in e.free_symbols:
            n = getattr(x, "name", "")
            if n in M.ints and n not in seen:
                subs[x] = inline(M.expr(M.ints[n].integrand), seen | {n})
        return e.xreplace(subs) if subs else e

    # (place, owner) -> {state part: summed coefficient}
    found: dict[tuple[str, str], dict[sp.Expr, sp.Expr]] = {}

    def add(place: str, owner: str, e: sp.Expr, end=None) -> None:
        e = inline(e)
        if end is not None:  # in a condition at an end, the field means its value there
            F, ax, side = end
            e = e.xreplace({M.symbols[F]: M.symbols[end_key(F, ax, side)]})
        here = found.setdefault((place, owner), {})
        # parts free of the state are kept whole (k*(1 - z/2)*B*P is one term, not two);
        # a part varying along an axis belongs to the term (delta(a)*i is not p(a)*i)
        held: dict[sp.Expr, sp.Expr] = {}

        def atoms(x: sp.Expr) -> sp.Expr:
            if x.is_Atom or isinstance(x, sp.Indexed):
                return x
            names = {getattr(y, "name", "") for y in x.free_symbols}
            if not names & live:
                if x not in held:
                    held[x] = sp.Dummy()
                return held[x]
            return x.func(*(atoms(a) for a in x.args))

        def reads(x: sp.Expr) -> bool:
            return bool({getattr(y, "name", "") for y in x.free_symbols} & live)

        def spread(x: sp.Expr) -> list[sp.Expr]:
            """Additive parts, a product multiplied out only over a sum every part of which
            reads the state (h*(C - M/V), b*(lysis of I + lysis of H)); (1 - frac) and
            (beta*frac + q) stay whole, as written."""
            if isinstance(x, sp.Add):
                return [y for a in x.args for y in spread(a)]
            if isinstance(x, sp.Mul):
                for i, f in enumerate(x.args):
                    if isinstance(f, sp.Add) and all(reads(a) for a in f.args):
                        rest = sp.Mul(*(x.args[:i] + x.args[i + 1 :]))
                        return [y for a in f.args for y in spread(rest * a)]
            return [x]

        back = {}
        for part in spread(atoms(e)):
            coef, state = sp.Integer(1), sp.Integer(1)
            for f in sp.Mul.make_args(part):
                if not back:
                    back = {d: x for x, d in held.items()}
                g = f.xreplace(back)
                names = {getattr(y, "name", "") for y in g.free_symbols}
                if names & (live | coords):
                    state *= g
                else:
                    coef *= g
            if state != 1:
                here[state] = here.get(state, sp.Integer(0)) + coef

    for t in M.ir.terms:
        place = _label(t.target, t.where)
        if t.kind in ("reaction", "rate"):
            add(place, t.target, M.expr(t.expr))
        elif t.kind == "hazard":
            add(place, t.target, -M.expr(t.expr) * M.symbols[t.target])
    for b in M.ir.boundaries:
        a = M.axes[b.axis]
        pos = _num(a.faces[0] if b.side == "lower" else a.faces[-1])
        sign = 1 if b.side == "lower" else -1  # a flux along the axis -> into the field
        end = (b.target, b.axis, b.side)
        if b.kind in ("flux", "inflow"):
            add(f"{b.target} at {b.axis}={pos}", b.target, sign * M.expr(b.value), end)
        elif b.kind == "dirichlet" and b.to is None:
            moving = [t for t in M.ir.terms if t.target == b.target and t.axis == b.axis]
            if moving and all(t.kind == "advection" for t in moving):
                # F(t, 0) = X where the flow enters: the renewal condition, a flux v*X
                v = sum((M.expr(t.expr) for t in moving), sp.Integer(0))
                add(f"{b.target} at {b.axis}={pos}", b.target, v * M.expr(b.value), end)
        elif b.kind == "outflow" and b.to is None:
            for t in M.ir.terms:
                if t.kind == "advection" and t.target == b.target and t.axis == b.axis:
                    leaving = sign * M.expr(t.expr) * M.symbols[b.target]
                    add(f"{b.target} leaving at {b.axis}={pos}", b.target, leaving, end)

    def participants(state: sp.Expr) -> set[str]:
        """Places the term is proportional to (a factor that is the place, or the place at
        an end); places read inside a bracket or a power do not count."""
        out = set()
        for f in sp.Mul.make_args(state):
            n = getattr(f, "name", None)
            if n not in states and n not in ends:
                continue
            if state.count(f) != 1:  # also inside a bracket: a signal (S/(Ks + S)), not mass
                continue
            out.add(n if n in states else ends[n])
        return out

    groups: dict[sp.Expr, list[tuple[str, str, sp.Expr]]] = {}
    for (place, owner), terms in found.items():
        for state, coef in terms.items():
            coef = sp.expand(coef)
            if coef != 0:
                groups.setdefault(state, []).append((place, owner, coef))
    shared, lone = [], []
    work = list(groups.items())
    while work:
        state, items = work.pop(0)
        # a place losing the state by two routes, -(ka + k)*C, shares only the part another
        # place has (+ka*C): keep that part (the rest is its own, e.g. faecal loss)
        for i, (p_i, o_i, c_i) in enumerate(items):
            if not isinstance(c_i, sp.Add):
                continue
            others = [c for j, (_, _, c) in enumerate(items) if j != i]
            part = next(
                (
                    a
                    for a in sp.Add.make_args(sp.expand(c_i))
                    for c in others
                    if sp.simplify(a + c) == 0 or sp.simplify(a - c) == 0
                ),
                None,
            )
            if part is not None:
                items[i] = (p_i, o_i, part)
        ref = min(
            (c for _, _, c in items),
            key=lambda c: sp.count_ops(-c if c.could_extract_minus_sign() else c),
        )
        # the same term up to a factor is believed when it is exactly +-1, when the two
        # coefficients share a parameter (Q/V1 and Q/V2), or when the shared part is more
        # than one place's value (I*h(a), B*P): delta*i and p*i are two processes
        rich = len(sp.Mul.make_args(state)) > 1 or not isinstance(state, sp.Symbol)
        keep, apart = [], []
        for item in items:
            c = item[2]
            ratio = sp.simplify(c / ref)
            ok = (
                c is ref
                or ratio in (1, -1)
                or rich
                or bool((c.free_symbols & ref.free_symbols) - {sp.Integer(1)})
            )
            (keep if ok else apart).append(item)
        if apart:
            work += [(state, [x]) for x in apart]
            items = keep
        owners = {o for _, o, _ in items}
        missing = sorted(participants(state) - owners)
        note = (
            f" Not in the equation of {', '.join(missing)}: if "
            f"{'it takes' if len(missing) == 1 else 'they take'} part (used up or produced), "
            "the term is missing there."
            if missing
            else ""
        )
        if _sign_at_typical(ref) < 0:
            ref = -ref
        term = _text(sp.nsimplify(ref * state, rational=True))
        size = {
            **{n: M.expr(c.volume) for n, c in M.comps.items() if c.volume},
            **{n: M.expr(f.weights) for n, f in M.fields.items() if f.weights},
        }
        net = sp.simplify(sum(c * size.get(o, 1) for _, o, c in items))

        def spread(o):  # continuous axes a place's terms are densities over
            f = M.fields.get(o)
            return () if f is None else tuple(a for a in f.axes if M.axes[a].type != "categorical")

        # amounts compare only between places measured alike: a rate in a compartment is an
        # amount per time, the same rate in a field over z is per unit of z as well
        alike = len({spread(o) for _, o, _ in items}) == 1
        balance = (
            " Amounts balance: what one place loses, the others gain." if net == 0 and alike else ""
        )
        if len({p for p, _, _ in items}) > 1:
            parts = ", ".join(
                f"{p} {_times(sp.nsimplify(sp.simplify(c / ref), rational=True))}"
                for p, _, c in items
            )
            shared += _wrap(f"{term}: {parts}.{balance}{note}", "  ")
        elif missing:
            p, _, c = items[0]
            lone += _wrap(
                f"{p}: {_signed(_text(sp.nsimplify(c * state, rational=True)))}.{note}", "  "
            )
    return shared, lone


def _carried(M: _Model, part: str) -> set[str]:
    """Places a term is proportional to: a factor that is another place's value, a slice of
    it or a named total or formula of it (``beta*lysis``, ``Q*C[sub=vascular]``). A place
    read inside a bracket or a power (feedback, e.g. ``(circ0/Circ)^gam``) does not count:
    material moving out of a place is proportional to what is there."""
    out: set[str] = set()
    for f in sp.Mul.make_args(parse(part, M.symbols)):
        if not isinstance(f, sp.Symbol):
            continue
        if f.name in M.fields or f.name in M.comps:
            out.add(f.name)
        elif f.name in M.defs or f.name in M.ints:
            out |= M.reads(f.name)
    return out


def _default_end(M: _Model, field: str, ax: str, side: str) -> str:
    adv = any(t.kind == "advection" and t.target == field and t.axis == ax for t in M.ir.terms)
    if M.axes[ax].kind == "one_way" and adv and side == "upper":
        return "outflow"
    return "zero_flux"


def _boundaries(M: _Model) -> list[str]:
    given = {(b.target, b.axis, b.side): b for b in M.ir.boundaries}
    out = []
    for f in M.ir.fields:
        for ax in f.axes:
            if M.axes[ax].type != "continuous":
                continue
            lo, hi = M.axes[ax].faces[0], M.axes[ax].faces[-1]
            for side, at in (("lower", lo), ("upper", hi)):
                b = given.get((f.name, ax, side))
                kind = b.kind if b else _default_end(M, f.name, ax, side)
                where = f"{f.name} at {ax}={side} ({ax} = {at:g})"
                default = "" if b else " (the default)"
                if kind == "zero_flux":
                    text = f"{where}: sealed, nothing crosses{default}."
                elif kind == "outflow":
                    to = f" into {b.to}" if b and b.to else ", leaving the model"
                    text = f"{where}: what the flow carries there goes out{to}{default}."
                elif kind == "inflow":
                    text = f"{where}: enters at rate {b.value} (per unit of the other axes)."
                elif kind == "flux":
                    text = (
                        f"{where}: the flux of {f.name} along {ax} there (toward larger {ax}) "
                        f"equals {b.value}, {f.name} meaning its value at this end."
                    )
                elif kind == "dirichlet":
                    text = f"{where}: held at {b.value}" + (
                        f"; what crosses is booked in {b.to}." if b.to else "."
                    )
                else:
                    text = f"{where}: exchange {b.h}*({b.value} - {f.name}) with the outside" + (
                        f"; what crosses is booked in {b.to}." if b.to else "."
                    )
                out += _wrap(text, "  ")
    return out or ["  none (no continuous axes)."]


def _starts(M: _Model) -> list[str]:
    ir = M.ir
    out = []
    zero = []
    covered: dict[str, list] = {}
    for it in ir.inits:
        covered.setdefault(it.target, []).append(it.where)
        part = _label(it.target, it.where)
        if it.value is not None:
            v = (
                M.expr(it.value)
                if not re.search(rf"\b({'|'.join(SHAPES)})\s*\(", it.value)
                else None
            )
            if v is not None and v.is_zero:
                zero.append(part)
                continue
            if it.target in M.comps:
                out.append(f"  {part} = {it.value}")
            elif v is None:
                out += _wrap(
                    f"{part} = {it.value}: the curve itself as a density, not scaled to an amount.",
                    "  ",
                )
            else:
                out += _wrap(f"{part} = {it.value}: a density (per unit of the axes).", "  ")
        elif it.shape and it.shape.strip().startswith("point("):
            out += _wrap(
                f"{part}: all of {it.total} in the one cell containing {it.shape.strip()[6:-1]}.",
                "  ",
            )
        else:
            out += _wrap(
                f"{part}: an amount {it.total} spread like {it.shape}, scaled so the cells "
                "hold exactly that amount.",
                "  ",
            )
    for name in [*M.fields, *M.comps]:
        wheres = covered.get(name)
        if not wheres:
            zero.append(name)
        elif all(wheres) and name in M.fields and not _covers(M, name, wheres):
            zero.append(f"the rest of {name}")
    if zero:
        out += _wrap("Start at 0 (not given, or given as 0): " + ", ".join(zero) + ".", "  ")
    return out or ["  everything starts at 0."]


def _covers(M: _Model, name: str, wheres) -> bool:
    """Whether slices ``wheres`` of field ``name`` together cover all of it."""
    cat = [a for a in M.fields[name].axes if M.axes[a].type == "categorical"]
    labels = [M.axes[a].labels for a in cat]
    import itertools

    for combo in itertools.product(*labels):
        cell = dict(zip(cat, combo, strict=True))
        if not any(all(cell.get(a) == lab for a, lab in w) for w in wheres):
            return False
    return True


def _extinctions(ir: ModelIR) -> list[str]:
    out = []
    for e in ir.extinctions:
        members = ", ".join(_label(m, e.member_where(i)) for i, m in enumerate(e.group))
        text = (
            f"{e.name}: {members}. When their total falls below {e.threshold:g} (after having "
            "been above it), all of them are set to 0 at that moment and the solve restarts; "
            "a group that has never reached the threshold (e.g. mutants seeded from 0) is "
            "never touched."
        )
        if e.noise_floor:
            text += (
                f" Noise floor: while the group's total is below {e.threshold:g}, the "
                "equations treat the whole group as 0 (stops ghost populations made by solver "
                "round-off from growing)."
            )
        out += _wrap(text, "  ")
    return out


def _safeguards(ir: ModelIR, report: Report) -> list[str]:
    from .backends.scipy_backend import FIRST_STEP_FLOOR

    out = [
        "  No value is clipped, capped or floored: negative values, if the solver makes any, are",
        "  kept and reported by diagnostics(sol).",
        "  Solver: BDF unless solve(method=...) says otherwise; a segment where BDF fails is "
        "retried once with Radau (said in sol.messages).",
        f"  Each segment (between doses) starts with a fresh step, never below "
        f"{FIRST_STEP_FLOOR:g} x its length.",
    ]
    flux_ends = [b for b in ir.boundaries if b.kind in ("flux", "inflow")]
    if flux_ends:
        names = ", ".join(sorted({f"{b.target} at {b.axis}={b.side}" for b in flux_ends}))
        out += _wrap(
            f"Ends with a flux condition ({names}): where the flow leaves through the end, it "
            "carries out the edge cell's value (upwind); the condition then sets the value "
            "read at the end. Where nothing sets that value (the flow leaves, nothing "
            "diffuses, and the condition does not depend on the value), it is the edge "
            "cell's, and what leaves is what the flow carries.",
            "  ",
        )
    floors = [e.name for e in ir.extinctions if e.noise_floor]
    if floors:
        out.append(f"  Noise floors (declared): {', '.join(floors)} (see Extinction groups).")
    cellwise = [
        i for i in report.notes if "cell by cell" in i.message or "where it stands" in i.message
    ]
    for i in cellwise:
        out += _wrap(f"{i.where}: {i.message}", "  ")
    return out


# ------------------------------------------------------------------------------ diagnostics


def diagnostics(model, sol=None) -> Report:
    """Accuracy guidance for a model, and checks of a solution (see the module docstring)."""
    ir = model.to_ir()
    cm = model.compile()
    issues: list[Issue] = []
    times, states = _samples(cm, sol)
    coeffs = [
        cm.transport_coefficients(t, y, None if sol is None else sol.theta)
        for t, y in zip(times, states, strict=True)
    ]
    M = _Model(ir)
    for (field, ax), _ in coeffs[0].items() if coeffs else []:
        issues += _transport_checks(M, cm, field, ax, [c[(field, ax)] for c in coeffs], sol)
    if sol is not None:
        issues += _solution_checks(cm, ir, sol)
    return Report(issues)


def _samples(cm, sol, n: int = 25):
    if sol is None:
        missing = cm.params.missing(cm.theta0)
        if missing:
            from .lower import MissingValues

            raise MissingValues(
                missing,
                "diagnostics needs values: pass a solution, "
                "m.diagnostics(m.solve(..., params={...}))",
            )
        return [0.0], [np.asarray(cm.initial_state())]
    idx = np.unique(np.linspace(0, len(sol.t) - 1, min(n, len(sol.t))).astype(int))
    return [float(sol.t[i]) for i in idx], [sol.y[i] for i in idx]


def _transport_checks(M: _Model, cm, field: str, ax: str, per_time, sol) -> list[Issue]:
    where = f"field '{field}', axis '{ax}'"
    out: list[Issue] = []
    faces = np.asarray(M.axes[ax].faces, dtype=float)
    widths = np.diff(faces)
    terms = [t for t in M.ir.terms if t.target == field and t.axis == ax]
    schemes = {t.scheme for t in terms if t.kind == "advection"}
    vs = [c["velocity"] for c in per_time if c["velocity"] is not None]
    Ds = [c["D"] for c in per_time if c["D"] is not None]
    dim = cm.layout.axes[field].index(ax)
    if Ds:
        dmin = min(float(np.min(D)) for D in Ds)
        if dmin < 0:
            out.append(
                Issue(
                    "warning",
                    where,
                    f"the diffusivity is negative somewhere (down to {dmin:.3g}): anti-diffusion "
                    "is ill-posed and makes peaks grow without bound; it is used as given, never "
                    "clipped. Check the formula of D.",
                )
            )
    if vs:
        vmax = max(float(np.max(np.abs(v))) for v in vs)
        vmin_signed = min(float(np.min(v)) for v in vs)
        if M.axes[ax].kind == "one_way" and vmin_signed < 0:
            out.append(
                Issue(
                    "warning",
                    where,
                    f"the velocity is negative somewhere (down to {vmin_signed:.3g}) on a one_way "
                    "axis: material would flow backwards through the inlet.",
                )
            )
        if vmax > 0:
            cross = float(np.min(widths)) / vmax
            msg = (
                f"fastest transport crosses a cell in {cross:.3g} time units "
                f"(|v| up to {vmax:.3g}, cells {_widths(widths)} wide)"
            )
            span = 0.0 if sol is None else float(sol.t[-1] - sol.t[0])
            if span > 0:
                msg += f"; over the run that is up to {span / cross:.3g} cells"
            if schemes == {"sharp"}:
                msg += (
                    "; with sharp the solver follows the front cell by cell, so its steps are "
                    "at most about this long"
                )
            elif _uniform(widths) and not M.varies_along(
                " + ".join(t.expr for t in terms if t.kind == "advection"), ax
            ):
                n = len(widths)
                msg += f"; erlang: a chain of {n} transit compartments, CV {1 / math.sqrt(n):.3g}"
            out.append(Issue("note", where, msg + "."))
        if Ds:
            pe = 0.0
            for v, D in zip(vs, Ds, strict=False):
                pe = max(pe, _peclet(np.asarray(v), np.asarray(D), widths, dim))
            if np.isinf(pe):
                pe_txt = "is infinite where D = 0"
            else:
                pe_txt = f"is up to {pe:.3g}"
            if pe > 2:
                tail = (
                    "transport dominates diffusion within a cell: first-order upwind (erlang) "
                    "then adds more spread (|v| x width / 2) than D itself, so the spread you "
                    "see is mostly the grid's. Use more cells or sharp if D is meant to set it."
                    if schemes == {"erlang"}
                    else "transport dominates diffusion within a cell; sharp keeps fronts, but "
                    "more cells resolve the diffusive layer."
                )
            else:
                tail = "diffusion is resolved within a cell."
            out.append(Issue("note", where, f"cell Péclet number |v| x width / D {pe_txt}: {tail}"))
    return out


def _widths(widths: np.ndarray) -> str:
    if _uniform(widths):
        return f"{widths[0]:.3g}"
    return f"{np.min(widths):.3g} to {np.max(widths):.3g}"


def _uniform(widths: np.ndarray) -> bool:
    return bool(np.allclose(widths, widths[0], rtol=1e-12, atol=0))


def _peclet(v: np.ndarray, D: np.ndarray, widths: np.ndarray, dim: int) -> float:
    """Largest |v| x width / D over the cells (face velocities averaged onto cells)."""
    v = np.moveaxis(np.atleast_1d(v), dim, 0) if v.ndim > dim else np.atleast_1d(v)
    vc = 0.5 * (np.abs(v[:-1]) + np.abs(v[1:])) if v.shape[0] == len(widths) + 1 else np.abs(v)
    D = np.moveaxis(np.atleast_1d(D), dim, 0) if D.ndim > dim else np.atleast_1d(D)
    w = widths.reshape((-1,) + (1,) * (vc.ndim - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        pe = np.where(np.abs(D) > 0, vc * w / np.abs(D), np.where(vc > 0, np.inf, 0.0))
    return float(np.max(pe))


def _solution_checks(cm, ir: ModelIR, sol) -> list[Issue]:
    from .backends.scipy_backend import atol_floor

    out: list[Issue] = []
    for m in sol.messages:
        out.append(Issue("warning", "solver", m))
    atol = float(np.min(np.atleast_1d(sol.options.get("atol", 1e-6))))
    guarded = {m for e in ir.extinctions if e.threshold >= atol for m in e.group}
    names = [*(f.name for f in ir.fields), *(c.name for c in ir.compartments)]
    amounts = [cm.amounts(y, sol.theta) for y in sol.y]
    # what crosses a Dirichlet or Robin end is booked in a compartment, in either direction
    booked = {b.to for b in ir.boundaries if b.to and b.kind in ("dirichlet", "robin")}
    # the doses given in this run (a regimen replaces the model's doses)
    dosed = sorted(d[0] for d in sol.doses)
    infusing = [(f[0], f[1]) for f in sol.infusions if f[3] > 0]  # refill over their windows
    big = [d for d in sol.doses if atol_floor(d[3]) > atol]
    if big:
        td, target, _, rise = max(big, key=lambda d: d[3])
        out.append(
            Issue(
                "warning",
                target,
                f"the dose at t = {td:g} raised its value by {rise:.3g}. That leaves rounding "
                f"errors of about {atol_floor(rise):.0e} in the states it feeds, above atol "
                f"({atol:.3g}): the solver's error estimate is then noise, and BDF fails or "
                "survives by chance (a slightly different rtol can change the outcome). Use "
                f"atol={atol_floor(rise):.0e} or larger.",
            )
        )
    for name in names:
        vals = np.asarray(sol[name])
        low = float(np.min(vals)) if vals.size else 0.0
        if low < -10 * atol and name not in booked:
            out.append(
                Issue(
                    "warning",
                    name,
                    f"went negative (down to {low:.3g}, beyond -10 x atol = {-10 * atol:.3g}): "
                    "the solver's tolerance allowed an undershoot. Tighten atol/rtol, or check for "
                    "a negative rate or diffusivity.",
                )
            )
        if name in guarded:
            continue
        total = np.array([a[name] for a in amounts])
        hit = next(  # a regrowth with no dose in between (a dose anywhere may refill it)
            (
                (i, j)
                for i, j in _dips_and_regrowths(total, atol)
                if not any(sol.t[i] <= x <= sol.t[j] for x in dosed)
                and not any(a <= sol.t[j] and sol.t[i] <= b for a, b in infusing)
            ),
            None,
        )
        stays = _dip(total, atol)
        if stays is not None:
            out.append(
                Issue(
                    "note",
                    name,
                    f"its total fell below atol ({atol:.3g}) at t ≈ {sol.t[stays]:.4g} and stayed "
                    "there. Below atol a population is not resolved, so this solve does not decide "
                    "whether it would have grown back. If that matters, make it an explicit "
                    "choice: "
                    "declare an extinction threshold (m.extinction), or tighten atol.",
                )
            )
        if hit is not None:
            i, j = hit
            out.append(
                Issue(
                    "warning",
                    name,
                    f"its total fell below atol ({atol:.3g}) at t ≈ {sol.t[i]:.4g} and grew back "
                    f"to {total[j]:.3g} by t ≈ {sol.t[j]:.4g}. Below atol the solver does not "
                    "resolve a population, so this regrowth may be a numerical artefact. Declare "
                    "an extinction threshold (m.extinction) if it should die out, or tighten atol "
                    "if the remnant is real.",
                )
            )
    if sol.ledger is not None:
        worst = max(sol.ledger.check().values(), default=0.0)
        level = "warning" if worst > 1e-6 else "note"
        out.append(
            Issue(
                level,
                "ledger",
                f"largest unexplained amount, relative: {worst:.2g}"
                + ("" if level == "note" else " — the books do not close; tighten rtol/atol."),
            )
        )
    return out


def _dip(total: np.ndarray, atol: float):
    """Index where the total, after it was last above 10 x atol, falls below atol and stays
    there to the end; None if it never does."""
    above = np.flatnonzero(total > 10 * atol)
    if above.size == 0:
        return None
    below = np.flatnonzero(total[above[-1] :] < atol)
    return int(above[-1] + below[0]) if below.size else None


def _dips_and_regrowths(total: np.ndarray, atol: float):
    """Every (i, j): above 10 x atol before i, below atol at i, back above 10 x atol at j."""
    seen_high, low_at = False, None
    for k, v in enumerate(total):
        if v > 10 * atol:
            if low_at is not None:
                yield low_at, k
                low_at = None
            seen_high = True
        elif v < atol and seen_high and low_at is None:
            low_at = k
