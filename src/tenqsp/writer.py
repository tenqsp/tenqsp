# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Any model as text in the TenQSP notation (``m.to_text()``; docs/language/notation.md).

The text reads back (``tq.Model.from_text``) to the same model: the same IR up to the
canonical form of ``tenqsp.canonical`` (V12). Python-API constructs without a word of their
own in the notation are written with the notation's equivalent:

- a transfer between two places becomes a flux with declared ends, ``-J`` in the equation
  of the source and ``+J`` in that of the destination (divided by volumes or weights, summed
  over the axes the other end lacks);
- a transfer out of the model (``to=None``) becomes a loss term ``-(rate)*F/(volume)``;
- a hazard becomes ``-(rate)*F``, a reaction ``+(expr)``, a switch ``trans_s(M, F)`` and a
  mutation ``trans_s(M, flux)``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from .expr import SHAPES
from .ir import ModelIR
from .model import _float_text

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class WriteError(ValueError):
    """A model uses something the notation cannot write."""


def _num(v: float) -> str:
    v = float(v)
    if v == int(v) and abs(v) < 1e5:
        return str(int(v))
    return _float_text(v)


def _label(s: str) -> str:
    return s if _NAME.fullmatch(s) else repr(s)


def _opt(key: str, value: str | None) -> str:
    if not value:
        return ""
    return f" {key}={value!r}" if (" " in value or "=" in value) else f" {key}={value}"


def _where(where) -> str:
    return "[" + ", ".join(f"{a}={_label(lab)}" for a, lab in where) + "]" if where else ""


def _ops(axes: str | tuple[str, ...]) -> str:
    return axes if isinstance(axes, str) else "{" + ",".join(axes) + "}"


def _p(expr: str) -> str:
    """A formula in brackets, unless it is one name or number."""
    e = expr.strip()
    while _enclosed(e):
        e = e[1:-1].strip()
    indexed = re.fullmatch(rf"{_NAME.pattern}\[[^\[\]]*\]", e)
    if _NAME.fullmatch(e) or indexed or (re.fullmatch(r"[0-9.eE+-]+", e) and e[0] not in "+-"):
        return e
    return f"({e})"


def _enclosed(e: str) -> bool:
    """``(…)`` with the first bracket closing at the end."""
    if not (e.startswith("(") and e.endswith(")")):
        return False
    depth = 0
    for i, ch in enumerate(e):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and i < len(e) - 1:
            return False
    return True


def _uniform(faces: tuple[float, ...]) -> bool:
    f = np.asarray(faces)
    return np.allclose(np.diff(f), (f[-1] - f[0]) / (len(f) - 1), rtol=1e-12, atol=0)


def _table(values: Iterable[float], labels: list[tuple[str, ...]]) -> str:
    vals = list(values)
    if len(labels) == 1:
        return (
            "{"
            + ", ".join(f"{_label(lab)}: {_num(v)}" for lab, v in zip(labels[0], vals, strict=True))
            + "}"
        )
    step = len(vals) // len(labels[0])
    inner = [
        f"{_label(lab)}: {_table(vals[i * step : (i + 1) * step], labels[1:])}"
        for i, lab in enumerate(labels[0])
    ]
    return (
        "{\n    " + ",\n    ".join(inner) + ",\n}"
        if len(labels) == 2
        else "{" + ", ".join(inner) + "}"
    )


def to_text(ir: ModelIR) -> str:
    """The model as text in the TenQSP notation."""
    axes = {a.name: a for a in ir.axes}
    fields = {f.name: f for f in ir.fields}
    comps = {c.name: c for c in ir.compartments}
    out: list[str] = [f"model {ir.name}", ""]

    for a in ir.axes:
        if a.type == "categorical":
            out.append(
                f"axis {a.name} categorical [{', '.join(_label(x) for x in a.labels)}]"
                + _opt("units", a.units)
            )
            continue
        if _uniform(a.faces):
            rng = f"in [{_num(a.faces[0])}, {_num(a.faces[-1])}] n={len(a.faces) - 1}"
        else:
            rng = "faces=[" + ", ".join(repr(float(f)) for f in a.faces) + "]"
        kind = "" if a.kind == "bidirectional" else f" kind={a.kind}"
        out.append(f"axis {a.name} {rng}{kind}" + _opt("units", a.units))
    if ir.axes:
        out.append("")

    for p in ir.params:
        if p.over is None:
            value = "" if p.value is None else f" = {_num(p.value)}"  # None: declared only
            out.append(f"param {p.name}{value}" + _opt("units", p.units))
        elif p.value is None:
            over = p.over if isinstance(p.over, tuple) else (p.over,)
            out.append(f"param {p.name}[{', '.join(over)}]" + _opt("units", p.units))
        else:
            over = p.over if isinstance(p.over, tuple) else (p.over,)
            table = _table(p.value, [axes[a].labels for a in over])
            out.append(f"param {p.name}[{', '.join(over)}] = {table}" + _opt("units", p.units))
    if ir.params:
        out.append("")

    for f in ir.fields:
        w = f" weights={f.weights}" if f.weights else ""
        out.append(f"field {f.name}({', '.join(f.axes)})" + _opt("units", f.units) + w)
    for c in ir.compartments:
        v = f" volume={c.volume}" if c.volume else ""
        out.append(f"compartment {c.name}" + _opt("units", c.units) + v)
    if ir.fields or ir.compartments:
        out.append("")

    for tr in ir.transitions:
        joint = isinstance(tr.axis, tuple)

        def cat(c, joint=joint) -> str:
            return "(" + ", ".join(_label(x) for x in c) + ")" if joint else _label(c)

        on = "(" + ", ".join(tr.axis) + ")" if joint else tr.axis
        moves = "\n".join(f"    {cat(f)} -> {cat(t)} = {r}" for f, t, r in tr.rates)
        out.append(f"transitions {tr.name} on {on}:\n{moves}")

    # named formulas and totals, in the order they were made
    for d in ir.definitions:
        out.append(f"{d.name} = {d.expr}")
    for i in ir.integrals:
        out.append(f"{i.name} = {_integral(i, fields, axes)}")
    if ir.definitions or ir.integrals or ir.transitions:
        out.append("")

    # equations: terms per place (field, slice or compartment)
    eqs: dict[tuple, list[str]] = {}
    sharp: dict[str, list[str]] = {}

    def add(target: str, where, term: str) -> None:
        eqs.setdefault((target, tuple(where)), []).append(term)

    for t in ir.terms:
        F = t.target
        if t.kind == "advection":
            v = t.expr.strip()
            add(F, (), f"- d_{t.axis}({F})" if v in ("1", "1.0") else f"- d_{t.axis}({_p(v)}*{F})")
            if any(
                u.kind == "advection"
                and u.target == F
                and u.axis == t.axis
                and u.scheme != t.scheme
                for u in ir.terms
            ):
                raise WriteError(
                    f"'{F}' has two transports along {t.axis} with different schemes; the "
                    "notation gives one scheme per field and axis (merge them, or use one scheme)."
                )
            if t.scheme == "sharp" and t.axis not in sharp.get(F, []):
                sharp.setdefault(F, []).append(t.axis)
        elif t.kind == "diffusion":
            add(F, (), f"+ d_{t.axis}({_p(t.expr)}*d_{t.axis}({F}))")
        elif t.kind == "reaction":
            add(F, t.where, f"+ {_p(t.expr)}")
        elif t.kind == "hazard":
            add(F, t.where, f"- {_p(t.expr)}*{F}")
        elif t.kind == "rate":
            add(F, (), f"+ {_p(t.expr)}")
        elif t.kind == "switch":
            add(F, t.where, f"+ trans_{_ops(t.axis)}({t.transitions}, {F}{_where(t.where)})")
        elif t.kind == "mutate":
            add(F, t.where, f"+ trans_{_ops(t.axis)}({t.transitions}, {t.expr})")
        else:
            raise WriteError(f"a '{t.kind}' term cannot be written in the notation yet.")

    fluxes = []
    taken: set[str] = {x.name for x in (*ir.axes, *ir.params, *ir.fields, *ir.compartments)}
    taken |= {x.name for x in (*ir.definitions, *ir.integrals, *ir.transitions)}
    for k, tr in enumerate(ir.transfers):
        src = (tr.source, tuple(tr.source_where))
        w_src = _weight(tr.source, fields, comps)
        if tr.to is None:  # leaves the model: a loss
            div = f"/{_p(w_src)}" if w_src else ""
            add(*src, f"- {_p(tr.rate)}*{tr.source}{div}")
            continue
        dst = (tr.to, tuple(tr.to_where))
        name = f"J{k + 1}"
        while name in taken:
            name += "_"
        taken.add(name)
        fluxes.append(
            f"flux {name}: {tr.source}{_where(tr.source_where)} -> {tr.to}{_where(tr.to_where)}"
            f" = {_p(tr.rate)}*{tr.source}"
        )
        free = _free(tr.source, tr.source_where, fields) + _free(tr.to, tr.to_where, fields)
        axes_j = list(dict.fromkeys(free))
        for end, sign in ((src, "-"), (dst, "+")):
            have = _free(end[0], end[1], fields)
            term = name
            for a in [a for a in axes_j if a not in have]:
                term = f"sum_{a}({term})"
            w = _weight(end[0], fields, comps)
            add(*end, f"{sign} {term}" + (f"/{_p(w)}" if w else ""))
    out += fluxes
    if fluxes:
        out.append("")

    # conditions written under their field's equation, as in a paper (IR 0.10); the older
    # statements below keep what this form does not say (routing, inflow, robin)
    under: dict[str, list[str]] = {}
    rest = []
    whole = {target for target, where in eqs if not where}
    for b in ir.boundaries:
        if b.target not in whole and b.kind != "flux":  # no equation to stand under
            rest.append(b)
            continue
        faces = axes[b.axis].faces
        at = f"    at {b.axis}={_num(faces[0] if b.side == 'lower' else faces[-1])}: "
        terms_b = [t for t in ir.terms if t.target == b.target and t.axis == b.axis]
        if b.kind == "flux":
            under.setdefault(b.target, []).append(
                f"{at}{_flux_text(b.target, b.axis, terms_b)} = {b.value}"
            )
        elif b.kind == "dirichlet" and b.to is None:
            under.setdefault(b.target, []).append(f"{at}{b.target} = {b.value}")
        elif b.kind == "outflow" and b.to is None:
            under.setdefault(b.target, []).append(f"{at}outflow")
        elif b.kind == "zero_flux" and terms_b:
            under.setdefault(b.target, []).append(
                f"{at}{_flux_text(b.target, b.axis, terms_b)} = 0"
            )
        else:
            rest.append(b)

    for (target, where), terms in eqs.items():
        if len(terms) == 1 and terms[0].startswith("+ ") and _enclosed(terms[0][2:]):
            terms = ["+ " + terms[0][3:-1]]  # one term: no brackets needed
        rhs = " ".join(terms)
        rhs = rhs[2:] if rhs.startswith("+ ") else "-" + rhs[2:] if rhs.startswith("- ") else rhs
        using = ""
        if not where and target in sharp:
            using = "  using sharp on " + ", ".join(sharp[target])
        out.append(f"d_t({target}{_where(where)}) = {rhs}{using}")
        if not where:
            out += under.pop(target, [])
    for target, lines in under.items():  # pragma: no cover - a moving field has an equation
        raise WriteError(f"conditions of '{target}' have no equation to stand under: {lines}")

    for b in rest:
        head = f"{b.kind} {b.target} at {b.axis}={b.side}"
        partner = f" with {b.to}" if b.to else ""
        if b.kind == "zero_flux":
            out.append(head)
        elif b.kind == "outflow":
            out.append(head + (f" to {b.to}" if b.to else ""))
        elif b.kind == "inflow":
            out.append(f"{head}: {b.value}")
        elif b.kind == "dirichlet":
            out.append(f"{head}: {b.value}{partner}")
        else:  # robin: h*(outside - F)
            out.append(f"{head}: {_p(b.h)}*({_p(b.value)} - {b.target}){partner}")
    if ir.boundaries:
        out.append("")

    for it in ir.inits:
        if it.value is not None:  # a density; shapes in it are not scaled to an amount
            shaped = re.search(rf"\b({'|'.join(SHAPES)}|shape)\s*\(", it.value)
            value = f"density({it.value})" if shaped else it.value
        else:
            value = f"{_p(it.total)}*{_shape_text(it.shape)}"
        out.append(f"{it.target}{_where(it.where)}(0) = {value}")
    field_axes = {f.name: f.axes for f in ir.fields}
    for ds in ir.doses:
        times = ", ".join(_num(t) for t in ds.times)
        free = [a for a in field_axes.get(ds.target, ()) if a not in dict(ds.where)]
        if ds.shape is not None:  # an amount spread by a shape, as an initial value
            each = [f"{_p(a)}*{_shape_text(ds.shape)}" for a in ds.amounts]
        elif free:  # a density added to the field
            each = [f"density({a})" for a in ds.amounts]
        else:
            each = list(ds.amounts)
        amounts = each[0] if len(set(each)) == 1 else ", ".join(each)
        out.append(f"dose {ds.target}{_where(ds.where)} at t={times}: {amounts}")
    for x in ir.lags:
        out.append(f"lag {x.target} = {x.value}")
    for x in ir.bioavailabilities:
        out.append(f"bioavailability {x.target} = {x.value}")
    for o in ir.observables:
        out.append(f"observable {o.name} = {o.expr}{_where(o.where)}")
    for e in ir.extinctions:
        members = ", ".join(f"{m}{_where(e.member_where(i))}" for i, m in enumerate(e.group))
        floor = " with noise_floor" if e.noise_floor else ""
        out.append(f"extinction {e.name}: {members} below {_num(e.threshold)}{floor}")
    return "\n".join(out).rstrip() + "\n"


def _flux_text(F: str, ax: str, terms) -> str:
    """The flux of F along ax as its equation's transport terms give it: v*F for a flow,
    -D*d_ax(F) for diffusion."""
    parts = []
    for t in terms:
        if t.kind not in ("advection", "diffusion"):
            continue
        v = t.expr.strip()
        if t.kind == "advection":
            parts.append(F if v in ("1", "1.0") else f"{_p(v)}*{F}")
        else:
            parts.append(f"-{_p(v)}*d_{ax}({F})")
    if parts == [F]:  # a flow at speed 1: 'F = ...' alone would read as a held value
        parts = [f"1*{F}"]
    return " + ".join(parts).replace("+ -", "- ")


def _weight(name: str, fields, comps) -> str | None:
    if name in fields:
        return fields[name].weights
    return comps[name].volume


def _free(name: str, where, fields) -> list[str]:
    if name not in fields:
        return []
    fixed = {a for a, _ in where}
    return [a for a in fields[name].axes if a not in fixed]


def _integral(i, fields, axes) -> str:
    over = list(i.over)
    if i.where:
        integrand = i.integrand.strip()
        if integrand in fields:
            inner = f"{integrand}{_where(i.where)}"
        else:
            f = re.fullmatch(rf"({_NAME.pattern})\s*/\s*\(?(.+?)\)?", integrand)
            if not (f and f.group(1) in fields and over == []):
                raise WriteError(
                    f"integral '{i.name}' slices a formula ({integrand}); the notation writes "
                    "only slices of fields."
                )
            inner = f"{f.group(1)}{_where(i.where)}"  # the value on the slice
            if not over:
                return inner
    else:
        inner = i.integrand
    text = inner
    for a in reversed(over):
        text = f"{'int' if axes[a].type == 'continuous' else 'sum'}_{a}({text})"
    return text


def _shape_text(shape: str) -> str:
    """A stored shape in the notation: named shapes as they are (a product of them for
    several axes), any other formula in shape(…)."""
    named = rf"({'|'.join(SHAPES)})\s*\([^()]*\)"
    s = shape.strip()
    if re.fullmatch(rf"{named}(\s*\*\s*{named})*", s) or re.fullmatch(
        rf"\(\s*{named}\s*\)(\s*\*\s*\(\s*{named}\s*\))*", s
    ):
        return s
    return f"shape({s})"
