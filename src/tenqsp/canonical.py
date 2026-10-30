# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Are two model descriptions the same model? (V12: API <-> notation round trip.)

The same model can be written in more than one way: ``m.hazard(B, k*P)`` and the notation's
``- k*P*B``; an integral named ``B_tot`` or made up as ``int_z_s``; ``div - ads`` as one
reaction or two; a transfer out of the model or a loss term; ``0.5`` or ``1/2``. The
canonical form used here removes those differences and nothing else:

- named formulas are written out (inlined), and integrals are named by what they integrate;
- the local terms of each place (a field, a slice of one, or a compartment) — reactions,
  losses (``-r*F``), a compartment's rate, transfers out of the model (``-rate*F/volume``)
  — are summed into one net rate, expanded;
- numbers are compared as floats; a zero starting value is the same as none.

Transport, diffusion, transitions, transfers between places, boundaries, starting values,
doses, outputs and extinction groups are compared one by one.
"""

from __future__ import annotations

import hashlib

import numpy as np
import sympy as sp

from .expr import parse
from .ir import ModelIR
from .lower import tables

__all__ = ["differences"]


class _Canon:
    def __init__(self, ir: ModelIR):
        self.ir = ir
        self.symbols = dict(tables(ir).symbols)
        self.defs = {d.name: d.expr for d in ir.definitions}
        self._def_cache: dict[str, sp.Expr] = {}
        self.integral_key: dict[str, sp.Symbol] = {}
        for i in ir.integrals:  # integrals may use integrals: resolve in dependency order
            self._integral(i.name)

    def _integral(self, name: str) -> sp.Symbol:
        if name in self.integral_key:
            return self.integral_key[name]
        spec = next(i for i in self.ir.integrals if i.name == name)
        integrand = self.expr(spec.integrand)
        key = f"{sp.srepr(integrand)}|{sorted(spec.over)}|{sorted(spec.where)}"
        sym = sp.Symbol("integral_" + hashlib.sha256(key.encode()).hexdigest()[:12])
        self.integral_key[name] = sym
        return sym

    def _definition(self, name: str) -> sp.Expr:
        if name not in self._def_cache:
            self._def_cache[name] = self.expr(self.defs[name])
        return self._def_cache[name]

    def expr(self, text: str | None) -> sp.Expr:
        """A formula with definitions inlined, integrals named by content, floats only."""
        if text is None:
            return sp.Integer(0)
        e = parse(text, self.symbols)
        subs = {}
        for s in e.free_symbols:
            if not isinstance(s, sp.Symbol):
                continue
            if s.name in self.defs:
                subs[s] = self._definition(s.name)
            elif any(i.name == s.name for i in self.ir.integrals):
                subs[s] = self._integral(s.name)
        if subs:
            e = e.xreplace(subs)
        return _floats(e)


def _num(n: sp.Number) -> sp.Number:
    """A number rounded to 12 significant digits; whole numbers as integers (2.0 is 2)."""
    v = float(n)
    if v.is_integer() and abs(v) < 1e15:
        return sp.Integer(int(v))
    return sp.Float(float(f"{v:.12g}"), 15)


def _floats(e: sp.Expr) -> sp.Expr:
    """Numbers made uniform (``0.5`` and ``1/2``; ``sqrt(6)`` written out and the float a
    Python model stored), then expanded."""
    for _ in range(2):  # again for numbers made by expanding (e.g. sqrt(6) evaluated)
        nums = {n: _num(n) for n in e.atoms(sp.Number) if not n.is_Integer}
        e = sp.expand(e.xreplace(nums) if nums else e)
    return e


def _same(a: sp.Expr, b: sp.Expr) -> bool:
    """Equal as formulas: structurally after expanding; with conditionals, branch by branch
    (a difference confined to one branch is found even where no random point falls); or else
    at random values of every symbol (so that numbers rounded differently inside functions,
    e.g. ``exp(x/sqrt(6))`` against the float a Python model stored, still compare equal)."""
    d = sp.expand(a - b)
    if d == 0:
        return True
    if d.has(sp.Piecewise):  # random points may all miss a branch: compare branch by branch
        f = sp.piecewise_fold(d)
        if isinstance(f, sp.Piecewise):
            return all(_same(e, sp.Integer(0)) for e, _ in f.args)
    atoms = sorted(d.free_symbols | d.atoms(sp.Indexed), key=str)
    if not atoms:
        return _close(a, b, {})
    rng = np.random.default_rng(12345)
    for _ in range(3):
        point = {s: sp.Float(rng.uniform(0.3, 1.7)) for s in atoms}
        if not _close(a, b, point):
            return False
    if d.has(*_SIGN_SENSITIVE):  # abs(v) and v agree at positive points: try both signs
        for k in range(3):  # all negative first, then mixed signs
            point = {
                s: sp.Float((-1 if k == 0 else rng.choice([-1, 1])) * rng.uniform(0.3, 1.7))
                for s in atoms
            }
            if not _close(a, b, point):
                return False
    return True


_SIGN_SENSITIVE = (sp.Abs, sp.Max, sp.Min, sp.sign, sp.Heaviside, sp.floor, sp.ceiling)


def _close(a: sp.Expr, b: sp.Expr, point: dict) -> bool:
    try:
        x = complex(a.xreplace(point).evalf(30))
        y = complex(b.xreplace(point).evalf(30))
    except (TypeError, ValueError):  # something that cannot be evaluated: not proven equal
        return False
    return abs(x - y) <= 1e-10 * max(abs(x), abs(y), 1e-300)


def _where(w) -> tuple:
    return tuple(sorted(tuple(x) for x in w))


def _canon(ir: ModelIR) -> dict:
    c = _Canon(ir)
    fields = {f.name: f for f in ir.fields}
    comps = {cp.name: cp for cp in ir.compartments}
    out: dict = {"name": ir.name}
    out["axes"] = {
        a.name: (
            a.type,
            tuple(np.round(a.faces, 14)),
            a.kind if a.type == "continuous" else "",
            a.units,
            a.labels,
        )
        for a in ir.axes
    }
    out["params"] = {
        p.name: (
            None if p.value is None else tuple(float(v) for v in np.ravel(p.value)),
            p.over if p.over is None or isinstance(p.over, str) else tuple(p.over),
            p.units,
        )
        for p in ir.params
    }
    out["fields"] = {f.name: (f.axes, f.units, c.expr(f.weights)) for f in ir.fields}
    out["compartments"] = {cp.name: (cp.units, c.expr(cp.volume)) for cp in ir.compartments}
    out["transitions"] = {
        tr.name: (
            tr.axis if isinstance(tr.axis, str) else tuple(tr.axis),
            {(str(f), str(t)): c.expr(r) for f, t, r in tr.rates},
        )
        for tr in ir.transitions
    }
    net: dict[tuple, sp.Expr] = {}
    moves: dict[tuple, sp.Expr] = {}
    for t in ir.terms:
        place = (t.target, _where(t.where))
        F = c.symbols[t.target]
        if t.kind in ("reaction", "rate"):
            net[place] = net.get(place, 0) + c.expr(t.expr)
        elif t.kind == "hazard":
            net[place] = net.get(place, 0) - c.expr(t.expr) * F
        elif t.kind in ("advection", "diffusion"):  # several on one axis add up
            key = (t.target, t.kind, t.axis, t.scheme if t.kind == "advection" else "")
            key += (_where(t.where),)
            moves[key] = moves.get(key, 0) + c.expr(t.expr)
        elif t.kind in ("switch", "mutate"):
            axis = t.axis if isinstance(t.axis, str) else tuple(t.axis)
            key = (t.target, t.kind, axis, t.transitions, _where(t.where))
            moves[key] = moves.get(key, 0) + c.expr(t.expr)
        else:
            key = (t.target, t.kind, _where(t.where))
            moves[key] = moves.get(key, 0) + c.expr(t.expr)
    transfers: dict[tuple, sp.Expr] = {}
    for tr in ir.transfers:
        if tr.to is None:  # a loss of the source: -rate*value/volume
            place = (tr.source, _where(tr.source_where))
            size = fields[tr.source].weights if tr.source in fields else comps[tr.source].volume
            loss = c.expr(tr.rate) * c.symbols[tr.source] / (c.expr(size) if size else 1)
            net[place] = net.get(place, 0) - _floats(loss)
        else:
            key = (tr.source, _where(tr.source_where), tr.to, _where(tr.to_where))
            transfers[key] = transfers.get(key, 0) + c.expr(tr.rate)
    out["net rate"] = {k: _floats(v) for k, v in net.items() if not _floats(v).is_zero}
    out["operators"] = moves
    out["transfers"] = transfers
    out["boundaries"] = {
        (b.target, b.axis, b.side): (b.kind, c.expr(b.value), c.expr(b.h), b.to)
        for b in ir.boundaries
    }
    inits = {}
    for it in ir.inits:
        key = (it.target, _where(it.where))
        if it.value is not None:
            v = c.expr(it.value)
            if not v.is_zero:
                inits[key] = ("value", v)
        else:
            inits[key] = ("amount and shape", c.expr(it.total), c.expr(it.shape))
    out["starting values"] = inits
    doses: dict[tuple, list] = {}
    for ds in ir.doses:
        shape = None if ds.shape is None else c.expr(ds.shape)
        doses.setdefault((ds.target, _where(ds.where)), []).extend(
            (float(t), c.expr(a), shape) for t, a in zip(ds.times, ds.amounts, strict=True)
        )
    out["doses"] = {k: sorted(v, key=lambda x: (x[0], str(x[1]))) for k, v in doses.items()}
    out["lags"] = {x.target: c.expr(x.value) for x in ir.lags}
    out["bioavailabilities"] = {x.target: c.expr(x.value) for x in ir.bioavailabilities}
    out["observables"] = {o.name: (c.expr(o.expr), _where(o.where)) for o in ir.observables}
    out["extinction groups"] = {
        e.name: (
            tuple((m, _where(e.member_where(i))) for i, m in enumerate(e.group)),
            float(e.threshold),
            e.noise_floor,
        )
        for e in ir.extinctions
    }
    return out


def _equal(a, b) -> bool:
    if isinstance(a, sp.Basic) or isinstance(b, sp.Basic):
        return _same(sp.sympify(a), sp.sympify(b))
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        return a == b or abs(a - b) <= 1e-14 * max(abs(a), abs(b))
    return a == b


def differences(a: ModelIR, b: ModelIR) -> list[str]:
    """What differs between two models in meaning (empty: the same model)."""
    ca, cb = _canon(a), _canon(b)
    out = []
    for part in ca:
        x, y = ca[part], cb[part]
        if not isinstance(x, dict):
            if x != y:
                out.append(f"{part}: {x!r} vs {y!r}")
            continue
        for k in sorted(set(x) | set(y), key=str):
            if k not in x:
                out.append(f"{part}: only in the second model: {k} = {y[k]}")
            elif k not in y:
                out.append(f"{part}: only in the first model: {k} = {x[k]}")
            elif not _equal(x[k], y[k]):
                out.append(f"{part} {k}: {x[k]} vs {y[k]}")
    return out
