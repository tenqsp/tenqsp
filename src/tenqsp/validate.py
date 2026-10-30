# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Model validation: find every problem in a ``ModelIR`` before compiling.

Each finding is an ``Issue`` with a level:

- ``error``: the model cannot be compiled as written;
- ``warning``: probably a mistake, or a default the modeller should know about;
- ``note``: a choice the library made, reported by ``explain()``.

Messages are written for modellers, not programmers: they name the item, say what is wrong
and suggest the fix.

Dimension checking
------------------
Every formula varies over some set of axes:

- a field over its axes;
- a coordinate over its own axis;
- ``k[s]`` over ``s``;
- an integral over its integrand's axes minus the ones integrated out;
- a definition over the axes of its formula.

A term's formula must vary only over axes its target has. A compartment's formula must not
vary at all; integrate first.
"""

from __future__ import annotations

import keyword
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import sympy as sp

from tenqsp.expr import NON_SMOOTH_TYPES as NON_SMOOTH
from tenqsp.expr import FormulaError, end_key, grads_used, parse, shapes_used
from tenqsp.ir import (
    BOUNDARY_KINDS,
    READABLE_VERSIONS,
    RESERVED_NAMES,
    TERM_KINDS,
    ModelIR,
)

Level = Literal["error", "warning", "note"]
AXIS_KINDS = ("one_way", "bidirectional", "spatial")


@dataclass(frozen=True)
class Issue:
    level: Level
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.where}: {self.message}"


class ModelError(ValueError):
    """Raised when a model has errors; the message lists all of them."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        errs = [i for i in issues if i.level == "error"]
        lines = "\n".join(f"  - {i.where}: {i.message}" for i in errs)
        super().__init__(f"The model has {len(errs)} error(s):\n{lines}")


@dataclass
class Report:
    """What ``m.validate()``, ``tq.validate(ir)`` and ``m.diagnostics()`` found: ``issues``
    (each with ``level``, ``where`` and ``message``), and the same split by level in
    ``errors``, ``warnings`` and ``notes``. ``ok`` is true when there are no errors.
    ``print(report)`` lists every issue, one per line."""

    issues: list[Issue]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def notes(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "note"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        return "\n".join(str(i) for i in self.issues) or "No issues."


class _Checker:
    def __init__(self, ir: ModelIR):
        self._noted_jumps: set = set()
        self.ir = ir
        self.issues: list[Issue] = []
        self.kind: dict[str, str] = {}  # name -> what it is
        self.axis_type: dict[str, str] = {a.name: a.type for a in ir.axes}
        self.axis_spec = {a.name: a for a in ir.axes}
        self.field_axes = {f.name: tuple(f.axes) for f in ir.fields}
        self.vector_over = {p.name: p.over for p in ir.params if p.over is not None}
        self.symbols: dict[str, sp.Basic] = {}
        self._dims_cache: dict[str, frozenset[str] | None] = {}
        self._defining: list[str] = []
        self.used: set[str] = set()

    def add(self, level: Level, where: str, message: str) -> None:
        issue = Issue(level, where, message)
        if issue not in self.issues:  # a formula checked from two places is reported once
            self.issues.append(issue)

    # ------------------------------------------------------------------ names
    def register_names(self) -> None:
        ir = self.ir
        groups = [
            ("axis", [a.name for a in ir.axes]),
            ("parameter", [p.name for p in ir.params]),
            ("field", [f.name for f in ir.fields]),
            ("compartment", [c.name for c in ir.compartments]),
            ("definition", [d.name for d in ir.definitions]),
            ("integral", [i.name for i in ir.integrals]),
            ("transitions", [t.name for t in ir.transitions]),
            ("observable", [o.name for o in ir.observables]),
            ("extinction group", [e.name for e in ir.extinctions]),
        ]
        # A model name may match a function name: 'beta' (a burst size) and 'beta(z, ...)' (a
        # shape) are told apart by the call syntax. Only time 't' is reserved.
        reserved = RESERVED_NAMES
        for what, names in groups:
            for n in names:
                if not n.isidentifier():
                    self.add(
                        "error",
                        f"{what} '{n}'",
                        "names must be letters, digits and _ and must not start with a digit.",
                    )
                if keyword.iskeyword(n):
                    self.add(
                        "error",
                        f"{what} '{n}'",
                        f"'{n}' is a reserved word of the formula language; choose another "
                        f"name (e.g. '{n}_').",
                    )
                if n in reserved:
                    self.add(
                        "error",
                        f"{what} '{n}'",
                        f"'{n}' is reserved (it means time); choose another name.",
                    )
                if n in self.kind:
                    self.add(
                        "error",
                        f"{what} '{n}'",
                        f"the name '{n}' is already used by a "
                        f"{self.kind[n]}; every name must be unique.",
                    )
                else:
                    self.kind[n] = what
        for n, what in self.kind.items():
            if what == "parameter" and n in self.vector_over:
                self.symbols[n] = sp.IndexedBase(n)
            elif what in ("axis", "parameter", "field", "compartment", "definition", "integral"):
                self.symbols[n] = sp.Symbol(n, real=True)
        self.symbols["t"] = sp.Symbol("t", real=True)
        # a field read at one end of each of its continuous axes, D(z=lower) (IR 0.10)
        self.end_of: dict[str, tuple[str, str, str]] = {}
        for f in ir.fields:
            for ax in f.axes:
                spec = self.axis_spec.get(ax)
                if spec is None or spec.type != "continuous" or not spec.faces:
                    continue
                for side, at in (("lower", spec.faces[0]), ("upper", spec.faces[-1])):
                    key = end_key(f.name, ax, side)
                    self.kind[key] = "end"
                    self.end_of[key] = (f.name, ax, side)
                    self.symbols[key] = sp.Symbol(key, real=True)
                    self.symbols[end_key(f.name, ax, f"#{float(at)!r}")] = self.symbols[key]

    def end_axes(self, key: str) -> tuple[str, ...]:
        field, ax, _ = self.end_of[key]
        return tuple(a for a in self.field_axes[field] if a != ax)

    # ------------------------------------------------------------------ dims of formulas
    def parse(self, text: str, where: str) -> sp.Expr | None:
        try:
            expr = parse(text, self.symbols)
        except FormulaError as e:
            self.add("error", where, str(e))
            return None
        bad = _not_finite_real(expr)
        if bad is not None:
            self.add(
                "error",
                where,
                f"'{text}' contains {bad}, so it is not a finite real number (check for a "
                "division by zero, or a log or square root of a negative number).",
            )
            return None
        return expr

    def dims(self, text: str, where: str) -> frozenset[str] | None:
        """Axes the formula varies over (``None`` if it could not be checked)."""
        expr = self.parse(text, where)
        if expr is None:
            return None
        out: set[str] = set()
        ok = True
        for sh in shapes_used(expr):
            c = sh.args[0]
            if not (
                isinstance(c, sp.Symbol)
                and self.kind.get(c.name) == "axis"
                and self.axis_type.get(c.name) == "continuous"
            ):
                self.add(
                    "error",
                    where,
                    f"{sh.shape_name}(...) needs a continuous axis as its first argument, "
                    f"not '{c}'.",
                )
                ok = False
            elif sh.shape_name == "point" and sh.args[1].is_number:
                faces = self.axis_spec[c.name].faces
                at = float(sh.args[1])
                if faces and not (faces[0] <= at <= faces[-1]):
                    self.add(
                        "error",
                        where,
                        f"point({c.name}, at={at:g}) is outside the axis {c.name}, which runs "
                        f"from {faces[0]:g} to {faces[-1]:g}.",
                    )
                    ok = False
        jumps = sorted(
            {NON_SMOOTH[type(f)] for f in expr.atoms(sp.Function) if type(f) in NON_SMOOTH}
        )
        if jumps and (where, tuple(jumps)) not in self._noted_jumps:
            self._noted_jumps.add((where, tuple(jumps)))
            self.add(
                "note",
                where,
                f"{', '.join(jumps)} jumps: the solver must cut its steps at each jump, and the "
                "derivative there is zero or undefined, so fitting does not see it. Prefer a "
                "smooth form if one exists.",
            )
        for g in grads_used(expr):
            f, ax = g.args
            if not (isinstance(f, sp.Symbol) and self.kind.get(f.name) == "field"):
                self.add("error", where, f"grad({f}, {ax}): '{f}' is not a field.")
                ok = False
            elif not (
                isinstance(ax, sp.Symbol)
                and ax.name in self.field_axes[f.name]
                and self.axis_type.get(ax.name) == "continuous"
            ):
                cont = [a for a in self.field_axes[f.name] if self.axis_type.get(a) == "continuous"]
                self.add(
                    "error",
                    where,
                    f"grad({f}, {ax}): '{ax}' is not a continuous axis of '{f}' "
                    f"(its continuous axes: {cont}).",
                )
                ok = False
        indexed = list(expr.atoms(sp.Indexed))
        for ind in indexed:
            base, idx = str(ind.base), tuple(str(i) for i in ind.indices)
            self.used.add(base)
            want = self.vector_over.get(base)
            want_t = want if isinstance(want, tuple) else (want,)
            if idx != want_t:
                self.add(
                    "error",
                    where,
                    f"'{base}' has one value per level of {_axes_text(want)}; "
                    f"write '{base}[{', '.join(want_t)}]', not '{ind}'.",
                )
                ok = False
            out.update(i for i in idx if i in self.axis_type)
        # Swap each k[s] for a fresh placeholder (not a number: '1 - cost[s]' -> 0 would erase
        # the rest of a product) and keep only the model's own names.
        stand_in = {ind: sp.Dummy() for ind in indexed}
        plain = expr.xreplace(stand_in).free_symbols - set(stand_in.values())
        for s in plain:
            n = s.name
            self.used.add(n)
            what = self.kind.get(n, "time" if n == "t" else None)
            if what == "axis":
                if self.axis_type[n] == "categorical":
                    self.add(
                        "error",
                        where,
                        f"'{n}' is a categorical axis; it can only be used as "
                        f"an index, like k[{n}].",
                    )
                    ok = False
                else:
                    out.add(n)
            elif what == "parameter":
                if n in self.vector_over:
                    over = self.vector_over[n]
                    over_t = over if isinstance(over, tuple) else (over,)
                    self.add(
                        "error",
                        where,
                        f"'{n}' has one value per level of {_axes_text(over)}; write "
                        f"'{n}[{', '.join(over_t)}]'.",
                    )
                    ok = False
            elif what == "field":
                out.update(self.field_axes[n])
            elif what == "end":
                out.update(self.end_axes(n))
            elif what in ("definition", "integral"):
                d = self.named_dims(n)
                if d is None:
                    ok = False
                else:
                    out.update(d)
            elif what in ("compartment", "time"):
                pass
            else:
                self.add("error", where, f"'{n}' is a {what} and cannot be used inside a formula.")
                ok = False
        return frozenset(out) if ok else None

    def forbid_state(self, text: str, where: str, what: str) -> None:
        """Initial values and dose amounts are fixed before the run: parameters and
        coordinates only, no states, integrals or time (also through definitions)."""
        try:
            expr = parse(text, self.symbols)
        except FormulaError:
            return  # already reported
        seen: set[str] = set()
        todo = [s.name for s in expr.free_symbols if isinstance(s, sp.Symbol)]
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            k = self.kind.get(n, "time" if n == "t" else None)
            if k in ("field", "compartment", "integral", "time", "end"):
                desc = "field at one end of an axis" if k == "end" else k
                self.add(
                    "error",
                    where,
                    f"{what} uses '{n}' (a {desc}); it may use parameters and coordinates only.",
                )
                return
            if k == "definition":
                d = next(x for x in self.ir.definitions if x.name == n)
                try:
                    sub = parse(d.expr, self.symbols)
                except FormulaError:
                    continue
                todo += [s.name for s in sub.free_symbols if isinstance(s, sp.Symbol)]

    def named_dims(self, name: str) -> frozenset[str] | None:
        if name in self._dims_cache:
            return self._dims_cache[name]
        if name in self._defining:
            cycle = " -> ".join([*self._defining[self._defining.index(name) :], name])
            self.add("error", f"definition '{name}'", f"circular definition: {cycle}.")
            return None
        self._defining.append(name)
        try:
            if self.kind[name] == "definition":
                d = next(x for x in self.ir.definitions if x.name == name)
                result = self.dims(d.expr, f"definition '{name}'")
            else:
                spec = next(x for x in self.ir.integrals if x.name == name)
                where = f"integral '{name}'"
                inner = self.dims(spec.integrand, where)
                if spec.where:
                    if self.check_slice_of(spec, where):
                        fixed = {a for a, _ in spec.where}
                        if fixed & set(spec.over):
                            self.add(
                                "error",
                                where,
                                f"{sorted(fixed & set(spec.over))} are fixed by the slice; "
                                "integrate over the axes it leaves free.",
                            )
                        if inner is not None:
                            inner = inner - frozenset(fixed)
                    else:
                        inner = None
                self.check_integral_weights(spec, where)
                result = None
                if inner is not None:
                    missing = [a for a in spec.over if a not in inner]
                    unknown = [a for a in spec.over if a not in self.axis_type]
                    if unknown:
                        self.add("error", where, f"unknown axis {unknown[0]!r} in 'over'.")
                    elif missing:
                        tail = (
                            f"Integrate only over axes of the integrand: {sorted(inner)}."
                            if inner
                            else "The integrand does not vary over any axis; is a field missing?"
                        )
                        self.add(
                            "error",
                            where,
                            f"the integrand does not vary over "
                            f"{', '.join(repr(m) for m in missing)}, so integrating over it "
                            f"is probably a mistake. {tail}",
                        )
                    else:
                        result = inner - frozenset(spec.over)
        finally:
            self._defining.pop()
        self._dims_cache[name] = result
        return result

    def check_slice_of(self, spec, where: str) -> bool:
        """The slice of an integral: of its integrand's field, or (IR 0.6) of every field
        an integrand formula uses."""
        if self.kind.get(spec.integrand.strip()) == "field":
            return self.check_where(spec.integrand.strip(), spec.where, where)
        try:
            fields = sorted(self._fields_in(parse(spec.integrand, self.symbols)))
        except FormulaError:
            return False
        if not fields:
            self.add("error", where, "a slice needs a field in the integrand to fix axes of.")
            return False
        return all([self.check_where(f, spec.where, where) for f in fields])

    def check_integral_weights(self, spec, where: str) -> None:
        """An integral of weighted fields is an amount: their weights must agree and every
        axis the weights vary over must be integrated over or fixed by the slice."""
        try:
            expr = parse(spec.integrand, self.symbols)
        except FormulaError:
            return
        fields = self._fields_in(expr)
        weights = {f.name: f.weights for f in self.ir.fields}
        found = {weights[f] for f in fields if weights.get(f)}
        if not found:
            return
        if len(found) > 1 or any(not weights.get(f) for f in fields):
            self.add(
                "error",
                where,
                f"the integrand mixes fields with different weights ({sorted(fields)}); "
                "integrate each separately.",
            )
            return
        wd = self.dims(next(iter(found)), where) or frozenset()
        covered = set(spec.over) | {a for a, _ in spec.where}
        if not wd <= covered:
            self.add(
                "error",
                where,
                f"the weights vary over {sorted(wd)}; integrate over (or fix with a slice) "
                f"{sorted(wd - covered)} too, so the result is an amount.",
            )

    def _fields_in(self, expr: sp.Expr, _seen: frozenset[str] = frozenset()) -> set[str]:
        out = set()
        for sym in expr.free_symbols:
            k = self.kind.get(sym.name)
            if k == "field":
                out.add(sym.name)
            elif k == "end":
                out.add(self.end_of[sym.name][0])
            elif k == "definition" and sym.name not in _seen:
                d = next(x for x in self.ir.definitions if x.name == sym.name)
                try:
                    sub = parse(d.expr, self.symbols)
                except FormulaError:
                    continue
                out |= self._fields_in(sub, _seen | {sym.name})
        return out

    def require_within(self, dims, allowed: tuple[str, ...], where: str, target_desc: str) -> None:
        if dims is None:
            return
        extra = sorted(dims - set(allowed))
        if extra:
            self.add(
                "error",
                where,
                f"this formula varies over {extra}, but {target_desc} "
                f"{'has axes ' + str(list(allowed)) if allowed else 'is a single number'}. "
                "Integrate or sum over the extra axes first (declare an integral).",
            )

    # ------------------------------------------------------------------ sections
    def check_axes(self) -> None:
        for a in self.ir.axes:
            w = f"axis '{a.name}'"
            if a.type == "continuous":
                f = np.asarray(a.faces, dtype=float)
                if not all(_is_number(x) for x in a.faces):
                    self.add("error", w, "faces must be finite numbers.")
                elif f.size < 2:
                    self.add("error", w, "needs at least 2 faces (1 cell).")
                elif not np.all(np.diff(f) > 0):
                    self.add("error", w, "faces must be strictly increasing.")
                if a.kind not in AXIS_KINDS:
                    self.add("error", w, f"kind must be one of {', '.join(AXIS_KINDS)}.")
                if a.labels:
                    self.add("error", w, "a continuous axis has no labels.")
            elif a.type == "categorical":
                if not a.labels:
                    self.add("error", w, "a categorical axis needs at least one label.")
                elif len(set(a.labels)) != len(a.labels):
                    self.add("error", w, "labels must be unique.")
                if a.faces:
                    self.add("error", w, "a categorical axis has no faces.")
            else:
                self.add("error", w, "type must be 'continuous' or 'categorical'.")

    def check_params(self) -> None:
        for p in self.ir.params:
            w = f"parameter '{p.name}'"
            values = p.value if isinstance(p.value, tuple) else (p.value,)
            if p.value is not None and not all(_is_number(v) for v in values):
                self.add("error", w, "values must be finite numbers (not true/false, inf or NaN).")
                continue
            if p.over is None:
                if p.value is not None and not isinstance(p.value, (int, float)):
                    self.add(
                        "error",
                        w,
                        "a scalar parameter needs a single number; to give one "
                        "value per category, set 'over' to a categorical axis.",
                    )
            else:
                over = p.over if isinstance(p.over, tuple) else (p.over,)
                bad = [a for a in over if self.axis_type.get(a) != "categorical"]
                if bad or len(set(over)) != len(over):
                    self.add(
                        "error", w, f"'over' must be different categorical axes, not {p.over!r}."
                    )
                    continue
                n = int(np.prod([len(self.axis_spec[a].labels) for a in over]))
                if p.value is not None and (not isinstance(p.value, tuple) or len(p.value) != n):
                    self.add(
                        "error",
                        w,
                        f"needs {n} values, one per level of {_axes_text(p.over)}.",
                    )

    def check_fields(self) -> None:
        for f in self.ir.fields:
            w = f"field '{f.name}'"
            if not f.axes:
                self.add(
                    "error",
                    w,
                    "a field needs at least one axis; use a compartment for a "
                    "single well-mixed quantity.",
                )
            if len(set(f.axes)) != len(f.axes):
                self.add("error", w, "an axis appears twice.")
            for ax in f.axes:
                if ax not in self.axis_type:
                    self.add("error", w, f"unknown axis '{ax}'.")
            if f.weights is not None:
                cat = tuple(a for a in f.axes if self.axis_type.get(a) == "categorical")
                self.check_size(
                    f.weights, f"{w}, weights", cat, f"the categorical axes {list(cat)}"
                )
        for c in self.ir.compartments:
            if c.volume is not None:
                self.check_size(c.volume, f"compartment '{c.name}', volume", (), "a single number")

    def check_size(self, text: str, where: str, allowed: tuple[str, ...], desc: str) -> None:
        """Weights and volumes: parameters only, varying at most over ``allowed``, > 0."""
        self.require_within(self.dims(text, where), allowed, where, desc)
        self.forbid_state(text, where, "a weight or volume")
        expr = self.parse(text, where)
        if expr is None:
            return
        for v in self.values_of(expr):
            if not v > 0:
                self.add("error", where, f"must be > 0 everywhere (found {v:g}).")
                return

    def values_of(self, expr: sp.Expr) -> list[float]:
        """Every value a formula of numbers and parameters takes (empty if it uses
        anything else)."""
        subs = {}
        for p in self.ir.params:
            if p.over is None and isinstance(p.value, (int, float)):
                subs[sp.Symbol(p.name, real=True)] = p.value
        variants = [expr.subs(subs)]
        for ind in expr.atoms(sp.Indexed):
            p = next((x for x in self.ir.params if x.name == str(ind.base)), None)
            if p is None or not isinstance(p.value, tuple):
                return []
            variants = [v.subs(ind, val) for v in variants for val in p.value]
        out = []
        for v in variants:
            if not v.is_number:
                return []
            out.append(float(v) if _not_finite_real(v) is None else float("nan"))
        return out

    def check_where(self, target: str, where, w: str, *, all_axes: bool = False) -> bool:
        """A slice of field ``target``: categorical axes of it, known labels, each once."""
        if not where:
            return True
        if self.kind.get(target) != "field":
            self.add("error", w, f"'{target}' is not a field; only fields can be sliced.")
            return False
        ok = True
        axes = [a for a, _ in where]
        if len(set(axes)) != len(axes):
            self.add("error", w, f"an axis is fixed twice in the slice {dict(where)}.")
            ok = False
        for ax, lab in where:
            if ax not in self.field_axes.get(target, ()) or self.axis_type.get(ax) != (
                "categorical"
            ):
                self.add("error", w, f"'{ax}' is not a categorical axis of '{target}'.")
                ok = False
            elif lab not in self.axis_spec[ax].labels:
                self.add(
                    "error",
                    w,
                    f"'{lab}' is not a category of '{ax}' (it has "
                    f"{list(self.axis_spec[ax].labels)}).",
                )
                ok = False
        if all_axes and ok:
            free = [a for a in self.field_axes.get(target, ()) if a not in dict(where)]
            if free:
                self.add("error", w, f"fix every axis of '{target}' (also {free}).")
                ok = False
        return ok

    def check_definitions_and_integrals(self) -> None:
        for d in self.ir.definitions:
            self.named_dims(d.name)
        for i in self.ir.integrals:
            self.named_dims(i.name)

    def check_transitions(self) -> None:
        for tr in self.ir.transitions:
            w = f"transitions '{tr.name}'"
            along = tuple(tr.axis) if isinstance(tr.axis, tuple) else (tr.axis,)
            specs = [self.axis_spec.get(a) for a in along]
            bad = [
                a
                for a, ax in zip(along, specs, strict=True)
                if ax is None or ax.type != "categorical"
            ]
            if bad:
                self.add("error", w, f"{bad} must be categorical axes.")
                continue
            if len(set(along)) != len(along):
                self.add(
                    "error",
                    w,
                    f"axes {list(along)} repeat; a joint transition needs different axes.",
                )
                continue
            joint = len(along) > 1
            seen = set()
            for frm, to, rate in tr.rates:
                move = f"{_category(frm)} -> {_category(to)}"
                for cat in (frm, to):
                    labels = cat if isinstance(cat, tuple) else (cat,)
                    if joint != isinstance(cat, tuple) or len(labels) != len(along):
                        self.add(
                            "error",
                            w,
                            f"category {_category(cat)!r} must give one label of each of "
                            f"{list(along)}.",
                        )
                        continue
                    for a, ax, lab in zip(along, specs, labels, strict=True):
                        if lab not in ax.labels:
                            self.add(
                                "error",
                                w,
                                f"axis '{a}' has no category '{lab}' (it has {list(ax.labels)}).",
                            )
                if frm == to:
                    self.add(
                        "error",
                        w,
                        f"'{move}': no self-transitions; the diagonal is "
                        "derived automatically so that cells are conserved.",
                    )
                if (frm, to) in seen:
                    self.add("error", w, f"'{move}' is given twice.")
                seen.add((frm, to))
                d = self.dims(rate, f"{w}, rate {move}")
                own = sorted(set(along) & (d or set()))
                if own:
                    self.add(
                        "error",
                        f"{w}, rate {move}",
                        f"the rate varies along {own}, the axis it moves between. A rate is "
                        f"the rate of leaving '{_category(frm)}' for '{_category(to)}': it may "
                        "vary over the field's other axes, not along this one. To depend on "
                        "the whole population, use a total (declare an integral).",
                    )
                expr = self.parse(rate, f"{w}, rate {move}")
                if expr is not None and expr.is_number and float(expr) < 0:
                    self.add("error", w, f"rate {move} is negative ({rate}); rates must be >= 0.")

    def check_terms(self) -> None:
        rates = [t.target for t in self.ir.terms if t.kind == "rate"]
        for c in sorted({c for c in rates if rates.count(c) > 1}):
            self.add(
                "error",
                f"compartment '{c}'",
                "has more than one equation ('rate' term); write the whole right-hand side in one.",
            )
        schemes: dict[tuple[str, str], set[str]] = {}
        for t in self.ir.terms:
            if t.kind == "advection":
                schemes.setdefault((t.target, str(t.axis)), set()).add(t.scheme)
        for (f, ax), sc in sorted(schemes.items()):
            if len(sc) > 1:
                self.add(
                    "error",
                    f"field '{f}' along '{ax}'",
                    f"advected with different schemes ({', '.join(sorted(sc))}); the velocities "
                    "are added and upwinded together, so use one scheme.",
                )
        for i, t in enumerate(self.ir.terms):
            w = f"term {i + 1} ({t.kind} on '{t.target}')"
            if t.kind not in TERM_KINDS:
                self.add("error", w, f"unknown kind; use one of {', '.join(TERM_KINDS[:-1])}.")
                continue
            if t.where:
                if t.kind not in ("reaction", "hazard", "switch", "mutate"):
                    self.add(
                        "error", w, "only reaction, hazard and transition terms act on a slice."
                    )
                    continue
                if not self.check_where(t.target, t.where, w):
                    continue
                if t.kind in ("switch", "mutate"):
                    along = tuple(t.axis) if isinstance(t.axis, tuple) else (t.axis,)
                    clash = sorted({a for a, _ in t.where} & set(along))
                    if clash:
                        self.add(
                            "error",
                            w,
                            f"the slice fixes {clash}, the axis it moves along; fix other axes "
                            "(e.g. move between organs only where sub='vascular').",
                        )
                        continue
            if t.kind == "noise":
                self.add(
                    "error",
                    w,
                    "stochastic ('noise') terms are planned for TenQSP v2 and are "
                    "not available yet.",
                )
                continue
            target_kind = self.kind.get(t.target)
            if target_kind not in ("field", "compartment"):
                self.add("error", w, f"'{t.target}' is not a field or compartment.")
                continue
            self.used.add(t.target)
            if t.kind == "rate":
                if target_kind != "compartment":
                    self.add(
                        "error",
                        w,
                        "'rate' terms are for compartments; for a field use a 'reaction' term.",
                    )
                    continue
                self.require_within(self.dims(t.expr, w), (), w, f"compartment '{t.target}'")
                continue
            if target_kind != "field":
                self.add(
                    "error",
                    w,
                    f"'{t.kind}' terms act on fields; for compartment "
                    f"'{t.target}' use a 'rate' term.",
                )
                continue
            axes = self.field_axes[t.target]
            if t.kind in ("advection", "diffusion"):
                if t.axis not in axes or self.axis_type.get(t.axis) != "continuous":
                    self.add(
                        "error",
                        w,
                        f"needs a continuous axis of '{t.target}' "
                        f"({[a for a in axes if self.axis_type.get(a) == 'continuous']}), "
                        f"not {t.axis!r}.",
                    )
                    continue
                if t.kind == "advection" and t.scheme not in ("erlang", "sharp"):
                    self.add("error", w, f"scheme must be 'erlang' or 'sharp', not {t.scheme!r}.")
            if t.kind in ("switch", "mutate"):
                along = tuple(t.axis) if isinstance(t.axis, tuple) else (t.axis,)
                if any(a not in axes or self.axis_type.get(a) != "categorical" for a in along):
                    self.add(
                        "error",
                        w,
                        f"needs categorical axes of '{t.target}', not {t.axis!r} "
                        f"('{t.target}' has {list(axes)}).",
                    )
                    continue
                tr = next((x for x in self.ir.transitions if x.name == t.transitions), None)
                if tr is None:
                    self.add("error", w, f"unknown transitions {t.transitions!r}.")
                elif tr.axis != t.axis:
                    self.add(
                        "error",
                        w,
                        f"transitions '{tr.name}' are on {_category(tr.axis)!r}, not "
                        f"{_category(t.axis)!r}.",
                    )
                else:
                    others = tuple(a for a in axes if a not in along)
                    for frm, to, rate in tr.rates:
                        d = self.dims(rate, f"transitions '{tr.name}'")
                        self.require_within(
                            None if d is None else d - set(along),  # along: reported above
                            others,
                            f"{w}, rate {_category(frm)} -> {_category(to)} of '{tr.name}'",
                            f"'{t.target}' (apart from {list(along)})",
                        )
                if t.kind == "switch":
                    continue  # acts on the target itself
            d = self.dims(t.expr, w)
            self.require_within(d, axes, w, f"field '{t.target}'")
            if t.kind == "advection" and d is not None:
                self.check_one_way(t, w)

    def _axes_and_state(self, expr: sp.Expr) -> tuple[frozenset[str], bool]:
        """Axes a parsed expression varies over, and whether it depends on the state or time
        (through definitions too). Assumes its names were already checked."""
        axes: set[str] = {str(j) for i in expr.atoms(sp.Indexed) for j in i.indices}
        state = False
        todo = [s.name for s in expr.free_symbols if isinstance(s, sp.Symbol)]
        seen: set[str] = set()
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            k = self.kind.get(n, "time" if n == "t" else None)
            if k == "field":
                axes.update(self.field_axes[n])
                state = True
            elif k == "end":
                axes.update(self.end_axes(n))
                state = True
            elif k in ("compartment", "time"):
                state = True
            elif k == "axis" and self.axis_type.get(n) == "continuous":
                axes.add(n)
            elif k in ("definition", "integral"):
                axes.update(self.named_dims(n) or ())
                if k == "integral":
                    state = True
                else:
                    d = next(x for x in self.ir.definitions if x.name == n)
                    try:
                        sub = parse(d.expr, self.symbols)
                    except FormulaError:
                        continue
                    todo += [s.name for s in sub.free_symbols if isinstance(s, sp.Symbol)]
        return frozenset(axes), state

    def check_switches(self) -> None:
        """Conditions that switch cell by cell are evaluated where they stand: note it (ADR
        0006). Conditions on time, compartments and totals are handled exactly by the solver
        and need no note."""
        ir = self.ir
        places = [(f"definition '{d.name}'", d.expr) for d in ir.definitions]
        places += [(f"integral '{i.name}'", i.integrand) for i in ir.integrals]
        places += [
            (f"term {k + 1} ({t.kind} on '{t.target}')", t.expr) for k, t in enumerate(ir.terms)
        ]
        for b in ir.boundaries:
            w = f"boundary {b.kind} of '{b.target}' at {b.axis}={b.side}"
            places += [(w, b.value), (w, b.h)]
        for tr in ir.transitions:
            places += [(f"transitions '{tr.name}'", r) for _, _, r in tr.rates]
        for w, text in places:
            if not text or " if " not in f" {text} ":
                continue
            try:
                expr = parse(text, self.symbols)
            except FormulaError:
                continue  # reported where the formula is checked
            cell = []
            for a in expr.atoms(sp.Rel):
                axes, state = self._axes_and_state(a)
                if axes and state:
                    cell.append(f"'{a}' (varies over {sorted(axes)})")
            if cell:
                self.add(
                    "note",
                    w,
                    f"{', '.join(sorted(cell))} switches cell by cell: each cell jumps when its "
                    "own condition flips, and the solver steps through those jumps (it stops "
                    "exactly only for conditions on time, compartments and totals). Prefer a "
                    "smooth form (e.g. sigmoid) if the jumps slow the solve.",
                )

    def check_gradients(self) -> None:
        """``grad(F, x)`` lives on the cell faces, so only an advection velocity along ``x``
        (written directly in the term, not through a definition) may use it."""
        ir = self.ir
        places: list[tuple[str, str, str | None]] = []  # (where, text, allowed axis)
        for d in ir.definitions:
            places.append((f"definition '{d.name}'", d.expr, None))
        for i in ir.integrals:
            places.append((f"integral '{i.name}'", i.integrand, None))
        for tr in ir.transitions:
            places += [(f"transitions '{tr.name}'", r, None) for _, _, r in tr.rates]
        for k, t in enumerate(ir.terms):
            allowed = t.axis if t.kind == "advection" else None
            places.append((f"term {k + 1} ({t.kind} on '{t.target}')", t.expr, allowed))
        for b in ir.boundaries:
            w = f"boundary {b.kind} of '{b.target}' at {b.axis}={b.side}"
            places += [(w, b.value, None), (w, b.h, None)]
        for o in ir.observables:
            places.append((f"observable '{o.name}'", o.expr, None))
        for it in ir.inits:
            places += [(f"init of '{it.target}'", x, None) for x in (it.value, it.total, it.shape)]
        for ds in ir.doses:
            places += [(f"dose into '{ds.target}'", a, None) for a in (*ds.amounts, ds.shape)]
        for k, tr in enumerate(ir.transfers):
            places.append((f"transfer {k + 1} from '{tr.source}'", tr.rate, None))
        places += [(f"weights of '{f.name}'", f.weights, None) for f in ir.fields]
        places += [(f"volume of '{c.name}'", c.volume, None) for c in ir.compartments]
        for w, text, allowed in places:
            if not text or "grad" not in text:
                continue
            try:
                expr = parse(text, self.symbols)
            except FormulaError:
                continue  # reported where the formula is checked
            for g in grads_used(expr):
                ax = g.args[1]
                if allowed is None:
                    self.add(
                        "error",
                        w,
                        f"grad({g.args[0]}, {ax}) (grad_{ax}({g.args[0]})) can only be used in "
                        f"a velocity along '{ax}' (gradients are taken at the cell faces, where "
                        "velocities live); write it inside the transport term, e.g. "
                        f"-d_{ax}(chi*grad_{ax}({g.args[0]})*B) (in Python, "
                        f"m.advect(B, axis={ax}, velocity=chi*tq.grad({g.args[0]}, {ax}))).",
                    )
                elif str(ax) != allowed:
                    self.add(
                        "error",
                        w,
                        f"grad({g.args[0]}, {ax}) is a gradient along '{ax}', but this velocity "
                        f"moves along '{allowed}'; only a gradient along the same axis can be "
                        "used.",
                    )

    def check_one_way(self, t, w: str) -> None:
        ax = self.axis_spec[t.axis]
        if ax.kind != "one_way":
            return
        expr = self.parse(t.expr, w)
        if expr is None:
            return
        expr = self.expand_definitions(expr)  # a named formula of parameters is no state
        if expr is None:
            return
        states = {
            n for n, k in self.kind.items() if k in ("field", "compartment", "integral", "end")
        }
        if {s.name for s in expr.free_symbols} & (states | {"t"}):
            self.add(
                "note",
                w,
                f"axis '{t.axis}' is one_way, so the velocity must stay >= 0. It "
                "depends on the state or time, so it is checked while solving.",
            )
            return
        values = self.sample_values(expr, ax)
        if values is not None and np.nanmin(values) < 0:
            self.add(
                "error",
                w,
                f"axis '{t.axis}' is one_way (e.g. age), but the velocity "
                f"'{t.expr}' is negative somewhere (min {np.nanmin(values):.3g}). Use kind "
                "'bidirectional' if material can move both ways.",
            )

    def expand_definitions(self, expr: sp.Expr, depth: int = 0) -> sp.Expr | None:
        """``expr`` with every definition replaced by its formula (None if one cannot be
        read, or definitions nest too deep, i.e. a cycle, reported elsewhere)."""
        defs = {d.name: d.expr for d in self.ir.definitions}
        named = [s for s in expr.free_symbols if isinstance(s, sp.Symbol) and s.name in defs]
        if not named:
            return expr
        if depth > 50:
            return None
        subs = {}
        for s in named:
            try:
                subs[s] = parse(defs[s.name], self.symbols)
            except FormulaError:
                return None
        return self.expand_definitions(expr.subs(subs), depth + 1)

    def sample_values(self, expr: sp.Expr, ax) -> np.ndarray | None:
        """Evaluate a parameter/coordinate expression at the axis faces for every category."""
        subs = {}
        for p in self.ir.params:
            if p.over is None and isinstance(p.value, (int, float)):
                subs[sp.Symbol(p.name, real=True)] = p.value
        expr = expr.subs(subs)
        indexed = list(expr.atoms(sp.Indexed))
        variants = [expr]
        for ind in indexed:
            p = next(x for x in self.ir.params if x.name == str(ind.base))
            if not isinstance(p.value, tuple):  # declared without values
                return None
            variants = [v.subs(ind, val) for v in variants for val in p.value]
        out = []
        faces = np.asarray(ax.faces, dtype=float)
        for v in variants:
            free = v.free_symbols
            if free - {sp.Symbol(ax.name, real=True)}:
                return None
            f = sp.lambdify(sp.Symbol(ax.name, real=True), v, ["scipy", "numpy"])
            out.append(np.broadcast_to(f(faces), faces.shape))
        return np.concatenate(out) if out else None

    def check_boundaries(self) -> None:
        seen = set()
        for b in self.ir.boundaries:
            w = f"boundary {b.kind} of '{b.target}' at {b.axis}={b.side}"
            if self.kind.get(b.target) != "field":
                self.add("error", w, f"'{b.target}' is not a field.")
                continue
            axes = self.field_axes[b.target]
            if b.axis not in axes or self.axis_type.get(b.axis) != "continuous":
                self.add("error", w, f"'{b.axis}' is not a continuous axis of '{b.target}'.")
                continue
            if b.side not in ("lower", "upper"):
                self.add("error", w, "side must be 'lower' or 'upper'.")
                continue
            if b.kind not in BOUNDARY_KINDS:
                self.add("error", w, f"kind must be one of {', '.join(BOUNDARY_KINDS)}.")
                continue
            key = (b.target, b.axis, b.side)
            if key in seen:
                self.add("error", w, "this end already has a boundary condition.")
            seen.add(key)
            rest = tuple(a for a in axes if a != b.axis)
            if b.kind == "flux":
                self.check_flux_condition(b, w, rest)
            if b.kind in ("inflow", "dirichlet", "robin"):
                self.require_within(
                    self.dims(b.value, w),
                    rest,
                    w,
                    f"the boundary of '{b.target}' at {b.axis}={b.side}",
                )
            if b.kind == "robin":
                self.require_within(
                    self.dims(b.h, w), rest, w, f"the boundary of '{b.target}' at {b.axis}={b.side}"
                )
            if b.to is not None:
                if b.kind not in ("outflow", "dirichlet", "robin"):
                    self.add(
                        "error",
                        w,
                        "'to' (routing to a compartment) is for outflow, dirichlet and robin "
                        "boundaries.",
                    )
                elif self.kind.get(b.to) != "compartment":
                    self.add(
                        "error",
                        w,
                        f"a boundary can be routed to a compartment; '{b.to}' is not one.",
                    )
            if b.kind == "outflow":
                adv = [
                    t
                    for t in self.ir.terms
                    if t.kind == "advection" and t.target == b.target and t.axis == b.axis
                ]
                if not adv:
                    self.add(
                        "error",
                        w,
                        "outflow is carried by advection, but there is no "
                        f"advection term for '{b.target}' along '{b.axis}'.",
                    )

    def check_flux_condition(self, b, w: str, rest: tuple[str, ...]) -> None:
        """A condition as written in a paper: the flux of the field along the axis at this end
        equals ``b.value``, in which the field's own name means its value at the end."""
        transport = [
            t
            for t in self.ir.terms
            if t.target == b.target and t.axis == b.axis and t.kind in ("advection", "diffusion")
        ]
        if not transport:
            self.add(
                "error",
                w,
                f"this condition sets the flux of '{b.target}' along '{b.axis}', but "
                f"'{b.target}' does not move along '{b.axis}' (no -d_{b.axis}(v*{b.target}) or "
                f"d_{b.axis}(D*d_{b.axis}({b.target})) in its equation).",
            )
            return
        if b.to is not None:
            self.add("error", w, "a flux condition is not routed anywhere ('to' is not used).")
        self.dims(b.value, w)  # names and indexing (the field's own axes are checked below)
        expr = self.parse(b.value, w)
        if expr is None:
            return
        expr = self.expand_definitions(expr)
        if expr is None:
            return
        u = sp.Dummy("u")
        own = {self.symbols[b.target]: u, self.symbols[end_key(b.target, b.axis, b.side)]: u}
        expr = expr.xreplace(own)
        slope = sp.diff(expr, u)
        if slope.has(u):
            self.add(
                "error",
                w,
                f"'{b.value}' is not linear in {b.target} (its value at this end): a boundary "
                "condition is written as a flux equal to terms that are each proportional to "
                f"{b.target} or free of it, e.g. -D*d_{b.axis}({b.target}) = h*(Cout - "
                f"{b.target}).",
            )
            return
        other = expr.xreplace({u: sp.Integer(0)}) + slope
        axes, _ = self._axes_and_state(other)
        extra = sorted(a for a in axes if a not in rest)
        if b.axis in extra:
            self.add(
                "error",
                w,
                f"'{b.value}' varies along '{b.axis}', but the condition holds at one end of "
                f"it; read another field at that end as F({b.axis}={b.side}).",
            )
        elif extra:
            self.add(
                "error",
                w,
                f"'{b.value}' varies over {extra}, which '{b.target}' at {b.axis}={b.side} does "
                "not have.",
            )

    def check_end_cycles(self) -> None:
        """A field's value at an end comes from its boundary condition there; one that needs
        itself (D at z=0 read in the condition that sets H at z=0, and back) is refused."""
        given = {(b.target, b.axis, b.side): b for b in self.ir.boundaries}
        integrands = {i.name: i.integrand for i in self.ir.integrals}

        def ends_in(text: str, seen: frozenset[str] = frozenset()) -> set[str]:
            """End values a formula reads: directly, through definitions or integrals."""
            try:
                e = self.expand_definitions(parse(text, self.symbols))
            except FormulaError:
                return set()
            if e is None:
                return set()
            out = set()
            for sym in e.free_symbols:
                k = self.kind.get(sym.name)
                if k == "end":
                    out.add(sym.name)
                elif k == "integral" and sym.name in integrands and sym.name not in seen:
                    out |= ends_in(integrands[sym.name], seen | {sym.name})
            return out

        needs: dict[str, set[str]] = {}
        for key, (field, ax, side) in self.end_of.items():
            # the transport along the axis is needed to compute the end value: reading the
            # same end there is a circle too
            found: set[str] = set()
            for t in self.ir.terms:
                if t.target == field and t.axis == ax:
                    found |= ends_in(t.expr)
            b = given.get((field, ax, side))
            if b is not None:  # the condition's own end enters linearly (solved for)
                for x in (b.value, b.h):
                    if x:
                        found |= ends_in(x) - {key}
            needs[key] = found
        state: dict[str, int] = {}

        def visit(k: str, path: list[str]) -> list[str] | None:
            state[k] = 1
            if k in needs.get(k, ()):
                return [k, k]
            for n in sorted(needs.get(k, ())):
                if state.get(n) == 1:
                    return [*path[path.index(n) :], n] if n in path else [*path, n]
                if state.get(n) is None:
                    cyc = visit(n, [*path, n])
                    if cyc:
                        return cyc
            state[k] = 2
            return None

        for k in sorted(needs):
            if state.get(k) is None:
                cyc = visit(k, [k])
                if cyc and len(cyc) == 2 and cyc[0] == cyc[1]:
                    field, ax, _ = self.end_of[cyc[0]]
                    self.add(
                        "error",
                        f"boundary of '{field}'",
                        f"the flow or diffusion of '{field}' along '{ax}' reads {cyc[0]}, but that "
                        "value is computed from the flow and diffusion there (and the end's "
                        "condition): it would need itself. Read another quantity in the "
                        "velocity or diffusivity.",
                    )
                elif cyc:
                    self.add(
                        "error",
                        f"boundary of '{self.end_of[k][0]}'",
                        "these values at the ends depend on each other in a circle: "
                        + " -> ".join(cyc)
                        + ". Give at least one of them a condition that does not read the "
                        "others.",
                    )
                    return

    def check_default_boundaries(self) -> None:
        """Report the defaults applied to ends without an explicit condition."""
        given = {(b.target, b.axis, b.side) for b in self.ir.boundaries}
        for f in self.ir.fields:
            for axn in f.axes:
                if self.axis_type.get(axn) != "continuous":
                    continue
                transport = [
                    t
                    for t in self.ir.terms
                    if t.target == f.name and t.axis == axn and t.kind in ("advection", "diffusion")
                ]
                kind = self.axis_spec[axn].kind
                for side in ("lower", "upper"):
                    if (f.name, axn, side) in given:
                        continue
                    if not transport:
                        continue
                    has_adv = any(t.kind == "advection" for t in transport)
                    if kind == "one_way" and has_adv and side == "lower":
                        # a field filled by doses or a starting value need not have an inflow
                        # (a dose moving down the gut): a note then, a warning otherwise
                        filled = any(d.target == f.name for d in self.ir.doses) or any(
                            i.target == f.name for i in self.ir.inits
                        )
                        self.add(
                            "note" if filled else "warning",
                            f"field '{f.name}', axis '{axn}'",
                            "no inflow given at the lower end of this one_way axis, so "
                            "nothing enters there (inflow 0)"
                            + (
                                "; the field is filled by its doses or starting value."
                                if filled
                                else ". Add an inflow condition if, for example, newly "
                                "infected cells should enter at age 0."
                            ),
                        )
                    elif kind == "one_way" and has_adv and side == "upper":
                        # what reaches this end would leave the model unnoticed: say so
                        # (decided 2026-10-28; files saved before IR 0.10 get it written in)
                        self.add(
                            "error",
                            f"field '{f.name}', axis '{axn}'",
                            f"the flow carries {f.name} to the upper end of '{axn}'; say what "
                            f"happens there: 'at {axn}=<end>: outflow' (it leaves the model, "
                            "recorded in the ledger; read it elsewhere as "
                            f"{f.name}({axn}=upper)), or a condition.",
                        )
                    else:
                        self.add(
                            "note",
                            f"field '{f.name}', axis '{axn}'",
                            f"{side} end: zero-flux (nothing crosses).",
                        )

    def check_inits(self) -> None:
        targets: dict[str, list] = {}
        for i, s in enumerate(self.ir.inits):
            w = f"init {i + 1} of '{s.target}'"
            what = self.kind.get(s.target)
            if what not in ("field", "compartment"):
                self.add("error", w, f"'{s.target}' is not a field or compartment.")
                continue
            axes = self.field_axes.get(s.target, ())
            fixed = []
            for axn, lab in s.where:
                ax = self.axis_spec.get(axn)
                if axn not in axes or ax is None or ax.type != "categorical":
                    self.add("error", w, f"'{axn}' is not a categorical axis of '{s.target}'.")
                elif lab not in ax.labels:
                    self.add(
                        "error", w, f"axis '{axn}' has no category '{lab}' ({list(ax.labels)})."
                    )
                fixed.append(axn)
            free_axes = tuple(a for a in axes if a not in fixed)
            if s.value is not None and (s.total is not None or s.shape is not None):
                self.add("error", w, "give either 'value' or 'total' + 'shape', not both.")
            elif s.value is not None:
                self.require_within(self.dims(s.value, w), free_axes, w, f"'{s.target}'")
                self.forbid_state(s.value, w, "an initial value")
            elif s.total is not None and s.shape is not None:
                if what == "compartment":
                    self.add("error", w, "a compartment has no shape; give 'value'.")
                self.require_within(self.dims(s.total, w), (), w, "a total amount")
                self.require_within(self.dims(s.shape, w), free_axes, w, f"'{s.target}'")
                self.forbid_state(s.total, w, "an initial total")
                self.forbid_state(s.shape, w, "an initial shape")
            else:
                self.add("error", w, "give 'value', or both 'total' and 'shape'.")
            # two inits of one target overlap unless some axis is fixed to different categories
            here = dict(s.where)
            for j, other_where in targets.get(s.target, []):
                there = dict(other_where)
                if not any(a in there and there[a] != lab for a, lab in here.items()):
                    self.add(
                        "error",
                        w,
                        f"overlaps init {j + 1} of '{s.target}' (both set the same cells); "
                        "give each part of the state one initial condition, using where=.",
                    )
            targets.setdefault(s.target, []).append((i, s.where))
        initialised = {s.target for s in self.ir.inits}
        for n, what in self.kind.items():
            if what in ("field", "compartment") and n not in initialised:
                self.add("note", f"{what} '{n}'", "no initial condition given; starts at 0.")

    def check_doses(self) -> None:
        for i, d in enumerate(self.ir.doses):
            w = f"dose {i + 1} into '{d.target}'"
            kind = self.kind.get(d.target)
            if kind not in ("compartment", "field"):
                self.add(
                    "error", w, f"doses go into compartments and fields; '{d.target}' is neither."
                )
                continue
            if d.where and not self.check_where(d.target, d.where, w):
                continue
            free = tuple(a for a in self.field_axes.get(d.target, ()) if a not in dict(d.where))
            free_cat = [a for a in free if self.axis_type.get(a) == "categorical"]
            if free_cat:
                self.add(
                    "error",
                    w,
                    f"fix every axis of '{d.target}' (also {free_cat}): a dose goes into one "
                    "category; only continuous axes may be left free (with a shape or a "
                    "density).",
                )
                continue
            if not free and d.shape is not None:
                self.add(
                    "error",
                    w,
                    "a shape spreads an amount over a field's free axes; a compartment or one "
                    "element of a field has none.",
                )
            elif d.shape is not None:
                self.require_within(self.dims(d.shape, w), free, w, f"'{d.target}'")
                self.forbid_state(d.shape, w, "a dose shape")
            if len(d.times) != len(d.amounts):
                self.add("error", w, f"{len(d.times)} times but {len(d.amounts)} amounts.")
            if any(t < 0 for t in d.times):
                self.add("error", w, "dose times must be >= 0.")
            if list(d.times) != sorted(d.times):
                self.add("error", w, "dose times must be in increasing order.")
            if not all(_is_number(t) for t in d.times):
                self.add("error", w, "dose times must be finite numbers.")
                continue
            density = bool(free) and d.shape is None  # amounts are densities over `free`
            for a in d.amounts:
                self.require_within(
                    self.dims(a, w),
                    free if density else (),
                    w,
                    "a dose density" if density else "a dose amount",
                )
                self.forbid_state(a, w, "a dose amount")

    def check_arrivals(self) -> None:
        """Lags and bioavailabilities: one per place (a field or compartment), each a single
        number from parameters, >= 0."""
        for kind, specs in (("lag", self.ir.lags), ("bioavailability", self.ir.bioavailabilities)):
            seen: set[str] = set()
            for x in specs:
                w = f"{kind} of '{x.target}'"
                if self.kind.get(x.target) not in ("field", "compartment"):
                    self.add("error", w, f"'{x.target}' is not a field or compartment.")
                if x.target in seen:
                    self.add("error", w, f"'{x.target}' has more than one {kind}; give one.")
                seen.add(x.target)
                self.check_nonnegative(x.value, w, f"a {kind}")
            dosed = {d.target for d in self.ir.doses}
            for x in specs:
                if self.ir.doses and x.target not in dosed:
                    self.add(
                        "note",
                        f"{kind} of '{x.target}'",
                        "the model gives no doses there; it acts on a regimen's doses only.",
                    )

    def check_nonnegative(self, text: str, where: str, desc: str) -> None:
        """A lag or bioavailability: parameters only, a single number, >= 0."""
        self.require_within(self.dims(text, where), (), where, desc)
        self.forbid_state(text, where, desc)
        expr = self.parse(text, where)
        if expr is None:
            return
        for v in self.values_of(expr):
            if not v >= 0:
                self.add("error", where, f"must be >= 0 (found {v:g}).")
                return

    def check_observables(self) -> None:
        for o in self.ir.observables:
            self.dims(o.expr, f"observable '{o.name}'")
            self.check_where(o.expr, o.where, f"observable '{o.name}'")

    def check_transfers(self) -> None:
        for i, tr in enumerate(self.ir.transfers):
            dst = "elimination" if tr.to is None else tr.to
            w = f"transfer {i + 1} ({tr.source} -> {dst})"
            free: list[set[str]] = []
            ok = True
            for name, where in ((tr.source, tr.source_where), (tr.to, tr.to_where)):
                if name is None:
                    continue
                k = self.kind.get(name)
                if k not in ("field", "compartment"):
                    self.add("error", w, f"'{name}' is not a compartment or field.")
                    ok = False
                    continue
                if not self.check_where(name, where, w):
                    ok = False
                    continue
                if k == "field":
                    axes = self.field_axes.get(name, ())
                    cont = [a for a in axes if self.axis_type.get(a) == "continuous"]
                    if cont:
                        self.add(
                            "error",
                            w,
                            f"'{name}' has continuous axes {cont}; transfers connect "
                            "compartments and fields over categorical axes (use boundary "
                            "conditions or reactions along continuous axes).",
                        )
                        ok = False
                    free.append({a for a in axes if a not in dict(where)})
            if not ok:
                continue
            if tr.to == tr.source and dict(tr.to_where) == dict(tr.source_where):
                self.add("error", w, "the source and the destination are the same.")
            if len(free) == 2 and free[0] and free[1] and free[0] != free[1]:
                self.add(
                    "error",
                    w,
                    f"the two slices leave different axes free ({sorted(free[0])} and "
                    f"{sorted(free[1])}); element by element they must match.",
                )
                continue
            allowed = set().union(*free) if free else set()
            # an axis a slice fixes is fine too: the rate is taken at that category
            allowed |= {a for a, _ in (*tr.source_where, *tr.to_where)}
            self.require_within(
                self.dims(tr.rate, f"{w}, rate"),
                tuple(sorted(allowed)),
                f"{w}, rate",
                "the transfer (over the axes its slices leave free)",
            )
            e = self.parse(tr.rate, f"{w}, rate")
            if e is not None and e.is_number and float(e) < 0:
                self.add("error", w, f"the rate is negative ({tr.rate}); rates must be >= 0.")

    def check_extinctions(self) -> None:
        for e in self.ir.extinctions:
            w = f"extinction group '{e.name}'"
            if not e.group:
                self.add("error", w, "the group is empty.")
            for m in e.group:
                if self.kind.get(m) not in ("field", "compartment"):
                    self.add("error", w, f"'{m}' is not a field or compartment.")
            if not e.threshold > 0:
                self.add("error", w, "threshold must be > 0.")
            if len(set(e.group)) != len(e.group):
                self.add("error", w, "a member is listed twice.")
        taken: dict[str, list[tuple[str, dict[str, str]]]] = {}  # member -> (group, slice)
        for e in self.ir.extinctions:
            w = f"extinction group '{e.name}'"
            if e.where and len(e.where) != len(e.group):
                self.add("error", w, "'where' needs one entry per member.")
                continue
            for i, m in enumerate(e.group):
                here = dict(e.member_where(i))
                if here and self.kind.get(m) != "field":
                    self.add("error", w, f"'{m}' is not a field; only fields can be sliced.")
                    continue
                for ax, lab in here.items():
                    if ax not in self.field_axes.get(m, ()) or self.axis_type.get(ax) != (
                        "categorical"
                    ):
                        self.add("error", w, f"'{ax}' is not a categorical axis of '{m}'.")
                    elif lab not in self.axis_spec[ax].labels:
                        self.add("error", w, f"'{lab}' is not a category of '{ax}'.")
                # two parts of one field overlap unless some axis is fixed to different labels
                for other, there in taken.get(m, []):
                    if not any(a in there and there[a] != lab for a, lab in here.items()):
                        part = f"'{m}'" if not here else f"{m}.at({here})"
                        self.add(
                            "error",
                            w,
                            f"{part} overlaps group '{other}'; a part of the state can belong "
                            "to one extinction group only (zeroing one group must not change "
                            "another).",
                        )
                taken.setdefault(m, []).append((e.name, here))

    def check_floor_feeds(self) -> None:
        """A floored group fed from outside it. Below its threshold the floor hides the group's
        own losses, so a feed too weak to hold it above the threshold fills it up, and it goes
        extinct, again and again (ADR 0005, decision 23)."""
        for e in self.ir.extinctions:
            if not e.noise_floor or (e.where and len(e.where) != len(e.group)):
                continue
            part = {m: dict(e.member_where(i)) for i, m in enumerate(e.group)}
            feeds = self._feeds_into(part)
            if feeds:
                self.add(
                    "warning",
                    f"extinction group '{e.name}'",
                    f"has a noise floor and is fed from outside the group ({'; '.join(feeds)}). "
                    "Below its threshold the floor hides the group's own losses, so the feed "
                    "fills it up; if the feed cannot hold it above the threshold, it fills up "
                    "and goes extinct again and again. Put the source in the group (e.g. the "
                    "blood compartment of a phage lineage), or leave out the noise floor.",
                )

    def _feeds_into(self, part: dict[str, dict[str, str]]) -> list[str]:
        """What adds material to the group ``part`` (member -> its slice) from outside it."""
        feeds = []
        for t in self.ir.terms:
            if t.target not in part or not _overlap(part[t.target], dict(t.where)):
                continue
            here = {**dict(t.where), **part[t.target]}
            if t.kind in ("reaction", "rate"):
                if not self._zero_in_group(t.expr, part, here):
                    feeds.append(f"{t.kind} on '{t.target}': {t.expr}")
            elif t.kind in ("switch", "mutate"):
                tr = next((x for x in self.ir.transitions if x.name == t.transitions), None)
                if tr is None:
                    continue
                along = tuple(t.axis) if isinstance(t.axis, tuple) else (t.axis,)
                for frm, to, _ in tr.rates:
                    labels = [(frm,), (to,)] if len(along) == 1 else [frm, to]
                    origin = {**here, **dict(zip(along, labels[0], strict=False))}
                    dest = {**here, **dict(zip(along, labels[1], strict=False))}
                    if _inside(part, t.target, dest) and not _inside(part, t.target, origin):
                        feeds.append(f"{t.kind} {_category(frm)} -> {_category(to)} into it")
        for b in self.ir.boundaries:
            if b.target in part and b.kind in ("inflow", "dirichlet", "robin"):
                if not self._zero_in_group(b.value, part, part[b.target]):
                    feeds.append(f"{b.kind} boundary of '{b.target}' at {b.axis}={b.side}")
            if b.to in part and not _inside(part, b.target, {}):
                feeds.append(f"outflow of '{b.target}' at {b.axis}={b.side} into '{b.to}'")
        for tr in self.ir.transfers:
            if tr.to in part and _overlap(part[tr.to], dict(tr.to_where)):
                if not _inside(part, tr.source, dict(tr.source_where)):
                    feeds.append(f"transfer {tr.source} -> {tr.to}")
        return feeds

    def _zero_in_group(self, text: str, part, here: dict[str, str]) -> bool:
        """Whether formula ``text``, evaluated on the slice ``here``, is identically zero when
        the group is (through definitions and integrals too)."""
        try:
            expr = self._floored(parse(text, self.symbols), part, here, frozenset())
            return expr == 0 or sp.simplify(expr) == 0
        except Exception:  # noqa: BLE001 - unreadable formulas are reported elsewhere
            return False

    def _floored(self, expr: sp.Expr, part, here: dict[str, str], seen: frozenset[str]):
        subs = {}
        for sym in expr.free_symbols:
            n = getattr(sym, "name", None)
            k = self.kind.get(n)
            if (k == "compartment" and n in part) or (k == "field" and _inside(part, n, here)):
                subs[sym] = 0
            elif k == "end" and _inside(part, self.end_of[n][0], here):
                subs[sym] = 0
            elif k in ("definition", "integral") and n not in seen:
                if k == "definition":
                    text = next(x for x in self.ir.definitions if x.name == n).expr
                    inner = here
                else:
                    spec = next(x for x in self.ir.integrals if x.name == n)
                    text = spec.integrand
                    inner = {a: lab for a, lab in here.items() if a not in spec.over}
                    inner.update(dict(spec.where))
                subs[sym] = self._floored(parse(text, self.symbols), part, inner, seen | {n})
        return expr.xreplace(subs)

    def check_usage(self) -> None:
        for p in self.ir.params:
            if p.name not in self.used:
                self.add("note", f"parameter '{p.name}'", "is not used anywhere.")
        for f in self.ir.fields:
            moved = any(f.name in (tr.source, tr.to) for tr in self.ir.transfers)
            if not moved and not any(t.target == f.name for t in self.ir.terms):
                self.add("warning", f"field '{f.name}'", "has no terms, so it never changes.")


def validate(ir: ModelIR, *, raise_on_error: bool = True) -> Report:
    """Check a model; return a ``Report`` (raise ``ModelError`` on errors by default)."""
    c = _Checker(ir)
    if ir.ir_version not in READABLE_VERSIONS:
        c.add(
            "error",
            "model",
            f"IR version {ir.ir_version!r} is not supported ({sorted(READABLE_VERSIONS)}).",
        )
    c.register_names()
    c.check_axes()
    c.check_params()
    c.check_fields()
    c.check_definitions_and_integrals()
    c.check_transitions()
    c.check_terms()
    c.check_gradients()
    c.check_switches()
    c.check_boundaries()
    c.check_end_cycles()
    c.check_default_boundaries()
    c.check_inits()
    c.check_doses()
    c.check_arrivals()
    c.check_observables()
    c.check_extinctions()
    c.check_floor_feeds()
    c.check_transfers()
    c.check_usage()
    sizes = ir.state_count()
    total = sum(sizes.values())
    if total == 0:
        c.add(
            "error",
            "model",
            "has no states: no equation d_t(...) for a compartment or a field. TenQSP solves "
            "differential equations; a model given only by formulas of t (a closed-form "
            "concentration, say) is not supported yet. Write at least one equation: d_t(X) = ... "
            "in the notation, m.ode(X, ...) in Python.",
        )
    if total > 100_000:
        biggest = max(sizes, key=sizes.get)
        c.add(
            "warning",
            "model size",
            f"{total:,} state variables (target <= 100,000). Largest: "
            f"'{biggest}' with {sizes[biggest]:,}. Consider fewer cells, a separate field for "
            "structures that exist only at some sites, or a reachable-genotype axis (ADR 0003).",
        )
    report = Report(c.issues)
    if raise_on_error and report.errors:
        raise ModelError(report.issues)
    return report


def validate_values(ir: ModelIR) -> None:
    """Only the checks that depend on parameter values: weights and volumes > 0, a
    ``one_way`` axis's velocity >= 0, lags and bioavailabilities >= 0. Run on the values a
    solve uses when they differ from the model's own (values given at simulation time);
    raises ``ModelError``."""
    c = _Checker(ir)
    c.register_names()
    for f in ir.fields:
        if f.weights is not None:
            cat = tuple(a for a in f.axes if c.axis_type.get(a) == "categorical")
            c.check_size(f.weights, f"field '{f.name}', weights", cat, f"the axes {list(cat)}")
    for comp in ir.compartments:
        if comp.volume is not None:
            c.check_size(comp.volume, f"compartment '{comp.name}', volume", (), "a number")
    for t in ir.terms:
        if t.kind == "advection" and t.axis is not None and t.axis in c.axis_spec:
            c.check_one_way(t, f"advection of '{t.target}' along '{t.axis}'")
    for kind, specs in (("lag", ir.lags), ("bioavailability", ir.bioavailabilities)):
        for x in specs:
            c.check_nonnegative(x.value, f"{kind} of '{x.target}'", f"a {kind}")
    errors = [i for i in c.issues if i.level == "error"]
    if errors:
        raise ModelError(errors)


def _not_finite_real(expr: sp.Basic):
    """The first part of a formula that is undefined or complex, else ``None``. (Signed
    infinities are left alone: shape functions use them as default bounds, and the formula
    grammar has no way to write one.)"""
    for part in sp.preorder_traversal(expr):
        if part in (sp.zoo, sp.nan, sp.I):
            return {
                sp.zoo: "a division by zero",
                sp.nan: "an undefined value",
                sp.I: "the imaginary unit",
            }[part]
        if part.is_number and part.is_extended_real is False:
            return f"the complex number {part}"
    return None


def _is_number(x) -> bool:
    """A finite real number (not a bool)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _category(c) -> str:
    """A category, or an axis list, as written: ``S`` or ``S,R``."""
    return ",".join(map(str, c)) if isinstance(c, tuple) else str(c)


def _axes_text(over) -> str:
    if isinstance(over, tuple):
        return "axes " + ", ".join(f"'{a}'" for a in over)
    return f"axis '{over}'"


def _inside(part: dict[str, dict[str, str]], name: str | None, where: dict[str, str]) -> bool:
    """Whether the part ``where`` of state ``name`` lies in the group ``part``."""
    return name in part and part[name].items() <= where.items()


def _overlap(a: dict[str, str], b: dict[str, str]) -> bool:
    """Whether two slices of one field share any element."""
    return all(b.get(ax, lab) == lab for ax, lab in a.items())
