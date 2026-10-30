# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Lowering: formula text -> whole-array JAX values (ADR 0002).

Every quantity a formula can name is turned into an array in one shared **frame**: an array
with one dimension per model axis, in the order the axes were declared. A quantity has the
axis's size on the dimensions it varies over and size 1 elsewhere, so ordinary broadcasting
evaluates a formula on every cell at once. Examples, for axes ``(z, s, a)``:

- the coordinate ``z``: shape ``(n_z, 1, 1)`` (cell centres);
- ``k[s]``: shape ``(1, n_s, 1)``, sliced out of the parameter vector ``theta``;
- the field ``I(a, z)``: shape ``(n_z, 1, n_a)``;
- a compartment or scalar parameter: a 0-d array.

Formulas are parsed once, when the model is compiled. They are evaluated by walking the SymPy
tree while JAX traces the right-hand side, so tracing cost does not grow with the grid and no
user text is ever executed.

ADR 0002 planned ``sympy.lambdify``. A tree walk is used instead because shape functions
(``normal(z, ...)`` etc.) need the cell *faces*, not the value at a point: in any formula, a
shape function means its cell-average density, truncated to the axis and normalised so it
integrates to 1 over the axis (exact cell masses from CDF differences).
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Mapping
from functools import reduce

import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import numpy as np
import sympy as sp

from tenqsp.expr import SHAPE_CLASSES, Grad, NormCdf, Probit, Rem, Shape, end_key, parse
from tenqsp.ir import ModelIR, ParamSpec

# 3-point Gauss-Legendre on [-1, 1]: cell averages of initial values (exact to degree 5).
GAUSS_NODES = (-np.sqrt(0.6), 0.0, np.sqrt(0.6))
GAUSS_WEIGHTS = (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0)  # sum 1: weights for an average


class CompileError(ValueError):
    """The model is valid but uses something the compiler cannot build yet."""


# ----------------------------------------------------------------------------- symbol table


@dataclasses.dataclass(frozen=True)
class Tables:
    """Names in a model and the SymPy symbols that stand for them."""

    kind: dict[str, str]  # name -> axis | parameter | field | compartment | definition |
    #                       integral | end (a field read at one end of an axis, "D(z=lower)")
    symbols: dict[str, sp.Basic]
    vector_over: dict[str, str | tuple[str, ...]]
    ends: dict[str, tuple[str, str, str]] = dataclasses.field(default_factory=dict)
    # end name -> (field, axis, side)


def tables(ir: ModelIR) -> Tables:
    kind: dict[str, str] = {}
    for what, items in [
        ("axis", ir.axes),
        ("parameter", ir.params),
        ("field", ir.fields),
        ("compartment", ir.compartments),
        ("definition", ir.definitions),
        ("integral", ir.integrals),
    ]:
        for x in items:
            kind[x.name] = what
    vector_over = {p.name: p.over for p in ir.params if p.over is not None}
    symbols: dict[str, sp.Basic] = {"t": sp.Symbol("t", real=True)}
    for n in kind:
        symbols[n] = sp.IndexedBase(n) if n in vector_over else sp.Symbol(n, real=True)
    # a field read at either end of each of its continuous axes (IR 0.10): D(z=lower) is
    # also reachable by the coordinate of that end, D(z=0)
    ends: dict[str, tuple[str, str, str]] = {}
    continuous = {a.name: a for a in ir.axes if a.type == "continuous"}
    for f in ir.fields:
        for ax in f.axes:
            if ax not in continuous or not continuous[ax].faces:
                continue
            bounds = {"lower": continuous[ax].faces[0], "upper": continuous[ax].faces[-1]}
            for side, at in bounds.items():
                key = end_key(f.name, ax, side)
                if key in kind:  # pragma: no cover - a name cannot contain brackets
                    continue
                kind[key] = "end"
                ends[key] = (f.name, ax, side)
                symbols[key] = sp.Symbol(key, real=True)
                symbols[end_key(f.name, ax, f"#{float(at)!r}")] = symbols[key]
    return Tables(kind, symbols, vector_over, ends)


# ----------------------------------------------------------------------------- frame


@dataclasses.dataclass(frozen=True)
class Frame:
    """Axis order and sizes of the shared broadcasting frame."""

    axes: tuple[str, ...]
    sizes: dict[str, int]
    types: dict[str, str]  # continuous | categorical

    @classmethod
    def of(cls, ir: ModelIR) -> Frame:
        sizes = {
            a.name: (len(a.faces) - 1 if a.type == "continuous" else len(a.labels)) for a in ir.axes
        }
        return cls(tuple(a.name for a in ir.axes), sizes, {a.name: a.type for a in ir.axes})

    @property
    def ndim(self) -> int:
        return len(self.axes)

    def pos(self, axis: str) -> int:
        return self.axes.index(axis)

    def along(self, axis: str, values):
        """A 1-D array placed on ``axis``'s dimension of the frame."""
        shape = [1] * self.ndim
        shape[self.pos(axis)] = -1
        return jnp.reshape(jnp.asarray(values), shape)

    def from_layout(self, arr, axes: tuple[str, ...]):
        """A field stored in its own axis order -> frame (size 1 on axes it lacks)."""
        order = sorted(range(len(axes)), key=lambda i: self.pos(axes[i]))
        arr = jnp.transpose(arr, order) if len(axes) > 1 else arr
        shape = [1] * self.ndim
        for i in order:
            shape[self.pos(axes[i])] = arr.shape[order.index(i)]
        return jnp.reshape(arr, shape)

    def to_layout(self, arr, axes: tuple[str, ...], what: str = "a formula"):
        """Frame -> the axis order ``axes`` (dimensions of other axes must have size 1).

        The result is broadcastable to the field's shape; it is not expanded.
        """
        arr = jnp.asarray(arr)
        if arr.ndim == 0:
            return jnp.reshape(arr, (1,) * len(axes))
        keep = [self.pos(a) for a in axes]
        extra = [d for d in range(self.ndim) if d not in keep and arr.shape[d] != 1]
        if extra:
            names = [self.axes[d] for d in extra]
            raise CompileError(f"{what} varies over {names}, which the target does not have.")
        squeezed = jnp.squeeze(arr, axis=tuple(d for d in range(self.ndim) if d not in keep))
        sorted_keep = sorted(keep)
        return jnp.transpose(squeezed, [sorted_keep.index(k) for k in keep])


# ----------------------------------------------------------------------------- geometry


def geometry(ir: ModelIR) -> dict[str, dict[str, np.ndarray]]:
    """Faces, centres and widths of every continuous axis (passed to the compiled functions as
    runtime inputs, so they are not baked into the compiled code as constants)."""
    out = {}
    for a in ir.axes:
        if a.type == "continuous":
            f = np.asarray(a.faces, dtype=np.float64)
            out[a.name] = {"faces": f, "centers": 0.5 * (f[1:] + f[:-1]), "widths": np.diff(f)}
    return out


# ----------------------------------------------------------------------------- parameters


class MissingValues(ValueError):
    """Parameters declared without a value were not given one."""

    def __init__(self, names: list[str], hint: str | None = None):
        self.names = list(names)
        base = list(dict.fromkeys(n.split("[")[0] for n in names))
        example = ", ".join(f"'{b}': ..." for b in base[:3])
        hint = hint or f"give them when solving or simulating, e.g. params={{{example}}}"
        super().__init__(
            f"No value for {', '.join(names)}: declared without a value in the model, so {hint}."
        )


@dataclasses.dataclass(frozen=True)
class ParamLayout:
    """Where each parameter sits in the flat parameter vector ``theta``."""

    index: dict[str, slice]
    over: dict[str, str | None]
    labels: dict[str, tuple[str, ...]]
    theta0: np.ndarray

    @classmethod
    def of(cls, ir: ModelIR) -> ParamLayout:
        index, over, labels, values = {}, {}, {}, []
        i = 0
        for p in ir.params:
            n = 1 if p.over is None else len(_combinations(ir, p.over))
            if p.value is None:  # declared without a value: NaN until one is given
                vals: tuple = (np.nan,) * n
            else:
                vals = (p.value,) if p.over is None else tuple(p.value)
            index[p.name] = slice(i, i + len(vals))
            over[p.name] = p.over
            if p.over is not None:
                labels[p.name] = _combinations(ir, p.over)
            values.extend(float(v) for v in vals)
            i += len(vals)
        return cls(index, over, labels, np.asarray(values, dtype=np.float64))

    @property
    def size(self) -> int:
        return self.theta0.size

    def theta(self, overrides: Mapping[str, object] | None = None) -> np.ndarray:
        """``theta0`` with some parameters replaced. A per-category parameter takes a list in
        label order or a dict by label (only the labels given are changed)."""
        th = self.theta0.copy()
        for name, value in (overrides or {}).items():
            if name not in self.index:
                import difflib

                same = [k for k in self.index if k.lower() == name.lower()]  # Cl for CL
                hint = same or difflib.get_close_matches(name, list(self.index), n=1)
                extra = f" Did you mean '{hint[0]}'?" if hint else ""
                raise KeyError(f"The model has no parameter '{name}'.{extra}")
            sl = self.index[name]
            if self.over[name] is None:
                th[sl] = _scalar(value, name)
            elif isinstance(value, Mapping):
                labels = self.labels[name]
                for lab, v in _flat_items(value):
                    if lab not in labels:
                        raise KeyError(f"'{name}' has no category {lab!r}; it has {list(labels)}.")
                    th[sl.start + labels.index(lab)] = _scalar(v, f"{name}[{lab}]")
            else:
                vals = np.asarray(value, dtype=np.float64).ravel()
                if vals.size != sl.stop - sl.start:
                    raise ValueError(
                        f"'{name}' needs {sl.stop - sl.start} values, one per level of "
                        f"'{self.over[name]}' {list(self.labels[name])}."
                    )
                th[sl] = vals
        return th

    def specs(self, ir: ModelIR, theta) -> tuple[ParamSpec, ...]:
        """``ir.params`` with the values in ``theta``."""
        th = np.asarray(theta, dtype=np.float64)
        out = []
        for p in ir.params:
            vals = tuple(float(v) for v in th[self.index[p.name]])
            out.append(dataclasses.replace(p, value=vals[0] if p.over is None else vals))
        return tuple(out)

    def missing(self, theta) -> list[str]:
        """Parameters with no value in ``theta`` (NaN): ``"k"``, or ``"k[R]"`` for one category
        of a per-category parameter (``"k[s]"`` when none of its categories has a value)."""
        th = np.asarray(theta, dtype=np.float64)[: self.size]
        out = []
        for name, sl in self.index.items():
            gone = np.isnan(th[sl])
            if not gone.any():
                continue
            if self.over[name] is None:
                out.append(name)
            elif gone.all():
                over = self.over[name]
                out.append(f"{name}[{over if isinstance(over, str) else ', '.join(over)}]")
            else:
                for lab, g in zip(self.labels[name], gone, strict=True):
                    if g:
                        lab = lab if isinstance(lab, str) else ", ".join(lab)
                        out.append(f"{name}[{lab}]")
        return out


def fields_in(plan: Plan, e: sp.Expr, _seen: frozenset[str] = frozenset()) -> set[str]:
    """Fields a formula uses, directly, at one of their ends (``D(z=lower)``) or through
    definitions."""
    out = set()
    for sym in e.free_symbols:
        what = plan.tables.kind.get(sym.name)
        if what == "field":
            out.add(sym.name)
        elif what == "end":
            out.add(plan.tables.ends[sym.name][0])
        elif what == "definition" and sym.name not in _seen:
            out |= fields_in(plan, plan.definitions[sym.name], _seen | {sym.name})
    return out


def integral_weights(ir: ModelIR, plan: Plan, e: sp.Expr) -> str | None:
    """The weights an integral multiplies by: those of the weighted field(s) its integrand
    uses (the validator requires them to agree), or None."""
    weights = {f.name: f.weights for f in ir.fields}
    found = sorted({weights[f] for f in fields_in(plan, e) if weights.get(f)})
    return found[0] if found else None


def _combinations(ir: ModelIR, over) -> tuple:
    """Labels of a per-category parameter in storage order: labels for one axis, tuples of
    labels (first axis slowest) for several."""
    if not isinstance(over, tuple):
        return ir.axis(over).labels
    import itertools

    return tuple(itertools.product(*(ir.axis(a).labels for a in over)))


def _flat_items(value: Mapping, prefix: tuple = ()):
    """``{"liver": {"vascular": 1}}`` -> ``(("liver", "vascular"), 1)``; one level: labels."""
    for k, v in value.items():
        if isinstance(v, Mapping):
            yield from _flat_items(v, (*prefix, k))
        else:
            yield ((*prefix, k) if prefix else k), v


def _scalar(v, name: str) -> float:
    a = np.asarray(v, dtype=np.float64)
    if a.ndim != 0:
        raise ValueError(f"'{name}' is a single number, not {v!r}.")
    if not np.isfinite(a):
        raise ValueError(f"'{name}' must be a finite number, not {float(a)}.")
    return float(a)


# ----------------------------------------------------------------------------- shapes


def _shape_cdf(name: str, x, params):
    """Cumulative distribution (possibly unnormalised) of a shape at positions ``x``."""
    if name == "normal":
        mean, sd = params
        return jss.ndtr((x - mean) / sd)
    if name == "lognormal":
        meanlog, sdlog = params
        pos = x > 0
        return jnp.where(pos, jss.ndtr((jnp.log(jnp.where(pos, x, 1.0)) - meanlog) / sdlog), 0.0)
    if name == "gamma":
        shape, rate = params
        return jss.gammainc(shape, rate * jnp.maximum(x, 0.0))
    if name == "exponential":
        (rate,) = params
        return -jnp.expm1(-rate * jnp.maximum(x, 0.0))
    if name == "beta":
        alpha, beta = params
        return jss.betainc(alpha, beta, jnp.clip(x, 0.0, 1.0))
    if name == "uniform":
        lower, upper = params
        # Unbounded ends are allowed: any increasing linear CDF works before normalisation.
        return jnp.clip(x, lower, upper)
    raise CompileError(f"unknown shape '{name}'")  # pragma: no cover


def shape_density(name: str, faces, params, ax: int, *, raw: bool = False):
    """Cell-average density of a shape on one axis, normalised to integrate to 1 over it.

    ``faces`` has the ``n + 1`` faces on dimension ``ax`` and size 1 elsewhere; ``params``
    broadcast against it. Returns an array with ``n`` along that dimension.

    ``raw=True`` gives the curve itself (a probability density, not rescaled after the axis
    cuts it), for a shape inside a density formula, ``density(normal(z, ...))``. ``point``
    has no curve and is always the whole unit in one cell.
    """
    if name == "point":
        (at,) = params
        # cell i holds the point when faces[i] <= at < faces[i+1]; at the upper end, the last
        # cell. A point within rounding of an interior face counts as on it (the cell above),
        # so at=0.3 is on the face 0.30000000000000004 of a grid of 10 cells on [0, 1].
        n_faces = faces.shape[ax]
        span = jnp.max(faces) - jnp.min(faces)
        on = jnp.abs(faces - at) <= 1e-9 * span
        last = jnp.arange(n_faces).reshape(faces.shape) == n_faces - 1
        cdf = jnp.where(last, (faces >= at) | on, (faces > at) & ~on).astype(faces.dtype)
        raw = False
    else:
        cdf = _shape_cdf(name, faces, params)
    mass = jnp.diff(cdf, axis=ax)
    widths = jnp.diff(faces, axis=ax)
    total = jnp.sum(mass, axis=ax, keepdims=True)
    if raw:
        if name == "uniform":  # the CDF above is not scaled: divide by the support's length;
            lower, upper = params  # an open end means the axis, as for an amount's spread
            length = jnp.asarray(upper - lower, dtype=faces.dtype)
            finite = jnp.isfinite(length)
            return mass / jnp.where(finite, length, total) / widths
        return mass / widths
    return mass / total / widths


# ----------------------------------------------------------------------------- evaluation


class Evaluator:
    """Evaluates parsed formulas in the frame, for one call of a compiled function.

    ``fields`` and ``compartments`` hold the current state (fields already in the frame);
    ``theta`` the parameters; ``geom`` the axis geometry. ``coords`` optionally replaces the
    coordinate of some axes (Gauss nodes for initial values). Definitions are recomputed under
    replaced coordinates; integrals always use cell values.

    ``face_axis`` (see ``at_faces``) evaluates a formula on the faces of one continuous
    axis, where velocities live: the coordinate is taken at the faces; every cell value that
    varies along the axis (fields, integrals, shape masses) is first interpolated onto the
    faces; ``grad(F, axis)`` is the difference of ``F`` across each face (ADR 0005).
    """

    def __init__(
        self,
        plan: Plan,
        *,
        t,
        theta,
        geom,
        fields: Mapping[str, object],
        compartments: Mapping[str, object],
        coords: Mapping[str, object] | None = None,
        integral_cache: dict | None = None,
        face_axis: str | None = None,
        end_hook=None,
        raw_shapes: bool = False,
    ):
        self.plan = plan
        # shapes are the curve itself (a density formula) instead of an amount's spread
        self.raw_shapes = raw_shapes
        self.t = t
        self.theta = theta
        self.geom = geom
        self.fields = fields
        self.compartments = compartments
        self.coords = dict(coords or {})
        self.defs: dict[str, object] = {}
        self.integrals = {} if integral_cache is None else integral_cache
        self.face_axis = face_axis
        self.guard = None  # inside a conditional value: where that value is used
        # computes a field's value at one end of an axis (the compiled model's boundary
        # treatment); None where no state is at hand
        self.end_hook = end_hook

    def with_coords(self, coords: Mapping[str, object]) -> Evaluator:
        """Same state, some coordinates replaced (integrals are shared)."""
        return Evaluator(
            self.plan,
            t=self.t,
            theta=self.theta,
            geom=self.geom,
            fields=self.fields,
            compartments=self.compartments,
            coords={**self.coords, **coords},
            integral_cache=self.integrals,
            face_axis=self.face_axis,
            end_hook=self.end_hook,
            raw_shapes=self.raw_shapes,
        )

    def with_raw_shapes(self) -> Evaluator:
        """Same state; shape functions are the curve itself (inside a density formula)."""
        return Evaluator(
            self.plan,
            t=self.t,
            theta=self.theta,
            geom=self.geom,
            fields=self.fields,
            compartments=self.compartments,
            coords=self.coords,
            integral_cache=self.integrals,
            face_axis=self.face_axis,
            end_hook=self.end_hook,
            raw_shapes=True,
        )

    def at_faces(self, axis: str) -> Evaluator:
        """Same state, evaluated on the faces of continuous ``axis`` (``n + 1`` values)."""
        return Evaluator(
            self.plan,
            t=self.t,
            theta=self.theta,
            geom=self.geom,
            fields=self.fields,
            compartments=self.compartments,
            coords={**self.coords, axis: self.plan.frame.along(axis, self.geom[axis]["faces"])},
            integral_cache=self.integrals,
            face_axis=axis,
            end_hook=self.end_hook,
            raw_shapes=self.raw_shapes,
        )

    def _faces(self, v):
        """Cell values -> face values along ``face_axis``: linear between the two centres
        (by distance, so nonuniform grids stay consistent), the end cells' values on the two
        boundary faces. Values constant along the axis are returned as they are."""
        ax = self.face_axis
        frame = self.plan.frame
        d, n = frame.pos(ax), frame.sizes[ax]
        v = jnp.asarray(v)
        if v.ndim < frame.ndim or v.shape[d] != n:
            return v
        if n == 1:
            return jnp.concatenate([v, v], axis=d)
        c, f = self.geom[ax]["centers"], self.geom[ax]["faces"]
        wr = frame.along(ax, (f[1:-1] - c[:-1]) / (c[1:] - c[:-1]))  # weight of the right cell
        mid = (1.0 - wr) * jax.lax.slice_in_dim(v, 0, n - 1, axis=d) + wr * jax.lax.slice_in_dim(
            v, 1, n, axis=d
        )
        return jnp.concatenate(
            [jax.lax.slice_in_dim(v, 0, 1, axis=d), mid, jax.lax.slice_in_dim(v, n - 1, n, axis=d)],
            axis=d,
        )

    def grad(self, e: Grad):
        """``grad(F, x)`` on the faces of ``x``: (F[i+1] - F[i]) / (centre distance) on the
        interior faces; each boundary face takes the nearest interior face's value."""
        ax = e.axis
        if self.face_axis != ax:
            raise CompileError(  # the validator prevents this
                f"'{e}' can only be used in an advection velocity along '{ax}'."
            )
        frame = self.plan.frame
        d, n = frame.pos(ax), frame.sizes[ax]
        u = self.fields[e.field]
        if n == 1:
            shape = list(u.shape)
            shape[d] = 2
            return jnp.zeros(shape)
        c = self.geom[ax]["centers"]
        g = (jax.lax.slice_in_dim(u, 1, n, axis=d) - jax.lax.slice_in_dim(u, 0, n - 1, axis=d)) / (
            frame.along(ax, jnp.diff(c))
        )
        return jnp.concatenate(
            [
                jax.lax.slice_in_dim(g, 0, 1, axis=d),
                g,
                jax.lax.slice_in_dim(g, n - 2, n - 1, axis=d),
            ],
            axis=d,
        )

    def centers_only(self) -> Evaluator:
        if not self.coords:
            return self
        return Evaluator(
            self.plan,
            t=self.t,
            theta=self.theta,
            geom=self.geom,
            fields=self.fields,
            compartments=self.compartments,
            integral_cache=self.integrals,
            end_hook=self.end_hook,
            raw_shapes=self.raw_shapes,
        )

    # -------------------------------------------------------------- names
    def name(self, n: str):
        plan = self.plan
        what = plan.tables.kind.get(n)
        if n == "t":
            return jnp.asarray(self.t)
        if what == "axis":
            if n in self.coords:
                return self.coords[n]
            return plan.frame.along(n, self.geom[n]["centers"])
        if what == "parameter":
            return self.theta[plan.params.index[n].start]
        if what == "field":
            return self.fields[n] if self.face_axis is None else self._faces(self.fields[n])
        if what == "compartment":
            return self.compartments[n]
        if what == "definition":
            if n not in self.defs:
                self.defs[n] = self.expr(plan.definitions[n])
            return self.defs[n]
        if what == "integral":
            if n not in self.integrals:
                self.integrals[n] = self.centers_only().integral(n)
            v = self.integrals[n]
            return v if self.face_axis is None else self._faces(v)
        if what == "end":
            if n not in self.integrals:  # shared like integrals (and lifted with them)
                if self.end_hook is None:
                    raise CompileError(f"'{n}' (a field at one end) cannot be used here.")
                self.integrals[n] = self.end_hook(n, self.centers_only())
            v = self.integrals[n]
            return v if self.face_axis is None else self._faces(v)
        raise CompileError(f"'{n}' cannot be used in a formula.")  # validator prevents this

    def integral(self, n: str):
        integrand, over = self.plan.integrals[n]
        val = self.expr(integrand)
        frame = self.plan.frame
        for ax, lab in self.plan.integral_where.get(n, ()):  # a slice: keep that category
            val = jnp.asarray(val)
            if val.ndim < frame.ndim:
                val = jnp.reshape(val, (1,) * (frame.ndim - val.ndim) + val.shape)
            d = frame.pos(ax)
            if val.shape[d] != 1:
                i = self.plan.ir.axis(ax).labels.index(lab)
                val = jax.lax.slice_in_dim(val, i, i + 1, axis=d)
        full = list(np.shape(val)) + [1] * (frame.ndim - np.ndim(val))
        val = jnp.reshape(val, full) if np.ndim(val) < frame.ndim else val
        for ax in over:
            d = frame.pos(ax)
            if val.shape[d] == 1:  # constant along this axis: integrate a broadcast copy
                val = jnp.broadcast_to(val, (*val.shape[:d], frame.sizes[ax], *val.shape[d + 1 :]))
            if frame.types[ax] == "continuous":
                val = val * frame.along(ax, self.geom[ax]["widths"])
            val = jnp.sum(val, axis=d, keepdims=True)
        return val

    def vector(self, base: str, index: tuple[str, ...]):
        sl = self.plan.params.index[base]
        if len(index) == 1:
            return self.plan.frame.along(index[0], self.theta[sl])
        frame = self.plan.frame
        table = jnp.reshape(self.theta[sl], tuple(frame.sizes[a] for a in index))
        return frame.from_layout(table, index)

    # -------------------------------------------------------------- tree walk
    def expr(self, e: sp.Basic):
        if isinstance(e, sp.Symbol):
            return self.name(e.name)
        if isinstance(e, sp.Indexed):
            return self.vector(str(e.base), tuple(str(i) for i in e.indices))
        if e.is_Number:
            if e in (sp.oo, -sp.oo):
                return jnp.inf if e == sp.oo else -jnp.inf
            return float(e)
        if isinstance(e, sp.Add):
            return reduce(lambda a, b: a + b, (self.expr(a) for a in e.args))
        if isinstance(e, sp.Mul):
            return reduce(lambda a, b: a * b, (self.expr(a) for a in e.args))
        if isinstance(e, sp.Pow):
            sig = _sigmoid_arg(e)
            if sig is not None:
                return jax.nn.sigmoid(self.expr(sig))
            base, ex = e.args
            b = self.expr(base)
            if not (ex.is_Integer and int(ex) >= 0):  # 1/x, sqrt(x), x^p: singular at 0
                b = self._safe(b, 1.0)
            if ex.is_Integer:
                return jnp.asarray(b) ** int(ex)
            if ex == sp.Rational(1, 2):
                return jnp.sqrt(b)
            if ex == sp.Rational(-1, 2):
                return 1.0 / jnp.sqrt(b)
            return jnp.power(b, self.expr(ex))
        if isinstance(e, Shape):
            v = self.shape(e)
            return v if self.face_axis is None else self._faces(v)
        if isinstance(e, Grad):
            return self.grad(e)
        fn = _FUNCS.get(type(e))
        if fn is not None:
            args = [self.expr(a) for a in e.args]
            if self.guard is not None and type(e) in _SAFE_ARGS:
                args = [self._safe(a, v) for a, v in zip(args, _SAFE_ARGS[type(e)], strict=False)]
            return fn(*args)
        if isinstance(e, (sp.Max, sp.Min)):
            op = jnp.maximum if isinstance(e, sp.Max) else jnp.minimum
            return reduce(op, (self.expr(a) for a in e.args))
        if isinstance(e, sp.Piecewise):
            return self.cases(e)
        if isinstance(e, sp.logic.boolalg.BooleanTrue):
            return jnp.asarray(True)
        if isinstance(e, sp.logic.boolalg.BooleanFalse):
            return jnp.asarray(False)
        if isinstance(e, sp.Rel):
            ev = self
            if e in self.plan.time_atoms:  # constant within a solver segment (ADR 0006)
                ts = self.theta[self.plan.switch_slot]
                ev = copy.copy(self)
                ev.t = jnp.where(jnp.isnan(ts), self.t, ts)
                ev.defs = {}
            return _RELATIONS[type(e)](ev.expr(e.lhs), ev.expr(e.rhs))
        if isinstance(e, sp.And):
            return reduce(jnp.logical_and, (self.expr(a) for a in e.args))
        if isinstance(e, sp.Or):
            return reduce(jnp.logical_or, (self.expr(a) for a in e.args))
        if isinstance(e, sp.Not):
            return jnp.logical_not(self.expr(e.args[0]))
        raise CompileError(f"cannot compile '{e}' ({type(e).__name__}).")

    # -------------------------------------------------------------- conditionals
    def cases(self, e: sp.Piecewise):
        """``v1 if c1 else v2 if c2 else ...``: the first true condition chooses. Each value is
        evaluated under a guard (where it is chosen); elsewhere the arguments of functions
        that are singular somewhere (log, 1/x, sqrt, ...) are replaced by a harmless number.
        The result is unchanged there (that value is not chosen), and derivatives cannot pick
        up the NaN or infinity of a value that is not chosen ("double where", ADR 0006)."""
        taken = None  # some earlier condition is true
        chosen = []
        for value, cond in e.args:
            c = self.expr(cond)
            pick = c if taken is None else jnp.logical_and(c, jnp.logical_not(taken))
            taken = c if taken is None else jnp.logical_or(taken, c)
            chosen.append((pick, self.guarded(pick).expr(value)))
        out = chosen[-1][1]
        for pick, v in reversed(chosen[:-1]):
            out = jnp.where(pick, v, out)
        return out

    def guarded(self, pick) -> Evaluator:
        """The same evaluator, with values used only where ``pick`` holds."""
        g = copy.copy(self)
        g.guard = pick if self.guard is None else jnp.logical_and(self.guard, pick)
        g.defs = {}  # definitions under a guard differ where the guard is false: not shared
        return g

    def _safe(self, x, harmless: float):
        if self.guard is None:
            return x
        return jnp.where(self.guard, x, harmless)

    def shape(self, e: Shape):
        coord = e.args[0]
        name = e.shape_name
        if not (
            isinstance(coord, sp.Symbol) and self.plan.frame.types.get(coord.name) == "continuous"
        ):
            raise CompileError(
                f"{name}(...): the first argument must be a continuous axis, not '{coord}'."
            )
        ax = coord.name
        faces = self.plan.frame.along(ax, self.geom[ax]["faces"])
        params = [self.centers_only().expr(p) for p in e.args[1:]]
        return shape_density(name, faces, params, self.plan.frame.pos(ax), raw=self.raw_shapes)


_RELATIONS: dict[type, Callable] = {
    sp.StrictLessThan: jnp.less,
    sp.LessThan: jnp.less_equal,
    sp.StrictGreaterThan: jnp.greater,
    sp.GreaterThan: jnp.greater_equal,
    sp.Equality: jnp.equal,
    sp.Unequality: jnp.not_equal,
}


def _sigmoid_arg(e: sp.Pow):
    """``1/(1 + exp(-x))`` -> x (evaluated with jax.nn.sigmoid: no overflow in derivatives)."""
    base, ex = e.args
    if ex != -1 or not isinstance(base, sp.Add) or len(base.args) != 2:
        return None
    one, other = base.args
    if one != 1:
        one, other = other, one
    if one == 1 and isinstance(other, sp.exp):
        return -other.args[0]
    return None


_FUNCS: dict[type, Callable] = {
    sp.exp: jnp.exp,
    sp.log: jnp.log,
    sp.Abs: jnp.abs,
    sp.sin: jnp.sin,
    sp.cos: jnp.cos,
    sp.tan: jnp.tan,
    sp.asin: jnp.arcsin,
    sp.acos: jnp.arccos,
    sp.atan: jnp.arctan,
    sp.atan2: jnp.arctan2,
    sp.sinh: jnp.sinh,
    sp.cosh: jnp.cosh,
    sp.tanh: jnp.tanh,
    sp.erf: jax.scipy.special.erf,
    sp.loggamma: jax.scipy.special.gammaln,
    sp.gamma: jax.scipy.special.gamma,
    sp.floor: jnp.floor,
    sp.ceiling: jnp.ceil,
    NormCdf: jax.scipy.special.ndtr,
    Probit: jax.scipy.special.ndtri,
    Rem: jnp.fmod,
}
# inside a conditional value: harmless arguments for functions singular somewhere, used where
# that value is not chosen (ADR 0006)
_SAFE_ARGS: dict[type, tuple[float, ...]] = {
    sp.log: (1.0,),
    sp.tan: (0.0,),
    sp.asin: (0.0,),
    sp.acos: (0.0,),
    sp.atan2: (1.0, 1.0),
    sp.loggamma: (1.0,),
    sp.gamma: (1.0,),
    Probit: (0.5,),
    Rem: (0.0, 1.0),
}


# ----------------------------------------------------------------------------- plan


@dataclasses.dataclass
class Plan:
    """Everything the compiled functions need, parsed once."""

    ir: ModelIR
    tables: Tables
    frame: Frame
    params: ParamLayout
    definitions: dict[str, sp.Expr]
    integrals: dict[str, tuple[sp.Expr, tuple[str, ...]]]
    # conditions on time and parameters only, and where their segment time sits in theta
    # (set by the compiler, ADR 0006)
    time_atoms: frozenset = frozenset()
    switch_slot: int | None = None
    integral_where: dict = dataclasses.field(default_factory=dict)  # name -> slice

    @classmethod
    def of(cls, ir: ModelIR) -> Plan:
        tab = tables(ir)
        plan = cls(
            ir=ir,
            tables=tab,
            frame=Frame.of(ir),
            params=ParamLayout.of(ir),
            definitions={},
            integrals={},
        )
        plan.definitions = {d.name: plan.parse(d.expr) for d in ir.definitions}
        plan.integrals = {}
        plan.integral_where = {}
        for i in ir.integrals:
            e = plan.parse(i.integrand)
            w = integral_weights(ir, plan, e)
            if w is not None:  # a weighted field: its integral is an amount
                e = e * plan.parse(w)
            plan.integrals[i.name] = (e, tuple(i.over))
            plan.integral_where[i.name] = tuple(i.where)
        return plan

    def parse(self, text: str) -> sp.Expr:
        return parse(text, self.tables.symbols)

    def coordinate_axes(self, e: sp.Expr, _seen: frozenset[str] = frozenset()) -> set[str]:
        """Continuous axes whose coordinate a formula uses, directly or through definitions
        (not inside shape functions, which handle their axis exactly)."""
        e = e.replace(lambda x: isinstance(x, Shape), lambda x: sp.Integer(1))
        out = set()
        for s in e.free_symbols:
            what = self.tables.kind.get(s.name)
            if what == "axis" and self.frame.types[s.name] == "continuous":
                out.add(s.name)
            elif what == "definition" and s.name not in _seen:
                out |= self.coordinate_axes(self.definitions[s.name], _seen | {s.name})
        return out


__all__ = [
    "CompileError",
    "Evaluator",
    "Frame",
    "GAUSS_NODES",
    "GAUSS_WEIGHTS",
    "MissingValues",
    "ParamLayout",
    "Plan",
    "SHAPE_CLASSES",
    "geometry",
    "shape_density",
    "tables",
]
