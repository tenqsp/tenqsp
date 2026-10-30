# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Compiler: a validated ``ModelIR`` -> pure JAX functions.

``CompiledModel`` holds, each jitted once per model structure:

- ``rhs(t, y, theta)``: time derivative of the flat state vector;
- ``sparse_jacobian(t, y, theta)``: exact ``d rhs / d y`` in sparse form, from the model's
  structure (``jacobian`` is the dense version, kept as a check for small models);
- ``initial_state(theta)``: the flat initial state (doses not included);
- ``dose_amounts(theta)``: the amount of every scheduled dose;
- ``observables(t, y, theta)``: the named observables;
- ``ledger_rates(t, y, theta)``: the conservation ledger: for every field and
  compartment, the rate at which each term moves amount in or out.

Parameters, axis geometry, the state and time are all array inputs: changing a parameter value
never recompiles. Every transport term is assembled in flux form: interior face fluxes from
the operators, the two boundary faces from the boundary condition, then one divergence per
field and axis.

State layout: fields in declaration order (each flattened row-major over its own axes), then
compartments.

Ledger. The amount of a field is its density times the cell widths, summed (plain sums over
categories). Interior transport fluxes cancel in that sum, so the change in amount is exactly
the sum of: reaction and hazard terms, transitions (zero by construction), and the flux through
each boundary end that is not zero-flux. Reaction, hazard and compartment rate formulas are
booked piece by piece: ``g*B*(1 - B/K) - k*B*P`` gives two lines, growth and adsorption, named
by the text as written. The solver integrates these channels alongside the state (quadrature
states), so the cumulative amounts carry the same accuracy as the solution.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import itertools
import re
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
import sympy as sp

from tenqsp.expr import end_key
from tenqsp.ir import BoundarySpec, ModelIR, TermSpec
from tenqsp.lower import (
    GAUSS_NODES,
    GAUSS_WEIGHTS,
    CompileError,
    Evaluator,
    MissingValues,
    ParamLayout,
    Plan,
    geometry,
)
from tenqsp.operators import (
    advective_flux_interior,
    advective_outflow_flux,
    assemble_faces,
    diffusive_flux_interior,
    dirichlet_diffusive_flux,
    divergence,
    robin_flux,
    upwind_end_flux,
)
from tenqsp.validate import validate, validate_values


def _validate(ir: ModelIR, stacklevel: int = 4) -> None:
    report = validate(ir)  # raises ModelError listing every problem
    for issue in report.warnings:
        if issue.where.startswith("extinction group"):  # these change the results
            warnings.warn(str(issue), UserWarning, stacklevel=stacklevel)


def _lu_size(cm) -> float | None:
    """Megabytes of the sparse LU factors of I - hJ at the starting state (what BDF factors),
    measured with SuperLU as the solver does (missing values set to 1); None if it fails."""
    import scipy.sparse.linalg as spla

    try:
        th = cm._stand_in_theta()
        y = np.asarray(cm.initial_state(th))
        J = cm.sparse_jacobian(0.0, y, th)
        lu = spla.splu((sps.identity(J.shape[0], format="csc") - 1e-3 * J).tocsc())
    except Exception:  # noqa: BLE001 - a size estimate only
        return None
    return ((lu.L.nnz + lu.U.nnz) * 12 + 2 * (J.shape[0] + 1) * 4) / 1e6


MAX_BUILDS = 8  # model structures whose builds are kept (least recently used dropped first)
_BUILDS: OrderedDict[str, CompiledModel] = OrderedDict()


def build(ir: ModelIR) -> CompiledModel:
    """The compiled model for ``ir``. Models with the same structure (same equations, axes,
    cells, number of doses; ``ModelIR.structure_hash``) share one build whatever their
    parameter values and dose times, so a value change never recompiles, even
    across model objects (e.g. ``tq.load`` of one file with different ``params=``)."""
    key = ir.structure_hash()
    base = _BUILDS.get(key)
    if base is None:
        base = _BUILDS[key] = CompiledModel(ir)
        while len(_BUILDS) > MAX_BUILDS:
            _BUILDS.popitem(last=False)
        return base
    _BUILDS.move_to_end(key)
    if base.ir.content_hash() != ir.content_hash():
        return base.with_values(ir)  # validates ir
    _validate(ir, stacklevel=3)  # another model object: checked (and warned about) as well
    return base


SWITCH_SAMPLES = 20_001  # samples per solve span for conditions on time without a closed form
DENSE_JACOBIAN_LIMIT = 5_000  # states; the dense Jacobian (a check only) is refused above this


# ----------------------------------------------------------------------------- state layout


@dataclasses.dataclass(frozen=True)
class StateLayout:
    """Where each field and compartment sits in the flat state vector."""

    names: tuple[str, ...]
    shapes: dict[str, tuple[int, ...]]
    slices: dict[str, slice]
    axes: dict[str, tuple[str, ...]]

    @classmethod
    def of(cls, ir: ModelIR, sizes: Mapping[str, int]) -> StateLayout:
        names, shapes, slices, axes = [], {}, {}, {}
        i = 0
        for f in ir.fields:
            shape = tuple(sizes[a] for a in f.axes)
            n = int(np.prod(shape))
            names.append(f.name)
            shapes[f.name], slices[f.name], axes[f.name] = shape, slice(i, i + n), tuple(f.axes)
            i += n
        for c in ir.compartments:
            names.append(c.name)
            shapes[c.name], slices[c.name], axes[c.name] = (), slice(i, i + 1), ()
            i += 1
        return cls(tuple(names), shapes, slices, axes)

    @property
    def size(self) -> int:
        return max((s.stop for s in self.slices.values()), default=0)

    def unpack(self, y) -> dict[str, object]:
        return {n: jnp.reshape(y[self.slices[n]], self.shapes[n]) for n in self.names}

    def pack(self, parts: Mapping[str, object]):
        return jnp.concatenate(
            [jnp.reshape(jnp.broadcast_to(parts[n], self.shapes[n]), (-1,)) for n in self.names]
        )


# ----------------------------------------------------------------------------- helpers


def _take(x, i, dim: int):
    return jnp.take(x, i, axis=dim)


def _rest(shape: tuple[int, ...], dim: int) -> tuple[int, ...]:
    return shape[:dim] + shape[dim + 1 :]


def _to_faces(v, dim: int, n: int, shape: tuple[int, ...]):
    """Velocity on the faces -> full field shape with ``n + 1`` faces along ``dim``.

    The face evaluator (``Evaluator.at_faces``) gives ``n + 1`` values along the axis, or
    1 for a velocity constant along it: coordinates are taken at the faces and cell values
    (fields, integrals) interpolated onto them before the formula is applied.
    """
    m = v.shape[dim]
    if m not in (1, n + 1):  # pragma: no cover - the face evaluator always gives n + 1 or 1
        raise CompileError(f"velocity has {m} values along the axis; expected {n + 1} or 1.")
    face_shape = shape[:dim] + (n + 1,) + shape[dim + 1 :]
    return jnp.broadcast_to(v, face_shape)


def additive_parts(text: str) -> list[str]:
    """Split a formula at its top-level ``+`` and ``-`` into signed pieces, keeping the text as
    written: ``"g*B*(1 - B/K) - k*B*P"`` -> ``["g*B*(1 - B/K)", "-k*B*P"]``."""
    text = text.strip().replace("^", "**")  # as in tenqsp.expr.parse
    parts: list[str] = []

    def walk(node, sign: int) -> None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub):
            walk(node.left, sign)
            walk(node.right, sign if isinstance(node.op, ast.Add) else -sign)
            return
        seg = ast.get_source_segment(text, node) or ast.unparse(node)
        if sign < 0:
            if isinstance(node, ast.IfExp):  # binds more loosely than a sign: keep it whole
                seg = f"-({seg})"
            else:
                seg = seg[1:] if seg.startswith("-") else f"-{seg}"
        parts.append(seg)

    walk(ast.parse(text, mode="eval").body, 1)
    return parts


def _axes_of(axis) -> tuple[str, ...]:
    """The axes of a transition: one name, or a tuple of names (joint)."""
    return tuple(axis) if isinstance(axis, tuple) else (axis,)


def _apply_moves(moves, src, dims: tuple[int, ...], size=1.0, fraction: bool = False):
    """Transfer ``rate * src[origin]`` from each origin category to its destination
    (``M[dest, origin]`` with the diagonal derived): nothing is created or lost in any cell.

    ``moves`` holds ``(origin, dest, rate)``: origin and dest give an index along each of
    ``dims``; the rate is an array broadcastable to ``src`` (constant along ``dims``), so it
    may vary over the other axes cell by cell.

    With category sizes (field weights), amounts are conserved: a switch rate is a clearance
    (it moves ``rate * src[origin]`` of amount); with ``fraction`` (mutate) the rate is the
    fraction of the origin's production, ``rate * src[origin] * size[origin]`` of amount.
    """
    du = jnp.zeros_like(src)
    sized = np.ndim(size) > 0
    if sized:
        size = jnp.broadcast_to(size, src.shape)
    for o, d, rate in moves:
        io = [slice(None)] * src.ndim
        id_ = [slice(None)] * src.ndim
        for k, dim in enumerate(dims):
            io[dim], id_[dim] = o[k], d[k]
        io, id_ = tuple(io), tuple(id_)
        flux = jnp.broadcast_to(rate, src.shape)[io] * src[io]
        if not sized:
            du = du.at[id_].add(flux).at[io].add(-flux)
        elif fraction:
            du = du.at[id_].add(flux * size[io] / size[id_]).at[io].add(-flux)
        else:
            du = du.at[id_].add(flux / size[id_]).at[io].add(-flux / size[io])
    return du


def _part_text(name: str, where) -> str:
    if not where:
        return name
    return f"{name}.at({', '.join(f'{a}={lab}' for a, lab in where)})"


@dataclasses.dataclass(frozen=True)
class _Boundary:
    kind: str
    value: sp.Expr | None = None
    h: sp.Expr | None = None
    to: str | None = None
    default: bool = False


# ----------------------------------------------------------------------------- compiled model


@dataclasses.dataclass(frozen=True)
class ExtinctionGroup:
    """A compiled extinction group: ``members`` marks its states; its total amount is
    ``(members * measure(theta)) @ y``."""

    name: str
    group: tuple[str, ...]
    threshold: float
    members: np.ndarray
    noise_floor: bool


def implicit_shift(g, root):
    """Zero in value, with the derivative of the root of ``g(t) = 0`` at ``root`` (found by a
    method with no useful derivative, such as bisection): by the implicit function theorem,
    d root = -(dg with t held) / (dg/dt). Add it to ``stop_gradient(root)``. (Step 4.4c.)"""
    root = jax.lax.stop_gradient(root)
    value, slope = jax.jvp(g, (root,), (jnp.ones_like(root),))
    shift = -value / jax.lax.stop_gradient(jnp.where(slope == 0, 1.0, slope))
    return shift - jax.lax.stop_gradient(shift)


class CompiledModel:
    """A model compiled to JAX functions. Build with ``model.compile()`` or
    ``CompiledModel(ir)``; solve with ``solve``."""

    def __init__(self, ir: ModelIR):
        _validate(ir)
        self.ir = ir
        self.plan = plan = Plan.of(ir)
        self.layout = StateLayout.of(ir, plan.frame.sizes)
        self.params = plan.params
        self.geom = jax.tree_util.tree_map(jnp.asarray, geometry(ir))
        # sizes of categories (field weights) and compartment volumes: formulas of parameters
        self.sizes: dict[str, sp.Expr] = {
            f.name: plan.parse(f.weights) for f in ir.fields if f.weights is not None
        }
        self.sizes.update(
            {c.name: plan.parse(c.volume) for c in ir.compartments if c.volume is not None}
        )
        self.cell_widths = self._cell_widths()
        self.extinction_groups = tuple(
            ExtinctionGroup(
                e.name, e.group, float(e.threshold), self._member_mask(e), e.noise_floor
            )
            for e in ir.extinctions
        )
        self._parse_slices_and_transfers()
        self._floors = [g for g in self.extinction_groups if g.noise_floor]
        self._parse_terms()
        # one theta slot per noise-floor group, after the time slot: 1 while the group is
        # floored, 0 while it counts, NaN outside a solve (then decided by its total)
        first = self.params.size + (self.plan.switch_slot is not None)
        self.floor_slots = [first + j for j in range(len(self._floors))]
        self._parse_inits_doses_observables()
        self._label_ledger()
        self.compile_seconds: float | None = None

        self._rhs = jax.jit(self._rhs_impl)
        self._jac = jax.jit(jax.jacfwd(self._rhs_impl, argnums=1))
        self._y0 = jax.jit(self._y0_impl)
        self._doses = jax.jit(self._dose_impl)
        self._event_shapes: dict = {}  # shapes of a regimen's doses into fields, parsed
        self._arrival = jax.jit(self._arrival_impl)
        self._measure = jax.jit(self._measure_impl)
        self._obs = jax.jit(self._obs_impl)
        self._ledger = jax.jit(self._ledger_impl)
        self._aug = jax.jit(self._aug_impl)  # state + ledger channels, used by the solver
        self._aug_jac = jax.jit(jax.jacfwd(self._aug_impl, argnums=1))
        self._integral_layout()
        # built on first use and shared with with_values() copies, so they compile nothing
        self._lazy: dict = {}
        self._dense = jax.jit(lambda t, y, th, g: self._dense_rows_impl(t, y, th, g, False))
        self._dense_aug = jax.jit(lambda t, y, th, g: self._dense_rows_impl(t, y, th, g, True))
        self._parse_switches()
        self._switch = jax.jit(self._switch_impl)

    # ------------------------------------------------------------------ public functions
    @property
    def n_states(self) -> int:
        return self.layout.size

    @property
    def n_theta(self) -> int:
        """Length of theta: the parameters, plus one slot for the segment time of conditions
        on time when the model has any (ADR 0006), plus one per noise-floor group (whether
        the solver has it floored, ADR 0005)."""
        return self.params.size + (self.plan.switch_slot is not None) + len(self._floors)

    def _with_slot(self, th: np.ndarray) -> np.ndarray:
        extra = self.n_theta - self.params.size
        return np.concatenate([th, np.full(extra, np.nan)]) if extra else th

    @property
    def theta0(self) -> np.ndarray:
        """Parameter vector with the values given in the model."""
        return self._with_slot(self.params.theta0.copy())

    def theta(self, params: Mapping[str, object] | None = None) -> np.ndarray:
        """Parameter vector with some values replaced, e.g. ``{"k": 2e-9}``."""
        return self._with_slot(self.params.theta(params))

    def rhs(self, t, y, theta=None):
        return self._rhs(_time(t), jnp.asarray(y), self._th(theta), self.geom)

    def jacobian(self, t, y, theta=None):
        """Dense ``d rhs / d y`` by ``jax.jacfwd``: a reference for checks on small models.
        The solver uses ``sparse_jacobian``."""
        if self.n_states > DENSE_JACOBIAN_LIMIT:
            raise ValueError(
                f"A dense Jacobian of {self.n_states:,} states would need "
                f"{self.n_states**2 * 8 / 1e9:.1f} GB; use sparse_jacobian()."
            )
        return self._jac(_time(t), jnp.asarray(y), self._th(theta), self.geom)

    def initial_state(self, theta=None):
        return self._y0(self._th(theta), self.geom)

    def dose_amounts(self, theta=None):
        """The amount of every entry of ``dose_schedule``; raises ValueError where one is not
        finite (a shape with no mass on its axis, e.g. a point outside it)."""
        out = self._doses(self._th(theta), self.geom)
        a = np.asarray(out)
        if not np.all(np.isfinite(a)):
            bad = sorted(
                {
                    tgt
                    for (_, tgt), x in zip(self.dose_schedule, a, strict=True)
                    if not np.isfinite(x)
                }
            )
            raise ValueError(
                f"The doses into {bad} are not finite. A shape may have no mass on its axis "
                "(e.g. a point or a normal outside it), or a formula divides by zero."
            )
        return out

    @property
    def has_arrival(self) -> bool:
        """Whether the model has a lag or a bioavailability (``m.lag``, ``m.bioavailability``)."""
        return bool(self.lag_exprs or self.bioavailability_exprs)

    def arrival(self, theta=None) -> tuple[np.ndarray, np.ndarray]:
        """Per state: the lag of a dose into it and the fraction of it that arrives (0 and 1
        where the model sets none). Raises ValueError for a value below 0 or not finite."""
        n = self.n_states
        if not self.has_arrival:
            return np.zeros(n), np.ones(n)
        lag, frac = (np.asarray(x) for x in self._arrival(self._th(theta), self.geom))
        for what, v, specs in (
            ("lag", lag, self.lag_exprs),
            ("bioavailability", frac, self.bioavailability_exprs),
        ):
            for sl, _, target in specs:
                x = float(v[sl.start])
                if not (np.isfinite(x) and x >= 0):
                    raise ValueError(
                        f"The {what} of the doses into '{target}' is {x:g}; it must be a finite "
                        "number >= 0."
                    )
        return lag, frac

    def observables(self, t, y, theta=None) -> dict[str, object]:
        return self._obs(_time(t), jnp.asarray(y), self._th(theta), self.geom)

    def ledger_rates(self, t, y, theta=None) -> dict[str, dict[str, float]]:
        """``{target: {channel: rate}}``: how fast each term moves amount into (+) or out of
        (-) each field and compartment at state ``y``."""
        v = np.asarray(self._ledger(_time(t), jnp.asarray(y), self._th(theta), self.geom))
        out: dict[str, dict[str, float]] = {n: {} for n in self.layout.names}
        for (target, _, label), x in zip(self.ledger_entries, v, strict=True):
            out[target][label] = float(x)
        return out

    def measure(self, theta=None) -> np.ndarray:
        """Per state: the amount one unit of it stands for (cell widths along continuous axes
        times the field's weights; a compartment's volume, or 1)."""
        return np.asarray(self._measure(self._th(theta), self.geom))

    def amounts(self, y, theta=None) -> dict[str, float]:
        """Amount in each field (value x cell widths x weights, summed) and compartment
        (value x volume)."""
        y = np.asarray(y)[..., : self.layout.size]
        w = self.measure(theta)
        return {name: (y[..., sl] * w[sl]).sum(axis=-1) for name, sl in self.layout.slices.items()}

    def transport_coefficients(self, t, y, theta=None) -> dict[tuple[str, str], dict]:
        """Per (field, continuous axis) with transport: ``"velocity"``, the total velocity on
        the faces (the axis has ``n + 1`` entries; ``None`` without advection), and ``"D"``,
        the total diffusivity in the cells (``None`` without diffusion); arrays in the field's
        axis order. Evaluated at state ``y`` as the right-hand side does (for
        ``m.diagnostics()``; not compiled, so not for use inside a solve)."""
        ev, state = self._evaluator(
            _time(t), jnp.asarray(y), self._th(theta), self.geom, floor=False
        )
        out: dict[tuple[str, str], dict] = {}
        for name, per_axis in self.transport.items():
            axes = self.layout.axes[name]
            shape = tuple(state[name].shape)
            for ax, spec in per_axis.items():
                dim = axes.index(ax)
                n = shape[dim]
                face_ev = ev.at_faces(ax)
                v = D = None
                for term, e in spec["terms"]:
                    where = f"{term.kind} of '{name}' along '{ax}'"
                    if term.kind == "advection":
                        x = _to_faces(self._on(face_ev.expr(e), axes, where), dim, n, shape)
                        v = x if v is None else v + x
                    else:
                        x = self._on(ev.expr(e), axes, where)
                        x = x if x.size == 1 else jnp.broadcast_to(x, shape)
                        D = x if D is None else D + x
                if v is None and D is None:
                    continue
                out[(name, ax)] = {
                    "velocity": None if v is None else np.asarray(v),
                    "D": None if D is None else np.asarray(D),
                }
        return out

    def compilations(self) -> dict[str, int]:
        """How many times each jitted function has been compiled (one per model structure)."""
        local = self._lazy["local_jit"]._cache_size() if "local_jit" in self._lazy else 0
        return {
            "rhs": self._rhs._cache_size(),
            "solver rhs (with ledger)": self._aug._cache_size(),
            "jacobian: field rows": local,
            "jacobian: compartment rows": self._dense._cache_size(),
            "jacobian: compartment and ledger rows": self._dense_aug._cache_size(),
            "dense jacobian (for checks)": self._jac._cache_size(),
            "initial state": self._y0._cache_size(),
            "dose amounts": self._doses._cache_size(),
            "dose lags and bioavailabilities": self._arrival._cache_size(),
            "measure": self._measure._cache_size(),
            "observables": self._obs._cache_size(),
            "ledger": self._ledger._cache_size(),
            "switch conditions": self._switch._cache_size(),
        }

    def _over_times(self, fn, T, Y, theta):
        """Evaluate ``fn(t, y, theta, geom)`` at each output time. One call per time, so the
        number of output times never triggers a new compilation (a batched call would)."""
        th = jnp.asarray(theta, dtype=jnp.float64)
        return [fn(_time(t), jnp.asarray(y), th, self.geom) for t, y in zip(T, Y, strict=True)]

    def unpack(self, y) -> dict[str, np.ndarray]:
        """Flat state -> ``{name: array}`` (fields in their axis order, compartments scalar)."""
        return {k: np.asarray(v) for k, v in self.layout.unpack(jnp.asarray(y)).items()}

    def _th(self, theta):
        th = self.theta0 if theta is None else np.asarray(theta, dtype=np.float64)
        if th.shape == (self.params.size,):
            th = self._with_slot(th)
        if th.shape != (self.n_theta,):
            raise ValueError(f"theta must have {self.params.size} values, not shape {th.shape}.")
        missing = self.params.missing(th)
        if missing:
            raise MissingValues(missing)
        return jnp.asarray(th)

    def check_values(self, theta) -> None:
        """Run the checks that depend on parameter values (weights and volumes > 0, one_way
        velocities >= 0) on ``theta`` when it differs from the model's own values, which were
        checked when the model was built. Remembers the last values it passed."""
        th = np.asarray(theta, dtype=np.float64)[: self.params.size]
        if np.array_equal(th, self.params.theta0, equal_nan=True):
            return
        key = th.tobytes()
        seen = self.__dict__.setdefault("_checked_values", [])
        if key in seen:
            return
        validate_values(dataclasses.replace(self.ir, params=self.params.specs(self.ir, th)))
        seen.append(key)
        del seen[:-64]

    def state_index(self, target: str, where=(), role: str = "dose") -> int:
        """The flat index of a compartment, or of one element of a field (``where`` fixing
        every axis, all categorical)."""
        import difflib

        if target not in self.layout.slices:
            places = list(self.layout.slices)
            hint = difflib.get_close_matches(target, places, n=1)
            extra = f" Did you mean '{hint[0]}'?" if hint else f" It has {places}."
            raise KeyError(f"{role}: the model has no compartment or field '{target}'.{extra}")
        sl = self.layout.slices[target]
        axes = self.layout.axes.get(target, ())
        fixed = dict(where)
        if len(fixed) != len(where):
            raise ValueError(f"{role}: an axis is fixed twice in {target}{list(where)}.")
        if not axes:
            if where:
                raise ValueError(f"{role}: '{target}' is a compartment; it has no axes to fix.")
            return sl.start
        if set(fixed) != set(axes):
            raise ValueError(
                f"{role}: '{target}' is a field over {list(axes)}; fix every axis, e.g. "
                f"{target}[{', '.join(f'{a}=...' for a in axes)}]."
            )
        idx = []
        for a in axes:
            spec = self.ir.axis(a)
            if spec.type != "categorical":
                raise ValueError(
                    f"{role}: '{a}' is a continuous axis; a dose goes into a compartment or one "
                    f"element of a field over categorical axes (on a continuous axis, use an "
                    f"inflow boundary or a source term)."
                )
            if fixed[a] not in spec.labels:
                raise KeyError(
                    f"{role}: axis '{a}' has no category {fixed[a]!r}; it has {list(spec.labels)}."
                )
            idx.append(spec.labels.index(fixed[a]))
        return sl.start + int(np.ravel_multi_index(tuple(idx), self.layout.shapes[target]))

    def slice_cells(self, target: str, where=()) -> tuple[np.ndarray, tuple[str, ...]]:
        """The flat indices of a compartment, a field or a slice of it (in the field's order),
        and the axes the slice leaves free."""
        if target not in self.layout.slices:
            self.state_index(target, where)  # raises with a suggestion
        sl = self.layout.slices[target]
        axes = self.layout.axes.get(target, ())
        fixed = dict(where)
        for a, lab in where:
            if a not in axes:
                raise ValueError(f"'{a}' is not an axis of '{target}' (it has {list(axes)}).")
            if self.ir.axis(a).type != "categorical":
                raise ValueError(
                    f"'{a}' is a continuous axis of '{target}'; a slice fixes categories only. "
                    f"To dose at one place along '{a}', give a shape, e.g. "
                    f'shape=tq.point("{a}", at=0).'
                )
            if lab not in self.ir.axis(a).labels:
                raise KeyError(
                    f"axis '{a}' has no category {lab!r}; it has {list(self.ir.axis(a).labels)}."
                )
        grid = np.arange(sl.start, sl.stop).reshape(self.layout.shapes[target])
        idx = tuple(
            self.ir.axis(a).labels.index(fixed[a]) if a in fixed else slice(None) for a in axes
        )
        return np.ravel(grid[idx]), tuple(a for a in axes if a not in fixed)

    def event_cells(self, target: str, where=(), shape=None, theta=None, role: str = "dose"):
        """Where a dose (or an infusion) of a regimen goes: ``(indices, shares)``, the states
        and the fraction of the amount each receives (adding up to 1). A compartment or one
        element of a field: that state. A field with free axes needs ``shape`` (formula text,
        as ``tq.point(g, at=0)`` gives), evaluated at ``theta``; cells with no share are left
        out. Derivatives do not flow through the shape."""
        cells, free = self.slice_cells(target, where)
        free_cat = [a for a in free if self.ir.axis(a).type == "categorical"]
        if free_cat:
            raise ValueError(
                f"{role} into '{target}': fix every axis of '{target}' (also {free_cat}), e.g. "
                f"{target}[{', '.join(f'{a}=...' for a in free_cat)}]; only continuous axes may "
                "be left free, with a shape."
            )
        if not free:
            if shape is not None:
                raise ValueError(
                    f"{role} into '{target}': a shape is for a field with free axes; "
                    f"'{target}' takes the amount as it is."
                )
            return cells, np.ones(1)
        if shape is None:
            raise ValueError(
                f"{role} into the field '{target}' (free axes {list(free)}): give a shape saying "
                f'where the amount goes, e.g. shape=tq.point("{free[0]}", at=0).'
            )
        key = ("event shape", target, tuple(where), str(shape))
        e = self._event_shapes.get(key)
        if e is None:
            e = self.plan.parse(str(shape))
            self._require_no_state(e, f"{role} shape into '{target}'")
            bad = sorted(self.plan.coordinate_axes(e) - set(free))
            if bad:
                raise ValueError(f"{role} into '{target}': the shape uses {bad}, not free axes.")
            self._event_shapes[key] = e
        th = self._th(theta)
        ev, _ = self._evaluator(0.0, jnp.zeros(self.layout.size), jnp.asarray(th), self.geom)
        amounts = self._spread(ev, self.geom, target, where, free, sp.Integer(1), e)
        amounts = np.ravel(np.asarray(amounts)) * np.asarray(self.measure(th))[cells]
        if not (np.all(np.isfinite(amounts)) and np.all(amounts >= 0) and amounts.sum() > 0):
            raise ValueError(
                f"{role} into '{target}': the shape {shape} has no mass on the grid (or is not "
                "finite there)."
            )
        keep = amounts > 0
        return cells[keep], amounts[keep] / amounts.sum()

    def with_values(self, ir: ModelIR) -> CompiledModel:
        """This build with another model's values: ``ir`` must have the same structure
        (``ModelIR.structure_hash``), differing only in parameter values and dose times.
        The jitted functions are shared, so nothing is compiled again."""
        if ir.structure_hash() != self.ir.structure_hash():
            raise ValueError("with_values needs a model with the same structure.")
        _validate(ir, stacklevel=4)  # every model is checked, shared build or not
        new = copy.copy(self)
        new.ir = ir
        new.params = ParamLayout.of(ir)
        new.dose_schedule = [
            (float(t), d.target)
            for d, cells in zip(ir.doses, self.dose_cells, strict=True)
            for t in d.times
            for _ in cells
        ]
        new._checked_values = []
        return new

    def solve(self, t_span, **kwargs):
        """Solve with the scipy backend; see ``solve``."""
        from tenqsp.backends.scipy_backend import solve

        return solve(self, t_span, **kwargs)

    # ------------------------------------------------------------------ parsing
    def _parse_terms(self) -> None:
        ir, plan = self.ir, self.plan
        self.field_terms: dict[str, list[tuple[TermSpec, sp.Expr | None]]] = {
            f.name: [] for f in ir.fields
        }
        # reaction, hazard and rate formulas are kept as signed pieces, booked separately
        self.rate_parts: dict[str, list[tuple[str, sp.Expr]]] = {
            c.name: [] for c in ir.compartments
        }
        self.term_parts: dict[str, list[list[tuple[str, sp.Expr]]]] = {
            f.name: [] for f in ir.fields
        }
        for t in ir.terms:
            pieces = (
                [(p, plan.parse(p)) for p in additive_parts(t.expr)]
                if t.kind in ("rate", "reaction", "hazard")
                else []
            )
            if t.kind == "rate":
                self.rate_parts[t.target].extend(pieces)
            else:
                expr = None if t.kind == "switch" else plan.parse(t.expr)
                self.field_terms[t.target].append((t, expr))
                self.term_parts[t.target].append(pieces)
        # name -> (axes, [(origin index, destination index, rate)]), one index per axis
        self.generators: dict[str, tuple[tuple[str, ...], list]] = {}
        for tr in ir.transitions:
            axes = _axes_of(tr.axis)

            def where(cat, axes=axes):
                cat = cat if isinstance(cat, tuple) else (cat,)
                return tuple(ir.axis(a).labels.index(c) for a, c in zip(axes, cat, strict=True))

            self.generators[tr.name] = (
                axes,
                [(where(frm), where(to), plan.parse(rate)) for frm, to, rate in tr.rates],
            )
        given: dict[tuple[str, str, str], BoundarySpec] = {
            (b.target, b.axis, b.side): b for b in ir.boundaries
        }
        self.transport: dict[str, dict[str, dict]] = {}
        for f in ir.fields:
            per_axis = {}
            for ax in f.axes:
                if plan.frame.types[ax] != "continuous":
                    continue
                terms = [
                    (t, e)
                    for t, e in self.field_terms[f.name]
                    if t.axis == ax and t.kind in ("advection", "diffusion")
                ]
                ends = {}
                for side in ("lower", "upper"):
                    b = given.get((f.name, ax, side))
                    ends[side] = self._boundary(b, f.name, ax, side, terms)
                if terms or any(not e.default for e in ends.values()):
                    per_axis[ax] = {"terms": terms, "ends": ends}
            self.transport[f.name] = per_axis

    def _boundary(self, b, field, ax, side, terms) -> _Boundary:
        has_adv = any(t.kind == "advection" for t, _ in terms)
        if b is None:  # defaults, as reported by the validator
            one_way = self.ir.axis(ax).kind == "one_way"
            if one_way and has_adv and side == "upper":
                return _Boundary("outflow", default=True)
            return _Boundary("zero_flux", default=True)
        parse = self.plan.parse
        if b.kind == "flux":  # flux along the axis = A + C * (the field's value at this end)
            sym = self.plan.tables.symbols
            u_b = sp.Dummy("u_b")
            rhs = self._expand(parse(b.value)).xreplace(
                {sym[field]: u_b, sym[end_key(field, ax, side)]: u_b}
            )
            return _Boundary("flux", value=rhs.subs(u_b, 0), h=sp.diff(rhs, u_b))
        return _Boundary(
            b.kind,
            value=parse(b.value) if b.kind in ("inflow", "dirichlet", "robin") else None,
            h=parse(b.h) if b.kind == "robin" else None,
            to=b.to,
        )

    def _parse_slices_and_transfers(self) -> None:
        """Masks of slices (in each target's own axis order) and parsed transfers."""
        self.transfers = [(tr, self.plan.parse(tr.rate)) for tr in self.ir.transfers]

    def _mask(self, name: str, where) -> np.ndarray | None:
        """1 on the slice ``where`` of ``name`` (its own axis order, broadcastable), or None
        for the whole of it."""
        if not where:
            return None
        axes = self.layout.axes[name]
        mask = np.ones([1] * len(axes))
        for ax, lab in where:
            onehot = (np.asarray(self.ir.axis(ax).labels) == lab).astype(float)
            shape = [1] * len(axes)
            shape[axes.index(ax)] = -1
            mask = mask * np.reshape(onehot, shape)
        return mask

    def _pick(self, x, where):
        """The value of frame array ``x`` on slice ``where`` (size 1 on the fixed axes)."""
        frame = self.plan.frame
        x = jnp.asarray(x)
        if x.ndim < frame.ndim:
            x = jnp.reshape(x, (1,) * (frame.ndim - x.ndim) + x.shape)
        for ax, lab in where:
            d = frame.pos(ax)
            i = self.ir.axis(ax).labels.index(lab)
            if x.shape[d] != 1:
                x = jax.lax.slice_in_dim(x, i, i + 1, axis=d)
        return x

    def _parse_inits_doses_observables(self) -> None:
        ir, plan = self.ir, self.plan
        self.inits = []
        for s in ir.inits:
            exprs = [plan.parse(x) for x in (s.value, s.total, s.shape) if x is not None]
            for e in exprs:
                self._require_no_state(e, f"initial condition of '{s.target}'")
            self.inits.append(
                (
                    s,
                    plan.parse(s.value) if s.value is not None else None,
                    plan.parse(s.total) if s.total is not None else None,
                    plan.parse(s.shape) if s.shape is not None else None,
                )
            )
        # every dose, one entry per state it reaches: a compartment or one element of a field
        # is one state; a dose into a field with free axes (IR 0.9) reaches every cell of the
        # slice, each with its share (formulas of theta, evaluated in _dose_impl)
        self.dose_schedule: list[tuple[float, str]] = []
        self.dose_index: list[int] = []  # the state each entry goes into
        self.dose_spread: list[bool] = []  # whether the entry is one cell of a field dose
        self.dose_parts: list[tuple] = []  # per dose: what _dose_impl evaluates
        self.dose_cells: list[np.ndarray] = []  # per DoseSpec: the states it reaches
        for d in ir.doses:
            cells, free = self.slice_cells(d.target, d.where)
            self.dose_cells.append(cells)
            shape = None
            if d.shape is not None:
                shape = plan.parse(d.shape)
                self._require_no_state(shape, f"dose shape into '{d.target}'")
            for t, a in zip(d.times, d.amounts, strict=True):
                e = plan.parse(a)
                self._require_no_state(e, f"dose amount into '{d.target}'")
                self.dose_parts.append((d.target, d.where, free, e, shape, cells))
                self.dose_schedule += [(float(t), d.target)] * len(cells)
                self.dose_index += [int(c) for c in cells]
                self.dose_spread += [bool(free)] * len(cells)
        # lags and bioavailabilities: (the states of the place, formula, place)
        self.lag_exprs: list[tuple[slice, sp.Expr, str]] = []
        self.bioavailability_exprs: list[tuple[slice, sp.Expr, str]] = []
        for specs, out, what in (
            (ir.lags, self.lag_exprs, "lag"),
            (ir.bioavailabilities, self.bioavailability_exprs, "bioavailability"),
        ):
            for x in specs:
                e = plan.parse(x.value)
                self._require_no_state(e, f"{what} of '{x.target}'")
                out.append((self.layout.slices[x.target], e, x.target))
        self.observable_exprs = {o.name: plan.parse(o.expr) for o in ir.observables}
        self.observable_where = {o.name: o.where for o in ir.observables}

    def _require_no_state(self, e: sp.Expr, what: str, _seen: frozenset[str] = frozenset()) -> None:
        kind = self.plan.tables.kind
        for s in e.free_symbols:
            k = kind.get(s.name, "time" if s.name == "t" else None)
            if k in ("field", "compartment", "integral", "time"):
                raise CompileError(
                    f"The {what} uses '{s.name}' (a {k}); initial values and dose amounts may "
                    "use parameters and coordinates only."
                )
            if k == "definition" and s.name not in _seen:
                self._require_no_state(self.plan.definitions[s.name], what, _seen | {s.name})

    def _label_ledger(self) -> None:
        """Name every ledger channel, in a fixed order: per field its reaction, hazard and
        transition terms, then each boundary end that is not zero-flux; per compartment its
        rate terms, then what is routed into it (outflows, Dirichlet/Robin exchange)."""
        entries: list[tuple[str, str, str]] = []
        used: set[tuple[str, str]] = set()

        def add(target: str, kind: str, label: str) -> tuple[str, str]:
            base, i = label, 2
            while (target, label) in used:
                label, i = f"{base} #{i}", i + 1
            used.add((target, label))
            entries.append((target, kind, label))
            return (target, label)

        self._term_keys: dict[str, list] = {}
        self._end_keys: dict[tuple[str, str, str], tuple[str, str]] = {}
        routed_into: dict[str, list] = {c.name: [] for c in self.ir.compartments}
        for f in self.ir.fields:
            keys = []
            for (t, _), pieces in zip(
                self.field_terms[f.name], self.term_parts[f.name], strict=True
            ):
                if t.kind in ("reaction", "hazard"):
                    keys.append([add(f.name, t.kind, f"{t.kind}: {text}") for text, _ in pieces])
                elif t.kind in ("switch", "mutate"):
                    keys.append(
                        [
                            add(
                                f.name,
                                "transition",
                                f"{t.kind} along {'×'.join(_axes_of(t.axis))} ({t.transitions})",
                            )
                        ]
                    )
                else:
                    keys.append([])
            self._term_keys[f.name] = keys
            for ax, spec in self.transport[f.name].items():
                for side, b in spec["ends"].items():
                    if b.kind == "zero_flux":
                        continue
                    if b.kind == "outflow":
                        where = f" to {b.to}" if b.to else " (absorbed)"
                    else:
                        where = f" with {b.to}" if b.to else ""
                    self._end_keys[(f.name, ax, side)] = add(
                        f.name, "boundary", f"boundary {ax}={side}: {b.kind}{where}"
                    )
                    if b.to is not None:
                        routed_into[b.to].append((f.name, ax, side, b.kind))
        self._transfer_keys: list[tuple] = []
        for tr in self.ir.transfers:
            src = _part_text(tr.source, tr.source_where)
            dst = "elimination" if tr.to is None else _part_text(tr.to, tr.to_where)
            out_key = add(tr.source, "transfer", f"transfer to {dst}")
            in_key = None if tr.to is None else add(tr.to, "transfer", f"transfer from {src}")
            self._transfer_keys.append((out_key, in_key))
        self._rate_keys: dict[str, list] = {}
        self._routed_keys: dict[tuple[str, str, str], tuple[str, str]] = {}
        for c in self.ir.compartments:
            self._rate_keys[c.name] = [
                add(c.name, "rate", f"rate: {text}") for text, _ in self.rate_parts[c.name]
            ]
            for f, ax, side, kind in routed_into[c.name]:
                what = "outflow from" if kind == "outflow" else f"{kind} exchange with"
                self._routed_keys[(f, ax, side)] = add(
                    c.name, "routed", f"{what} {f} at {ax}={side}"
                )
        self.ledger_entries: tuple[tuple[str, str, str], ...] = tuple(entries)

    # ------------------------------------------------------------------ conditionals
    def _rhs_formulas(self) -> list[sp.Expr]:
        """Every parsed formula that can change the right-hand side."""
        ir, plan = self.ir, self.plan
        out = list(plan.definitions.values()) + [e for e, _ in plan.integrals.values()]
        out += [plan.parse(t.expr) for t in ir.terms]
        for b in ir.boundaries:
            if b.kind in ("inflow", "dirichlet", "robin"):
                out.append(plan.parse(b.value))
            if b.kind == "robin":
                out.append(plan.parse(b.h))
        for tr in ir.transitions:
            out += [plan.parse(r) for _, _, r in tr.rates]
        out += [plan.parse(tr.rate) for tr in ir.transfers]
        return out

    def _expand(self, e: sp.Expr, _seen: frozenset[str] = frozenset()) -> sp.Expr:
        """Definitions replaced by their formulas (integrals stay as names)."""
        subs = {}
        for s in e.free_symbols:
            if self.plan.tables.kind.get(s.name) == "definition" and s.name not in _seen:
                subs[s] = self._expand(self.plan.definitions[s.name], _seen | {s.name})
        return e.xreplace(subs) if subs else e

    def _parse_switches(self) -> None:
        """Sort every comparison in the right-hand side (ADR 0006):

        - on time and scalar parameters only: solved for its switch times, which become
          segment boundaries like doses (``switch_times``);
        - one number that depends on the state (compartments, integrals over everything):
          a solver event, so the integration stops exactly at the switch (``switch_values``);
        - anything that varies cell by cell: evaluated where it stands (the validator notes
          it); and conditions on coordinates and parameters only never switch in time.
        """
        kind = self.plan.tables.kind
        t_sym = sp.Symbol("t", real=True)
        atoms: dict[sp.Rel, sp.Rel] = {}  # as written -> with definitions expanded
        for e in self._rhs_formulas():
            for a in e.atoms(sp.Rel):
                atoms.setdefault(a, self._expand(a))
        self.switch_roots: list[tuple[sp.Rel, Callable]] = []
        self.switch_roots_jax: list[Callable | None] = []  # the same roots, traceable in JAX
        self.switch_sampled: list[sp.Rel] = []
        self.switch_events: list[sp.Rel] = []
        self.switch_cells: list[sp.Rel] = []
        scalar_params = [p.name for p in self.ir.params if p.over is None]
        time_atoms = set()
        for written, a in atoms.items():
            names = {s.name for s in a.free_symbols} | {str(i.base) for i in a.atoms(sp.Indexed)}
            states = {n for n in names if kind.get(n) in ("field", "compartment", "integral")}
            coords = {n for n in names if kind.get(n) == "axis"}
            vectors = {str(i.base) for i in a.atoms(sp.Indexed)}
            if "t" not in names and not states:
                continue  # fixed in time
            if not states and not coords and not vectors:
                roots = self._time_roots(a, t_sym, scalar_params)
                if roots is not None:
                    self.switch_roots.append((a, roots[0]))
                    self.switch_roots_jax.append(roots[1])
                else:  # no closed form (e.g. sin(t) > 0): located numerically per solve
                    self.switch_sampled.append(a)
                time_atoms.add(written)
                continue
            shape = jax.eval_shape(
                lambda y, th, a=a: self._evaluator(0.0, y, th, self.geom)[0].expr(a.lhs - a.rhs),
                jax.ShapeDtypeStruct((self.layout.size,), np.float64),
                jax.ShapeDtypeStruct((self.params.size + 1,), np.float64),
            ).shape
            if int(np.prod(shape)) == 1:
                self.switch_events.append(a)
            else:
                self.switch_cells.append(a)
        # Between two switch times every condition on time is constant. A solve evaluates
        # them once per segment, at its midpoint, held in an extra slot of theta (NaN outside
        # a solve: then the current time is used). Otherwise the right-hand side at a segment's
        # first point would still see the old branch, and the solver would fight a jump at
        # its starting point.
        self.plan.time_atoms = frozenset(time_atoms)
        self.plan.switch_slot = self.params.size if time_atoms else None

    def _time_roots(self, a: sp.Rel, t_sym, scalar_params):
        """Switch times of a comparison of time and scalar parameters, as a function of theta;
        ``None`` if it cannot be solved in closed form (it then becomes an event)."""
        try:
            sol = sp.solveset(a.lhs - a.rhs, t_sym, domain=sp.S.Reals)
        except (NotImplementedError, TypeError, ValueError):
            return None
        # only a proven finite set of roots will do (sin(t) = 0 has infinitely many: event)
        if not isinstance(sol, sp.FiniteSet) or not sol or any(r.has(t_sym) for r in sol):
            return None
        roots = list(sol)
        syms = [sp.Symbol(n, real=True) for n in scalar_params]
        f = sp.lambdify(syms, roots, ["scipy", "numpy"])
        idx = [self.params.index[n].start for n in scalar_params]
        host = lambda th, f=f, idx=idx: np.atleast_1d(  # noqa: E731
            np.asarray(f(*np.asarray(th)[idx]), dtype=np.complex128)
        )
        try:  # the same roots in JAX (the JAX backend: under grad or vmap theta is traced)
            fj = sp.lambdify(syms, roots, "jax")

            def traced(th, fj=fj, idx=idx):
                vals = fj(*(th[i] for i in idx))
                r = jnp.stack([jnp.asarray(v, dtype=jnp.complex128) for v in vals])
                ok = (jnp.abs(r.imag) < 1e-12) & jnp.isfinite(r.real)
                return jnp.where(ok, r.real, jnp.inf)  # no switch: +inf (never reached)

            traced(jnp.asarray(self.params.theta0))  # fails here if a function has no JAX form
        except Exception:  # noqa: BLE001 - the JAX backend then refuses the model
            traced = None
        return host, traced

    def switch_times(self, theta=None, t_span=None) -> np.ndarray:
        """Times at which a condition on time switches (sorted). Closed-form roots are exact;
        conditions without one (e.g. ``sin(t) > 0``) are sampled at ``SWITCH_SAMPLES`` points
        over ``t_span`` and each sign change is polished to round-off (brentq); a switch pair
        closer together than ``(t1 - t0) / SWITCH_SAMPLES`` can be missed."""
        th = self._th(theta)
        out = []
        for _, f in self.switch_roots:
            r = f(th)
            out.extend(float(x.real) for x in r if abs(x.imag) < 1e-12 and np.isfinite(x.real))
        if self.switch_sampled and t_span is not None:
            out.extend(self._sampled_switch_times(th, *(float(v) for v in t_span)))
        return np.unique(np.asarray(out, dtype=np.float64))

    def switch_times_jax(self, theta, t0, t1, capacity: int, overflow: bool = False):
        """``switch_times`` inside JAX (theta may be traced): closed-form roots, and each
        sampled condition's sign changes on the same ``SWITCH_SAMPLES`` grid, bisected to
        round-off. Sorted, padded with +inf; at most ``capacity`` per sampled condition
        (the JAX backend sizes it from a host count). ``overflow=True``: also whether some
        condition had more (the extra ones are left out)."""
        parts = [f(theta) for f in self.switch_roots_jax]
        over = jnp.bool_(False)
        if self.switch_sampled:
            ts = jnp.linspace(t0, t1, SWITCH_SAMPLES)
            g = jax.vmap(self._time_conditions_impl, in_axes=(0, None, None))(ts, theta, self.geom)
            for k in range(len(self.switch_sampled)):
                gk = g[:, k]

                def cond_k(t, k=k):
                    return self._time_conditions_impl(t, theta, self.geom)[k]

                over = over | (jnp.sum(gk[1:-1] == 0.0) > capacity)
                over = over | (jnp.sum(gk[:-1] * gk[1:] < 0) > capacity)
                (zero,) = jnp.nonzero(gk[1:-1] == 0.0, size=capacity, fill_value=-1)
                on_grid = jax.lax.stop_gradient(ts[1:-1][zero])
                on_grid = on_grid + jax.vmap(implicit_shift, in_axes=(None, 0))(cond_k, on_grid)
                parts.append(jnp.where(zero >= 0, on_grid, jnp.inf))
                (i,) = jnp.nonzero(gk[:-1] * gk[1:] < 0, size=capacity, fill_value=-1)
                lo, hi = ts[jnp.maximum(i, 0)], ts[jnp.maximum(i, 0) + 1]

                def bisect(_, lohi, gk=gk, i=i, cond_k=cond_k):
                    lo, hi = lohi
                    mid = 0.5 * (lo + hi)
                    same = jnp.sign(jax.vmap(cond_k)(mid)) == jnp.sign(gk[jnp.maximum(i, 0)])
                    return jnp.where(same, mid, lo), jnp.where(same, hi, mid)

                lo, hi = jax.lax.fori_loop(0, 64, bisect, (lo, hi))
                root = jax.lax.stop_gradient(0.5 * (lo + hi))
                root = root + jax.vmap(implicit_shift, in_axes=(None, 0))(cond_k, root)
                parts.append(jnp.where(i >= 0, root, jnp.inf))
        if not parts:
            times = jnp.full((1,), jnp.inf)
        else:
            times = jnp.sort(jnp.concatenate([jnp.atleast_1d(p) for p in parts]))
        return (times, over) if overflow else times

    def _time_conditions_impl(self, t, theta, geom):
        ev, _ = self._evaluator(t, jnp.zeros(self.layout.size), theta, geom)
        return jnp.stack([jnp.reshape(ev.expr(a.lhs - a.rhs), ()) for a in self.switch_sampled])

    def _sampled_switch_times(self, th, t0: float, t1: float) -> list[float]:
        from scipy.optimize import brentq

        if "time_conds" not in self._lazy:
            self._lazy["time_conds"] = jax.jit(
                jax.vmap(self._time_conditions_impl, in_axes=(0, None, None))
            )
        ts = np.linspace(t0, t1, SWITCH_SAMPLES)
        g = np.asarray(self._lazy["time_conds"](jnp.asarray(ts), th, self.geom))  # (samples, atoms)
        out = []
        for k in range(g.shape[1]):
            gk = g[:, k]
            out.extend(ts[1:-1][gk[1:-1] == 0.0])
            for i in np.flatnonzero(gk[:-1] * gk[1:] < 0):

                def f(t, k=k):
                    return float(self._lazy["time_conds"](jnp.asarray([t]), th, self.geom)[0, k])

                out.append(brentq(f, ts[i], ts[i + 1], xtol=1e-14, rtol=4 * np.finfo(float).eps))
        return out

    def _switch_impl(self, t, y, theta, geom):
        if not self.switch_events:
            return jnp.zeros((0,))
        ev, _ = self._evaluator(t, y, theta, geom)
        return jnp.stack([jnp.reshape(ev.expr(a.lhs - a.rhs), ()) for a in self.switch_events])

    def switch_values(self, t, y, theta=None) -> np.ndarray:
        """``lhs - rhs`` of each state-dependent switch (``switch_events``) at ``(t, y)``."""
        return np.asarray(
            self._switch(_time(t), jnp.asarray(y)[: self.layout.size], self._th(theta), self.geom)
        )

    # ------------------------------------------------------------------ extinction groups
    def _cell_widths(self) -> np.ndarray:
        """Per state: the product of its cell widths along continuous axes (1 for categories
        and compartments)."""
        w = np.ones(self.layout.size)
        for f in self.ir.fields:
            axes = self.layout.axes[f.name]
            cell = np.ones(self.layout.shapes[f.name])
            for i, ax in enumerate(axes):
                if self.plan.frame.types[ax] == "continuous":
                    shape = [1] * len(axes)
                    shape[i] = -1
                    cell = cell * np.reshape(np.asarray(self.geom[ax]["widths"]), shape)
            w[self.layout.slices[f.name]] = cell.ravel()
        return w

    def _size(self, name: str, ev: Evaluator):
        """Weights of a field (in its own axis order, broadcastable) or volume of a
        compartment (0-d); 1.0 without."""
        e = self.sizes.get(name)
        if e is None:
            return 1.0
        axes = self.layout.axes.get(name, ())
        return self._on(ev.expr(e), axes, f"the {'weights' if axes else 'volume'} of '{name}'")

    def _measure_impl(self, theta, geom):
        if not self.sizes:
            return jnp.asarray(self.cell_widths)
        ev = Evaluator(self.plan, t=0.0, theta=theta, geom=geom, fields={}, compartments={})
        parts = []
        for name in self.layout.names:
            sl = self.layout.slices[name]
            size = jnp.broadcast_to(self._size(name, ev), self.layout.shapes[name])
            parts.append(jnp.asarray(self.cell_widths[sl]) * jnp.ravel(size))
        return jnp.concatenate(parts)

    def group_weights(self, theta=None) -> list[np.ndarray]:
        """Per extinction group: ``w @ y`` is its total amount."""
        w = self.measure(theta)
        return [w * g.members for g in self.extinction_groups]

    def _member_mask(self, e) -> np.ndarray:
        """States of an extinction group: whole members, or the slices in ``e.where``."""
        mask = np.zeros(self.layout.size, dtype=bool)
        for i, m in enumerate(e.group):
            part = np.ones(self.layout.shapes[m], dtype=bool)
            axes = self.layout.axes[m]
            for ax, lab in e.member_where(i):
                d = axes.index(ax)
                keep = np.asarray(self.ir.axis(ax).labels) == lab
                shape = [1] * len(axes)
                shape[d] = -1
                part = part & np.reshape(keep, shape)
            mask[self.layout.slices[m]] |= part.ravel()
        return mask

    def group_totals(self, y, theta=None) -> np.ndarray:
        """Total amount of each extinction group at state(s) ``y`` (numpy)."""
        y = np.asarray(y)[..., : self.layout.size]
        return np.stack([y @ w for w in self.group_weights(theta)], axis=-1)

    def _apply_floor(self, y, theta, geom):
        """Noise floor: a floored group is seen as zero by every equation. Its own
        rates are computed from that zero, so sources from outside still add to it, but it
        cannot feed itself. During a solve a group is floored while it is disarmed (it has
        not reached its threshold since the start or its last extinction), held in its theta
        slot; the floor therefore switches only where the solver stops anyway (ADR 0005).
        Outside a solve (slot NaN) it is floored while its total is below the threshold."""
        w = self._measure_impl(theta, geom)
        for g, slot in zip(self._floors, self.floor_slots, strict=True):
            flag = theta[slot]
            by_total = jnp.dot(w * jnp.asarray(g.members), y) < g.threshold
            below = jnp.where(jnp.isnan(flag), by_total, flag > 0.5)
            y = jnp.where(jnp.asarray(g.members) & below, 0.0, y)
        return y

    # ------------------------------------------------------------------ evaluators
    def _evaluator(
        self, t, y, theta, geom, q=None, *, floor: bool = True
    ) -> tuple[Evaluator, dict[str, object]]:
        """Evaluator for state ``y``. With ``q`` (flat integral values), the integrals are
        taken from it instead of being computed from ``y`` (lifting, ADR 0003). Groups with a
        noise floor are applied (``floor=False`` for outputs, which report the state as is)."""
        if floor and self._floors:
            y = self._apply_floor(y, theta, geom)
        state = self.layout.unpack(y)
        frame = self.plan.frame
        fields = {f.name: frame.from_layout(state[f.name], f.axes) for f in self.ir.fields}
        comps = {c.name: state[c.name] for c in self.ir.compartments}
        cache = None if q is None else self._split_q(q)
        ev = Evaluator(
            self.plan,
            t=t,
            theta=theta,
            geom=geom,
            fields=fields,
            compartments=comps,
            integral_cache=cache,
            end_hook=self._end_hook,
        )
        return ev, state

    def _on(self, value, axes: tuple[str, ...], what: str):
        return self.plan.frame.to_layout(value, axes, what)

    # ------------------------------------------------------------------ right-hand side
    def _rhs_impl(self, t, y, theta, geom):
        return self.layout.pack(self._assemble(t, y, theta, geom, None))

    def _ledger_impl(self, t, y, theta, geom):
        book: dict = {}
        self._assemble(t, y, theta, geom, book)
        return self._stack_ledger(book)

    def _aug_impl(self, t, z, theta, geom):
        """Right-hand side of the state followed by the ledger channels (quadrature states)."""
        book: dict = {}
        rates = self._assemble(t, z[: self.layout.size], theta, geom, book)
        return jnp.concatenate([self.layout.pack(rates), self._stack_ledger(book)])

    def _stack_ledger(self, book):
        keys = [(target, label) for target, _, label in self.ledger_entries]
        if set(book) != set(keys):  # pragma: no cover - internal consistency
            raise CompileError(f"ledger channels {sorted(set(book) ^ set(keys))} out of step.")
        if not keys:
            return jnp.zeros((0,))
        return jnp.stack([jnp.reshape(book[k], ()) for k in keys])

    def _assemble(self, t, y, theta, geom, book, q=None):
        """Rates of every field and compartment; with ``book`` a dict, also fill in the ledger
        channel amounts (keyed as in ``ledger_entries``). With ``q``, integrals are held
        at those values. Ledger channels are amounts per time (values x widths x weights, or
        x volume)."""
        ev, state = self._evaluator(t, y, theta, geom, q)
        rates: dict[str, object] = {}
        routed: dict[str, list] = {c.name: [] for c in self.ir.compartments}
        sizes = {n: self._size(n, ev) for n in self.sizes}
        for f in self.ir.fields:
            rates[f.name] = self._field_rate(
                f.name, state[f.name], ev, geom, routed, book, sizes.get(f.name, 1.0)
            )
        self._transfer(ev, state, rates, sizes, book)
        for c in self.ir.compartments:
            vol = sizes.get(c.name, 1.0)
            total = jnp.zeros(())
            for (_, e), key in zip(self.rate_parts[c.name], self._rate_keys[c.name], strict=True):
                r = jnp.reshape(self._on(ev.expr(e), (), f"the equation of '{c.name}'"), ())
                total = total + r
                if book is not None:
                    book[key] = r * vol
            for key, r in routed[c.name]:  # amounts per time
                total = total + r / vol
                if book is not None:
                    book[key] = r
            rates[c.name] = rates.get(c.name, 0.0) + total
        return rates

    def _transfer(self, ev: Evaluator, state, rates, sizes, book):
        """``m.transfer`` terms: amount per time = rate x the source's value, element by
        element over the axes the slices leave free (summed into a compartment)."""
        frame = self.plan.frame
        for (tr, rate), (out_key, in_key) in zip(self.transfers, self._transfer_keys, strict=True):
            if tr.source in self.layout.axes and self.layout.axes[tr.source]:
                src = self._pick(ev.fields[tr.source], tr.source_where)
            else:
                src = jnp.reshape(state[tr.source], (1,) * frame.ndim)
            # a rate varying over an axis a slice fixes (Kp[organ] on C[organ=liver]) is
            # taken at that category: the source's, else the destination's
            r = self._pick(self._pick(ev.expr(rate), tr.source_where), tr.to_where)
            amount = r * src  # frame: one value per free element
            # every element of the slice(s) counts, also where the amount is the same for all
            elements = [1] * frame.ndim
            for name, where in ((tr.source, tr.source_where), (tr.to, tr.to_where)):
                fixed = dict(where)
                for a in self.layout.axes.get(name, ()) if name else ():
                    if a not in fixed:
                        elements[frame.pos(a)] = frame.sizes[a]
            amount = jnp.broadcast_to(amount, tuple(elements))
            total = jnp.sum(amount)
            self._deliver(tr.source, tr.source_where, -amount, -total, rates, sizes)
            if tr.to is not None:
                self._deliver(tr.to, tr.to_where, amount, total, rates, sizes)
            if book is not None:
                book[out_key] = -total
                if in_key is not None:
                    book[in_key] = total

    def _deliver(self, name, where, amount, total, rates, sizes):
        """Add ``amount`` (frame, per element of the slice) to ``name``'s rate, as a change of
        its value (divided by its weights or volume)."""
        axes = self.layout.axes.get(name, ())
        size = sizes.get(name, 1.0)
        if not axes:
            add = total / size
        else:
            shape = self.layout.shapes[name]
            add = jnp.broadcast_to(self._on(amount, axes, "a transfer"), shape) / size
            mask = self._mask(name, where)
            if mask is not None:
                add = add * mask
        rates[name] = rates.get(name, 0.0) + add

    def _field_rate(self, name: str, u, ev: Evaluator, geom, routed, book, size=1.0):
        axes = self.layout.axes[name]
        shape = u.shape
        du = jnp.zeros(shape)
        terms = zip(
            self.field_terms[name], self.term_parts[name], self._term_keys[name], strict=True
        )
        for (t, e), pieces, keys in terms:
            where = f"{t.kind} term of '{name}'"
            mask = self._mask(name, t.where)
            if t.kind == "reaction":
                cs = [self._on(ev.expr(p), axes, where) for _, p in pieces]
            elif t.kind == "hazard":
                cs = [-self._on(ev.expr(p), axes, where) * u for _, p in pieces]
            elif t.kind in ("switch", "mutate"):
                along, moves = self.generators[t.transitions]
                rated = []
                for o, d, r in moves:
                    rate = self._on(ev.expr(r), axes, f"transition rate of {t.transitions}")
                    rated.append((o, d, rate if mask is None else rate * mask))
                src = (
                    u
                    if t.kind == "switch"
                    else jnp.broadcast_to(self._on(ev.expr(e), axes, where), shape)
                )
                dims = tuple(axes.index(a) for a in along)
                cs = [_apply_moves(rated, src, dims, size, t.kind == "mutate")]
                mask = None  # already restricted through the rates
            else:
                continue
            for c, key in zip(cs, keys, strict=True):
                if mask is not None:
                    c = c * mask
                du = du + c
                if book is not None:
                    book[key] = self._total(jnp.broadcast_to(c, shape) * size, axes, geom)
        for ax, spec in self.transport[name].items():
            du = du + self._transport(name, ax, spec, u, ev, geom, routed, book, size)
        return du

    def _coefficients(self, name, ax, spec, ev: Evaluator):
        """Transport of ``name`` along ``ax``: ``([(scheme, v)], [D])``, the velocity on the
        faces and the diffusivity on the cells (at most one of each: several terms act as
        one)."""
        axes = self.layout.axes[name]
        dim = axes.index(ax)
        shape = self.layout.shapes[name]
        n = shape[dim]
        velocities, diffusivities = [], []
        face_ev = ev.at_faces(ax)  # velocities live on the faces (cell values interpolated)
        for t, e in spec["terms"]:
            where = f"{t.kind} of '{name}' along '{ax}'"
            if t.kind == "advection":
                v = self._on(face_ev.expr(e), axes, where)
                velocities.append((t.scheme, _to_faces(v, dim, n, shape)))
            else:
                D = self._on(ev.expr(e), axes, where)
                diffusivities.append(D if D.size == 1 else jnp.broadcast_to(D, shape))
        # several terms act as one: upwind on the total face velocity, and one
        # diffusivity (the sum), so splitting a velocity into parts changes nothing
        schemes = {sc for sc, _ in velocities}
        if len(schemes) > 1:
            raise CompileError(
                f"'{name}' is advected along '{ax}' by several terms with different schemes "
                f"({', '.join(sorted(schemes))}); use one scheme for all of them."
            )
        if velocities:
            velocities = [(velocities[0][0], sum(v for _, v in velocities))]
        if len(diffusivities) > 1:
            total_D = sum(diffusivities)
            diffusivities = [total_D if total_D.size == 1 else jnp.broadcast_to(total_D, shape)]
        return velocities, diffusivities

    def _end(self, name, ax, side, spec, u, ev: Evaluator, geom, velocities, diffusivities):
        """One end of ``name`` along ``ax``: the flux through it (positive toward larger
        coordinates, the field's shape without the axis) and the field's value there, as the
        boundary condition defines them. The value is what a formula reads as
        ``name(ax=side)``; computing both here keeps the two consistent (IR 0.10)."""
        axes = self.layout.axes[name]
        dim = axes.index(ax)
        shape = u.shape
        n = shape[dim]
        rest_axes = axes[:dim] + axes[dim + 1 :]
        rest = _rest(shape, dim)
        b = spec["ends"][side]
        idx, cell = (0, 0) if side == "lower" else (n, n - 1)
        width = geom[ax]["widths"][cell]
        ue = _take(u, cell, dim)
        sgn = 1.0 if side == "lower" else -1.0  # flux along the axis -> flux into the domain
        zero = jnp.zeros(rest)

        def formula(e, what):
            return jnp.broadcast_to(
                self._on(ev.expr(e), rest_axes, f"{what} of '{name}' at {ax}={side}"), rest
            )

        De = None
        if diffusivities:
            D = diffusivities[0]
            De = jnp.reshape(D, ()) if D.size == 1 else _take(D, cell, dim)
        g = zero if De is None else De / (0.5 * width)  # conductance of the half-cell
        ve = _take(velocities[0][1], idx, dim) if velocities else None
        v_into = zero if ve is None else sgn * ve

        v_in, v_out = jnp.maximum(v_into, 0.0), jnp.minimum(v_into, 0.0)

        def level(a_in, h):
            """Value at the end where the flux into the domain is ``a_in - h * value`` and
            the transport carries ``v_in * value + v_out * edge + g * (value - edge)`` into
            it: upwind, so a flow leaving through the end carries the edge cell's value
            (``v_in``, ``v_out``: the parts of the velocity into and out of the domain).
            Where nothing sets the value (the flow leaves, nothing diffuses, h <= 0) it is
            the edge cell's (listed by explain())."""
            den = v_in + g + h
            ok = den > 0
            return jnp.where(ok, (a_in + (g - v_out) * ue) / jnp.where(ok, den, 1.0), ue)

        if b.kind == "zero_flux":
            return zero, ue
        if b.kind == "outflow":
            F = zero
            if ve is not None:
                F = F + advective_outflow_flux(u, ve, dim=dim, side=side)
            return F, ue
        if b.kind == "inflow":
            J = formula(b.value, "boundary")
            # J enters; anything the flow carries outward through this end leaves (upwind)
            F = sgn * J
            if ve is not None:
                F = F + (jnp.minimum(ve, 0.0) if side == "lower" else jnp.maximum(ve, 0.0)) * ue
            return F, level(J + v_out * ue, 0.0)
        if b.kind == "flux":  # flux along the axis = A + C * value (IR 0.10)
            A = formula(b.value, "boundary condition")
            C = formula(b.h, "boundary condition")
            a_in, h = sgn * A, -sgn * C
            cf = level(a_in, h)
            # where nothing sets the value (see level), the flow carries out what reaches it
            J = jnp.where(v_in + g + h > 0, a_in - h * cf, v_out * ue)
            return sgn * J, cf
        if b.kind == "dirichlet":
            ub = formula(b.value, "boundary")
            F = upwind_end_flux(u, ub, velocities, dim=dim, side=side, idx=idx)
            if De is not None:
                F = F + dirichlet_diffusive_flux(u, ub, De, dim=dim, side=side, width_edge=width)
            return F, ub
        # robin: the exchange, plus any flow across the end
        out = formula(b.value, "boundary")
        h = jnp.broadcast_to(
            self._on(ev.expr(b.h), rest_axes, f"Robin coefficient of '{name}'"), rest
        )
        if velocities and De is not None:  # flow and diffusion: one half-cell balance
            F = robin_flux(u, h, out, dim=dim, side=side, D_edge=De, width_edge=width, v_face=ve)
            v_in = jnp.maximum(v_into, 0.0)
            den = v_in + h + g
            ok = den > 0
            cf = jnp.where(ok, ((v_in + h) * out + g * ue) / jnp.where(ok, den, 1.0), ue)
            return F, cf
        # upwind flow (outside level in, edge value out) plus the exchange
        F = upwind_end_flux(u, out, velocities, dim=dim, side=side, idx=idx)
        F = F + robin_flux(
            u, h, out, dim=dim, side=side, D_edge=De, width_edge=width if De is not None else None
        )
        if De is None:
            return F, ue
        den = h + g
        ok = den > 0
        return F, jnp.where(ok, (h * out + g * ue) / jnp.where(ok, den, 1.0), ue)

    def _end_hook(self, key: str, ev: Evaluator):
        """``F(z=side)`` in a formula: the field's value at that end (frame, size 1 along z)."""
        field, ax, side = self.plan.tables.ends[key]
        axes = self.layout.axes[field]
        u = self._on(ev.fields[field], axes, f"field '{field}'")
        spec = self.transport[field].get(ax)
        if spec is None:  # no transport and default ends along this axis: the edge cell
            closed = _Boundary("zero_flux", default=True)
            spec = {"terms": [], "ends": {"lower": closed, "upper": closed}}
        vel, dif = self._coefficients(field, ax, spec, ev)
        _, cf = self._end(field, ax, side, spec, u, ev, ev.geom, vel, dif)
        dim = axes.index(ax)
        return self.plan.frame.from_layout(cf, axes[:dim] + axes[dim + 1 :])

    def _transport(self, name, ax, spec, u, ev: Evaluator, geom, routed, book, size=1.0):
        axes = self.layout.axes[name]
        dim = axes.index(ax)
        shape = u.shape
        n = shape[dim]
        g = geom[ax]
        rest_axes = axes[:dim] + axes[dim + 1 :]
        velocities, diffusivities = self._coefficients(name, ax, spec, ev)
        ends = spec["ends"]
        flux, level = {}, {}
        for side in ("lower", "upper"):
            flux[side], level[side] = self._end(
                name, ax, side, spec, u, ev, geom, velocities, diffusivities
            )

        # interior faces
        F_int = jnp.zeros(shape[:dim] + (max(n - 1, 0),) + shape[dim + 1 :])
        if n > 1:
            for scheme, v in velocities:
                edge, one_sided = {}, {"lower": False, "upper": False}
                if scheme == "sharp":
                    for side in ("lower", "upper"):
                        b = ends[side]
                        if b.kind == "robin" or (b.kind in ("inflow", "flux") and diffusivities):
                            # the edge value is not set by the flow alone (diffusion, or an
                            # exchange resistance): limit the end slope from the interior
                            one_sided[side] = True
                        elif b.kind in ("inflow", "flux", "dirichlet"):
                            edge[side] = level[side]
                F_int = F_int + advective_flux_interior(
                    u,
                    jax.lax.slice_in_dim(v, 1, n, axis=dim),
                    dim=dim,
                    centers=g["centers"],
                    faces=g["faces"],
                    scheme=scheme,
                    lower_value=edge.get("lower"),
                    upper_value=edge.get("upper"),
                    one_sided=(one_sided["lower"], one_sided["upper"]),
                )
            for D in diffusivities:
                F_int = F_int + diffusive_flux_interior(
                    u, jnp.reshape(D, ()) if D.size == 1 else D, dim=dim, widths=g["widths"]
                )

        for side in ("lower", "upper"):
            b = ends[side]
            F = flux[side]
            # amounts: weights never vary along a continuous axis, so drop that dimension
            w_rest = size if np.ndim(size) == 0 else jnp.take(size, 0, axis=dim)
            if b.kind != "zero_flux" and book is not None:
                into = F if side == "lower" else -F
                book[self._end_keys[(name, ax, side)]] = self._total(into * w_rest, rest_axes, geom)
            if b.to is not None:  # outflow, or dirichlet/robin exchange with a compartment
                leaving = F if side == "upper" else -F
                routed[b.to].append(
                    (
                        self._routed_keys[(name, ax, side)],
                        self._total(leaving * w_rest, rest_axes, geom),
                    )
                )

        faces = assemble_faces(flux["lower"], F_int, flux["upper"], dim)
        return divergence(faces, g["widths"], dim)

    def _total(self, x, axes: tuple[str, ...], geom):
        """Amount in ``x`` over ``axes``: widths on continuous axes, plain sums on categories."""
        for i, ax in enumerate(axes):
            if self.plan.frame.types[ax] == "continuous":
                shape = [1] * len(axes)
                shape[i] = -1
                x = x * jnp.reshape(geom[ax]["widths"], shape)
        return jnp.sum(x)

    # ------------------------------------------------------------------ sparse Jacobian
    # ADR 0003: J = dg/dy + dg/dq . dq/dy, with q the integrals the model declares. Field rows
    # come from coloured forward derivatives of g (integrals held fixed) and of g with respect
    # to q; dq/dy from coloured reverse derivatives; compartment (and ledger) rows, which
    # depend on whole fields, from one reverse derivative each.

    def _ends_used(self) -> list[str]:
        """Field-at-end values the model's formulas may read (both ends of every axis a
        formula reads one end of; an unused one costs one small row, never accuracy)."""
        text = self.ir.to_json()
        used = []
        for key, (field, ax, _) in self.plan.tables.ends.items():
            if re.search(rf"\b{re.escape(field)}\s*\(\s*{re.escape(ax)}\s*=", text):
                used.append(key)
        return used

    def _integral_layout(self) -> None:
        self._end_names = tuple(self._ends_used())
        names = [i.name for i in self.ir.integrals] + list(self._end_names)
        shapes = {}
        if names:
            y = jax.ShapeDtypeStruct((self.layout.size,), jnp.float64)
            th = jax.ShapeDtypeStruct((self.n_theta,), jnp.float64)
            out = jax.eval_shape(
                lambda y, th, g: self._integrals_impl(0.0, y, th, g), y, th, self.geom
            )
            shapes = {n: tuple(out[n].shape) for n in names}
        offsets, i = {}, 0
        for n in names:
            size = int(np.prod(shapes[n], dtype=np.int64))
            offsets[n] = slice(i, i + size)
            i += size
        self._q_names, self._q_shapes, self._q_slices, self.n_integrals = (
            tuple(names),
            shapes,
            offsets,
            i,
        )

    def _integrals_impl(self, t, y, theta, geom) -> dict[str, object]:
        ev, _ = self._evaluator(t, y, theta, geom)
        names = [i.name for i in self.ir.integrals] + list(getattr(self, "_end_names", ()))
        return {n: jnp.asarray(ev.name(n)) for n in names}

    def _q_impl(self, t, y, theta, geom):
        """All integrals, flattened (each in the frame's axis order)."""
        vals = self._integrals_impl(t, y, theta, geom)
        if not vals:
            return jnp.zeros((0,))
        return jnp.concatenate([jnp.ravel(vals[n]) for n in self._q_names])

    def _split_q(self, q) -> dict[str, object]:
        return {n: jnp.reshape(q[self._q_slices[n]], self._q_shapes[n]) for n in self._q_names}

    @property
    def n_field_states(self) -> int:
        return sum(int(np.prod(self.layout.shapes[f.name])) for f in self.ir.fields)

    @property
    def sparse_structure(self) -> dict:
        """Patterns and colourings of the Jacobian parts (built once, on first use)."""
        if "structure" not in self._lazy:
            self._lazy["structure"] = self._build_structure()
        return self._lazy["structure"]

    def _build_structure(self) -> dict:
        from tenqsp import sparsity as spy

        frame, n, n_f, n_q = (
            self.plan.frame,
            self.layout.size,
            self.n_field_states,
            self.n_integrals,
        )
        nodes = {
            name: spy.Node(
                name,
                self.layout.axes[name],
                self.layout.slices[name].start,
                int(np.prod(self.layout.shapes[name], dtype=np.int64)),
            )
            for name in self.layout.names
        }
        for q in self._q_names:
            axes = tuple(a for a, k in zip(frame.axes, self._q_shapes[q], strict=True) if k > 1)
            sl = self._q_slices[q]
            nodes[q] = spy.Node(q, axes, n + sl.start, sl.stop - sl.start)

        # local part: field rows x (state, integrals)
        rows, cols = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
        for field, deps in spy.field_dependencies(self).items():
            for src, reach in deps.items():
                r, c = spy.block(nodes[field], nodes[src], reach, frame)
                rows.append(r)
                cols.append(c)
        A = _sorted_pattern(np.concatenate(rows), np.concatenate(cols), (n_f, n + n_q))
        colours = spy.colour_columns(A["rows"], A["cols"], (n_f, n + n_q))

        # integrals: q rows x state columns, coloured by rows (reverse mode)
        rows, cols = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
        for q in self._q_names:
            for src, over in spy.quantity_sources(self, q).items():
                if not self.layout.axes[src]:
                    continue  # compartment columns: forward mode, one pass each (below)
                reach = {ax: spy.Reach(band=spy.FULL if ax in over else 0) for ax in frame.axes}
                qn = spy.Node(q, nodes[q].axes, nodes[q].offset - n, nodes[q].size)
                r, c = spy.block(qn, nodes[src], reach, frame)
                rows.append(r)
                cols.append(c)
        Dq = _sorted_pattern(np.concatenate(rows), np.concatenate(cols), (n_q, n))
        row_colours = spy.colour_columns(Dq["cols"], Dq["rows"], (n, n_q))
        n_colours = int(colours.max(initial=-1)) + 1
        n_row_colours = int(row_colours.max(initial=-1)) + 1
        comp_idx = np.asarray(
            [self.layout.slices[c.name].start for c in self.ir.compartments], dtype=np.int64
        )
        arrays = {
            "comp_idx": jnp.asarray(comp_idx),
            "colours": jnp.asarray(colours),
            "A_rows": jnp.asarray(A["rows"]),
            "A_pick": jnp.asarray(colours[A["cols"]]),
            "row_colours": jnp.asarray(row_colours),
            "D_cols": jnp.asarray(Dq["cols"]),
            "D_pick": jnp.asarray(row_colours[Dq["rows"]]),
        }
        return {
            "A": A,
            "Dq": Dq,
            "comp_idx": comp_idx,
            "n_colours": n_colours,
            "n_row_colours": n_row_colours,
            "arrays": arrays,
        }

    def _local(self, t, y, theta, geom, arrays):
        if "local_jit" not in self._lazy:
            S = self.sparse_structure
            nc, nr = S["n_colours"], S["n_row_colours"]
            self._lazy["local_jit"] = jax.jit(
                lambda t, y, th, g, a: self._local_impl(t, y, th, g, a, nc, nr)
            )
        return self._lazy["local_jit"](t, y, theta, geom, arrays)

    def _local_impl(self, t, y, theta, geom, S, n_colours: int, n_row_colours: int):
        """Values of the local part (pattern order) and of dq/dy (pattern order)."""
        n, n_f = self.layout.size, self.n_field_states
        q = self._q_impl(t, y, theta, geom)
        w = jnp.concatenate([y, q])

        def g(w):
            rates = self._assemble(t, w[:n], theta, geom, None, q=w[n:])
            return self.layout.pack(rates)[:n_f]

        seeds = (S["colours"][None, :] == jnp.arange(n_colours)[:, None]).astype(w.dtype)
        comp = jax.vmap(lambda s: jax.jvp(g, (w,), (s,))[1])(seeds)  # (colours, n_f)
        A_data = comp[S["A_pick"], S["A_rows"]]
        if n_row_colours:
            _, pull = jax.vjp(lambda y: self._q_impl(t, y, theta, geom), y)
            cot = (S["row_colours"][None, :] == jnp.arange(n_row_colours)[:, None]).astype(y.dtype)
            compq = jax.vmap(lambda c: pull(c)[0])(cot)  # (row colours, n)
            D_data = compq[S["D_pick"], S["D_cols"]]
        else:
            D_data = jnp.zeros((0,))
        # dq / d(compartments): a compartment can enter every integral, so by forward mode
        if self.n_integrals and S["comp_idx"].size:
            unit = jax.nn.one_hot(S["comp_idx"], n, dtype=y.dtype)
            q_of = lambda y: self._q_impl(t, y, theta, geom)  # noqa: E731
            Dc = jax.vmap(lambda u: jax.jvp(q_of, (y,), (u,))[1])(unit).T  # (n_q, n_comp)
        else:
            Dc = jnp.zeros((self.n_integrals, 0))
        return A_data, D_data, Dc

    def _dense_rows_impl(self, t, y, theta, geom, ledger: bool):
        """Exact rows of the compartments (and, with ``ledger``, the ledger channels)."""
        n_f = self.n_field_states

        def rows(y):
            if ledger:
                book: dict = {}
                rates = self._assemble(t, y, theta, geom, book)
                return jnp.concatenate([self.layout.pack(rates)[n_f:], self._stack_ledger(book)])
            return self._rhs_impl(t, y, theta, geom)[n_f:]

        return jax.jacrev(rows)(y)

    def sparse_jacobian(self, t, y, theta=None, *, ledger: bool = False) -> sps.csc_matrix:
        """``d rhs / d y`` as a sparse matrix, never formed densely (ADR 0003).

        With ``ledger=True``, the Jacobian of the system the solver integrates: the state
        followed by the ledger channels (whose own columns are zero).
        """
        S = self.sparse_structure
        n, n_f = self.layout.size, self.n_field_states
        y = jnp.asarray(np.asarray(y, dtype=np.float64)[:n])
        th, tt = self._th(theta), _time(t)
        A_data, D_data, Dc = self._local(tt, y, th, self.geom, S["arrays"])
        dense = self._dense_aug if ledger else self._dense
        rows = dense(tt, y, th, self.geom) if (n > n_f or ledger) else np.zeros((0, n))
        return self.assemble_sparse((A_data, D_data, Dc, rows), ledger=ledger)

    def jacobian_pieces(self, t, y, theta, *, ledger: bool = False):
        """The parts of ``sparse_jacobian``, computed inside JAX (traceable; the JAX
        backend's sparse tier, step 4.5): the local values (pattern order), the integrals'
        rows (pattern order) and compartment columns, and the dense rows of the
        compartments (and, with ``ledger``, the ledger channels). Fixed shapes; assembled
        on the host by ``assemble_sparse``."""
        S = self.sparse_structure
        n, n_f = self.layout.size, self.n_field_states
        y = y[:n]
        A_data, D_data, Dc = self._local_impl(
            t, y, theta, self.geom, S["arrays"], S["n_colours"], S["n_row_colours"]
        )
        if n > n_f or ledger:
            rows = self._dense_rows_impl(t, y, theta, self.geom, ledger)
        else:
            rows = jnp.zeros((0, n))
        return A_data, D_data, Dc, rows

    def assemble_sparse(self, pieces, *, ledger: bool = False) -> sps.csc_matrix:
        """The sparse Jacobian from ``jacobian_pieces`` (numbers, on the host)."""
        S = self.sparse_structure
        n, n_f, n_q = self.layout.size, self.n_field_states, self.n_integrals
        A_data, D_data, Dc, rows = (np.asarray(p) for p in pieces)
        A = S["A"]
        Am = sps.csr_matrix((A_data, A["cols"], A["indptr"]), shape=(n_f, n + n_q))
        J_f = Am[:, :n]
        if n_q:
            Dq = S["Dq"]
            Dm = sps.csr_matrix((D_data, Dq["cols"], Dq["indptr"]), shape=(n_q, n))
            if S["comp_idx"].size:
                r, c = np.nonzero(Dc)
                Dm = Dm + sps.csr_matrix((Dc[r, c], (r, S["comp_idx"][c])), shape=(n_q, n))
            J_f = J_f + Am[:, n:] @ Dm
        if n > n_f or ledger:
            J = sps.vstack([J_f, sps.csr_matrix(rows)], format="csr")
        else:
            J = J_f
        if ledger and len(self.ledger_entries):
            J = sps.hstack([J, sps.csr_matrix((J.shape[0], len(self.ledger_entries)))])
        return J.tocsc()

    def jacobian_jax(self, t, y, theta, *, ledger: bool = False):
        """``sparse_jacobian`` inside JAX, assembled as a dense matrix (the JAX backend's
        dense tier): the same coloured passes (ADR 0003), so a Jacobian costs ~20 directional
        derivatives instead of ``n``. ``theta`` is the full vector (traceable)."""
        S = self.sparse_structure
        n, n_f, n_q = self.layout.size, self.n_field_states, self.n_integrals
        y = y[:n]
        A_data, D_data, Dc = self._local(t, y, theta, self.geom, S["arrays"])
        A = S["A"]
        rows_A = np.repeat(np.arange(n_f), np.diff(np.asarray(A["indptr"])))
        Am = jnp.zeros((n_f, n + n_q)).at[rows_A, np.asarray(A["cols"])].add(A_data)
        J_f = Am[:, :n]
        if n_q:
            Dq = S["Dq"]
            rows_D = np.repeat(np.arange(n_q), np.diff(np.asarray(Dq["indptr"])))
            Dm = jnp.zeros((n_q, n)).at[rows_D, np.asarray(Dq["cols"])].add(D_data)
            comp_idx = np.asarray(S["comp_idx"])
            if comp_idx.size:
                Dm = Dm.at[:, comp_idx].add(Dc)
            J_f = J_f + Am[:, n:] @ Dm
        parts = [J_f]
        if n > n_f or ledger:
            parts.append(self._dense_rows_impl(t, y, theta, self.geom, ledger))
        J = jnp.concatenate(parts) if len(parts) > 1 else J_f
        n_l = len(self.ledger_entries) if ledger else 0
        if n_l:
            J = jnp.concatenate([J, jnp.zeros((J.shape[0], n_l))], axis=1)
        return J

    # ------------------------------------------------------------------ initial state
    def _y0_impl(self, theta, geom):
        frame = self.plan.frame
        zeros_state = {f.name: jnp.zeros((1,) * frame.ndim) for f in self.ir.fields}
        zeros_comp = {c.name: jnp.zeros(()) for c in self.ir.compartments}
        ev = Evaluator(
            self.plan, t=0.0, theta=theta, geom=geom, fields=zeros_state, compartments=zeros_comp
        )
        # a value is a density: a shape in it is the curve itself (density(normal(z, ...)))
        ev_density = ev.with_raw_shapes()
        values = {n: jnp.zeros(self.layout.shapes[n]) for n in self.layout.names}
        for spec, value, total, shape in self.inits:
            name = spec.target
            what = f"initial condition of '{name}'"
            if name not in self.layout.axes or not self.layout.axes[name]:  # compartment
                values[name] = jnp.reshape(self._on(ev.expr(value), (), what), ())
                continue
            axes = self.layout.axes[name]
            fixed = dict(spec.where)
            free = tuple(a for a in axes if a not in fixed)
            free_shape = tuple(frame.sizes[a] for a in free)
            if value is not None:
                arr = jnp.broadcast_to(
                    self._on(self._cell_average(ev_density, value, free), free, what), free_shape
                )
            else:
                arr = self._spread(ev, geom, name, spec.where, free, total, shape)
            idx = tuple(
                self.ir.axis(a).labels.index(fixed[a]) if a in fixed else slice(None) for a in axes
            )
            values[name] = values[name].at[idx].set(arr)
        return self.layout.pack(values)

    def _spread(self, ev: Evaluator, geom, name: str, where, free, total, shape):
        """An amount ``total`` spread over the free axes of ``name[where]`` by ``shape``: the
        values of the slice's cells (cut to the grid, scaled so they hold exactly ``total``)."""
        what = f"amount and shape of '{name}'"
        axes, fixed = self.layout.axes[name], dict(where)
        free_shape = tuple(self.plan.frame.sizes[a] for a in free)
        dens = jnp.broadcast_to(
            self._on(self._cell_average(ev, shape, free), free, what), free_shape
        )
        amount = jnp.reshape(self._on(ev.expr(total), (), what), ())
        size = self._size(name, ev)
        if np.ndim(size):  # weights: the cells hold amount / (width x weight)
            full = jnp.broadcast_to(size, self.layout.shapes[name])
            size = full[
                tuple(
                    self.ir.axis(a).labels.index(fixed[a]) if a in fixed else slice(None)
                    for a in axes
                )
            ]
        return amount * dens / self._total(dens * size, free, geom)

    def _cell_average(self, ev: Evaluator, e: sp.Expr, free: tuple[str, ...]):
        """Cell average of a formula over the free continuous axes it depends on
        (3-point Gauss per axis; shape functions are exact by construction)."""
        axes = sorted(self.plan.coordinate_axes(e) & set(free))
        if not axes:
            return ev.expr(e)
        frame, geom = self.plan.frame, ev.geom
        out = 0.0
        for combo in itertools.product(range(3), repeat=len(axes)):
            coords, w = {}, 1.0
            for ax, k in zip(axes, combo, strict=True):
                g = geom[ax]
                coords[ax] = frame.along(ax, g["centers"] + GAUSS_NODES[k] * 0.5 * g["widths"])
                w *= GAUSS_WEIGHTS[k]
            out = out + w * ev.with_coords(coords).expr(e)
        return out

    # ------------------------------------------------------------------ doses, observables
    def _dose_impl(self, theta, geom):
        """The amount of every entry of ``dose_schedule``."""
        if not self.dose_parts:
            return jnp.zeros((0,))
        ev, _ = self._evaluator(0.0, jnp.zeros(self.layout.size), theta, geom)
        measure, out = None, []
        for target, where, free, e, shape, cells in self.dose_parts:
            if not free:
                out.append(jnp.reshape(self._on(ev.expr(e), (), "a dose amount"), (1,)))
                continue
            what = f"dose into '{target}'"
            if shape is None:  # a density added to the field (a shape in it: the curve itself)
                arr = jnp.broadcast_to(
                    self._on(self._cell_average(ev.with_raw_shapes(), e, free), free, what),
                    tuple(self.plan.frame.sizes[a] for a in free),
                )
            else:
                arr = self._spread(ev, geom, target, where, free, e, shape)
            if measure is None:
                measure = self._measure_impl(theta, geom)
            out.append(jnp.ravel(arr) * measure[cells])
        return jnp.concatenate(out)

    def _arrival_impl(self, theta, geom):
        """Per state: the lag of a dose into it, and the fraction of it that arrives."""
        n = self.layout.size
        lag, frac = jnp.zeros(n), jnp.ones(n)
        if not self.has_arrival:
            return lag, frac
        ev, _ = self._evaluator(0.0, jnp.zeros(n), theta, geom)
        for sl, e, _ in self.lag_exprs:
            lag = lag.at[sl].set(jnp.reshape(self._on(ev.expr(e), (), "a lag"), ()))
        for sl, e, _ in self.bioavailability_exprs:
            frac = frac.at[sl].set(jnp.reshape(self._on(ev.expr(e), (), "a bioavailability"), ()))
        return lag, frac

    def _obs_impl(self, t, y, theta, geom):
        ev, _ = self._evaluator(t, y, theta, geom, floor=False)
        out = {}
        for name, e in self.observable_exprs.items():
            v = jnp.asarray(ev.expr(e))
            if self.observable_where[name]:
                v = self._pick(v, self.observable_where[name])
            if v.ndim < self.plan.frame.ndim:
                v = jnp.reshape(v, (1,) * self.plan.frame.ndim) if v.ndim == 0 else v
            out[name] = v
        return out

    def observable_axes(self, name: str) -> tuple[str, ...]:
        """Axes an observable varies over (in the model's axis order)."""
        th = self._stand_in_theta()  # only the shape matters
        v = self.observables(0.0, self.initial_state(th), th)[name]
        return tuple(a for a, n in zip(self.plan.frame.axes, np.shape(v), strict=True) if n > 1)

    # ------------------------------------------------------------------ report
    def _stand_in_theta(self) -> np.ndarray:
        """``theta0`` with any value the model leaves out set to 1: enough to compile, time
        and size the model (never used for results)."""
        th = self.theta0.copy()
        gone = np.isnan(th[: self.params.size])
        th[: self.params.size][gone] = 1.0
        return th

    def warm_up(self) -> dict[str, float]:
        """Compile everything once and time it (seconds). Parameters without a value are
        set to 1 for this (timings do not depend on the values)."""
        th = self._stand_in_theta()
        timings = {}
        t0 = time.perf_counter()
        y = self.initial_state(th).block_until_ready()
        self.dose_amounts(th).block_until_ready()
        if self.has_arrival:
            self.arrival(th)
        timings["initial state"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.rhs(0.0, y, th).block_until_ready()
        timings["rhs compile"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        reps = 20
        for _ in range(reps):
            self.rhs(0.0, y, th).block_until_ready()
        timings["rhs call"] = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        self.sparse_structure  # noqa: B018 - builds the patterns and colourings
        timings["jacobian structure"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.sparse_jacobian(0.0, y, th)
        timings["jacobian compile"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        J = self.sparse_jacobian(0.0, y, th)
        timings["jacobian call"] = time.perf_counter() - t0
        timings["jacobian nonzeros"] = J.nnz
        self.compile_seconds = timings["rhs compile"] + timings["jacobian compile"]
        return timings

    def summary(self) -> str:
        """State count per field (and the axes driving it), parameters, compile and call times."""
        sizes = self.plan.frame.sizes
        lines = [f"Compiled model '{self.ir.name}': {self.n_states:,} state variables"]
        for f in self.ir.fields:
            prod_txt = " x ".join(f"{sizes[a]} ({a})" for a in f.axes)
            lines.append(
                f"  field {f.name}: {int(np.prod(self.layout.shapes[f.name])):,} = {prod_txt}"
            )
        if self.ir.compartments:
            names = ", ".join(c.name for c in self.ir.compartments)
            lines.append(f"  compartments: {len(self.ir.compartments)} ({names})")
        lines.append(f"  parameters: {self.params.size} numbers in theta")
        missing = self.params.missing(self.theta0)
        if missing:
            lines.append(
                f"  (no value in the model for {', '.join(missing)}: timed with 1 in their place)"
            )
        timings = self.warm_up()
        lines.append(
            f"  compile: right-hand side {timings['rhs compile']:.2f} s, "
            f"Jacobian {timings['jacobian compile']:.2f} s"
        )
        lines.append(f"  one right-hand-side call: {timings['rhs call'] * 1e3:.3f} ms")
        n = self.n_states
        S = self.sparse_structure
        lines.append(
            f"  Jacobian: sparse, {timings['jacobian nonzeros']:,} non-zeros of {n:,} x {n:,}; "
            f"{S['n_colours']} forward and {S['n_row_colours']} reverse passes"
            f" + {len(self.ir.compartments)} for the compartments"
            f" ({self.n_integrals:,} integral values lifted); "
            f"one evaluation {timings['jacobian call'] * 1e3:.1f} ms"
        )
        nnz = timings["jacobian nonzeros"]
        sparse_mb = (nnz * 12 + (n + 1) * 4) / 1e6  # CSC: value + row index, column pointers
        lu = _lu_size(self)
        lu_text = "" if lu is None else f"; its LU factors (the solver's I - hJ) {lu:.2g} MB"
        lines.append(
            f"  memory: state {n * 8 / 1e6:.2g} MB; Jacobian {sparse_mb:.2g} MB sparse "
            f"(a dense one would need {n * n * 8 / 1e6:,.3g} MB){lu_text}"
        )
        if n > 100_000:
            lines.append("  WARNING: above the 100,000-state target (ADR 0003).")
        return "\n".join(lines)


def _sorted_pattern(rows, cols, shape) -> dict[str, np.ndarray]:
    """Unique (row, col) pairs sorted row-major, with CSR row pointers."""
    rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
    key = np.unique(rows * shape[1] + cols)
    r, c = key // shape[1], key % shape[1]
    indptr = np.zeros(shape[0] + 1, dtype=np.int64)
    np.add.at(indptr, r + 1, 1)
    return {"rows": r, "cols": c, "indptr": np.cumsum(indptr)}


def _time(t):
    return jnp.asarray(t, dtype=jnp.float64)
