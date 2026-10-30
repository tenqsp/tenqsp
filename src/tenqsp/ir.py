# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Intermediate representation (IR): the single description of a model.

Both front ends, the Python API and the PDE-notation text, build a ``ModelIR``. The
compiler only ever reads a ``ModelIR``. Everything here is plain data:

- immutable dataclasses;
- expressions stored as formula text (ADR 0002 grammar);
- JSON round-trips exactly;
- a content hash (``ModelIR.content_hash``) identifies a model structure independent of
  Python object identity.

``validate`` checks a model and returns *every* problem it finds, each in plain language.

Conventions
-----------
- Transition matrices follow ``M[dest, origin]``. Only off-diagonal rates are given (as
  ``from -> to`` pairs); the diagonal is derived, so transitions conserve mass by construction.
- Integrals (over continuous axes, quadrature) and sums (over categorical axes) are declared
  as ``IntegralSpec`` and used by name in formulas, so the compiler can lift them out of
  the Jacobian (ADR 0003).
- Dose targets are required, and no field has a silent default target.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

IR_VERSION = "0.10"
# older versions this reader still accepts, with what changed since:
#   0.1 -> 0.2: ExtinctionSpec gained noise_floor (default False)
#   0.2 -> 0.3: ExtinctionSpec gained where (slices of fields; default: whole members)
#   0.3 -> 0.4: TransitionSpec.axis (and a switch/mutate TermSpec.axis) may be a tuple of
#               axes: joint transitions, whose categories are tuples of labels
#   0.4 -> 0.5: hierarchical compartments: ParamSpec.over may be a tuple of axes;
#               FieldSpec.weights, CompartmentSpec.volume; `where` (slices) on TermSpec,
#               IntegralSpec, ObservableSpec, DoseSpec; TransferSpec (all default to absent)
#   0.5 -> 0.6: IntegralSpec.where may slice any integrand whose fields have the sliced axes
#               (before: only an integrand that is one field); `over` may then be empty, the
#               value of a formula on a slice (the notation's C[organ=liver] in a formula)
#   0.6 -> 0.7: ParamSpec.value may be None: a parameter declared without a value, given
#               when the model is solved or simulated (decided 2026-10-24)
#   0.7 -> 0.8: lags and bioavailabilities of the doses into a place (LagSpec,
#               BioavailabilitySpec; decided 2026-10-25; both default to none)
#   0.8 -> 0.9: doses into a field over continuous axes (decided 2026-10-27): DoseSpec.where
#               may leave axes free; DoseSpec.shape (default none) spreads each amount over
#               them, as an initial value's total and shape; without a shape the amounts are
#               densities added to the field
#   0.9 -> 0.10: boundary conditions as written in papers (decided 2026-10-28): BoundarySpec
#               kind "flux" (the flux along the axis at that end equals ``value``, in which
#               the field's own name means its value at that end), and formulas may read a
#               field at one end of a continuous axis, ``at_end(F, z, lower)``
READABLE_VERSIONS = frozenset(
    {"0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.10"}
)

Where = tuple[tuple[str, str], ...]  # a slice: (axis, label) pairs fixing categories

AxisType = Literal["continuous", "categorical"]
TermKind = Literal[
    "advection", "diffusion", "reaction", "hazard", "rate", "switch", "mutate", "noise"
]
BoundaryKind = Literal["zero_flux", "inflow", "outflow", "dirichlet", "robin", "flux"]
TERM_KINDS: tuple[str, ...] = (
    "advection",
    "diffusion",
    "reaction",
    "hazard",
    "rate",
    "switch",
    "mutate",
    "noise",
)
BOUNDARY_KINDS: tuple[str, ...] = (
    "zero_flux",
    "inflow",
    "outflow",
    "dirichlet",
    "robin",
    "flux",
)
RESERVED_NAMES = frozenset({"t"})  # time


@dataclass(frozen=True)
class AxisSpec:
    """A continuous axis (``faces``, ``kind``) or a categorical axis (``labels``)."""

    name: str
    type: AxisType
    faces: tuple[float, ...] = ()
    kind: str = "bidirectional"
    units: str = ""
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParamSpec:
    """A parameter: a scalar, or one value per level of categorical axis ``over``; ``over`` may
    be a tuple of axes (one value per combination, flattened with the first axis slowest).
    ``value`` is None for a parameter declared without a value (a default): it must then be
    given when the model is solved or simulated."""

    name: str
    value: float | tuple[float, ...] | None
    over: str | tuple[str, ...] | None = None
    units: str = ""


@dataclass(frozen=True)
class FieldSpec:
    """A population density over one or more axes.

    ``weights`` (a formula of parameters over some of its categorical axes, e.g.
    ``V[organ, sub]``) gives each category a size, as cell widths do on continuous axes: the
    field then holds a concentration, and its amount is value x widths x weight. Transfers
    between categories of different size conserve amount."""

    name: str
    axes: tuple[str, ...]
    units: str = ""
    weights: str | None = None


@dataclass(frozen=True)
class CompartmentSpec:
    """A well-mixed scalar state (free phage, drug amount, nutrient). With ``volume`` (a
    formula of parameters) it holds a concentration and its amount is value x volume."""

    name: str
    units: str = ""
    volume: str | None = None


@dataclass(frozen=True)
class DefinitionSpec:
    """A named intermediate expression, reusable in other formulas."""

    name: str
    expr: str


@dataclass(frozen=True)
class IntegralSpec:
    """``name = integral of integrand over the axes in `over``` (quadrature on continuous axes,
    a sum on categorical axes). The result varies over the integrand's remaining axes."""

    name: str
    integrand: str
    over: tuple[str, ...]
    where: Where = ()


@dataclass(frozen=True)
class TransitionSpec:
    """Rates between categories of one axis, as ``(from, to, rate expression)`` triples.

    A joint transition changes several axes at once: ``axis`` is then a tuple of axis names
    and each category a tuple of labels, one per axis in that order.
    """

    name: str
    axis: str | tuple[str, ...]
    rates: tuple[tuple[Any, Any, str], ...]


@dataclass(frozen=True)
class TermSpec:
    """One term in the equation for ``target``.

    kind:

    - ``advection``: ``-d_axis(expr * target)`` (expr is the velocity); uses ``scheme``.
    - ``diffusion``: ``d_axis(expr * d_axis(target))`` (expr is the diffusivity).
    - ``reaction``: adds ``expr`` pointwise (sources minus sinks).
    - ``hazard``: removes ``expr * target`` (expr is a per-capita rate).
    - ``rate``: adds ``expr`` to a compartment's ``d/dt``.
    - ``switch``: first-order transitions of ``target`` along ``axis`` with ``transitions``
      (``axis`` is a tuple of axes for a joint transition, as in its ``TransitionSpec``).
    - ``mutate``: redistributes the production flux ``expr`` along ``axis`` with
      ``transitions`` (daughters change category; the flux itself is added by a reaction term).
      (``Model.transition`` makes a ``switch`` when ``of=`` is the field itself, a ``mutate``
      otherwise; ``trans_s(M, X)`` in the notation likewise.)
    - ``noise``: reserved for v2 (stochastic terms); rejected by the validator.
    """

    kind: TermKind
    target: str
    expr: str = "0"
    axis: str | tuple[str, ...] | None = None
    scheme: str = "erlang"
    transitions: str | None = None
    where: Where = ()  # reaction, hazard, switch, mutate: only on this slice of the target


@dataclass(frozen=True)
class BoundarySpec:
    """Condition at one end of a continuous axis of a field.

    - ``zero_flux``: nothing crosses.
    - ``inflow``: ``value`` is the flux entering (e.g. renewal ``k*B*P``).
    - ``outflow``: material carried out by advection, absorbed or routed to compartment ``to``.
    - ``dirichlet``: the field is held at ``value`` on the boundary.
    - ``robin``: exchange ``h * (outside - u)`` with ``value`` = outside level.
    - ``flux`` (IR 0.10): the flux along the axis at this end (positive toward larger
      coordinates, as in a paper's ``v*I(t, 0) = ...`` or ``-D*dC/dz = ...``) equals
      ``value``, a formula in which the field's own name means its value at this end; it
      must be linear in that value.
    """

    target: str
    axis: str
    side: Literal["lower", "upper"]
    kind: BoundaryKind
    value: str = "0"
    h: str = "0"
    to: str | None = None


@dataclass(frozen=True)
class InitSpec:
    """Initial condition of ``target`` (optionally restricted to categories ``where``).

    Give either ``value`` (a density formula) or ``total`` + ``shape`` (amount and shape; the
    shape is truncated to the grid and normalised so the discrete total equals ``total``).
    """

    target: str
    where: tuple[tuple[str, str], ...] = ()
    value: str | None = None
    total: str | None = None
    shape: str | None = None


@dataclass(frozen=True)
class DoseSpec:
    """Bolus doses at scheduled times into a compartment or one element of a field (``where``
    fixing every axis): ``amounts`` are amounts. Into a field with axes left free (IR 0.9),
    as an initial value: with ``shape``, each amount is spread over the free axes by that
    shape (cut to the grid, scaled so the cells receive exactly the amount); without, the
    amounts are densities (formulas of the free axes) added to the field."""

    target: str
    times: tuple[float, ...]
    amounts: tuple[str, ...]
    where: Where = ()
    shape: str | None = None


@dataclass(frozen=True)
class LagSpec:
    """Every dose into ``target`` (a compartment, or every element of a field) arrives
    ``value`` later (a formula of parameters, >= 0): a bolus at its time + lag, an infusion
    shifted whole. Applies to the model's doses and to a regimen's alike."""

    target: str
    value: str


@dataclass(frozen=True)
class BioavailabilitySpec:
    """Only the fraction ``value`` (a formula of parameters, >= 0) of every dose into
    ``target`` arrives: bolus amounts and infusion rates are scaled by it."""

    target: str
    value: str


@dataclass(frozen=True)
class ObservableSpec:
    name: str
    expr: str
    where: Where = ()  # expr names a field: only this slice of it


@dataclass(frozen=True)
class TransferSpec:
    """Material moved from ``source`` to ``to`` at ``rate`` x the source's value (an amount
    per time; with weights or volumes the rate is a clearance or flow). Either end is a
    compartment or a slice of a field (``source_where``, ``to_where``); ``to=None`` removes
    it (elimination). Element by element over the axes the slices leave free."""

    source: str
    rate: str
    source_where: Where = ()
    to: str | None = None
    to_where: Where = ()


@dataclass(frozen=True)
class ExtinctionSpec:
    """A lineage group zeroed when its total falls below ``threshold``.

    ``where`` (one entry per member, or empty for whole members) fixes categories of a field
    member, ``(("s", "R"),)`` meaning only the part of the field with ``s = R`` (a slice)."""

    name: str
    group: tuple[str, ...]
    threshold: float
    noise_floor: bool = False
    where: tuple[tuple[tuple[str, str], ...], ...] = ()

    def member_where(self, i: int) -> tuple[tuple[str, str], ...]:
        return self.where[i] if self.where else ()


def _explicit_outflows(kwargs: dict[str, Any]) -> tuple[BoundarySpec, ...]:
    """Before IR 0.10 the upper end of a one_way axis along which a field flows was an
    outflow when nothing was said; since then it must be stated. Older files get it
    written in, so they behave as before."""
    one_way = {a.name for a in kwargs["axes"] if a.kind == "one_way"}
    given = {(b.target, b.axis, b.side) for b in kwargs["boundaries"]}
    added = []
    for t in kwargs["terms"]:
        key = (t.target, t.axis, "upper")
        if t.kind == "advection" and t.axis in one_way and key not in given:
            given.add(key)
            added.append(BoundarySpec(t.target, t.axis, "upper", "outflow"))
    return (*kwargs["boundaries"], *added)


_LISTS: dict[str, type] = {
    "axes": AxisSpec,
    "params": ParamSpec,
    "fields": FieldSpec,
    "compartments": CompartmentSpec,
    "definitions": DefinitionSpec,
    "integrals": IntegralSpec,
    "transitions": TransitionSpec,
    "terms": TermSpec,
    "boundaries": BoundarySpec,
    "inits": InitSpec,
    "doses": DoseSpec,
    "lags": LagSpec,
    "bioavailabilities": BioavailabilitySpec,
    "observables": ObservableSpec,
    "extinctions": ExtinctionSpec,
    "transfers": TransferSpec,
}


@dataclass(frozen=True)
class ModelIR:
    """A complete model description."""

    name: str
    axes: tuple[AxisSpec, ...] = ()
    params: tuple[ParamSpec, ...] = ()
    fields: tuple[FieldSpec, ...] = ()
    compartments: tuple[CompartmentSpec, ...] = ()
    definitions: tuple[DefinitionSpec, ...] = ()
    integrals: tuple[IntegralSpec, ...] = ()
    transitions: tuple[TransitionSpec, ...] = ()
    terms: tuple[TermSpec, ...] = ()
    boundaries: tuple[BoundarySpec, ...] = ()
    inits: tuple[InitSpec, ...] = ()
    doses: tuple[DoseSpec, ...] = ()
    lags: tuple[LagSpec, ...] = ()
    bioavailabilities: tuple[BioavailabilitySpec, ...] = ()
    observables: tuple[ObservableSpec, ...] = ()
    extinctions: tuple[ExtinctionSpec, ...] = ()
    transfers: tuple[TransferSpec, ...] = ()
    ir_version: str = IR_VERSION
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"ir_version": self.ir_version, "name": self.name}
        for key in _LISTS:
            d[key] = [_to_plain(dataclasses.asdict(x)) for x in getattr(self, key)]
        if self.meta:
            d["meta"] = self.meta
        return d

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelIR:
        version = d.get("ir_version")
        if version not in READABLE_VERSIONS:
            raise ValueError(
                f"This model was saved with IR version {version!r}; this TenQSP reads versions "
                f"{sorted(READABLE_VERSIONS)}."
            )
        # older files load as the current version (every later field has a default)
        unknown = set(d) - {"ir_version", "name", "meta", *_LISTS}
        if unknown:
            raise ValueError(f"Unknown keys in model file: {sorted(unknown)}.")
        kwargs: dict[str, Any] = {"name": d["name"], "meta": d.get("meta", {})}
        for key, spec_cls in _LISTS.items():
            kwargs[key] = tuple(_from_plain(spec_cls, item) for item in d.get(key, []))
        if tuple(int(x) for x in version.split(".")) < (0, 10):
            kwargs["boundaries"] = _explicit_outflows(kwargs)
        return cls(**kwargs)

    @classmethod
    def from_json(cls, text: str) -> ModelIR:
        return cls.from_dict(json.loads(text))

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON (structure + values, not ``meta``)."""
        d = self.to_dict()
        d.pop("meta", None)
        # numbers that compare equal hash equally (50 and 50.0), as the dataclasses compare
        return hashlib.sha256(json.dumps(_floats(d), sort_keys=True).encode()).hexdigest()

    def structure_hash(self) -> str:
        """SHA-256 of the structure only: as ``content_hash``, without parameter values and
        dose times (the number of doses stays: it is structural). Two models with the same
        structure share one build."""
        d = self.to_dict()
        d.pop("meta", None)
        for p in d["params"]:
            p["value"] = None
        for dose in d["doses"]:
            dose["times"] = len(dose["times"])
        return hashlib.sha256(json.dumps(_floats(d), sort_keys=True).encode()).hexdigest()

    # ------------------------------------------------------------------ lookups
    def axis(self, name: str) -> AxisSpec:
        return _find(self.axes, name, "axis")

    def field_spec(self, name: str) -> FieldSpec:
        return _find(self.fields, name, "field")

    def state_count(self) -> dict[str, int]:
        """Number of state variables per field and compartment."""
        sizes = {
            a.name: (len(a.faces) - 1 if a.type == "continuous" else len(a.labels))
            for a in self.axes
        }
        out = {}
        for f in self.fields:
            n = 1
            for ax in f.axes:
                n *= sizes.get(ax, 0)
            out[f.name] = n
        for c in self.compartments:
            out[c.name] = 1
        return out


def _find(items, name, what):
    for x in items:
        if x.name == name:
            return x
    raise KeyError(f"No {what} named '{name}'.")


def _floats(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return float(x)
    if isinstance(x, list):
        return [_floats(v) for v in x]
    if isinstance(x, dict):
        return {k: _floats(v) for k, v in x.items()}
    return x


def _to_plain(x):
    if isinstance(x, tuple):
        return [_to_plain(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}
    return x


def _from_plain(spec_cls, item: dict[str, Any]):
    names = {f.name for f in dataclasses.fields(spec_cls)}
    unknown = set(item) - names
    if unknown:
        raise ValueError(f"Unknown keys for {spec_cls.__name__}: {sorted(unknown)}.")
    kw = {}
    for k, v in item.items():
        kw[k] = _to_tuple(v)
    return spec_cls(**kw)


def _to_tuple(v):
    if isinstance(v, list):
        return tuple(_to_tuple(x) for x in v)
    return v
