# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Simulation: one model built once, run with values, regimens and times given per run
(design decisions 2026-10-24).

- ``Scenario``: parameter values + regimen + output times (+ start), named, savable.
- ``Simulator`` (``m.simulator(outputs=[...])``): called with a scenario and/or
  ``params=``, ``regimen=``, ``times=``; returns a ``SimResult``.

Values are layered: the model's own values (defaults), then the scenario's ``params``, then
``params=`` in the call; each layer replaces only the names it gives. A regimen given (in the
scenario or the call) **replaces** the doses written in the model; not giving one keeps the
model's doses; ``Regimen()`` means no doses. Changing any value never rebuilds the model.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from tenqsp.regimen import Regimen


class _ModelDoses:
    """Sentinel: no regimen given, so the model's own doses are used."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<the model's doses>"

    def __bool__(self) -> bool:
        return False


MODEL_DOSES = _ModelDoses()


@dataclasses.dataclass(frozen=True)
class Scenario:
    """What to run: ``params`` (values by name, as ``m.solve(params=)``), ``regimen`` (a
    ``Regimen``; left out: the model's own doses; ``Regimen()``: none),
    ``times`` (output times), ``start`` (default 0) and ``initial`` (the starting state: a
    ``PreRun``, or values by state name; left out: the model's own
    initial values), under a ``name``."""

    name: str = ""
    params: Mapping = dataclasses.field(default_factory=dict)
    regimen: Regimen | _ModelDoses = MODEL_DOSES
    times: tuple[float, ...] | None = None
    start: float | None = None
    initial: Mapping | None = None

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError(f"Scenario name must be text, not {self.name!r}.")
        if not isinstance(self.params, Mapping) or not all(isinstance(k, str) for k in self.params):
            raise TypeError("Scenario params must be a mapping of parameter names to values.")
        object.__setattr__(self, "params", {k: _plain(v) for k, v in self.params.items()})
        if not isinstance(self.regimen, (Regimen, _ModelDoses)):
            raise TypeError(
                f"Scenario regimen must be a tq.Regimen (e.g. tq.bolus(...)), not {self.regimen!r}."
            )
        if self.times is not None:
            object.__setattr__(self, "times", _times(self.times))
        if self.start is not None:
            object.__setattr__(self, "start", _finite(self.start, "start"))
        if self.initial is not None:
            initial = getattr(self.initial, "state", self.initial)  # a PreRun
            if not isinstance(initial, Mapping) or not all(isinstance(k, str) for k in initial):
                raise TypeError(
                    "Scenario initial takes a pre-run (sim.pre_run(...)) or a mapping of state "
                    f"names to values, not {self.initial!r}."
                )
            object.__setattr__(self, "initial", {k: _plain(v) for k, v in initial.items()})

    def replace(self, **changes) -> Scenario:
        """A copy with changes. ``params`` given here are merged into this scenario's (the
        names given change, the others stay; for a per-category parameter, the labels given
        change, the others stay); every other field is replaced."""
        if "params" in changes:
            merged = dict(self.params)
            for name, value in dict(changes["params"]).items():
                old = merged.get(name)
                if isinstance(old, Mapping) and isinstance(value, Mapping):
                    value = {**old, **value}
                merged[name] = value
            changes["params"] = merged
        return dataclasses.replace(self, **changes)

    def initial_vector(self, cm):
        """The starting state as one number per state of the compiled model, NaN where the
        model's own initial value stays; None if no ``initial`` is given."""
        if self.initial is None:
            return None
        layout = cm.layout
        y = np.full(cm.n_states, np.nan)
        for name, value in self.initial.items():
            if name not in layout.names:
                raise KeyError(
                    f"initial: '{name}' is not a field or compartment of the model; they are "
                    f"{list(layout.names)}."
                )
            v = np.asarray(value, dtype=np.float64)
            shape = tuple(layout.shapes[name])
            try:
                v = np.broadcast_to(v, shape)
            except ValueError:
                raise ValueError(
                    f"initial: '{name}' has shape {v.shape}; the model's is {shape}."
                ) from None
            y[layout.slices[name]] = v.reshape(-1)
        return y

    # ------------------------------------------------------------------ JSON
    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "params": self.params}
        if isinstance(self.regimen, Regimen):  # absent: the model's doses
            d["regimen"] = self.regimen.to_list()
        if self.times is not None:
            d["times"] = list(self.times)
        if self.start is not None:
            d["start"] = self.start
        if self.initial is not None:
            d["initial"] = self.initial
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> Scenario:
        unknown = set(d) - {"name", "params", "regimen", "times", "start", "initial"}
        if unknown:
            raise ValueError(f"Unknown keys in a scenario: {sorted(unknown)}.")
        regimen = Regimen.from_list(d["regimen"]) if "regimen" in d else MODEL_DOSES
        return cls(
            name=d.get("name", ""),
            params=d.get("params", {}),
            regimen=regimen,
            times=d.get("times"),
            start=d.get("start"),
            initial=d.get("initial"),
        )

    def save(self, path) -> None:
        """Write the scenario as JSON (readable, and diffable under version control)."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path) -> Scenario:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class SimResult:
    """The outputs of one run: ``r["conc"]`` (shape ``(n_t, ...)``), ``r.t``, ``r.scenario``
    (what was run, with every layer of values merged), ``r.to_xarray()``, and the full
    ``Solution`` as ``r.solution`` (all states, ledger, messages)."""

    def __init__(self, solution, outputs: tuple[str, ...], scenario: Scenario):
        self.solution = solution
        self.outputs = outputs
        self.scenario = scenario

    @property
    def t(self) -> np.ndarray:
        return self.solution.t

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self.outputs:
            raise KeyError(
                f"'{name}' is not an output of this simulator; its outputs are "
                f"{list(self.outputs)} (everything else is in r.solution)."
            )
        return self.solution[name]

    def __contains__(self, name: str) -> bool:
        return name in self.outputs

    def to_xarray(self):
        ds = self.solution.to_xarray()[list(self.outputs)]
        ds.attrs["scenario"] = self.scenario.name
        return ds

    def __repr__(self) -> str:
        name = f" '{self.scenario.name}'" if self.scenario.name else ""
        return (
            f"<SimResult{name}: {list(self.outputs)} at {len(self.t)} times "
            f"from {self.t[0]:g} to {self.t[-1]:g}>"
        )


class Simulator:
    """A model ready to run many times: ``sim = m.simulator(outputs=["conc"])``, then
    ``sim(scenario)`` or ``sim(params=..., regimen=..., times=...)``. Solver settings
    (``backend``, ``method``, ``rtol``, ``atol``, ``max_step``, ``ledger``, ``max_steps``) are
    fixed here, so every run is comparable. Changing values never
    rebuilds the model."""

    def __init__(self, model, outputs: Sequence[str] | None = None, **options):
        allowed = {"method", "rtol", "atol", "max_step", "ledger", "backend", "max_steps"}
        unknown = set(options) - allowed
        if unknown:
            raise TypeError(
                f"Unknown simulator options {sorted(unknown)}; they are {sorted(allowed)}. "
                "Values, regimens and times are given per run."
            )
        self.model = model
        self.options = options
        ir = model.to_ir()
        observables = [o.name for o in ir.observables]
        places = [f.name for f in ir.fields] + [c.name for c in ir.compartments]
        if outputs is None:
            outputs = observables or places
        elif isinstance(outputs, str):
            outputs = [outputs]
        outputs = tuple(outputs)
        known = set(observables) | set(places)
        for o in outputs:
            if o not in known:
                import difflib

                hint = difflib.get_close_matches(o, sorted(known), n=1)
                extra = f" Did you mean '{hint[0]}'?" if hint else ""
                raise KeyError(
                    f"'{o}' is not an observable, field or compartment of the model.{extra} "
                    f"Observables: {observables}; fields and compartments: {places}."
                )
        if len(set(outputs)) != len(outputs):
            raise ValueError(f"An output is listed twice: {list(outputs)}.")
        self.outputs = outputs

    def __call__(
        self,
        scenario: Scenario | None = None,
        *,
        params: Mapping | None = None,
        regimen: Regimen | _ModelDoses = MODEL_DOSES,
        times=None,
        start=None,
    ) -> SimResult:
        sc = scenario if scenario is not None else Scenario()
        if not isinstance(sc, Scenario):
            raise TypeError(f"The first argument is a tq.Scenario, not {sc!r}.")
        given = (params, times, start)
        if any(x is not None for x in given) or regimen is not MODEL_DOSES:
            changes: dict = {}
            if params is not None:
                changes["params"] = params
            if isinstance(regimen, Regimen):
                changes["regimen"] = regimen
            elif regimen is not MODEL_DOSES:
                raise TypeError(f"regimen= takes a tq.Regimen, not {regimen!r}.")
            if times is not None:
                changes["times"] = times
            if start is not None:
                changes["start"] = start
            sc = sc.replace(**changes)
        if sc.times is None:
            raise ValueError("Give the output times: times=[...] (in the call or the scenario).")
        t0 = 0.0 if sc.start is None else sc.start
        if sc.times[0] < t0:
            raise ValueError(
                f"The output times start at {sc.times[0]:g}, before the start t = {t0:g}."
            )
        t1 = sc.times[-1]
        if not t1 > t0:
            raise ValueError(f"The last output time ({t1:g}) must be after the start ({t0:g}).")
        sol = self.model.solve(
            (t0, t1),
            times=np.asarray(sc.times),
            params=sc.params or None,
            regimen=sc.regimen if isinstance(sc.regimen, Regimen) else None,
            **({"initial": sc.initial_vector(self.model.compile())} if sc.initial else {}),
            **self.options,
        )
        return SimResult(sol, self.outputs, sc)

    def function(self, scenario: Scenario | None = None, *, wrt, derivatives="auto"):
        """The simulation as a JAX function of the numbers named in ``wrt``, for estimation
        and any JAX code: ``f = sim.function(scenario, wrt=["CL", "V"])``; ``f.names``,
        ``f.x0`` (their values in the scenario), ``f(x)`` -> ``{output: array}``, with ``x`` a
        1-D array in ``f.names`` order; ``jax.grad``, ``jax.jit`` and ``jax.vmap`` work on
        it. A failed run gives NaN (``f.status(x)`` says why). Names: a parameter (``"CL"``,
        ``"k"`` for every entry, ``"k[R]"``), or a number of the scenario's regimen, from 1
        in time order: ``"dose 1 amount"``, ``"dose 1 time"``, ``"infusion 1 rate"`` (or
        ``amount``, ``duration``, ``start``). Always uses the JAX solver; see
        ``gradients``. ``derivatives=``: ``"auto"`` (the default: ``"forward"`` up to 40
        names in ``wrt``, ``"reverse"`` above), ``"forward"`` (sensitivities carried along
        the run; cost grows with the number of names) or ``"reverse"`` (the run recorded,
        then replayed backwards; a gradient costs the same whatever the number of names);
        ``f.derivatives`` says which. ``derivatives=False``: for simulating (a population with
        ``jax.vmap``, a sweep) without differentiating: no record of the runs, so much less
        memory and 2-3x faster; differentiating it raises."""
        from tenqsp.gradients import function

        return function(self, scenario, wrt=wrt, derivatives=derivatives)

    def sensitivity(self, scenario: Scenario | None = None, *, wrt, relative: bool = False):
        """d output / d each name in ``wrt`` at every output time, at the scenario's values:
        ``s = sim.sensitivity(scenario, wrt=["CL", "V"])``, then ``s["conc"]["CL"]``,
        ``s.values["conc"]``, ``s.table()``. ``relative=True``: d log y / d log theta (NaN
        where y is 0). A failing run raises SolverError. Names as ``function``."""
        from tenqsp.gradients import sensitivity

        return sensitivity(self, scenario, wrt=wrt, relative=relative)

    def batch(self, scenarios, *, times=None, n_jobs: int = 1):
        """Many scenarios in one call: ``b = sim.batch(tq.grid(params={"CL": [3, 5, 8]},
        regimens={...}), times=...)`` or ``sim.batch([sc1, sc2, ...])``. Returns a
        ``BatchResult``: ``b["conc"]`` (scenario x time x ...),
        ``b.table()``, ``b.to_xarray()``, ``b.failed``. Every scenario needs the same number of
        output times (``times=`` here sets them for all). A scenario that fails gives NaN and
        is listed in ``b.failed``; the others still run. ``n_jobs``: that many processes. On
        ``backend="jax"`` the scenarios of one shape run as one ``jax.vmap`` call. See
        ``explore``."""
        from tenqsp.explore import batch

        return batch(self, scenarios, times=times, n_jobs=n_jobs)

    def pre_run(self, scenario: Scenario | None = None, *, until="steady", t_max: float = 1e6):
        """Run ahead and keep the end state, to start other scenarios from it:
        ``pre = sim.pre_run(sc, until="steady")`` (or ``until=336``, a fixed time), then
        ``sim(sc2.replace(initial=pre))``. ``"steady"``: with a regimen that repeats at even
        intervals, until two intervals in a row end in the same state (the end is just
        before the next dose); otherwise until every rate of change is within the
        tolerances, after the doses. Gives up at ``t_max`` with a warning (``pre.reached``
        False). The end state is a fixed starting point: derivatives (``sim.function``) do
        not flow back through it. See ``explore``."""
        from tenqsp.explore import pre_run

        return pre_run(self, scenario, until=until, t_max=t_max)

    def __repr__(self) -> str:
        return f"<Simulator of '{self.model.name}': outputs {list(self.outputs)}>"


def _times(x) -> tuple[float, ...]:
    ts = tuple(_finite(v, "output time") for v in np.ravel(np.asarray(x, dtype=object)))
    if not ts:
        raise ValueError("times is empty.")
    if any(b < a for a, b in zip(ts, ts[1:], strict=False)):
        raise ValueError("times must be in increasing order.")
    return ts


def _finite(x, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"The {what} must be a number, not {x!r}.") from None
    if not math.isfinite(v):
        raise ValueError(f"The {what} must be finite, not {v}.")
    return v


def _plain(v):
    """JSON-friendly copies of parameter values (numbers, lists, label dicts)."""
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v
