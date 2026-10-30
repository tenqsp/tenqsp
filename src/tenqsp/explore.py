# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Exploring a model: many scenarios in one call, and pre-runs (step 4.8; design
note ``docs/design/step4_8_explore.md``, design decisions 2026-10-27).

- ``grid``: the cross product of parameter values (and regimens) as a list of scenarios.
- ``batch`` -> ``BatchResult``: every scenario of a list, one
  labelled result (``b["conc"]``: scenario x time x ...; ``b.table()``; ``b.to_xarray()``).
  A failed scenario gives NaN and is listed in ``b.failed``; the others still run.
  ``n_jobs``: that many processes. On ``backend="jax"`` the scenarios of one shape run as one
  ``jax.vmap`` call (no derivatives, step 4.6).
- ``pre_run`` -> ``PreRun``: run for a fixed time or until steady,
  and start a scenario from the end state (``sc.replace(initial=pre)``). The pre-run's end
  is a fixed starting point: derivatives do not flow back through it (design decision 4).
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import math
import os
import warnings
from collections.abc import Mapping, Sequence

import numpy as np

from tenqsp.regimen import Bolus, Infusion, Regimen

# --------------------------------------------------------------------------------- grids


class Grid(list):
    """A list of scenarios made by ``grid``, which also knows what was varied:
    ``columns`` (the parameter names, then ``"regimen"`` if regimens were given) and, per
    scenario, ``labels[i]`` (its value of each column)."""

    def __init__(self, scenarios, columns, labels):
        super().__init__(scenarios)
        self.columns = tuple(columns)
        self.labels = list(labels)


def grid(params: Mapping | None = None, regimens=None, *, base=None) -> Grid:
    """Every combination of the values given, as scenarios: ``tq.grid(params={"CL": [3, 5],
    "V": [40, 60]}, regimens={"100 q12h": r1, "200 q24h": r2})`` makes 2 x 2 x 2 = 8, the
    last name varying fastest (regimens last). ``params``: a list of values per name (a
    per-category value is a dict, e.g. ``{"k": [{"S": 1}, {"S": 2}]}``); ``regimens``: a
    dict of named regimens, or a list (named "regimen 1", ...). ``base``: a scenario whose
    other settings (times, start, other values) every scenario starts from."""
    from tenqsp.simulate import Scenario

    params = dict(params or {})
    for name, values in params.items():
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"grid params take a list of values per name; '{name}' has {values!r}.")
        if len(values) == 0:
            raise ValueError(f"grid params: '{name}' has no values.")
    if regimens is not None and not isinstance(regimens, Mapping):
        regimens = {f"regimen {i + 1}": r for i, r in enumerate(regimens)}
    for label, r in (regimens or {}).items():
        if not isinstance(r, Regimen):
            raise TypeError(f"grid regimens take tq.Regimen values; '{label}' is {r!r}.")
    base = base if base is not None else Scenario()
    if not isinstance(base, Scenario):
        raise TypeError(f"base= takes a tq.Scenario, not {base!r}.")
    names = list(params)
    axes = [list(params[n]) for n in names]
    columns = list(names)
    if regimens:
        axes.append(list(regimens))
        columns.append("regimen")
    scenarios, labels = [], []
    for combo in itertools.product(*axes):
        values = dict(zip(columns, combo, strict=True))
        changes: dict = {}
        if names:
            changes["params"] = {n: values[n] for n in names}
        if regimens:
            changes["regimen"] = regimens[values["regimen"]]
        name = ", ".join(
            str(v) if c == "regimen" else f"{c}={_short(v)}" for c, v in values.items()
        )
        changes["name"] = name if not base.name else f"{base.name}: {name}"
        scenarios.append(base.replace(**changes))
        labels.append(values)
    return Grid(scenarios, columns, labels)


def _short(v) -> str:
    if isinstance(v, Mapping):
        return "{" + ", ".join(f"{k}: {_short(x)}" for k, x in v.items()) + "}"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


# --------------------------------------------------------------------------------- batches


@dataclasses.dataclass
class BatchResult:
    """The outputs of many scenarios: ``b["conc"]`` (scenario x time x ...), ``b.t``,
    ``b.scenarios``, ``b.failed`` (``(index, name, reason)`` of each scenario that failed;
    its outputs are NaN), ``b.table()`` (pandas, long), ``b.to_xarray()``. ``b[i]`` runs
    scenario ``i`` again and returns its full ``SimResult`` (the same
    numbers; a batch keeps only the outputs, to keep its memory small)."""

    simulator: object
    scenarios: list
    t: np.ndarray
    values: dict[str, np.ndarray]
    axes: dict[str, tuple[tuple[str, tuple], ...]]  # per output: (axis, coordinates)
    failed: list[tuple[int, str, str]]
    columns: tuple[str, ...]
    labels: list[dict]

    @property
    def outputs(self) -> tuple[str, ...]:
        return tuple(self.values)

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self.values:
                raise KeyError(
                    f"'{key}' is not an output of this batch; they are {list(self.values)}."
                )
            return self.values[key]
        return self.simulator(self.scenarios[key])

    def table(self):
        """One row per scenario x time x output (x field cell): columns ``scenario``, the
        varied values (``self.columns``), ``time``, the output's axes, ``output``, ``value``.
        Needs pandas."""
        import pandas as pd

        frames = []
        for out, arr in self.values.items():
            axes = self.axes[out]
            idx = np.indices(arr.shape).reshape(arr.ndim, -1)
            data = {
                "scenario": np.array([s.name for s in self.scenarios], dtype=object)[idx[0]],
            }
            for c in self.columns:
                col = np.array([_cell(lab.get(c)) for lab in self.labels], dtype=object)
                data[c] = col[idx[0]]
            data["time"] = self.t[idx[1]]
            for k, (axis, coords) in enumerate(axes):
                data[axis] = np.asarray(coords, dtype=object)[idx[2 + k]]
            data["output"] = out
            data["value"] = arr.reshape(-1)
            frames.append(pd.DataFrame(data))
        table = pd.concat(frames, ignore_index=True)
        for c in self.columns:  # numbers stay numbers where they all are
            try:
                table[c] = pd.to_numeric(table[c])
            except (TypeError, ValueError):
                pass
        return table

    def to_xarray(self):
        """An ``xarray.Dataset``: each output with dims (scenario, time, its axes); the
        varied values as coordinates along ``scenario``. Needs xarray."""
        import xarray as xr

        coords = {
            "scenario": [s.name for s in self.scenarios],
            "time": self.t,
        }
        for c in self.columns:
            coords[c] = ("scenario", [_cell(lab.get(c)) for lab in self.labels])
        data = {}
        for out, arr in self.values.items():
            dims = ["scenario", "time"]
            for axis, values in self.axes[out]:
                dims.append(axis)
                coords.setdefault(axis, list(values))
            data[out] = (dims, arr)
        return xr.Dataset(data, coords=coords)

    def __repr__(self) -> str:
        bad = f", {len(self.failed)} failed" if self.failed else ""
        return (
            f"<BatchResult: {len(self.scenarios)} scenarios{bad}, {list(self.values)} at "
            f"{len(self.t)} times>"
        )


def _cell(v):
    if isinstance(v, (Mapping, list, tuple)):
        return _short(v)
    return v


def batch(sim, scenarios, *, times=None, n_jobs: int = 1) -> BatchResult:
    """See ``batch``."""
    from tenqsp.simulate import Scenario

    if isinstance(scenarios, Scenario):
        scenarios = [scenarios]
    given = scenarios
    scenarios = list(scenarios)
    if not scenarios:
        raise ValueError("batch() needs at least one scenario.")
    for s in scenarios:
        if not isinstance(s, Scenario):
            raise TypeError(f"batch() takes tq.Scenario objects (or a tq.grid), not {s!r}.")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError(f"n_jobs must be a whole number >= 1, not {n_jobs!r}.")
    if isinstance(given, Grid):  # what was varied, before the times are set
        columns, labels = given.columns, given.labels
    else:
        columns, labels = _varied(scenarios)
    if times is not None:
        scenarios = [s.replace(times=times) for s in scenarios]
    for s in scenarios:
        if s.times is None:
            raise ValueError("Give the output times: times=[...] in batch() or in each scenario.")
    first = np.asarray(scenarios[0].times)
    for s in scenarios:
        if len(s.times) != len(first):
            raise ValueError(
                f"Every scenario of a batch needs the same number of output times; "
                f"'{s.name}' has {len(s.times)}, '{scenarios[0].name}' {len(first)}. Run "
                "them one by one with sim(...)."
            )
    same_times = all(np.array_equal(np.asarray(s.times), first) for s in scenarios)
    n = len(scenarios)
    if n_jobs == 1 or n == 1:
        results = _run_chunk(sim, scenarios, list(range(n)))
    else:
        results = _run_parallel(sim, scenarios, n_jobs)
    axes = next((r["axes"] for r in results if r["axes"] is not None), None)
    if axes is None:  # every scenario failed: the shapes from the model
        axes = _output_axes(sim)
    values = {}
    for out in sim.outputs:
        shape = (len(first), *(len(c) for _, c in axes[out]))
        arr = np.full((n, *shape), np.nan)
        for r in results:
            if r["values"] is not None:
                arr[r["index"]] = np.reshape(r["values"][out], shape)
        values[out] = arr
    failed = [(r["index"], scenarios[r["index"]].name, r["error"]) for r in results if r["error"]]
    if failed:
        warnings.warn(
            f"{len(failed)} of {n} scenarios failed (their outputs are NaN; see b.failed). "
            f"First: '{failed[0][1]}': {failed[0][2]}",
            stacklevel=3,
        )
    t = first if same_times else np.full(len(first), np.nan)
    return BatchResult(sim, scenarios, t, values, axes, failed, columns, labels)


def _varied(scenarios) -> tuple[tuple[str, ...], list[dict]]:
    """For a plain list: the parameters whose values differ between scenarios."""
    flat = []
    for s in scenarios:
        row = {}
        for name, v in s.params.items():
            if isinstance(v, Mapping):
                for label, x in v.items():
                    row[f"{name}[{label}]"] = x
            else:
                row[name] = v
        flat.append(row)
    names = []
    for row in flat:
        for k in row:
            if k not in names:
                names.append(k)
    columns = [k for k in names if len({repr(r.get(k)) for r in flat}) > 1]
    return tuple(columns), [{c: r.get(c) for c in columns} for r in flat]


def _output_axes(sim) -> dict:
    """When every scenario failed: no run to read the outputs' axes from, so each output is
    kept as scenario x time (all NaN)."""
    return {name: () for name in sim.outputs}


def _axes_of(result) -> dict:
    """Per output: ((axis, coordinates), ...) from one run's solution."""
    sol = result.solution
    out = {}
    for name in result.outputs:
        axes = sol.observable_axes.get(name, sol.axes.get(name, ()))
        out[name] = tuple((a, tuple(np.asarray(sol.coords[a]["values"]).tolist())) for a in axes)
    return out


def _run_chunk(sim, scenarios, indices) -> list[dict]:
    """Run the scenarios ``indices``: one by one on scipy; on JAX, those of one shape as one
    ``jax.vmap`` call."""
    if sim.options.get("backend", "scipy") == "jax":
        return _run_jax(sim, scenarios, indices)
    return [_run_one(sim, scenarios[i], i) for i in indices]


def _run_one(sim, scenario, i) -> dict:
    from tenqsp.backends.scipy_backend import SolverError

    try:
        r = sim(scenario)
    except (SolverError, ValueError, FloatingPointError) as e:
        return {"index": i, "values": None, "axes": None, "error": str(e)}
    values = {o: np.asarray(r[o]) for o in sim.outputs}
    return {"index": i, "values": values, "axes": _axes_of(r), "error": ""}


# ---------------------------------------------------------------------- the JAX backend


def _shape_key(scenario) -> tuple:
    """Scenarios with the same key run as one ``jax.vmap`` call: the same regimen shape
    (targets, and how a dose into a field is spread), output times, start, starting state
    and per-category labels given. Everything else that varies (values, dose amounts and
    times) is an input of that one call."""
    reg = scenario.regimen
    if isinstance(reg, Regimen):
        shape = (
            "regimen",
            tuple((b.target, b.where, repr(b.shape)) for b in reg.boluses),
            tuple((f.target, f.where, repr(f.shape)) for f in reg.infusions),
        )
    else:
        shape = ("model doses",)
    params = tuple(
        sorted(
            (k, tuple(sorted(v))) if isinstance(v, Mapping) else (k, None)
            for k, v in scenario.params.items()
        )
    )
    initial = None if scenario.initial is None else repr(sorted(scenario.initial.items()))
    return (shape, scenario.times, scenario.start, params, initial)


def _wrt_and_x(scenario) -> tuple[list[str], list[float]]:
    names, x = [], []
    for k, v in scenario.params.items():
        if isinstance(v, Mapping):
            for label, val in v.items():
                names.append(f"{k}[{label}]")
                x.append(float(val))
        elif np.ndim(v) == 0:
            names.append(k)
            x.append(float(v))
        else:
            raise ValueError(
                f"'{k}' has a list of values; on the JAX backend give per-category values "
                "by label (a dict)."
            )
    reg = scenario.regimen
    if isinstance(reg, Regimen):
        for i, b in enumerate(sorted(reg.boluses, key=lambda b: b.time), start=1):
            names += [f"dose {i} amount", f"dose {i} time"]
            x += [b.amount, b.time]
        for i, f in enumerate(sorted(reg.infusions, key=lambda f: f.start), start=1):
            names += [f"infusion {i} rate", f"infusion {i} duration", f"infusion {i} start"]
            x += [f.rate, f.duration, f.start]
    return names, x


def _run_jax(sim, scenarios, indices) -> list[dict]:
    import jax
    import jax.numpy as jnp

    from tenqsp.backends.scipy_backend import SolverError

    groups: dict[tuple, list[int]] = {}
    for i in indices:
        groups.setdefault(_shape_key(scenarios[i]), []).append(i)
    results = []
    for members in groups.values():
        template = scenarios[members[0]]
        try:
            names, _ = _wrt_and_x(template)
            if not names:
                raise ValueError("nothing varies")
            f = sim.function(template, wrt=names, derivatives=False)
            X = np.array([_wrt_and_x(scenarios[i])[1] for i in members])
            batch_f = jax.jit(jax.vmap(lambda x, f=f: f._impl(x, f._data)))  # outputs, status
            outs, status = batch_f(jnp.asarray(X))
            outs = {o: v[:, : f._n_t] for o, v in outs.items()}
        except (ValueError, KeyError, TypeError, SolverError):
            # not one vmap call (nothing varies, a list-valued parameter, a template that
            # does not run): one by one, each reporting its own failure
            results += [_run_one(sim, scenarios[i], i) for i in members]
            continue
        axes = _axes_of(sim(template)) if members else None
        for k, i in enumerate(members):
            code = int(status[k])
            if code != 0:
                from tenqsp.gradients import _STATUS

                why = _STATUS.get(code, f"status {code}")
                results.append({"index": i, "values": None, "axes": None, "error": why})
                continue
            values = {o: np.asarray(outs[o][k]) for o in sim.outputs}
            results.append({"index": i, "values": values, "axes": axes, "error": ""})
    return results


# ------------------------------------------------------------------- several processes


def _run_parallel(sim, scenarios, n_jobs) -> list[dict]:
    """``n_jobs`` processes (started fresh: JAX does not survive ``fork``), each with its own
    build of the model, each running a share of the scenarios."""
    import concurrent.futures as cf
    import multiprocessing as mp

    n = len(scenarios)
    shares = [list(range(k, n, n_jobs)) for k in range(min(n_jobs, n))]
    payload = (sim.model.to_ir(), sim.outputs, sim.options)
    ctx = mp.get_context("spawn")
    with (
        _one_thread_per_worker(),
        cf.ProcessPoolExecutor(
            max_workers=len(shares), mp_context=ctx, initializer=_worker_init, initargs=(payload,)
        ) as pool,
    ):
        futures = [pool.submit(_worker_run, [scenarios[i] for i in s], s) for s in shares]
        results = []
        for fut in futures:
            results += fut.result()
    return sorted(results, key=lambda r: r["index"])


# Each worker's BLAS (numpy's OpenBLAS) and XLA would otherwise start a thread per core, and
# idle BLAS threads spin: two 10,000-state solves at once took > 400 s each instead of 46 s
# (one thread each: 37 s, also faster alone). Read once, when the worker imports numpy and
# JAX, so it is set in the environment the workers start with; a value already set wins.
_WORKER_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}


@contextlib.contextmanager
def _one_thread_per_worker():
    added = [k for k in _WORKER_ENV if k not in os.environ]
    os.environ.update({k: _WORKER_ENV[k] for k in added})
    try:
        yield
    finally:
        for k in added:
            os.environ.pop(k, None)


_WORKER: dict = {}


def _worker_init(payload):
    import tenqsp as tq

    ir, outputs, options = payload
    _WORKER["sim"] = tq.Model.from_ir(ir).simulator(outputs=list(outputs), **options)


def _worker_run(scenarios, indices):
    sim = _WORKER["sim"]
    by_index = dict(zip(indices, scenarios, strict=True))
    order = list(indices)
    full = [by_index[i] for i in order]
    results = _run_chunk(sim, full, list(range(len(full))))
    for r in results:
        r["index"] = order[r["index"]]
    return results


# --------------------------------------------------------------------------------- pre-runs


@dataclasses.dataclass(frozen=True)
class PreRun:
    """The end of a pre-run: ``state`` (by name, as ``sol`` gives them), ``y`` (flat),
    ``t`` (how long it ran), ``reached`` (``until="steady"``: whether it got there), and
    ``cycles`` (dosing intervals run, for a steady state under a repeated regimen). Start a
    scenario from it with ``sc.replace(initial=pre)``."""

    state: dict
    y: np.ndarray
    t: float
    reached: bool
    cycles: int = 0
    until: str = "steady"  # "steady", or "time" for a run of a fixed length

    def __repr__(self) -> str:
        if self.until == "time":
            return f"<PreRun: {self.t:g} time units (a fixed time)>"
        how = "steady" if self.reached else "not steady"
        extra = f" ({self.cycles} dosing intervals)" if self.cycles else ""
        return f"<PreRun: {self.t:g} time units, {how}{extra}>"


STEADY_FIRST = 1.0  # the first stretch of a steady-state search (then doubling)


def pre_run(sim, scenario=None, *, until="steady", t_max: float = 1e6) -> PreRun:
    """See ``pre_run``."""
    from tenqsp.simulate import Scenario

    sc = scenario if scenario is not None else Scenario()
    if not isinstance(sc, Scenario):
        raise TypeError(f"The first argument is a tq.Scenario, not {sc!r}.")
    if not (isinstance(t_max, (int, float)) and t_max > 0):
        raise ValueError(f"t_max must be a number > 0, not {t_max!r}.")
    cm = sim.model.compile()
    t0 = 0.0 if sc.start is None else sc.start
    start = sc.initial_vector(cm)
    if isinstance(until, (int, float)) and not isinstance(until, bool):
        if not until > 0:
            raise ValueError(f"until= takes 'steady' or a time > 0, not {until!r}.")
        reg = _before(sc.regimen, t0 + until)
        sol = _solve(sim, cm, sc, t0, t0 + until, reg, start)
        return _pre(cm, sol, until, reached=True, until="time")
    if until != "steady":
        raise ValueError(f"until= takes 'steady' or a time > 0, not {until!r}.")
    if isinstance(sc.regimen, Regimen) and len(sc.regimen.events) > 1:
        try:
            _period(sc.regimen)
        except ValueError as e:
            raise ValueError(
                f"until='steady' with doses needs a regimen that repeats at even intervals "
                f"(e.g. tq.bolus(..., every=12, n=...)); this one does not ({e}). Use "
                "until=<time> instead."
            ) from None
        return _steady_cycles(sim, cm, sc, t0, start, t_max)
    return _steady_rest(sim, cm, sc, t0, start, t_max)


def _solve(sim, cm, sc, a, b, regimen, start):
    opts = {k: v for k, v in sim.options.items()}
    return sim.model.solve(
        (a, b),
        times=[b],
        params=sc.params or None,
        regimen=regimen,
        initial=start,
        **opts,
    )


def _solve_at(sim, sc, a, b, regimen, start, times):
    return sim.model.solve(
        (a, b),
        times=times,
        params=sc.params or None,
        regimen=regimen,
        initial=start,
        **dict(sim.options),
    )


def _pre(cm, sol, t, reached, cycles=0, until="steady") -> PreRun:
    y = np.asarray(sol.y[-1], dtype=np.float64)
    state = {k: np.asarray(v) for k, v in cm.unpack(y).items()}
    return PreRun(state=state, y=y, t=float(t), reached=reached, cycles=cycles, until=until)


def _before(regimen, end):
    """The regimen's doses before ``end`` (an infusion started before it runs on to its
    end only up to ``end``, as the solve stops there); the model's doses: None."""
    if not isinstance(regimen, Regimen):
        return None
    return Regimen(
        e for e in regimen.events if (e.time < end if isinstance(e, Bolus) else e.start < end)
    )


def _rates_small(cm, sol, rtol, atol) -> bool:
    y = np.asarray(sol.y[-1], dtype=np.float64)
    f = np.asarray(cm.rhs(float(sol.t[-1]), y, sol.theta))
    return bool(np.all(np.abs(f) <= np.asarray(atol) + rtol * np.abs(y)))


def _max_lag(cm, sc) -> float:
    """The longest lag of any dose (m.lag) at the scenario's values: doses arrive that much
    after they are given."""
    lag, _ = cm.arrival(cm.theta(sc.params or None))
    return float(np.max(lag, initial=0.0))


def _steady_rest(sim, cm, sc, t0, start, t_max) -> PreRun:
    """No repeated regimen: give its doses (waiting until the last has arrived, lags
    included), then run on in doubling stretches until every rate of change is within the
    tolerances. The clock runs on from stretch to stretch, so equations that read t see the
    right time."""
    rtol = sim.options.get("rtol", 1e-6)
    atol = sim.options.get("atol", 1e-6)
    reg = sc.regimen if isinstance(sc.regimen, Regimen) else None
    last = t0
    if reg is not None and reg.events:
        last = max(e.time if isinstance(e, Bolus) else e.stop for e in reg.events)
        last += _max_lag(cm, sc)
    elif reg is None and cm.dose_schedule:  # the model's own doses
        last = max(td for td, _ in cm.dose_schedule) + _max_lag(cm, sc)
    first = max(STEADY_FIRST, last - t0)
    sol = _solve(sim, cm, sc, t0, t0 + first, reg if reg is not None else None, start)
    run = first
    while not _rates_small(cm, sol, rtol, atol):
        if run >= t_max:
            warnings.warn(
                f"pre_run: not steady after {run:g} time units (t_max); the state is still "
                "changing. Give a larger t_max, or until=<time>.",
                stacklevel=3,
            )
            return _pre(cm, sol, run, reached=False)
        step = min(run, t_max - run)
        y = np.asarray(sol.y[-1], dtype=np.float64)
        sol = _solve(sim, cm, sc, t0 + run, t0 + run + step, Regimen(), y)
        run += step
    return _pre(cm, sol, run, reached=True)


def _period(regimen) -> tuple[float, list]:
    """The regimen's repeating unit: (interval, the doses of one interval, times relative to
    its start); raises if the doses do not repeat."""

    def when(e):
        return e.time if isinstance(e, Bolus) else e.start

    def same(e1, e2):
        if type(e1) is not type(e2) or e1.target != e2.target or e1.where != e2.where:
            return False
        if isinstance(e1, Bolus):
            return math.isclose(e1.amount, e2.amount)
        return math.isclose(e1.rate, e2.rate) and math.isclose(e1.duration, e2.duration)

    times = sorted({when(e) for e in regimen.events})
    if len(times) < 2:
        raise ValueError("one dosing time")
    gaps = np.diff(times)
    tau = float(gaps[0])
    if not np.allclose(gaps, tau, rtol=1e-9, atol=1e-12) or tau <= 0:
        raise ValueError("uneven intervals")
    first = sorted((e for e in regimen.events if when(e) == times[0]), key=repr)
    for t in times[1:]:
        group = sorted((e for e in regimen.events if when(e) == t), key=repr)
        if len(group) != len(first) or not all(map(same, first, group)):
            raise ValueError("different doses")
    unit = [
        dataclasses.replace(e, **({"time": 0.0} if isinstance(e, Bolus) else {"start": 0.0}))
        for e in first
    ]
    for e in unit:
        if isinstance(e, Infusion) and e.duration > tau:
            raise ValueError("an infusion longer than the interval")
    return tau, unit


def _steady_cycles(sim, cm, sc, t0, start, t_max) -> PreRun:
    """A repeated regimen: the doses given every interval, the run doubled in length until
    the state half-way through the last interval matches the one an interval before, within
    the tolerances. ``pre`` holds the trough: the state one interval after the last dose,
    just before the next would be given. Each run starts at the scenario's start, so
    equations that read t and doses with a lag (arriving in a later interval) come out
    right."""
    tau, unit = _period(sc.regimen)
    rtol = sim.options.get("rtol", 1e-6)
    atol = np.asarray(sim.options.get("atol", 1e-6))
    first = min(e.time if isinstance(e, Bolus) else e.start for e in sc.regimen.events)

    def given(e, at):
        return dataclasses.replace(e, **({"time": at} if isinstance(e, Bolus) else {"start": at}))

    k = 2  # doses given: k + 1, at first, first + tau, ..., first + k*tau
    while True:
        reg = Regimen(given(e, first + j * tau) for j in range(k + 1) for e in unit)
        end = first + (k + 1) * tau
        times = [first + (k - 1.5) * tau, first + (k - 0.5) * tau, end]
        sol = _solve_at(sim, sc, t0, end, reg, start, times)
        y_prev, y_now = np.asarray(sol.y[0]), np.asarray(sol.y[1])
        run = end - t0
        if np.all(np.abs(y_now - y_prev) <= atol + rtol * np.abs(y_now)):
            return _pre(cm, sol, run, reached=True, cycles=k + 1)
        if run >= t_max:
            warnings.warn(
                f"pre_run: not at steady state after {k + 1} dosing intervals ({run:g} time "
                "units, t_max).",
                stacklevel=3,
            )
            return _pre(cm, sol, run, reached=False, cycles=k + 1)
        k *= 2
