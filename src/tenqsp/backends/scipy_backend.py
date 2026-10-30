# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""scipy backend: ``solve_ivp`` with BDF (default) or Radau, and an exact JAX Jacobian
(ADR 0001: scipy BDF is the default forward solver at every size).

Doses split the run into segments. Each segment starts with a fresh step size at
the post-dose state. If BDF fails on a segment, that segment is retried once with Radau and
the retry is reported in ``Solution.messages``. LSODA is not offered.

The Jacobian is computed directly in sparse form from the model's structure (coloured
derivatives with the integrals lifted, ADR 0003; ``CompiledModel.sparse_jacobian``), and
scipy factorises it with a sparse LU. It is exact, not a finite-difference estimate.

The conservation ledger is integrated alongside the state: one extra "quadrature"
variable per ledger channel, whose derivative is that channel's rate. A linear multistep
method keeps any linear relation between the variables, so ``amount = initial + channels``
holds to the accuracy of the Newton iterations, not just to the truncation error.

Each segment's first step follows scipy's rule (Hairer, Nørsett & Wanner) but is never below
``FIRST_STEP_FLOOR`` times the segment length. After a large bolus the rule can propose
steps near 1e-12 h; at that size Newton's corrections are round-off, scipy's convergence-rate
test reads the noise as divergence, and the step is halved until BDF gives up ("required step
size is less than spacing between numbers", V14; pbisim saw the same at a 1e10 PFU bolus).
Any start from 1e-10 to 1e-1 h then integrates at the same cost; error control shrinks a
start that is too large, so the floor costs at most a few rejected steps.

The ledger variables never steer the integration: their absolute tolerance is infinite, so
their scaled errors count as zero in step-size and Newton control. scipy's error norm is a
root mean square over all variables, so the zeros would loosen the tolerance of the states by
``sqrt((n + n_q) / n)``; ``rtol`` and the states' ``atol`` are multiplied by
``sqrt(n / (n + n_q))`` to cancel that exactly, and a run with the ledger is as accurate as a
run without it. The identity above still holds, because Newton with the exact Jacobian
preserves linear invariants whether or not the ledger variables are in the convergence test.
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from tenqsp._threads import limited
from tenqsp.compile import CompiledModel
from tenqsp.ledger import build_ledger
from tenqsp.lower import CompileError
from tenqsp.results import Solution

METHODS = ("BDF", "Radau")
MAX_EVENTS = 10_000  # switch and extinction-group events per solve
FIRST_STEP_FLOOR = 1e-8  # relative to the segment length; see the module docstring
_ORDER = {"BDF": 1, "Radau": 3}  # the orders scipy's own starting-step rule uses
_TINY = float(np.finfo(np.float64).tiny)  # smallest normal number: event offsets


class SolverError(RuntimeError):
    """The integrator failed; the message says where and what was tried."""


# A large bolus leaves rounding errors of about this fraction of its size in the states it
# feeds; with atol below that, BDF's error estimate is noise and the run fails or survives by
# chance. Measured on test_v13_v14.bolus_model: plain BDF passed 8 of 8 nudged runs from
# atol = 1e-14 x the dose up (doses 1e9-1e11, bacteria 7e6-9e8), and down to 0 of 8 below it
# (2026-10-24).
ROUNDOFF_PER_DOSE = 1e-14


def atol_floor(rise: float) -> float:
    """The smallest atol at which a dose raising a state's value by ``rise`` is safe."""
    return ROUNDOFF_PER_DOSE * abs(rise)


def dose_hint(doses, t: float, atol_min: float) -> str:
    """A sentence about the largest dose given by ``t`` whose rounding exceeds ``atol``."""
    big = [d for d in doses if d[0] <= t and atol_floor(d[3]) > atol_min]
    if not big:
        return ""
    td, target, _, rise = max(big, key=lambda d: d[3])
    return (
        f" The dose into '{target}' at t = {td:g} raised its value by {rise:.3g}; with atol "
        f"below about {atol_floor(rise):.0e} its rounding errors can make BDF fail. Try "
        f"atol={atol_floor(rise):.0e} or larger."
    )


@limited  # one BLAS thread for SuperLU (tenqsp._threads)
def solve(
    cm: CompiledModel,
    t_span: Sequence[float],
    *,
    times: Sequence[float] | None = None,
    params: Mapping[str, object] | None = None,
    theta=None,
    method: str = "BDF",
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_step: float = np.inf,
    ledger: bool = True,
    regimen=None,
    initial=None,
    max_steps: int = 100_000,
) -> Solution:
    """Integrate a compiled model.

    Parameters
    ----------
    t_span
        ``(start, end)``.
    times
        Output times (default: 201 evenly spaced times over ``t_span``). Must lie in
        ``t_span``.
    params
        Parameter values to change for this run, e.g. ``{"k": 2e-9}`` or, per category,
        ``{"k": {"R": 0}}``. Never recompiles.
    theta
        The whole parameter vector instead (see ``cm.theta()``); not together with ``params``.
    method
        ``"BDF"`` (default) or ``"Radau"``.
    rtol, atol, max_step
        Passed to ``scipy.integrate.solve_ivp`` (``atol`` a number, or one per state).
    ledger
        Keep the conservation ledger (``sol.ledger``; default). ``False`` skips it.
    regimen
        A ``Regimen`` that **replaces** the doses written in the model
        (``Regimen()``: no doses). ``None`` (default): the model's own doses.
    initial
        The starting state instead of the model's initial values: one number per state
        (``cm.n_states``), NaN where the model's own value stays (a pre-run's end state,
        ``pre_run``). ``None`` (default): the model's.
    max_steps
        At most this many solver steps per segment (between doses, switches and events), as
        on the JAX backend; more raises ``SolverError`` instead of crawling on with
        tiny steps (typically tolerances near rounding for a fast process).
    """
    if method not in METHODS:
        hint = (
            " LSODA is not offered (it can stall silently on stiff models)."
            if method.upper() == "LSODA"
            else ""
        )
        raise ValueError(f"method must be one of {', '.join(METHODS)}, not {method!r}.{hint}")
    t0, t1 = (float(x) for x in t_span)
    if not t1 > t0:
        raise ValueError(f"t_span must be increasing, got {tuple(t_span)}.")
    if params is not None and theta is not None:
        raise ValueError(
            "Give either params= (changes by name) or theta= (whole vector), not both."
        )
    th = np.array(cm._th(cm.theta(params) if theta is None else theta))
    cm.check_values(th)
    slot = cm.plan.switch_slot  # segment time of conditions on time (ADR 0006)
    t_out = np.linspace(t0, t1, 201) if times is None else np.asarray(times, dtype=np.float64)
    if t_out.ndim != 1 or t_out.size == 0:
        raise ValueError("times must be a non-empty list of numbers.")
    if np.any(np.diff(t_out) < 0):
        raise ValueError("times must be in increasing order.")
    if t_out[0] < t0 or t_out[-1] > t1:
        raise ValueError(f"times must lie within t_span {t0, t1}.")
    # a time given twice gets a row each (as on the JAX backend): solved once, then repeated
    t_out, t_rows = np.unique(t_out, return_inverse=True)
    n = cm.n_states
    n_q = len(cm.ledger_entries) if ledger else 0
    atol_states = np.asarray(atol, dtype=np.float64)
    if atol_states.ndim > 1 or atol_states.size not in (1, n) or np.any(atol_states <= 0):
        raise ValueError(f"atol must be a positive number or {n} positive numbers (one per state).")
    # ledger variables: infinite tolerance, so they never steer the integration; the states'
    # tolerances are rescaled so the RMS error norm over n + n_q variables equals the norm
    # over the n states alone (see the module docstring)
    shrink = np.sqrt(n / (n + n_q)) if n_q else 1.0
    rtol_given = rtol
    rtol = rtol * shrink
    atol = np.concatenate([np.broadcast_to(atol_states * shrink, (n,)), np.full(n_q, np.inf)])

    # doses: group by time; a dose before the start would be silently lost, so refuse it
    measure = cm.measure(th)  # amount per unit of each state (widths, weights, volumes)
    group_w = cm.group_weights(th)
    if regimen is None:  # the model's own doses (amounts may be formulas of parameters)
        amounts = np.asarray(cm.dose_amounts(th))
        boluses = [  # a field dose's cells with no share are left out
            (td, target, index, float(amt))
            for (td, target), index, amt, spread in zip(
                cm.dose_schedule, cm.dose_index, amounts, cm.dose_spread, strict=True
            )
            if not (spread and amt == 0)
        ]
        infusions = []
    else:  # a regimen replaces them; a dose into a field is spread over its cells
        boluses, infusions = [], []
        for b in regimen.boluses:
            cells, shares = cm.event_cells(b.target, b.where, b.shape, th)
            boluses += [
                (b.time, b.target, int(c), b.amount * w) for c, w in zip(cells, shares, strict=True)
            ]
        for f in regimen.infusions:
            cells, shares = cm.event_cells(f.target, f.where, f.shape, th, "infusion")
            infusions += [
                (f.start, f.stop, f.target, int(c), f.rate * w)
                for c, w in zip(cells, shares, strict=True)
            ]
    for td, target, _, _ in boluses:
        if td < t0:
            raise ValueError(
                f"A dose into '{target}' at t = {td:g} is before the start t = {t0:g}."
            )
    for ts, _, target, _, _ in infusions:
        if ts < t0:
            raise ValueError(
                f"An infusion into '{target}' starts at t = {ts:g}, before the start t = {t0:g}."
            )
    # lags and bioavailabilities (m.lag, m.bioavailability): when and how much arrives
    lag, frac = cm.arrival(th)
    events: dict[float, list[tuple[int, float]]] = {}
    given: list[tuple[float, str, float, float]] = []  # (arrival, target, amount, fraction)
    doses: list[tuple[float, str, float, float]] = []  # (arrival, target, arrived, rise)
    for td, target, index, amt in boluses:
        td, f = td + float(lag[index]), float(frac[index])
        if td <= t1:
            events.setdefault(td, []).append((index, amt * f / measure[index]))
            given.append((td, target, amt, f))
            doses.append((float(td), target, float(amt * f), float(amt * f / measure[index])))
    # (start, stop, target, rate, fraction) in t_span, as arrived
    infused: list[tuple[float, float, str, float, float]] = []
    infusion_rates = []  # (start, stop, index, rise per time of the state)
    for ts, te, target, index, rate in infusions:
        ts, te, f = ts + float(lag[index]), te + float(lag[index]), float(frac[index])
        if ts < t1:
            infused.append((ts, min(te, t1), target, rate, f))
            infusion_rates.append((ts, te, index, rate * f / measure[index]))

    y = np.asarray(cm.initial_state(th), dtype=np.float64)
    if initial is not None:
        start = np.asarray(initial, dtype=np.float64)
        if start.shape != y.shape:
            raise ValueError(f"initial= has {start.size} numbers; the model has {y.size} states.")
        y = np.where(np.isnan(start), y, start)
    bad = [k for k, v in cm.unpack(y).items() if not np.all(np.isfinite(v))]
    if bad:
        raise ValueError(
            f"The initial values of {bad} are not finite. A shape function may have no mass on "
            "its axis (e.g. a normal centred far outside the axis), or a formula divides by zero."
        )

    def apply(t, y):
        for index, rise in events.get(t, []):
            y[index] += rise
        return y

    y_initial = y.copy()
    rhs, geom = (cm._aug if ledger else cm._rhs), cm.geom
    u = np.zeros(n + n_q)  # infusions running in the current segment (constant on it)

    def fun(t, y):
        f = np.asarray(rhs(np.float64(t), y, th, geom))
        return f + u if infusion_rates else f

    def jacobian(t, y):
        return cm.sparse_jacobian(t, y[:n], th, ledger=ledger)

    y = apply(t0, np.concatenate([y, np.zeros(n_q)]))
    groups = cm.extinction_groups
    armed = [False] * len(groups)
    floor_of = dict(zip((g.name for g in cm._floors), cm.floor_slots, strict=True))

    def set_armed(i, value):  # a floored group counts (no floor) exactly while it is armed
        armed[i] = value
        if groups[i].name in floor_of:
            th[floor_of[groups[i].name]] = 0.0 if value else 1.0

    def arm(y):  # a group reaching its threshold (start, doses) is armed
        totals = cm.group_totals(y, th) if groups else ()
        for i, tot in enumerate(totals):
            set_armed(i, armed[i] or bool(tot >= groups[i].threshold))

    arm(y)
    n_sw = len(cm.switch_events)
    # next crossing to watch for, per state-dependent switch: +1 up, -1 down, 0 (the value
    # is exactly on the boundary) leaving the boundary either way
    direction = np.zeros(n_sw)

    def aim(t, y):  # watch each switch for the crossing away from where it now stands
        if n_sw:
            v = cm.switch_values(t, y[:n], th)
            direction[:] = np.where(v > 0, -1.0, np.where(v < 0, 1.0, 0.0))

    # conditions on time switch at known times: segment boundaries, like doses
    has_time_switches = cm.switch_roots or cm.switch_sampled
    switches = (
        [s for s in cm.switch_times(th, (t0, t1)) if t0 < s < t1] if has_time_switches else []
    )
    edges = {x for ts, te, _, _ in infusion_rates for x in (ts, te) if t0 < x < t1}
    stops = sorted({td for td in events if t0 < td <= t1} | set(switches) | edges)
    bounds = [t0, *[s for s in stops if s < t1], t1]
    out_t, out_y, messages = [], [], []
    removed: list[tuple[float, str, float]] = []  # (time, member, -amount) at extinctions
    extinctions: list[tuple[float, str]] = []
    stats = {"nfev": 0, "njev": 0, "nlu": 0, "segments": 0, "extinctions": 0}

    def extinct(i, te, y):  # the whole group goes to zero (in place), booked in the ledger
        g = groups[i]
        for member in dict.fromkeys(g.group):
            sl = cm.layout.slices[member]
            part = g.members[sl]  # the whole member, or its slice
            amount = float(y[sl][part] @ measure[sl][part])
            removed.append((te, member, -amount))
            y[sl] = np.where(part, 0.0, y[sl])
        set_armed(i, False)
        extinctions.append((te, g.name))
        stats["extinctions"] += 1

    n_events = 0
    switched = None  # (time, condition) of the last state-dependent switch
    start = time.perf_counter()
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if slot is not None:
            th[slot] = 0.5 * (a + b)  # conditions on time are constant on (a, b)
        if infusion_rates:
            mid = 0.5 * (a + b)
            u[:] = 0.0
            for ts, te, index, rise in infusion_rates:
                if ts <= mid < te:
                    u[index] += rise
        aim(a, y)
        while True:  # integrate [a, b], stopping at each group or switch event
            wanted = t_out[(t_out >= a) & (t_out < b)]
            t_eval = np.concatenate([wanted, [b]])
            watch = _group_events(cm, group_w, armed, n) + _switch_events(cm, direction, th, n)
            res = _attempt(
                fun, jacobian, a, b, y, t_eval, method, rtol, atol, max_step, stats, watch,
                max_steps, (rtol_given, atol_states),
            )  # fmt: skip
            if not res.success and method == "BDF":
                messages.append(
                    f"BDF failed on [{a:g}, {b:g}] ({res.message}); retried with Radau."
                    + dose_hint(doses, b, float(np.min(atol_states)))
                )
                res = _attempt(
                    fun, jacobian, a, b, y, t_eval, "Radau", rtol, atol, max_step, stats,
                    watch, max_steps, (rtol_given, atol_states),
                )  # fmt: skip
            if not res.success:
                reached = f" (reached t = {res.t[-1]:g})" if len(res.t) else ""
                hint = ""
                if switched is not None and switched[0] == a:
                    hint = (
                        f" The condition '{switched[1]}' had just switched at t = {a:g}; if the "
                        "model pushes itself straight back across it, the solution slides along "
                        "the boundary and no step size works. Replace the switch by a smooth "
                        "form (e.g. with sigmoid), or model what happens on the boundary."
                    )
                if not hint:
                    hint = dose_hint(doses, b, float(np.min(atol_states)))
                raise SolverError(
                    f"The solver failed between t = {a:g} and t = {b:g}{reached}: "
                    f"{res.message}{hint}"
                )
            stats["segments"] += 1
            if res.status != 1:  # reached b
                out_t.append(res.t[:-1])
                out_y.append(res.y[:, :-1].T)
                y = res.y[:, -1].copy()
                break
            # stopped at an event (group or switch): the earliest one
            i, te = min(
                ((k, float(ts[0])) for k, ts in enumerate(res.t_events) if len(ts)),
                key=lambda kv: kv[1],
            )
            # scipy returns lists, not arrays, when no output time came before the event
            t_seg = np.asarray(res.t, dtype=np.float64)
            y_seg = np.asarray(res.y, dtype=np.float64).reshape(len(y), t_seg.size)
            keep = t_seg < te
            out_t.append(t_seg[keep])
            out_y.append(y_seg[:, keep].T)
            y = res.y_events[i][0].copy()
            n_events += 1
            if n_events > MAX_EVENTS:
                raise SolverError(
                    f"More than {MAX_EVENTS} switch or extinction events by t = {te:g}: a "
                    "condition keeps flipping back and forth (the model pushes itself back "
                    "across it). Replace the switch by a smooth form, e.g. with sigmoid."
                )
            a = te
            # Several events can fall on the same instant (two strains dying together); scipy
            # reports only the first, and a group already below its threshold at the restart
            # would never see a downward crossing. So every group and switch is settled here.
            if i < len(groups):
                if armed[i]:
                    extinct(i, te, y)  # fell below its threshold
                else:
                    set_armed(i, True)  # reached its threshold: armed from now on
            totals = cm.group_totals(y, th) if groups else ()
            for j, tot in enumerate(totals):
                if j == i:  # settled above (its total is at the threshold up to round-off)
                    continue
                if armed[j] and tot <= groups[j].threshold:
                    extinct(j, te, y)
                elif not armed[j] and tot >= groups[j].threshold:
                    set_armed(j, True)
            before = direction.copy()
            aim(te, y)  # every switch from where it now stands (zeroing may move them)
            if i >= len(groups):  # the condition that switched: watch for the way back
                k = i - len(groups)
                switched = (te, cm.switch_events[k])
                if before[k] != 0:  # its value at the root is zero up to round-off
                    direction[k] = -before[k]
            if a >= b:
                break
        y = apply(b, y)
        arm(y)
    if slot is not None:
        th[slot] = np.nan  # outputs: conditions on time use each output time itself
    th[cm.floor_slots] = np.nan  # outputs: floors decided by the totals
    if t_out[-1] == t1:
        out_t.append(np.array([t1]))
        out_y.append(y[None, :])
    stats["wall_seconds"] = time.perf_counter() - start
    T = np.concatenate(out_t)
    Z = np.concatenate(out_y) if out_y else np.zeros((0, n + n_q))
    if T.size == t_out.size and t_rows.size != t_out.size:
        T, Z = T[t_rows], Z[t_rows]
    Y = Z[:, :n]
    book = (
        build_ledger(cm, T, Y, Z[:, n:], th, y_initial, given, t0, removed, infused)
        if ledger
        else None
    )
    sol = _solution(cm, T, Y, th, method, stats, messages, book, extinctions)
    sol.doses = doses
    sol.infusions = [(float(a), float(b), tg, float(r * f)) for a, b, tg, r, f in infused]
    user_atol = float(atol_states) if atol_states.size == 1 else atol_states.copy()
    sol.options = {
        "method": method,
        "backend": "scipy",
        "rtol": rtol_given,
        "atol": user_atol,
        "max_step": max_step,
    }
    return sol


def _switch_events(cm: CompiledModel, direction: np.ndarray, th, n: int) -> list:
    """One terminal event per state-dependent condition (ADR 0006): ``lhs - rhs``, watched
    in the direction away from where it stands, so a restart exactly on the boundary does not
    see the same crossing again. A value exactly on the boundary (a compartment starting at
    0 for ``C > 0``) is watched until it leaves the boundary, in either direction. The values
    of all conditions are computed once per state."""
    if not len(direction):
        return []
    last: dict = {}

    def values(t, y):
        key = (t, y[:n].tobytes())
        if last.get("key") != key:
            last["key"], last["v"] = key, cm.switch_values(t, y[:n], th)
        return last["v"]

    out = []
    for k, d in enumerate(direction):
        if d == 0:  # exactly on the boundary: watch |v| leaving it, either way

            def crossing(t, y, k=k):
                return abs(float(values(t, y)[k])) - _TINY

            crossing.direction = 1.0
        else:  # shifted by the smallest normal number, so a value that stays on the
            # boundary (a state at rest on its threshold) is not a crossing at every step

            def crossing(t, y, k=k, d=d):
                return float(values(t, y)[k]) - d * _TINY

            crossing.direction = float(d)
        crossing.terminal = True
        out.append(crossing)
    return out


def _group_events(cm: CompiledModel, group_w: list, armed: list[bool], n: int) -> list:
    """One terminal event per extinction group: its total minus its threshold, watched
    downward once the group is armed, upward until then."""
    out = []
    for g, w_g, is_armed in zip(cm.extinction_groups, group_w, armed, strict=True):

        def crossing(t, y, w=w_g, thr=g.threshold):
            return float(y[:n] @ w) - thr

        crossing.terminal = True
        crossing.direction = -1.0 if is_armed else 1.0
        out.append(crossing)
    return out


class _TooManySteps(Exception):
    def __init__(self, t: float, h: float):
        self.t, self.h = t, h


def _counted(method: str, limit: int):
    """scipy's integrator class for ``method``, stopping after ``limit`` steps."""
    from scipy.integrate import BDF, Radau

    base = {"BDF": BDF, "Radau": Radau}[method]

    class Counted(base):
        def _step_impl(self):
            self._tq_steps = getattr(self, "_tq_steps", 0) + 1
            if self._tq_steps > limit:
                raise _TooManySteps(float(self.t), float(self.h_abs))
            return super()._step_impl()

    return Counted


def _attempt(
    fun, jac, a, b, y, t_eval, method, rtol, atol, max_step, stats, events=(),
    max_steps=100_000, given=None,
):  # fmt: skip
    """One try at a segment. An error raised inside the integrator (e.g. a singular or
    non-finite matrix) counts as a failure, so it can be retried; the work done is counted
    either way. More than ``max_steps`` steps raises ``SolverError`` at once (another
    method would crawl as well)."""
    try:
        res = _segment(fun, jac, a, b, y, t_eval, method, rtol, atol, max_step, events, max_steps)
    except _TooManySteps as stop:
        rtol_given, atol_states = given if given is not None else (rtol, atol)
        raise SolverError(
            f"The solver took more than {max_steps} steps between t = {a:g} and t = {b:g} and "
            f"reached only t = {stop.t:.6g} (last step {stop.h:.2g}): it is crawling, not "
            "failing. The usual cause is a tolerance near the limit of rounding for a fast "
            f"process (here rtol = {rtol_given:g}, atol = {float(np.min(atol_states)):g}): "
            "loosen rtol (e.g. 1e-8) or atol, or allow more steps (max_steps=...)."
        ) from None
    except CompileError:  # a problem in the model, not the integration
        raise
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, RuntimeError) as err:
        # RuntimeError: scipy's sparse LU ("Factor is exactly singular")
        return _Failed(f"{type(err).__name__}: {err}")
    for key in ("nfev", "njev", "nlu"):
        stats[key] += int(getattr(res, key))
    return res


@dataclasses.dataclass
class _Failed:
    message: str
    success: bool = False
    t: tuple = ()


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if np.size(x) else 0.0


def first_step(fun, a, b, y, method, rtol, atol, max_step) -> float:
    """Starting step for a segment: scipy's rule (``select_initial_step``, Hairer, Nørsett &
    Wanner, Solving ODEs I, II.4), floored at ``FIRST_STEP_FLOOR * (b - a)``."""
    span = b - a
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = atol + np.abs(y) * rtol  # infinite for the ledger variables: they count 0
        f0 = np.asarray(fun(a, y))
        d0, d1 = _rms(y / scale), _rms(f0 / scale)
        h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
        h0 = min(h0, span)
        f1 = np.asarray(fun(a + h0, y + h0 * f0))
        d2 = _rms((f1 - f0) / scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (_ORDER[method] + 1))
    h = min(100 * h0, h1, max_step)
    return float(min(max(h, FIRST_STEP_FLOOR * span), span, max_step))


def _segment(fun, jac, a, b, y, t_eval, method, rtol, atol, max_step, events=(), max_steps=100_000):
    with warnings.catch_warnings():
        # scipy's BDF allocates its difference table with np.empty and, on the first step,
        # subtracts a row it has not yet written (bdf.py, "D[order + 2] = d - D[order + 1]").
        # If that leftover memory holds a NaN pattern numpy warns; the row is overwritten
        # before it is ever used, so results are unaffected. Only that message from that
        # module is silenced.
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in subtract",
            category=RuntimeWarning,
            module=r"scipy\.integrate\._ivp\.bdf",
        )
        return solve_ivp(
            fun,
            (a, b),
            y,
            method=_counted(method, max_steps),
            t_eval=t_eval,
            jac=jac,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            first_step=first_step(fun, a, b, y, method, rtol, atol, max_step),
            events=list(events) or None,
        )


def _solution(
    cm: CompiledModel, T, Y, th, method, stats, messages, book=None, extinctions=()
) -> Solution:
    states = {}
    for name in cm.layout.names:
        sl, shape = cm.layout.slices[name], cm.layout.shapes[name]
        states[name] = Y[:, sl].reshape((T.size, *shape))
    observables, obs_axes = {}, {}
    if cm.observable_exprs:
        per_time = cm._over_times(cm._obs, T, Y, th)
        frame = cm.plan.frame
        for name in per_time[0]:
            v = np.stack([np.asarray(o[name]) for o in per_time])
            keep = tuple(a for a, k in zip(frame.axes, v.shape[1:], strict=True) if k > 1)
            obs_axes[name] = keep
            observables[name] = v.reshape((T.size, *(frame.sizes[a] for a in keep)))
    return Solution(
        t=T,
        y=Y,
        states=states,
        observables=observables,
        observable_axes=obs_axes,
        axes=dict(cm.layout.axes),
        theta=th,
        method=method,
        stats=stats,
        messages=messages,
        ledger=book,
        extinctions=list(extinctions),
        coords=_coords(cm.ir),
        units={
            **{f.name: f.units for f in cm.ir.fields},
            **{c.name: c.units for c in cm.ir.compartments},
        },
        model=cm.ir.name,
    )


def _coords(ir) -> dict[str, dict]:
    out = {}
    for a in ir.axes:
        if a.type == "categorical":
            out[a.name] = {"kind": "categorical", "units": a.units, "values": a.labels}
        else:
            f = np.asarray(a.faces, float)
            out[a.name] = {
                "kind": a.kind,
                "units": a.units,
                "values": 0.5 * (f[1:] + f[:-1]),
                "faces": f,
            }
    return out
