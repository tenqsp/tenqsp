# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""JAX backend: the model solved by the JAX BDF (``jax_bdf``) inside one
jitted program (step 4.3; design note ``docs/design/step4_3_jax_bdf.md``).

It does what the scipy backend does, in the same way, so the two agree (V9):

- doses split the run into segments; each segment starts afresh (order 1, scipy's first
  step with the same floor). A bolus is a jump at its boundary; an infusion is a constant
  input between its start and stop, which are boundaries too;
- an output time on a boundary is reported after the doses there (right-continuous);
- the ledger channels are integrated alongside the state with infinite absolute tolerance,
  and ``rtol``/``atol`` are rescaled exactly as in the scipy backend.

JAX needs fixed array sizes, so doses, infusions and output times sit in **slots** whose
number is rounded up to a power of two (design decision 2026-10-24): runs up to that many
share one build, and a run that needs more builds once more at the next size.

Conditions (ADR 0006) as on scipy: switch times of conditions on time are segment boundaries,
computed inside JAX (closed-form roots lambdified to JAX; sampled conditions on the same grid,
bisected to round-off), with each segment's midpoint held in the time slot of theta. Each
state-dependent condition is a terminal event, watched away from where it stands; after it
switches, it is watched for the way back (the scipy backend's rules, same offsets).

Extinction groups as on scipy: each group is a terminal event on its total
(upward until armed, downward after); at a crossing it is zeroed exactly there and the amount
removed is booked in the ledger; groups crossing at the same instant are settled together;
noise-floor groups are floored, through their theta slot, while disarmed.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tenqsp.backends.jax_bdf import (
    EVENT,
    NOT_FINITE,
    OK,
    STATUS_TEXT,
    bdf,
    empty_record,
    while_running,
)
from tenqsp.backends.scipy_backend import SolverError, _solution, dose_hint
from tenqsp.compile import CompiledModel
from tenqsp.ledger import build_ledger


def slots(k: int) -> int:
    """The number of slots for ``k`` items: the next power of two, at least 8."""
    return max(8, 1 << max(0, int(k) - 1).bit_length())


# The linear-algebra tier (design note step4_5_sparse_tier.md): a dense LU in JAX up to this
# many states, SuperLU on the host above (``jax_sparse``). Measured 2026-10-24: a dense LU
# of 1,261 states costs 25-70 ms against 2 ms for scipy's sparse LU.
SPARSE_FROM_STATES = 250
# On a GPU the dense LU runs on the GPU and the sparse tier's host calls go back to the CPU:
# measured on an RTX A6000 (step 4.6, design decision 2), the dense tier's batches are 2x
# faster at 353 states, 1.2x at 505, and slower from ~550 (64 members) to ~650 (1,000).
SPARSE_FROM_STATES_GPU = 500


def _platform() -> str:
    return jax.default_backend()


def tier(cm: CompiledModel) -> str:
    """``"dense"`` or ``"sparse"``: how this model's iteration matrix is factorised (by size;
    the threshold depends on the device JAX runs on)."""
    gpu = _platform() == "gpu"
    return (
        "sparse"
        if cm.n_states > (SPARSE_FROM_STATES_GPU if gpu else SPARSE_FROM_STATES)
        else "dense"
    )


def linear_parts(cm: CompiledModel, ledger: bool):
    """``(jac, LinearSolver)`` for this model's tier: ``jac(t, z, th)``."""
    from tenqsp.backends import jax_bdf, jax_sparse

    if tier(cm) == "sparse":
        return (
            lambda t, z, th: cm.jacobian_pieces(t, z, th, ledger=ledger),
            jax_sparse.linear_solver(cm, ledger=ledger),
        )
    return lambda t, z, th: cm.jacobian_jax(t, z, th, ledger=ledger), jax_bdf.DENSE


def _unsupported(cm: CompiledModel) -> str | None:
    if any(f is None for f in cm.switch_roots_jax):
        return "a condition on time whose switch times use a function without a JAX form"
    return None


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
):
    """Integrate with the JAX BDF. Same arguments and result as
    ``solve``, plus ``max_steps`` (per segment). There is
    no Radau fallback inside JAX: a failure raises ``SolverError`` naming where, and
    suggesting ``backend="scipy"``."""
    run = _prepare(
        cm, t_span, times, params, theta, method, rtol, atol, max_step, ledger, regimen, initial
    )
    t0, t1, t_out, n, th = run.t0, run.t1, run.t_out, run.n, run.th
    infusions, atol_states, y_initial = run.infusions, run.atol_states, run.y_initial
    program = _program(cm, ledger=ledger, model_doses=regimen is None, max_steps=max_steps)
    start = time.perf_counter()
    out = jax.block_until_ready(program(**run.args))
    wall = time.perf_counter() - start
    doses = _check(cm, out, run)
    Z = np.asarray(out["ys"])[: t_out.size]
    Y = Z[:, :n]
    stats = {
        k: int(out[k])
        for k in ("n_steps", "n_rejected", "nfev", "njev", "nlu", "segments", "n_events")
    }
    stats["wall_seconds"] = wall
    n_ext = int(out["rec"]["n"])
    stats["extinctions"] = n_ext
    extinctions, removed = [], []
    rec_t, rec_g = np.asarray(out["rec"]["t"]), np.asarray(out["rec"]["g"])
    rec_amount = np.asarray(out["rec"]["amount"])
    for r_ in range(n_ext):
        g = cm.extinction_groups[int(rec_g[r_])]
        te = float(rec_t[r_])
        extinctions.append((te, g.name))
        for member in dict.fromkeys(g.group):
            removed.append((te, member, float(rec_amount[r_, cm.layout.names.index(member)])))
    book = None
    lag, frac = run.lag, run.frac
    infused = [  # (start, stop, target, rate, fraction arriving), as arrived
        (ts + lag[i], min(te + lag[i], t1), tg, r, float(frac[i]))
        for ts, te, i, r, tg in infusions
        if ts + lag[i] < t1
    ]
    if ledger:
        amounts = np.asarray(out["b_amount"])[: len(run.b_time)]
        given = [  # (arrival, target, amount given, fraction arriving)
            (float(td + lag[i]), tg, float(a), float(frac[i]))
            for td, tg, a, i in zip(run.b_time, run.b_target, amounts, run.b_index, strict=True)
            if td + lag[i] <= t1
        ]
        book = build_ledger(cm, t_out, Y, Z[:, n:], th, y_initial, given, t0, removed, infused)
    sol = _solution(cm, t_out, Y, th, "BDF (JAX)", stats, [], book, extinctions)
    sol.doses = doses
    sol.infusions = [(float(a), float(b), tg, float(r * f)) for a, b, tg, r, f in infused]
    user_atol = float(atol_states) if atol_states.size == 1 else atol_states.copy()
    sol.options = {
        "method": "BDF",
        "backend": "jax",
        "rtol": rtol,
        "atol": user_atol,
        "max_step": max_step,
    }
    return sol


class _Run:
    """The host-side inputs of one run (``_prepare``)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _prepare(
    cm, t_span, times, params, theta, method, rtol, atol, max_step, ledger, regimen, initial=None
):
    """Check the arguments and lay them out in slots for the program (shared by
    ``solve``, ``record`` and the replay). ``initial``: the starting state (one
    number per state; NaN: the model's own initial value), or None."""
    if method != "BDF":
        raise ValueError(f"The JAX backend has method 'BDF' only, not {method!r}.")
    why = _unsupported(cm)
    if why is not None:
        raise NotImplementedError(
            f"The JAX backend does not yet handle {why}; use backend='scipy' for this model."
        )
    t0, t1 = (float(x) for x in t_span)
    if not t1 > t0:
        raise ValueError(f"t_span must be increasing, got {tuple(t_span)}.")
    if params is not None and theta is not None:
        raise ValueError(
            "Give either params= (changes by name) or theta= (whole vector), not both."
        )
    th = np.array(cm._th(cm.theta(params) if theta is None else theta))
    cm.check_values(th)
    if regimen is None and cm.dose_parts:
        cm.dose_amounts(th)  # raises for a dose that is not finite (a shape off its axis)
    lag, frac = cm.arrival(th)  # raises for a lag or bioavailability below 0
    t_out = np.linspace(t0, t1, 201) if times is None else np.asarray(times, dtype=np.float64)
    if t_out.ndim != 1 or t_out.size == 0:
        raise ValueError("times must be a non-empty list of numbers.")
    if np.any(np.diff(t_out) < 0):
        raise ValueError("times must be in increasing order.")
    if t_out[0] < t0 or t_out[-1] > t1:
        raise ValueError(f"times must lie within t_span {t0, t1}.")
    n = cm.n_states
    n_q = len(cm.ledger_entries) if ledger else 0
    atol_states = np.asarray(atol, dtype=np.float64)
    if atol_states.ndim > 1 or atol_states.size not in (1, n) or np.any(atol_states <= 0):
        raise ValueError(f"atol must be a positive number or {n} positive numbers (one per state).")
    shrink = np.sqrt(n / (n + n_q)) if n_q else 1.0  # as the scipy backend (module docstring)
    atol_all = np.concatenate([np.broadcast_to(atol_states * shrink, (n,)), np.full(n_q, np.inf)])

    # doses: (time, index) on the host; amounts may be formulas of theta (the model's doses)
    # a regimen's dose into a field is spread over cells: one slot per cell, with its share
    # (b_event, b_share: which bolus of the regimen each slot comes from, and its fraction)
    b_event = b_share = i_event = i_share = None
    if regimen is None:
        b_time = np.array([td for td, _ in cm.dose_schedule], dtype=np.float64)
        b_index = np.array(cm.dose_index, dtype=np.int64)
        b_target = [tg for _, tg in cm.dose_schedule]
        b_amount = None  # from the model's formulas, in JAX
        infusions = []
    else:
        slots_b = [
            (k, b, int(c), w)
            for k, b in enumerate(regimen.boluses)
            for c, w in zip(*cm.event_cells(b.target, b.where, b.shape, th), strict=True)
        ]
        b_time = np.array([b.time for _, b, _, _ in slots_b], dtype=np.float64)
        b_index = np.array([c for _, _, c, _ in slots_b], dtype=np.int64)
        b_target = [b.target for _, b, _, _ in slots_b]
        b_amount = np.array([b.amount * w for _, b, _, w in slots_b], dtype=np.float64)
        b_event = np.array([k for k, _, _, _ in slots_b], dtype=np.int64)
        b_share = np.array([w for _, _, _, w in slots_b], dtype=np.float64)
        slots_i = [
            (k, f, int(c), w)
            for k, f in enumerate(regimen.infusions)
            for c, w in zip(
                *cm.event_cells(f.target, f.where, f.shape, th, "infusion"), strict=True
            )
        ]
        infusions = [(f.start, f.stop, c, f.rate * w, f.target) for _, f, c, w in slots_i]
        i_event = np.array([k for k, _, _, _ in slots_i], dtype=np.int64)
        i_share = np.array([w for _, _, _, w in slots_i], dtype=np.float64)
    for td, tg in zip(b_time, b_target, strict=True):
        if td < t0:
            raise ValueError(f"A dose into '{tg}' at t = {td:g} is before the start t = {t0:g}.")
    for ts, _, _, _, tg in infusions:
        if ts < t0:
            raise ValueError(
                f"An infusion into '{tg}' starts at t = {ts:g}, before the start t = {t0:g}."
            )

    K, M, T = slots(len(b_time)), slots(len(infusions)), slots(t_out.size)
    # switch times of sampled conditions on time: as many slots as the host finds (+ room)
    sw_slots = slots(cm.switch_times(th, (t0, t1)).size) if cm.switch_sampled else 1
    pad = lambda a, k, v: np.concatenate([a, np.full(k - len(a), v)])  # noqa: E731
    args = dict(
        theta=jnp.asarray(th),
        t0=t0,
        t1=t1,
        t_out=jnp.asarray(pad(t_out, T, np.inf)),
        b_time=jnp.asarray(pad(b_time, K, np.inf)),
        b_index=jnp.asarray(pad(b_index, K, 0).astype(np.int64)),
        b_amount=None if b_amount is None else jnp.asarray(pad(b_amount, K, 0.0)),
        i_start=jnp.asarray(pad(np.array([f[0] for f in infusions]), M, np.inf)),
        i_stop=jnp.asarray(pad(np.array([f[1] for f in infusions]), M, np.inf)),
        i_index=jnp.asarray(pad(np.array([f[2] for f in infusions], dtype=np.int64), M, 0)),
        i_rate=jnp.asarray(pad(np.array([f[3] for f in infusions]), M, 0.0)),
        sw_capacity=jnp.zeros(sw_slots),
        rtol=rtol * shrink,
        atol=jnp.asarray(atol_all),
        max_step=float(max_step),
    )
    y_initial = np.asarray(cm.initial_state(th), dtype=np.float64)
    if initial is not None:
        start = np.asarray(initial, dtype=np.float64)
        if start.shape != y_initial.shape:
            raise ValueError(f"initial= has {start.size} numbers; the model has {n} states.")
        args["y_start"] = jnp.asarray(start)
        y_initial = np.where(np.isnan(start), y_initial, start)
    bad = [k for k, v in cm.unpack(y_initial).items() if not np.all(np.isfinite(v))]
    if bad:
        raise ValueError(
            f"The initial values of {bad} are not finite. A shape function may have no mass on "
            "its axis (e.g. a normal centred far outside the axis), or a formula divides by zero."
        )
    return _Run(
        t0=t0, t1=t1, t_out=t_out, n=n, th=th, b_time=b_time, b_index=b_index,
        b_target=b_target, infusions=infusions, atol_states=atol_states, args=args,
        y_initial=y_initial, model_doses=regimen is None, lag=lag, frac=frac,
        b_event=b_event, b_share=b_share, i_event=i_event, i_share=i_share,
    )  # fmt: skip


def _check(cm, out, run) -> list:
    """The doses as they arrived (time, target, amount, rise); raises SolverError on a
    failed run."""
    b_target, b_index, t1, th = run.b_target, run.b_index, run.t1, run.th
    b_time = run.b_time + run.lag[b_index]  # arrival times (m.lag)
    atol_states = run.atol_states
    status = int(out["status"])
    arrived = np.asarray(out["b_amount"])[: len(b_time)] * run.frac[b_index]
    measure = np.asarray(cm.measure(th))
    spread = cm.dose_spread if run.model_doses else [False] * len(b_time)
    doses = [  # a field dose's cells with no share are not listed
        (float(td), tg, float(a), float(a / measure[i]))
        for td, tg, a, i, sp_ in zip(b_time, b_target, arrived, b_index, spread, strict=True)
        if td <= t1 and not (sp_ and a == 0)
    ]
    if status == TOO_MANY_EXTINCTIONS:
        raise SolverError(
            f"More than {EXTINCTION_RECORDS} extinctions in one run on the JAX backend; use "
            "backend='scipy' for this run."
        )
    if status == BAD_ARRIVAL:
        raise SolverError("A lag or bioavailability is below 0 or not finite.")
    if status in (BAD_REGIMEN, SWITCH_CAPACITY):  # pragma: no cover - the host checks first
        raise SolverError(f"The run cannot start: status {status}.")
    if status == TOO_MANY_EVENTS:
        raise SolverError(
            f"More than {MAX_EVENTS} switch events by t = {float(out['t']):g}: a condition keeps "
            "flipping back and forth (the model pushes itself back across it). Replace the "
            "switch by a smooth form, e.g. with sigmoid."
        )
    if status != OK:
        hint = ""
        k = int(out["k_switch"])
        if k >= 0 and float(out["t_switch"]) == float(out["a_fail"]):
            hint = (
                f" The condition '{cm.switch_events[k]}' had just switched at t = "
                f"{float(out['a_fail']):g}; if the model pushes itself straight back across it, "
                "the solution slides along the boundary and no step size works. Replace the "
                "switch by a smooth form (e.g. with sigmoid), or model what happens on the "
                "boundary."
            )
        if not hint:
            hint = dose_hint(doses, float(out["t"]), float(atol_states.min()))
        if status == NOT_FINITE:  # not the solver's fault: no other solver does better
            raise SolverError(
                f"The JAX solver stopped at t = {float(out['t']):g}: {STATUS_TEXT[status]}. "
                "Check the parameter values and the formulas that divide or take logarithms."
            )
        raise SolverError(
            f"The JAX solver failed at t = {float(out['t']):g} ({STATUS_TEXT[status]}).{hint} "
            "The scipy backend retries a failing segment with Radau: try backend='scipy'."
        )
    return doses


TINY = float(np.finfo(np.float64).tiny)  # as the scipy backend: event offsets
MAX_EVENTS = 10_000  # as the scipy backend
TOO_MANY_EVENTS = 4
TOO_MANY_EXTINCTIONS = 5
BAD_ARRIVAL = 6  # a lag or bioavailability below 0 (checked on the host before a solve)
BAD_REGIMEN = 7  # a dose or infusion before the start, or a negative amount, rate or duration
SWITCH_CAPACITY = 8  # more switch times than the slots sized at the scenario's values
EXTINCTION_RECORDS = 256  # extinctions recorded per run (time, group, amounts removed)
NO_STEP = 2**62  # "first step" of an unused segment row: after every step, so sorted


def arrive(cm, theta, geom, measure, b_time, b_index, b_amount, i_start, i_stop, i_index, i_rate):
    """Doses as they arrive (``m.lag``, ``m.bioavailability``): bolus times and infusion
    starts and stops shifted by the lag of their place, amounts and rates scaled by its
    bioavailability, as rises of the state's value; and whether any lag or bioavailability
    is below 0 or not finite. Shared by the program and the replay."""
    lag, frac = cm._arrival_impl(theta, geom)
    b_time = b_time + lag[b_index]
    i_start, i_stop = i_start + lag[i_index], i_stop + lag[i_index]
    rise = b_amount * frac[b_index] / measure[b_index]
    i_rise = i_rate * frac[i_index] / measure[i_index]
    bad = ~jnp.all(jnp.isfinite(lag) & (lag >= 0) & jnp.isfinite(frac) & (frac >= 0))
    return b_time, i_start, i_stop, rise, i_rise, bad


def _program(
    cm: CompiledModel,
    *,
    ledger: bool,
    model_doses: bool,
    max_steps: int,
    record: int = 0,
    steer: bool = False,
):
    """The jitted solve for this model and these options, built once (shared by every model
    with the same structure, like the rest of the build). ``record``: if positive, the run
    is recorded for the replay (step 4.4a), with room for that many steps and segments.

    ``steer`` (step 4.4d, design note §6.3): the program takes ``dot=``, a direction of its
    inputs (``theta``, ``b_time``, ``b_amount``, ``i_start``, ``i_stop``, ``i_rate``), and the
    derivative of the solution along it steers the steps (``bdf``,
    ``sens=``). The derivative is carried across boundaries with the jump a moving dose or
    infusion boundary makes (the right-hand side before minus after, times how fast the
    boundary moves); across an event it is kept as it is (zero for a group zeroed there):
    it only chooses steps, the derivatives themselves come from the replay."""
    if record and ledger:
        raise ValueError("A recorded run has no ledger.")
    if steer and ledger:
        raise ValueError("A steered run has no ledger.")
    key = ("jax_program", ledger, model_doses, max_steps) + ((record,) if record else ())
    key += ("steer",) if steer else ()
    key += ("sparse",) if tier(cm) == "sparse" else ()
    if key in cm._lazy:
        return cm._lazy[key]
    cm.sparse_structure  # noqa: B018 - built here, outside any trace (it is cached on cm)
    jac_of, linear = linear_parts(cm, ledger)
    geom = cm.geom
    rhs = cm._aug_impl if ledger else cm._rhs_impl
    n = cm.n_states
    slot = cm.plan.switch_slot  # conditions on time: constant on a segment (ADR 0006)
    E = len(cm.switch_events)  # state-dependent conditions: events
    groups = cm.extinction_groups  # extinction groups: events too
    G = len(groups)
    members = jnp.asarray(np.array([g.members for g in groups], dtype=bool).reshape(G, n))
    thr = jnp.asarray(np.array([g.threshold for g in groups], dtype=np.float64))
    floor_group = np.array([i for i, g in enumerate(groups) if g.noise_floor], dtype=np.int64)
    floor_slot = np.array(cm.floor_slots, dtype=np.int64)
    names = cm.layout.names
    state_name = np.zeros(n, dtype=np.int64)
    for li, name in enumerate(names):
        state_name[cm.layout.slices[name]] = li

    def program(
        theta,
        t0,
        t1,
        t_out,
        b_time,
        b_index,
        b_amount,
        i_start,
        i_stop,
        i_index,
        i_rate,
        sw_capacity,
        rtol,
        atol,
        max_step,
        y_start=None,
        dot=None,
    ):
        n_all = atol.shape[0]

        def inputs(theta, b_time, b_amount, i_start, i_stop, i_rate):
            measure = cm._measure_impl(theta, geom)
            if model_doses:
                k = b_time.shape[0]
                amounts = cm._dose_impl(theta, geom)
                b_amount = jnp.zeros(k).at[: amounts.shape[0]].set(amounts)
            b_time, i_start, i_stop, rise, i_rise, bad = arrive(
                cm, theta, geom, measure, b_time, b_index, b_amount, i_start, i_stop, i_index,
                i_rate,
            )  # fmt: skip
            y0 = cm._y0_impl(theta, geom)
            if y_start is not None:  # a given starting state (a pre-run's end): fixed
                y0 = jnp.where(jnp.isnan(y_start), y0, y_start)
            return (measure, b_amount, b_time, i_start, i_stop, rise, i_rise, y0), bad

        if model_doses:
            b_amount = jnp.zeros_like(b_time)
        primals = (theta, b_time, b_amount, i_start, i_stop, i_rate)
        if steer:
            names_in = ("theta", "b_time", "b_amount", "i_start", "i_stop", "i_rate")
            tangents = tuple(
                jnp.zeros_like(p) if dot.get(k) is None else dot[k]
                for k, p in zip(names_in, primals, strict=True)
            )
            got, dgot, bad_arrival = jax.jvp(inputs, primals, tangents, has_aux=True)
            theta_dot = tangents[0]
            if slot is not None:  # constant on each segment: not moved here
                theta_dot = theta_dot.at[slot].set(0.0)
            if floor_slot.size:
                theta_dot = theta_dot.at[floor_slot].set(0.0)
            _, _, b_time_dot, i_start_dot, i_stop_dot, rise_dot, i_rise_dot, y0_dot = dgot
        else:
            got, bad_arrival = inputs(*primals)
        measure, b_amount, b_time, i_start, i_stop, rise, i_rise, y0 = got
        z = jnp.concatenate([y0, jnp.zeros(n_all - n)])
        if slot is not None:
            sw_times, sw_over = cm.switch_times_jax(
                theta, t0, t1, sw_capacity.shape[0], overflow=True
            )
        else:
            sw_times, sw_over = jnp.full((1,), jnp.inf), jnp.bool_(False)
        # a regimen moved out of range (sim.function at a new x; solve checks on the host)
        bad_regimen = (
            jnp.any(b_time < t0)
            | jnp.any(i_start < t0)
            | jnp.any(i_stop < i_start)
            | jnp.any(b_amount < 0)
            | jnp.any(i_rate < 0)
        )

        W = measure[None, :] * members  # (G, n): a group's total is W[g] @ y

        def totals(z):
            return W @ z[:n]

        def with_floors(th, armed):  # a floored group counts (no floor) exactly while armed
            if floor_group.size == 0:
                return th
            return th.at[floor_slot].set(jnp.where(armed[floor_group], 0.0, 1.0))

        def extinct(i, te, z, armed, rec, when, zm):
            """Group i to zero (if ``when``), its amount booked as removed; ``zm`` marks the
            groups zeroed."""
            gone = jnp.where(members[i], z[:n] * measure, 0.0)
            by_name = jax.ops.segment_sum(gone, jnp.asarray(state_name), len(names))
            z = jnp.where(when, z.at[:n].set(jnp.where(members[i], 0.0, z[:n])), z)
            armed = jnp.where(when, armed.at[i].set(False), armed)
            k = jnp.minimum(rec["n"], EXTINCTION_RECORDS - 1)
            rec = {
                "t": jnp.where(when, rec["t"].at[k].set(te), rec["t"]),
                "g": jnp.where(when, rec["g"].at[k].set(i), rec["g"]),
                "amount": jnp.where(when, rec["amount"].at[k].set(-by_name), rec["amount"]),
                "n": rec["n"] + jnp.where(when, 1, 0),
            }
            return z, armed, rec, zm.at[i].set(zm[i] | when)

        def apply_doses(z, at):
            hit = b_time == at
            return z.at[b_index].add(jnp.where(hit, rise, 0.0))

        def segment_inputs(a, armed):
            """theta and the infusion input of the segment starting at ``a``."""
            b, _ = next_boundary(a)
            mid = 0.5 * (a + jnp.minimum(b, t1))
            th = theta if slot is None else theta.at[slot].set(mid)
            th = with_floors(th, armed)
            on = (i_start <= mid) & (mid < i_stop)
            return th, jnp.zeros(n_all).at[i_index].add(jnp.where(on, i_rise, 0.0)), on

        def across(at, at_dot, z_before, z_after, s_before, f_before, armed_after):
            """The derivative just after the boundary at ``at`` (moving at ``at_dot``): its
            doses' amounts, and the jump of the right-hand side (before minus after)."""
            hit = b_time == at
            s = s_before.at[b_index].add(jnp.where(hit, rise_dot, 0.0))
            th_a, u_a, _ = segment_inputs(at, armed_after)
            f_after = rhs(at, z_after, th_a, geom) + u_a
            return s + (f_before - f_after) * at_dot

        def next_boundary(a):
            """The next boundary after ``a``, and which candidate it is (for the replay:
            dose times, infusion starts, infusion stops, switch times, then t1)."""
            cands = jnp.concatenate([b_time, i_start, i_stop, sw_times, jnp.array([t1])])
            later = jnp.where(cands > a, cands, jnp.inf)
            k = jnp.argmin(later)
            return later[k], k

        def switch_values(t, z, th):
            return cm._switch_impl(t, z[:n], th, geom)

        def aim(t, z, th):  # watch each switch for the crossing away from where it stands
            v = switch_values(t, z, th)
            return jnp.where(v > 0, -1.0, jnp.where(v < 0, 1.0, 0.0))

        def seg_cond(s):
            return (s["a"] < t1) & (s["status"] == OK)

        def seg_body(s):
            a = s["a"]
            b, b_which = next_boundary(a)
            b = jnp.minimum(b, t1)
            mid = 0.5 * (a + b)
            th_base = theta if slot is None else theta.at[slot].set(mid)
            th = with_floors(th_base, s["armed"])
            armed = s["armed"]
            on = (i_start <= mid) & (mid < i_stop)
            u = jnp.zeros(n_all).at[i_index].add(jnp.where(on, i_rise, 0.0))
            d = jnp.where(s["fresh"], aim(a, s["z"], th), s["d"]) if E else s["d"]

            def fun(t, z):
                return rhs(t, z, th, geom) + u

            def events(t, z):  # groups, then switches, as the scipy backend's watch list
                v = switch_values(t, z, th) if E else jnp.zeros(0)
                sw = jnp.where(d == 0, jnp.abs(v) - TINY, v - d * TINY)
                return jnp.concatenate([totals(z) - thr, sw])

            event_dir = jnp.concatenate([jnp.where(armed, -1.0, 1.0), jnp.where(d == 0, 1.0, d)])

            def jac(t, z):  # the coloured Jacobian (infusions are constant: no part of it)
                return jac_of(t, z, th)

            sens = None
            if steer:
                u_dot = jnp.zeros(n_all).at[i_index].add(jnp.where(on, i_rise_dot, 0.0))

                def fun_s(t, z, s_):  # the derivative of fun along (s_, the direction)
                    return jax.jvp(
                        lambda z_, th_, u_: rhs(t, z_, th_, geom) + u_,
                        (z, th, u),
                        (s_, theta_dot, u_dot),
                    )[1]

                sens = (fun_s, s["zs"])

            r = bdf(
                fun,
                jac,
                a,
                b,
                s["z"],
                t_out,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                max_steps=max_steps,
                ys=s["ys"],
                i_out=s["i_out"],
                out_before=b,
                events=events if (E or G) else None,
                event_dir=event_dir if (E or G) else None,
                record=s["steps"] if record else None,
                rec_base=s["n_steps"],
                skip=~seg_cond(s),
                out_step=s["out_step"] if record else None,
                sens=sens,
                linear=linear,
            )
            ok, at_event = r.status == OK, r.status == EVENT
            rec = s["rec"]
            # a boundary reached: its doses, then groups that reach their threshold are armed
            z_ok = apply_doses(r.y, b)
            armed_ok = armed | (totals(z_ok) >= thr)
            # an event: the group that crossed dies (armed) or is armed; every other group is
            # settled at the same instant (several can cross together); then every switch is
            # watched from where it now stands, the one that switched for the way back
            k = jnp.maximum(r.event, 0)
            z_ev, armed_ev = r.y, armed
            zm = jnp.zeros(G, dtype=bool)
            is_group = at_event & (r.event < G)
            for i in range(G):
                hit = is_group & (k == i)
                z_ev, armed_ev, rec, zm = extinct(i, r.t, z_ev, armed_ev, rec, hit & armed[i], zm)
                armed_ev = jnp.where(hit & ~armed[i], armed_ev.at[i].set(True), armed_ev)
            tot = totals(z_ev)
            for j in range(G):
                other = at_event & ~(is_group & (k == j))
                z_ev, armed_ev, rec, zm = extinct(
                    j, r.t, z_ev, armed_ev, rec, other & armed_ev[j] & (tot[j] <= thr[j]), zm
                )
                armed_ev = jnp.where(
                    other & ~armed_ev[j] & (tot[j] >= thr[j]), armed_ev.at[j].set(True), armed_ev
                )
            if E:
                ks = jnp.maximum(r.event - G, 0)
                d_ev = aim(r.t, z_ev, with_floors(th_base, armed_ev))
                flip = at_event & (r.event >= G) & (d[ks] != 0)
                d_ev = d_ev.at[ks].set(jnp.where(flip, -d[ks], d_ev[ks]))
                d = jnp.where(at_event, d_ev, d)
            # an event exactly at the boundary (its root is the segment's end): the boundary's
            # doses too, which the next segment (starting there) would not see
            dosed = at_event & (r.t >= b)
            z_ev = jnp.where(dosed, apply_doses(z_ev, b), z_ev)
            armed_ev = armed_ev | (dosed & (totals(z_ev) >= thr))
            z_end = jnp.where(ok, z_ok, jnp.where(at_event, z_ev, r.y))
            if steer:
                cands_dot = jnp.concatenate(
                    [b_time_dot, i_start_dot, i_stop_dot, jnp.zeros_like(sw_times), jnp.zeros(1)]
                )
                b_dot = jnp.where(b < t1, cands_dot[b_which], 0.0)
                zs_ok = across(b, b_dot, r.y, z_ok, r.s, fun(b, r.y), armed_ok)
                zs_ev = jnp.where(z_ev == r.y, r.s, 0.0)  # zeroed there: no derivative
                zs_end = jnp.where(ok, zs_ok, jnp.where(at_event, zs_ev, r.s))
            armed = jnp.where(ok, armed_ok, jnp.where(at_event, armed_ev, armed))
            switched = at_event & (r.event >= G)
            extra = {}
            if record:  # this segment, for the replay
                j = jnp.minimum(s["segments"], record)  # past the end: dropped
                put = lambda name, v: s["segs"][name].at[j].set(v, mode="drop")  # noqa: E731
                extra = {
                    "steps": r.record,
                    "out_step": r.out_step,
                    "segs": {
                        "a": put("a", a),
                        "mid": put("mid", mid),
                        "armed": put("armed", s["armed"]),
                        "h0": put("h0", r.h0),
                        "first": put("first", s["n_steps"]),
                        "nsteps": put("nsteps", r.n_steps),
                        "event": put("event", at_event),
                        "t_end": put("t_end", jnp.where(ok, b, r.t)),
                        "zero": put("zero", zm & at_event),
                        "which": put("which", b_which),
                        "ev": put("ev", r.event),
                        "dosed": put("dosed", dosed),
                    },
                }
            n_events = s["n_events"] + jnp.where(at_event, 1, 0)
            status = jnp.where(at_event, OK, r.status)
            status = jnp.where(n_events > MAX_EVENTS, TOO_MANY_EVENTS, status)
            status = jnp.where(rec["n"] > EXTINCTION_RECORDS, TOO_MANY_EXTINCTIONS, status)
            return {
                "a": jnp.where(ok, b, r.t),
                "z": z_end,
                "d": d,
                "armed": armed,
                "rec": rec,
                "fresh": ok,  # a new boundary segment starts: aim again there
                "ys": r.ys,
                "i_out": r.i_out,
                "status": status,
                "a_fail": jnp.where(ok | at_event, s["a_fail"], a),
                "t_switch": jnp.where(switched, r.t, s["t_switch"]),
                "k_switch": jnp.where(switched, r.event - G, s["k_switch"]),
                "n_events": n_events,
                "n_steps": s["n_steps"] + r.n_steps,
                "n_rejected": s["n_rejected"] + r.n_rejected,
                "nfev": s["nfev"] + r.nfev,
                "njev": s["njev"] + r.njev,
                "nlu": s["nlu"] + r.nlu,
                "segments": s["segments"] + 1,
                **extra,
                **({"zs": zs_end} if steer else {}),
            }

        zero = jnp.int64(0)
        z_start = apply_doses(z, t0)
        state = {
            "a": jnp.asarray(t0, dtype=jnp.float64),
            "z": z_start,
            "d": jnp.zeros(E),
            "armed": totals(z_start) >= thr,  # a group at its threshold at the start is armed
            "rec": {
                "t": jnp.full(EXTINCTION_RECORDS, jnp.nan),
                "g": jnp.full(EXTINCTION_RECORDS, -1, dtype=jnp.int64),
                "amount": jnp.zeros((EXTINCTION_RECORDS, len(names))),
                "n": zero,
            },
            "fresh": jnp.bool_(True),
            "ys": jnp.full((t_out.shape[0], n_all), jnp.nan),
            "i_out": zero,
            "status": jnp.select(
                [bad_arrival, bad_regimen, sw_over], [BAD_ARRIVAL, BAD_REGIMEN, SWITCH_CAPACITY], OK
            ).astype(jnp.int64),
            "a_fail": jnp.asarray(jnp.nan),
            "t_switch": jnp.asarray(jnp.nan),
            "k_switch": jnp.int64(-1),
            "n_events": zero,
            "n_steps": zero,
            "n_rejected": zero,
            "nfev": zero,
            "njev": zero,
            "nlu": zero,
            "segments": zero,
        }
        if steer:  # the start: the initial values, then any doses at t0
            th_0, u_0, _ = segment_inputs(t0, state["armed"])
            f_0 = rhs(t0, z, th_0, geom) + u_0
            # a dose at t0 that moves: the start runs undosed until it (f before)
            at_t0 = b_time == t0
            state["zs"] = across(
                t0,
                jnp.where(jnp.any(at_t0), b_time_dot[jnp.argmax(at_t0)], 0.0),
                z,
                z_start,
                jnp.concatenate([y0_dot, jnp.zeros(n_all - n)]),
                f_0,
                state["armed"],
            )
        if record:
            state["steps"] = empty_record(record, n_all)
            state["out_step"] = jnp.full(t_out.shape[0], -1, dtype=jnp.int64)
            state["segs"] = {
                "a": jnp.full(record, jnp.nan),
                "mid": jnp.full(record, jnp.nan),
                "armed": jnp.zeros((record, G), dtype=bool),
                "h0": jnp.ones(record),
                "first": jnp.full(record, NO_STEP, dtype=jnp.int64),
                "nsteps": jnp.zeros(record, dtype=jnp.int64),
                "event": jnp.zeros(record, dtype=bool),
                "t_end": jnp.full(record, jnp.nan),
                "zero": jnp.zeros((record, G), dtype=bool),
                "which": jnp.zeros(record, dtype=jnp.int64),
                "ev": jnp.full(record, -1, dtype=jnp.int64),
                "dosed": jnp.zeros(record, dtype=bool),
            }
        s = (
            while_running(seg_cond, seg_body, state, unkept="steps")
            if record
            else (jax.lax.while_loop(seg_cond, seg_body, state))
        )
        # output times at the end: after the doses there (scipy backend: the same)
        last = (t_out == t1)[:, None] & (jnp.arange(t_out.shape[0]) >= s["i_out"])[:, None]
        ys = jnp.where(last, s["z"], s["ys"])
        return {**s, "ys": ys, "t": s["a"], "b_amount": b_amount}

    jitted = jax.jit(program)
    cm._lazy[key] = jitted
    return jitted
