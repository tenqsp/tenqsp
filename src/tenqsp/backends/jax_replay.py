# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Record a JAX-backend run and replay it (step 4.4a; design note
``docs/design/step4_4_gradients.md`` §3).

The JAX BDF decides its own steps inside a ``while_loop``, which reverse-mode differentiation
cannot pass through. So a run is done twice:

1. ``record``: the ordinary JAX program, which also writes down every accepted step
   (its end time, order, the step-size changes applied to the difference array before and
   after it, and its Newton correction) and every segment (where it starts, its first step,
   the theta slots in force, and whether it ended at its boundary or at an event, with the
   extinction groups zeroed there);
2. ``replay``: the same steps as a ``lax.scan`` over the record, with no error control,
   step rejection or Newton iteration; doses, infusions, floors and extinctions are applied
   where the record says. Each step's *value* is taken from the record (the difference
   array it started from), while its *derivative* flows through the replayed arithmetic
   (``recorded``). So the replay's outputs equal the run's to round-off, and nothing is
   propagated that could amplify a difference. (Rebuilding the values from the recorded
   corrections alone is unstable: each step is then an extrapolation from the previous
   ones, which amplifies round-off at orders above 1; seen: 0.4 of 100 after ~400 order-5
   steps. Re-solving each step by Newton instead is stable, but agrees with the run only
   to its Newton tolerance, and growing solutions amplify even that.)

Step 4.4b gives each replayed step the derivative of its implicit equation (the implicit
function theorem, with the same LU), so ``jax.grad`` and ``jax.jvp`` of the replay are the
derivatives of the computed solution.

The record has room for a fixed number of steps (and as many segments), a power of two
remembered per model: a run that needs more is recorded once more at the next size.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from tenqsp.backends import jax_backend as jb
from tenqsp.backends import jax_sparse
from tenqsp.backends.jax_bdf import _ALPHA, MAX_ORDER, _change_D, dense, predict, update, when
from tenqsp.compile import CompiledModel

FIRST_CAPACITY = 1024  # steps (and segments) recorded at the first try
EVENT_NUDGE = 1e-8  # relative: how far an event's state is moved back to read the slope
BLOCK = 32  # record rows replayed (or skipped) together


@dataclasses.dataclass
class Recording:
    """A recorded run: the program's inputs (``run``), the step and segment records, and the
    run's own outputs ``ys`` (one row per requested time)."""

    run: object
    steps: dict
    segs: dict
    out_step: jax.Array
    n_steps: int
    n_segments: int
    capacity: int
    max_steps: int
    ys: np.ndarray


def record(
    cm: CompiledModel,
    t_span: Sequence[float],
    *,
    times: Sequence[float] | None = None,
    params: Mapping[str, object] | None = None,
    theta=None,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_step: float = np.inf,
    regimen=None,
    initial=None,
    max_steps: int = 100_000,
    direction=None,
) -> Recording:
    """Run the model on the JAX backend (no ledger) and record it. Same arguments as
    ``solve``; raises ``SolverError`` as it does.
    ``direction`` (one entry per theta entry): the run's steps are chosen for its derivative
    along it too, not only for the solution (``jax_backend._program``, ``steer``)."""
    why = jb._unsupported(cm)
    if why is not None:
        raise NotImplementedError(
            f"The JAX backend does not yet handle {why}; use backend='scipy' for this model."
        )
    run = jb._prepare(
        cm, t_span, times, params, theta, "BDF", rtol, atol, max_step, False, regimen, initial
    )
    capacity = cm._lazy.get("record_capacity", FIRST_CAPACITY)
    while True:
        program = jb._program(
            cm,
            ledger=False,
            model_doses=run.model_doses,
            max_steps=max_steps,
            record=capacity,
            steer=direction is not None,
        )
        extra = {}
        if direction is not None:
            direction = jnp.asarray(direction, dtype=jnp.float64)
            if direction.shape != run.args["theta"].shape:
                raise ValueError(f"direction has one entry per theta entry ({run.th.size}).")
            extra = {"dot": {"theta": direction}}
        out = jax.block_until_ready(program(**run.args, **extra))
        jb._check(cm, out, run)
        n_steps, n_segments = int(out["n_steps"]), int(out["segments"])
        if max(n_steps, n_segments) <= capacity:
            break
        capacity = jb.slots(max(n_steps, n_segments))  # once more, with room for this run
    cm._lazy["record_capacity"] = max(capacity, cm._lazy.get("record_capacity", 0))
    return Recording(
        run=run,
        steps=out["steps"],
        segs=out["segs"],
        out_step=out["out_step"],
        n_steps=n_steps,
        n_segments=n_segments,
        capacity=capacity,
        max_steps=max_steps,
        ys=np.asarray(out["ys"])[: run.t_out.size],
    )


def replay(
    cm: CompiledModel,
    rec: Recording,
    theta=None,
    *,
    dose_times=None,
    dose_amounts=None,
    infusion_starts=None,
    infusion_stops=None,
    infusion_rates=None,
) -> jax.Array:
    """The recorded run's outputs, recomputed from the record: one row per requested time,
    shape (n_t, n_states). ``theta`` (default: the recorded run's) may be traced, and so may
    the regimen's numbers (one per bolus or infusion, in the regimen's order; default: the
    recorded run's). Their values must be the recorded ones: the replay gives derivatives
    at the recorded run, not a new run."""
    th = jnp.asarray(rec.run.th if theta is None else theta, dtype=jnp.float64)
    a = rec.run.args

    def slots(name, given, fill):
        if given is None:
            return a[name]
        given = jnp.asarray(given, dtype=jnp.float64)
        k = a[name].shape[0] - given.shape[0]
        return jnp.concatenate([given, jnp.full(k, fill)])

    if dose_amounts is not None and rec.run.model_doses:
        raise ValueError("The model's own doses take their amounts from the parameters.")
    program = _replay_program(cm, rec.run.model_doses)
    ys = program(
        th,
        rec.steps,
        rec.segs,
        rec.out_step,
        jnp.int64(rec.n_steps),
        a["t0"],
        a["t1"],
        a["t_out"],
        slots("b_time", dose_times, jnp.inf),
        a["b_index"],
        None if rec.run.model_doses else slots("b_amount", dose_amounts, 0.0),
        slots("i_start", infusion_starts, jnp.inf),
        slots("i_stop", infusion_stops, jnp.inf),
        a["i_index"],
        slots("i_rate", infusion_rates, 0.0),
        a["sw_capacity"],
        a.get("y_start"),
    )
    return ys[: rec.run.t_out.size]


def recorded(value, computed):
    """``value`` (from the record), with the derivative of ``computed`` (the replayed
    arithmetic that would produce it)."""
    return value + (computed - jax.lax.stop_gradient(computed))


def _replay_program(cm: CompiledModel, model_doses: bool):
    sparse = jb.tier(cm) == "sparse"
    key = ("jax_replay", model_doses) + (("sparse",) if sparse else ())
    if key in cm._lazy:
        return cm._lazy[key]
    cm.sparse_structure  # noqa: B018 - built here, outside any trace (it is cached on cm)
    geom = cm.geom
    n = cm.n_states
    slot = cm.plan.switch_slot
    groups = cm.extinction_groups
    G = len(groups)
    # numpy, not JAX, constants: the program may first be built inside someone's trace
    members = np.array([g.members for g in groups], dtype=bool).reshape(G, n)
    thr = np.array([g.threshold for g in groups], dtype=np.float64)
    E = len(cm.switch_events)
    floor_group = np.array([i for i, g in enumerate(groups) if g.noise_floor], dtype=np.int64)
    floor_slot = np.array(cm.floor_slots, dtype=np.int64)

    def program(
        theta,
        steps,
        segs,
        out_step,
        n_total,
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
        y_start=None,
    ):
        measure = cm._measure_impl(theta, geom)
        if model_doses:  # as the program: the model's doses are formulas of theta
            amounts = cm._dose_impl(theta, geom)
            b_amount = jnp.zeros(b_time.shape[0]).at[: amounts.shape[0]].set(amounts)
        # as they arrive (m.lag, m.bioavailability): the lag moves the boundaries below
        b_time, i_start, i_stop, rise, i_rise, _ = jb.arrive(
            cm, theta, geom, measure, b_time, b_index, b_amount, i_start, i_stop, i_index, i_rate
        )

        def apply_doses(z, at):
            return z.at[b_index].add(jnp.where(b_time == at, rise, 0.0))

        # the boundaries a segment can end at, in the program's order (step 4.4c: they move
        # with theta and the regimen, and the segments stretch with them)
        sw = (
            cm.switch_times_jax(theta, t0, t1, sw_capacity.shape[0])
            if slot is not None
            else jnp.full((1,), jnp.inf)
        )
        cands = jnp.concatenate([b_time, i_start, i_stop, sw, jnp.atleast_1d(t1)])
        W = measure[None, :] * members

        def event_value(k, t, y, th):  # the program's events: group totals, then switches
            v = W @ y - thr
            if E:
                v = jnp.concatenate([v, cm._switch_impl(t, y, th, geom)])
            return v[jnp.maximum(k, 0)]

        def segment(k):  # theta, infusion input and start of segment k
            mid = segs["mid"][k]
            th = theta if slot is None else theta.at[slot].set(mid)
            if floor_group.size:
                th = th.at[floor_slot].set(jnp.where(segs["armed"][k][floor_group], 0.0, 1.0))
            on = (i_start <= mid) & (mid < i_stop)
            u = jnp.zeros(n).at[i_index].add(jnp.where(on, i_rise, 0.0))
            return th, u

        @jax.custom_jvp
        def correction(d, t_new, y_predict, psi, cc, th, u):
            """The step's Newton correction: the recorded value ``d``, with the derivative
            of the solution of its implicit equation
                G = cc (f(t_new, y_predict + d) + u) - psi - d = 0
            (implicit function theorem: (I - cc J) d' = the derivative of G with d held)."""
            return d

        @correction.defjvp
        def correction_jvp(primals, tangents):
            d, t_new, y_predict, psi, cc, th, u = primals

            def G(t_new, y_predict, psi, cc, th, u):
                return cc * (cm._rhs_impl(t_new, y_predict + d, th, geom) + u) - psi - d

            _, dG = jax.jvp(G, primals[1:], tangents[1:])
            y_new = y_predict + d
            if sparse:  # SuperLU on the host (step 4.5), transposed in reverse mode
                pieces = cm.jacobian_pieces(t_new, y_new, th, ledger=False)

                def matvec(x):
                    jx = jax.jvp(lambda yy: cm._rhs_impl(t_new, yy, th, geom), (y_new,), (x,))[1]
                    return x - cc * jx

                return d, jax_sparse.solve_step(cm, pieces, cc, dG, matvec)
            J = cm.jacobian_jax(t_new, y_new, th, ledger=False)
            lu = jsl.lu_factor(jnp.eye(n) - cc * J)
            return d, jsl.lu_solve(lu, dG)

        S = steps["t"].shape[0]
        seg_of = jnp.searchsorted(segs["first"], jnp.arange(S), side="right") - 1

        def body(c, x):
            i, t_rec, order, D_rec, pre, grow, order_next, d = x
            D, t, a, z_next, a_next = c
            k = jnp.maximum(seg_of[i], 0)
            th, u = segment(k)
            first = i == segs["first"][k]
            last = i == segs["first"][k] + segs["nsteps"][k] - 1
            # where the segment starts and ends. A segment ending at a boundary stretches its
            # steps between its (moving) start and end; one ending at an event keeps its
            # steps' sizes, shifted with its start, and the event time moves on its own
            a_rec, e_rec, at_event = segs["a"][k], segs["t_end"][k], segs["event"][k]
            a = jnp.where(first, recorded(a_rec, a_next), a)
            e_bound = recorded(e_rec, cands[segs["which"][k]])
            stretch = jnp.where(at_event, 1.0, (e_bound - a) / (e_rec - a_rec))
            t_new = recorded(t_rec, a + (t_rec - a_rec) * stretch)
            # a segment starts afresh: order 1, the recorded first step (scipy's start)
            h0 = recorded(segs["h0"][k], segs["h0"][k] * stretch)
            f0 = cm._rhs_impl(a, z_next, th, geom) + u
            D0 = jnp.zeros_like(D).at[0].set(z_next).at[1].set(f0 * h0)
            D = jnp.where(first, D0, D)
            t = jnp.where(first, a, t)
            # the step, with the step-size changes the run made before it; its value is the
            # recorded one (the run's, exactly), its derivative that of the replayed
            # arithmetic (``recorded``)
            D = recorded(D_rec, jnp.where(pre != 1.0, _change_D(D, order, pre), D))
            h = t_new - t
            y_predict, psi = predict(D, order)
            d = correction(d, t_new, y_predict, psi, h / jnp.asarray(_ALPHA)[order], th, u)
            Dn = update(D, d, order)
            Dn = jnp.where(grow != 1.0, _change_D(Dn, order_next, grow), Dn)
            h_abs = h * grow
            # the end of a segment: its boundary (doses there) or an event (groups zeroed).
            # An event's time t* is where the step's interpolant crosses; it moves by
            # dt* = -(dg with t held) / (dg/dt), and the state there by the slope of the
            # solution just before t* times dt*. That slope is the right-hand side on the
            # side the step came from. Neither the interpolant (the step that found the
            # event straddles the kink: seen -0.65 for a true -1 at a rate switch) nor the
            # state at t* (it can sit a round-off past the boundary) gives it; so the state
            # is nudged back across, by 1e-8 relative along the event function's gradient,
            # towards the sign it had at the step's start.
            # The interpolant is read at the recorded event time moved with the segment's
            # start (its steps are shifted with it): so the shift cancels inside the
            # interpolant, whose slope on this step blends both sides of the kink, and
            # enters only through the slope above (an event after an event or a dose that
            # moves: seen 0.77 of the true derivative of the second event time otherwise).
            tau = jax.lax.stop_gradient(e_rec) + (a - jax.lax.stop_gradient(a))
            y_tau = dense(Dn, t_new, h_abs, order_next, tau)
            if G or E:
                ev = segs["ev"][k]
                # Only the last step of a segment that ends at an event uses what follows. On
                # every other step tau lies beyond the step, and y_tau extrapolates its
                # polynomial there (a concentration can come out negative, and C^1.5 is then
                # NaN): discarded by ``where`` below, but a NaN there still makes the
                # reverse-mode derivative NaN (0 x NaN). Those steps read their own end.
                use = at_event & last
                tau = jnp.where(use, tau, t_new)
                y_tau = jnp.where(use, y_tau, Dn[0])
                y_sg = jax.lax.stop_gradient(y_tau)
                grad_g = jax.grad(lambda yy: event_value(ev, tau, yy, th))(y_sg)
                g_start = event_value(ev, t, D[0], th)
                side = jnp.where(g_start < 0, -1.0, 1.0)
                norm = jnp.sqrt(jnp.sum(grad_g**2))
                nudge = EVENT_NUDGE * jnp.max(jnp.abs(y_sg)) / jnp.where(norm == 0, 1.0, norm)
                y_left = y_sg + side * nudge * grad_g
                slope = jax.lax.stop_gradient(cm._rhs_impl(tau, y_left, th, geom) + u)
                _, rate = jax.jvp(
                    lambda tt, yy: event_value(ev, tt, yy, th),
                    (tau, y_sg),
                    (jnp.ones_like(tau), slope),
                )
                rate = jax.lax.stop_gradient(rate)
                g_now = event_value(ev, tau, y_tau, th)
                shift = -g_now / jnp.where(rate == 0, 1.0, rate)
                e_event = tau + (shift - jax.lax.stop_gradient(shift))
                y_event = y_tau + slope * (e_event - tau)
                e = jnp.where(at_event, e_event, e_bound)
            else:
                y_event = y_tau
                e = e_bound
            y_end = jnp.where(at_event, y_event, Dn[0])
            zeroed = jnp.any(segs["zero"][k][:, None] & members, axis=0) if G else False
            z_ev = jnp.where(zeroed, 0.0, y_end)
            z_ev = jnp.where(segs["dosed"][k], apply_doses(z_ev, e_bound), z_ev)  # at its boundary
            z_end = jnp.where(at_event, z_ev, apply_doses(y_end, e))
            valid = i < n_total
            z_next = jnp.where(valid & last, z_end, z_next)
            a_next = jnp.where(valid & last, e, a_next)
            new = (jnp.where(valid, Dn, D), jnp.where(valid, t_new, t), a, z_next, a_next)
            return new, (Dn, t_new, h_abs, order_next)

        y0 = cm._y0_impl(theta, geom)
        if y_start is not None:  # a given starting state: fixed, as in the run
            y0 = jnp.where(jnp.isnan(y_start), y0, y_start)
        z0 = apply_doses(y0, t0)
        # a dose or infusion start at t0 that moves (a dose time, a lag from 0): the run starts
        # undosed until it, so the state just after t0 moves by (f before - f after) per unit
        # of its time; one-sided (later), since earlier than t0 is not a run
        th0, _ = segment(0)

        def moving(t):  # 0, with the derivative of t, where t is at t0
            at = t == t0
            return jnp.where(at, t, t0) - jax.lax.stop_gradient(jnp.where(at, t, t0))

        def without(k):  # only a dose at t0 is taken back (a later one would leave a negative
            # state, where C^1.5 is NaN, and 0 x NaN is NaN even in the value)
            back = jnp.where(b_time[k] == t0, rise[k], 0.0)
            return cm._rhs_impl(t0, z0.at[b_index[k]].add(-back), th0, geom)

        f_after = cm._rhs_impl(t0, z0, th0, geom)
        f_before = jax.vmap(without)(jnp.arange(b_time.shape[0]))  # each dose not yet given
        z0 = z0 + jnp.sum((f_before - f_after) * moving(b_time)[:, None], axis=0)
        z0 = z0.at[i_index].add(-i_rise * moving(i_start))
        D_init = jnp.zeros((MAX_ORDER + 3, n))
        # unused rows (NaN times) feed computations that ``where`` then discards; in reverse
        # mode their zero cotangents still meet the NaNs (0 x NaN), so they get the first
        # step's numbers instead (row by row, in the scan: a copy of the whole record would
        # double its memory)
        first_row = jax.tree.map(lambda v: v[0], steps)
        xs = (
            jnp.arange(S),
            steps["t"],
            steps["order"],
            steps["D"],
            steps["pre"],
            steps["grow"],
            steps["order_next"],
            steps["d"],
        )

        def body_or_skip(c, x):
            """The record has room for more steps than the run took: rows past its last step
            are skipped (no Jacobian, no factorisation), not replayed and discarded."""

            def skip(c, x):
                D, t = c[0], c[1]
                return c, (D, t, jnp.ones_like(t), x[2])

            used = x[0] < n_total
            row0 = (
                x[0],
                first_row["t"],
                first_row["order"],
                first_row["D"],
                first_row["pre"],
                first_row["grow"],
                first_row["order_next"],
                first_row["d"],
            )
            x = jax.tree.map(lambda v, v0: jnp.where(used, v, v0), x, row0)
            return when(used, body, skip, c, x)

        # checkpointed: reverse mode recomputes a step's Jacobian and LU rather than keeping
        # an n x n factor per step
        step = jax.checkpoint(body_or_skip)
        B = min(BLOCK, S)  # S is a multiple of BLOCK (or smaller than it)

        def block(c, xb):
            """BLOCK rows of the record. A block past the run's last step is skipped whole
            (under ``vmap``: past every member's), so unused room in the record costs one
            test per block instead of a pass through every row, forward and in reverse."""

            def skip(c, xb):
                D, t = c[0], c[1]
                return c, (
                    jnp.broadcast_to(D, (B, *D.shape)),
                    jnp.broadcast_to(t, (B,)),
                    jnp.ones(B),
                    xb[2],
                )

            return when(xb[0][0] < n_total, lambda c, xb: jax.lax.scan(step, c, xb), skip, c, xb)

        t0 = jnp.asarray(t0, dtype=jnp.float64)
        init = (D_init, t0, t0, z0, t0)
        xs_b = jax.tree.map(lambda v: v.reshape(S // B, B, *v.shape[1:]), xs)
        (_, _, _, z_final, _), out_b = jax.lax.scan(block, init, xs_b)
        Ds, ts, hs, os = jax.tree.map(lambda v: v.reshape(S, *v.shape[2:]), out_b)

        def output(j):
            k = jnp.maximum(out_step[j], 0)
            tau = jnp.where(out_step[j] >= 0, t_out[j], ts[k])  # padded times are +inf
            inside = dense(Ds[k], ts[k], hs[k], os[k], tau)
            return jnp.where(out_step[j] >= 0, inside, z_final)

        return jax.vmap(output)(jnp.arange(t_out.shape[0]))

    jitted = jax.jit(program)
    cm._lazy[key] = jitted
    return jitted
