# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""A variable-order BDF integrator in JAX: a close port of ``scipy.integrate.BDF`` (step 4.3;
design note ``docs/design/step4_3_jax_bdf.md``; ADR 0001 decision 3).

The algorithm is scipy's, so the two backends take the same steps up to round-off (V45):

- orders 1-5 with the NDF coefficients (Shampine & Reichelt 1997, the MATLAB ode15s method);
- quasi-constant step size: the solution history is a difference array ``D`` of shape
  (MAX_ORDER + 3, n); a step-size change rescales it (``_change_D``);
- simplified Newton with the iteration matrix ``I - c J`` factorised once and reused; the
  Jacobian is recomputed only after a Newton failure;
- error control and order selection with scipy's constants.

Everything is one ``lax.while_loop`` over *attempts* (a step may take several: a Newton
failure with a fresh Jacobian, a halved step, an error-test rejection), carrying fixed-shape
arrays, so it compiles once and passes through ``jit`` and ``vmap``. scipy's nested loops
(step -> attempts -> Newton with a refreshed Jacobian) become flags in the carry:
``new_step`` (the next attempt starts a step: clamp h, use the stored Jacobian) and
``retry`` (redo Newton at the same t_new with a fresh Jacobian).

Where scipy's arrays have a length that depends on the order, these use the full
MAX_ORDER-sized arrays with the unused part masked to zero, which gives the same numbers.

One call integrates one span: doses, infusions and the segments between them are the
caller's (``jax_backend``). Within the span it stops at the first event
(``events=``), records its steps for the replay (``record=``), and solves its linear systems
with a pluggable ``LinearSolver`` (dense LU, or the sparse tier's host SuperLU).

``sens=`` (step 4.4d, design note ``docs/design/step4_4d_api.md`` §6.3): the derivative of the
solution along one direction of its parameters is carried too, and steers the steps. It has
its own difference array, rescaled and updated with the state's; after each converged Newton
solve its corrector (linear) is solved with the same factorised matrix, iterated as Newton
is (no convergence: as a Newton failure); its error enters the error test and the order
choice (the larger of the two norms, as CVODES with ``errconS``). So a solution at rest whose
derivative is not cannot take steps too coarse for the derivative.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jax.custom_batching import custom_vmap

MAX_ORDER = 5
NEWTON_MAXITER = 4
MIN_FACTOR = 0.2
MAX_FACTOR = 10.0
EPS = float(np.finfo(np.float64).eps)
FIRST_STEP_FLOOR = 1e-8  # as the scipy backend: relative to the span (V14)

_KAPPA = np.array([0, -0.1850, -1 / 9, -0.0823, -0.0415, 0])
_GAMMA = np.hstack((0, np.cumsum(1 / np.arange(1, MAX_ORDER + 1))))
_ALPHA = (1 - _KAPPA) * _GAMMA
_ERROR_CONST = _KAPPA * _GAMMA + 1 / np.arange(1, MAX_ORDER + 2)

# statuses
OK, TOO_SMALL_STEP, TOO_MANY_STEPS, EVENT = 0, 1, 2, 3
NOT_FINITE = 9  # 4-8 are the JAX backend's (jax_backend.py)
STATUS_TEXT = {
    OK: "reached the end",
    TOO_SMALL_STEP: "required step size is less than spacing between numbers",
    TOO_MANY_STEPS: "too many steps",
    EVENT: "stopped at an event",
    NOT_FINITE: "the rates of change are not finite numbers (NaN or infinity), e.g. from a "
    "division by zero or a parameter value out of range",
}
ROOT_ITERATIONS = 64  # bisection on the step's interpolant: an interval of h shrinks to
# round-off well before this (scipy uses brentq to 4 eps)


@custom_vmap
def _any_member(p):
    """``p`` itself; under ``jax.vmap``, whether it holds for any member (not batched)."""
    return p


@_any_member.def_vmap
def _any_member_vmap(axis_size, in_batched, p):
    return _any_member(jnp.any(p) if in_batched[0] else p), False


def when(need, work: Callable, keep: Callable, *operands):
    """``work(*operands)`` if ``need``, else ``keep(*operands)``: ``jax.lax.cond``, which
    ``jax.vmap`` would turn into computing both for every member (a Jacobian and an LU at
    every attempt). Here, under ``vmap``, ``work`` runs for the whole batch only when some
    member needs it, each member then taking its own result (step 4.6); ``keep`` must be
    cheap."""

    def some(*ops):
        return jax.tree.map(lambda a, b: jnp.where(need, a, b), work(*ops), keep(*ops))

    return jax.lax.cond(_any_member(need), some, keep, *operands)


def _field(c, name):
    return getattr(c, name) if hasattr(c, "_replace") else c[name]


def _set_field(c, name, v):
    return c._replace(**{name: v}) if hasattr(c, "_replace") else {**c, name: v}


def while_running(running: Callable, body: Callable, init, *, unkept: str):
    """``jax.lax.while_loop(running, body, init)``, cheap under ``jax.vmap`` (step 4.6).

    ``vmap`` runs a loop until every member is done, and keeps a done member's carry by
    copying the whole carry at every iteration: a copy of the step record (rows x orders x
    states, hundreds of MB) per step. Here the loop's test is not batched (some member is
    running) and done members keep their carry by the same copy, except the field
    ``unkept`` (the record): ``body`` must leave it unchanged for a member that is not
    running (its writes are dropped). Without ``vmap``, nothing is copied."""

    def guarded(c):
        go = running(c)
        new = body(c)

        def keep_done():
            kept = jax.tree.map(
                lambda a, b: jnp.where(go, a, b),
                _set_field(new, unkept, None),
                _set_field(c, unkept, None),
            )
            return _set_field(kept, unkept, _field(new, unkept))

        return jax.lax.cond(_any_member(~go), keep_done, lambda: new)

    return jax.lax.while_loop(lambda c: _any_member(running(c)), guarded, init)


class LinearSolver(NamedTuple):
    """How the iteration matrix ``I - c J`` is factorised and solved (the dense tier here;
    the sparse tier, step 4.5, in ``jax_sparse``). ``J`` is whatever
    ``jac(t, y)`` returns (a pytree of fixed shape); ``F`` a pytree of fixed shape too.

    - ``factor(J, c) -> F``;
    - ``solve(F, b) -> x`` with ``(I - c J) x = b``;
    - ``empty(J) -> F``: a placeholder of the same shapes (before the first factorisation);
    - ``refactor(F, J, c, need) -> F`` (optional): ``factor(J, c)`` where ``need``, else
      ``F``, for a solver that does better under ``jax.vmap`` than ``when`` (which
      factorises every member when any needs it);
    - ``release(F) -> 0`` (optional): called once when the integration ends, for a solver
      whose factorisations live outside JAX (the sparse tier's host SuperLU). It returns an
      integer 0, added to the status (which every caller reads), so the call is not optimised
      away."""

    factor: Callable
    solve: Callable
    empty: Callable
    refactor: Callable | None = None
    release: Callable | None = None


def _dense_factor(J, c):
    return jsl.lu_factor(jnp.eye(J.shape[0]) - c * J)


DENSE = LinearSolver(
    factor=_dense_factor,
    solve=lambda F, b: jsl.lu_solve(F, b),
    empty=lambda J: jsl.lu_factor(jnp.eye(J.shape[0])),
)


def _rms(x):
    """scipy's ``norm``: the root mean square."""
    return jnp.linalg.norm(x) / np.sqrt(x.size)


def _R(order, factor):
    """scipy's ``compute_R`` at full size, zero outside the (order + 1) block (the block does
    not depend on the size, so it equals scipy's matrix)."""
    i = jnp.arange(1, MAX_ORDER + 1, dtype=jnp.float64)[:, None]
    j = jnp.arange(1, MAX_ORDER + 1, dtype=jnp.float64)[None, :]
    M = jnp.zeros((MAX_ORDER + 1, MAX_ORDER + 1))
    M = M.at[1:, 1:].set((i - 1 - factor * j) / i)
    M = M.at[0].set(1.0)
    R = jnp.cumprod(M, axis=0)
    k = jnp.arange(MAX_ORDER + 1)
    return jnp.where((k[:, None] <= order) & (k[None, :] <= order), R, 0.0)


def _change_D(D, order, factor):
    """Rescale the difference array for a step-size change by ``factor`` (scipy's change_D)."""
    RU = _R(order, factor) @ _R(order, 1.0)
    top = RU.T @ D[: MAX_ORDER + 1]
    keep = (jnp.arange(MAX_ORDER + 1) <= order)[:, None]
    return D.at[: MAX_ORDER + 1].set(jnp.where(keep, top, D[: MAX_ORDER + 1]))


def _rows_upto(order, rows: int):
    return (jnp.arange(rows) <= order)[:, None]


def dense(D, t, h, order, tau):
    """scipy's BdfDenseOutput at time ``tau``, for the step ending at ``t`` (size ``h``)."""
    j = jnp.arange(MAX_ORDER, dtype=jnp.float64)
    x = (tau - (t - h * j)) / (h * (1 + j))
    p = jnp.where(jnp.arange(MAX_ORDER) < order, jnp.cumprod(x), 0.0)
    return D[1 : MAX_ORDER + 1].T @ p + D[0]


def predict(D, order):
    """The predicted state and scipy's ``psi`` for a step at ``order``."""
    y_predict = jnp.sum(jnp.where(_rows_upto(order, MAX_ORDER + 3), D, 0.0), axis=0)
    g = jnp.where(jnp.arange(1, MAX_ORDER + 1) <= order, jnp.asarray(_GAMMA)[1:], 0.0)
    psi = D[1 : MAX_ORDER + 1].T @ g / jnp.asarray(_ALPHA)[order]
    return y_predict, psi


def update(D, d, order):
    """The difference array after a step accepted with Newton correction ``d``."""
    D = D.at[order + 2].set(d - D[order + 1]).at[order + 1].set(d)

    def add(i, Dx):
        k = MAX_ORDER - i
        return jnp.where(k <= order, Dx.at[k].add(Dx[k + 1]), Dx)

    return jax.lax.fori_loop(0, MAX_ORDER + 1, add, D)


class Result(NamedTuple):
    """``ys`` at the output times (rows past the last reached time are NaN), the final ``t``
    and ``y``, a ``status`` (``OK``, ``TOO_SMALL_STEP``, ``TOO_MANY_STEPS``, ``EVENT``) and
    counters."""

    ys: jax.Array
    i_out: jax.Array
    t: jax.Array
    y: jax.Array
    status: jax.Array
    event: jax.Array  # index of the event that stopped the run (status EVENT), else -1
    n_steps: jax.Array
    n_rejected: jax.Array
    nfev: jax.Array
    njev: jax.Array
    nlu: jax.Array
    h0: jax.Array  # the first step size of the run
    record: dict | None  # the step record (``record=``), else None
    out_step: jax.Array | None  # per output row, the index of the step that wrote it
    s: jax.Array | None = None  # the derivative carried (``sens=``) at ``t``, else None


# The step record (step 4.4a): enough to replay a run without error control or Newton.
# For each accepted step: its end time, the order used, the difference array the step
# started from (``D``, after every step-size change since the previous step) and the
# product of those changes (``pre``), the Newton correction ``d`` (y_new - y_predict), and
# the step-size change and order chosen right after it (``grow``, ``order_next``; 1 and
# the same order if none).
RECORD_FIELDS = ("t", "order", "D", "pre", "grow", "order_next", "d")


def empty_record(steps: int, n: int) -> dict:
    """Buffers for ``steps`` accepted steps of an ``n``-state run (unused rows stay put)."""
    return {
        "t": jnp.full(steps, jnp.nan),
        "order": jnp.ones(steps, dtype=jnp.int64),
        "D": jnp.zeros((steps, MAX_ORDER + 3, n)),
        "pre": jnp.ones(steps),
        "grow": jnp.ones(steps),
        "order_next": jnp.ones(steps, dtype=jnp.int64),
        "d": jnp.zeros((steps, n)),
    }


class _Carry(NamedTuple):
    t: jax.Array
    h_abs: jax.Array  # scipy's self.h_abs: the step size to try at the next step
    h_loc: jax.Array  # the step size of the current attempt (scipy's local h_abs)
    t_new: jax.Array  # saved for a retry with a fresh Jacobian
    order: jax.Array
    D: jax.Array
    J: object  # what ``jac`` returns (a dense matrix in the dense tier)
    F: object  # the factorised iteration matrix (``LinearSolver``)
    lu_valid: jax.Array
    n_equal: jax.Array
    new_step: jax.Array
    retry: jax.Array
    current_jac: jax.Array
    ys: jax.Array
    i_out: jax.Array
    status: jax.Array
    done: jax.Array
    n_steps: jax.Array
    n_rejected: jax.Array
    n_attempts: jax.Array
    nfev: jax.Array
    njev: jax.Array
    nlu: jax.Array
    g: jax.Array  # event values at the last accepted state
    event: jax.Array
    t_event: jax.Array
    y_event: jax.Array
    pre: jax.Array  # product of step-size changes since the last accepted step
    record: dict | None
    out_step: jax.Array | None
    Ds: jax.Array  # the difference array of the derivative carried (``sens=``; else 0 wide)
    s_event: jax.Array


def first_step(fun, t0, t1, y0, f0, rtol, atol, max_step, floor=FIRST_STEP_FLOOR):
    """Starting step: scipy's rule (``select_initial_step``, order 1 for BDF), floored at
    ``floor * (t1 - t0)`` as the scipy backend does (V14)."""
    span = t1 - t0
    scale = atol + jnp.abs(y0) * rtol
    d0, d1 = _rms(y0 / scale), _rms(f0 / scale)
    h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)
    h0 = jnp.minimum(h0, span)
    f1 = fun(t0 + h0, y0 + h0 * f0)
    d2 = _rms((f1 - f0) / scale) / h0
    h1 = jnp.where(
        (d1 <= 1e-15) & (d2 <= 1e-15),
        jnp.maximum(1e-6, h0 * 1e-3),
        (0.01 / jnp.maximum(d1, d2)) ** (1 / 2),
    )
    h = jnp.minimum(jnp.minimum(100 * h0, h1), max_step)
    return jnp.minimum(jnp.minimum(jnp.maximum(h, floor * span), span), max_step)


def _newton(fun, t_new, y_predict, c, psi, lu_piv, scale, tol, solve=jsl.lu_solve):
    """scipy's ``solve_bdf_system``: (converged, iterations, y, d, function evaluations)."""
    n = y_predict.shape[0]

    def cond(s):
        k, _, _, _, done, _ = s
        return (~done) & (k < NEWTON_MAXITER)

    def body(s):
        k, y, d, old, _, _ = s
        f = fun(t_new, y)
        finite = jnp.all(jnp.isfinite(f))
        dy = solve(lu_piv, c * f - psi - d)
        dy_norm = _rms(dy / scale)
        has_rate = old >= 0
        rate = jnp.where(has_rate, dy_norm / jnp.where(has_rate, old, 1.0), 0.0)
        left = (NEWTON_MAXITER - k).astype(jnp.float64)
        diverging = has_rate & ((rate >= 1) | (rate**left / (1 - rate) * dy_norm > tol))
        stop = (~finite) | diverging
        converged = (~stop) & ((dy_norm == 0) | (has_rate & (rate / (1 - rate) * dy_norm < tol)))
        y = jnp.where(stop, y, y + dy)
        d = jnp.where(stop, d, d + dy)
        return k + 1, y, d, dy_norm, stop | converged, converged

    init = (jnp.int64(0), y_predict, jnp.zeros(n), jnp.float64(-1.0), False, False)
    k, y, d, _, _, converged = jax.lax.while_loop(cond, body, init)
    return converged, k, y, d


def bdf(
    fun: Callable,
    jac: Callable,
    t0,
    t1,
    y0,
    t_out,
    *,
    rtol,
    atol,
    max_step=jnp.inf,
    h0=None,
    max_steps: int = 100_000,
    ys=None,
    i_out=0,
    out_before=jnp.inf,
    events: Callable | None = None,
    event_dir=None,
    record: dict | None = None,
    rec_base=0,
    out_step=None,
    sens: tuple | None = None,
    sens_out: bool = False,
    linear: LinearSolver = DENSE,
    skip=False,
) -> Result:
    """Integrate ``y' = fun(t, y)`` from ``t0`` to ``t1`` with BDF, returning the state at the
    increasing times ``t_out`` (within [t0, t1]; entries past t1, e.g. +inf padding, stay
    NaN). ``jac(t, y)`` is the Jacobian, in the form ``linear`` factorises (default: dense).
    ``h0``: the first step (default: scipy's rule with the backend's floor). ``atol``: a
    number or one per state. Traceable: call it inside ``jax.jit`` (``fun``/``jac`` closed
    over).

    Segments of a longer run share one output buffer: ``ys`` (rows already written are kept)
    and ``i_out`` (the next row to write); only output times strictly before ``out_before``
    are written (a time on a segment boundary belongs to the next segment, after its
    doses).

    ``events(t, y)`` returns the values of terminal events, watched as ``solve_ivp`` does:
    ``event_dir`` +1 counts upward crossings only, -1 downward only, 0 both. After each
    accepted step the crossings are located on the step's interpolant (bisection to
    round-off); the run stops at the earliest one with status ``EVENT``, returning ``t`` and
    ``y`` there and the event's index; output times at or after it are left for the caller.

    ``record`` (from ``empty_record``): each accepted step is written at row
    ``rec_base`` + its number in this call (rows past the end are dropped; the caller
    compares the step count with the capacity). ``out_step`` (one entry per output time):
    the row of the step whose interpolant gave each output written here.

    ``sens=(fun_s, s0)``: carry the derivative ``s`` of the solution along a direction of its
    parameters, ``s' = fun_s(t, y, s)`` (the derivative of ``fun`` along ``(s, the direction)``)
    from ``s0``, and let its error steer the steps too (module docstring); its value at the
    end is ``Result.s``. ``s0`` may hold several directions side by side (``p x n`` entries),
    each with its own derivative equation (``fun_s`` takes and returns them so), all in the
    error test. ``sens_out``: the output rows hold the derivative at each output time too,
    after the state (``ys`` then has ``n + p x n`` columns).

    ``skip``: take no step (a member of a batch whose own loop is done: nothing written to
    ``record``, see ``while_running``)."""
    y0 = jnp.asarray(y0, dtype=jnp.float64)
    n = y0.shape[0]
    t0 = jnp.asarray(t0, dtype=jnp.float64)
    t1 = jnp.asarray(t1, dtype=jnp.float64)
    t_out = jnp.asarray(t_out, dtype=jnp.float64)
    rtol = jnp.asarray(rtol, dtype=jnp.float64)
    atol = jnp.broadcast_to(jnp.asarray(atol, dtype=jnp.float64), (n,))
    max_step = jnp.asarray(max_step, dtype=jnp.float64)
    newton_tol = jnp.maximum(10 * EPS / rtol, jnp.minimum(0.03, rtol**0.5))
    alpha = jnp.asarray(_ALPHA)
    error_const = jnp.asarray(_ERROR_CONST)

    f0 = fun(t0, y0)
    h_abs = first_step(fun, t0, t1, y0, f0, rtol, atol, max_step) if h0 is None else h0
    h_abs = jnp.asarray(h_abs, dtype=jnp.float64)
    D = jnp.zeros((MAX_ORDER + 3, n)).at[0].set(y0).at[1].set(f0 * h_abs)
    J = jac(t0, y0)
    T = t_out.shape[0]
    if sens is not None:
        fun_s, s0 = sens
        s0 = jnp.asarray(s0, dtype=jnp.float64)
        m = s0.shape[0]  # one direction (n), or several side by side (p x n)
        Ds0 = jnp.zeros((MAX_ORDER + 3, m)).at[0].set(s0).at[1].set(fun_s(t0, y0, s0) * h_abs)
        p = m // n
        atol_s = jnp.tile(atol, p)
        if p == 1:
            solve_s = linear.solve
        elif linear is DENSE:  # every direction in one solve with the same factors

            def solve_s(F, b):
                return jsl.lu_solve(F, b.reshape(p, n).T).T.reshape(-1)
        else:  # each direction with the same factorised matrix

            def solve_s(F, b):
                return jax.vmap(lambda bb: linear.solve(F, bb))(b.reshape(p, n)).reshape(-1)
    else:
        Ds0 = jnp.zeros((MAX_ORDER + 3, 0))
    carry_s = sens is not None
    if events is None:
        events = lambda t, y: jnp.zeros((0,))  # noqa: E731
        event_dir = jnp.zeros((0,))
    event_dir = jnp.asarray(event_dir, dtype=jnp.float64)
    E = event_dir.shape[0]
    g0 = events(t0, y0)

    def body(c: _Carry) -> _Carry:
        # ---- the start of a step: clamp h to [min_step, max_step] (scipy _step_impl)
        min_step = 10 * jnp.abs(jnp.nextafter(c.t, jnp.inf) - c.t)
        too_big = c.h_abs > max_step
        too_small = c.h_abs < min_step
        clamp = c.new_step & (too_big | too_small)
        h_clamped = jnp.where(too_big, max_step, min_step)
        D = jnp.where(
            clamp, _change_D(c.D, c.order, h_clamped / jnp.where(clamp, c.h_abs, 1.0)), c.D
        )
        pre = c.pre * jnp.where(clamp, h_clamped / jnp.where(clamp, c.h_abs, 1.0), 1.0)
        Ds = c.Ds
        if carry_s:
            Ds = jnp.where(
                clamp, _change_D(Ds, c.order, h_clamped / jnp.where(clamp, c.h_abs, 1.0)), Ds
            )
        h = jnp.where(c.new_step, jnp.where(clamp, h_clamped, c.h_abs), c.h_loc)
        n_equal = jnp.where(clamp, 0, c.n_equal)
        current_jac = jnp.where(c.new_step, False, c.current_jac)

        # ---- the attempt
        # a step size that is not a number (rates of change not finite: every comparison is
        # False, so nothing else would end the run)
        not_finite = ~jnp.isfinite(h)
        underflow = ((~c.retry) & (h < min_step)) | not_finite
        t_new = jnp.where(c.retry, c.t_new, c.t + h)
        past = (~c.retry) & (t_new > t1)
        t_new = jnp.where(past, t1, t_new)
        f_past = jnp.abs(t_new - c.t) / h
        D = jnp.where(past, _change_D(D, c.order, f_past), D)
        if carry_s:
            Ds = jnp.where(past, _change_D(Ds, c.order, f_past), Ds)
        pre = pre * jnp.where(past, f_past, 1.0)
        n_equal = jnp.where(past, 0, n_equal)
        lu_valid = c.lu_valid & ~past
        h = jnp.where(c.retry, h, t_new - c.t)

        order = c.order
        y_predict, psi = predict(D, order)
        scale = atol + rtol * jnp.abs(y_predict)
        cc = h / alpha[order]

        need_lu = ~lu_valid & ~c.done  # (a done member: see while_running)
        if linear.refactor is None:
            F = when(need_lu, lambda: linear.factor(c.J, cc), lambda: c.F)
        else:
            F = linear.refactor(c.F, c.J, cc, need_lu)
        nlu = c.nlu + jnp.where(lu_valid, 0, 1)
        converged, n_iter, y_new, d = _newton(
            fun, t_new, y_predict, cc, psi, F, scale, newton_tol, linear.solve
        )
        nfev = c.nfev + n_iter
        if carry_s:  # the derivative's corrector, with the same matrix (module docstring)
            s_predict, psi_s = predict(Ds, order)
            scale_s = atol_s + rtol * jnp.abs(s_predict)
            conv_s, n_iter_s, s_new, ds = when(
                converged,
                lambda: _newton(
                    lambda t, s: fun_s(t, y_new, s),
                    t_new,
                    s_predict,
                    cc,
                    psi_s,
                    F,
                    scale_s,
                    newton_tol,
                    solve_s,
                ),
                lambda: (jnp.bool_(False), jnp.int64(0), s_predict, jnp.zeros_like(s_predict)),
            )
            converged = converged & conv_s
            nfev = nfev + n_iter_s

        # ---- not converged: a fresh Jacobian first, then halve the step
        refresh = (~converged) & (~current_jac)
        halve = (~converged) & current_jac
        J = when(refresh, lambda: jac(t_new, y_predict), lambda: c.J)
        njev = c.njev + jnp.where(refresh, 1, 0)

        # ---- converged: the error test
        safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)
        scale_new = atol + rtol * jnp.abs(y_new)
        error_norm = _rms(error_const[order] * d / scale_new)
        if carry_s:
            scale_s_new = atol_s + rtol * jnp.abs(s_new)
            error_norm = jnp.maximum(error_norm, _rms(error_const[order] * ds / scale_s_new))
        reject = converged & (error_norm > 1)
        accept = converged & ~reject
        reject_factor = jnp.maximum(
            MIN_FACTOR, safety * error_norm ** (-1 / (order + 1).astype(jnp.float64))
        )
        shrink = jnp.where(halve, 0.5, jnp.where(reject, reject_factor, 1.0))
        D = jnp.where(halve | reject, _change_D(D, order, shrink), D)
        if carry_s:
            Ds = jnp.where(halve | reject, _change_D(Ds, order, shrink), Ds)
        h_next = h * shrink
        n_equal = jnp.where(halve | reject, 0, n_equal)
        lu_valid_next = jnp.where(refresh | halve, False, True)  # reject keeps the old LU

        # ---- accepted: update the differences (scipy's "elegant code")
        Dacc = update(D, d, order)
        Dsacc = update(Ds, ds, order) if carry_s else Ds
        n_eq_acc = n_equal + 1
        # order and step size for the next step, once order + 1 equal steps have been taken
        choose = n_eq_acc >= order + 1
        err_m = jnp.where(
            order > 1, _rms(error_const[order - 1] * Dacc[order] / scale_new), jnp.inf
        )
        err_p = jnp.where(
            order < MAX_ORDER,
            _rms(error_const[jnp.minimum(order + 1, MAX_ORDER)] * Dacc[order + 2] / scale_new),
            jnp.inf,
        )
        if carry_s:
            err_m = jnp.maximum(
                err_m,
                jnp.where(
                    order > 1, _rms(error_const[order - 1] * Dsacc[order] / scale_s_new), jnp.inf
                ),
            )
            err_p = jnp.maximum(
                err_p,
                jnp.where(
                    order < MAX_ORDER,
                    _rms(
                        error_const[jnp.minimum(order + 1, MAX_ORDER)]
                        * Dsacc[order + 2]
                        / scale_s_new
                    ),
                    jnp.inf,
                ),
            )
        norms = jnp.stack([err_m, error_norm, err_p])
        powers = -1.0 / (order + jnp.arange(3)).astype(jnp.float64)
        factors = jnp.where(norms == 0, jnp.inf, norms**powers)  # scipy: 0 ** -p = inf
        new_order = order + jnp.argmax(factors) - 1
        grow = jnp.minimum(MAX_FACTOR, safety * jnp.max(factors))
        order_acc = jnp.where(choose, new_order, order)
        Dacc = jnp.where(choose, _change_D(Dacc, order_acc, grow), Dacc)
        if carry_s:
            Dsacc = jnp.where(choose, _change_D(Dsacc, order_acc, grow), Dsacc)
        h_abs_acc = jnp.where(choose, h * grow, h)
        n_eq_acc = jnp.where(choose, 0, n_eq_acc)
        lu_valid_acc = ~choose

        # events crossed in this step (solve_ivp's find_active_events), located on the
        # interpolant as solve_ivp does with the dense output of the step just taken
        g_new = jnp.where(accept, events(t_new, y_new), c.g)
        up = (c.g <= 0) & (g_new >= 0)
        down = (c.g >= 0) & (g_new <= 0)
        active = accept & (
            ((event_dir > 0) & up) | ((event_dir < 0) & down) | ((event_dir == 0) & (up | down))
        )

        def g_at(tau):  # every event at its own time tau[k]
            ys_tau = jax.vmap(lambda tt: dense(Dacc, t_new, h_abs_acc, order_acc, tt))(tau)
            return jax.vmap(lambda tt, yy, k: events(tt, yy)[k])(tau, ys_tau, jnp.arange(E))

        def bisect(_, lohi):
            lo, hi = lohi
            mid = 0.5 * (lo + hi)
            gm = g_at(mid)
            to_hi = (jnp.sign(gm) == jnp.sign(g_new)) | (gm == 0)
            return jnp.where(to_hi, lo, mid), jnp.where(to_hi, mid, hi)

        def locate():
            lo = jnp.full((E,), c.t)
            hi = jnp.full((E,), t_new)
            _, hi = jax.lax.fori_loop(0, ROOT_ITERATIONS, bisect, (lo, hi))
            roots = jnp.where(active, hi, jnp.inf)
            k = jnp.argmin(roots)
            te = roots[k]
            s_te = dense(Dsacc, t_new, h_abs_acc, order_acc, te) if carry_s else c.s_event
            return k, te, dense(Dacc, t_new, h_abs_acc, order_acc, te), s_te

        if E:
            any_event = jnp.any(active)
            ev_k, t_ev, y_ev, s_ev = when(
                any_event,
                locate,
                lambda: (jnp.int64(-1), jnp.asarray(jnp.inf), jnp.zeros(n), c.s_event),
            )
        else:  # no events: nothing to locate (and nothing to index)
            any_event = jnp.bool_(False)
            ev_k, t_ev, y_ev, s_ev = jnp.int64(-1), jnp.asarray(jnp.inf), jnp.zeros(n), c.s_event
        out_limit = jnp.minimum(out_before, t_ev)

        # outputs passed by this step, from the interpolant as scipy builds it after step()
        def out_cond(s):
            i = s[0]
            tau = t_out[jnp.minimum(i, T - 1)]
            return (i < T) & (tau <= t_new) & (tau < out_limit)

        row = rec_base + c.n_steps  # this step's row in the record

        def out_body(s):
            i, ys, os = s
            y_i = dense(Dacc, t_new, h_abs_acc, order_acc, t_out[i])
            if sens_out:
                y_i = jnp.concatenate([y_i, dense(Dsacc, t_new, h_abs_acc, order_acc, t_out[i])])
            ys = ys.at[i].set(y_i)
            return i + 1, ys, None if os is None else os.at[i].set(row)

        i_out, ys, out_step_new = when(
            accept,
            lambda: jax.lax.while_loop(out_cond, out_body, (c.i_out, c.ys, c.out_step)),
            lambda: (c.i_out, c.ys, c.out_step),
        )

        status = jnp.where(underflow, jnp.where(not_finite, NOT_FINITE, TOO_SMALL_STEP), c.status)
        n_steps = c.n_steps + jnp.where(accept & ~underflow, 1, 0)
        n_attempts = c.n_attempts + 1
        acc = accept & ~underflow
        stopped = acc & any_event
        reached = acc & (t_new >= t1)
        # the limit counts steps still needed: a step that ends the run or meets an event is not
        # one too many
        over = (n_steps >= max_steps) & ~stopped & ~reached
        status = jnp.where((status == OK) & over, TOO_MANY_STEPS, status)
        status = jnp.where(stopped & (status == OK), EVENT, status)
        pick = lambda a, b: jnp.where(acc, a, b)  # noqa: E731
        rec = c.record
        if rec is not None:
            # out of range: dropped (a step not taken, or a member done: while_running)
            at = jnp.where(acc & ~c.done, row, rec["t"].shape[0])
            put = lambda name, v: rec[name].at[at].set(v, mode="drop")  # noqa: E731
            rec = {
                "t": put("t", t_new),
                "order": put("order", order),
                "D": put("D", D),
                "pre": put("pre", pre),
                "grow": put("grow", jnp.where(choose, grow, 1.0)),
                "order_next": put("order_next", order_acc),
                "d": put("d", d),
            }
        pre_next = jnp.where(acc, 1.0, pre * shrink)
        return _Carry(
            t=pick(t_new, c.t),
            h_abs=pick(h_abs_acc, c.h_abs),
            h_loc=pick(h_abs_acc, h_next),
            t_new=t_new,
            order=pick(order_acc, order),
            D=jnp.where(acc, Dacc, D),
            J=J,
            F=F,
            lu_valid=pick(lu_valid_acc, lu_valid_next),
            n_equal=pick(n_eq_acc, n_equal),
            new_step=acc,
            retry=refresh & ~underflow,
            current_jac=current_jac | refresh,
            ys=ys,
            i_out=i_out,
            status=status,
            done=reached | (status != OK),
            n_steps=n_steps,
            n_rejected=c.n_rejected + jnp.where(reject | halve, 1, 0),
            n_attempts=n_attempts,
            nfev=nfev,
            njev=njev,
            nlu=nlu,
            g=jnp.where(acc, g_new, c.g),
            event=jnp.where(stopped, ev_k, c.event),
            t_event=jnp.where(stopped, t_ev, c.t_event),
            y_event=jnp.where(stopped, y_ev, c.y_event),
            pre=pre_next,
            record=rec,
            out_step=out_step_new,
            Ds=jnp.where(acc, Dsacc, Ds),
            s_event=jnp.where(stopped, s_ev, c.s_event),
        )

    init = _Carry(
        t=t0,
        h_abs=h_abs,
        h_loc=h_abs,
        t_new=t0,
        order=jnp.int64(1),
        D=D,
        J=J,
        F=linear.empty(J),
        lu_valid=jnp.bool_(False),
        n_equal=jnp.int64(0),
        new_step=jnp.bool_(True),
        retry=jnp.bool_(False),
        current_jac=jnp.bool_(False),
        ys=jnp.full((T, n + (Ds0.shape[1] if sens_out else 0)), jnp.nan) if ys is None else ys,
        i_out=jnp.asarray(i_out, dtype=jnp.int64),
        status=jnp.int64(OK),
        done=jnp.asarray(skip, dtype=bool),
        n_steps=jnp.int64(0),
        n_rejected=jnp.int64(0),
        n_attempts=jnp.int64(0),
        nfev=jnp.int64(1),
        njev=jnp.int64(1),
        nlu=jnp.int64(0),
        g=g0,
        event=jnp.int64(-1),
        t_event=jnp.asarray(jnp.inf),
        y_event=jnp.zeros(n),
        pre=jnp.float64(1.0),
        record=record,
        out_step=None if out_step is None else jnp.asarray(out_step, dtype=jnp.int64),
        Ds=Ds0,
        s_event=jnp.zeros(Ds0.shape[1]),
    )
    c = while_running(lambda c: ~c.done, body, init, unkept="record")
    if linear.release is not None:
        c = c._replace(status=c.status + linear.release(c.F))  # every caller reads status
    at_event = c.status == EVENT
    return Result(
        ys=c.ys,
        i_out=c.i_out,
        t=jnp.where(at_event, c.t_event, c.t),
        y=jnp.where(at_event, c.y_event, c.D[0]),
        status=c.status,
        event=c.event,
        n_steps=c.n_steps,
        n_rejected=c.n_rejected,
        nfev=c.nfev,
        njev=c.njev,
        nlu=c.nlu,
        h0=h_abs,
        record=c.record,
        out_step=c.out_step,
        s=jnp.where(at_event, c.s_event, c.Ds[0]) if carry_s else None,
    )


def fixed_order(fun, jac, D0, t0, h, order: int, n_steps: int, newton_tol: float = 1e-14):
    """``n_steps`` steps of size ``h`` at a fixed ``order`` from the difference array ``D0``
    (D0[j] = the j-th backward difference of y at t0, spacing h), with the same Newton
    solve and difference update as ``bdf`` but no error control. For the order test
    (V45): the global error of order k falls like h^k. Returns y at t0 + n_steps h."""
    D = jnp.zeros((MAX_ORDER + 3, D0.shape[1])).at[: D0.shape[0]].set(D0)
    n = D.shape[1]
    alpha = jnp.asarray(_ALPHA)

    def step(i, D):
        t_new = t0 + (i + 1) * h
        y_predict, psi = predict(D, order)
        cc = h / alpha[order]
        lu = jsl.lu_factor(jnp.eye(n) - cc * jac(t_new, y_predict))
        _, _, _, d = _newton(fun, t_new, y_predict, cc, psi, lu, jnp.ones(n), newton_tol)
        D = D.at[order + 2].set(d - D[order + 1]).at[order + 1].set(d)
        for k in reversed(range(order + 1)):
            D = D.at[k].add(D[k + 1])
        return D

    return jax.lax.fori_loop(0, n_steps, step, D)[0]
