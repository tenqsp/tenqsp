# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The native solver (ADR 0011): a port of ``jax_bdf.bdf`` (scipy's BDF with TenQSP's forward
sensitivities) and of the JAX program's segment loop, for numba.

This file is a template: ``rhs(t, z, th)``, ``jac(t, z, th)`` and
``sens(t, z, th, S, TD)`` (the derivative of ``rhs`` along each row of ``(S, TD)``) are the
model's kernels, which ``program.py`` writes in front of this source, one module per model
structure, so that numba caches the compiled solver on disk with them. Nothing here is
imported on its own.

The steps are those of ``jax_bdf.bdf`` up to round-off: the same predictor, Newton iteration,
error constants, sensitivity corrector (solved with the same factorised matrix, iterated as
Newton is) and error scale (``atol + rtol |s| + SENS_FLOOR max|s|`` per direction), order and
step-size choice. Not ported: events, the step record, stacked units.
"""

import numpy as np
from numba import njit

MAX_ORDER = 5
NEWTON_MAXITER = 4
MIN_FACTOR = 0.2
MAX_FACTOR = 10.0
EPS = 2.220446049250313e-16
TINY = 2.2250738585072014e-308
FIRST_STEP_FLOOR = 1e-8
SENS_FLOOR = 1e-12
OK, TOO_SMALL_STEP, TOO_MANY_STEPS, NOT_FINITE = 0, 1, 2, 9
BAD_REGIMEN = 6  # as jax_backend

_KAPPA = np.array([0.0, -0.1850, -1 / 9, -0.0823, -0.0415, 0.0])
_GAMMA = np.zeros(MAX_ORDER + 1)
for _k in range(1, MAX_ORDER + 1):
    _GAMMA[_k] = _GAMMA[_k - 1] + 1.0 / _k
_ALPHA = (1 - _KAPPA) * _GAMMA
_ERROR_CONST = _KAPPA * _GAMMA + 1.0 / np.arange(1, MAX_ORDER + 2)


@njit(cache=True, error_model="numpy")
def _rms(x):
    s = 0.0
    for v in x.ravel():
        s += v * v
    return np.sqrt(s / max(x.size, 1))


@njit(cache=True, error_model="numpy")
def _lu_factor(A):
    n = A.shape[0]
    LU = A.copy()
    piv = np.arange(n)
    for k in range(n):
        p = k
        m = abs(LU[k, k])
        for i in range(k + 1, n):
            if abs(LU[i, k]) > m:
                m = abs(LU[i, k])
                p = i
        if p != k:
            for j in range(n):
                tmp = LU[k, j]
                LU[k, j] = LU[p, j]
                LU[p, j] = tmp
            tp = piv[k]
            piv[k] = piv[p]
            piv[p] = tp
        d = LU[k, k]
        if d == 0.0:
            continue
        for i in range(k + 1, n):
            LU[i, k] /= d
            f = LU[i, k]
            if f != 0.0:
                for j in range(k + 1, n):
                    LU[i, j] -= f * LU[k, j]
    return LU, piv


@njit(cache=True, error_model="numpy")
def _lu_solve(LU, piv, b):
    n = b.shape[0]
    x = np.empty(n)
    for i in range(n):
        x[i] = b[piv[i]]
    for i in range(n):
        s = x[i]
        for j in range(i):
            s -= LU[i, j] * x[j]
        x[i] = s
    for i in range(n - 1, -1, -1):
        s = x[i]
        for j in range(i + 1, n):
            s -= LU[i, j] * x[j]
        x[i] = s / LU[i, i]
    return x


@njit(cache=True, error_model="numpy")
def _R_into(R, order, factor):
    """scipy's compute_R at full size, zero outside the (order + 1) block, into ``R``."""
    for j in range(MAX_ORDER + 1):
        R[0, j] = 1.0 if j <= order else 0.0
    for i in range(1, MAX_ORDER + 1):
        for j in range(MAX_ORDER + 1):
            if i > order or j > order:
                R[i, j] = 0.0
            elif j == 0:
                R[i, j] = R[i - 1, j] * 0.0 if i > 0 else 1.0
            else:
                R[i, j] = R[i - 1, j] * ((i - 1 - factor * j) / i)


def _u_table():
    U = np.zeros((MAX_ORDER + 1, MAX_ORDER + 1, MAX_ORDER + 1))
    for order in range(MAX_ORDER + 1):
        for j in range(MAX_ORDER + 1):
            U[order, 0, j] = 1.0 if j <= order else 0.0
        for i in range(1, MAX_ORDER + 1):
            for j in range(MAX_ORDER + 1):
                if i > order or j > order:
                    U[order, i, j] = 0.0
                elif j == 0:
                    U[order, i, j] = 0.0
                else:
                    U[order, i, j] = U[order, i - 1, j] * ((i - 1 - 1.0 * j) / i)
    return U


_U = _u_table()  # R(order, 1), one per order: a constant of the change_D product


@njit(cache=True, error_model="numpy")
def _change_D(D, order, factor, W):
    """scipy's change_D, in place (D: (MAX_ORDER + 3, m)); ``W``: workspace with ``R``,
    ``RU`` ((MAX_ORDER + 1)^2 each) and rows for the new top block."""
    R = W[0]
    RU = W[1]
    _R_into(R, order, factor)
    U = _U[order]
    for i in range(order + 1):
        for j in range(order + 1):
            acc = 0.0
            for k in range(order + 1):
                acc += R[i, k] * U[k, j]
            RU[i, j] = acc
    m = D.shape[1]
    top = W[2]
    for i in range(order + 1):
        for j in range(m):
            acc = 0.0
            for k in range(order + 1):
                acc += RU[k, i] * D[k, j]
            top[i, j] = acc
    for i in range(order + 1):
        for j in range(m):
            D[i, j] = top[i, j]


@njit(cache=True, error_model="numpy")
def _predict(D, order, yp, psi):
    m = D.shape[1]
    a = _ALPHA[order]
    for j in range(m):
        acc = D[0, j]
        ps = 0.0
        for i in range(1, order + 1):
            acc += D[i, j]
            ps += D[i, j] * _GAMMA[i]
        yp[j] = acc
        psi[j] = ps / a


@njit(cache=True, error_model="numpy")
def _update(D, d, order):
    m = D.shape[1]
    for j in range(m):
        D[order + 2, j] = d[j] - D[order + 1, j]
        D[order + 1, j] = d[j]
    for k in range(order, -1, -1):
        for j in range(m):
            D[k, j] += D[k + 1, j]


@njit(cache=True, error_model="numpy")
def _dense_into(out, D, t, h, order, tau):
    m = D.shape[1]
    for i in range(m):
        out[i] = D[0, i]
    p = 1.0
    for j in range(order):
        p *= (tau - (t - h * j)) / (h * (1 + j))
        for i in range(m):
            out[i] += D[j + 1, i] * p


@njit(cache=True, error_model="numpy")
def _rms_ratio(x, scale, c):
    """rms(c * x / scale), with no array made."""
    acc = 0.0
    for i in range(x.size):
        v = c * x[i] / scale[i]
        acc += v * v
    return np.sqrt(acc / max(x.size, 1))


@njit(cache=True, error_model="numpy")
def _state_scale(scale, atol, rtol, y):
    for i in range(y.size):
        scale[i] = atol[i] + rtol * abs(y[i])


@njit(cache=True, error_model="numpy")
def _sens_scale(scale, atol, rtol, s, P, n):
    for k in range(P):
        big = 0.0
        for i in range(n):
            v = abs(s[k * n + i])
            if v > big:
                big = v
        for i in range(n):
            scale[k * n + i] = atol[i] + rtol * abs(s[k * n + i]) + SENS_FLOOR * big


@njit(cache=True, error_model="numpy")
def _fun(t, z, th, u):
    return rhs(t, z, th)[0] + u  # noqa: F821 - the model's kernel (module docstring)


@njit(cache=True, error_model="numpy")
def _fun_s(t, z, th, s, TD, UD, P, n):
    S = s.reshape(P, n)
    out = sens(t, z, th, S, TD)[0] + UD  # noqa: F821 - the model's kernel
    return np.ascontiguousarray(out).ravel()


@njit(cache=True, error_model="numpy")
def _first_step(t0, t1, y0, f0, th, u, rtol, atol, max_step):
    span = t1 - t0
    scale = atol + np.abs(y0) * rtol
    d0, d1 = _rms_ratio(y0, scale, 1.0), _rms_ratio(f0, scale, 1.0)
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    h0 = min(h0, span)
    f1 = _fun(t0 + h0, y0 + h0 * f0, th, u)
    d2 = _rms_ratio(f1 - f0, scale, 1.0) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** 0.5
    h = min(min(100 * h0, h1), max_step)
    return min(min(max(h, FIRST_STEP_FLOOR * span), span), max_step)


@njit(cache=True, error_model="numpy")
def _lu_solve_into(LU, piv, b, x, off, n):
    """x[off:off+n] = solve(LU, b[off:off+n])."""
    for i in range(n):
        x[off + i] = b[off + piv[i]]
    for i in range(n):
        acc = x[off + i]
        for j in range(i):
            acc -= LU[i, j] * x[off + j]
        x[off + i] = acc
    for i in range(n - 1, -1, -1):
        acc = x[off + i]
        for j in range(i + 1, n):
            acc -= LU[i, j] * x[off + j]
        x[off + i] = acc / LU[i, i]


@njit(cache=True, error_model="numpy")
def _newton(t_new, yp, c, psi, LU, piv, scale, tol, th, u, y, d, rhs_buf, dy,
            sens_mode, z, TD, UD, P, n):  # fmt: skip
    """scipy's solve_bdf_system, in place: y (from yp) and d; the state (``sens_mode``
    False) or the derivative's corrector at the state ``z`` (True, P directions). Returns
    (converged, iterations)."""
    m = yp.size
    for i in range(m):
        y[i] = yp[i]
        d[i] = 0.0
    old = -1.0
    k = 0
    while k < NEWTON_MAXITER:
        if sens_mode:
            f = _fun_s(t_new, z, th, y, TD, UD, P, n)
        else:
            f = _fun(t_new, y, th, u)
        k += 1
        finite = True
        for i in range(m):
            v = c * f[i] - psi[i] - d[i]
            if not np.isfinite(v):
                finite = False
            rhs_buf[i] = v
        if not finite:
            return False, k
        for q in range(m // n):
            _lu_solve_into(LU, piv, rhs_buf, dy, q * n, n)
        dn = _rms_ratio(dy[:m], scale, 1.0)
        rate = dn / old if old >= 0 else 0.0
        if old >= 0 and (rate >= 1 or rate ** (NEWTON_MAXITER - k + 1) / (1 - rate) * dn > tol):
            return False, k
        for i in range(m):
            y[i] += dy[i]
            d[i] += dy[i]
        if dn == 0 or (old >= 0 and rate / (1 - rate) * dn < tol):
            return True, k
        old = dn
    return False, k


@njit(cache=True, error_model="numpy")
def _bdf(t0, t1, y0, s0, th, u, TD, UD, P, t_out, ys, ss, i_out, out_before, rtol, atol,
         max_step, max_steps):  # fmt: skip
    """One segment, as jax_bdf.bdf without events or record; writes rows of ``ys``/``ss``
    from ``i_out`` for output times before ``out_before``. Returns (status, i_out, y, s,
    steps)."""
    n = y0.size
    m = s0.size
    T = t_out.size
    newton_tol = max(10 * EPS / rtol, min(0.03, rtol**0.5))
    f0 = _fun(t0, y0, th, u)
    h_abs = _first_step(t0, t1, y0, f0, th, u, rtol, atol, max_step)
    D = np.zeros((MAX_ORDER + 3, n))
    D[0] = y0
    D[1] = f0 * h_abs
    Ds = np.zeros((MAX_ORDER + 3, m))
    if P:
        Ds[0] = s0
        Ds[1] = _fun_s(t0, y0, th, s0, TD, UD, P, n) * h_abs
    W = np.zeros((3, MAX_ORDER + 1, max(MAX_ORDER + 1, n, m)))
    Wn = W[:, :, :n] if False else W
    # workspaces
    yp = np.empty(n)
    psi = np.empty(n)
    y_new = np.empty(n)
    d = np.empty(n)
    rb = np.empty(max(n, m))
    dy = np.empty(max(n, m))
    scale = np.empty(n)
    scale_new = np.empty(n)
    sp = np.empty(m)
    psi_s = np.empty(m)
    s_new = np.empty(m)
    ds = np.empty(m)
    scale_s = np.empty(m)
    scale_s_new = np.empty(m)
    J = np.ascontiguousarray(jac(t0, y0, th)[0])  # noqa: F821 - the model's kernel
    LU = np.eye(n)
    piv = np.arange(n)
    lu_valid = False
    t = t0
    order = 1
    n_equal = 0
    steps = 0
    y = y0.copy()
    s = s0.copy()
    Wd = np.zeros((3, MAX_ORDER + 1, max(n, MAX_ORDER + 1)))
    Ws = np.zeros((3, MAX_ORDER + 1, max(m, MAX_ORDER + 1)))
    _ = Wn
    while t < t1:
        min_step = max(10 * abs(np.nextafter(t, np.inf) - t), 10 * TINY)
        h = h_abs
        if h > max_step or h < min_step:
            hc = max_step if h > max_step else min_step
            _change_D(D, order, hc / h, Wd)
            if P:
                _change_D(Ds, order, hc / h, Ws)
            h = hc
            n_equal = 0
        current_jac = False
        accepted = False
        while not accepted:
            if not np.isfinite(h):
                return NOT_FINITE, i_out, y, s, steps
            if h < min_step:
                return TOO_SMALL_STEP, i_out, y, s, steps
            t_new = t + h
            if t_new > t1:
                f = abs(t1 - t) / h
                _change_D(D, order, f, Wd)
                if P:
                    _change_D(Ds, order, f, Ws)
                n_equal = 0
                lu_valid = False
                t_new = t1
            h = t_new - t
            retry = True
            while retry:
                retry = False
                _predict(D, order, yp, psi)
                _state_scale(scale, atol, rtol, yp)
                cc = h / _ALPHA[order]
                if not lu_valid:
                    LU, piv = _lu_factor(np.eye(n) - cc * J)
                    lu_valid = True
                converged, n_iter = _newton(t_new, yp, cc, psi, LU, piv, scale, newton_tol,
                                            th, u, y_new, d, rb, dy, False, yp, TD, UD, P,
                                            n)  # fmt: skip
                if converged and P:
                    _predict(Ds, order, sp, psi_s)
                    _sens_scale(scale_s, atol, rtol, sp, P, n)
                    converged, _k = _newton(t_new, sp, cc, psi_s, LU, piv, scale_s,
                                            newton_tol, th, u, s_new, ds, rb, dy, True,
                                            y_new, TD, UD, P, n)  # fmt: skip
                if not converged:
                    if not current_jac:
                        J = np.ascontiguousarray(jac(t_new, yp, th)[0])  # noqa: F821
                        current_jac = True
                        lu_valid = False
                        retry = True
                        continue
                    _change_D(D, order, 0.5, Wd)
                    if P:
                        _change_D(Ds, order, 0.5, Ws)
                    h *= 0.5
                    n_equal = 0
                    lu_valid = False
                    break
                safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)
                _state_scale(scale_new, atol, rtol, y_new)
                error_norm = _rms_ratio(d, scale_new, _ERROR_CONST[order])
                if P:
                    _sens_scale(scale_s_new, atol, rtol, s_new, P, n)
                    error_norm = max(error_norm, _rms_ratio(ds, scale_s_new, _ERROR_CONST[order]))
                if error_norm > 1:
                    factor = max(MIN_FACTOR, safety * error_norm ** (-1.0 / (order + 1)))
                    _change_D(D, order, factor, Wd)
                    if P:
                        _change_D(Ds, order, factor, Ws)
                    h *= factor
                    n_equal = 0
                    break  # a rejected step keeps the factorisation (as scipy)
                accepted = True
                _update(D, d, order)
                if P:
                    _update(Ds, ds, order)
                steps += 1
                t = t_new
                for i in range(n):
                    y[i] = y_new[i]
                for i in range(m):
                    s[i] = s_new[i]
                n_equal += 1
                h_acc = h
                order_acc = order
                if n_equal >= order + 1:
                    err_m = np.inf
                    err_p = np.inf
                    if order > 1:
                        err_m = _rms_ratio(D[order], scale_new, _ERROR_CONST[order - 1])
                        if P:
                            err_m = max(err_m, _rms_ratio(Ds[order], scale_s_new,
                                                          _ERROR_CONST[order - 1]))  # fmt: skip
                    if order < MAX_ORDER:
                        err_p = _rms_ratio(D[order + 2], scale_new, _ERROR_CONST[order + 1])
                        if P:
                            err_p = max(err_p, _rms_ratio(Ds[order + 2], scale_s_new,
                                                          _ERROR_CONST[order + 1]))  # fmt: skip
                    best = 0
                    fbest = -1.0
                    for q in range(3):
                        nq = err_m if q == 0 else (error_norm if q == 1 else err_p)
                        fq = np.inf if nq == 0 else nq ** (-1.0 / (order + q))
                        if fq > fbest:
                            fbest = fq
                            best = q
                    order_acc = order + best - 1
                    grow = min(MAX_FACTOR, safety * fbest)
                    _change_D(D, order_acc, grow, Wd)
                    if P:
                        _change_D(Ds, order_acc, grow, Ws)
                    h_acc = h * grow
                    n_equal = 0
                    lu_valid = False
                while i_out < T and t_out[i_out] <= t and t_out[i_out] < out_before:
                    _dense_into(ys[i_out], D, t, h_acc, order_acc, t_out[i_out])
                    if P:
                        _dense_into(ss[i_out], Ds, t, h_acc, order_acc, t_out[i_out])
                    i_out += 1
                order = order_acc
                h_abs = h_acc
                if steps >= max_steps and t < t1:
                    return TOO_MANY_STEPS, i_out, y, s, steps
    return OK, i_out, y, s, steps


@njit(cache=True, nogil=True, error_model="numpy")
def run_one(theta, theta_dot, y0, y0_dot, b_time, b_time_dot, b_index, rise, rise_dot,
            i_start, i_start_dot, i_stop, i_stop_dot, i_index, i_rise, i_rise_dot,
            t0, t1, t_out, rtol, atol, max_step, max_steps):  # fmt: skip
    """One scenario, as the JAX program's segment loop without events: boluses applied at
    the start of each segment, infusions constant on it, the sensitivities' jumps at moving
    boundaries (``across``). Returns ys (T, n), S (T, P, n), status, most steps in a
    segment."""
    n = y0.size
    P = theta_dot.shape[0]
    T = t_out.size
    ys = np.full((T, n), np.nan)
    ss = np.full((T, P * n), np.nan)
    z = y0.copy()
    zs = np.ascontiguousarray(y0_dot).ravel().copy()
    if np.any(b_time < t0) or np.any(i_start < t0) or np.any(i_stop < i_start):
        return ys, ss.reshape(T, P, n), BAD_REGIMEN, 0
    # boundaries: doses, infusion starts and stops, then t1
    nb = b_time.size
    ni = i_start.size
    cands = np.concatenate((b_time, i_start, i_stop, np.array([t1])))
    cand_dot = np.zeros((P, cands.size))
    for q in range(P):
        cand_dot[q, :nb] = b_time_dot[q]
        cand_dot[q, nb : nb + ni] = i_start_dot[q]
        cand_dot[q, nb + ni : nb + 2 * ni] = i_stop_dot[q]
    i_out = 0
    most = 0
    a = t0
    # the start: any doses at t0, and (a dose at t0 that moves) the jump of the derivative
    z_start = z.copy()
    first_at_t0 = -1
    for k in range(nb):
        if b_time[k] == a:
            z_start[b_index[k]] += rise[k]
            if first_at_t0 < 0:
                first_at_t0 = k
    if P:
        u0, ud0 = _inputs_at(a, cands, i_start, i_stop, i_index, i_rise, i_rise_dot, P, n, t1)
        f_before = _fun(a, z, theta, u0)
        f_after = _fun(a, z_start, theta, u0)
        for q in range(P):
            for k in range(nb):
                if b_time[k] == a:
                    zs[q * n + b_index[k]] += rise_dot[q, k]
            if first_at_t0 >= 0:
                bd = b_time_dot[q, first_at_t0]
                for i in range(n):
                    zs[q * n + i] += (f_before[i] - f_after[i]) * bd
    z = z_start
    while a < t1:
        b = np.inf
        which = cands.size - 1
        for k in range(cands.size):
            if cands[k] > a and cands[k] < b:
                b = cands[k]
                which = k
        b = min(b, t1)
        u, UD = _inputs_at(a, cands, i_start, i_stop, i_index, i_rise, i_rise_dot, P, n, t1)
        status, i_out, y_end, s_end, steps = _bdf(
            a, b, z, zs, theta, u, theta_dot, UD, P, t_out, ys, ss, i_out, b, rtol, atol,
            max_step, max_steps,
        )  # fmt: skip
        most = max(most, steps)
        if status != OK:
            return ys, ss.reshape(T, P, n), status, most
        # the boundary: its doses; the derivative jumps where the boundary moves
        z_after = y_end.copy()
        for k in range(nb):
            if b_time[k] == b:
                z_after[b_index[k]] += rise[k]
        zs_after = s_end.copy()
        if P:
            f_before = _fun(b, y_end, theta, u)
            u_a, _ = _inputs_at(b, cands, i_start, i_stop, i_index, i_rise, i_rise_dot, P, n,
                                t1)  # fmt: skip
            f_after = _fun(b, z_after, theta, u_a)
            for q in range(P):
                for k in range(nb):
                    if b_time[k] == b:
                        zs_after[q * n + b_index[k]] += rise_dot[q, k]
                bd = cand_dot[q, which] if b < t1 else 0.0
                for i in range(n):
                    zs_after[q * n + i] += (f_before[i] - f_after[i]) * bd
        z = z_after
        zs = zs_after
        a = b
    # output times at the end: after the doses there
    for i in range(i_out, T):
        if t_out[i] == t1:
            ys[i] = z
            if P:
                ss[i] = zs
    return ys, ss.reshape(T, P, n), OK, most


@njit(cache=True, error_model="numpy")
def _inputs_at(a, cands, i_start, i_stop, i_index, i_rise, i_rise_dot, P, n, t1):
    """The infusion input of the segment starting at ``a`` (constant on it), and its
    derivative along each direction."""
    b = np.inf
    for k in range(cands.size):
        if cands[k] > a and cands[k] < b:
            b = cands[k]
    mid = 0.5 * (a + min(b, t1))
    u = np.zeros(n)
    UD = np.zeros((P, n))
    for k in range(i_start.size):
        if i_start[k] <= mid and mid < i_stop[k]:
            u[i_index[k]] += i_rise[k]
            for q in range(P):
                UD[q, i_index[k]] += i_rise_dot[q, k]
    return u, UD
