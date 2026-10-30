# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Step 4.3a: the JAX BDF core (V45). It is a port of scipy's BDF, so on the same problem with
the same first step it must take the same steps (up to round-off) and give the same answers;
and each order 1-5 must converge at its order. Expected values come from scipy itself, from
matrix exponentials, and from the convergence order of the NDF formulas."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sl
from scipy.integrate._ivp.bdf import BDF

import tenqsp  # noqa: F401 - enables float64
from tenqsp.backends.jax_bdf import OK, TOO_SMALL_STEP, bdf, fixed_order


def scipy_bdf(fun, jac, y0, t1, rtol, atol, h0, t_out):
    """scipy's BDF driven step by step: (accepted steps, LU count, Jacobians, y at t_out)."""
    fun, jac = jax.jit(fun), jax.jit(jac)  # compiled once: scipy calls them thousands of times
    s = BDF(
        lambda t, y: np.asarray(fun(t, y)),
        0.0,
        np.asarray(y0, float),
        t1,
        rtol=rtol,
        atol=atol,
        jac=lambda t, y: np.asarray(jac(t, y)),
        first_step=h0,
    )
    steps, out, i = 0, np.full((len(t_out), len(y0)), np.nan), 0
    while s.status == "running":
        s.step()
        steps += 1
        dense = s.dense_output()
        while i < len(t_out) and t_out[i] <= s.t:
            out[i] = dense(t_out[i])
            i += 1
    return steps, s.nlu, s.njev, out


def robertson(t, y):
    return jnp.array(
        [
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2,
        ]
    )


def van_der_pol(t, y, mu=100.0):
    return jnp.array([y[1], mu * ((1 - y[0] ** 2) * y[1] - y[0])])


CASES = {
    "robertson": (robertson, [1.0, 0.0, 0.0], 1e5, 1e-6, 1e-10, 1e-6),
    "van der Pol, mu = 100": (van_der_pol, [2.0, 0.0], 200.0, 1e-4, 1e-6, 1e-4),
}


@pytest.mark.parametrize("name", list(CASES))
def test_same_steps_and_answers_as_scipy(name):
    fun, y0, t1, rtol, atol, h0 = CASES[name]
    jac = jax.jacfwd(fun, argnums=1)
    t_out = np.linspace(0, t1, 31)
    r = jax.jit(lambda y0: bdf(fun, jac, 0.0, t1, y0, t_out, rtol=rtol, atol=atol, h0=h0))(
        jnp.asarray(y0)
    )
    steps, nlu, njev, expected = scipy_bdf(fun, jac, y0, t1, rtol, atol, h0, t_out)
    assert int(r.status) == OK
    # round-off can nudge a step-size decision, so allow 2% (identical on this machine)
    assert abs(int(r.n_steps) - steps) <= max(2, 0.02 * steps), (int(r.n_steps), steps)
    assert abs(int(r.nlu) - nlu) <= max(2, 0.02 * nlu), (int(r.nlu), nlu)
    scale = np.max(np.abs(expected), axis=0) + atol
    np.testing.assert_allclose(np.asarray(r.ys) / scale, expected / scale, atol=100 * rtol)
    assert njev >= 1 and int(r.njev) >= 1


def test_a_linear_stiff_system_against_the_matrix_exponential():
    rng = np.random.default_rng(45)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    A = Q @ np.diag([-1.0, -10.0, -1e2, -1e3, -1e4, -1e5]) @ Q.T  # stiffness ratio 1e5
    y0 = rng.normal(size=6)
    t_out = np.linspace(0, 5, 11)
    expected = np.array([sl.expm(A * t) @ y0 for t in t_out])
    fun = lambda t, y: jnp.asarray(A) @ y  # noqa: E731
    jac = lambda t, y: jnp.asarray(A)  # noqa: E731
    for rtol in (1e-6, 1e-9):
        r = bdf(fun, jac, 0.0, 5.0, y0, t_out, rtol=rtol, atol=rtol * 1e-3)
        err = np.max(np.abs(np.asarray(r.ys) - expected))
        assert err < 100 * rtol * np.max(np.abs(y0)), (rtol, err)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_each_order_converges_at_its_order(order):
    """Fixed order and step from an exact history: the global error at t = 1 falls like h^k
    (observed order within 0.3 of k)."""
    A = np.array([[-1.0, 2.0], [-2.0, -1.0]])  # a decaying rotation
    y0 = np.array([1.0, 0.5])
    exact = lambda t: sl.expm(A * t) @ y0  # noqa: E731
    fun = lambda t, y: jnp.asarray(A) @ y  # noqa: E731
    jac = lambda t, y: jnp.asarray(A)  # noqa: E731
    errors = []
    hs = [1 / 16, 1 / 32, 1 / 64]
    for h in hs:
        past = np.array([exact(-j * h) for j in range(order + 1)])  # y(t0), y(t0 - h), ...
        D0 = np.zeros((order + 1, 2))
        diffs = past.copy()
        for j in range(order + 1):  # backward differences of the exact values
            D0[j] = diffs[0]
            diffs = diffs[:-1] - diffs[1:]
        y = fixed_order(fun, jac, jnp.asarray(D0), 0.0, h, order, int(round(1 / h)))
        errors.append(np.max(np.abs(np.asarray(y) - exact(1.0))))
    observed = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(np.abs(observed - order) < 0.3), (order, errors, observed)


def test_compiles_once_and_runs_under_vmap():
    def fun(t, y, k):
        return jnp.array([-k * y[0], k * y[0] - 0.5 * y[1]])

    def solve(k, y0):
        f = lambda t, y: fun(t, y, k)  # noqa: E731
        return bdf(
            f, jax.jacfwd(f, argnums=1), 0.0, 4.0, y0, jnp.array([1.0, 4.0]), rtol=1e-8, atol=1e-10
        )

    run = jax.jit(solve)
    ks = np.array([0.5, 1.0, 2.0, 4.0])
    one_by_one = [np.asarray(run(k, jnp.array([1.0, 0.0])).ys) for k in ks]
    assert run._cache_size() == 1  # four values, one compilation
    batch = jax.jit(jax.vmap(solve, in_axes=(0, None)))(ks, jnp.array([1.0, 0.0]))
    np.testing.assert_allclose(np.asarray(batch.ys), np.array(one_by_one), rtol=1e-12)
    exact = np.exp(-ks * 4.0)
    np.testing.assert_allclose(np.asarray(batch.ys)[:, 1, 0], exact, rtol=1e-6, atol=1e-9)


def test_a_failing_run_reports_its_status_instead_of_hanging():
    """y' = y^2 from y(0) = 1 blows up at t = 1: the step size collapses."""
    fun = lambda t, y: y**2  # noqa: E731
    jac = lambda t, y: jnp.diag(2 * y)  # noqa: E731
    r = bdf(fun, jac, 0.0, 2.0, jnp.array([1.0]), jnp.array([0.5, 2.0]), rtol=1e-6, atol=1e-9)
    assert int(r.status) == TOO_SMALL_STEP
    assert float(r.t) == pytest.approx(1.0, abs=1e-3)
    np.testing.assert_allclose(np.asarray(r.ys)[0], [2.0], rtol=1e-4)  # y(0.5) = 2
    assert np.isnan(np.asarray(r.ys)[1, 0])  # never reached


def test_an_event_stops_the_run_at_the_crossing():
    """y' = -y from 1, watching y - 0.5 downward: it stops at ln 2 with y = 0.5; outputs at
    and after the event are left unwritten."""
    fun = lambda t, y: -y  # noqa: E731
    jac = lambda t, y: -jnp.eye(1)  # noqa: E731
    watch = lambda t, y: y - 0.5  # noqa: E731
    t_out = jnp.array([0.5, 1.0, 2.0])
    r = bdf(
        fun,
        jac,
        0.0,
        5.0,
        jnp.array([1.0]),
        t_out,
        rtol=1e-10,
        atol=1e-12,
        events=watch,
        event_dir=jnp.array([-1.0]),
    )
    assert int(r.status) == 3 and int(r.event) == 0
    assert float(r.t) == pytest.approx(np.log(2), rel=1e-9)
    assert float(r.y[0]) == pytest.approx(0.5, rel=1e-9)
    np.testing.assert_allclose(np.asarray(r.ys)[0], [np.exp(-0.5)], rtol=1e-8)
    assert np.all(np.isnan(np.asarray(r.ys)[1:]))
    up = bdf(
        fun,
        jac,
        0.0,
        5.0,
        jnp.array([1.0]),
        jnp.array([5.0]),
        rtol=1e-8,
        atol=1e-10,
        events=watch,
        event_dir=jnp.array([1.0]),
    )
    assert int(up.status) == 0  # a downward crossing does not count
