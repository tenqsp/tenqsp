# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Compiler + scipy backend (Phase 1 step 1.4). Oracles come from the mathematics, written
independently of the compiler: exact solutions, hand-written ODE chains, a hand-assembled
right-hand side for the toy phage model, and scipy.stats for initial shapes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as st
from scipy.integrate import solve_ivp

import tenqsp as tq
from tenqsp.backends import scipy_backend
from tenqsp.compile import CompiledModel
from tenqsp.examples import faces, phage_latency_model, toy_phage_model
from tenqsp.operators import advective_flux_interior

RNG = np.random.default_rng(20261011)


def steady_state(cm: CompiledModel, theta=None):
    """Exact steady state of a *linear* model: y* = y - J^-1 f(y) at y = 0."""
    y = jnp.zeros(cm.n_states)
    J = np.asarray(cm.jacobian(0.0, y, theta))
    f = np.asarray(cm.rhs(0.0, y, theta))
    return -np.linalg.solve(J, f)


# ---------------------------------------------------------------------------- compartments


def test_decay_matches_exact_solution():
    m = tq.Model("decay")
    P = m.compartment("P", init=1e8)
    delta = m.param("delta", 0.1)
    m.ode(P, -delta * P)
    t = np.array([0.0, 2.5, 10.0, 24.0])
    sol = m.solve((0, 24), times=t, rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(sol["P"], 1e8 * np.exp(-0.1 * t), rtol=1e-8)
    sol2 = m.solve((0, 24), times=t, params={"delta": 0.2}, rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(sol2["P"], 1e8 * np.exp(-0.2 * t), rtol=1e-8)


def test_logistic_growth_matches_exact_solution():
    m = tq.Model("logistic")
    B = m.compartment("B", init=1e3)
    g, K = m.params(g=1.2, K=1e9)
    m.ode(B, g * B * (1 - B / K))
    t = np.linspace(0, 30, 7)
    sol = m.solve((0, 30), times=t, rtol=1e-10, atol=1e-6)
    exact = 1e9 / (1 + (1e9 / 1e3 - 1) * np.exp(-1.2 * t))
    np.testing.assert_allclose(sol["B"], exact, rtol=1e-7)


def test_doses_are_added_and_reported_after_the_dose():
    m = tq.Model("doses")
    P = m.compartment("P")
    delta, D0 = m.params(delta=0.3, D0=5e8)
    m.ode(P, -delta * P)
    m.dose(P, times=[0.0, 5.0], amounts=[D0, 1e8])
    t = np.array([0.0, 4.0, 5.0, 7.5, 10.0])
    sol = m.solve((0, 10), times=t, rtol=1e-10, atol=1e-6)
    exact = 5e8 * np.exp(-0.3 * t) + 1e8 * np.exp(-0.3 * (t - 5)) * (t >= 5)
    np.testing.assert_allclose(sol["P"], exact, rtol=1e-8)
    assert sol.stats["segments"] == 2
    # dose amounts are inputs: a new D0 needs no recompilation
    sol2 = m.solve((0, 10), times=t, params={"D0": 1e8}, rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(sol2.at("P", 0.0), 1e8, rtol=1e-12)


# ---------------------------------------------------------------------------- age axis


def transit_model(n: int, T: float = 2.0):
    """Depot -> age axis (n erlang cells over [0, T], speed 1) -> central, eliminated."""
    m = tq.Model("transit")
    a = m.axis("a", 0, T, n, kind="one_way")
    A = m.compartment("A")
    C = m.compartment("C")
    I = m.field("I", axes=[a])
    ka, ke = m.params(ka=1.5, ke=0.2)
    m.advect(I, axis=a, velocity=1.0, scheme="erlang")
    m.inflow(I, axis=a, at="lower", flux=ka * A)
    m.outflow(I, axis=a, at="upper", to=C)
    m.ode(A, -ka * A)
    m.ode(C, -ke * C)
    m.dose(A, times=[0.0], amounts=[100.0])
    return m


def test_erlang_age_axis_is_a_transit_compartment_chain():
    """At model level: the age axis equals the Savic transit chain (rate n/T), with
    the outflow routed into the central compartment."""
    n, T, ka, ke = 6, 2.0, 1.5, 0.2
    cm = transit_model(n, T).compile()
    ktr = n / T

    def chain(t, x):  # x = [A, X_1..X_n (amounts), C]
        A, X, C = x[0], x[1:-1], x[-1]
        inflow = np.concatenate([[ka * A], ktr * X[:-1]])
        return np.concatenate([[-ka * A], inflow - ktr * X, [ktr * X[-1] - ke * C]])

    # right-hand sides agree exactly at a random state (field density x width = amount)
    y = RNG.uniform(0, 10, cm.n_states)
    w = T / n
    st_ = cm.unpack(y)
    x = np.concatenate([[st_["A"]], st_["I"] * w, [st_["C"]]])
    d = cm.unpack(cm.rhs(0.0, y))
    dx = chain(0.0, x)
    np.testing.assert_allclose(d["I"] * w, dx[1:-1], rtol=1e-13)
    np.testing.assert_allclose([d["A"], d["C"]], [dx[0], dx[-1]], rtol=1e-13)

    t = np.linspace(0, 12, 13)
    sol = cm.solve((0, 12), times=t, rtol=1e-10, atol=1e-9)
    ref = solve_ivp(chain, (0, 12), np.r_[100.0, np.zeros(n + 1)], t_eval=t, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(sol["C"], ref.y[-1], rtol=1e-7, atol=1e-8)


def test_routed_outflow_conserves_the_total():
    """Nothing is eliminated: depot + transit + central stays at the dose."""
    m = transit_model(8)
    cm = m.compile()
    sol = cm.solve((0, 10), params={"ke": 0.0}, rtol=1e-10, atol=1e-10)
    w = 2.0 / 8
    total = sol["A"] + sol["I"].sum(axis=1) * w + sol["C"]
    np.testing.assert_allclose(total, 100.0, rtol=1e-9)


def test_mckendrick_steady_state_sharp():
    """V4 at model level: constant inflow J, death mu: u(a) = (J/v) exp(-mu a / v)."""
    J, v, mu, n = 2.0, 1.0, 0.7, 80
    m = tq.Model("mckendrick")
    a = m.axis("a", 0, 3, n, kind="one_way")
    I = m.field("I", axes=[a])
    m.advect(I, axis=a, velocity=v, scheme="sharp")
    m.inflow(I, axis=a, at="lower", flux=J)
    m.outflow(I, axis=a, at="upper")
    m.hazard(I, mu)
    cm = m.compile()
    # the sharp scheme is nonlinear (limiter): integrate to steady state instead
    sol = cm.solve((0, 40), times=[40.0], rtol=1e-10, atol=1e-10)
    f = np.asarray(a.faces)
    exact = (J / v) * (v / mu) * (np.exp(-mu * f[:-1] / v) - np.exp(-mu * f[1:] / v)) / np.diff(f)
    # error measured against the inflow level J/v, as in the operator-level V4 test
    assert np.max(np.abs(sol["I"][-1] - exact)) / (J / v) < 2e-3


# ---------------------------------------------------------------------------- toy phage model


def toy_rhs_by_hand(y, n_z=30, n_a=40):
    """The toy phage model assembled by hand (NumPy for reactions, transitions and integrals;
    the step-1.1 operators for the sharp age transport), independently of the compiler."""
    g_max, K, beta, delta, h_max, tau, s_h, w_amp, Dz, mu = (
        1.0, 1e9, 50.0, 0.1, 5.0, 0.8, 0.05, 0.5, 0.01, 1e-6,
    )  # fmt: skip
    k, cost = np.array([1e-9, 1e-12]), np.array([0.0, 0.1])
    zf = np.linspace(0, 1, n_z + 1)
    zc, dz = 0.5 * (zf[1:] + zf[:-1]), 1 / n_z
    af = np.linspace(0, 1.5, n_a + 1)
    ac, da = 0.5 * (af[1:] + af[:-1]), 1.5 / n_a
    B = y[: 2 * n_z].reshape(n_z, 2)
    I = y[2 * n_z : 2 * n_z + n_a * n_z].reshape(n_a, n_z)
    P = y[-1]

    B_tot = B.sum() * dz
    div = g_max * (1 - zc)[:, None] * (1 - cost)[None, :] * B * (1 - B_tot / K)
    ads = k[None, :] * (1 - 0.5 * zc)[:, None] * B * P
    h = h_max / (1 + np.exp(-(ac - tau) / s_h))
    ads_z = ads.sum(axis=1)
    ads_total = ads.sum() * dz
    lysis = (h[:, None] * I).sum() * da * dz
    Q = np.array([[-mu, mu], [mu, -mu]])  # Q[dest, origin]

    def z_transport(u, v_faces=None):  # zero-flux both ends, along axis 0 of u (z first)
        F = np.zeros((n_z + 1, *u.shape[1:]))
        if v_faces is not None:
            v = v_faces[1:-1].reshape(-1, *[1] * (u.ndim - 1))
            F[1:-1] += np.maximum(v, 0) * u[:-1] + np.minimum(v, 0) * u[1:]
        F[1:-1] += -Dz * (u[1:] - u[:-1]) / dz
        return -(F[1:] - F[:-1]) / dz

    dB = div - ads + div @ Q.T + z_transport(B, w_amp * (0.3 - zf))
    F_int = np.asarray(
        advective_flux_interior(
            jnp.asarray(I), jnp.ones(n_a - 1), dim=0, centers=ac, faces=af, scheme="sharp",
            lower_value=jnp.asarray(ads_z),
        )
    )  # fmt: skip
    F = np.concatenate([ads_z[None, :], F_int, I[-1:, :]], axis=0)  # inflow, interior, outflow
    dI = -(F[1:] - F[:-1]) / da - h[:, None] * I + z_transport(I.T).T
    dP = beta * lysis - ads_total - delta * P
    return np.concatenate([dB.ravel(), dI.ravel(), [dP]])


def test_toy_phage_rhs_matches_hand_assembly():
    cm = toy_phage_model().compile()
    for _ in range(3):
        y = RNG.uniform(0, 1e7, cm.n_states)
        y[-1] = RNG.uniform(1e6, 1e9)
        np.testing.assert_allclose(
            np.asarray(cm.rhs(0.0, y)), toy_rhs_by_hand(y), rtol=1e-11, atol=1e-6
        )


def test_toy_phage_solve_runs_and_stays_near_nonnegative():
    m = toy_phage_model(n_z=10, n_a=12)
    m._terms = [  # erlang ageing keeps this smoke test fast (the sharp front costs time steps)
        t
        if t.kind != "advection" or t.target != "I"
        else type(t)(**{**vars(t), "scheme": "erlang"})
        for t in m._terms
    ]
    sol = m.solve((0, 12), times=np.linspace(0, 12, 5))
    assert sol.at("phage", 0.0) == pytest.approx(1e9)
    assert sol["bacteria"][0] == pytest.approx(1e7, rel=1e-12)
    assert min(sol["B"].min(), sol["I"].min()) > -1e-3  # sub-atol undershoot only
    assert sol.observable_axes["bacteria"] == ()


# ---------------------------------------------------------------------------- categories


def test_mutation_supply_comes_from_the_parent_flux():
    """V13 in miniature: growth 1.0 (S) and 0.5 (R), S->R at mu: R gains mu * g_S * B_S."""
    m = tq.Model("mut")
    s = m.axis("s", labels=["S", "R"])
    x = m.axis("x", 0, 1, 1)
    B = m.field("B", axes=[s, x])
    g = m.param("g", {"S": 1.0, "R": 0.5}, over=s)
    mu = m.param("mu", 1e-6)
    M = m.transitions("M", axis=s, rates={"S->R": mu})
    m.reaction(B, g[s] * B)
    m.transition(B, axis=s, matrix=M, of=g[s] * B)
    cm = m.compile()
    BS, BR = 3e8, 2e2
    d = cm.unpack(cm.rhs(0.0, np.array([BS, BR])))["B"][:, 0]
    np.testing.assert_allclose(
        d, [1.0 * BS - 1e-6 * 1.0 * BS, 0.5 * BR + 1e-6 * 1.0 * BS], rtol=1e-14
    )


def test_switching_conserves_cells_and_follows_the_matrix():
    m = tq.Model("switch")
    s = m.axis("s", labels=["A", "B", "C"])
    x = m.axis("x", 0, 2, 4)
    F = m.field("F", axes=[x, s])
    q1, q2 = m.params(q1=0.3, q2=0.05)
    Q = m.transitions("Q", axis=s, rates={"A->B": q1, "B->C": q2, "C->A": q1 + q2})
    m.transition(F, axis=s, matrix=Q, of=F)
    cm = m.compile()
    y = RNG.uniform(0, 1, cm.n_states)
    u = y.reshape(4, 3)
    d = np.asarray(cm.rhs(0.0, y)).reshape(4, 3)
    G = np.array([[-0.3, 0, 0.35], [0.3, -0.05, 0], [0, 0.05, -0.35]])  # G[dest, origin]
    np.testing.assert_allclose(d, u @ G.T, rtol=1e-14)
    np.testing.assert_allclose(d.sum(axis=1), 0, atol=1e-15)


def test_velocity_per_category_moves_each_class_its_own_way():
    m = tq.Model("two_way")
    s = m.axis("s", labels=["up", "down"])
    x = m.axis("x", 0, 1, 5)
    B = m.field("B", axes=[s, x])
    c = m.param("c", {"up": 1.0, "down": -2.0}, over=s)
    m.advect(B, axis=x, velocity=c[s])
    cm = m.compile()
    y = RNG.uniform(0, 1, 10)
    u = y.reshape(2, 5)
    d = np.asarray(cm.rhs(0.0, y)).reshape(2, 5)
    w = 0.2
    F_up = np.r_[0.0, 1.0 * u[0, :-1], 0.0]  # zero-flux walls
    F_dn = np.r_[0.0, -2.0 * u[1, 1:], 0.0]
    np.testing.assert_allclose(d[0], -(F_up[1:] - F_up[:-1]) / w, rtol=1e-14)
    np.testing.assert_allclose(d[1], -(F_dn[1:] - F_dn[:-1]) / w, rtol=1e-14)


# ---------------------------------------------------------------------------- boundaries


def test_curved_velocity_is_taken_at_the_faces_and_inflow_can_enter_from_above():
    """Velocity -(1 + x^2) (downward), inflow J at the upper end, outflow at the lower end.
    A curved velocity distinguishes face values from averaged centre values."""
    m = tq.Model("down")
    x = m.axis("x", 0, 1, 6)
    u = m.field("u", axes=[x])
    m.advect(u, axis=x, velocity=-(1 + x**2))
    m.inflow(u, axis=x, at="upper", flux=2.0)
    m.outflow(u, axis=x, at="lower")
    cm = m.compile()
    y = RNG.uniform(0, 1, 6)
    f = np.linspace(0, 1, 7)
    v = -(1 + f**2)
    F = np.r_[v[0] * y[0], v[1:-1] * y[1:], -2.0]  # upwind = the cell above; J enters downward
    np.testing.assert_allclose(np.asarray(cm.rhs(0.0, y)), -(F[1:] - F[:-1]) / (1 / 6), rtol=1e-14)


@pytest.mark.parametrize(
    ("kind", "extra", "expected"),
    [
        ("dirichlet", {"value": 1.5}, 1.5),
        ("robin", {"value": 2.0, "h": 3.0}, 2.0),
    ],
)
def test_diffusion_boundaries_reach_the_boundary_value(kind, extra, expected):
    m = tq.Model("bc")
    x = m.axis("x", 0, 1, 20, kind="spatial")
    u = m.field("u", axes=[x])
    m.diffuse(u, axis=x, D=0.1)
    m.boundary(u, axis=x, at="lower", kind=kind, **extra)
    np.testing.assert_allclose(steady_state(m.compile()), expected, rtol=1e-10)


def test_robin_steady_flux_by_hand():
    """Robin at lower (outside 2, h 3), Dirichlet 0 at upper, D = 0.1 on [0, 1]: the steady
    profile is linear, u(0) = 2 - q/h and u(1) = 0 with q = D (u0 - 0) / 1."""
    m = tq.Model("robin")
    x = m.axis("x", 0, 1, 10, kind="spatial")
    u = m.field("u", axes=[x])
    m.diffuse(u, axis=x, D=0.1)
    m.boundary(u, axis=x, at="lower", kind="robin", value=2.0, h=3.0)
    m.boundary(u, axis=x, at="upper", kind="dirichlet", value=0.0)
    ys = steady_state(m.compile())
    u0 = 2.0 / (1 + 0.1 / 3.0)  # from D*u0 = h*(2 - u0)
    np.testing.assert_allclose(ys, u0 * (1 - x.centers), rtol=1e-10)


def test_dirichlet_inflow_by_advection():
    m = tq.Model("adv_dirichlet")
    x = m.axis("x", 0, 1, 10, kind="spatial")
    u = m.field("u", axes=[x])
    m.advect(u, axis=x, velocity=2.0)
    m.boundary(u, axis=x, at="lower", kind="dirichlet", value=3.0)
    m.outflow(u, axis=x, at="upper")
    m.hazard(u, 0.0)
    np.testing.assert_allclose(steady_state(m.compile()), 3.0, rtol=1e-12)


# ---------------------------------------------------------------------------- initial states


SHAPES = [
    (lambda z: tq.normal(z, mean=0.4, sd=0.15), st.norm(0.4, 0.15)),
    (lambda z: tq.lognormal(z, median=0.3, cv=0.5), st.lognorm(np.sqrt(np.log(1.25)), scale=0.3)),
    (lambda z: tq.gamma(z, mean=0.5, cv=0.4), st.gamma(1 / 0.16, scale=0.5 * 0.16)),
    (lambda z: tq.exponential(z, mean=0.3), st.expon(scale=0.3)),
    (lambda z: tq.beta(z, mean=0.3, concentration=8), st.beta(2.4, 5.6)),
    (lambda z: tq.uniform(z, lower=0.2, upper=0.7), st.uniform(0.2, 0.5)),
]


@pytest.mark.parametrize(("shape", "dist"), SHAPES)
def test_initial_shape_gives_exact_cell_masses(shape, dist):
    fz = np.sort(np.r_[0.0, RNG.uniform(0, 1, 11), 1.0])  # nonuniform cells
    m = tq.Model("init")
    z = m.axis("z", faces=fz)
    B = m.field("B", axes=[z])
    m.reaction(B, 0)
    m.init(B, total=1e7, shape=shape(z))
    amounts = np.asarray(m.compile().initial_state()) * np.diff(fz)
    cdf = dist.cdf(fz)
    np.testing.assert_allclose(
        amounts, 1e7 * np.diff(cdf) / (cdf[-1] - cdf[0]), rtol=1e-9, atol=1e-6
    )
    assert amounts.sum() == pytest.approx(1e7, rel=1e-14)


def test_point_shape_puts_everything_in_one_cell():
    m = tq.Model("pt")
    z = m.axis("z", 0, 1, 4)
    B = m.field("B", axes=[z])
    m.reaction(B, 0)
    m.init(B, total=8.0, shape=tq.point(z, at=0.5))  # on a face: the upper cell
    np.testing.assert_allclose(np.asarray(m.compile().initial_state()) * 0.25, [0, 0, 8, 0])


def test_initial_value_is_a_cell_average():
    fz = np.array([0.0, 0.1, 0.35, 0.8, 1.0])
    m = tq.Model("avg")
    z = m.axis("z", faces=fz)
    B = m.field("B", axes=[z])
    m.reaction(B, 0)
    m.init(B, z**2 + 3 * z**4)
    a, b = fz[:-1], fz[1:]
    exact = ((b**3 - a**3) / 3 + 3 * (b**5 - a**5) / 5) / (b - a)
    np.testing.assert_allclose(np.asarray(m.compile().initial_state()), exact, rtol=1e-13)


def test_general_shape_is_normalised_to_the_total():
    m = tq.Model("gen")
    z = m.axis("z", 0, 2, 7)
    B = m.field("B", axes=[z])
    m.reaction(B, 0)
    m.init(B, total=5.0, shape=tq.exp(-z))
    amounts = np.asarray(m.compile().initial_state()) * (2 / 7)
    assert amounts.sum() == pytest.approx(5.0, rel=1e-14)
    f = np.linspace(0, 2, 8)
    exact = 5.0 * np.diff(-np.exp(-f)) / (1 - np.exp(-2))
    np.testing.assert_allclose(amounts, exact, rtol=1e-6)


def test_split_and_where_fill_the_right_categories():
    cm = toy_phage_model().compile()
    B = cm.unpack(cm.initial_state())["B"]  # (z, s)
    assert B[:, 0].sum() / 30 == pytest.approx(1e7, rel=1e-13)
    assert np.all(B[:, 1] == 0)
    m = tq.Model("split")
    z = m.axis("z", 0, 1, 10)
    s = m.axis("s", labels=["S", "R"])
    F = m.field("F", axes=[s, z])
    m.reaction(F, 0)
    m.init(F, total=1e7, shape=tq.uniform(z), split={s: {"S": 0.99, "R": 0.01}})
    np.testing.assert_allclose(
        m.compile().unpack(m.compile().initial_state())["F"].sum(axis=1) / 10, [9.9e6, 1e5]
    )


# ---------------------------------------------------------------------------- compile once


def test_new_parameter_values_never_recompile():
    cm = toy_phage_model(n_z=6, n_a=8).compile()
    y = np.asarray(cm.initial_state())
    for i in range(20):
        th = cm.theta({"g_max": 1.0 + 0.01 * i, "k": {"S": 1e-9 * (1 + i)}})
        cm.rhs(0.5 * i, y, th).block_until_ready()
        cm.initial_state(th).block_until_ready()
    assert cm._rhs._cache_size() == 1
    assert cm._y0._cache_size() == 1


def test_compile_is_cached_per_model_description():
    m = transit_model(4)
    assert m.compile() is m.compile()
    m.observable("central", m["C"])
    assert m.compile() is m.compile()


# -------------------------------------------------------------------- infection-age phage example


def test_prd_latency_example_runs():
    m = phage_latency_model(n=20)
    sol = m.solve((0, 24), backend="scipy", method="BDF", times=[0.0, 12.0, 24.0])
    assert sol.at("P", 0.0) == pytest.approx(1e7 + 1e8)
    assert sol["burst_rate"].shape == (3,)
    assert np.all(np.isfinite(sol.y))


# ---------------------------------------------------------------------------- mistakes


def test_solver_options_are_checked():
    m = transit_model(4)
    with pytest.raises(ValueError, match="LSODA is not offered"):
        m.solve((0, 1), method="LSODA")
    with pytest.raises(KeyError, match="Did you mean 'ka'"):
        m.solve((0, 1), params={"kaa": 1.0})
    with pytest.raises(TypeError):
        m.solve((0, 1), rtoll=1e-3)  # misspelt option: never silently ignored
    with pytest.raises(ValueError, match="backend"):
        m.solve((0, 1), backend="diffrax")
    with pytest.raises(ValueError, match="within t_span"):
        m.solve((0, 1), times=[0, 2])


def test_dose_before_the_start_is_refused():
    m = transit_model(4)
    with pytest.raises(ValueError, match="before the start"):
        m.solve((1, 5))


def test_initial_values_may_not_use_the_state():
    m = tq.Model("bad_init")
    A = m.compartment("A", init=1.0)
    C = m.compartment("C")
    m.init(C, 2 * A)
    m.ode(A, -A)
    m.ode(C, A)
    with pytest.raises(tq.ModelError, match="parameters and coordinates only"):
        m.compile()


@pytest.mark.scipy_backend
def test_bdf_failure_is_retried_with_radau(monkeypatch):
    real = scipy_backend._segment
    calls = []

    def flaky(fun, jac, a, b, y, t_eval, method, *rest):
        calls.append(method)
        res = real(fun, jac, a, b, y, t_eval, method, *rest)
        if method == "BDF":
            res.success, res.message = False, "Required step size is less than spacing"
        return res

    monkeypatch.setattr(scipy_backend, "_segment", flaky)
    sol = transit_model(4).solve((0, 2))
    assert calls == ["BDF", "Radau"]
    assert "retried with Radau" in sol.messages[0]


def test_jax_x64_is_on():
    assert jax.config.jax_enable_x64
    assert faces(0, 1, 2) == (0.0, 0.5, 1.0)


# ---------------------------------------------------------------------------- Phase 1 review
# Regression tests for defects found by the independent Phase 1 review (2026-10-11).


def test_caret_is_power_with_power_precedence():
    import sympy as sp

    from tenqsp.expr import parse

    x, a = sp.Symbol("x"), sp.Symbol("a")
    names = {"x": x, "a": a}
    assert parse("2*x^2", names) == 2 * x**2
    assert parse("a - x^2", names) == a - x**2
    assert parse("-x^2", names) == -(x**2)
    assert parse("x^2^3", names) == x ** (2**3)


def test_caret_in_a_model_equation():
    m = tq.Model("caret")
    P = m.compartment("P", init=3.0)
    m.ode(P, "2*P^2 - 2*P^2")
    assert float(m.compile().rhs(0.0, np.array([3.0]))[0]) == 0.0


def _two_velocity_model(parts, scheme="erlang"):
    m = tq.Model("split_v")
    x = m.axis("x", 0, 1, 8)
    u = m.field("u", axes=[x])
    for v in parts:
        m.advect(u, axis=x, velocity=v, scheme=scheme)
    return m.compile()


@pytest.mark.parametrize("scheme", ["erlang", "sharp"])
def test_velocity_written_as_several_terms_acts_as_their_sum(scheme):
    """Upwinding uses the total face velocity: +1 and -1 is no motion at all, and 0.6 + 0.4
    is exactly velocity 1."""
    y = RNG.uniform(0, 1, 8)
    still = np.asarray(_two_velocity_model([1.0, -1.0], scheme).rhs(0.0, y))
    np.testing.assert_allclose(still, 0.0, atol=1e-15)
    split = np.asarray(_two_velocity_model([0.6, 0.4], scheme).rhs(0.0, y))
    whole = np.asarray(_two_velocity_model([1.0], scheme).rhs(0.0, y))
    np.testing.assert_allclose(split, whole, rtol=1e-14, atol=1e-15)


def test_mixed_schemes_on_one_axis_are_refused():
    m = tq.Model("mixed")
    x = m.axis("x", 0, 1, 4)
    u = m.field("u", axes=[x])
    m.advect(u, axis=x, velocity=1.0, scheme="erlang")
    m.advect(u, axis=x, velocity=0.5, scheme="sharp")
    with pytest.raises(tq.ModelError, match="different schemes"):
        m.compile()


def test_negative_diffusivity_is_not_clipped():
    """A cell-varying negative D must act like the same constant negative D (no hidden
    clamp); only D == 0 blocks a face."""
    from tenqsp.operators import diffusive_flux_interior

    w = np.full(5, 0.2)
    u = (np.arange(5) + 0.5) * 0.2
    F_const = diffusive_flux_interior(u, -0.1, dim=0, widths=w)
    F_cells = diffusive_flux_interior(u, np.full(5, -0.1), dim=0, widths=w)
    np.testing.assert_allclose(F_cells, F_const, rtol=1e-14)
    blocked = diffusive_flux_interior(u, np.array([0.1, 0.1, 0.0, 0.1, 0.1]), dim=0, widths=w)
    np.testing.assert_array_equal(np.asarray(blocked)[1:3], 0.0)


def test_new_number_of_output_times_does_not_recompile():
    m = transit_model(4)
    m.observable("central", m["C"])
    for times in ([0, 1, 2, 3, 4], [4], np.linspace(0, 4, 17)):
        m.solve((0, 4), times=times)
    counts = m.compile().compilations()
    assert counts["observables"] == counts["ledger"] == 1, counts


def test_flow_leaving_through_an_inflow_end_leaves():
    """Inflow J at the lower end while the flow runs downward (v = -1): J enters and the
    upwind flux v*u0 leaves through the same face."""
    m = tq.Model("inflow_out")
    x = m.axis("x", 0, 1, 4)
    u = m.field("u", axes=[x])
    m.advect(u, axis=x, velocity=-1.0)
    m.inflow(u, axis=x, at="lower", flux=2.0)
    y = RNG.uniform(0, 1, 4)
    F = np.r_[2.0 - y[0], -y[1:], 0.0]  # faces 0..4; upwind of v < 0 is the cell above
    np.testing.assert_allclose(
        np.asarray(m.compile().rhs(0.0, y)), -(F[1:] - F[:-1]) / 0.25, rtol=1e-14
    )


def test_velocity_from_cell_values_is_interpolated_by_distance():
    """Velocity c*u (cell values) on a geometric grid: interior face values are the linear
    interpolation between the two neighbouring centres."""
    m = tq.Model("nonuniform_v")
    x = m.axis("x", 0, 1, 5, spacing="geometric", ratio=2.0)
    u = m.field("u", axes=[x])
    m.advect(u, axis=x, velocity=0.3 * u)
    f, c, w = x.faces, x.centers, x.widths
    y = RNG.uniform(0.5, 1.5, 5)
    vc = 0.3 * y
    lam = (f[1:-1] - c[:-1]) / (c[1:] - c[:-1])
    vf = (1 - lam) * vc[:-1] + lam * vc[1:]
    F = np.r_[0.0, vf * y[:-1], 0.0]  # v > 0: upwind is the cell below; zero-flux ends
    np.testing.assert_allclose(
        np.asarray(m.compile().rhs(0.0, y)), -(F[1:] - F[:-1]) / w, rtol=1e-13
    )


@pytest.mark.scipy_backend
def test_integrator_error_is_retried_with_radau(monkeypatch):
    real = scipy_backend._segment
    calls = []

    def crashing(fun, jac, a, b, y, t_eval, method, *rest):
        calls.append(method)
        if method == "BDF":
            raise ValueError("array must not contain infs or NaNs")
        return real(fun, jac, a, b, y, t_eval, method, *rest)

    monkeypatch.setattr(scipy_backend, "_segment", crashing)
    sol = transit_model(4).solve((0, 2))
    assert calls == ["BDF", "Radau"]
    assert "ValueError" in sol.messages[0] and "retried with Radau" in sol.messages[0]
    assert sol.stats["nfev"] > 0
