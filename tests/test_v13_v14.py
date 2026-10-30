# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""V13 (mutation acts on the parent's division flux) and V14 (a 1e10 bolus at a dose junction
on the default method).

V13 oracle: with constant growth rates and one-way mutation S -> R at mu per division,
    B_S(t) = B_S0 exp(g_S (1 - mu) t)
    B_R(t) = B_R0 exp(g_R t)
             + mu g_S B_S0 (exp(g_S (1 - mu) t) - exp(g_R t)) / (g_S (1 - mu) - g_R).
pbisim scaled the mutant supply by the destination's rate g_R instead of the parent's g_S,
under-seeding a costly mutant by exactly g_R / g_S; the test checks the difference is visible.

V14: pbisim's scipy BDF failed instantly at a 1e10 PFU site bolus. Here the same kind of model
(free phage adsorbing to bacteria, infected cells by infection age, burst) gets a 1e10 bolus
at t = 2 and must integrate with BDF alone: no Radau retry, ledger closed, and the result
agrees with a tight Radau reference.
"""

from __future__ import annotations

import numpy as np
import pytest

import tenqsp as tq

G = {"S": 1.0, "R": 0.5}
MU = 1e-6


def mutants(*, with_nutrient: bool = False, mu: float = MU):
    m = tq.Model("mutation")
    s = m.axis("s", labels=["S", "R"])
    B = m.field("B", axes=[s])
    g = m.param("g", G, over=s)
    mu_p = m.param("mu", mu)
    M = m.transitions("M", axis=s, rates={"S->R": mu_p})
    if with_nutrient:  # Monod growth; consumption charged on the pre-mutation division flux
        N = m.compartment("N")
        Ks, Y = m.params(Ks=1.0, Y=1e-9)
        div = m.define("div", g[s] * N / (Ks + N) * B)
        m.ode(N, -Y * m.integral(div, over=s))
        m.init(N, 10.0)
    else:
        div = m.define("div", g[s] * B)
    m.reaction(B, div)
    m.transition(B, axis=s, matrix=M, of=div)
    m.init(B, 1e8, where={s: "S"})
    m.init(B, 10.0, where={s: "R"})
    return m


def closed_form(t, gS=1.0, gR=0.5, mu=MU, S0=1e8, R0=10.0):
    a = gS * (1 - mu)
    BS = S0 * np.exp(a * t)
    BR = R0 * np.exp(gR * t) + mu * gS * S0 * (np.exp(a * t) - np.exp(gR * t)) / (a - gR)
    return BS, BR


def test_v13_mutant_supply_follows_the_parents_rate():
    t = np.linspace(0, 10, 11)
    sol = mutants().solve((0, 10), times=t, rtol=1e-11, atol=1e-6)
    BS, BR = closed_form(t)
    np.testing.assert_allclose(sol["B"][:, 0], BS, rtol=1e-8)
    np.testing.assert_allclose(sol["B"][:, 1], BR, rtol=1e-8)
    # the pbisim defect (supply at the destination's rate) would give a visibly different R
    _, BR_wrong = closed_form(t, gS=1.0, gR=0.5, mu=MU * 0.5 / 1.0)
    assert abs(sol["B"][-1, 1] / BR_wrong[-1] - 1) > 0.3
    assert max(sol.ledger.check().values()) < 1e-10


def test_v13_mutation_does_not_change_nutrient_consumption():
    """At any state, dN/dt is the same with and without mutation: consumption is charged on
    the division flux before it is redistributed among daughters."""
    with_mu = mutants(with_nutrient=True).compile()
    without = mutants(with_nutrient=True, mu=0.0).compile()
    rng = np.random.default_rng(13)
    for _ in range(5):
        y = rng.uniform(0.5, 2.0, with_mu.n_states) * np.array([1e8, 1e3, 5.0])[: with_mu.n_states]
        a, b = with_mu.unpack(with_mu.rhs(0.0, y)), without.unpack(without.rhs(0.0, y))
        assert a["N"] == pytest.approx(float(b["N"]), rel=1e-15)
        # the R class gains mu * g_S * f(N) * B_S exactly
        BS, N = y[with_mu.layout.slices["B"]][0], y[with_mu.layout.slices["N"]][0]
        f = N / (1.0 + N)
        assert a["B"][1] - b["B"][1] == pytest.approx(MU * 1.0 * f * BS, rel=1e-9)
        assert np.sum(a["B"]) == pytest.approx(float(np.sum(b["B"])), rel=1e-14)


# --------------------------------------------------------------------------- V14
def bolus_model(B0: float, *, n: int = 40):
    """A logistic infection-age phage toy with illustrative values (g = 1/h, K = 1e9/mL,
    k = 1e-7 mL/h, beta = 100, delta = 0.1/h, tau = 0.5 h, s = 0.03 h, h_max = 30/h), and a
    1e10 phage bolus at t = 2 h."""
    m = tq.Model("bolus")
    age = m.axis("age", 0.0, 1.0, n, kind="one_way", units="h")
    B, P = m.compartment("B", init=B0), m.compartment("P", init=0.0)
    I = m.field("I", axes=[age], init=0.0)
    k, beta, g, K, delta, h_max, tau, s = m.params(
        k=1e-7, beta=100.0, g=1.0, K=1e9, delta=0.1, h_max=30.0, tau=0.5, s=0.03
    )
    h = h_max * tq.sigmoid((age.coord - tau) / s)
    m.ode(B, g * B * (1 - B / K) - k * B * P)
    m.advect(I, axis=age, velocity=1.0, scheme="sharp")
    m.inflow(I, axis=age, at="lower", flux=k * B * P)
    m.outflow(I, axis=age, at="upper")
    m.hazard(I, h)
    m.ode(P, beta * m.integral(h * I, over=age) - k * B * P - delta * P)
    m.dose(P, times=[2.0], amounts=[1e10])
    return m


@pytest.mark.parametrize("B0", [1e6, 1e8])
@pytest.mark.parametrize("ledger", [True, pytest.param(False, marks=pytest.mark.slow)])
@pytest.mark.scipy_backend
def test_v14_large_bolus_on_the_default_method(B0, ledger):
    m = bolus_model(B0)
    t = np.linspace(0, 12, 25)
    sol = m.solve((0, 12), times=t, ledger=ledger)
    assert sol.method == "BDF" and sol.messages == [], sol.messages
    ref = m.solve((0, 12), times=t, method="Radau", rtol=1e-9, atol=1e-6, ledger=False)
    for name in ("B", "P"):
        np.testing.assert_allclose(sol[name], ref[name], rtol=1e-3, atol=1e-2)
    if ledger:
        assert max(sol.ledger.check().values()) < 1e-10
    assert sol["P"][5] > 1e9  # the bolus arrived


@pytest.mark.parametrize("ledger", [True, False])
@pytest.mark.scipy_backend
def test_v14_the_bolus_arrives_in_full(ledger):
    """An oracle independent of the solver: across one microsecond around the dose the phage
    count rises by exactly the dose (the dynamics move at most ~2e5 phage in that time)."""
    sol = bolus_model(1e8).solve((0, 3), times=[2 - 5e-7, 2 + 5e-7], ledger=ledger, rtol=1e-8)
    assert (sol["P"][1] - sol["P"][0]) / 1e10 == pytest.approx(1.0, abs=1e-4)


def test_first_step_is_floored_but_respects_max_step():
    from tenqsp.backends.scipy_backend import FIRST_STEP_FLOOR, first_step

    def stiff(t, y):  # a scale on which scipy's rule proposes a tiny start
        return np.array([-1e12 * y[0], 1e-3])

    y = np.array([1e10, 0.0])
    h = first_step(stiff, 2.0, 12.0, y, "BDF", 1e-6, np.full(2, 1e-6), np.inf)
    assert h == pytest.approx(FIRST_STEP_FLOOR * 10.0)
    assert first_step(stiff, 2.0, 12.0, y, "BDF", 1e-6, np.full(2, 1e-6), 1e-9) == 1e-9
    gentle = first_step(lambda t, y: -y, 0.0, 10.0, np.ones(2), "BDF", 1e-6, 1e-6, np.inf)
    assert FIRST_STEP_FLOOR * 10 < gentle < 10


def test_atol_per_state():
    m = bolus_model(1e6, n=10)
    n = m.compile().n_states
    a = m.solve((0, 3), times=[3], atol=np.full(n, 1e-6))
    b = m.solve((0, 3), times=[3], atol=1e-6)
    np.testing.assert_array_equal(a.y, b.y)
    with pytest.raises(ValueError, match="one per state"):
        m.solve((0, 3), times=[3], atol=np.full(n + 1, 1e-6))
    with pytest.raises(ValueError, match="positive"):
        m.solve((0, 3), times=[3], atol=0.0)


# ----------------------------------------------------- V14: rounding after a large dose
# Below atol ~ 1e-14 x a dose, the error estimate of the states the dose feeds is rounding,
# and BDF fails or survives by chance (ROUNDOFF_PER_DOSE, measured on bolus_model).


def test_solutions_record_the_doses_given():
    m = tq.Model.from_text(
        "axis s categorical [S, R]\nparam w[s] = {S: 2, R: 4}\nfield C(s) weights=w[s]\n"
        "compartment A\nd_t(C) = -0.1*C\nd_t(A) = -A\ndose A at t=1: 5\n",
        name="w",
    )
    for backend in ("scipy", "jax"):
        own = m.solve((0, 3), times=[3], backend=backend)
        assert own.doses == [(1.0, "A", 5.0, 5.0)]
        reg = tq.bolus("C[s=R]", 8, times=[1, 4]) + tq.bolus("A", 2, times=[0.5])
        sol = m.solve((0, 3), times=[3], regimen=reg, backend=backend)
        assert sorted(sol.doses) == [(0.5, "A", 2.0, 2.0), (1.0, "C", 8.0, 2.0)]  # 8 / w = 2


@pytest.mark.scipy_backend  # the tight run completes only through scipy's Radau retry
def test_diagnostics_warn_when_atol_is_below_a_doses_rounding():
    m = bolus_model(1e8, n=10)
    loose = m.diagnostics(m.solve((0, 3), times=[3], rtol=1e-8, atol=1e-3))
    assert not [i for i in loose.issues if "rounding" in i.message]
    tight = m.diagnostics(m.solve((0, 3), times=[3], rtol=1e-8, atol=1e-6))
    (w,) = [i for i in tight.issues if i.where == "P" and "rounding" in i.message]
    assert w.level == "warning" and w.where == "P" and "atol=1e-04" in w.message


def test_a_regrowth_after_a_regimens_dose_is_not_flagged():
    """The dip-and-regrowth check reads the doses of the run, not the model's (a regimen
    replaced them)."""
    m = tq.Model("refill")
    m.ode(m.compartment("B", init=1.0), "-5*B")
    reg = tq.bolus("B", 1.0, times=[0.0]) + tq.bolus("B", 1.0, times=[10.0])
    sol = m.solve((0, 12), times=np.linspace(0, 12, 25), regimen=reg)
    assert not [i for i in m.diagnostics(sol).issues if "grew back" in i.message]


@pytest.mark.scipy_backend  # the tight run completes only through scipy's Radau retry
def test_failures_after_a_large_dose_suggest_an_atol():
    m = bolus_model(1e8)
    reg = tq.bolus("P", 1e11, times=[2.0])  # BDF fails 8 of 8 at atol 1e-6
    with pytest.raises(tq.SolverError, match=r"dose into 'P' at t = 2.*atol=1e-03"):
        m.solve((0, 3), times=[3], rtol=1e-8, atol=1e-6, regimen=reg, backend="jax")
    sol = m.solve((0, 3), times=[3], rtol=1e-8, atol=1e-6, regimen=reg, ledger=False)
    assert any("retried with Radau" in x and "atol=1e-03" in x for x in sol.messages)
