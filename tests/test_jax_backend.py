# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Step 4.3b: the JAX backend on whole models, with doses, infusions, output times and the
ledger, against the scipy backend and closed forms (part of V9); slots and compile-once."""

from __future__ import annotations

import numpy as np
import pytest

import tenqsp as tq
from tenqsp.backends import jax_backend
from tenqsp.examples import toy_phage_model

PK = """param ka, CL
param V = 30
compartment depot, centr
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
observable conc = centr/V
dose depot at t=0: 100
"""
P = {"ka": 1.2, "CL": 4}
TIGHT = {"rtol": 1e-10, "atol": 1e-12}


def oral(t, doses, ka=1.2, CL=4.0, V=30.0):
    ke, t = CL / V, np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    for td, amt in doses:
        s = np.clip(t - td, 0, None)
        out += amt * ka / (V * (ka - ke)) * (np.exp(-ke * s) - np.exp(-ka * s))
    return out


@pytest.fixture(scope="module")
def pk():
    return tq.Model.from_text(PK, name="pk")


def test_the_models_dose_and_a_regimen_match_the_closed_form_and_scipy(pk):
    t = np.linspace(0, 96, 97)
    sol = pk.solve((0, 96), times=t, params=P, backend="jax", **TIGHT)
    np.testing.assert_allclose(sol["conc"], oral(t, [(0, 100)]), rtol=1e-7, atol=1e-10)
    q12 = tq.bolus("depot", 100, start=0, every=12, n=7)
    j = pk.solve((0, 96), times=t, params=P, regimen=q12, backend="jax", **TIGHT)
    s = pk.solve((0, 96), times=t, params=P, regimen=q12, **TIGHT)
    expected = oral(t, [(12 * k, 100) for k in range(7)])
    np.testing.assert_allclose(j["conc"], expected, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(j["conc"], s["conc"], rtol=1e-9, atol=1e-12)
    assert max(j.ledger.check().values()) < 1e-9
    assert j.stats["segments"] == 7 and j.options["backend"] == "jax"


def test_an_infusion_is_booked_like_on_scipy(pk):
    t = np.linspace(0, 24, 49)
    reg = tq.infusion("centr", rate=50, duration=2, start=3)
    j = pk.solve((0, 24), times=t, params=P, regimen=reg, backend="jax", **TIGHT)
    s = pk.solve((0, 24), times=t, params=P, regimen=reg, **TIGHT)
    np.testing.assert_allclose(j["conc"], s["conc"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(j.ledger.moved["centr"]["infusions"][-1], 100.0)
    assert max(j.ledger.check().values()) < 1e-9


def test_fields_transport_and_integrals_match_scipy():
    m = toy_phage_model(8, 6)
    t = np.linspace(0, 6, 13)
    j = m.solve((0, 6), times=t, backend="jax", rtol=1e-8, atol=1e-6)
    s = m.solve((0, 6), times=t, rtol=1e-8, atol=1e-6)
    for name in ("B", "I", "P"):
        scale = np.max(np.abs(s[name]))
        np.testing.assert_allclose(j[name] / scale, s[name] / scale, atol=1e-6)
    assert max(j.ledger.check().values()) < 1e-8


def test_a_dose_into_one_element_of_a_weighted_field():
    text = "axis s categorical [S, R]\nparam w[s] = {S: 2, R: 4}\nfield C(s) weights=w[s]\n"
    m = tq.Model.from_text(text + "d_t(C) = -0.1*C\n", name="w")
    reg = tq.bolus("C[s=R]", 8, times=[1])
    j = m.solve((0, 3), times=[0, 1, 2, 3], regimen=reg, backend="jax", **TIGHT)
    np.testing.assert_allclose(j["C"][:, 1], [0, 2, 2 * np.exp(-0.1), 2 * np.exp(-0.2)], rtol=1e-8)
    assert np.all(j["C"][:, 0] == 0)


def test_slots_one_build_per_power_of_two(pk):
    program = jax_backend._program(pk.compile(), ledger=True, model_doses=False, max_steps=100_000)
    before = program._cache_size()
    for k in range(1, 9):  # 1..8 doses, 3..8 output times: all in 8 slots
        reg = tq.bolus("depot", 10.0 * k, times=np.linspace(0, 20, k))
        pk.solve((0, 24), times=np.linspace(0, 24, 3 + k % 6), params=P, regimen=reg, backend="jax")
    assert program._cache_size() - before <= 1  # this test's slot sizes: one build
    reg9 = tq.bolus("depot", 1.0, times=np.linspace(0, 20, 9))
    pk.solve((0, 24), times=[24], params=P, regimen=reg9, backend="jax")
    assert program._cache_size() - before <= 2  # 9 doses: the 16-slot build
    assert jax_backend.slots(1) == 8 and jax_backend.slots(9) == 16 and jax_backend.slots(16) == 16


def test_a_failure_says_where_and_suggests_scipy():
    m = tq.Model.from_text("compartment A\nd_t(A) = A^2\nA(0) = 1\n", name="blowup")
    with pytest.raises(tq.SolverError, match=r"failed at t = (0\.99|1).*backend='scipy'"):
        m.solve((0, 2), times=[0.5, 2], backend="jax")


# ----------------------------------------------------------------------------- 4.3c
# the condition models of test_conditionals.py, with their exact answers, on the JAX backend


def test_a_time_switch_is_a_boundary_and_its_time_a_value():
    m = tq.Model("on")
    A = m.compartment("A", init=0.0)
    m.params(t_on=5.0, kin=1.0)
    m.ode(A, "(kin if t > t_on else 0) - 0.1*A")
    t = np.array([0, 4, 5, 6, 20.0])
    for t_on in (5.0, 7.5):
        sol = m.solve((0, 20), times=t, params={"t_on": t_on}, backend="jax", **TIGHT)
        exact = np.where(t > t_on, 10 * (1 - np.exp(-0.1 * (t - t_on))), 0.0)
        np.testing.assert_allclose(sol["A"], exact, atol=1e-8)
        assert sol.stats["segments"] == 2


def test_a_short_pulse_and_a_sampled_condition_on_time():
    import jax.numpy as jnp

    pulse = tq.Model("pulse")
    pulse.ode(pulse.compartment("A", init=0.0), "1000 if 30 < t <= 30.001 else 0")
    end = pulse.solve((0, 100), times=[100], backend="jax", **TIGHT)["A"][-1]
    assert end == pytest.approx(1.0, rel=1e-8)
    days = tq.Model("days")
    days.ode(days.compartment("A", init=0.0), "1 if sin(t) > 0 else 0")
    cm = days.compile()
    found = np.asarray(cm.switch_times_jax(jnp.asarray(cm.theta0), 0.0, 10.0, 8))
    np.testing.assert_allclose(found[:3], cm.switch_times(None, (0, 10)), rtol=1e-15)
    sol = days.solve((0, 2 * np.pi + 1), times=[np.pi, 2 * np.pi + 1], backend="jax", **TIGHT)
    np.testing.assert_allclose(sol["A"], [np.pi, np.pi + 1], rtol=1e-8)


def test_state_switches_stop_exactly():
    m = tq.Model("sat")
    C = m.compartment("C", init=10.0)
    m.ode(C, -tq.cases((C > 2, 0.5), otherwise=0.1) * C)
    ts = np.log(5) / 0.5
    t = np.array([0, 1, ts, 5, 20.0])
    sol = m.solve((0, 20), times=t, backend="jax", **TIGHT)
    exact = np.where(t < ts, 10 * np.exp(-0.5 * t), 2 * np.exp(-0.1 * (t - ts)))
    np.testing.assert_allclose(sol["C"], exact, rtol=1e-8)
    assert sol.stats["n_events"] == 1 and max(sol.ledger.check().values()) < 1e-12
    crowd = tq.Model("crowd")
    x = crowd.axis("x", 0, 1, 6)
    B = crowd.field("B", axes=[x])
    B_tot = crowd.integral(B, over=x, name="B_tot")
    crowd.diffuse(B, axis=x, D=0.01)
    crowd.reaction(B, tq.cases((B_tot > 5, 0.3), otherwise=1.0) * B)
    crowd.init(B, 2.0)
    t = np.array([0, 0.5, 1.0, 2.0, 4.0])
    sol = crowd.solve((0, 4), times=t, backend="jax", **TIGHT)
    ts = np.log(5 / 2)
    exact = np.where(t < ts, 2 * np.exp(t), 5 * np.exp(0.3 * (t - ts)))
    np.testing.assert_allclose(crowd.compile().amounts(sol.y)["B"], exact, rtol=1e-8)


def test_switches_on_their_boundary_and_sliding():
    on = tq.Model("drug_on")
    C, B = on.compartment("C", init=0.0), on.compartment("B", init=1.0)
    on.ode(C, "-0.5*C")
    on.ode(B, tq.cases((C > 0, -1.0 * B), otherwise=0.3 * B))
    on.dose(C, times=[5.0], amounts=[1.0])
    sol = on.solve((0, 10), times=[5.0, 10.0], backend="jax", **TIGHT)
    np.testing.assert_allclose(sol["B"], [np.exp(1.5), np.exp(1.5 - 5)], rtol=1e-7)
    rest = tq.Model("at_rest")
    C, X = rest.compartment("C", init=1.0), rest.compartment("X", init=0.0)
    rest.ode(C, "-0.5*(C - 1)")
    rest.ode(X, tq.cases((C > 1.0, 1.0), otherwise=0.0))
    assert rest.solve((0, 5), times=[5], backend="jax")["X"][-1] == 0.0
    chatter = tq.Model("chatter")
    chatter.ode(chatter.compartment("X", init=1.0), "-1 if X > 0 else 1")
    with pytest.raises(tq.SolverError, match="'X > 0' had just switched at t = 1"):
        chatter.solve((0, 3), backend="jax")


def test_the_simulator_runs_on_either_backend(pk):
    sc = tq.Scenario(
        params=P,
        regimen=tq.bolus("depot", 100, start=0, every=12, n=3),
        times=np.linspace(0, 48, 25),
    )
    a = pk.simulator(outputs=["conc"], backend="jax", **TIGHT)(sc)
    b = pk.simulator(outputs=["conc"], **TIGHT)(sc)
    np.testing.assert_allclose(a["conc"], b["conc"], rtol=1e-9, atol=1e-12)


def test_v14_a_large_bolus_on_jax():
    """A 1e10 phage bolus into a logistic infection-age phage toy (test_v13_v14.bolus_model).
    Below an absolute tolerance of ~1e-4 the error estimate of the youngest infected cells is
    round-off
    (it stays ~5e-3 however small the step), and BDF survives only by chance, on scipy as on
    JAX (2 in 8 nudged runs each; measured 2026-10-24). With atol above it, both always pass:
    the bolus arrives in full, and the run matches scipy."""
    from test_v13_v14 import bolus_model

    m = bolus_model(1e8)
    sol = m.solve((0, 3), times=[2 - 5e-7, 2 + 5e-7, 3], rtol=1e-8, atol=1e-4, backend="jax")
    assert (sol["P"][1] - sol["P"][0]) / 1e10 == pytest.approx(1.0, abs=1e-4)
    ref = m.solve((0, 3), times=[2 - 5e-7, 2 + 5e-7, 3], rtol=1e-8, atol=1e-4)
    for name in ("B", "P"):  # B falls to ~1e-260 by t = 3: compare against the largest value
        scale = np.max(np.abs(ref[name]))
        np.testing.assert_allclose(sol[name] / scale, ref[name] / scale, atol=1e-5)
