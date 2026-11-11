# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Models given by formulas of t (no differential equations; NONMEM's $PRED). Oracles: the
formulas written with NumPy, their derivatives written by hand, and the same model written
with equations and solved."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq

PROGRESSION = """model progression
param E0 = 50, prog = 2
observable E = E0 + prog*t
observable late = 1 if t >= 12 else 0
"""

SINGLE_DOSE = """model theoph_sd
param D = 320, ka = 1.5, V = 0.5, ke = 0.09
observable conc = D*ka/(V*(ka - ke))*(exp(-ke*t) - exp(-ka*t))
"""

SINGLE_DOSE_ODE = """model theoph_ode
param D = 320, ka = 1.5, V = 0.5, ke = 0.09
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - ke*centr
depot(0) = D
observable conc = centr/V
"""

TIMES = [0.25, 0.5, 1, 2, 3.5, 5, 7, 9, 12, 24]


def closed_form(t, D=320, ka=1.5, V=0.5, ke=0.09):
    t = np.asarray(t, float)
    return D * ka / (V * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def test_solve_evaluates_the_formulas_at_the_output_times():
    m = tq.Model.from_text(PROGRESSION)
    sol = m.solve((0, 52), times=[0, 4, 12, 12, 26, 52])
    np.testing.assert_array_equal(sol.t, [0, 4, 12, 12, 26, 52])
    np.testing.assert_array_equal(sol["E"], [50, 58, 74, 74, 102, 154])
    np.testing.assert_array_equal(sol["late"], [0, 0, 1, 1, 1, 1])  # a condition on t itself
    assert sol.y.shape == (6, 0) and sol.ledger is None
    np.testing.assert_array_equal(m.solve((0, 52), times=[1, 2], params={"prog": 3})["E"], [53, 56])


def test_both_backends_and_any_tolerance_give_the_formulas_exactly():
    m = tq.Model.from_text(SINGLE_DOSE)
    want = closed_form(TIMES)
    for backend in ("scipy", "jax"):
        for rtol in (1e-3, 1e-10):
            sol = m.solve((0, 24), times=TIMES, backend=backend, rtol=rtol)
            np.testing.assert_allclose(sol["conc"], want, rtol=1e-14)
            assert sol.options["backend"] == backend


def test_the_formula_equals_the_same_model_written_with_equations():
    f = tq.Model.from_text(SINGLE_DOSE).solve((0, 24), times=TIMES)["conc"]
    o = tq.Model.from_text(SINGLE_DOSE_ODE).solve((0, 24), times=TIMES, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(f, o["conc"], rtol=1e-7)


def test_function_gives_the_exact_derivatives():
    sim = tq.Simulator(tq.Model.from_text(SINGLE_DOSE), outputs=["conc"])
    f = sim.function(tq.Scenario(times=TIMES), wrt=["ka", "V", "ke"])
    x = jnp.array([1.5, 0.5, 0.09])
    np.testing.assert_allclose(f(x)["conc"], closed_form(TIMES), rtol=1e-14)
    assert f.status(x) == "ok"
    J = np.asarray(jax.jacfwd(lambda x: f(x)["conc"])(x))
    Jr = np.asarray(jax.jacrev(lambda x: f(x)["conc"])(x))
    t = np.asarray(TIMES)
    ka, V, ke, D = 1.5, 0.5, 0.09, 320.0
    c = closed_form(t)
    e_ke, e_ka = np.exp(-ke * t), np.exp(-ka * t)
    a = D * ka / (V * (ka - ke))
    d_ka = c * (1 / ka - 1 / (ka - ke)) + a * t * e_ka
    d_V = -c / V
    d_ke = c / (ka - ke) - a * t * e_ke
    want = np.stack([d_ka, d_V, d_ke], axis=1)
    np.testing.assert_allclose(J, want, rtol=1e-12)
    np.testing.assert_allclose(Jr, want, rtol=1e-12)


def test_function_without_derivatives_and_a_padded_batch():
    sim = tq.Simulator(tq.Model.from_text(PROGRESSION), outputs=["E"])
    fs = sim.function(
        [tq.Scenario(times=[0, 12]), tq.Scenario(times=[1, 2, 52])], wrt=["E0", "prog"]
    )
    out = np.asarray(fs(fs.x0)["E"])
    np.testing.assert_array_equal(out[0, :2], [50, 74])
    assert np.isnan(out[0, 2])  # past the first scenario's times
    np.testing.assert_array_equal(out[1], [52, 54, 154])
    assert fs.status(fs.x0) == ("ok", "ok")
    g = sim.function(tq.Scenario(times=[1, 10]), wrt=["prog"], derivatives=False)
    np.testing.assert_array_equal(g(g.x0)["E"], [52, 70])
    with pytest.raises(ValueError, match="derivatives=False"):
        jax.grad(lambda x: g(x)["E"].sum())(g.x0)
    np.testing.assert_array_equal(
        sim.sensitivity(tq.Scenario(times=[1, 10]), wrt=["prog"])["E"]["prog"], [1, 10]
    )
    np.testing.assert_array_equal(sim.batch([tq.Scenario(times=[1, 2])])[0]["E"], [52, 54])


def test_vmap_over_parameters():
    sim = tq.Simulator(tq.Model.from_text(SINGLE_DOSE), outputs=["conc"])
    f = sim.function(tq.Scenario(times=TIMES), wrt=["ka", "ke"])
    X = jnp.array([[1.5, 0.09], [1.0, 0.2]])
    out = np.asarray(jax.vmap(lambda x: f(x)["conc"])(X))
    np.testing.assert_allclose(out[1], closed_form(TIMES, ka=1.0, ke=0.2), rtol=1e-14)


def test_the_python_api():
    m = tq.Model("progression")
    m.param("E0", 50)
    m.param("prog", 2)
    m.observable("E", "E0 + prog*t")
    np.testing.assert_array_equal(m.solve((0, 1), times=[0, 1])["E"], [50, 52])


def test_text_latex_and_explain():
    m = tq.Model.from_text(PROGRESSION)
    again = tq.Model.from_text(m.to_text())
    np.testing.assert_array_equal(again.solve((0, 9), times=[9])["E"], [68])
    assert r"E(t) &= E_{0} + \mathrm{prog} \, t" in m.to_latex()
    text = m.explain()
    assert "no states, given by formulas of t" in text and "E = E0 + prog*t" in text
    assert "no solver" in text


HINT = "a model given by formulas of t has no doses"


@pytest.mark.parametrize(
    "line", ["dose E at t=0: 1", "E(0) = 3", "lag E = 1", "extinction E below 1"]
)
def test_statements_that_need_a_state_are_refused_with_the_way_out(line):
    with pytest.raises(tq.NotationError, match=HINT) as e:
        tq.Model.from_text(PROGRESSION + line + "\n")
    assert "D = DOSE" in str(e.value)


def test_a_regimen_and_a_pre_run_are_refused():
    m = tq.Model.from_text(PROGRESSION)
    with pytest.raises(ValueError, match="a dose has nowhere to go.*D = DOSE"):
        m.solve((0, 9), times=[1], regimen=tq.bolus("E", 1, times=[0]))
    with pytest.raises(ValueError, match="no state to run ahead to"):
        tq.Simulator(m).pre_run(tq.Scenario(times=[1]))


def test_a_model_with_states_still_needs_no_observable():
    m = tq.Model.from_text("model decay\nparam k = 1\nd_t(A) = -k*A\nA(0) = 1\n")
    np.testing.assert_allclose(m.solve((0, 1), times=[1])["A"], [np.exp(-1)], rtol=1e-5)
