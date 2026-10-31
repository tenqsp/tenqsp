# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``sim.function(..., derivatives=)``: forward sensitivities or record-and-replay (design
decision 2026-10-31: "auto" picks forward up to 40 names in ``wrt``, reverse above; either
can be asked for). Oracles: closed forms differentiated by JAX (oral PK with a lag, F and
two boluses; a decay whose rate switches at a level, the switch time moving with the
parameters; a decay that goes extinct), and forward = reverse."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq
from tenqsp.gradients import FORWARD_MAX

TIGHT = {"rtol": 1e-10, "atol": 1e-12}
T = np.array([1, 2, 4, 7, 12.0])
MODES = ["forward", "reverse"]


def chain_of(n_params):
    k = ", ".join(f"k{i}=0.01" for i in range(n_params))
    total = " + ".join(f"k{i}" for i in range(n_params))
    m = tq.Model.from_text(f"param {k}\ncompartment A\nd_t(A) = -({total})*A\nA(0) = 1\n")
    return m.simulator(outputs=["A"]), [f"k{i}" for i in range(n_params)]


def test_auto_is_forward_up_to_40_names_and_reverse_above():
    sc = tq.Scenario(times=[1.0, 2.0])
    sim, names = chain_of(FORWARD_MAX + 1)
    assert FORWARD_MAX == 40
    assert sim.function(sc, wrt=names[:FORWARD_MAX]).derivatives == "forward"
    assert sim.function(sc, wrt=names).derivatives == "reverse"
    assert sim.function(sc, wrt=names, derivatives="forward").derivatives == "forward"
    assert sim.function(sc, wrt=names[:2], derivatives="reverse").derivatives == "reverse"
    assert sim.function(sc, wrt=names[:2], derivatives=True).derivatives == "forward"  # = auto
    assert sim.function(sc, wrt=names[:2], derivatives=False).derivatives is False


@pytest.mark.parametrize("bad", ["replay", "Forward", 1, None])
def test_other_values_of_derivatives_raise(bad):
    sim, names = chain_of(2)
    with pytest.raises(TypeError, match="'auto' \\(the default\\), 'forward', 'reverse' or False"):
        sim.function(tq.Scenario(times=[1.0]), wrt=names, derivatives=bad)


def test_reverse_keeps_a_record_and_forward_does_not():
    sim, names = chain_of(2)
    sc = tq.Scenario(times=[1.0])
    assert sim.function(sc, wrt=names, derivatives="reverse").capacity > 0
    assert sim.function(sc, wrt=names, derivatives="forward").capacity == 0


ORAL = """model pk
param ka = 1.2, CL = 4, V = 30, tlag = 0.5, F = 0.8
compartment depot, centr
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
lag depot = tlag
bioavailability depot = F
observable conc = centr/V
"""
ORAL_WRT = ["ka", "CL", "V", "tlag", "F", "dose 1 amount", "dose 2 time"]


def oral_conc(p, t=T):
    """conc for two boluses into depot (amounts a1 and 100 at times 0 and t2), lag and F."""
    ka, CL, V, tlag, F, a1, t2 = p
    k = CL / V
    out = 0.0
    for a, td in ((a1, 0.0), (100.0, t2)):
        s = jnp.maximum(t - td - tlag, 0.0)
        out = out + F * a * ka / (ka - k) * (jnp.exp(-k * s) - jnp.exp(-ka * s))
    return out / V


@pytest.fixture(scope="module")
def oral():
    sim = tq.Model.from_text(ORAL).simulator(outputs=["conc"], **TIGHT)
    sc = tq.Scenario(regimen=tq.bolus("depot", 100, times=[0, 5]), times=T)
    return sim, sc


@pytest.mark.parametrize("mode", MODES)
def test_oral_values_jacobian_and_gradient_match_the_closed_form(oral, mode):
    sim, sc = oral
    f = sim.function(sc, wrt=ORAL_WRT, derivatives=mode)
    x = jnp.asarray(f.x0)
    np.testing.assert_allclose(f(x)["conc"], oral_conc(x), rtol=1e-7)
    J = jax.jacfwd(lambda x: f(x)["conc"])(x)
    np.testing.assert_allclose(J, jax.jacfwd(oral_conc)(x), rtol=1e-5, atol=1e-8)
    data = oral_conc(x) * 1.1

    def loss(fn):
        return lambda x: jnp.sum((fn(x) - data) ** 2)

    g = jax.grad(loss(lambda x: f(x)["conc"]))(x)
    np.testing.assert_allclose(g, jax.grad(loss(oral_conc))(x), rtol=1e-5, atol=1e-10)


def test_forward_and_reverse_agree_under_vmap(oral):
    sim, sc = oral
    fs = [sim.function(sc, wrt=ORAL_WRT, derivatives=mode) for mode in MODES]
    X = jnp.asarray(fs[0].x0) * jnp.asarray([[1.0], [1.2], [0.8]])
    jac = [jax.vmap(jax.jacfwd(lambda x, f=f: f(x)["conc"]))(X) for f in fs]
    np.testing.assert_allclose(jac[0], jac[1], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(jac[0], jax.vmap(jax.jacfwd(oral_conc))(X), rtol=1e-5, atol=1e-8)


SWITCH = """model switch
param k1 = 0.5, k2 = 0.1, c = 20
compartment A
d_t(A) = -(k1 if A > c else k2)*A
A(0) = 100
"""


def switch_A(p, t):
    """A0 e^{-k1 t} until A reaches c at ts = ln(A0/c)/k1, then c e^{-k2 (t - ts)}."""
    k1, k2, c = p
    ts = jnp.log(100.0 / c) / k1
    return jnp.where(t < ts, 100.0 * jnp.exp(-k1 * t), c * jnp.exp(-k2 * (t - ts)))


@pytest.mark.parametrize("mode", MODES)
def test_a_switch_time_that_moves_with_the_parameters(mode):
    t = np.array([1.0, 2.0, 4.0, 8.0, 15.0])  # the switch is at ln 5 / 0.5 = 3.2
    sim = tq.Model.from_text(SWITCH).simulator(outputs=["A"], **TIGHT)
    f = sim.function(tq.Scenario(times=t), wrt=["k1", "k2", "c"], derivatives=mode)
    x = jnp.asarray(f.x0)
    np.testing.assert_allclose(f(x)["A"], switch_A(x, t), rtol=1e-7)
    J = jax.jacfwd(lambda x: f(x)["A"])(x)
    np.testing.assert_allclose(J, jax.jacfwd(switch_A)(x, t), rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("mode", MODES)
def test_an_extinct_population_has_zero_derivative_after_the_crossing(mode):
    m = tq.Model("decay")
    B = m.compartment("B", init=100.0)
    k = m.param("k", 0.5)
    m.ode(B, -k * B)
    m.extinction(B, threshold=1.0)
    t = np.array([2.0, 5.0, 12.0])  # extinct at ln 100 / 0.5 = 9.2
    f = m.simulator(outputs=["B"], **TIGHT).function(tq.Scenario(times=t), wrt=["k"],
                                                      derivatives=mode)  # fmt: skip
    J = np.asarray(jax.jacfwd(lambda x: f(x)["B"])(jnp.asarray(f.x0)))[:, 0]
    np.testing.assert_allclose(J[:2], -t[:2] * 100 * np.exp(-0.5 * t[:2]), rtol=1e-6)
    assert J[2] == 0.0


@pytest.mark.parametrize("mode", MODES)
def test_a_failed_run_gives_nan_and_says_why(oral, mode):
    sim, sc = oral
    f = sim.function(sc, wrt=["CL", "dose 2 time"], derivatives=mode)
    bad = jnp.asarray([4.0, -1.0])  # the 2nd dose moved before the start
    assert f.status(bad).startswith("a dose or infusion moved before the start")
    assert np.all(np.isnan(f(bad)["conc"]))
    g = jax.grad(lambda x: jnp.sum(f(x)["conc"]))(bad)
    assert np.all(np.isnan(g))
