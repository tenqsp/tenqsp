# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Step 4.4d: ``sim.function`` and ``sim.sensitivity`` (design note
docs/design/step4_4d_api.md §2-§4; design decisions 2026-10-25). Oracles: closed forms of
one-compartment PK (oral boluses with a lag and F, zero-order infusions) differentiated by
JAX, forward = reverse, the scaling identity d log C / d log CL + d log C / d log V = -1, and
the simulator's own outputs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq
from tenqsp.backends.scipy_backend import SolverError

TIGHT = {"rtol": 1e-10, "atol": 1e-12}
T = np.array([1, 2, 4, 7, 12.0])
ORAL = """model pk
param ka = 1.2, CL = 4, V = 30, tlag = 0.5, F = 0.8
compartment depot, centr
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
lag depot = tlag
bioavailability depot = F
observable conc = centr/V
"""


def oral_conc(p, t=T):
    """conc for two boluses into depot (amounts a1, a2 at times 0 and t2), lag and F."""
    ka, CL, V, tlag, F, a1, t2 = p
    k = CL / V
    out = 0.0
    for a, td in ((a1, 0.0), (100.0, t2)):
        s = jnp.maximum(t - td - tlag, 0.0)
        out = out + F * a * ka / (ka - k) * (jnp.exp(-k * s) - jnp.exp(-ka * s))
    return out / V


def iv_conc(CL, V, rate, start, dur, t=T):
    """conc for a zero-order infusion into centr."""
    k = CL / V

    def up(s):
        s = jnp.maximum(s, 0.0)
        return rate / CL * (1 - jnp.exp(-k * s))

    return up(t - start) - up(t - start - dur)


@pytest.fixture(scope="module")
def oral():
    m = tq.Model.from_text(ORAL)
    sim = m.simulator(outputs=["conc"], **TIGHT)
    sc = tq.Scenario(regimen=tq.bolus("depot", 100, times=[0, 5]), times=T)
    return m, sim, sc


def test_f_is_the_simulation_as_a_function_of_x(oral):
    m, sim, sc = oral
    f = sim.function(sc, wrt=["CL", "V", "tlag"])
    assert f.names == ("CL", "V", "tlag") and np.allclose(f.x0, [4, 30, 0.5])
    assert f.unpack(f.x0) == {"CL": 4.0, "V": 30.0, "tlag": 0.5}
    y = np.asarray(f(f.x0)["conc"])
    assert y.shape == (T.size,)
    assert np.allclose(y, sim(sc)["conc"], rtol=1e-8, atol=1e-12)
    # at another x: a new run there, not derivatives extrapolated from x0
    x = jnp.array([6.0, 20.0, 1.0])
    other = sim(sc, params={"CL": 6.0, "V": 20.0, "tlag": 1.0})["conc"]
    assert np.allclose(np.asarray(f(x)["conc"]), other, rtol=1e-8, atol=1e-12)
    assert f.status(x) == "ok"


def test_v8_derivatives_match_the_closed_form(oral):
    _, sim, sc = oral
    names = ["ka", "CL", "V", "tlag", "F", "dose 1 amount", "dose 2 time"]
    f = sim.function(sc, wrt=names)
    assert np.allclose(f.x0, [1.2, 4, 30, 0.5, 0.8, 100, 5])
    J = np.asarray(jax.jacfwd(lambda x: f(x)["conc"])(f.x0))
    R = np.asarray(jax.jacrev(lambda x: f(x)["conc"])(f.x0))
    exact = np.asarray(jax.jacfwd(oral_conc)(jnp.asarray(f.x0)))
    assert np.allclose(J, exact, rtol=1e-6, atol=1e-9)
    assert np.allclose(J, R, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(
    "what, held",
    [("duration", "rate"), ("rate", "duration"), ("amount", "duration"), ("start", "duration")],
)
def test_infusion_numbers_and_what_each_keeps(what, held):
    m = tq.Model.from_text(
        "param CL = 4, V = 30\ncompartment centr\nd_t(centr) = -CL/V*centr\n"
        "observable conc = centr/V\n"
    )
    sim = m.simulator(outputs=["conc"], **TIGHT)
    # starts and stops off the output times (there the derivative is one-sided)
    sc = tq.Scenario(regimen=tq.infusion("centr", rate=50, duration=2, start=0.5), times=T)
    f = sim.function(sc, wrt=["CL", f"infusion 1 {what}"])
    x0 = {"duration": 2.0, "rate": 50.0, "amount": 100.0, "start": 0.5}[what]
    assert f.x0[1] == x0

    def exact(x):
        CL, v = x
        rate, start, dur = 50.0, 0.5, 2.0
        if what == "duration":
            dur = v  # the rate is kept: the amount follows
        elif what == "rate":
            rate = v
        elif what == "amount":
            rate = v / dur  # the duration is kept
        else:
            start = v  # both ends move
        return iv_conc(CL, 30.0, rate, start, dur)

    J = np.asarray(jax.jacfwd(lambda x: f(x)["conc"])(f.x0))
    assert np.allclose(J, np.asarray(jax.jacfwd(exact)(jnp.asarray(f.x0))), rtol=1e-6, atol=1e-9)
    assert held in ("rate", "duration")


def test_failures_give_nan_with_a_reason(oral):
    _, sim, sc = oral
    f = sim.function(sc, wrt=["tlag"])
    y = f(jnp.array([-1.0]))["conc"]
    assert np.all(np.isnan(y))
    assert f.status(jnp.array([-1.0])) == "a lag or bioavailability below 0 or not finite"
    g = jax.grad(lambda x: jnp.sum(f(x)["conc"]))(jnp.array([-1.0]))
    assert np.all(np.isnan(g))
    assert np.all(np.isfinite(jax.grad(lambda x: jnp.sum(f(x)["conc"]))(f.x0)))


def test_a_run_needing_more_steps_than_the_record_holds_is_nan():
    m = tq.Model.from_text("param w = 1\ncompartment X, Y\nd_t(X) = w*Y\nd_t(Y) = -w*X\nX(0) = 1\n")
    sim = m.simulator(outputs=["X"], rtol=1e-8, atol=1e-10)
    f = sim.function(tq.Scenario(times=[10.0]), wrt=["w"], derivatives="reverse")
    assert np.isfinite(f(f.x0)["X"][0]) and np.isclose(f(f.x0)["X"][0], np.cos(10), atol=1e-6)
    fast = jnp.array([200.0])
    assert (
        f.status(fast) == "too many steps for the record (make a new sim.function(...) at this x)"
    )
    assert np.isnan(f(fast)["X"][0])
    # forward sensitivities keep no record: the same x simply runs
    g = sim.function(tq.Scenario(times=[10.0]), wrt=["w"], derivatives="forward")
    assert g.status(fast) == "ok" and np.isclose(g(fast)["X"][0], np.cos(2000), atol=1e-4)


def test_jit_vmap_and_compile_once(oral):
    m, sim, sc = oral
    f = sim.function(sc, wrt=["CL", "V"])
    impl = f._impl
    loss = jax.jit(jax.grad(lambda x: jnp.sum(f(x)["conc"] ** 2)))
    loss(f.x0)
    f(f.x0)
    before = (impl._cache_size(), loss._cache_size())
    xs = f.x0 * (1 + 0.05 * np.random.default_rng(0).standard_normal((20, 2)))
    for x in xs:
        f(x)
        loss(x)
    assert (impl._cache_size(), loss._cache_size()) == before
    batch = jax.vmap(lambda x: f(x)["conc"])(jnp.asarray(xs[:4]))
    # batched linear algebra rounds differently, so steps may differ: equal to the tolerance
    assert np.allclose(batch, np.stack([f(x)["conc"] for x in xs[:4]]), rtol=1e-9, atol=0)
    # another scenario of the same shape (other times, values, doses) shares the build
    other = tq.Scenario(
        regimen=tq.bolus("depot", 50, times=[1, 3]), times=T + 0.5, params={"ka": 2.0}
    )
    g = sim.function(other, wrt=["V", "CL"])
    assert g._impl is impl
    g(g.x0)
    assert impl._cache_size() == before[0]


def test_wrt_names(oral):
    m = tq.Model.from_text(
        "param g = 1\nparam k[s] = {S: 0.5, R: 0.2}\naxis s categorical [S, R]\nfield B(s)\n"
        "d_t(B) = (g - k[s])*B\nB(0) = 1\n"
    )
    sim = m.simulator(outputs=["B"], **TIGHT)
    f = sim.function(tq.Scenario(times=[1, 2]), wrt=["k", "g"])
    assert f.names == ("k[S]", "k[R]", "g")
    assert f(f.x0)["B"].shape == (2, 2)
    one = sim.function(tq.Scenario(times=[1, 2]), wrt=["k[ R ]"])
    assert one.names == ("k[R]",)
    J = jax.jacfwd(lambda x: one(x)["B"])(one.x0)
    assert np.allclose(J[:, 1, 0], -np.array([1, 2]) * np.exp(np.array([1, 2]) * 0.8), rtol=1e-6)
    assert np.all(J[:, 0, 0] == 0)
    _, osim, osc = oral
    cases = [
        (["Cl"], "Did you mean 'CL'"),
        (["centr"], "'centr' is a compartment, not something to differentiate"),
        (["conc"], "an observable"),
        (["dose 3 time"], "the regimen has 2 dose"),
        (["infusion 1 rate"], "the regimen has 0 infusion"),
        (["CL", "CL"], "given twice"),
        ([], "at least one"),
    ]
    for wrt, message in cases:
        with pytest.raises(ValueError, match=message):
            osim.function(osc, wrt=wrt)
    with pytest.raises(ValueError, match="uses the model's own doses"):
        osim.function(tq.Scenario(times=T), wrt=["dose 1 time"])
    reg = tq.infusion("centr", rate=1, duration=2)
    with pytest.raises(ValueError, match="its rate or its amount, not both"):
        osim.function(
            tq.Scenario(regimen=reg, times=T), wrt=["infusion 1 rate", "infusion 1 amount"]
        )


def test_sensitivity_table_and_relative(oral):
    _, sim, sc = oral
    s = sim.sensitivity(sc, wrt=["CL", "V"])
    f = sim.function(sc, wrt=["CL", "V"])
    J = np.asarray(jax.jacfwd(lambda x: f(x)["conc"])(f.x0))
    assert np.allclose(s["conc"]["CL"], J[:, 0]) and np.allclose(s["conc"]["V"], J[:, 1])
    assert np.allclose(s.values["conc"], sim(sc)["conc"], rtol=1e-8)
    r = sim.sensitivity(sc, wrt=["CL", "V"], relative=True)
    # doubling CL and V together halves the concentration: the two add up to -1
    assert np.allclose(r["conc"]["CL"] + r["conc"]["V"], -1.0, atol=1e-8)
    table = r.table()
    assert list(table.columns) == [("conc", "CL"), ("conc", "V")]
    assert np.allclose(table.index, T)
    # where the output is 0, relative sensitivity is undefined
    early = sc.replace(times=[0.25, 1.0])
    z = sim.sensitivity(early, wrt=["CL"], relative=True)
    assert np.isnan(z["conc"]["CL"][0]) and np.isfinite(z["conc"]["CL"][1])


def test_sensitivity_raises_on_a_failed_run(oral):
    m, _, sc = oral
    sim = m.simulator(outputs=["conc"], max_steps=5)
    with pytest.raises(SolverError, match="too many steps"):
        sim.sensitivity(sc, wrt=["CL"])
