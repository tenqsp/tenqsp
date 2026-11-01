# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``sim.function([sc1, sc2, ...], wrt=...)``: several scenarios (each with its own regimen,
output times, start and starting state) as one JAX function, run in one ``jax.vmap`` call of
one build. Oracles: the closed form of oral one-compartment PK (superposed over the
boluses), differentiated by JAX; and each scenario's own ``sim.function``, which the padding
(slots at t = +inf) must not change."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq
from tenqsp.gradients import FunctionBatch

TIGHT = {"rtol": 1e-10, "atol": 1e-12}
PK = """model pk
param ka=1.2, CL=4, V=30
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
observable conc = centr/V
"""


def closed(x, doses, t):
    """Oral one-compartment PK: conc at times t for boluses (time, amount) into depot."""
    ka, V = x[0], 30.0
    ke = x[1] / V
    out = jnp.zeros(len(t))
    for td, amt in doses:
        dt = jnp.asarray(t) - td
        c = amt * ka / (V * (ka - ke)) * (jnp.exp(-ke * dt) - jnp.exp(-ka * dt))
        out = out + jnp.where(dt >= 0, c, 0.0)
    return out


DOSES = [[(0.0, 100.0)], [(0.0, 50.0), (12.0, 80.0)], [(2.0, 30.0), (6.0, 30.0), (10.0, 30.0)]]
TIMES = [(0.5, 1, 2, 4, 8), tuple(np.linspace(0.5, 24, 17)), (3, 7, 11, 20)]


def scenarios():
    out = []
    for i, (doses, times) in enumerate(zip(DOSES, TIMES, strict=True)):
        reg = tq.Regimen()
        for td, amt in doses:
            reg = reg + tq.bolus("depot", amt, times=[td])
        out.append(
            tq.Scenario(name=f"s{i}", params={"ka": 1.0 + 0.3 * i}, regimen=reg, times=times)
        )
    return out


@pytest.fixture(scope="module")
def sim():
    return tq.Model.from_text(PK).simulator(outputs=["conc"], **TIGHT)


@pytest.mark.parametrize("mode", ["forward", "reverse"])
def test_values_and_derivatives_against_the_closed_form(sim, mode):
    scs = scenarios()
    fs = sim.function(scs, wrt=["ka", "CL"], derivatives=mode)
    assert isinstance(fs, FunctionBatch) and len(fs) == 3 and fs.derivatives == mode
    assert fs.names == ("ka", "CL")
    np.testing.assert_allclose(fs.x0, [[1.0, 4.0], [1.3, 4.0], [1.6, 4.0]])
    X = jnp.asarray(fs.x0) * jnp.array([1.1, 0.9])
    Y = fs(X)["conc"]
    J = jax.jacfwd(lambda X: fs(X)["conc"])(X)  # (scenarios, times, scenarios, names)
    assert Y.shape == (3, 17)
    for i in range(3):
        n = len(TIMES[i])
        exact = closed(X[i], DOSES[i], TIMES[i])
        np.testing.assert_allclose(Y[i, :n], exact, rtol=1e-7, atol=1e-10)
        assert np.all(np.isnan(Y[i, n:]))  # past this scenario's own times
        dexact = jax.jacfwd(lambda x, i=i: closed(x, DOSES[i], TIMES[i]))(X[i])
        np.testing.assert_allclose(J[i, :n, i], dexact, rtol=1e-6, atol=1e-9)
        others = [j for j in range(3) if j != i]
        assert np.all(J[i, :n, others] == 0)  # a scenario depends only on its own row


@pytest.mark.parametrize("mode", ["forward", "reverse", False])
def test_padding_changes_nothing(sim, mode):
    """A scenario padded to a larger one's size gives exactly what it gives unpadded."""
    s0, s1, s2 = scenarios()
    alone = sim.function([s0, s0], wrt=["ka", "CL"], derivatives=mode)
    for big in (s1, s2):
        padded = sim.function([s0, big], wrt=["ka", "CL"], derivatives=mode)
        np.testing.assert_array_equal(
            padded(padded.x0)["conc"][0, :5], alone(alone.x0)["conc"][0, :5]
        )
        if mode:
            g = [
                jax.grad(lambda X, f=f: jnp.sum(jnp.where(f.mask, f(X)["conc"], 0)[0]))(
                    jnp.asarray(f.x0)
                )[0]
                for f in (alone, padded)
            ]
            np.testing.assert_array_equal(g[0], g[1])


@pytest.mark.parametrize("mode", ["forward", "reverse", False])
def test_each_scenario_as_its_own_function(sim, mode):
    """The batch agrees with each scenario's own function within the solver's tolerance
    (vmap's batched arithmetic rounds differently, which can move a step; rtol 1e-10)."""
    scs = scenarios()
    fs = sim.function(scs, wrt=["ka", "CL"], derivatives=mode)
    Y = fs(fs.x0)["conc"]
    for i, sc in enumerate(scs):
        f = sim.function(sc, wrt=["ka", "CL"], derivatives=mode)
        y = f(f.x0)["conc"]
        np.testing.assert_allclose(Y[i, : len(y)], y, rtol=1e-8)
        if mode:
            g = jax.grad(lambda X, i=i: jnp.sum(jnp.where(fs.mask, fs(X)["conc"], 0)[i]))(
                jnp.asarray(fs.x0)
            )
            g1 = jax.grad(lambda x, f=f: jnp.sum(f(x)["conc"]))(f.x0)
            np.testing.assert_allclose(g[i], g1, rtol=1e-7)


def test_times_mask_and_a_masked_objective(sim):
    fs = sim.function(scenarios(), wrt=["ka", "CL"])
    assert list(fs.n_times) == [5, 17, 4]
    assert fs.mask.shape == fs.t.shape == (3, 17)
    np.testing.assert_array_equal(fs.t[2, :4], [3, 7, 11, 20])
    assert np.isnan(fs.t[2, 4]) and not fs.mask[2, 4] and fs.mask[1].all()
    data = np.where(fs.mask, 0.1, np.nan)

    def sse(X):
        r = jnp.where(fs.mask, fs(X)["conc"] - data, 0.0)
        return jnp.sum(r**2)

    g = jax.grad(sse)(jnp.asarray(fs.x0))
    assert g.shape == (3, 2) and np.all(np.isfinite(g))
    assert np.isfinite(jax.jit(sse)(jnp.asarray(fs.x0)))


def test_starts_starting_states_and_infusions(sim):
    a = tq.Scenario(regimen=tq.infusion("centr", rate=10, duration=2, start=1), times=(1, 2, 5))
    b = tq.Scenario(regimen=tq.Regimen(), times=(4, 6), start=3, initial={"centr": 60.0})
    with pytest.raises(ValueError, match="scenario 2 \\(no name\\): wrt: 'infusion 1 rate'"):
        sim.function([a, b], wrt=["CL", "infusion 1 rate"])
    fs = sim.function([a, b], wrt=["CL"])
    Y = fs(fs.x0)["conc"]
    for i, sc in enumerate([a, b]):
        f = sim.function(sc, wrt=["CL"])
        np.testing.assert_allclose(Y[i, : len(sc.times)], f(f.x0)["conc"], rtol=1e-12)
    np.testing.assert_allclose(Y[1, 0], 2.0 * np.exp(-4 / 30), rtol=1e-8)  # from t = 3


def test_dose_numbers_in_wrt(sim):
    scs = scenarios()
    fs = sim.function(scs, wrt=["dose 1 amount", "dose 1 time"])
    np.testing.assert_allclose(fs.x0, [[100, 0], [50, 0], [30, 2]])
    J = jax.jacfwd(lambda X: fs(X)["conc"])(jnp.asarray(fs.x0))
    exact = jax.jacfwd(lambda a: closed(jnp.array([1.6, 4.0]), [(2.0, a)], [3.0]))(30.0)
    np.testing.assert_allclose(J[2, 0, 2, 0], exact[0], rtol=1e-6)


def test_status_per_scenario(sim):
    fs = sim.function(scenarios(), wrt=["dose 1 time"])
    X = jnp.asarray(fs.x0).at[1, 0].set(-5.0)  # scenario 2's dose before the start
    assert fs.status(X) == ("ok", fs.status(X)[1], "ok")
    assert "before the start" in fs.status(X)[1]
    Y = fs(X)["conc"]
    assert np.all(np.isnan(Y[1])) and np.all(np.isfinite(Y[0, :5]))


def test_one_build_for_new_scenarios_of_any_size(sim):
    sim.function(scenarios(), wrt=["ka", "CL"])
    cm = sim.model.compile()
    before = {k for k in cm._lazy if isinstance(k, tuple) and k[0] == "batch"}
    more = [s.replace(times=s.times[:2]) for s in scenarios()]
    fs = sim.function(more, wrt=["ka", "CL"])
    after = {k for k in cm._lazy if isinstance(k, tuple) and k[0] == "batch"}
    assert after == before
    assert fs(fs.x0)["conc"].shape == (3, 2)


def test_errors(sim):
    sc = scenarios()[0]
    with pytest.raises(ValueError, match="at least one scenario"):
        sim.function([], wrt=["CL"])
    with pytest.raises(TypeError, match="Every scenario is a tq.Scenario"):
        sim.function([sc, {"CL": 3}], wrt=["CL"])
    with pytest.raises(ValueError, match="use the model's own doses"):
        sim.function([sc, tq.Scenario(times=(1, 2))], wrt=["CL"])
    fs = sim.function([sc, sc], wrt=["CL"])
    with pytest.raises(ValueError, match=r"X has one row per scenario"):
        fs(jnp.ones(2))
    assert "2 scenarios, up to 5 times" in repr(fs)


def test_the_models_own_doses():
    m = tq.Model.from_text(PK.replace("observable", "dose depot at t=0, 12: 100\nobservable"))
    sim = m.simulator(outputs=["conc"], **TIGHT)
    a, b = tq.Scenario(times=(1, 2, 4)), tq.Scenario(params={"ka": 2.0}, times=tuple(range(1, 20)))
    fs = sim.function([a, b], wrt=["ka", "CL"])
    Y = fs(fs.x0)["conc"]
    np.testing.assert_allclose(Y[0, :3], closed(jnp.array([1.2, 4.0]), [(0.0, 100.0)], a.times),
                               rtol=1e-7)  # fmt: skip
    exact = closed(jnp.array([2.0, 4.0]), [(0.0, 100.0), (12.0, 100.0)], b.times)
    np.testing.assert_allclose(Y[1], exact, rtol=1e-7)
