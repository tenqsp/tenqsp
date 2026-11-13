# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``f(x, rtol=..., atol=...)``: one call of ``sim.function`` at other tolerances than the
simulator's, with no new compile. Oracles: a simulator built at those tolerances (the same
program and the same inputs, so the same numbers), and the closed form of oral
one-compartment PK (a tighter tolerance comes closer to it)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq

PK = """model pk
param ka=1.2, CL=4, V=30
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
observable conc = centr/V
"""
T = np.array([0.5, 1, 2, 4, 8, 12.0])
SC = tq.Scenario(regimen=tq.bolus("depot", 100, times=[0]), times=T)


def closed(x, t=T):
    ka, CL = x
    ke = CL / 30.0
    return 100 * ka / (30.0 * (ka - ke)) * (jnp.exp(-ke * t) - jnp.exp(-ka * t))


@pytest.fixture(scope="module")
def m():
    return tq.Model.from_text(PK)


@pytest.mark.parametrize("derivatives", ["forward", "reverse", False])
def test_a_call_at_other_tolerances_is_a_simulator_built_with_them(m, derivatives):
    room = {"record_room": 4} if derivatives == "reverse" else {}
    f = m.simulator(outputs=["conc"]).function(
        SC, wrt=["ka", "CL"], derivatives=derivatives, **room
    )
    tight = {"rtol": 1e-8, "atol": 1e-10}  # within that reverse record's room (see below)
    g = m.simulator(outputs=["conc"], **tight).function(
        SC, wrt=["ka", "CL"], derivatives=derivatives
    )
    x = jnp.asarray(f.x0)
    np.testing.assert_array_equal(f(x, **tight)["conc"], g(x)["conc"])
    assert f.status(x, **tight) == "ok"
    if derivatives:
        jac = lambda h, **o: jax.jacfwd(lambda v: h(v, **o)["conc"])(x)  # noqa: E731
        np.testing.assert_allclose(jac(f, **tight), jac(g), rtol=1e-12, atol=1e-15)
    # without them, the simulator's own (rtol = atol = 1e-6): further from the closed form
    exact = closed(x)
    err = lambda out: np.max(np.abs(out["conc"] - exact))  # noqa: E731
    assert err(f(x)) > 10 * err(f(x, **tight))


def test_no_new_compile(m):
    f = m.simulator(outputs=["conc"]).function(SC, wrt=["ka", "CL"])
    x = jnp.asarray(f.x0)
    f(x)
    loss = jax.jit(lambda v, r: jnp.sum(f(v, rtol=r)["conc"] ** 2))
    loss(x, 1e-6)
    before = (f._impl._cache_size(), loss._cache_size())
    for r in (1e-4, 1e-8, 1e-10):
        f(x, rtol=r, atol=r)
        loss(x, r)
    jax.grad(loss)(x, 1e-9)
    assert (f._impl._cache_size(), loss._cache_size()) == before


def test_one_atol_per_state_and_mistakes(m):
    f = m.simulator(outputs=["conc"]).function(SC, wrt=["ka"])
    x = jnp.asarray(f.x0)
    np.testing.assert_array_equal(f(x, atol=[1e-9, 1e-9])["conc"], f(x, atol=1e-9)["conc"])
    with pytest.raises(ValueError, match="2 positive numbers"):
        f(x, atol=[1e-9, 1e-9, 1e-9])
    with pytest.raises(ValueError, match="rtol must be positive"):
        f(x, rtol=0.0)
    with pytest.raises(ValueError, match="atol must be positive"):
        f.status(x, atol=-1.0)
    with pytest.raises(TypeError):
        f(x, 1e-8)  # keyword only: a tolerance is never taken for a value of x


def test_several_scenarios(m):
    scs = [SC, SC.replace(regimen=tq.bolus("depot", 50, times=[0, 6]), times=tuple(T[:4]))]
    tight = {"rtol": 1e-10, "atol": 1e-12}
    fs = m.simulator(outputs=["conc"]).function(scs, wrt=["ka", "CL"])
    gs = m.simulator(outputs=["conc"], **tight).function(scs, wrt=["ka", "CL"])
    X = jnp.asarray(fs.x0)
    np.testing.assert_array_equal(fs(X, **tight)["conc"], gs(X)["conc"])
    assert fs.status(X, **tight) == ("ok", "ok")


def test_a_much_tighter_tolerance_can_fill_the_reverse_record():
    """The record was sized at the simulator's tolerance: a run needing far more steps gives
    NaN and says so (forward derivatives keep no record, so they simply run)."""
    # a model of its own: a build of the same shape with a somewhat larger record is shared
    own = tq.Model.from_text(PK.replace("model pk", "model pk_record"))
    sim = own.simulator(outputs=["conc"], rtol=1e-3, atol=1e-3)
    f = sim.function(SC, wrt=["ka", "CL"], derivatives="reverse")
    x = jnp.asarray(f.x0)
    assert "record" in f.status(x, rtol=1e-12, atol=1e-14)
    assert np.all(np.isnan(f(x, rtol=1e-12, atol=1e-14)["conc"]))
    g = sim.function(SC, wrt=["ka", "CL"], derivatives="forward")
    assert g.status(x, rtol=1e-12, atol=1e-14) == "ok"
