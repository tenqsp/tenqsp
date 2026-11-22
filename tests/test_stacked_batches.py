# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Stacked batches (ADR 0010): ``sim.function([sc1, sc2, ...], wrt=..., batch="stacked")`` runs
the scenarios as one system, with one sequence of steps and the error norm of the least easy
unit. Oracles: the closed form of oral one-compartment PK (superposed over the boluses), the
same batch with each scenario's own steps (``batch="vmap"``), and central differences."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_function_batch import PK, TIGHT, closed

import tenqsp as tq

# doses at different times and of different numbers, output times of different lengths
DOSES = [[(0.0, 100.0)], [(0.0, 50.0), (12.0, 80.0)], [(2.0, 30.0), (6.0, 30.0), (10.0, 30.0)]]
TIMES = [(0.5, 1, 2, 4, 8), tuple(np.linspace(0.5, 24, 17)), (3, 7, 11, 20)]
INFUSED = """model inf
param CL=4, V=30, k12=0.5, k21=0.3
d_t(centr) = -CL/V*centr - k12*centr + k21*periph
d_t(periph) = k12*centr - k21*periph
observable conc = centr/V
"""


@pytest.fixture(scope="module")
def m():
    return tq.Model.from_text(PK.replace("model pk", "model pk_stacked"))


def scenarios():
    out = []
    for i, (doses, times) in enumerate(zip(DOSES, TIMES, strict=True)):
        reg = tq.Regimen()
        for td, amt in doses:
            reg = reg + tq.bolus("depot", amt, times=[td])
        out.append(tq.Scenario(params={"ka": 1.0 + 0.3 * i}, regimen=reg, times=times))
    return out


@pytest.mark.parametrize("mode", ["forward", False])
def test_values_and_derivatives_against_the_closed_form(m, mode):
    fs = m.simulator(outputs=["conc"], **TIGHT).function(
        scenarios(), wrt=["ka", "CL"], derivatives=mode, batch="stacked"
    )
    assert fs.batch == "stacked"
    X = jnp.asarray(fs.x0)
    Y = fs(X)["conc"]
    for i, (doses, times) in enumerate(zip(DOSES, TIMES, strict=True)):
        np.testing.assert_allclose(Y[i, : len(times)], closed(X[i], doses, times), rtol=1e-8)
        assert np.all(np.isnan(Y[i, len(times) :]))  # past its own times
    if mode:
        J = jax.jacfwd(lambda X_: jnp.where(fs.mask, fs(X_)["conc"], 0.0))(X)
        for i, (doses, times) in enumerate(zip(DOSES, TIMES, strict=True)):
            Jc = jax.jacfwd(lambda x, d=doses, t=times: closed(x, d, t))(X[i])
            np.testing.assert_allclose(J[i, : len(times), i], Jc, rtol=1e-6, atol=1e-10)
            others = [j for j in range(3) if j != i]
            assert np.all(J[i, :, others] == 0)  # a scenario depends only on its own row


def test_stacked_agrees_with_one_by_one_within_the_tolerance(m):
    """At the default tolerance a scenario's values depend slightly on the batch (the shared
    steps are chosen for the least easy one): within rtol of each scenario's own run."""
    sim = m.simulator(outputs=["conc"])
    st = sim.function(scenarios(), wrt=["ka", "CL"], batch="stacked")
    vm = sim.function(scenarios(), wrt=["ka", "CL"], batch="vmap")
    X = jnp.asarray(st.x0)
    a, b = st(X)["conc"], vm(X)["conc"]
    np.testing.assert_allclose(a[st.mask], b[st.mask], rtol=2e-5, atol=1e-8)
    ga = jax.grad(lambda X_: jnp.sum(jnp.where(st.mask, st(X_)["conc"], 0.0)))(X)
    gb = jax.grad(lambda X_: jnp.sum(jnp.where(vm.mask, vm(X_)["conc"], 0.0)))(X)
    np.testing.assert_allclose(ga, gb, rtol=1e-4)


def test_dose_times_and_amounts_in_wrt(m):
    """Derivatives by each scenario's own dose time and amount, against central differences
    of the values (tight tolerances). The step is not small: moving one scenario's dose moves
    a boundary of the shared run, so every scenario's steps change, and the values move by
    round-off of the solver's tolerance (a difference step of 1e-6 sees it)."""
    sim = m.simulator(outputs=["conc"], **TIGHT)
    fs = sim.function(scenarios(), wrt=["dose 1 time", "dose 1 amount"], batch="stacked")
    X = jnp.asarray(fs.x0).at[:, 0].add(0.25)  # off the output times

    def total(X_):
        return jnp.sum(jnp.where(fs.mask, fs(X_)["conc"], 0.0) ** 2)

    g = jax.grad(total)(X)
    h = 1e-4
    for i in range(3):
        for k in range(2):
            E = jnp.zeros_like(X).at[i, k].set(h * max(1.0, abs(float(X[i, k]))))
            fd = (total(X + E) - total(X - E)) / (2 * E[i, k])
            np.testing.assert_allclose(g[i, k], fd, rtol=1e-5)


def test_infusions_and_starting_states():
    model = tq.Model.from_text(INFUSED)
    sim = model.simulator(outputs=["conc"], **TIGHT)
    scs = [
        tq.Scenario(
            regimen=tq.infusion("centr", rate=50, start=0, duration=2), times=(1, 2, 5, 10)
        ),
        tq.Scenario(regimen=tq.infusion("centr", rate=20, start=1, duration=6), times=(1, 3, 8)),
        tq.Scenario(initial={"centr": 40.0}, regimen=tq.Regimen(), times=(0.5, 4, 12)),
    ]
    st = sim.function(scs, wrt=["CL", "V"], batch="stacked")
    vm = sim.function(scs, wrt=["CL", "V"], batch="vmap")
    X = jnp.asarray(st.x0)
    np.testing.assert_allclose(st(X)["conc"][st.mask], vm(X)["conc"][vm.mask], rtol=1e-8)
    J = [jax.jacfwd(lambda X_, f=f: jnp.where(f.mask, f(X_)["conc"], 0.0))(X) for f in (st, vm)]
    np.testing.assert_allclose(J[0], J[1], rtol=1e-6, atol=1e-12)


def test_second_derivatives_forward_over_forward(m):
    fs = m.simulator(outputs=["conc"], **TIGHT).function(scenarios(), wrt=["ka", "CL"],
                                                          batch="stacked")  # fmt: skip
    X = jnp.asarray(fs.x0)
    H = jax.hessian(lambda X_: jnp.sum(jnp.where(fs.mask, fs(X_)["conc"], 0.0)))(X)
    for i, (doses, times) in enumerate(zip(DOSES, TIMES, strict=True)):
        Hc = jax.hessian(lambda x, d=doses, t=times: jnp.sum(closed(x, d, t)))(X[i])
        np.testing.assert_allclose(H[i, :, i, :], Hc, rtol=1e-5, atol=1e-10)


def test_vmap_by_default_stacked_when_asked_and_why_not(m):
    """``batch="vmap"`` by default (stacking is slower where the scenarios differ much: the
    shared steps are those of the least easy at every moment); ``"stacked"`` when asked, and
    refused with the reason where it does not apply."""
    sim = m.simulator(outputs=["conc"])
    assert sim.function(scenarios(), wrt=["ka"]).batch == "vmap"
    assert sim.function(scenarios(), wrt=["ka"], batch="stacked").batch == "stacked"
    with pytest.raises(ValueError, match="reverse derivatives"):
        sim.function(scenarios(), wrt=["ka"], derivatives="reverse", batch="stacked")
    with pytest.raises(ValueError, match="batch= is 'vmap'"):
        sim.function(scenarios(), wrt=["ka"], batch="auto")
    with pytest.raises(ValueError, match="for a list of scenarios"):
        sim.function(scenarios()[0], wrt=["ka"], batch="vmap")
    with pytest.raises(ValueError, match="one scenario"):
        sim.function(scenarios()[:1], wrt=["ka"], batch="stacked")
    # doses at many different times: the shared run would restart at each
    times = [[0.37 * i + 0.11 * j for j in range(3)] for i in range(12)]
    many = [tq.Scenario(regimen=tq.bolus("depot", 10, times=t), times=(30,)) for t in times]
    with pytest.raises(ValueError, match="different times"):
        sim.function(many, wrt=["ka"], batch="stacked")


def test_events_stay_one_by_one():
    ext = tq.Model.from_text(
        """model ext
param r=1, k=2
d_t(B) = r*B - k*B
B(0) = 10
extinction B below 1
observable y = B
"""
    )
    sim = ext.simulator(outputs=["y"])
    scs = [tq.Scenario(params={"k": k}, times=(1, 2, 4)) for k in (1.5, 2.0)]
    assert sim.function(scs, wrt=["r"]).batch == "vmap"
    with pytest.raises(ValueError, match="events"):
        sim.function(scs, wrt=["r"], batch="stacked")


def test_a_failing_run_fails_the_batch_and_status_says_which(m):
    sim = m.simulator(outputs=["conc"])
    fs = sim.function(scenarios(), wrt=["dose 1 time"], batch="stacked")
    X = jnp.asarray(fs.x0).at[1, 0].set(-5.0)  # scenario 2's dose before the start
    assert np.all(np.isnan(fs(X)["conc"]))
    st = fs.status(X)
    assert st[0] == "ok" and st[2] == "ok" and "before the start" in st[1]


def test_new_times_of_the_same_shape_compile_nothing(m):
    """The layout of the output times is an input of the compiled function, not a constant."""
    sim = m.simulator(outputs=["conc"])
    fs = sim.function(scenarios(), wrt=["ka", "CL"], batch="stacked")
    fs(fs.x0)
    cm = m.compile()
    before = {k for k in cm._lazy if isinstance(k, tuple) and k[0] == "jax_stacked_function"}
    later = [s.replace(times=tuple(t + 0.5 for t in s.times)) for s in scenarios()]
    gs = sim.function(later, wrt=["ka", "CL"], batch="stacked")
    gs(gs.x0)
    after = {k for k in cm._lazy if isinstance(k, tuple) and k[0] == "jax_stacked_function"}
    assert after == before
