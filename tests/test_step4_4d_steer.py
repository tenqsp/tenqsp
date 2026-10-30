# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Step 4.4d: steps chosen for the derivative too (design note docs/design/step4_4d_api.md
§6.3; design decision 2026-10-25: option B, as CVODES with errconS). Oracles: closed forms of
a response started at rest whose derivative is not at rest, the Friberg example of §6.3
against a run with small steps, and the derivative carried inside the run against the
replay's exact derivative (doses, lags, F, dose times, infusions)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq
from tenqsp.backends import jax_backend as jb
from tenqsp.backends import jax_replay
from tenqsp.backends.jax_bdf import bdf
from tenqsp.gradients import _inputs

# an indirect response at its baseline: R(0) is a number (from data), kin/kout match it
IDR = """model idr
param kin = 2, kout = 0.2
compartment R
d_t(R) = kin - kout*R
R(0) = 10
"""
T = np.array([2.0, 10, 30, 100])


def test_the_integrator_carries_the_derivative_and_is_steered_by_it():
    # y' = k (c - y) from y = c: at rest; d y / d c = 1 - exp(-k t) is not
    k = 0.3

    def run(sens):
        return jax.jit(
            lambda: bdf(
                lambda t, y: k * (1.0 - y),
                lambda t, y: jnp.array([[-k]]),
                0.0,
                100.0,
                jnp.ones(1),
                jnp.array([1.0, 100.0]),
                rtol=1e-6,
                atol=1e-6,
                sens=sens,
            )
        )()

    plain = run(None)
    steered = run((lambda t, y, s: k * (1.0 - s), jnp.zeros(1)))
    assert plain.s is None and steered.s is not None
    assert int(steered.n_steps) > 3 * int(plain.n_steps)
    assert np.isclose(float(steered.s[0]), 1 - np.exp(-k * 100), rtol=1e-5)
    # the solution itself is the same (at rest)
    assert np.allclose(steered.ys, plain.ys, rtol=1e-12)


def test_a_response_at_its_baseline():
    """R at rest; dR/dkin = (1 - e^-kout t)/kout, dR/dkout = -R0 (1 - e^-kout t)/kout."""
    m = tq.Model.from_text(IDR)
    sim = m.simulator(outputs=["R"])  # default tolerances, no max_step
    f = sim.function(tq.Scenario(times=T), wrt=["kin", "kout"])
    assert np.allclose(f(f.x0)["R"], 10.0)
    J = np.asarray(jax.jacfwd(lambda x: f(x)["R"])(f.x0))
    grow = 1 - np.exp(-0.2 * T)
    assert np.allclose(J[:, 0], grow / 0.2, rtol=1e-4)
    assert np.allclose(J[:, 1], -10 * grow / 0.2, rtol=1e-4)
    # unsteered, the run takes steps for the solution only, and the derivative is off
    cm = m.compile()
    rec = jax_replay.record(cm, (0, 100), times=T)
    _, G = jax.jvp(
        lambda th: jax_replay.replay(cm, rec, th),
        (jnp.asarray(rec.run.th),),
        (jnp.asarray(np.eye(rec.run.th.size)[0]),),
    )
    assert not np.allclose(np.asarray(G)[:, 0], grow / 0.2, rtol=1e-3)


@pytest.mark.slow  # 10 s: in the slow tier (fast tier < 2 min)
def test_the_friberg_example_of_the_design_note():
    sys.path.insert(0, str(Path(__file__).parent))
    from test_gallery_phase2 import friberg

    m = friberg()
    times = tq.Scenario(times=[100.0, 500.0])
    f = m.simulator(outputs=["Circ"]).function(times, wrt=["circ0"])
    fine = m.simulator(outputs=["Circ"], max_step=1.0, rtol=1e-10, atol=1e-12)
    g = fine.function(times, wrt=["circ0"])
    J = np.asarray(jax.jacfwd(lambda x: f(x)["Circ"])(f.x0))[:, 0]
    ref = np.asarray(jax.jacfwd(lambda x: g(x)["Circ"])(g.x0))[:, 0]
    assert np.allclose(J, ref, rtol=1e-4)  # was 1.021 against 1.092 at t = 500


ORAL = """model pk
param ka = 1.2, CL = 4, V = 30, tlag = 0.5, F = 0.8
compartment depot, centr
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
lag depot = tlag
bioavailability depot = F
"""


def _carried_and_exact(sim, sc, wrt):
    """The derivative the steered run carries to its end, and the replay's, along the
    steering direction."""
    f = sim.function(sc, wrt=wrt)
    cm = f._cm
    program = jb._program(cm, ledger=False, model_doses=False, max_steps=100_000, steer=True)
    x0 = jnp.asarray(f.x0)
    args, dot = _inputs(x0, f._data, False)
    out = program(**args, dot=dot)
    v = f._data["steer"] * jnp.where(x0 != 0, jnp.abs(x0), 1.0)
    _, exact = jax.jvp(lambda x: f(x)["centr"][-1], (x0,), (v,))
    return float(out["zs"][cm.layout.names.index("centr")]), float(exact)


def test_the_carried_derivative_across_doses_lags_and_dose_times():
    m = tq.Model.from_text(ORAL)
    sim = m.simulator(outputs=["centr"], rtol=1e-9, atol=1e-12)
    sc = tq.Scenario(regimen=tq.bolus("depot", 100, times=[0, 5]), times=[1, 3, 12.0])
    wrt = ["ka", "CL", "tlag", "F", "dose 1 amount", "dose 2 time"]
    carried, exact = _carried_and_exact(sim, sc, wrt)
    assert np.isclose(carried, exact, rtol=1e-5)


def test_the_carried_derivative_across_infusion_starts_and_stops():
    m = tq.Model.from_text("param CL = 4, V = 30\ncompartment centr\nd_t(centr) = -CL/V*centr\n")
    sim = m.simulator(outputs=["centr"], rtol=1e-9, atol=1e-12)
    reg = tq.infusion("centr", rate=50, duration=2, start=0.5)
    sc = tq.Scenario(regimen=reg, times=[1, 3, 12.0])
    for wrt in (["CL", "infusion 1 start"], ["infusion 1 duration"], ["infusion 1 amount"]):
        carried, exact = _carried_and_exact(sim, sc, wrt)
        assert np.isclose(carried, exact, rtol=1e-5), wrt


def test_steering_adds_steps_only_where_the_derivative_needs_them():
    # a one-compartment decline: the solution and its derivatives move together
    m = tq.Model.from_text(
        "param CL = 4, V = 30\ncompartment centr\nd_t(centr) = -CL/V*centr\ncentr(0) = 100\n"
    )
    cm = m.compile()
    plain = jax_replay.record(cm, (0, 24), times=[24.0])
    th = plain.run.th
    steered = jax_replay.record(cm, (0, 24), times=[24.0], direction=th)
    assert steered.n_steps <= 2 * plain.n_steps
