# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Step 4.4a: a JAX-backend run recorded and replayed (design note
docs/design/step4_4_gradients.md §3). The replay takes each step's value from the record and
recomputes the rest (segment starts, doses, extinctions, outputs), so it must reproduce the
run to round-off on every feature the backend has. ``pytest --backend=jax --replay`` checks
the same on every JAX solve of the test suite."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tenqsp as tq
from tenqsp.backends import jax_replay
from tenqsp.examples import toy_phage_model, toy_three_axis_model

PK = """param ka, CL
param V = 30
compartment depot, centr
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
dose depot at t=0: 100
"""
P = {"ka": 1.2, "CL": 4}
TIGHT = {"rtol": 1e-8, "atol": 1e-10}


def replayed(m, t_span, **kw):
    cm = m.compile()
    rec = jax_replay.record(cm, t_span, **kw)
    return rec, np.asarray(jax_replay.replay(cm, rec))


def assert_same(rec, ys, rel=1e-13):
    scale = np.max(np.abs(rec.ys), axis=0) + 1e-300
    assert np.max(np.abs(ys - rec.ys) / scale) <= rel


@pytest.fixture(scope="module")
def pk():
    return tq.Model.from_text(PK, name="pk")


def test_the_models_own_dose_and_a_regimen(pk):
    t = np.linspace(0, 96, 97)
    rec, ys = replayed(pk, (0, 96), times=t, params=P, **TIGHT)
    assert rec.n_segments == 1
    assert_same(rec, ys)
    reg = tq.bolus("depot", 100, start=0, every=12, n=7)
    rec, ys = replayed(pk, (0, 96), times=t, params=P, regimen=reg, **TIGHT)
    assert rec.n_segments == 7 and rec.n_steps > 1024  # also: recorded again, at 2048
    assert_same(rec, ys)
    sol = pk.solve((0, 96), times=t, params=P, regimen=reg, backend="jax", ledger=False, **TIGHT)
    np.testing.assert_array_equal(rec.ys, sol.y)  # the recorded run is the ordinary run


def test_an_infusion_and_a_dose_into_a_weighted_field_element():
    m = tq.Model.from_text(
        "axis s categorical [S, R]\nparam w[s] = {S: 2, R: 4}\nfield C(s) weights=w[s]\n"
        "compartment A\nd_t(C) = -0.1*C\nd_t(A) = -A\n",
        name="w",
    )
    reg = tq.bolus("C[s=R]", 8, times=[1]) + tq.infusion("A", rate=5, duration=2, start=0.5)
    rec, ys = replayed(m, (0, 6), times=np.linspace(0, 6, 13), regimen=reg, **TIGHT)
    assert rec.n_segments == 4  # boundaries at 0.5, 1 and 2.5
    assert_same(rec, ys)


def test_switches_on_time_and_on_a_state():
    on = tq.Model("on")
    A = on.compartment("A", init=0.0)
    on.params(t_on=5.0, kin=1.0)
    on.ode(A, "(kin if t > t_on else 0) - 0.1*A")
    rec, ys = replayed(on, (0, 20), times=[0, 4, 5, 6, 20.0], params={"t_on": 7.5}, **TIGHT)
    assert rec.n_segments == 2
    assert_same(rec, ys)
    sat = tq.Model("sat")
    C = sat.compartment("C", init=10.0)
    sat.ode(C, -tq.cases((C > 2, 0.5), otherwise=0.1) * C)
    rec, ys = replayed(sat, (0, 20), times=np.linspace(0, 20, 21), **TIGHT)
    assert rec.n_segments == 2 and bool(rec.segs["event"][0])  # ended at the switch
    assert_same(rec, ys)


def test_an_extinction_and_a_noise_floor():
    m = tq.Model("decay")
    B, P_ = m.compartment("B", init=100.0), m.compartment("P", init=1e-12)
    m.ode(B, -0.5 * B)
    m.ode(P_, 5 * P_)
    m.extinction(B, threshold=1.0)
    m.extinction(P_, threshold=1.0, noise_floor=True)
    rec, ys = replayed(m, (0, 20), times=np.linspace(0, 20, 21), **TIGHT)
    assert np.asarray(rec.segs["zero"])[0].tolist() == [True, False]  # B zeroed at the event
    assert ys[-1, 0] == 0.0 and ys[-1, 1] == 1e-12  # B extinct, the ghost floored
    assert_same(rec, ys)


@pytest.mark.parametrize(
    "build", [lambda: toy_phage_model(8, 6), lambda: toy_three_axis_model(4, 6)]
)
def test_field_models(build):
    m = build()
    rec, ys = replayed(m, (0, 6), times=np.linspace(0, 6, 13), rtol=1e-8, atol=1e-6)
    assert_same(rec, ys)


def test_the_replay_is_a_traceable_function_of_theta(pk):
    """No derivatives yet (4.4b); but it runs under jit with theta traced, and the record
    is reused: one build for new values."""
    cm = pk.compile()
    rec = jax_replay.record(cm, (0, 24), times=[1, 24], params=P, **TIGHT)
    f = jax.jit(lambda th: jax_replay.replay(cm, rec, th))
    np.testing.assert_allclose(np.asarray(f(rec.run.th)), rec.ys, rtol=1e-14)
    jax_replay.replay(cm, rec)
    program = jax_replay._replay_program(cm, True)
    before = program._cache_size()
    jax_replay.replay(cm, rec, rec.run.th * 1.0)
    assert program._cache_size() == before
