# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Regressions for the Phase 4 gate review (independent reviewers, 2026-10-27): the JAX
solver's segments and limits, sim.function at a new x, scenarios, regimens and diagnostics.
Each test is one finding, checked against a closed form or the scipy backend."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq

DECAY = "param k = 0.2\ncompartment A\nd_t(A) = -k*A\n"


# ---------------------------------------------------------------- the JAX solver


def test_switch_times_beyond_the_room_sized_at_x0_are_a_failure_not_a_cut():
    m = tq.Model.from_text(
        "param w = 1\ncompartment A\nd_t(A) = 1 if sin(w*t) > 0 else 0\n", name="sw_room"
    )
    f = m.simulator(outputs=["A"]).function(
        tq.Scenario(times=[0.0, 100.0]), wrt=["w"], derivatives=False
    )
    assert float(f(jnp.array([1.0]))["A"][-1]) == pytest.approx(50.2655, rel=1e-4)
    out = f(jnp.array([3.0]))["A"]  # three times the switches: the room is too small
    assert np.isnan(out).all()
    assert "more switch times" in f.status(jnp.array([3.0]))


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_a_dose_is_given_when_an_event_falls_on_its_time(backend):
    """A rises at rate 1 to its threshold 5 at t = 5 (arming its group), where 100 is dosed."""
    m = tq.Model.from_text(
        "compartment A\nd_t(A) = 1\nA(0) = 0\nextinction A below 5\ndose A at t=5: 100\n",
        name="ev_dose",
    )
    sol = m.solve((0, 6), times=[5.5, 6], backend=backend)
    assert np.allclose(np.asarray(sol["A"]).ravel(), [105.5, 106.0])
    assert max(sol.ledger.check().values()) < 1e-9


def test_a_dose_at_an_event_time_through_the_replay():
    m = tq.Model.from_text(
        "param r = 1\ncompartment A\nd_t(A) = r\nA(0) = 0\nextinction A below 5\n"
        "dose A at t=5: 100\n",
        name="ev_dose_r",
    )
    f = m.simulator(outputs=["A"]).function(tq.Scenario(times=[5.5, 6.0]), wrt=["r"])
    assert np.allclose(f(jnp.array([1.0]))["A"], [105.5, 106.0])


@pytest.mark.parametrize("derivatives", [True, False])
def test_a_regimen_moved_out_of_range_is_a_failure(derivatives):
    m = tq.Model.from_text(DECAY, name="moved")
    sc = tq.Scenario(
        regimen=tq.bolus("A", 100, times=[1]) + tq.infusion("A", rate=10, duration=2, start=3),
        times=[2.0, 6.0],
    )
    f = m.simulator(outputs=["A"]).function(
        sc, wrt=["dose 1 time", "infusion 1 duration"], derivatives=derivatives
    )
    assert np.isfinite(f(jnp.array([1.0, 2.0]))["A"]).all()
    for x in ([-0.5, 2.0], [1.0, -2.0]):  # a dose before t = 0; a negative duration
        assert np.isnan(f(jnp.array(x))["A"]).all()
        assert "moved before the start" in f.status(jnp.array(x))


def test_a_run_needing_exactly_max_steps_steps_succeeds():
    m = tq.Model.from_text(DECAY + "A(0) = 1\n", name="steps")
    n = m.solve((0, 10), backend="jax").stats["n_steps"]
    m.solve((0, 10), backend="jax", max_steps=n)  # no error
    with pytest.raises(tq.SolverError, match="too many steps"):
        m.solve((0, 10), backend="jax", max_steps=n - 1)


# ---------------------------------------------------------------- scenarios and regimens


def test_per_category_values_merge_label_by_label():
    sc = tq.Scenario(params={"k": {"S": 2}, "V": 1}).replace(params={"k": {"R": 3}})
    assert sc.params == {"k": {"S": 2, "R": 3}, "V": 1}
    m = tq.Model.from_text(
        "axis s categorical [S, R]\nparam k[s] = {S: 1, R: 1}\nfield B(s)\nB(0) = 1\n"
        "d_t(B) = -k[s]*B\n",
        name="percat",
    )
    out = m.simulator(outputs=["B"])(
        tq.Scenario(params={"k": {"S": 2}}), params={"k": {"R": 3}}, times=[1.0]
    )
    assert np.allclose(np.asarray(out["B"]).ravel(), [math.exp(-2), math.exp(-3)], rtol=1e-4)


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"EVID": [1, 4]}, "EVID 4"),
        ({"EVID": [1, 3]}, "EVID 3"),
        ({"SS": [1, 0]}, "steady state"),
        ({"ID": [1, 2]}, "several subjects"),
        ({"ADDL": [1.5, 0], "II": [12, 0]}, "whole number"),
        ({"RATE": [50, 0], "DUR": [2, 0]}, "not both"),
    ],
)
def test_a_dose_table_refuses_what_it_cannot_read(extra, match):
    table = {"TIME": [0, 24], "AMT": [100, 100], "CMT": ["depot", "depot"], **extra}
    with pytest.raises(ValueError, match=match):
        tq.regimen_from_table(table)


def test_a_dose_table_of_one_subject_reads():
    table = {"ID": [7, 7], "TIME": [0, 24], "AMT": [100, 0], "CMT": ["depot"] * 2, "EVID": [1, 0]}
    assert len(tq.regimen_from_table(table).boluses) == 1


@pytest.mark.parametrize(
    "entry, error",
    [
        ({"kind": "bolus", "target": "A", "time": 0, "amount": -100}, ValueError),
        ({"kind": "bolus", "target": "A", "time": "soon", "amount": 1}, TypeError),
        ({"kind": "bolus", "target": "A", "time": math.inf, "amount": 1}, ValueError),
        ({"kind": "infusion", "target": "A", "start": 0, "duration": -1, "rate": 1}, ValueError),
        ({"kind": "infusion", "target": "A", "start": 0, "duration": 1, "rate": -1}, ValueError),
    ],
)
def test_a_regimen_read_from_json_is_checked(entry, error):
    with pytest.raises(error):
        tq.Regimen.from_list([entry])


@pytest.mark.parametrize("times", [[0, 12, 24, 24], [0, 12, 12, 24]])
def test_a_time_given_twice_gets_a_row_each_on_both_backends(times):
    m = tq.Model.from_text(DECAY + "A(0) = 1\ndose A at t=12: 1\n", name="twice")
    rows = [
        np.asarray(m.solve((0, 24), times=times, backend=b)["A"]).ravel() for b in ("scipy", "jax")
    ]
    assert rows[0].shape == (4,) and np.allclose(rows[0], rows[1], rtol=1e-6)


# ---------------------------------------------------------------- diagnostics


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_an_infusion_refills_without_an_artefact_warning(backend):
    m = tq.Model.from_text(
        "param CL = 300, V = 30\ncompartment A volume=V\nd_t(A) = -CL/V*A\n", name="refill"
    )
    regimen = tq.bolus("A", 1, times=[0]) + tq.infusion("A", rate=10, duration=5, start=10)
    sol = m.solve((0, 20), regimen=regimen, times=np.linspace(0, 20, 81), backend=backend)
    assert sol.infusions == [(10.0, 15.0, "A", 10.0)]
    assert "artefact" not in str(m.diagnostics(sol))
