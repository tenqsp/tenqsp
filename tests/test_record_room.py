# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``sim.function(..., record_room=)``: the record of reverse derivatives holds this many times
the steps of the run at ``f.x0`` (default 2; design decision 2026-11-12). Oracle: an
oscillator, u = cos(w t), whose solver steps grow with w, so a run far from x0 overflows a
small record and fits a larger one, with the derivative of the closed form."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tenqsp as tq
from tenqsp.gradients import RECORD_ROOM, record_slots

OSC = tq.Model.from_text("""
param w = 1
compartment u, v
d_t(u) = v
d_t(v) = -w^2*u
u(0) = 1
""")
T_END = 20.0
SC = tq.Scenario(times=[T_END])
TOL = {"rtol": 1e-8, "atol": 1e-10}


def reverse(record_room=None):
    sim = OSC.simulator(outputs=["u"], **TOL)
    return sim.function(SC, wrt=["w"], derivatives="reverse", record_room=record_room)


def test_the_record_holds_twice_the_steps_of_the_run_at_x0_by_default():
    f = reverse()
    assert f.record_room == RECORD_ROOM == 2
    need = f._need
    assert record_slots(2 * need) <= f.capacity <= 1.5 * record_slots(2 * need)


def test_a_run_needing_more_steps_says_which_room_to_ask_for():
    f = reverse()
    far = np.array([6.0])  # about 6x the steps of the run at w = 1
    assert np.isnan(np.asarray(f(far)["u"])).all()
    msg = f.status(far)
    assert "too many steps for the record" in msg and "record_room=4" in msg
    assert f"{f.capacity:,}" in msg


def test_more_room_runs_far_from_x0_with_the_exact_derivative():
    f = reverse(record_room=12)
    far = np.array([6.0])
    assert f.status(far) == "ok"
    u = np.asarray(f(far)["u"])[0]
    du = float(jax.grad(lambda x: f(x)["u"][0])(far)[0])
    assert np.isclose(u, np.cos(6.0 * T_END), atol=1e-5)
    assert np.isclose(du, -T_END * np.sin(6.0 * T_END), rtol=1e-4, atol=1e-4)
    assert f.capacity >= 6 * f._need


@pytest.mark.parametrize("derivatives", ["forward", False, "auto"])
def test_record_room_without_a_record_raises(derivatives):
    sim = OSC.simulator(outputs=["u"])
    with pytest.raises(ValueError, match="record_room= sizes the record of reverse"):
        sim.function(SC, wrt=["w"], derivatives=derivatives, record_room=3)


@pytest.mark.parametrize("bad", [0.5, 0, float("nan"), True, "2"])
def test_record_room_is_a_number_of_at_least_1(bad):
    sim = OSC.simulator(outputs=["u"])
    with pytest.raises((TypeError, ValueError), match="record_room="):
        sim.function(SC, wrt=["w"], derivatives="reverse", record_room=bad)


def test_a_batch_takes_record_room_too():
    sim = OSC.simulator(outputs=["u"], **TOL)
    fs = sim.function([SC, SC], wrt=["w"], derivatives="reverse", record_room=12)
    X = np.array([[1.0], [6.0]])
    assert fs.status(X) == ("ok", "ok")
    small = sim.function([SC, SC], wrt=["w"], derivatives="reverse")
    assert small.status(X)[1].startswith("too many steps for the record")
