# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The carried derivative's error scale (``jax_bdf._sens_scale``). A derivative that starts
from 0 after a dose while the state does not (a field the dose reaches through its
neighbours) has, in its first steps, an error estimate as large as itself. With
``atol + rtol |s|`` alone the step could only pass if ``atol`` were large; with a small
``atol`` every step was rejected down to round-off ("required step size is less than
spacing between numbers"). Oracle: central differences of the scipy backend at tight
tolerances."""

from __future__ import annotations

import numpy as np
import pytest

import tenqsp as tq

SURFACE = """model surface_dose
axis r in [0, 1] n=6
param k=0.7, D=0.2
field c(r)
d_t(c) = d_r(D*d_r(c)) - c
    at r=1: c = P
d_t(P) = -k*P
observable o = int_r(c)
"""
TIMES = [0.0, 5.0]


def _o_end(m, times, amounts=(10.0, 10.0)):
    reg = tq.Regimen()
    for t, a in zip(times, amounts, strict=True):
        reg = reg + tq.bolus("P", a, times=[t])
    s = m.solve((0, 10), times=[0.0, 10.0], regimen=reg, rtol=1e-11, atol=1e-14)
    return float(np.asarray(s["o"])[-1])


@pytest.mark.parametrize("name", ["dose 2 time", "dose 2 amount"])
def test_a_derivative_that_starts_after_a_dose_with_a_small_atol(name):
    m = tq.Model.from_text(SURFACE)
    sim = m.simulator(outputs=["o"], rtol=1e-6, atol=1e-20)
    sc = tq.Scenario(regimen=tq.bolus("P", 10, times=TIMES), times=[0.0, 10.0])
    f = sim.function(sc, wrt=[name], derivatives="forward")
    assert f.status(f.x0) == "ok"
    s = sim.sensitivity(sc, wrt=[name])["o"][name][-1]
    h = 1e-3
    if name == "dose 2 time":
        fd = (_o_end(m, [0.0, 5.0 + h]) - _o_end(m, [0.0, 5.0 - h])) / (2 * h)
    else:
        fd = (_o_end(m, TIMES, (10.0, 10.0 + h)) - _o_end(m, TIMES, (10.0, 10.0 - h))) / (2 * h)
    assert np.isclose(s, fd, rtol=1e-4)


LATE = SURFACE.replace("n=6", "n=20 geometry=spherical").replace("k=0.7", "k=0.02")


def test_a_large_dose_moved_late_in_the_run_with_a_small_atol():
    # dose 2 at t = 900: the direction moves it in proportion to its value, so its derivative
    # is huge next to the surface and tiny a few cells in, below the rounding of the solve
    m = tq.Model.from_text(LATE)
    sim = m.simulator(outputs=["o"], rtol=1e-6, atol=1e-20)
    sc = tq.Scenario(regimen=tq.bolus("P", 29.4, times=[0.0, 900.0]), times=[0.0, 905.0])
    f = sim.function(sc, wrt=["dose 1 time", "dose 2 time"], derivatives="forward")
    assert f.status(f.x0) == "ok"
    s = sim.sensitivity(sc, wrt=["dose 2 time"])["o"]["dose 2 time"][-1]

    def o_end(t2):
        reg = tq.bolus("P", 29.4, times=[0.0]) + tq.bolus("P", 29.4, times=[t2])
        sol = m.solve((0, 905), times=[0.0, 905.0], regimen=reg, rtol=1e-11, atol=1e-14)
        return float(np.asarray(sol["o"])[-1])

    h = 1e-3
    assert np.isclose(s, (o_end(900 + h) - o_end(900 - h)) / (2 * h), rtol=1e-4)
