# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``NegativeValuesWarning``: a run whose state goes below -10 x atol while its values are
otherwise far larger warns once, names the state, and says whether the negative part grew; a
failure after such a dip says so in its message. A state negative by design (it starts below
zero, or its dip is a sizeable share of its values) is left alone. Oracle: compartments whose
rates are set piecewise in time, so where and how far they go below zero is known exactly."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import tenqsp as tq
from tenqsp.backends.scipy_backend import TAIL_FRACTION

TOL = {"rtol": 1e-8, "atol": 1e-9}
BACKENDS = ["scipy", "jax"]


def model(rate: str, start: float = 1.0) -> tq.Model:
    return tq.Model.from_text(f"compartment A\nd_t(A) = {rate}\nA(0) = {start}\n")


def run(m, backend, t_end=3.0, **options):
    """The solution and the NegativeValuesWarning messages of one run."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sol = m.solve((0, t_end), backend=backend, **{**TOL, **options})
    return sol, [str(w.message) for w in caught if w.category is tq.NegativeValuesWarning]


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_dip_below_ten_atol_warns_once(backend):
    # A falls from 1 at rate 1 + 1e-5 for one time unit: it ends at -1e-5, far beyond -1e-8
    sol, msgs = run(model("-(1 + 1e-5) if t < 1 else 0"), backend)
    assert np.isclose(np.min(sol["A"]), -1e-5, rtol=1e-6)
    assert len(msgs) == 1
    assert msgs[0].startswith("'A' went negative (down to -1e-05 at t = 1.005;")
    assert "grew" not in msgs[0] and "smaller atol for 'A'" in msgs[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_dip_that_grows_says_so(backend):
    # -1e-6 at t = 1, then A' = 2A: -1e-6 x e^4 at t = 3 (atol well below 1e-6 to pin it)
    sol, msgs = run(model("-(1 + 1e-6) if t < 1 else 2*A"), backend, atol=1e-14)
    assert np.isclose(np.min(sol["A"]), -1e-6 * np.exp(4), rtol=1e-4)
    assert len(msgs) == 1 and "at t = 3, and the negative part grew" in msgs[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_states_negative_by_design_do_not_warn(backend):
    _, msgs = run(model("-2"), backend)  # down to -5: a sizeable share of its values
    assert msgs == []
    _, msgs = run(model("-1e-3", start=-1.0), backend)  # starts below zero
    assert msgs == []
    _, msgs = run(model("-1 if t < 1 else 0"), backend)  # reaches 0 to rounding
    assert msgs == []


def test_the_tail_fraction_separates_tails_from_values():
    # A reaches 0 at t = 1, then the equations drive it down at rate x: -2x at t = 3
    for x, warns in ((0.25 * TAIL_FRACTION, True), (TAIL_FRACTION, False)):
        _, msgs = run(model(f"-1 if t < 1 else -{x}"), "scipy")
        assert len(msgs) == warns, x


@pytest.mark.parametrize("backend", BACKENDS)
def test_any_dip_counts_where_the_equations_keep_the_state_nonnegative(backend):
    # a population killed below atol undershoots, then grows away from 0 once the killing
    # stops: far larger than TAIL_FRACTION of its values, yet B' = (g - k)B cannot cross 0
    m = tq.Model.from_text("""
param g=1, kmax=3.9
k_kill = kmax if t < 10 else 0
compartment B
d_t(B) = g*B - k_kill*B
B(0) = 1e6
""")
    sol, msgs = run(m, backend, t_end=40.0, rtol=1e-6, atol=1e-6)
    assert np.min(sol["B"]) < -1e5  # the troubleshooting guide's example: -2.2e6 at t = 40
    assert len(msgs) == 1 and "'B' went negative" in msgs[0] and "grew" in msgs[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_tolerances_per_state_are_used(backend):
    # one state: a per-state atol of 1e-4 puts -1e-5 within -10 x atol, so no warning
    _, msgs = run(model("-(1 + 1e-5) if t < 1 else 0"), backend, atol=[1e-4])
    assert msgs == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_failure_after_a_dip_says_so(backend):
    # below zero from t = 1; at t = 2 the rate takes sqrt(A) of a negative A and the run fails
    m = model("-(1 + 1e-5) if t < 1 else (0 if t < 2 else -sqrt(A))")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # numpy's warnings on sqrt of a negative number
        with pytest.raises(tq.SolverError, match="Before the failure, 'A' went negative"):
            m.solve((0, 3), backend=backend, **TOL)


@pytest.mark.scipy_backend  # under --backend=jax the caller is the suite's own wrapper
def test_it_is_a_user_warning_that_points_at_the_caller():
    assert issubclass(tq.NegativeValuesWarning, UserWarning)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model("-(1 + 1e-5) if t < 1 else 0").solve((0, 3), **TOL)
    (w,) = [w for w in caught if w.category is tq.NegativeValuesWarning]
    assert w.filename == __file__


def test_it_can_be_silenced():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=tq.NegativeValuesWarning)
        model("-(1 + 1e-5) if t < 1 else 0").solve((0, 3), **TOL)


def test_a_simulator_run_warns_too():
    sim = model("-(1 + 1e-5) if t < 1 else 0").simulator(outputs=["A"], **TOL)
    with pytest.warns(tq.NegativeValuesWarning, match="'A' went negative"):
        sim(tq.Scenario(times=[0.5, 2.0]))
