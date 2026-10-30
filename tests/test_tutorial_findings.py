# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Details found while writing the tutorials (2026-10-27): explain() wording, validation of
velocities through named formulas, per-category values in a flux between slices."""

from __future__ import annotations

import numpy as np
import pytest

import tenqsp as tq

TRANSIT = """
axis z in [0, 1] n=3 kind=one_way
param MTT=100
ktr = 4/MTT
T_mat = 3/ktr
field Transit(z)
d_t(Transit) = -d_z(Transit/T_mat)
    at z=1: outflow
"""


def test_a_velocity_through_named_formulas_of_parameters_is_not_called_state_dependent():
    text = tq.Model.from_text(TRANSIT, name="t").explain()
    assert "depends on the state or time" not in text


def test_a_negative_velocity_through_a_named_formula_is_still_refused():
    """...and now when the model is read, not only while solving."""
    with pytest.raises(tq.notation.NotationError, match="one_way .* negative somewhere"):
        tq.Model.from_text(TRANSIT.replace("T_mat = 3/ktr", "T_mat = -3/ktr"), name="t")


def test_a_compartment_changed_only_by_fluxes_is_not_said_to_stay_put():
    m = tq.Model.from_text(
        "param k=1\nflux J: A -> B = k*A\nd_t(A) = -J\nd_t(B) = J\nA(0) = 1\n", name="f"
    )
    text = m.explain()
    assert "stays as it starts" not in text


ORGANS = """
axis organ categorical [gut, liver]
param Kp[organ] = {gut: 2, liver: 4}, Q[organ] = {gut: 1, liver: 3}, CL=8, Kp_liver=4, Q_liver=3
field C(organ)
flux J_met: C[organ=liver] -> Elim = CL/Kp[organ]*C
flux J_in: Pool -> C[organ=liver] = Q[organ]*Pool
d_t(C[organ=liver]) = J_in - J_met
d_t(Elim) = J_met
d_t(Pool) = -J_in
Pool(0) = 10
C(0) = 1
"""


def test_a_per_category_value_in_a_flux_is_taken_at_the_slices_category():
    """Kp[organ] on C[organ=liver] is Kp[liver]; Q[organ] into C[organ=liver] is Q[liver]:
    the same run as with the liver's values written as plain parameters."""
    named = ORGANS.replace("CL/Kp[organ]*C", "CL/Kp_liver*C").replace(
        "Q[organ]*Pool", "Q_liver*Pool"
    )
    t = [0.0, 0.5, 1.0, 2.0]
    a = tq.Model.from_text(ORGANS, name="o").solve((0, 2), times=t, rtol=1e-10, atol=1e-12)
    b = tq.Model.from_text(named, name="o").solve((0, 2), times=t, rtol=1e-10, atol=1e-12)
    for k in ("C", "Elim", "Pool"):
        assert np.allclose(a[k], b[k], rtol=1e-10, atol=1e-12), k
    assert max(a.ledger.check().values()) < 1e-10


def test_blas_runs_one_thread_while_tenqsp_factors_and_solves():
    """SuperLU calls BLAS; idle OpenBLAS threads spin (a 9,600-state solve used 3,400 CPU
    seconds in 54 s): TenQSP limits BLAS to one thread inside, and restores it after."""
    threadpoolctl = pytest.importorskip("threadpoolctl")
    from tenqsp import _threads

    before = [i["num_threads"] for i in threadpoolctl.threadpool_info() if i["user_api"] == "blas"]
    with _threads.blas_threads():
        inside = [
            i["num_threads"] for i in threadpoolctl.threadpool_info() if i["user_api"] == "blas"
        ]
    after = [i["num_threads"] for i in threadpoolctl.threadpool_info() if i["user_api"] == "blas"]
    assert inside and set(inside) == {1}
    assert after == before


def test_a_crawling_solve_stops_with_an_explanation():
    """A solve that needs more than max_steps steps in one segment stops with a message
    (it used to crawl on silently: a PBPK model at PS = 1e4 and rtol = 1e-10 took tiny
    first-order steps near steady state for minutes). Also with Radau, not retried."""
    from tenqsp.backends.scipy_backend import SolverError, solve

    cm = tq.Model.from_text("param k=1\nd_t(A) = -k*A\nA(0) = 1\n", name="d").compile()
    for method in ("BDF", "Radau"):  # the scipy backend itself (not routed by --backend)
        with pytest.raises(SolverError, match="more than 100 steps .* crawling"):
            solve(cm, (0, 1), times=[1], max_step=1e-3, max_steps=100, method=method)
    assert solve(cm, (0, 1), times=[1], max_step=1e-3, max_steps=2000).t[-1] == 1
