# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""V32-V35: whole-body PBPK as C[organ, sub].

Oracles: series-resistance hepatic clearance (closed form), mass balance, partition
equilibrium at steady state, the same model written by hand as 30 ODEs in amounts, and the
perfusion-limited model as the PS -> infinity limit. Values are illustrative.
"""

from __future__ import annotations

import numpy as np
import pytest
from pbpk_models import KP, ORGANS, pbpk, pbpk_by_hand, perfusion_limited, series_clearance
from scipy.integrate import solve_ivp

TIGHT = {"rtol": 1e-10, "atol": 1e-12}
DOSE = 100.0


def bolus(m):
    m.dose(m["Cven"], times=[0], amounts=[DOSE])
    return m


def test_v32_infusion_steady_state_is_rate_over_hepatic_clearance():
    R = 10.0
    sol = pbpk(R=R).solve((0, 800), times=[800], **TIGHT)
    assert sol["Cart"][-1] == pytest.approx(R / series_clearance(50, 10, 30), rel=1e-9)


def test_v33_auc_is_dose_over_clearance_and_the_books_close():
    m = bolus(pbpk())
    C = m["C"]
    body = m.integral(C, over=[m["organ"], m["sub"]], name="body")  # weighted: an amount
    m.observable("in_body", body)
    sol = m.solve((0, 600), times=[1, 24, 600], **TIGHT)
    assert sol["AUC"][-1] == pytest.approx(DOSE / series_clearance(50, 10, 30), rel=1e-7)
    assert max(sol.ledger.check().values()) < 1e-10
    cm = m.compile()
    a = cm.amounts(sol.y)
    np.testing.assert_allclose(sol["in_body"].ravel(), a["C"], rtol=1e-12)
    eliminated = -sol.ledger.moved["C"]["transfer to elimination"]
    np.testing.assert_allclose(a["C"] + a["Cart"] + a["Cven"] + eliminated, DOSE, rtol=1e-9)


def test_v34_partition_equilibrium_in_non_eliminating_organs():
    sol = pbpk(R=10.0).solve((0, 800), times=[800], **TIGHT)
    c = sol["C"][-1]  # (organ, sub)
    for i, o in enumerate(ORGANS):
        if o == "liver":
            continue
        vasc, inter, cell = c[i]
        assert inter == pytest.approx(vasc, rel=1e-8), o
        assert cell / KP[o] == pytest.approx(vasc, rel=1e-8), o
    li = ORGANS.index("liver")  # the eliminating organ: a gradient towards the cells
    assert c[li, 0] > c[li, 1] > c[li, 2] / KP["liver"]


def test_v35_tensor_form_equals_the_hand_written_odes():
    t = np.array([0.1, 1, 6, 24, 96.0])
    sol = bolus(pbpk()).solve((0, 96), times=t, **TIGHT)
    rhs, vol = pbpk_by_hand()
    y0 = np.zeros(30)
    y0[28] = DOSE  # venous pool
    ref = solve_ivp(rhs, (0, 96), y0, t_eval=t, method="Radau", rtol=1e-12, atol=1e-14).y
    conc = ref / vol[:, None]
    np.testing.assert_allclose(sol["C"].reshape(len(t), -1), conc[:27].T, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(sol["Cart"], conc[27], rtol=1e-7)
    np.testing.assert_allclose(sol["Cven"], conc[28], rtol=1e-7)
    np.testing.assert_allclose(sol["AUC"], ref[29], rtol=1e-7)


def test_v35_large_permeability_approaches_the_perfusion_limited_model():
    """The membrane resistances 1/PS vanish as PS grows: the difference from the
    perfusion-limited model falls in proportion to 1/PS (tenfold per decade)."""
    t = np.array([0.5, 2, 8, 24.0])
    y0 = np.zeros(len(ORGANS) + 2)
    y0[-1] = DOSE
    ref = solve_ivp(
        perfusion_limited(), (0, 24), y0, t_eval=t, method="Radau", rtol=1e-12, atol=1e-14
    ).y
    c_art = ref[-2] / 1.7
    errors = []
    for ps in (1e2, 1e3, 1e4):
        sol = bolus(pbpk(PS1=ps, PS2=ps)).solve((0, 24), times=t, **TIGHT)
        errors.append(np.max(np.abs(sol["Cart"] / c_art - 1)))
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all((ratios > 8) & (ratios < 13)) and errors[-1] < 2e-2, errors  # error ~ 1/PS


@pytest.mark.scipy_backend
def test_pbpk_sparse_jacobian_and_one_compile_for_new_values():
    m = pbpk()
    cm = m.compile()
    rng = np.random.default_rng(32)
    for _ in range(3):
        y = rng.uniform(0.0, 2.0, cm.n_states)
        np.testing.assert_allclose(
            cm.sparse_jacobian(0.0, y).toarray(), np.asarray(cm.jacobian(0.0, y)), atol=1e-12
        )
    for CLint in (10.0, 30.0):
        m.solve((0, 10), times=[10], params={"CLint": CLint, "V": {"liver": {"cellular": 2.0}}})
    assert cm.compilations()["solver rhs (with ledger)"] == 1
