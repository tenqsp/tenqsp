# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The whole-body PBPK model (notation_models/pbpk.tq) with a second drug: blood pools per
drug, each feeding every organ by a flux, and per-organ slices read in formulas.

Oracle: the drugs do not interact, so each drug's run equals the one-drug model run with that
drug's values. Also: the sparse Jacobian equals the dense one, and the ledger closes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_sparse_jacobian import _random_point

import tenqsp as tq

ONE = (Path(__file__).parent / "notation_models" / "pbpk.tq").read_text()
TIGHT = {"rtol": 1e-10, "atol": 1e-12}
TIMES = [0.5, 4.0, 24.0]
DRUGS = {"d1": {"R": 10.0, "CLint": 30.0}, "d2": {"R": 3.0, "CLint": 5.0}}


def two_drugs() -> str:
    t = ONE.replace("model pbpk", "model pbpk2\naxis drug categorical [d1, d2]")
    t = t.replace("CLint=30, ", "").replace("R=0, ", "")
    t = t.replace(
        "param Q_gut=65",
        "param CLint[drug] = {d1: 30, d2: 5}\nparam R[drug] = {d1: 10, d2: 3}\nparam Q_gut=65",
    )
    t = t.replace("field C(organ, sub)", "field C(drug, organ, sub)")
    t = t.replace(
        "compartment Cart volume=V_art\ncompartment Cven volume=V_ven",
        "Va(drug) = V_art\nVv(drug) = V_ven\n"
        "field Cart(drug) weights=Va\nfield Cven(drug) weights=Vv",
    )
    t = t.replace("/V_art", "/Va").replace("/V_ven", "/Vv")
    t = t.replace("-CLint/", "-CLint[drug]/").replace("+ R/Vv", "+ R[drug]/Vv")
    t = t.replace("d_t(AUC)  = Cart\n", "")
    assert "AUC" not in t and "compartment" not in t
    return t + "observable cell = C[sub=cellular]\nobservable cell_one = C[drug=d2, sub=cellular]\n"


def one_drug(R: float, CLint: float) -> tq.Model:
    t = ONE.replace("CLint=30", f"CLint={CLint}").replace("R=0", f"R={R}")
    return tq.Model.from_text(t + "observable cell = C[sub=cellular]\n")


@pytest.fixture(scope="module")
def model():
    return tq.Model.from_text(two_drugs())


def test_it_reads_and_says_the_pools_take_the_sum(model):
    text = " ".join(model.explain().split())  # one line: explain wraps its text
    assert "Cart -> C[sub=vascular]" in text
    assert "Cart gives or takes the sum over organ" in text
    assert "Cven gives or takes the sum over organ" in text


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_each_drug_is_the_one_drug_model(model, backend):
    sol = model.solve((0, 24), times=TIMES, backend=backend, **TIGHT)
    assert np.asarray(sol["Cart"]).shape == (len(TIMES), 2)
    for i, (_, v) in enumerate(DRUGS.items()):
        ref = one_drug(**v).solve((0, 24), times=TIMES, backend=backend, **TIGHT)
        for name in ("Cart", "Cven"):
            np.testing.assert_allclose(np.asarray(sol[name])[:, i], ref[name], rtol=1e-8)
        np.testing.assert_allclose(np.asarray(sol["C"])[:, i], ref["C"], rtol=1e-8, atol=1e-14)
        # a slice read with organ left free: the cellular concentration in every organ
        np.testing.assert_allclose(
            np.asarray(sol["cell"])[:, i], np.asarray(ref["C"])[..., 2], rtol=1e-8, atol=1e-14
        )
        np.testing.assert_allclose(np.asarray(ref["cell"]), np.asarray(ref["C"])[..., 2])
    np.testing.assert_allclose(sol["cell_one"], np.asarray(sol["cell"])[:, 1])


def test_the_books_close(model):
    sol = model.solve((0, 24), times=TIMES, **TIGHT)
    assert max(sol.ledger.check().values()) < 1e-10


def test_sparse_jacobian_equals_dense(model):
    cm = model.compile()
    t, y, th = _random_point(cm, 3)
    dense = np.asarray(cm.jacobian(t, y, th))
    sparse = cm.sparse_jacobian(t, y, th)
    np.testing.assert_allclose(sparse.toarray(), dense, rtol=0, atol=1e-13 * np.abs(dense).max())


def test_a_flux_between_unrelated_axes_is_still_refused():
    """A pool per drug cannot feed a field over organs only: neither end's free axes are
    all free at the other end."""
    text = """model bad
axis drug categorical [d1, d2]
axis organ categorical [liver, lung]
Vp(drug) = 1
field B(drug) weights=Vp
field C(organ)
flux J: B -> C = 2*B
d_t(C) = sum_drug(J)
d_t(B) = -sum_organ(J)/Vp"""
    with pytest.raises(tq.NotationError, match="one end's free axes must all be free"):
        tq.Model.from_text(text)
