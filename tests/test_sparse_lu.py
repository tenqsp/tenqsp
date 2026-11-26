# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The sparse LU's ordering method, chosen once per model (``tenqsp._sparse_lu``).

COLAMD, scipy's default, fills in everything below a full row; an inflow that depends on an
integral of the field makes one. Such a model is factored with MMD on A^T + A instead, the
results do not change beyond rounding, and a run repeats to the last digit.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

import tenqsp as tq
from tenqsp import _sparse_lu
from tenqsp.compile import _lu_size

RENEWAL = """
model renewal
axis z in [0, 1] n={n} kind=one_way
param b=3, K=10, v=0.5, mu=0.2, kd=0.1
field u(z)
Q = int_z(u)
d_t(u) = -d_z(v*u) - mu*u
    at z=0: v*u = b*Q*(1 - Q/K)
    at z=1: outflow
d_t(M) = v*u(z=1) - kd*M
u(0) = 1
"""


def renewal(n: int) -> tq.Model:
    return tq.Model.from_text(RENEWAL.format(n=n))


def test_factor_solves_as_splu_does():
    rng = np.random.default_rng(3)
    n = 200
    A = sps.random_array((n, n), density=0.02, random_state=rng, format="csc")
    A = (A + sps.diags_array(np.abs(A).sum(axis=1) + 1.0)).tocsc()
    o = _sparse_lu.Ordering()
    b = rng.normal(size=(n, 3))
    for _ in range(3):  # the first factorization, then the kept method
        lu = o.factor(A)
        np.testing.assert_allclose(A @ lu.solve(b), b, atol=1e-12)
        np.testing.assert_allclose(A.T @ lu.solve(b, trans="T"), b, atol=1e-12)


def test_an_integral_inflow_moves_the_model_off_colamd():
    m = renewal(1000)
    cm = m.compile()
    m.solve((0, 1), times=[1])
    # the solver's matrix carries the ledger's rows too: one entry per matrix size
    assert set(_sparse_lu.ordering(cm)._spec.values()) == {_sparse_lu.ALTERNATIVE}
    # the estimate in summary() is the alternative's: ~4,500 nonzeros, not ~500,000
    assert _lu_size(cm) < 0.5


def test_results_do_not_depend_on_the_method(monkeypatch):
    m = renewal(300)
    times = [0.5, 2.0]
    learned = m.solve((0, 2), times=times)
    monkeypatch.setattr(_sparse_lu, "FILL_FACTOR", np.inf)  # COLAMD always
    monkeypatch.setattr(_sparse_lu, "_ORDERINGS", type(_sparse_lu._ORDERINGS)())
    colamd = m.solve((0, 2), times=times)
    for name in ("u", "M"):
        np.testing.assert_allclose(
            np.asarray(learned[name]), np.asarray(colamd[name]), rtol=1e-6, atol=1e-9
        )


@pytest.mark.parametrize("n", [40, 400])
def test_a_model_without_the_full_row_keeps_colamd(n):
    """Transport alone: no integral, no full row; the alternative is never chosen, and a second
    run repeats the first to the last digit."""
    m = tq.Model.from_text(
        f"""
model decay
axis z in [0, 1] n={n} kind=one_way
param v=0.5, mu=0.2
field u(z)
d_t(u) = -d_z(v*u) - mu*u
    at z=0: v*u = 0
    at z=1: outflow
u(0) = 1
"""
    )
    first = np.asarray(m.solve((0, 1), times=[1])["u"])
    again = np.asarray(m.solve((0, 1), times=[1])["u"])
    np.testing.assert_array_equal(first, again)
    cm = m.compile()
    # empty when no sparse matrix was factored (a small model on the JAX backend's dense tier)
    assert set(_sparse_lu.ordering(cm)._spec.values()) <= {_sparse_lu.DEFAULT}
