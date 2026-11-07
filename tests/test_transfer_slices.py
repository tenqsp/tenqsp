# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``m.transfer`` with an end that fixes an axis by its slice, ``A.at(organ="liver")``.

Oracle, by hand: with ``A, B`` over (drug, organ) and rate r, ``A[liver] -> B`` gives every
element of B ``r * A[d, liver]`` and A[d, liver] loses the sum over organ; the reverse,
``B -> A[liver]``, gives A[d, liver] the sum over organ of ``r * B[d, o]``. The rates add up
to zero and the ledger closes.
"""

from __future__ import annotations

import numpy as np
import pytest

import tenqsp as tq

R = 2.0


def _model(source_slice, *, reverse=False):
    m = tq.Model("slices")
    d = m.axis("drug", labels=["d1", "d2"])
    o = m.axis("organ", labels=["gut", "liver"])
    A = m.field("A", axes=[d, o])
    B = m.field("B", axes=[d, o])
    m.init(A, 1.0)
    m.init(B, 0.5)
    if reverse:
        m.transfer(B, to=A.at(**source_slice), rate=str(R))
    else:
        m.transfer(A.at(**source_slice), to=B, rate=str(R))
    return m


def _rates(m):
    cm = m.compile()
    y = np.arange(1.0, 9.0)  # A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] (drug x organ)
    du = np.asarray(cm.rhs(0.0, y))
    sl = cm.layout.slices
    return du[sl["A"]].reshape(2, 2), du[sl["B"]].reshape(2, 2)


@pytest.mark.parametrize(
    ("where", "dA", "dB"),
    [
        ({"organ": "liver"}, [[0, -8], [0, -16]], [[4, 4], [8, 8]]),
        ({"drug": "d1", "organ": "liver"}, [[0, -16], [0, 0]], [[4, 4], [4, 4]]),
    ],
)
def test_from_a_slice(where, dA, dB):
    m = _model(where)
    assert m.validate(raise_on_error=False).ok
    a, b = _rates(m)
    np.testing.assert_allclose(a, dA)
    np.testing.assert_allclose(b, dB)


def test_into_a_slice():
    a, b = _rates(_model({"organ": "liver"}, reverse=True))
    np.testing.assert_allclose(b, -R * np.array([[5, 6], [7, 8]]))
    np.testing.assert_allclose(a, [[0, R * 11], [0, R * 15]])


@pytest.mark.parametrize("reverse", [False, True])
def test_the_ledger_closes(reverse):
    sol = _model({"organ": "liver"}, reverse=reverse).solve(
        (0, 1), times=[0.5, 1.0], rtol=1e-10, atol=1e-12
    )
    assert max(sol.ledger.check().values()) < 1e-8
