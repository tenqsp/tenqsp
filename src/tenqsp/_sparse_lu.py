# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Sparse LU factorizations with a column-ordering method chosen once per model.

The stiff solvers factor ``I - c J`` many times per run: scipy's BDF and Radau, and the sparse
tier of the JAX backend. ``scipy.sparse.linalg.splu`` orders the columns with COLAMD by
default. That suits most models, but COLAMD fails badly on one structure: a full row with
entries below it. An inflow that depends on an integral of a field (a renewal boundary,
``at z=0: v*u = b*Q`` with ``Q = int_z(u)``) makes such a row, and COLAMD fills in everything
below it. On a 1,000-cell example the factors had 502,502 nonzeros with COLAMD and 4,500
with MMD on A^T + A (28.6 ms against 2.0 ms per factorization), and the run took 62 s
instead of 8 s.

So at a model's first factorization (for each matrix size: with and without the ledger's
rows) the matrix is factored with both methods, and MMD on A^T + A is kept only if it fills in
less than ``1 / FILL_FACTOR`` as much as COLAMD; that factorization is the one returned. Every
later factorization of the model uses the kept method. The choice is made once and from the
fill, which is deterministic, so the same call gives the same numbers every time.

What was measured before settling on this:

- **A wide margin.** At their first factorization the renewal models had 8 to 15 times less
  fill with MMD; every reference model (phage, plaque, PBPK, spheroids, absorption, the
  paper's models) was within 1.9 times. Within that range MMD can be slower although it
  fills in less: its ordering is dearer to compute, and splu computes it at every call (a
  6,121-state phage model: half the fill, 87 ms against 62 ms).
- **COLAMD stays the default.** It bounds the fill for any row pivoting, and in a run SuperLU
  pivots away from the diagonal.
- **The model's own matrix decides, not its structure.** The structure counts every entry the
  Jacobian can have; a matrix in a run is sparser (a value that is 0 at the time drops out)
  and pivots differently. On a three-axis phage model the structure favored MMD, and the run
  was slower with it (2.4 s against 1.3 s).
- **The method is kept, not one ordering.** An ordering computed once and reused filled in
  1.6 to 2.2 times more than one computed on each matrix (phage reference models).
- **Not by timing.** Timing both methods at checkpoints chose well, but made the method, and
  so the last digits of a run, depend on machine load and on earlier runs.
"""

from __future__ import annotations

import threading
import weakref

from scipy.sparse.linalg import splu

DEFAULT = "COLAMD"
ALTERNATIVE = "MMD_AT_PLUS_A"
FILL_FACTOR = 5  # the alternative is kept only with less than 1 / FILL_FACTOR of COLAMD's fill


class Ordering:
    """The column-ordering method of one model's iteration matrices, one per matrix size
    (with and without the ledger's rows), chosen at the model's first factorization."""

    def __init__(self):
        self._spec: dict[int, str] = {}
        self._lock = threading.Lock()

    def spec(self, n: int) -> str | None:
        """The method for size ``n``; None before the first factorization."""
        return self._spec.get(n)

    def factor(self, A):
        """``A`` (CSC) factored by SuperLU with this model's ordering method (the first time:
        with both, keeping the one chosen); raises as ``splu`` does."""
        n = A.shape[0]
        spec = self._spec.get(n)
        if spec is not None:
            return splu(A, permc_spec=spec)
        lus = {s: splu(A, permc_spec=s) for s in (DEFAULT, ALTERNATIVE)}
        fill = {s: lu.L.nnz + lu.U.nnz for s, lu in lus.items()}
        best = ALTERNATIVE if FILL_FACTOR * fill[ALTERNATIVE] < fill[DEFAULT] else DEFAULT
        with self._lock:
            best = self._spec.setdefault(n, best)  # another thread may have chosen first
        return lus[best]


_ORDERINGS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_LOCK = threading.Lock()


def ordering(cm) -> Ordering:
    """The ``Ordering`` of the compiled model ``cm`` (made on first use, kept with it)."""
    got = _ORDERINGS.get(cm)
    if got is None:
        with _LOCK:
            got = _ORDERINGS.setdefault(cm, Ordering())
    return got
