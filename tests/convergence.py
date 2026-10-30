# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Grid-convergence helper shared by the convergence tests.

``assert_converges(quantity, grids, expected_order, exact=...)`` evaluates ``quantity(n)`` on
each grid, measures the observed order between successive grids, and fails if the finest
observed order differs from ``expected_order`` by more than ``tol``. Without ``exact``, the
order comes from three successive grids (``log2`` of the ratio of successive differences, grids
doubling) and the Richardson-extrapolated value is returned.

Solver tolerance: callers solve with tolerances well below the finest-grid spatial error; the
helper checks that the finest error is at least ``floor`` so the solver floor cannot pass as
spatial error.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def observed_orders(errors: Sequence[float], grids: Sequence[int]) -> np.ndarray:
    e, n = np.asarray(errors, float), np.asarray(grids, float)
    return np.log(e[:-1] / e[1:]) / np.log(n[1:] / n[:-1])


def assert_converges(
    quantity: Callable[[int], float],
    grids: Sequence[int],
    expected_order: float,
    *,
    exact: float | None = None,
    tol: float = 0.3,
    floor: float = 1e-12,
) -> dict:
    values = np.array([float(quantity(n)) for n in grids])
    if exact is not None:
        errors = np.abs(values - exact)
        orders = observed_orders(errors, grids)
        best = values[-1]
    else:
        if len(grids) < 3:
            raise ValueError("without an exact value, give at least three grids")
        diffs = np.abs(np.diff(values))
        errors = diffs
        orders = observed_orders(diffs, grids[1:])
        p = orders[-1]
        r = grids[-1] / grids[-2]
        best = values[-1] + (values[-1] - values[-2]) / (r**p - 1)
    report = {"values": values, "errors": errors, "orders": orders, "extrapolated": best}
    assert errors[-1] > floor, f"finest error {errors[-1]:.1e} is at the solver floor: {report}"
    assert abs(orders[-1] - expected_order) <= tol, (
        f"observed order {orders[-1]:.2f}, expected {expected_order} ± {tol}: {report}"
    )
    return report
