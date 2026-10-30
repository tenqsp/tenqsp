# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Axes: geometry, quadrature and validation."""

import numpy as np
import pytest

import tenqsp  # noqa: F401  (enables float64)
from tenqsp.grids import CategoricalAxis, ContinuousAxis


def test_uniform_geometry():
    ax = ContinuousAxis.uniform("a", 0.0, 2.0, 4, kind="one_way", units="h")
    assert ax.n == 4
    np.testing.assert_allclose(ax.faces, [0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(ax.centers, [0.25, 0.75, 1.25, 1.75])
    np.testing.assert_allclose(ax.widths, 0.5)
    np.testing.assert_allclose(ax.center_distances, 0.5)
    assert ax.is_uniform and ax.lower == 0.0 and ax.upper == 2.0


def test_geometric_widths_sum_and_ratio():
    ax = ContinuousAxis.geometric("x", 0.0, 1.0, 5, ratio=0.5)
    np.testing.assert_allclose(ax.widths.sum(), 1.0)
    np.testing.assert_allclose(ax.widths[1:] / ax.widths[:-1], 0.5)
    assert not ax.is_uniform


def test_integral_is_total_amount():
    ax = ContinuousAxis.geometric("x", 0.0, 3.0, 7, ratio=1.3)
    u = np.arange(1.0, 8.0)  # cell averages
    assert float(ax.integral(u)) == pytest.approx(float(np.sum(u * ax.widths)))
    # along dimension 1 of a 2-D field
    U = np.stack([u, 2 * u])
    np.testing.assert_allclose(
        np.asarray(ax.integral(U, axis=1)), [np.sum(u * ax.widths), 2 * np.sum(u * ax.widths)]
    )


def test_geometry_arrays_are_read_only():
    ax = ContinuousAxis.uniform("z", 0, 1, 3)
    with pytest.raises(ValueError):
        ax.faces[0] = 5.0


@pytest.mark.parametrize(
    "faces, kind, msg",
    [
        ([0.0], "spatial", "at least 2"),
        ([0.0, 1.0, 1.0], "spatial", "strictly increasing"),
        ([0.0, 1.0], "sideways", "kind must be one of"),
    ],
)
def test_continuous_axis_validation(faces, kind, msg):
    with pytest.raises(ValueError, match=msg):
        ContinuousAxis("x", np.asarray(faces), kind=kind)


def test_categorical_axis():
    s = CategoricalAxis("s", ["S", "R"])
    assert s.n == 2 and s.index("R") == 1
    with pytest.raises(KeyError, match="has no category 'X'"):
        s.index("X")
    with pytest.raises(ValueError, match="unique"):
        CategoricalAxis("s", ["S", "S"])
