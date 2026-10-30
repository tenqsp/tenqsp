# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""A shape inside a density is the curve itself; point() on a face rounded by the grid.

Oracles from the math: on z in [0, 1], the N(0, 0.5) curve holds Phi(2) - Phi(0) of its mass;
the uniform curve on [0, 2] has density 1/2.
"""

import numpy as np
import pytest
from scipy.stats import norm

import tenqsp as tq

INSIDE = norm.cdf(2.0) - norm.cdf(0.0)  # mass of N(0, 0.5) on [0, 1]


def start(value: str, n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    m = tq.Model.from_text(f"axis z in [0, 1] n={n}\nfield B(z)\nd_t(B) = 0\nB(0) = {value}\n")
    sol = m.solve((0, 1), times=[0.0])
    return np.asarray(sol["B"])[0], np.diff(sol.coords["z"]["faces"])


@pytest.mark.parametrize(
    "value, total",
    [
        ("density(normal(z, mean=0, sd=0.5))", INSIDE),
        ("1e7*density(normal(z, mean=0, sd=0.5))", 1e7 * INSIDE),
        ("density(1e7*normal(z, mean=0, sd=0.5))", 1e7 * INSIDE),
        ("density(normal(z, mean=0, sd=0.5))*2", 2 * INSIDE),
        ("1e7*normal(z, mean=0, sd=0.5)", 1e7),  # an amount and a shape: spread, exactly 1e7
        ("density(uniform(z, lower=0, upper=2))", 0.5),
        ("density(uniform(z))", 1.0),  # open ends: the axis
    ],
)
def test_density_is_the_curve(value, total):
    B, w = start(value)
    assert np.sum(B * w) == pytest.approx(total, rel=1e-12)


def test_density_round_trip():
    m = tq.Model.from_text("axis z in [0, 1] n=20\nfield B(z)\nd_t(B) = 0\n"
                           "B(0) = 3*density(normal(z, mean=0, sd=0.5))\n")  # fmt: skip
    back = tq.Model.from_text(m.to_text())
    a = np.asarray(m.solve((0, 1), times=[0.0])["B"])
    b = np.asarray(back.solve((0, 1), times=[0.0])["B"])
    assert np.array_equal(a, b)


def test_python_value_with_shape_is_a_density():
    m = tq.Model("p")
    z = m.axis("z", 0, 1, 50)
    B = m.field("B", axes=[z])
    m.reaction(B, 0 * B)
    m.init(B, tq.normal(z, mean=0, sd=0.5))
    sol = m.solve((0, 1), times=[0.0])
    w = np.diff(sol.coords["z"]["faces"])
    assert np.sum(np.asarray(sol["B"])[0] * w) == pytest.approx(INSIDE, rel=1e-12)


def test_density_dose_is_the_curve():
    m = tq.Model.from_text("axis z in [0, 1] n=50\nfield B(z)\nd_t(B) = 0\n"
                           "dose B at t=0: 4*density(normal(z, mean=0, sd=0.5))\n")  # fmt: skip
    sol = m.solve((0, 1), times=[1.0])
    w = np.diff(sol.coords["z"]["faces"])
    assert np.sum(np.asarray(sol["B"])[-1] * w) == pytest.approx(4 * INSIDE, rel=1e-12)


@pytest.mark.parametrize("at, cell", [(0.0, 0), (0.3, 3), (0.5, 5), (0.6, 6), (0.7, 7), (0.75, 7),
                                      (1.0, 9)])  # fmt: skip
def test_point_on_a_face_goes_in_the_cell_above(at, cell):
    B, _ = start(f"1*point(z, at={at})", n=10)
    assert int(np.argmax(B)) == cell and np.count_nonzero(B) == 1


def test_density_must_multiply():
    with pytest.raises(tq.NotationError, match="density"):
        tq.Model.from_text("axis z in [0, 1] n=5\nfield B(z)\nd_t(B) = 0\n"
                           "B(0) = 1 + density(normal(z, mean=0, sd=0.5))\n")  # fmt: skip
