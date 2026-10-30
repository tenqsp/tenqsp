# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 step 2.3: velocities that depend on fields, chemotaxis ``grad``, and fronts.

Oracles from the mathematics:
- face values by hand (flux splitting with the gradient across each face);
- chemotaxis up a fixed attractant S with zero-flux ends: the steady state is the Boltzmann
  profile n ∝ exp(chi S / D);
- Keller-Segel (gallery §5, derived there): the growth rate of a cos(pi x / L) mode is the
  largest eigenvalue of [[-Dn k^2, chi n0 k^2], [alpha, -(Dc k^2 + beta)]];
- Fisher-KPP with D = r = 1: the exact travelling wave (1 + e^{(x - c t)/√6})^-2, c = 5/√6
  (Ablowitz & Zeppetella 1979).
"""

from __future__ import annotations

import numpy as np
import pytest
from convergence import assert_converges
from scipy.integrate import quad

import tenqsp as tq

TIGHT = {"rtol": 1e-11, "atol": 1e-13}


# --------------------------------------------------------------------------- face values
@pytest.mark.parametrize(
    "faces", [np.linspace(0, 1, 7), np.array([0, 0.1, 0.3, 0.35, 0.6, 0.9, 1.0])],
    ids=["uniform", "uneven"],
)  # fmt: skip
def test_velocity_with_a_gradient_and_the_coordinate_by_hand(faces):
    """v = chi grad(S, x) + w (x - 0.5) S on the faces: the gradient across each face (over
    the distance between the two centres), the coordinate at the face, S interpolated to the
    face linearly by distance (end cells' values on the ends, and the nearest interior
    gradient there). Mixing the coordinate with a field was refused in Phase 1. The uneven
    grid has width ratios up to 5 between neighbours, where the interpolation weights matter."""
    n, chi, w = len(faces) - 1, 0.7, 0.3
    m = tq.Model("by_hand")
    x = m.axis("x", faces=list(faces), kind="spatial")
    S = m.field("S", axes=[x])
    B = m.field("B", axes=[x])
    m.advect(B, axis=x, velocity=chi * tq.grad(S, x) + w * (x.coord - 0.5) * S)
    m.outflow(B, axis=x, at="upper")
    m.outflow(B, axis=x, at="lower")
    cm = m.compile()
    rng = np.random.default_rng(23)
    s, b = rng.uniform(0.5, 2.0, n), rng.uniform(0.5, 2.0, n)
    y = np.concatenate([s, b])
    dB = np.asarray(cm.unpack(cm.rhs(0.0, y))["B"])

    width = np.diff(faces)
    c = 0.5 * (faces[1:] + faces[:-1])
    g = np.diff(s) / np.diff(c)
    g = np.concatenate([g[:1], g, g[-1:]])
    wr = (faces[1:-1] - c[:-1]) / (c[1:] - c[:-1])  # distance from the left centre
    s_f = np.concatenate([s[:1], (1 - wr) * s[:-1] + wr * s[1:], s[-1:]])
    v = chi * g + w * (faces - 0.5) * s_f
    left = np.concatenate([[0.0], b])  # upwind values: left cell (none outside the lower end)
    right = np.concatenate([b, [0.0]])
    F = np.maximum(v, 0) * left + np.minimum(v, 0) * right
    np.testing.assert_allclose(dB, -(F[1:] - F[:-1]) / width, rtol=1e-13)
    np.testing.assert_allclose(cm.unpack(cm.rhs(0.0, y))["S"], 0.0)  # S has no terms


# --------------------------------------------------------------------------- Boltzmann
CHI, DB = 0.8, 0.1


def attractant(x):
    return np.exp(-(((x - 0.6) / 0.2) ** 2)) + 0.5 * x


def boltzmann(n: int, scheme: str):
    """B climbs a fixed attractant S (S has no equation of its own) and diffuses."""
    m = tq.Model("boltzmann")
    x = m.axis("x", 0, 1, n, kind="spatial")
    S = m.field("S", axes=[x])
    B = m.field("B", axes=[x])
    m.init(S, tq.exp(-(((x.coord - 0.6) / 0.2) ** 2)) + 0.5 * x.coord)
    m.advect(B, axis=x, velocity=CHI * tq.grad(S, x), scheme=scheme)
    m.diffuse(B, axis=x, D=DB)
    m.init(B, 1.0)
    return m


def boltzmann_cell_averages(n: int) -> np.ndarray:
    def dens(x):
        return np.exp(CHI * attractant(x) / DB)

    faces = np.linspace(0, 1, n + 1)
    cells = np.array(
        [
            quad(dens, a, b, epsabs=0, epsrel=1e-13)[0]
            for a, b in zip(faces[:-1], faces[1:], strict=True)
        ]
    )
    return cells / cells.sum() * n  # total amount 1 on [0, 1]


@pytest.mark.parametrize("scheme, order", [("sharp", 2.0), ("erlang", 1.0)])
def test_chemotaxis_reaches_the_boltzmann_profile(scheme, order):
    def l1_error(n):
        sol = boltzmann(n, scheme).solve((0, 40), times=[40], **TIGHT)
        u = sol["B"][-1]
        assert abs(u.sum() / n - 1) < 1e-12 and u.min() > 0
        return np.sum(np.abs(u - boltzmann_cell_averages(n))) / n

    assert_converges(l1_error, [40, 80, 160], order, exact=0.0)


# ------------------------------------------------- velocity from cell values, rough grid
W_DRIFT = 2.0
_BASE = np.random.default_rng(5).uniform(0.5, 2.0, 10)  # widths of 10 base cells, ratio <= 4
_BASE_FACES = np.concatenate([[0.0], np.cumsum(_BASE)]) / _BASE.sum()


def _rough_faces(n: int) -> np.ndarray:
    """Each of 10 uneven base cells split into n/10 equal parts (refines a rough grid)."""
    k = n // 10
    parts = [
        np.linspace(a, b, k + 1) for a, b in zip(_BASE_FACES[:-1], _BASE_FACES[1:], strict=True)
    ]
    return np.unique(np.concatenate(parts))


@pytest.mark.parametrize("scheme, order", [("sharp", 2.0), ("erlang", 1.0)])
def test_drift_made_of_cell_values_on_a_rough_grid(scheme, order):
    """v = w S with S a field (interpolated from the cells to the faces), diffusion D,
    zero-flux ends: the steady state is n ∝ exp((w/D) int_0^x S), here with S = sin 3x + x.
    Cell amounts converge at the scheme's order on a grid of uneven cells."""

    def dens(x):
        return np.exp(W_DRIFT * ((1 - np.cos(3 * x)) / 3 + x * x / 2) / DB)

    def l1_error(n):
        f = _rough_faces(n)
        m = tq.Model("drift")
        x = m.axis("x", faces=list(f), kind="spatial")
        S = m.field("S", axes=[x])
        B = m.field("B", axes=[x])
        m.init(S, tq.sin(3 * x.coord) + x.coord)
        m.advect(B, axis=x, velocity=W_DRIFT * S, scheme=scheme)
        m.diffuse(B, axis=x, D=DB)
        m.init(B, 1.0)
        amount = m.solve((0, 30), times=[30], **TIGHT)["B"][-1] * np.diff(f)
        exact = np.array(
            [
                quad(dens, a, b, epsabs=0, epsrel=1e-13)[0]
                for a, b in zip(f[:-1], f[1:], strict=True)
            ]
        )
        return np.sum(np.abs(amount / amount.sum() - exact / exact.sum()))

    assert_converges(l1_error, [40, 80, 160], order, exact=0.0)


# --------------------------------------------------------------------------- V43
DN, DC, ALPHA, BETA, N0 = 0.1, 1.0, 1.0, 1.0, 1.0


def keller_segel(n: int, chi: float, eps: float = 1e-6, scheme: str = "sharp"):
    """n_t = Dn n_xx - (chi n c_x)_x,  c_t = Dc c_xx + alpha n - beta c, zero flux on [0, 1],
    starting from the uniform state plus a small bump."""
    m = tq.Model("keller_segel")
    x = m.axis("x", 0, 1, n, kind="spatial")
    N = m.field("n", axes=[x])
    C = m.field("c", axes=[x])
    X, A, B = m.params(chi=chi, alpha=ALPHA, beta=BETA)
    m.advect(N, axis=x, velocity=X * tq.grad(C, x), scheme=scheme)
    m.diffuse(N, axis=x, D=DN)
    m.diffuse(C, axis=x, D=DC)
    m.reaction(C, A * N - B * C)
    m.init(N, N0 * (1 + eps * tq.exp(-(((x.coord - 0.3) / 0.1) ** 2))))
    m.init(C, ALPHA * N0 / BETA)
    return m


def linear_growth_rate(chi: float, k: float = np.pi) -> float:
    M = np.array([[-DN * k**2, chi * N0 * k**2], [ALPHA, -(DC * k**2 + BETA)]])
    return float(np.max(np.linalg.eigvals(M).real))


@pytest.mark.parametrize("chi", [5.0, 0.5], ids=["unstable", "stable"])
def test_v43_keller_segel_linear_growth_rate(chi):
    """The cos(pi x) part of the perturbation grows (or decays) at the predicted rate, measured
    between t = 1 and 2 (the fast eigenvalue has died out by then). Instability needs
    chi alpha n0 > Dn (Dc k^2 + beta)."""
    n = 100
    sol = keller_segel(n, chi).solve((0, 2), times=[1.0, 2.0], rtol=1e-10, atol=1e-16)
    xc = (np.arange(n) + 0.5) / n
    mode = np.cos(np.pi * xc)  # the discrete cosines are orthogonal on cell centres
    a1, a2 = (np.dot(sol["n"][i] - N0, mode) for i in range(2))
    rate = np.log(abs(a2 / a1))
    assert rate == pytest.approx(linear_growth_rate(chi), rel=1e-3)
    assert (rate > 0) == (chi * ALPHA * N0 > DN * (DC * np.pi**2 + BETA))
    np.testing.assert_allclose(sol["n"].mean(axis=1), sol["n"][0].mean(), rtol=1e-13)
    assert max(sol.ledger.check().values()) < 1e-10


def test_keller_segel_blows_up_into_a_peak_but_stays_positive():
    """Past the linear phase the cells aggregate: the books still close, nothing goes
    negative, and the total is kept."""
    sol = keller_segel(60, 5.0, eps=1e-2).solve((0, 3), times=[0, 1, 3])
    assert sol["n"][-1].max() > 1.5 * sol["n"][0].max()
    assert sol["n"].min() > 0
    np.testing.assert_allclose(sol["n"].mean(axis=1), sol["n"][0].mean(), rtol=1e-10)
    assert max(sol.ledger.check().values()) < 1e-10


# --------------------------------------------------------------------------- V42
C_FISHER = 5 / np.sqrt(6)


def fisher_exact(x, t):
    return (1 + np.exp((x - C_FISHER * t) / np.sqrt(6))) ** -2


def fisher(n: int, L0: float = -10.0, L1: float = 30.0):
    """u_t = u_xx + u (1 - u); both ends held at the exact solution (time-dependent)."""
    m = tq.Model("fisher")
    x = m.axis("x", L0, L1, n, kind="spatial")
    u = m.field("u", axes=[x])
    m.diffuse(u, axis=x, D=1.0)
    m.reaction(u, u * (1 - u))
    held = "(1 + exp(({x} - 5/sqrt(6)*t)/sqrt(6)))^(-2)"
    m.boundary(u, axis=x, at="lower", kind="dirichlet", value=held.format(x=L0))
    m.boundary(u, axis=x, at="upper", kind="dirichlet", value=held.format(x=L1))
    m.init(u, (1 + tq.exp(x.coord / np.sqrt(6))) ** -2)
    return m


def test_v42_fisher_travelling_wave():
    t_end = 4.0

    def max_error(n):
        sol = fisher(n).solve((0, t_end), times=[t_end], **TIGHT)
        faces = np.linspace(-10, 30, n + 1)
        gx, gw = np.polynomial.legendre.leggauss(6)
        c, h = 0.5 * (faces[1:] + faces[:-1]), faces[1] - faces[0]
        exact = sum(
            0.5 * wi * fisher_exact(c + 0.5 * h * xi, t_end) for xi, wi in zip(gx, gw, strict=True)
        )
        return np.max(np.abs(sol["u"][-1] - exact))

    assert_converges(max_error, [100, 200, 400], 2.0, exact=0.0)


# --------------------------------------------------------------------------- validation
def _two_fields():
    m = tq.Model("bad")
    x = m.axis("x", 0, 1, 5, kind="spatial")
    y = m.axis("y", 0, 1, 4, kind="spatial")
    s = m.axis("s", labels=["a", "b"])
    S = m.field("S", axes=[x, y])
    B = m.field("B", axes=[x, y, s])
    P = m.compartment("P")
    return m, x, y, s, S, B, P


@pytest.mark.parametrize(
    "build, message",
    [
        (
            lambda m, x, y, s, S, B, P: m.reaction(B, tq.grad(S, x)),
            "only be used in a velocity along",
        ),
        (lambda m, x, y, s, S, B, P: m.advect(B, axis=y, velocity=tq.grad(S, x)), "same axis"),
        (lambda m, x, y, s, S, B, P: m.advect(B, axis=x, velocity="grad(P, x)"), "not a field"),
        (
            lambda m, x, y, s, S, B, P: m.advect(B, axis=x, velocity="grad(S, s)"),
            "not a continuous axis",
        ),
        (
            lambda m, x, y, s, S, B, P: m.advect(
                B, axis=x, velocity=m.define("drift", tq.grad(S, x))
            ),
            "only be used in a velocity along",
        ),
    ],
    ids=["in a reaction", "other axis", "of a compartment", "categorical axis", "in a definition"],
)
def test_gradients_are_checked(build, message):
    m, *handles = _two_fields()
    build(m, *handles)
    with pytest.raises(tq.ModelError, match=message):
        m.compile()


def test_grad_needs_a_field_and_an_axis():
    m, x, y, s, S, B, P = _two_fields()
    m.advect(B, axis=x, velocity="grad(S)")
    with pytest.raises(tq.ModelError, match="grad takes a field and an axis"):
        m.compile()
    m, x, y, s, S, B, P = _two_fields()
    with pytest.raises(ValueError, match="categorical"):
        tq.grad(S, s)
    with pytest.raises(TypeError, match="must be a field"):
        tq.grad(P, x)
