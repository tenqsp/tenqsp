# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 step 2.2: drift-diffusion steady states and the boundary conditions that exchange
with the outside.

Every oracle comes from the mathematics, not from the code:
- V5, V29: zero-flux steady states of Fokker-Planck equations, n ∝ exp(∫ a/D) / D;
- V26: the closed-form availability of the dispersion model with Danckwerts ends;
- V27: the cosh profile and effectiveness factor of a reacting slab, and the series-resistance
  steady state of a slab exchanging with a bulk compartment;
- Robin with advection: the steady advection-diffusion-decay equation solved by hand
  (two exponentials, two boundary conditions).
"""

from __future__ import annotations

import numpy as np
import pytest
from convergence import assert_converges
from scipy.integrate import quad
from scipy.special import erf

import tenqsp as tq

TIGHT = {"rtol": 1e-11, "atol": 1e-14}


# --------------------------------------------------------------------------- V5
def ou(n: int, scheme: str, D: float = 0.5, L: float = 4.0):
    """dx = -x dt + sqrt(2D) dW: drift -x, diffusion D, zero-flux ends at ±L."""
    m = tq.Model("ou")
    x = m.axis("x", -L, L, n, kind="bidirectional")
    u = m.field("u", axes=[x])
    m.advect(u, axis=x, velocity=-x.coord, scheme=scheme)
    m.diffuse(u, axis=x, D=D)
    m.init(u, total=1.0, shape=tq.normal(x, mean=1.5, sd=0.3))
    return m


def gaussian_cell_averages(n: int, D: float = 0.5, L: float = 4.0) -> np.ndarray:
    faces = np.linspace(-L, L, n + 1)
    cdf = 0.5 * (1 + erf(faces / np.sqrt(2 * D)))
    return np.diff(cdf) / np.diff(faces) / (cdf[-1] - cdf[0])


@pytest.mark.parametrize("scheme, order", [("sharp", 2.0), ("erlang", 1.0)])
def test_v5_ornstein_uhlenbeck_steady_state(scheme, order):
    """Velocity -x changes sign at 0: the steady state is the Gaussian with variance D,
    with no negative values and the mass kept to round-off."""

    def l1_error(n):
        sol = ou(n, scheme).solve((0, 30), times=[30], **TIGHT)
        u = sol["u"][-1]
        assert u.min() > 0
        assert abs(np.sum(u) * 8 / n - 1) < 1e-13
        return np.sum(np.abs(u - gaussian_cell_averages(n))) * 8 / n

    assert_converges(l1_error, [40, 80, 160], order, exact=0.0)


# --------------------------------------------------------------------------- V29
TH, XI0, D0 = 2.0, 0.3, 0.05


def dormancy(n: int, *, g=0.0, d=0.0, E=0.0):
    """Gallery Model 7.3: dormancy depth xi in [0, 1], SDE drift a = -TH (xi - XI0), diffusion
    D(xi) = D0 (1 + xi). The Fokker-Planck flux needs the drift corrected by -dD/dxi; growth
    falls and drug susceptibility falls with depth, crowding through the total N."""
    m = tq.Model("dormancy")
    xi = m.axis("xi", 0, 1, n, kind="bidirectional")
    B = m.field("B", axes=[xi])
    gp, dp, Ep, K = m.params(g=g, d=d, E=E, K=10.0)
    N = m.integral(B, over=xi, name="N")
    m.advect(B, axis=xi, velocity=-TH * (xi.coord - XI0) - D0, scheme="sharp")
    m.diffuse(B, axis=xi, D=D0 * (1 + xi.coord))
    m.reaction(B, (gp * (1 - xi.coord) * (1 - N / K) - dp - Ep * (1 - xi.coord) ** 2) * B)
    m.init(B, total=1.0, shape=tq.normal(xi, mean=0.8, sd=0.05))
    return m


def dormancy_steady_cell_averages(n: int) -> np.ndarray:
    """n(xi) ∝ exp(∫ a/D) / D = (1 + xi)^(TH (1 + XI0) / D0 - 1) exp(-TH xi / D0)."""
    p = TH * (1 + XI0) / D0 - 1

    def dens(x):
        return np.exp(p * np.log1p(x) - TH * x / D0 - (p * np.log1p(XI0) - TH * XI0 / D0))

    faces = np.linspace(0, 1, n + 1)
    cells = np.array(
        [
            quad(dens, a, b, epsabs=0, epsrel=1e-13)[0]
            for a, b in zip(faces[:-1], faces[1:], strict=True)
        ]
    )
    return cells / np.diff(faces) / cells.sum()


def test_v29_dormancy_conserves_mass_and_reaches_its_steady_state():
    """No growth, death or drug: the total stays 1 to round-off, and the profile converges
    at second order to the exact zero-flux steady state (variable diffusivity, Itô drift
    correction)."""

    def l1_error(n):
        sol = dormancy(n).solve((0, 40), times=[5, 40], **TIGHT)
        totals = sol["B"].sum(axis=1) / n
        np.testing.assert_allclose(totals, 1.0, rtol=0, atol=1e-13)
        return np.sum(np.abs(sol["B"][-1] - dormancy_steady_cell_averages(n))) / n

    assert_converges(l1_error, [40, 80, 160], 2.0, exact=0.0)


def test_v29_dormancy_books_close_with_growth_death_and_drug():
    sol = dormancy(60, g=1.0, d=0.1, E=0.5).solve((0, 10), times=[2, 5, 10])
    assert max(sol.ledger.check().values()) < 1e-10
    assert sol["B"].min() > 0


# --------------------------------------------------------------------------- V26
def availability(DN: float, RN: float) -> float:
    """Roberts & Rowland (1986) / Wehner-Wilhelm: F = C(1)/C_in with Danckwerts ends."""
    a = np.sqrt(1 + 4 * RN * DN)
    return (
        4
        * a
        / ((1 + a) ** 2 * np.exp((a - 1) / (2 * DN)) - (1 - a) ** 2 * np.exp(-(a + 1) / (2 * DN)))
    )


def dispersion(n: int, DN: float, RN: float, scheme: str = "sharp", bolus: bool = False):
    """dC/dθ = DN C'' - C' - RN C on Z in [0, 1]. Danckwerts: C - DN C' = C_in at the inlet is
    a prescribed total flux (inflow 1·C_in), C' = 0 at the outlet is an outflow."""
    m = tq.Model("dispersion")
    z = m.axis("Z", 0, 1, n, kind="spatial")
    C = m.field("C", axes=[z])
    D, R, Cin = m.params(DN=DN, RN=RN, Cin=1.0)
    m.advect(C, axis=z, velocity=1.0, scheme=scheme)
    m.diffuse(C, axis=z, D=D)
    m.hazard(C, R)
    m.inflow(C, axis=z, at="lower", flux=Cin)
    m.outflow(C, axis=z, at="upper")
    if bolus:
        m.init(C, total=1.0, shape=tq.normal(z, mean=0.1, sd=0.03))
    else:
        m.init(C, 0.0)
    return m


def outlet(n, DN, RN, scheme="sharp", t_end=30.0):
    sol = dispersion(n, DN, RN, scheme).solve((0, t_end), times=[t_end], **TIGHT)
    assert max(sol.ledger.check().values()) < 1e-10
    return sol["C"][-1][-1]


def test_v26_dispersion_availability_converges_at_second_order():
    DN, RN = 0.17, 2.0
    assert_converges(
        lambda n: outlet(n, DN, RN),
        [20, 40, 80],
        2.0,
        exact=availability(DN, RN),
        tol=0.6,  # 2.6 here, falling toward 2 on finer grids
    )
    # the upwind (erlang) scheme adds numerical dispersion: first order
    assert_converges(
        lambda n: outlet(n, DN, RN, "erlang"), [20, 40, 80], 1.0, exact=availability(DN, RN)
    )


def test_v26_dispersion_limits():
    """Small dispersion number: parallel-tube F = exp(-RN); large: well-stirred 1/(1 + RN)."""
    RN = 2.0
    assert abs(outlet(400, 1e-3, RN, t_end=40) / np.exp(-RN) - 1) < 0.01
    assert abs(outlet(100, 1e3, RN, t_end=40) * (1 + RN) - 1) < 0.01


def test_v26_dispersion_bolus_mass_balance():
    """A bolus through the organ, no feed: what came out + what was eliminated + what is
    left = the dose."""
    m = dispersion(60, 0.17, 2.0, bolus=True)
    sol = m.solve((0, 30), times=[1, 30], params={"Cin": 0.0})
    moved = sol.ledger.moved["C"]
    out = -moved["boundary Z=upper: outflow (absorbed)"][-1]
    eliminated = -moved["hazard: RN"][-1]
    left = sol.ledger.amount["C"][-1]
    np.testing.assert_allclose(out + eliminated + left, 1.0, rtol=1e-10)
    assert out > 0.05 and eliminated > 0.5


# --------------------------------------------------------------------------- V27
L_, DE, KR = 1.0, 0.1, 0.4  # Thiele modulus phi = L sqrt(k / De) = 2
PHI = L_ * np.sqrt(KR / DE)


def biofilm(n: int, *, kind="dirichlet", h=0.5, V=2.0, R=1.0, CL=0.3):
    """Gallery Model 6 (Stewart 1996): a reacting slab sealed at z = 0, open to the liquid at
    z = L. ``dirichlet``: the liquid is held at 1. ``bulk``: the liquid is a compartment
    (amount M in volume V, infusion R, clearance CL) exchanging through a mass-transfer
    coefficient h, the amount crossing the surface taken from M."""
    m = tq.Model("biofilm")
    z = m.axis("z", 0, L_, n, kind="spatial")
    C = m.field("C", axes=[z])
    m.diffuse(C, axis=z, D=DE)
    m.hazard(C, KR)
    if kind == "dirichlet":
        m.boundary(C, axis=z, at="upper", kind="dirichlet", value=1.0)
    else:
        M = m.compartment("M")
        m.boundary(C, axis=z, at="upper", kind="robin", value=M / V, h=h, to=M)
        m.ode(M, R - CL * M / V)
        m.init(M, 0.0)
    m.init(C, 0.0)
    return m


def test_v27_biofilm_cosh_profile_and_effectiveness_factor():
    """C(z)/C_b = cosh(phi z/L)/cosh(phi); mean / C_b = tanh(phi)/phi; second order."""

    def profile_error(n):
        sol = biofilm(n).solve((0, 60), times=[60], **TIGHT)
        faces = np.linspace(0, L_, n + 1)
        exact = np.diff(np.sinh(PHI * faces / L_)) / (PHI / L_) / np.diff(faces) / np.cosh(PHI)
        return np.max(np.abs(sol["C"][-1] - exact))

    assert_converges(profile_error, [10, 20, 40], 2.0, exact=0.0)

    def eta(n):
        return float(np.mean(biofilm(n).solve((0, 60), times=[60], **TIGHT)["C"][-1]))

    assert_converges(eta, [10, 20, 40], 2.0, exact=np.tanh(PHI) / PHI)


def test_v27_biofilm_exchanges_mass_with_the_bulk_compartment():
    """Robin surface routed to the bulk: the steady bulk level is R / (CL + G), with G the
    series conductance h·phi·tanh(phi) / (Bi + phi·tanh(phi)), Bi = hL/De; the books of the
    slab and of the bulk both close, and what one gains the other loses."""
    h, V, R, CL = 0.5, 2.0, 1.0, 0.3
    pt = PHI * np.tanh(PHI)
    G = h * pt / (h * L_ / DE + pt)
    exact = R / (CL + G)

    def bulk(n):
        sol = biofilm(n, kind="bulk", h=h, V=V, R=R, CL=CL).solve((0, 200), times=[200], **TIGHT)
        return sol["M"][-1] / V

    assert_converges(bulk, [10, 20, 40], 2.0, exact=exact)
    sol = biofilm(20, kind="bulk").solve((0, 50), times=[10, 50])
    assert max(sol.ledger.check().values()) < 1e-10
    into_slab = sol.ledger.moved["C"]["boundary z=upper: robin with M"]
    out_of_bulk = sol.ledger.moved["M"]["robin exchange with C at z=upper"]
    np.testing.assert_allclose(into_slab, -out_of_bulk, rtol=1e-14)
    assert into_slab[-1] > 0


# --------------------------------------------------------------------------- Robin + advection
def steady_amount(v, Dc, k, h, cout, J, robin_at):
    """∫_0^1 C dz for D C'' - v C' - k C = 0 (v > 0). Ends: ``robin_at="lower"``: inlet
    v C - D C' = v cout + h (cout - C), outlet C' = 0. ``robin_at="upper"``: inlet v C - D C'
    = J, outlet D C' = h (cout - C) (the flow leaves carrying C)."""
    r = np.sqrt(v * v + 4 * k * Dc)
    m1, m2 = (v + r) / (2 * Dc), (v - r) / (2 * Dc)
    if robin_at == "lower":
        A = np.array([[v + h - Dc * m1, v + h - Dc * m2], [m1 * np.exp(m1), m2 * np.exp(m2)]])
        rhs = np.array([(v + h) * cout, 0.0])
    else:
        A = np.array(
            [[v - Dc * m1, v - Dc * m2], [(Dc * m1 + h) * np.exp(m1), (Dc * m2 + h) * np.exp(m2)]]
        )
        rhs = np.array([J, h * cout])
    a, b = np.linalg.solve(A, rhs)
    return a * np.expm1(m1) / m1 + b * np.expm1(m2) / m2


def robin_advection(n, robin_at, scheme, *, v=1.0, Dc=0.2, k=1.5, h=2.0, cout=3.0, J=1.0):
    m = tq.Model("robin_adv")
    z = m.axis("z", 0, 1, n, kind="spatial")
    C = m.field("C", axes=[z])
    W = m.compartment("W")  # the outside reservoir, credited with whatever crosses the end
    m.advect(C, axis=z, velocity=v, scheme=scheme)
    m.diffuse(C, axis=z, D=Dc)
    m.hazard(C, k)
    if robin_at == "lower":
        m.boundary(C, axis=z, at="lower", kind="robin", value=cout, h=h, to=W)
        m.outflow(C, axis=z, at="upper")
    else:
        m.inflow(C, axis=z, at="lower", flux=J)
        m.boundary(C, axis=z, at="upper", kind="robin", value=cout, h=h, to=W)
    m.ode(W, 0)
    m.init(C, 0.0)
    m.init(W, 0.0)
    return m


@pytest.mark.parametrize("robin_at", ["lower", "upper"])
def test_robin_end_with_advection(robin_at):
    """Flow entering through a Robin end carries the outside level, flow leaving carries the
    edge value, and the exchange acts in series with the half-cell of diffusion: the steady
    amount converges at second order (sharp) to the hand solution, and every unit crossing
    the end is debited from (or credited to) the reservoir."""
    exact = steady_amount(1.0, 0.2, 1.5, 2.0, 3.0, 1.0, robin_at)

    def amount(n):
        sol = robin_advection(n, robin_at, "sharp").solve((0, 20), times=[20], **TIGHT)
        return sol.ledger.amount["C"][-1]

    assert_converges(amount, [40, 80, 160], 2.0, exact=exact)
    assert_converges(
        lambda n: (
            robin_advection(n, robin_at, "erlang")
            .solve((0, 20), times=[20], **TIGHT)
            .ledger.amount["C"][-1]
        ),
        [40, 80, 160],
        1.0,
        exact=exact,
    )
    sol = robin_advection(30, robin_at, "sharp").solve((0, 5), times=[1, 5])
    assert max(sol.ledger.check().values()) < 1e-10
    label = f"boundary z={robin_at}: robin with W"
    np.testing.assert_allclose(
        sol.ledger.moved["C"][label],
        -sol.ledger.moved["W"][f"robin exchange with C at z={robin_at}"],
        rtol=1e-14,
    )


def test_dirichlet_end_routed_to_a_compartment():
    """A Dirichlet end can exchange with a compartment too: the reservoir loses what enters."""
    m = tq.Model("held")
    z = m.axis("z", 0, 1, 20, kind="spatial")
    C = m.field("C", axes=[z])
    Res = m.compartment("Res")
    m.diffuse(C, axis=z, D=0.1)
    m.boundary(C, axis=z, at="lower", kind="dirichlet", value=2.0, to=Res)
    m.ode(Res, 0)
    m.init(C, 0.0)
    m.init(Res, 10.0)
    sol = m.solve((0, 5), times=[5])
    np.testing.assert_allclose(sol["Res"][-1] + sol.ledger.amount["C"][-1], 10.0, rtol=1e-10)
    assert sol.ledger.amount["C"][-1] > 0.5


@pytest.mark.parametrize(
    "kind, extra, message",
    [
        ("zero_flux", {}, "dirichlet and robin"),
        ("dirichlet", {"value": 1.0}, "not a compartment|compartment"),
    ],
)
def test_routing_a_boundary_is_checked(kind, extra, message):
    m = tq.Model("bad")
    z = m.axis("z", 0, 1, 5, kind="spatial")
    C = m.field("C", axes=[z])
    other = m.field("O", axes=[z])
    target = m.compartment("W") if kind == "zero_flux" else other
    with pytest.raises(ValueError, match=message):
        m.boundary(C, axis=z, at="lower", kind=kind, to=target, **extra)


def _robin_switched_on(diffusion: bool, n: int):
    m = tq.Model("robin_on")
    x = m.axis("x", 0, 1, n)
    N, C = m.field("N", axes=[x]), m.compartment("C", init=0.0)
    m.params(k=2.0)
    if diffusion:
        m.diffuse(N, axis=x, D=0.1)
    m.boundary(N, axis=x, at="upper", kind="robin", value=1.0, h="k*C")
    m.ode(C, "-0.1*C")
    m.init(N, 0.0)
    m.dose(C, times=[1.0], amounts=[1.0])
    return m


def test_robin_coefficient_reaching_zero():
    """h = k C with C = 0 before a dose: the exchange is closed, and its derivative must stay
    finite (1/h used to give 0 * inf = NaN in the Jacobian). One cell without diffusion:
    dN/dt = k C (1 - N), so N = 1 - exp(-k (1 - e^{-0.1 (t - 1)}) / 0.1) after the dose."""
    t = np.array([1.0, 2.0, 4.0])
    sol = _robin_switched_on(False, 1).solve((0, 4), times=t, **TIGHT)
    expect = 1 - np.exp(-2.0 * (1 - np.exp(-0.1 * (t - 1))) / 0.1)
    np.testing.assert_allclose(sol["N"][:, 0], expect, rtol=1e-8, atol=1e-14)
    cm = _robin_switched_on(True, 5).compile()
    y = np.zeros(cm.n_states)
    J = cm.sparse_jacobian(0.0, y).toarray()
    assert np.all(np.isfinite(J))
    np.testing.assert_allclose(J, np.asarray(cm.jacobian(0.0, y)), atol=1e-14)
    assert _robin_switched_on(True, 5).solve((0, 3), times=[3]).messages == []
