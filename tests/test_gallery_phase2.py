# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Gallery regression tests for Phase 2: V18-V25, V30, V31, V39, V40.

Each test builds a model from the pharmacometric and mathematical-biology literature with the
Python API and checks it against an oracle from the mathematics: closed forms, the method of
characteristics, the Euler-Lotka root, the published ODE form written out by hand, or moments
of the transfer function. Parameter values are illustrative unless stated otherwise.
Equation checks against the original papers (2026-10-19) are recorded in each model's
docstring: what was read (original or secondary), notation differences, and what remains
unverified. The paper wins; nothing here contradicts a paper that was read.
"""

from __future__ import annotations

import numpy as np
import pytest
from convergence import assert_converges, observed_orders
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import gammainc, gammaln

import tenqsp as tq

SLOW = pytest.mark.slow
TIGHT = {"rtol": 1e-10, "atol": 1e-12}


def _cell_averages(f, faces, nodes=8):
    x, w = np.polynomial.legendre.leggauss(nodes)
    lo, hi = faces[:-1, None], faces[1:, None]
    return (f(0.5 * (lo + hi) + 0.5 * (hi - lo) * x) * w).sum(axis=1) / 2


# ======================================================================= Model 1: McKendrick
def mckendrick(n, *, beta=None, hazard="mu", A=10.0, scheme="erlang", init=None, **params):
    """Gallery Model 1 (McKendrick 1926): dn/dt + dn/da = -mu(a) n, n(0, t) = int beta n da.
    ``beta`` is a formula of the age ``a`` (the birth rate at age a)."""
    m = tq.Model("mckendrick")
    a = m.axis("a", 0, A, n, kind="one_way")
    u = m.field("n", axes=[a])
    m.params(**params)
    m.advect(u, axis=a, velocity=1.0, scheme=scheme)
    births = m.integral(f"({beta})*n", over=a, name="births") if beta else 0.0
    m.inflow(u, axis=a, at="lower", flux=births)
    m.outflow(u, axis=a, at="upper")
    m.hazard(u, hazard)
    m.init(u, init or "exp(-((a - 1)/0.25)^2)")
    m.observable("N", m.integral(u, over=a, name="total"))
    return m, a


@pytest.mark.parametrize("scheme", ["erlang", "sharp"])
def test_v18_no_births_total_decays_exponentially(scheme):
    """beta = 0, mu constant: N(t) = N0 exp(-mu t) while nothing reaches the end of the axis."""
    mu = 0.5
    m, _ = mckendrick(100, scheme=scheme, mu=mu)
    t = np.linspace(0, 3, 7)
    sol = m.solve((0, 3), times=t, **TIGHT)
    np.testing.assert_allclose(sol["N"], sol["N"][0] * np.exp(-mu * t), rtol=1e-9)
    assert abs(sol.ledger.moved["n"]["boundary a=upper: outflow (absorbed)"][-1]) < 1e-15


@pytest.mark.parametrize("scheme", ["erlang", "sharp"])
def test_v19_constant_births_and_deaths_give_exponential_growth(scheme):
    """beta, mu constant: dN/dt = (beta - mu) N exactly; the ledger shows births in, deaths
    out and nothing through the far end."""
    b, mu = 0.8, 0.5
    m, _ = mckendrick(100, beta="b", scheme=scheme, b=b, mu=mu)
    t = np.linspace(0, 3, 7)
    sol = m.solve((0, 3), times=t, **TIGHT)
    np.testing.assert_allclose(sol["N"], sol["N"][0] * np.exp((b - mu) * t), rtol=1e-9)
    moved = sol.ledger.moved["n"]
    np.testing.assert_allclose(
        moved["boundary a=lower: inflow"] + moved["hazard: mu"],
        sol["N"] - sol["N"][0],
        rtol=1e-9,
        atol=1e-12,
    )
    assert abs(moved["boundary a=upper: outflow (absorbed)"][-1]) < 1e-15
    assert max(sol.ledger.check().values()) < 1e-10


@pytest.mark.parametrize(
    ("scheme", "order", "grids"),
    [("erlang", 1.0, (120, 240, 480)), ("sharp", 2.0, (60, 120, 240, 480))],
)
def test_v20_profile_follows_the_characteristics(scheme, order, grids):
    """beta = 0, mu(a) = mu0 (1 + a): n(a, t) = n0(a - t) S(a) / S(a - t),
    S(a) = exp(-mu0 (a + a^2/2)). The L1 error of the cell averages falls at the scheme's
    order (erlang 1, sharp 2; within 0.3)."""
    mu0, t_end = 0.4, 1.5

    def S(x):
        return np.exp(-mu0 * (x + x * x / 2))

    def exact(x):
        return np.exp(-((((x - t_end) - 1.5) / 0.3) ** 2)) * S(x) / S(x - t_end)

    errors = []
    for n in grids:
        m, a = mckendrick(
            n, hazard="mu0*(1 + a)", A=6.0, scheme=scheme, init="exp(-((a - 1.5)/0.3)^2)", mu0=mu0
        )
        sol = m.solve((0, t_end), times=[t_end], ledger=False, rtol=1e-11, atol=1e-13)
        f = np.asarray(a.faces)
        errors.append(np.sum(np.abs(sol["n"][-1] - _cell_averages(exact, f)) * np.diff(f)))
    orders = observed_orders(errors, grids)
    assert abs(orders[-1] - order) <= 0.3, (errors, orders)


def _euler_lotka_window(b0, mu, a1, a2):
    """r with b0 int_a1^a2 exp(-(mu + r) a) da = 1 (births at rate b0 between ages a1, a2)."""
    return brentq(
        lambda r: b0 * (np.exp(-(mu + r) * a1) - np.exp(-(mu + r) * a2)) / (mu + r) - 1,
        -mu + 1e-9,
        20.0,
    )


@pytest.mark.parametrize(
    ("scheme", "n", "tol"), [("sharp", 100, 1e-3), pytest.param("erlang", 400, 1e-2, marks=SLOW)]
)
def test_v21_stable_age_profile_and_growth_rate_from_euler_lotka(scheme, n, tol):
    """beta(a) = b0 on [a1, a2], mu constant: after a long time the normalised age profile is
    exp(-(r + mu) a) and N grows at r, the Euler-Lotka root."""
    b0, mu, a1, a2 = 2.5, 0.3, 1.0, 2.0
    r = _euler_lotka_window(b0, mu, a1, a2)
    m, a = mckendrick(
        n,
        beta="b0 if a > a1 and a < a2 else 0",
        A=5.0,
        scheme=scheme,
        init="1",
        b0=b0,
        mu=mu,
        a1=a1,
        a2=a2,
    )
    sol = m.solve((0, 80), times=[78, 80], ledger=False, **TIGHT)
    f = np.asarray(a.faces)
    w = np.diff(f)
    prof = sol["n"][-1] / np.sum(sol["n"][-1] * w)
    exact = -np.diff(np.exp(-(mu + r) * f)) / (mu + r) / w
    exact /= np.sum(exact * w)
    assert np.sum(np.abs(prof - exact) * w) < tol
    r_sim = np.log(sol["N"][1] / sol["N"][0]) / 2
    assert abs(r_sim / r - 1) < tol


# ================================================================ Model 2: transit (Savic)
def savic(N, *, mtt=2.0, ka=1.2, cl_v=0.3, dose=100.0):
    """Gallery Model 2 (Savic et al. 2007): the dose starts in the first of N = n + 1 cells of
    a maturation axis z in [0, 1] moving at 1/MTT; the outflow feeds the absorption
    compartment Aa, which feeds the central compartment Ac.

    Paper check: 2007 full text not accessible; the same authors' PAGE 2004 poster gives
    k_tr = (n + 1)/MTT and the input Dose k_tr (k_tr t)^n exp(-k_tr t)/n!, as here. Savic used
    Stirling's n! ~ sqrt(2 pi) n^(n + 1/2) e^-n for non-integer n; the tests use the exact
    Gamma(n + 1) (gammaln), so they check the chain, not the approximation."""
    m = tq.Model("savic")
    z = m.axis("z", 0, 1, N, kind="one_way")
    u = m.field("u", axes=[z])
    Aa, Ac = m.compartment("Aa", init=0.0), m.compartment("Ac", init=0.0)
    m.params(mtt=mtt, ka=ka, cl_v=cl_v)
    m.advect(u, axis=z, velocity="1/mtt")
    m.inflow(u, axis=z, at="lower", flux=0.0)
    m.outflow(u, axis=z, at="upper", to=Aa)
    m.ode(Aa, "-ka*Aa")
    m.ode(Ac, "ka*Aa - cl_v*Ac")
    m.init(u, total=dose, shape=tq.point(z, at=0.5 / N))
    return m, z


def savic_odes(n, mtt, ka, cl_v):
    """The published ODE form: A_0 (dose) -> A_1..A_n -> Aa -> Ac, k_tr = (n + 1)/MTT."""
    ktr = (n + 1) / mtt

    def rhs(_, x):
        A, Aa, Ac = x[: n + 1], x[n + 1], x[n + 2]
        into = np.concatenate([[0.0], ktr * A[:-1]])
        return np.concatenate([into - ktr * A, [ktr * A[-1] - ka * Aa, ka * Aa - cl_v * Ac]])

    return rhs


@pytest.mark.parametrize("n", [0, 3, 7])
def test_v22_erlang_cells_are_the_savic_transit_chain(n):
    """N = n + 1 erlang cells = the transit chain with k_tr = (n + 1)/MTT (cell amount =
    density x width), to solver tolerance."""
    mtt, ka, cl_v, dose = 2.0, 1.2, 0.3, 100.0
    m, z = savic(n + 1, mtt=mtt, ka=ka, cl_v=cl_v, dose=dose)
    t = np.linspace(0, 12, 25)
    sol = m.solve((0, 12), times=t, rtol=1e-12, atol=1e-12, ledger=False)
    y0 = np.zeros(n + 3)
    y0[0] = dose
    ref = solve_ivp(
        savic_odes(n, mtt, ka, cl_v), (0, 12), y0, t_eval=t, method="Radau", rtol=1e-13, atol=1e-13
    ).y
    np.testing.assert_allclose(sol["u"] * np.asarray(z.widths), ref[: n + 1].T, atol=1e-9)
    np.testing.assert_allclose(sol["Aa"], ref[n + 1], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(sol["Ac"], ref[n + 2], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("n", [2, 5])
def test_v23_outflow_is_the_erlang_input(n):
    """Savic's analytic input: the outflow into Aa is Dose k_tr (k_tr t)^n exp(-k_tr t)/n!.
    Checked as a rate (the upwind flux v u_N at the end) and, with ka = 0, as its integral
    Aa(t) = Dose P(n + 1, k_tr t) (regularised incomplete gamma)."""
    mtt, dose = 2.0, 100.0
    ktr = (n + 1) / mtt
    m, _ = savic(n + 1, mtt=mtt, ka=0.0, dose=dose)
    t = np.linspace(0.25, 8, 32)
    sol = m.solve((0, 8), times=t, rtol=1e-12, atol=1e-12, ledger=False)
    erlang = dose * ktr * np.exp(n * np.log(ktr * t) - ktr * t - gammaln(n + 1))
    np.testing.assert_allclose(sol["u"][:, -1] / mtt, erlang, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sol["Aa"], dose * gammainc(n + 1, ktr * t), rtol=1e-9, atol=1e-10)


# ======================================================================= Model 3: Friberg
FRIBERG = {"mtt": 100.0, "gamma": 0.17, "circ0": 5.0, "slope": 0.5, "ke": 0.1}


def friberg(*, dose=None, **p):
    """Gallery Model 3 (Friberg et al. 2002): Prol and Circ are compartments, the three transit
    compartments are 3 erlang cells of a maturation axis z in [0, 1] at speed k_tr/3
    (k_tr = 4/MTT, k_prol = k_tr = k_circ), E_drug = slope x Conc, Conc a one-compartment
    drug. Illustrative values; MTT and gamma within the published ranges.

    Paper check: the abstract confirms the structure (proliferating compartment, three
    transit compartments, circulating compartment; baseline, MTT and gamma estimated; linear
    or Emax drug effect); the equations as written match an implementation derived from the
    paper (DTAT vignette, k_tr = 4/MTT, all compartments starting at Circ0). The full text
    (per-drug Table 4) was not accessible, so no published per-drug values are used."""
    p = {**FRIBERG, **p}
    m = tq.Model("friberg")
    z = m.axis("z", 0, 1, 3, kind="one_way")
    T = m.field("Transit", axes=[z])
    Prol = m.compartment("Prol", init=p["circ0"])
    Circ = m.compartment("Circ", init=p["circ0"])
    Conc = m.compartment("Conc", init=0.0)
    m.params(**p)
    m.define("ktr", "4/mtt")
    m.advect(T, axis=z, velocity="ktr/3")
    m.inflow(T, axis=z, at="lower", flux="ktr*Prol")
    m.outflow(T, axis=z, at="upper", to=Circ)
    m.ode(Prol, "ktr*Prol*(1 - slope*Conc)*(circ0/Circ)^gamma - ktr*Prol")
    m.ode(Circ, "-ktr*Circ")
    m.ode(Conc, "-ke*Conc")
    m.init(T, 3 * p["circ0"])  # density x width 1/3 = circ0 in each transit compartment
    if dose is not None:
        m.dose(Conc, times=[0.0], amounts=[dose])
    return m


def friberg_odes(mtt, gamma, circ0, slope, ke):
    ktr = 4 / mtt

    def rhs(_, x):
        prol, t1, t2, t3, circ, conc = x
        return [
            ktr * prol * (1 - slope * conc) * (circ0 / circ) ** gamma - ktr * prol,
            ktr * (prol - t1),
            ktr * (t1 - t2),
            ktr * (t2 - t3),
            ktr * t3 - ktr * circ,
            -ke * conc,
        ]

    return rhs


def test_v24_friberg_equals_the_published_ode_model():
    """3 erlang cells = the published 5-ODE model (plus the drug), through the nadir and the
    rebound, to solver tolerance."""
    t = np.linspace(0, 600, 61)
    sol = friberg(dose=1.8).solve((0, 600), times=t, rtol=1e-12, atol=1e-12, ledger=False)
    c0 = FRIBERG["circ0"]
    ref = solve_ivp(
        friberg_odes(**FRIBERG),
        (0, 600),
        [c0, c0, c0, c0, c0, 1.8],
        t_eval=t,
        method="Radau",
        rtol=1e-13,
        atol=1e-13,
    ).y
    np.testing.assert_allclose(sol["Circ"], ref[4], rtol=1e-9)
    np.testing.assert_allclose(sol["Prol"], ref[0], rtol=1e-9)
    np.testing.assert_allclose(sol["Transit"] / 3, ref[1:4].T, rtol=1e-9)
    assert sol["Circ"].min() < 0.8 * c0  # a real nadir, not a trivial run


def test_v24_friberg_steady_state_is_preserved():
    """No drug: every compartment at Circ0 is a steady state (RHS residual < 1e-12) and stays
    there over the run."""
    m = friberg()
    cm = m.compile()
    y0 = np.asarray(cm.initial_state())
    assert np.max(np.abs(np.asarray(cm.rhs(0.0, y0)))) < 1e-12
    sol = m.solve((0, 500), times=[100, 500], ledger=False, **TIGHT)
    np.testing.assert_allclose(sol["Circ"], FRIBERG["circ0"], rtol=1e-12)


# ===================================================================== Model 4: lifespan
@pytest.mark.parametrize("scheme", ["erlang", "sharp"])
def test_v25_lifespan_step_response_ramp_then_plateau(scheme):
    """Krzyzanski et al. 1999 (dR/dt = k_in(t) - k_in(t - T_R), R0 = k_in0 T_R, confirmed from
    the first author's PAGE 2010 poster): cells live exactly T_R; a step k_in0 -> k_in1 at
    t = 0 gives R(t) = k_in0 T_R + (k_in1 - k_in0) min(t, T_R). The kinks at 0 and T_R are
    smeared by the scheme: the maximum error falls with an order below 1 (documented, V25;
    about 0.5 for erlang, 0.7 for sharp), sharp beats erlang and the smeared region narrows."""
    kin0, kin1, TR = 2.0, 5.0, 4.0
    t = np.linspace(0, 8, 401)
    exact = kin0 * TR + (kin1 - kin0) * np.minimum(t, TR)
    errors, widths = [], []
    for n in (40, 80, 160):
        m = tq.Model("lifespan")
        a = m.axis("a", 0, TR, n, kind="one_way")
        u = m.field("u", axes=[a])
        m.params(kin1=kin1)
        m.advect(u, axis=a, velocity=1.0, scheme=scheme)
        m.inflow(u, axis=a, at="lower", flux="kin1")
        m.outflow(u, axis=a, at="upper")
        m.init(u, kin0)
        m.observable("R", m.integral(u, over=a, name="cells"))
        err = np.abs(m.solve((0, 8), times=t, ledger=False, **TIGHT)["R"] - exact)
        errors.append(err.max())
        smeared = t[err > 2e-3 * (kin1 - kin0) * TR]
        widths.append(np.ptp(smeared) if smeared.size else 0.0)
    orders = observed_orders(errors, [40, 80, 160])
    assert np.all((orders > 0.35) & (orders < 1.0)), (errors, orders)
    assert widths[0] > widths[1] > widths[2], widths
    if scheme == "sharp":
        assert errors[-1] < 0.3 * 0.38  # erlang's error at n = 160 is about 0.38


# ==================================================================== Model 8: HIV age model
HIV = {"s": 1e4, "d": 0.01, "k": 2.4e-8, "delta": 1.0, "p": 5000.0, "c": 23.0}


def hiv_age_model(
    n=40, A=40.0, *, eps_rt=0.3, eps_pi=0.2, production="p", death="delta", scheme="erlang"
):
    """Gallery Model 8 (Nelson et al. 2004; Rong et al. 2007, combination therapy): T and V
    compartments, infected cells i(a, t) by infection age; new infections (1 - eps_RT) k V T
    enter at age 0, virions are produced at (1 - eps_PI) int p(a) i da. Illustrative values
    in the range of Perelson-type target-cell models (per mL and day).

    Paper check: Nelson et al. 2004 read in full. The untreated model matches (their T*, P(a),
    Pmax, beta are our i, p(a), p*, theta; P(a) = Pmax (1 - exp(-beta (a - a1))) for a >= a1).
    Their printed closed form of the burst size (Eqs 21, 26) agrees with integrating Eq 3 only
    at a1 = 0; the V31 oracle integrates Eq 3 directly. The drug factors follow the gallery's
    standard (non-age-structured) form. Rong et al. 2007 may model the RT inhibitor
    differently (cells reverting during the pre-RT age; seen in the authors' 2006 precursor,
    the 2007 full text was not accessible): open, these tests do not claim Rong's form."""
    m = tq.Model("hiv_age")
    a = m.axis("a", 0, A, n, kind="one_way", units="day")
    i = m.field("i", axes=[a], units="cells/mL/day")
    T = m.compartment("T", init=HIV["s"] / HIV["d"], units="cells/mL")
    V = m.compartment("V", init=1e-3, units="copies/mL")
    m.params(**HIV, eps_rt=eps_rt, eps_pi=eps_pi, pstar=1000.0, theta=1.0, a1=0.5)
    m.advect(i, axis=a, velocity=1.0, scheme=scheme)
    m.inflow(i, axis=a, at="lower", flux="(1 - eps_rt)*k*V*T")
    m.outflow(i, axis=a, at="upper")
    m.hazard(i, death)
    m.ode(T, "s - d*T - (1 - eps_rt)*k*V*T")
    produced = m.integral(f"({production})*i", over=a, name="produced")
    m.ode(V, (1 - m["eps_pi"]) * produced - m["c"] * V)
    m.init(i, 0.0)
    m.observable("I", m.integral(i, over=a, name="infected"))
    return m, a


def test_v30_constant_death_and_production_is_the_three_ode_model():
    """delta(a) = delta, p(a) = p: I = int i da obeys dI/dt = (1 - eps_RT) k V T - delta I and
    the system is the standard target-cell-limited model (T, I, V), to solver tolerance."""
    eps_rt, eps_pi = 0.3, 0.2
    m, _ = hiv_age_model(60, A=60.0, eps_rt=eps_rt, eps_pi=eps_pi)  # nothing reaches a = 60
    t = np.linspace(0, 60, 31)
    # the virus starts at 1e-3 copies/mL and infected-cell densities are tiny early on, so
    # the absolute tolerance must sit far below them
    sol = m.solve((0, 60), times=t, rtol=1e-12, atol=1e-18)
    h = HIV

    def rhs(_, x):
        T, I, V = x
        new = (1 - eps_rt) * h["k"] * V * T
        return [
            h["s"] - h["d"] * T - new,
            new - h["delta"] * I,
            (1 - eps_pi) * h["p"] * I - h["c"] * V,
        ]

    ref = solve_ivp(
        rhs, (0, 60), [h["s"] / h["d"], 0.0, 1e-3], t_eval=t, method="Radau", rtol=1e-13, atol=1e-18
    ).y
    np.testing.assert_allclose(sol["T"], ref[0], rtol=1e-8)
    np.testing.assert_allclose(sol["I"], ref[1], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(sol["V"], ref[2], rtol=1e-8, atol=1e-12)
    assert sol["V"].max() > 1e3  # the infection takes off (R0 > 1)
    assert abs(sol.ledger.moved["i"]["boundary a=upper: outflow (absorbed)"][-1]) < 1e-12


def test_v31_r0_by_quadrature_on_the_grid():
    """The quadrature behind R0 (the model's dynamics are checked in the next test):
    R0 = (1 - eps_RT)(1 - eps_PI)(k T0 / c) int p(a) exp(-int_0^a delta) da with Nelson's
    production ramp p(a) = p* (1 - exp(-theta (a - a1))) after the eclipse a1, delta constant:
    int = p* exp(-delta a1) theta / (delta (delta + theta)). The grid's quadrature converges
    at order 2 (the eclipse kink on a cell face)."""
    h, eps_rt, eps_pi, pstar, theta, a1 = HIV, 0.3, 0.2, 1000.0, 1.0, 0.5
    T0 = h["s"] / h["d"]
    scale = (1 - eps_rt) * (1 - eps_pi) * h["k"] * T0 / h["c"]
    exact = scale * pstar * np.exp(-h["delta"] * a1) * theta / (h["delta"] * (h["delta"] + theta))

    def r0(n):
        m, a = hiv_age_model(n, A=20.0)
        q = m.integral("pstar*max(0, 1 - exp(-theta*(a - a1)))*exp(-delta*a)", over=a, name="q")
        m.observable("R0", f"(1 - eps_rt)*(1 - eps_pi)*k*(s/d)/c*{q}")
        return m.solve((0, 1e-6), times=[1e-6], ledger=False)["R0"][-1]

    report = assert_converges(r0, [40, 80, 160], 2.0, exact=exact, tol=0.2)
    assert report["errors"][-1] / exact < 2e-3


def test_v31_early_growth_rate_is_the_root_of_the_renewal_equation():
    """The model itself, with Nelson's production ramp: while target cells are still at T0
    the system is linear, i(a, t) = b(t - a) exp(-delta a), and V grows as exp(r t) with r
    the root of

        r + c = (1 - eps_RT)(1 - eps_PI) k T0 int p(a) exp(-(delta + r) a) da
              = (1 - eps_RT)(1 - eps_PI) k T0 p* exp(-(delta + r) a1) theta
                / ((delta + r)(delta + r + theta))

    (gallery check iii). The slope of log V over days 8-12 approaches r: `sharp` to 5e-4
    at 160 cells, `erlang` at first order (its transit chain lets cells reach production
    early)."""
    h, eps_rt, eps_pi, pstar, theta, a1 = HIV, 0.3, 0.2, 20000.0, 1.0, 0.5
    gain = (1 - eps_rt) * (1 - eps_pi) * h["k"] * h["s"] / h["d"]

    def balance(r):
        s = h["delta"] + r
        return gain * pstar * np.exp(-s * a1) * theta / (s * (s + theta)) - (r + h["c"])

    r = brentq(balance, -0.5, 20.0)
    ramp = "pstar*max(0, 1 - exp(-theta*(a - a1)))"

    def slope(n, scheme):
        m, _ = hiv_age_model(n, A=20.0, production=ramp, scheme=scheme)
        sol = m.solve(
            (0, 12), times=[8.0, 12.0], params={"pstar": pstar}, rtol=1e-10, atol=1e-16,
            ledger=False,
        )  # fmt: skip
        assert sol["T"][-1] > (1 - 1e-5) * h["s"] / h["d"]  # still the linear phase
        return np.log(sol["V"][1] / sol["V"][0]) / 4.0

    assert abs(slope(160, "sharp") / r - 1) < 1e-3
    assert_converges(lambda n: slope(n, "erlang"), [40, 80, 160], 1.0, exact=r, tol=0.3)


# ==================================================================== CAT (Yu & Amidon)
def cat(N, *, tsi=199 / 60, ka=0.6, scheme="erlang", dose=100.0):
    """Compartmental absorption and transit (Yu et al. 1996; Yu & Amidon 1999): the small
    intestine as N cells of z in [0, 1] moving at 1/T_si (T_si = 199 min), absorption ka from
    every cell into ``absorbed``, the end emptying into ``colon``. N = 7 is the CAT model.

    Paper check (abstracts): 7 compartments, T_si = 199 min; Yu & Amidon 1999 give
    F_a = 1 - (1 + 0.54 P_eff)^-7, which is 1 - (1 + K_a/K_t)^-7 with K_a = 2 P_eff/R,
    K_t = 7/T_si, R = 1.75 cm. The compartment ODEs were not read in full text."""
    m = tq.Model("cat")
    z = m.axis("z", 0, 1, N, kind="one_way")
    M = m.field("M", axes=[z], units="mg")
    absorbed = m.compartment("absorbed", init=0.0, units="mg")
    colon = m.compartment("colon", init=0.0, units="mg")
    m.params(tsi=tsi, ka=ka)
    m.advect(M, axis=z, velocity="1/tsi", scheme=scheme)
    m.inflow(M, axis=z, at="lower", flux=0.0)
    m.outflow(M, axis=z, at="upper", to=colon)
    m.hazard(M, "ka")
    m.ode(absorbed, f"ka*{m.integral(M, over=z, name='in_gut')}")
    m.init(M, total=dose, shape=tq.point(z, at=0.5 / N))
    return m


def test_v39_seven_erlang_cells_are_the_cat_model_and_give_its_fraction_absorbed():
    """7 erlang cells = dM_n/dt = K_t (M_{n-1} - M_n) - K_a M_n, K_t = 7/T_si, to solver
    tolerance; the fraction absorbed is F_a = 1 - (1 + K_a/K_t)^-7 and nothing is lost."""
    tsi, ka, dose = 199 / 60, 0.6, 100.0
    kt = 7 / tsi
    t = np.linspace(0, 40, 41)
    sol = cat(7, tsi=tsi, ka=ka, dose=dose).solve((0, 40), times=t, rtol=1e-12, atol=1e-12)

    def rhs(_, x):
        Mn = x[:7]
        into = np.concatenate([[0.0], kt * Mn[:-1]])
        return np.concatenate([into - (kt + ka) * Mn, [ka * Mn.sum(), kt * Mn[-1]]])

    y0 = np.zeros(9)
    y0[0] = dose
    ref = solve_ivp(rhs, (0, 40), y0, t_eval=t, method="Radau", rtol=1e-13, atol=1e-13).y
    np.testing.assert_allclose(sol["M"] / 7, ref[:7].T, atol=1e-9)
    np.testing.assert_allclose(sol["absorbed"], ref[7], rtol=1e-9, atol=1e-9)
    fa = 1 - (1 + ka / kt) ** -7
    assert sol["absorbed"][-1] / dose == pytest.approx(fa, rel=1e-9)
    total = sol["absorbed"] + sol["colon"] + sol["M"].sum(axis=1) / 7
    np.testing.assert_allclose(total, dose, rtol=1e-11)
    assert max(sol.ledger.check().values()) < 1e-10


def test_v39_more_cells_tend_to_plug_flow():
    """N erlang cells absorb exactly 1 - (1 + K_a T_si/N)^-N, which tends to the plug-flow
    value 1 - exp(-K_a T_si) (Ni et al. 1980; Amidon et al. 1988) as N grows; the sharp scheme
    gets closer on the same grid."""
    tsi, ka = 199 / 60, 0.6
    plug = 1 - np.exp(-ka * tsi)
    fa = {}
    for N in (7, 28):
        for scheme in ("erlang", "sharp"):
            sol = cat(N, tsi=tsi, ka=ka, scheme=scheme).solve(
                (0, 40), times=[40], ledger=False, **TIGHT
            )
            fa[scheme, N] = sol["absorbed"][-1] / 100.0
        assert fa["erlang", N] == pytest.approx(1 - (1 + ka * tsi / N) ** -N, rel=1e-9)
        assert abs(fa["sharp", N] - plug) < abs(fa["erlang", N] - plug)
    assert abs(fa["erlang", 28] - plug) < abs(fa["erlang", 7] - plug) / 3


# ========================================================== capillary-tissue (Sangren-Sheppard)
SS = {"L": 1.0, "v": 1.0, "V": 1.0, "Vt": 2.0, "PS": 5.0, "t0": 1.0, "sg": 0.1}


def capillary_tissue(n, scheme):
    """Sangren & Sheppard 1953: dC/dt + v dC/dx = -k1 (C - Ct), dCt/dt = k2 (C - Ct),
    k1 = PS/V, k2 = PS/Vt, plasma volume V and tissue volume Vt spread over the length L. A
    Gaussian pulse (mean t0, sd sg, unit amount) enters at x = 0. ``W1`` and ``W2`` integrate
    M(t) and 2 t M(t), M the amount inside: for a linear system int M dt and int 2 t M dt are
    the first two moments of the transit time times the dose (moment identity).

    Paper check: citation confirmed; the 1953 text was not accessible. A secondary source
    (JSim "Sanshe") describes it as this two-region convection-exchange model with PS, plasma
    and tissue volumes. The moments are derived here from the equations as written."""
    p = SS
    m = tq.Model("capillary_tissue")
    x = m.axis("x", 0, p["L"], n, kind="one_way")
    C, Ct = m.field("C", axes=[x]), m.field("Ct", axes=[x])
    W1, W2 = m.compartment("W1", init=0.0), m.compartment("W2", init=0.0)
    m.params(**p)
    m.advect(C, axis=x, velocity="v", scheme=scheme)
    m.inflow(C, axis=x, at="lower", flux="(L/V)*exp(-(t - t0)^2/(2*sg^2))/(sg*2.5066282746310002)")
    m.outflow(C, axis=x, at="upper")
    m.reaction(C, "-(PS/V)*(C - Ct)")
    m.reaction(Ct, "(PS/Vt)*(C - Ct)")
    inside = m.integral("(V*C + Vt*Ct)/L", over=x, name="inside")
    m.ode(W1, inside)
    m.ode(W2, f"2*t*{inside}")
    return m


def _transit_moments(n, scheme):
    sol = capillary_tissue(n, scheme).solve(
        (0, 80), times=[80], ledger=False, rtol=1e-10, atol=1e-13
    )
    t0, sg = SS["t0"], SS["sg"]
    mean = sol["W1"][-1]  # unit dose
    second = sol["W2"][-1] + t0**2 + sg**2  # E[t_out^2]
    return mean, second - (t0 + mean) ** 2 - sg**2  # mean and variance of the transit time


def test_v40_mean_transit_time_is_exact_with_erlang_cells():
    """Mean transit time (L/v)(1 + Vt/V) (gallery §5, from H(s) = exp(-(L/v)(s + k1 s/(s +
    k2)))). With erlang cells each cell is exactly a well-mixed capillary-tissue unit, so the
    mean is exact on every grid and the variance is the continuous value 2 (L/v) k1/k2^2 plus
    the cells' own mean^2/N (tanks in series)."""
    p = SS
    tau, k1, k2 = p["L"] / p["v"], p["PS"] / p["V"], p["PS"] / p["Vt"]
    mean_exact = tau * (1 + p["Vt"] / p["V"])
    for n in (10, 40):
        mean, var = _transit_moments(n, "erlang")
        assert mean == pytest.approx(mean_exact, rel=1e-8)
        assert var == pytest.approx(2 * tau * k1 / k2**2 + mean_exact**2 / n, rel=1e-7)


def test_v40_sharp_variance_converges_at_second_order():
    """With the sharp scheme the transit-time variance tends to the continuous Sangren-Sheppard
    value 2 (L/v) k1/k2^2 at second order, and the mean to (L/v)(1 + Vt/V)."""
    p = SS
    tau, k1, k2 = p["L"] / p["v"], p["PS"] / p["V"], p["PS"] / p["Vt"]
    var_exact = 2 * tau * k1 / k2**2
    report = assert_converges(
        lambda n: _transit_moments(n, "sharp")[1], [20, 40, 80], 2.0, exact=var_exact
    )
    assert report["errors"][-1] < 2e-3
    mean, _ = _transit_moments(80, "sharp")
    assert mean == pytest.approx(tau * (1 + p["Vt"] / p["V"]), rel=1e-4)
