# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""A tumour cell population on a continuous drug-resistance level, with selection, phenotypic
diffusion and a drug-induced drift (`tenqsp.examples.drug_tolerance_model`; Lorenzi,
Chisholm & Clairambault, Biol Direct 2016, plus the drift term of Chisholm, Lorenzi & Lorz,
Commun Math Sci 2016).

The oracles come from the mathematics. The fitness gamma - eta*x^2 - c*(x - 1)^2 - d*rho is
quadratic in x, the drift v = vbar*c does not depend on x and the diffusion beta is
constant, so on the whole line a Gaussian start stays Gaussian. Writing n = exp(-A x^2 +
B x + const) and matching powers of x gives, with eta_c = eta + c, x_c = c/eta_c and
gamma_c = gamma - eta*c/eta_c (the fitness is gamma_c - eta_c*(x - x_c)^2 - d*rho):

    variance  s' = 2*beta - 2*eta_c*s^2
    mean      m' = v + 2*eta_c*s*(x_c - m)
    size      rho' = rho*(gamma_c - eta_c*((m - x_c)^2 + s) - d*rho)

At a constant drug level the equilibrium is s = sqrt(beta/eta_c), m = x_c + v/(2 sqrt(beta
eta_c)), rho = (gamma_c - sqrt(beta*eta_c) - v^2/(4*beta))/d. With v = 0 these are Theorems
1-2 of Lorenzi et al. 2016 (published); the drift terms are derived here. The model is
solved on [-4, 4] with sealed ends, where the Gaussian's tails are below 1e-12 of its peak.
The profile is compared with the exact Gaussian averaged over each cell (a start
`normal(x, ...)` is exactly that), so the difference is the grid's error alone: second order
in the cell width for the `sharp` scheme with diffusion.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import ndtr

from tenqsp.examples import drug_tolerance_model

GAMMA, ETA, BETA, K, VBAR = 0.66, 0.132, 0.01, 1e8, 0.02
D = GAMMA / K
TIGHT = {"rtol": 1e-11, "atol": 1e-4}  # cells per unit of x: up to ~1e8


def _moment_rhs(vbar, c):
    def rhs(_, y):
        rho, m, s = y
        eta_c, x_c = ETA + c, c / (ETA + c)
        gamma_c = GAMMA - ETA * c / (ETA + c)
        return [
            rho * (gamma_c - eta_c * ((m - x_c) ** 2 + s) - D * rho),
            vbar * c + 2 * eta_c * s * (x_c - m),
            2 * BETA - 2 * eta_c * s * s,
        ]

    return rhs


def _moments(times, y0, *, vbar=VBAR, c0=0.0, ke=0.0, doses=()):
    """(rho, mean, variance) at `times` from the three ODEs, with the drug c(t) = its level
    after the last dose times exp(-ke (t - that dose)). Solved piecewise between doses."""
    times = np.asarray(times, float)
    edges = sorted({0.0, *[t for t, _ in doses if t > 0], float(times[-1])})
    out = np.empty((3, times.size))
    y, c = np.asarray(y0, float), c0 + sum(a for t, a in doses if t == 0)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (times >= lo) & (times <= hi)

        def rhs(t, yy, lo=lo, c=c):
            return _moment_rhs(vbar, c * np.exp(-ke * (t - lo)))(t, yy)

        sol = solve_ivp(rhs, (lo, hi), y, method="Radau", rtol=1e-12, atol=1e-12, dense_output=True)
        out[:, sel] = sol.sol(times[sel])
        y = sol.y[:, -1]
        c = c * np.exp(-ke * (hi - lo)) + sum(a for t, a in doses if t == hi)
    return out


def _cells(n, rho, m, s, lower=-4.0, upper=4.0):
    """The Gaussian rho*N(m, s) averaged over each of the n cells."""
    f = np.linspace(lower, upper, n + 1)
    return rho * np.diff(ndtr((f - m) / np.sqrt(s))) / np.diff(f)


def _equilibrium(c, vbar):
    eta_c = ETA + c
    v = vbar * c
    rho = (GAMMA - ETA * c / eta_c - np.sqrt(BETA * eta_c) - v**2 / (4 * BETA)) / D
    return rho, c / eta_c + v / (2 * np.sqrt(BETA * eta_c)), np.sqrt(BETA / eta_c)


def _profile_error(sol, n, rho, m, s, row=-1):
    exact = _cells(n, rho, m, s)
    return np.max(np.abs(np.asarray(sol["n"])[row] - exact)) / exact.max()


def test_drug_free_equilibrium_is_the_published_gaussian():
    """No drug (Lorenzi et al. 2016, Theorem 1): the population settles to a Gaussian at the
    most proliferative level x = 0, variance sqrt(beta/eta), size (gamma - sqrt(beta*eta))/d,
    whatever it started from."""
    n = 320
    sol = drug_tolerance_model(n).solve(
        (0, 300), times=[0, 300], params={"x0": 0.4, "s0": 0.3}, **TIGHT
    )
    rho, m, s = _equilibrium(0.0, VBAR)
    assert abs(m) < 1e-15 and abs(rho - (GAMMA - np.sqrt(BETA * ETA)) / D) < 1e-6
    size, mean = float(np.asarray(sol["size"])[-1]), float(np.asarray(sol["mean_x"])[-1])
    assert abs(size / rho - 1) < 2e-4, (size, rho)
    assert abs(mean) < 1e-4, mean
    assert _profile_error(sol, n, rho, m, s) < 1e-3


@pytest.mark.parametrize("vbar", [0.0, VBAR])
def test_constant_drug_equilibrium(vbar):
    """A constant drug level c = 1 per day. Without drift (vbar = 0) the equilibrium is
    Theorem 2 of Lorenzi et al. 2016: mean c/(eta + c) = 0.883. With the drug-induced drift
    the mean moves up by vbar*c/(2 sqrt(beta (eta + c))) and the size falls by
    (vbar c)^2/(4 beta d); the variance does not change."""
    n, c = 320, 1.0
    sol = drug_tolerance_model(n).solve(
        (0, 300), times=[0, 300], params={"c0": c, "vbar": vbar}, **TIGHT
    )
    rho, m, s = _equilibrium(c, vbar)
    size, mean = float(np.asarray(sol["size"])[-1]), float(np.asarray(sol["mean_x"])[-1])
    assert abs(size / rho - 1) < 2e-4, (size, rho)
    assert abs(mean - m) < 3e-4, (mean, m)  # the grid: 1e-4 with the drift, 1e-5 without
    assert _profile_error(sol, n, rho, m, s) < 1e-3
    if vbar:
        no_drift = _equilibrium(c, 0.0)
        assert m - no_drift[1] > 0.09 and rho < no_drift[0]  # the effect is not negligible


def test_gaussian_at_all_times_under_weekly_doses():
    """Weekly doses into C, eliminated with half-life ln 2/0.5 days: selection and drift
    follow the drug level. The size, mean and the whole profile follow the three moment
    ODEs (solved independently with scipy) at every time, through the bottleneck and the
    regrowth."""
    n = 320
    m = drug_tolerance_model(n)
    m.dose("C", times=[0.0, 7.0, 14.0, 21.0], amounts=[2.0] * 4)
    t = np.linspace(0, 35, 71)
    sol = m.solve((0, 35), times=t, params={"ke": 0.5}, **TIGHT)
    doses = [(day, 2.0) for day in (0.0, 7.0, 14.0, 21.0)]
    rho, mean, s = _moments(t, [1e6, 0.0, 0.25], ke=0.5, doses=doses)
    size = np.asarray(sol["size"])
    assert np.max(np.abs(size / rho - 1)) < 1e-3
    assert np.max(np.abs(np.asarray(sol["mean_x"]) - mean)) < 2e-4  # observed 1e-4 (grid)
    assert min(size) < 0.6e6 and size[-1] > 1e7  # a real bottleneck, then regrowth
    errs = [_profile_error(sol, n, rho[i], mean[i], s[i], row=i) for i in (10, 30, 50, 70)]
    assert max(errs) < 2e-3, errs
    assert max(sol.ledger.check().values()) < 1e-10


def test_without_trait_dependence_the_size_is_logistic():
    """eta = 0 and no drug: every level grows at gamma - d*rho, so the size is the logistic
    curve K/(1 + (K/rho0 - 1) exp(-gamma t)) exactly, whatever the profile does (here it
    spreads by diffusion from an off-centre start). No grid error enters."""
    t = np.linspace(0, 30, 31)
    sol = drug_tolerance_model(80).solve(
        (0, 30), times=t, params={"eta": 0.0, "x0": 1.5, "s0": 0.2, "beta": 0.05},
        rtol=1e-12, atol=1e-6,
    )  # fmt: skip
    exact = K / (1 + (K / 1e6 - 1) * np.exp(-GAMMA * t))
    assert np.max(np.abs(np.asarray(sol["size"]) / exact - 1)) < 1e-8
    assert np.asarray(sol["var_x"])[-1] > 0.04 + 2 * 0.05 * 30 * 0.5  # it did spread


def test_the_books_close_under_drug_and_drift():
    """Every cell born, killed by the drug or lost to competition is in the ledger; drift
    and diffusion move cells without creating or losing any."""
    sol = drug_tolerance_model(160).solve(
        (0, 40), times=np.linspace(0, 40, 9), params={"c0": 1.5, "ke": 0.1, "vbar": 0.05},
        rtol=1e-10, atol=1e-4,
    )  # fmt: skip
    assert max(sol.ledger.check().values()) < 1e-10


def test_a_bounded_trait_with_sealed_ends_conserves_cells_while_they_drift():
    """On [0, 1] with sealed ends (the setting of Chisholm et al. 2015) and no births or
    deaths, the drift piles the cells against the resistant end and the total stays put
    exactly (no Gaussian here: the ends matter). The no-flux equilibrium is
    proportional to exp(v x/beta). The drift v = vbar*C is set with C = 1e-12 and
    vbar = v/1e-12, so the drug's kill (at most 1e-12 per day) removes < 4e-10 of the cells."""
    n, v = 100, 0.05
    sol = drug_tolerance_model(n, lower=0.0, upper=1.0).solve(
        (0, 400), times=[0, 400],
        params={"gamma": 0.0, "eta": 0.0, "c0": 1e-12, "vbar": v / 1e-12, "x0": 0.3, "s0": 0.1},
        rtol=1e-11, atol=1e-10,
    )  # fmt: skip
    size = np.asarray(sol["size"])
    assert abs(size[-1] / size[0] - 1) < 1e-9
    f = np.linspace(0, 1, n + 1)
    a = v / BETA
    exact = size[0] * np.diff(np.exp(a * f)) / (np.exp(a) - 1) / np.diff(f)
    err = np.max(np.abs(np.asarray(sol["n"])[-1] - exact)) / exact.max()
    assert err < 2e-3, err


@pytest.mark.slow
def test_grid_convergence_is_second_order():
    """The profile and the size at day 10 under a constant drug, on 160, 320 and 640 cells:
    errors fall by about 4 per halving of the cell width (observed orders 1.8-1.9; from 80
    cells, still before the asymptotic range, 1.6)."""
    T = 10.0
    rho, m, s = _moments([0.0, T], [1e6, 0.0, 0.25], c0=1.0)[:, -1]
    prof, size = [], []
    for n in (160, 320, 640):
        sol = drug_tolerance_model(n).solve((0, T), times=[0, T], params={"c0": 1.0}, **TIGHT)
        prof.append(_profile_error(sol, n, rho, m, s))
        size.append(abs(float(np.asarray(sol["size"])[-1]) / rho - 1))
    p_order = np.log2(np.array(prof[:-1]) / np.array(prof[1:]))
    s_order = np.log2(np.array(size[:-1]) / np.array(size[1:]))
    assert np.all((p_order > 1.75) & (p_order < 2.25)), (prof, p_order)
    assert np.all((s_order > 1.75) & (s_order < 2.25)), (size, s_order)
    assert prof[-1] < 3e-4
