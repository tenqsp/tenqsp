# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Gallery regression tests for Phase 5: V36, V37 (Model 10, plaque fronts) and
the physiological oral absorption model (Phase 5 plan §2; checks at the end of this file).

Model 10 (`tenqsp.examples.plaque_model`): phage spreading through a bacterial lawn, with
infected cells structured by infection age. The oracles come from linearizing at the
leading edge, where the lawn is untouched (B = B0) and phage are few (gallery §3, Model 10):

    growth rate (well mixed):  r = -k B0 - delta + beta k B0 f^(r)
    front speed c for a decay rate lambda:
        lambda c = D lambda^2 - k B0 - delta + beta k B0 f^(lambda c)
    c* = min over lambda of c(lambda);  without latency (f^ = 1) c* = 2 sqrt(D r)

with f^(s) = int f(a) e^(-s a) da the Laplace transform of the lysis-age density
f(a) = h(a) exp(-int_0^a h), computed here by quadrature on a fine grid. A start that decays
like exp(-lambda0 x) with lambda0 below the minimizing lambda* spreads at c(lambda0) (the
linear theory exactly); a steeper start gives c*, which the latent period's phage reservoir
behind the front can push higher (gallery caveat: 6-15 %), so that case asserts a range.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq, minimize_scalar

from tenqsp.examples import oral_absorption_model, plaque_model

SLOW = pytest.mark.slow
V = dict(D=400.0, k=1e-8, B0=1e9, beta=50.0, delta=0.05, h_max=30.0, tau=0.5, s=0.03)


def _f_hat(p, a_max=1.5):
    """Laplace transform of the lysis-age density, to the end of the age axis."""
    a = np.linspace(0.0, a_max, 30001)
    h = p["h_max"] / (1 + np.exp(-(a - p["tau"]) / p["s"]))
    f = h * np.exp(-cumulative_trapezoid(h, a, initial=0.0))
    return lambda z: np.trapezoid(f * np.exp(-z * a), a)


def _growth_rate(p):
    F, kB = _f_hat(p), p["k"] * p["B0"]
    return brentq(lambda r: r + kB + p["delta"] - p["beta"] * kB * F(r), 1e-6, 100.0)


def _speed(p, lam):
    F, kB = _f_hat(p), p["k"] * p["B0"]
    return brentq(
        lambda c: lam * c - p["D"] * lam**2 + kB + p["delta"] - p["beta"] * kB * F(lam * c),
        1e-9,
        1e6,
    )


def _min_speed(p):
    res = minimize_scalar(lambda lam: _speed(p, lam), bounds=(1e-4, 2.0), method="bounded")
    return res.fun, res.x


def _front_speed(sol, x, B0, t_from, x_to):
    """Speed of the point where the lawn has fallen to half, fitted from t_from until the
    front reaches x_to."""
    B = np.asarray(sol["B"]) / B0
    front = np.array([x[np.where(row < 0.5)[0].max()] if (row < 0.5).any() else 0 for row in B])
    sel = (sol.t >= t_from) & (front < x_to) & (front > 0)
    assert sel.sum() >= 10, front
    return np.polyfit(sol.t[sel], front[sel], 1)[0]


def _x(n, L=600.0):
    return (np.arange(n) + 0.5) * L / n


def test_v36_well_mixed_growth_rate_is_the_characteristic_root():
    """A uniform start on a closed domain is the well-mixed model; from very few phage the
    lawn stays untouched (depleted by < 1e-6), so the phage grow at the root r. The bursts
    start in step and take a few generations to spread out: the rate is fitted late."""
    r = _growth_rate(V)
    m = plaque_model(2, 80, L=10.0)
    t = np.linspace(0.0, 9.0, 361)
    sol = m.compile().solve(
        (0.0, 9.0), times=t, params={"P0": 1e-30, "lam0": 0.0}, rtol=1e-10, atol=1e-45
    )
    sel = t >= 7.0
    rate = np.polyfit(t[sel], np.log(np.asarray(sol["phage"])[sel]), 1)[0]
    assert abs(rate / r - 1) < 3e-3, (rate, r)
    B = np.asarray(sol["bacteria"])
    assert 1 - B[-1] / B[0] < 1e-6
    assert max(sol.ledger.check().values()) < 1e-9


@SLOW
def test_v37_without_latency_the_front_moves_at_two_sqrt_D_r():
    """No latent period (Yin & McCaskill limit): a pulled Fisher-KPP front, c* = 2 sqrt(D r),
    r = (beta - 1) k B0 - delta."""
    p = dict(V, beta=2.0, delta=5.0)
    c_star = 2 * np.sqrt(p["D"] * ((p["beta"] - 1) * p["k"] * p["B0"] - p["delta"]))
    cm = plaque_model(240, latency=False).compile()
    sol = cm.solve((0.0, 8.0), times=np.linspace(0, 8, 81), params=p, rtol=1e-6)
    c = _front_speed(sol, _x(240), p["B0"], 3.0, 500.0)
    assert abs(c / c_star - 1) < 0.03, (c, c_star)
    assert max(sol.ledger.check().values()) < 1e-9


@SLOW
def test_v37_a_shallow_start_spreads_at_the_linear_speed_of_its_decay_rate():
    """With latency, a start decaying like exp(-0.1 x) (below lambda* ~ 0.19 per um) sets the
    front speed to c(0.1) of the linear theory."""
    lam0 = 0.1
    assert lam0 < _min_speed(V)[1]
    expected = _speed(V, lam0)
    sol = (
        plaque_model(240, 40)
        .compile()
        .solve((0.0, 10.0), times=np.linspace(0, 10, 101), params={"lam0": lam0}, rtol=1e-6)
    )
    c = _front_speed(sol, _x(240), V["B0"], 4.0, 500.0)
    assert abs(c / expected - 1) < 0.01, (c, expected)
    assert max(sol.ledger.check().values()) < 1e-9


@SLOW
@pytest.mark.parametrize("delta", [0.05, 3.0])
def test_v37_with_latency_a_steep_start_is_at_least_the_minimal_speed(delta):
    """A steep start: speed >= c*, pushed above it by the phage behind the front (by 14 % at
    delta = 0.05 per h, 7 % at 3 per h here; the gap shrinks as phage decay faster)."""
    p = dict(V, delta=delta)
    c_star, _ = _min_speed(p)
    sol = (
        plaque_model(240, 40)
        .compile()
        .solve((0.0, 12.0), times=np.linspace(0, 12, 121), params={"delta": delta}, rtol=1e-6)
    )
    c = _front_speed(sol, _x(240), p["B0"], 4.0, 500.0)
    assert c_star * 0.97 <= c <= c_star * 1.2, (c, c_star)
    assert max(sol.ledger.check().values()) < 1e-9


# ============================================================ physiological oral absorption
# tenqsp.examples.oral_absorption_model. With no colon
# absorption, no faeces and no clearance, the body compartment counts exactly what the small
# intestine absorbed, so its end value / Dose is the fraction absorbed F_a, which has closed
# forms: plug flow 1 - exp(-int ka dt) along the transit; CAT (Yu & Amidon 1996) for 7
# well-stirred cells; the dispersion model with Danckwerts ends (as V26).
T_SI = 199 / 60
GUT_ONLY = {"ka_col": 0.0, "k_col": 0.0, "CL": 0.0}
TIGHT = {"rtol": 1e-9, "atol": 1e-11}


def _f_absorbed(m, params, t_end=40.0):
    sol = m.compile().solve((0, t_end), times=[0, t_end], params={**GUT_ONLY, **params}, **TIGHT)
    assert max(sol.ledger.check().values()) < 1e-10
    return float(np.asarray(sol["Ac"])[-1]) / 100.0


def test_weibull_gastric_emptying_follows_the_weibull_distribution():
    """The stomach holds Dose exp(-(t/eta)^beta) and empties at Dose x the Weibull density;
    first order (beta = 1) exactly, beta > 1 to first order in the cell width (the dose
    starts spread over the first cell)."""
    t = np.linspace(0, 6, 61)
    for (eta, b), n_g, tol in [((0.36, 1.0), 160, 1e-10), ((1.0, 1.5), 320, 4e-3)]:
        sol = (
            oral_absorption_model(n_g=n_g, n_x=20)
            .compile()
            .solve((0, 6), times=t, params={"eta": eta, "beta_ge": b}, **TIGHT)
        )
        survival = 100 * np.exp(-((t / eta) ** b))
        density = 100 * b / eta * (t / eta) ** (b - 1) * np.exp(-((t / eta) ** b))
        assert np.max(np.abs(np.asarray(sol["stomach"]) - survival)) < tol * 100
        late = t >= 0.3
        err = np.abs(np.asarray(sol["emptying"]) - density)[late]
        assert np.max(err) < tol * density.max(), (eta, b)


def test_plug_flow_fraction_absorbed():
    """Constant ka, no dispersion: F_a = 1 - exp(-ka T_si), converging as the grid refines."""
    exact = 1 - np.exp(-0.5 * T_SI)
    errs = [
        abs(_f_absorbed(oral_absorption_model(80, n), {"ka_ratio": 1.0, "ka_si": 0.5}) / exact - 1)
        for n in (100, 200)
    ]
    assert errs[1] < 5e-4 and errs[1] < 0.6 * errs[0], errs


def test_regional_absorption_falling_log_linearly_along_the_intestine():
    """ka(x) = ka_si r^x: F_a = 1 - exp(-ka_si T_si (r - 1) / ln r) (atenolol-like defaults)."""
    r, ka = 1 / 3, 0.185
    exact = 1 - np.exp(-ka * T_SI * (r - 1) / np.log(r))
    assert abs(_f_absorbed(oral_absorption_model(80, 200), {}) / exact - 1) < 1.5e-3


def test_seven_upwind_cells_are_the_cat_model():
    """The erlang scheme on 7 cells is the CAT model's 7 compartments, so F_a is exactly
    Yu & Amidon's 1 - (1 + ka T_si / 7)^-7."""
    m = oral_absorption_model(80, 7, scheme="erlang")
    exact = 1 - (1 + 0.5 * T_SI / 7) ** -7
    assert abs(_f_absorbed(m, {"ka_ratio": 1.0, "ka_si": 0.5}) / exact - 1) < 1e-8


@pytest.mark.parametrize("DN", [0.05, 0.3])
def test_dispersion_model_fraction_absorbed(DN):
    """Spreading D_x along the intestine (Ni et al. 1980), Danckwerts ends: F_a = 1 - the
    dispersion model's availability with D_N = D_x T_si and R_N = ka T_si."""
    a = np.sqrt(1 + 4 * (0.5 * T_SI) * DN)
    escapes = (
        4
        * a
        / ((1 + a) ** 2 * np.exp((a - 1) / (2 * DN)) - (1 - a) ** 2 * np.exp(-(a + 1) / (2 * DN)))
    )
    f = _f_absorbed(
        oral_absorption_model(80, 100), {"ka_ratio": 1.0, "ka_si": 0.5, "D_x": DN / T_SI}, 60.0
    )
    assert abs(f / (1 - escapes) - 1) < 2e-5


def test_the_dose_is_accounted_for():
    """Stomach + small intestine + colon + body = the dose when nothing leaves the body."""
    t = np.linspace(0, 48, 97)
    sol = (
        oral_absorption_model()
        .compile()
        .solve((0, 48), times=t, params={"CL": 0.0, "k_col": 0.0}, **TIGHT)
    )
    total = sum(np.asarray(sol[k]) for k in ("stomach", "small_intestine", "Col", "Ac"))
    assert np.max(np.abs(total - 100.0)) < 1e-6
    assert max(sol.ledger.check().values()) < 1e-10


def test_drug_reaching_the_last_stomach_age_moves_on_into_the_intestine():
    """With a short stomach axis much of the dose reaches g_max; it enters the intestine with
    the emptied drug (read there as G(g=g_max)), so with no elimination nothing is lost."""
    m = oral_absorption_model(n_g=20, n_x=20, g_max=0.5)
    t = np.linspace(0, 12, 13)
    sol = m.solve((0, 12), times=t, params={"CL": 0.0, "k_col": 0.0}, rtol=1e-10, atol=1e-12)
    total = (
        np.asarray(sol["stomach"])
        + np.asarray(sol["small_intestine"])
        + np.asarray(sol["Col"])
        + np.asarray(sol["Ac"])
    )
    np.testing.assert_allclose(total, 100.0, rtol=1e-8)
    assert np.asarray(sol["stomach"])[1] < 1e-6  # all of it has left the stomach by 1 h
