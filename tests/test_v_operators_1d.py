# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""V1–V4 and the 1-D part of V15 (manufactured solutions), on the raw operators.

Oracles come from mathematics (exact solutions, a hand-written transit chain, observed order
of convergence), never from the operator code itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

import tenqsp  # noqa: F401  (enables float64)
from tenqsp.grids import ContinuousAxis
from tenqsp.operators import (
    advective_flux_interior,
    advective_outflow_flux,
    assemble_faces,
    diffusive_flux_interior,
    dirichlet_diffusive_flux,
    divergence,
    robin_flux,
)


def observed_order(ns, errors):
    """Least-squares slope of log(error) against log(1/n)."""
    return float(np.polyfit(np.log(1.0 / np.asarray(ns)), np.log(np.asarray(errors)), 1)[0])


def finest_order(ns, errors):
    """Order between the two finest grids: the asymptotic order, free of coarse-grid effects."""
    return float(np.log(errors[-2] / errors[-1]) / np.log(ns[-1] / ns[-2]))


def advect_rhs(ax: ContinuousAxis, v: float, scheme: str, inflow=0.0):
    """du/dt for constant velocity v > 0: inflow at the lower face, outflow at the upper face."""
    v_int = jnp.full(ax.n - 1, v)

    def rhs(u):
        F_int = advective_flux_interior(
            u,
            v_int,
            dim=0,
            centers=ax.centers,
            faces=ax.faces,
            scheme=scheme,
            lower_value=inflow / v,
        )
        F_up = advective_outflow_flux(u, v, dim=0, side="upper")
        return divergence(assemble_faces(inflow, F_int, F_up, dim=0), ax.widths, dim=0)

    return rhs


# ---------------------------------------------------------------------------- unit checks


def test_upwind_by_hand_both_directions():
    ax = ContinuousAxis.uniform("x", 0.0, 2.0, 4)  # width 0.5
    u = jnp.array([4.0, 3.0, 2.0, 1.0])
    F = advective_flux_interior(u, jnp.ones(3), dim=0, centers=ax.centers, faces=ax.faces)
    np.testing.assert_allclose(F, [4.0, 3.0, 2.0])  # taken from the left cell
    du = divergence(assemble_faces(0.0, F, 0.0, dim=0), ax.widths, dim=0)
    np.testing.assert_allclose(du, [-8.0, 2.0, 2.0, 4.0])
    F_neg = advective_flux_interior(u, -jnp.ones(3), dim=0, centers=ax.centers, faces=ax.faces)
    np.testing.assert_allclose(F_neg, [-3.0, -2.0, -1.0])  # taken from the right cell


def test_operators_act_along_any_dimension():
    ax = ContinuousAxis.uniform("x", 0.0, 1.0, 5)
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.uniform(size=(3, 5, 2)))  # axis on dim 1
    for scheme in ("erlang", "sharp"):
        F = advective_flux_interior(
            u, 0.7 * jnp.ones((1, 4, 1)), dim=1, centers=ax.centers, faces=ax.faces, scheme=scheme
        )
        for i in range(3):
            for j in range(2):
                ref = advective_flux_interior(
                    u[i, :, j],
                    0.7 * jnp.ones(4),
                    dim=0,
                    centers=ax.centers,
                    faces=ax.faces,
                    scheme=scheme,
                )
                np.testing.assert_allclose(F[i, :, j], ref, rtol=1e-14)
    Fd = diffusive_flux_interior(u, 0.3, dim=1, widths=ax.widths)
    np.testing.assert_allclose(
        Fd[1, :, 0], diffusive_flux_interior(u[1, :, 0], 0.3, dim=0, widths=ax.widths)
    )


@pytest.mark.parametrize("scheme", ["erlang", "sharp"])
def test_mass_balance_is_exact(scheme):
    """Change of total amount equals boundary in minus out, for any field and velocity sign."""
    ax = ContinuousAxis.geometric("x", 0.0, 1.0, 12, ratio=1.15)
    rng = np.random.default_rng(1)
    for _ in range(5):
        u = jnp.asarray(rng.uniform(0, 5, ax.n))
        v_int = jnp.asarray(rng.normal(size=ax.n - 1))
        D = jnp.asarray(rng.uniform(0, 1, ax.n))
        F_int = advective_flux_interior(
            u, v_int, dim=0, centers=ax.centers, faces=ax.faces, scheme=scheme
        )
        F_int = F_int + diffusive_flux_interior(u, D, dim=0, widths=ax.widths)
        F_lo = robin_flux(u, 2.0, 1.5, dim=0, side="lower")
        F_up = advective_outflow_flux(u, 0.8, dim=0, side="upper")
        du = divergence(assemble_faces(F_lo, F_int, F_up, dim=0), ax.widths, dim=0)
        assert float(jnp.sum(du * ax.widths)) == pytest.approx(float(F_lo - F_up), abs=1e-12)


def test_sharp_scheme_creates_no_new_extrema():
    """Face states lie between neighbouring averages -> a nonnegative field stays nonnegative."""
    ax = ContinuousAxis.uniform("x", 0.0, 1.0, 50)
    u = jnp.where(jnp.abs(jnp.asarray(ax.centers) - 0.3) < 0.1, 1.0, 0.0)  # square pulse
    du = advect_rhs(ax, 1.0, "sharp")(u)
    # wherever u == 0, du/dt must be >= 0 (nothing can be taken from an empty cell)
    assert float(jnp.min(jnp.where(u == 0.0, du, 0.0))) >= -1e-12


def test_diffusion_harmonic_face_and_blocking():
    ax = ContinuousAxis.uniform("x", 0.0, 2.0, 2)  # two cells of width 1
    u = jnp.array([1.0, 0.0])
    # D = 1 and 3: series average over centre distance 1 is 1/(0.5/1 + 0.5/3) = 1.5
    F = diffusive_flux_interior(u, jnp.array([1.0, 3.0]), dim=0, widths=ax.widths)
    np.testing.assert_allclose(F, [1.5])
    F0 = diffusive_flux_interior(u, jnp.array([1.0, 0.0]), dim=0, widths=ax.widths)
    np.testing.assert_allclose(F0, [0.0])
    g = jax.grad(lambda D: diffusive_flux_interior(u, D, dim=0, widths=ax.widths).sum())(
        jnp.array([1.0, 0.0])
    )
    assert bool(jnp.all(jnp.isfinite(g)))  # no NaN from the blocked branch


def test_boundary_fluxes_by_hand():
    u = jnp.array([2.0, 5.0, 3.0])
    assert float(
        dirichlet_diffusive_flux(u, 0.0, 0.5, dim=0, side="lower", width_edge=0.2)
    ) == pytest.approx(-10.0)
    assert float(
        dirichlet_diffusive_flux(u, 0.0, 0.5, dim=0, side="upper", width_edge=0.2)
    ) == pytest.approx(15.0)
    assert float(robin_flux(u, 2.0, 1.0, dim=0, side="lower")) == pytest.approx(
        -2.0
    )  # (1 - 2)*2 enters
    assert float(robin_flux(u, 2.0, 1.0, dim=0, side="upper")) == pytest.approx(
        4.0
    )  # (3 - 1)*2 leaves
    # with diffusion: resistances 1/h = 0.5 and (0.2/2)/0.5 = 0.2 in series -> (1 - 2)/0.7 enters
    assert float(
        robin_flux(u, 2.0, 1.0, dim=0, side="lower", D_edge=0.5, width_edge=0.2)
    ) == pytest.approx(-1 / 0.7)
    assert float(advective_outflow_flux(u, 2.0, dim=0, side="upper")) == pytest.approx(6.0)
    assert float(advective_outflow_flux(u, 2.0, dim=0, side="lower")) == 0.0  # flow points inward


# ---------------------------------------------------------------------------- V2 Erlang


def test_v2_erlang_equals_transit_chain():
    """Upwind with N cells == N-compartment chain; exit times are gamma(N, rate N v / A)."""
    N, A, v = 5, 2.0, 1.0
    ax = ContinuousAxis.uniform("a", 0.0, A, N, kind="one_way")
    k = N * v / A
    rhs = advect_rhs(ax, v, "erlang")

    def chain(u):  # amounts per unit width, as the finite-volume state
        inflow = jnp.concatenate([jnp.zeros(1), u[:-1]])
        return k * (inflow - u)

    u = jnp.asarray(np.random.default_rng(2).uniform(size=N))
    np.testing.assert_allclose(rhs(u), chain(u), rtol=1e-12, atol=0.0)

    # Unit mass enters cell 0 at t = 0; the exit rate through the upper face is the gamma pdf.
    u0 = np.zeros(N)
    u0[0] = 1.0 / ax.widths[0]
    f = jax.jit(rhs)
    ts = np.linspace(0.1, 8.0, 40)
    sol = integrate.solve_ivp(
        lambda t, y, f=f: np.asarray(f(jnp.asarray(y))),
        (0, 8.0),
        u0,
        method="BDF",
        t_eval=ts,
        rtol=1e-11,
        atol=1e-14,
    )
    exit_rate = v * sol.y[-1]
    np.testing.assert_allclose(
        exit_rate, stats.gamma.pdf(ts, a=N, scale=1.0 / k), rtol=1e-6, atol=1e-10
    )


# ---------------------------------------------------------------------------- V1 advection


def _bump(x, c):
    return np.where(np.abs(x - c) < 0.15, np.cos(np.pi * (x - c) / 0.3) ** 2, 0.0)


def _bump_cell_avg(ax, c):
    xs, ws = np.polynomial.legendre.leggauss(8)
    out = np.zeros(ax.n)
    for xi, wi in zip(xs, ws, strict=True):
        out += 0.5 * wi * _bump(ax.centers + 0.5 * ax.widths * xi, c)
    return out


@pytest.mark.parametrize("scheme", ["erlang", "sharp"])
def test_v1_advection_converges(scheme):
    """A smooth bump moved by v*T; L1 error vs the exact shift decreases with refinement."""
    v, T = 1.0, 0.4
    ns, errs = [50, 100, 200], []
    for n in ns:
        ax = ContinuousAxis.uniform("x", 0.0, 1.0, n, kind="one_way")
        f = jax.jit(advect_rhs(ax, v, scheme))
        sol = integrate.solve_ivp(
            lambda t, y, f=f: np.asarray(f(jnp.asarray(y))),
            (0, T),
            _bump_cell_avg(ax, 0.25),
            method="RK45",
            rtol=1e-9,
            atol=1e-12,
        )
        u_T = sol.y[:, -1]
        assert u_T.min() > -1e-9, "positivity"
        errs.append(np.sum(np.abs(u_T - _bump_cell_avg(ax, 0.25 + v * T)) * ax.widths))
    order = observed_order(ns, errs)
    assert errs[-1] < errs[0]
    assert order > (0.6 if scheme == "erlang" else 1.4), (
        f"{scheme}: order {order:.2f}, errors {errs}"
    )


def test_v1_sharp_is_more_accurate_than_erlang():
    ax = ContinuousAxis.uniform("x", 0.0, 1.0, 100, kind="one_way")
    out = {}
    for scheme in ("erlang", "sharp"):
        f = jax.jit(advect_rhs(ax, 1.0, scheme))
        sol = integrate.solve_ivp(
            lambda t, y, f=f: np.asarray(f(jnp.asarray(y))),
            (0, 0.4),
            _bump_cell_avg(ax, 0.25),
            method="RK45",
            rtol=1e-9,
            atol=1e-12,
        )
        out[scheme] = np.sum(np.abs(sol.y[:, -1] - _bump_cell_avg(ax, 0.65)) * ax.widths)
    assert out["sharp"] < 0.3 * out["erlang"]


# ---------------------------------------------------------------------------- V3 diffusion


def test_v3_diffusion_mode_decay_second_order():
    """Zero-flux on [0, L]: cos(pi x / L) decays at rate D (pi/L)^2; error is O(dx^2)."""
    D, L = 0.7, 2.0
    rate_exact = D * (np.pi / L) ** 2
    ns, errs = [10, 20, 40, 80], []
    for n in ns:
        ax = ContinuousAxis.uniform("x", 0.0, L, n, kind="spatial")
        # cell averages of cos(pi x / L)
        k = np.pi / L
        u = (np.sin(k * ax.faces[1:]) - np.sin(k * ax.faces[:-1])) / (k * ax.widths)
        F_int = diffusive_flux_interior(jnp.asarray(u), D, dim=0, widths=ax.widths)
        du = divergence(assemble_faces(0.0, F_int, 0.0, dim=0), ax.widths, dim=0)
        rate = -float(jnp.dot(du, u * ax.widths) / jnp.dot(u, u * ax.widths))
        errs.append(abs(rate - rate_exact))
    assert observed_order(ns, errs) == pytest.approx(2.0, abs=0.1)


# ---------------------------------------------------------------------------- V4 McKendrick


@pytest.mark.parametrize(
    "scheme, min_order",
    # sharp is in the slow tier; sharp with renewal stays fast (test_gate_model_level)
    [("erlang", 0.85), pytest.param("sharp", 1.7, marks=pytest.mark.slow)],
)
def test_v4_mckendrick_stable_age_distribution(scheme, min_order):
    """du/dt + v du/da = -mu u, inflow J at a = 0 -> steady u(a) = (J/v) exp(-mu a / v)."""
    J, v, mu, A = 3.0, 1.0, 1.5, 4.0
    ns, errs = [40, 80, 160], []
    for n in ns:
        ax = ContinuousAxis.uniform("a", 0.0, A, n, kind="one_way")
        transport = advect_rhs(ax, v, scheme, inflow=J)
        f = jax.jit(lambda u, tr=transport: tr(u) - mu * u)
        sol = integrate.solve_ivp(
            lambda t, y, f=f: np.asarray(f(jnp.asarray(y))),
            (0, 40.0),
            np.zeros(n),
            method="BDF",
            rtol=1e-10,
            atol=1e-13,
        )
        u_ss = sol.y[:, -1]
        lam = mu / v
        exact = (
            (J / v)
            * (np.exp(-lam * ax.faces[:-1]) - np.exp(-lam * ax.faces[1:]))
            / (lam * ax.widths)
        )
        errs.append(np.max(np.abs(u_ss - exact)) / (J / v))
    order = finest_order(ns, errs)
    assert order > min_order, f"{scheme}: order {order:.2f}, errors {errs}"


# ---------------------------------------------------------------------------- V15 (1-D MMS)


def _cell_avg(fn, ax, nq=6):
    xs, ws = np.polynomial.legendre.leggauss(nq)
    return sum(
        0.5 * wi * fn(ax.centers + 0.5 * ax.widths * xi) for xi, wi in zip(xs, ws, strict=True)
    )


def _steady_solve(rhs, n):
    """rhs is affine in u: rhs(u) = A u + b -> solve A u = -b with a dense Jacobian."""
    b = rhs(jnp.zeros(n))
    A = jax.jacfwd(rhs)(jnp.zeros(n))
    return np.linalg.solve(np.asarray(A), -np.asarray(b))


# u*(x) = 1 + 0.5 sin(2x) + x^2 on [0, 1]; D(x) = 0.2 (1 + x); v constant.
def _u(x):
    return 1 + 0.5 * np.sin(2 * x) + x**2


def _du(x):
    return np.cos(2 * x) + 2 * x


def _d2u(x):
    return -2 * np.sin(2 * x) + 2


def _D(x):
    return 0.2 * (1 + x)


def _source(x, v):  # S = d/dx(v u) - d/dx(D du/dx) makes u* steady
    return v * _du(x) - (0.2 * _du(x) + _D(x) * _d2u(x))


@pytest.mark.parametrize(
    "case, v, lower_bc, faces_kind, min_order",
    [
        ("diffusion, Dirichlet both ends", 0.0, "dirichlet", "uniform", 1.85),
        ("diffusion, nonuniform grid", 0.0, "dirichlet", "smooth", 1.85),
        ("diffusion, Robin lower end", 0.0, "robin", "uniform", 1.85),
        ("advection-diffusion, upwind", 1.0, "dirichlet", "uniform", 0.9),
    ],
)
def test_v15_mms_1d(case, v, lower_bc, faces_kind, min_order):
    h_robin, u0, u1 = 3.0, _u(0.0), _u(1.0)
    # Robin data that u* satisfies: flux into the domain h (C - u(0)) = -D(0) u'(0)
    C_out = u0 - _D(0.0) * _du(0.0) / h_robin
    ns, errs = [20, 40, 80, 160], []
    for n in ns:
        if faces_kind == "uniform":
            ax = ContinuousAxis.uniform("x", 0.0, 1.0, n)
        else:  # smoothly stretched grid, refined near x = 1
            xi = np.linspace(0, 1, n + 1)
            ax = ContinuousAxis("x", np.sin(0.5 * np.pi * xi))
        D_cells = jnp.asarray(_D(ax.centers))
        S = jnp.asarray(_cell_avg(lambda x: _source(x, v), ax))

        def rhs(u, ax=ax, D_cells=D_cells, S=S):
            F_int = diffusive_flux_interior(u, D_cells, dim=0, widths=ax.widths)
            if v:
                F_int = F_int + advective_flux_interior(
                    u, jnp.full(ax.n - 1, v), dim=0, centers=ax.centers, faces=ax.faces
                )
            if lower_bc == "dirichlet":
                F_lo = dirichlet_diffusive_flux(
                    u, u0, _D(0.0), dim=0, side="lower", width_edge=ax.widths[0]
                )
            else:
                F_lo = robin_flux(
                    u, h_robin, C_out, dim=0, side="lower", D_edge=_D(0.0), width_edge=ax.widths[0]
                )
            F_lo = F_lo + v * u0  # advective inflow carries the exact boundary value
            F_up = dirichlet_diffusive_flux(
                u, u1, _D(1.0), dim=0, side="upper", width_edge=ax.widths[-1]
            )
            F_up = F_up + advective_outflow_flux(u, v, dim=0, side="upper")
            return divergence(assemble_faces(F_lo, F_int, F_up, dim=0), ax.widths, dim=0) + S

        u_num = _steady_solve(rhs, ax.n)
        errs.append(np.max(np.abs(u_num - _cell_avg(_u, ax))))
    order = observed_order(ns, errs)
    assert order > min_order, f"{case}: observed order {order:.2f}, errors {errs}"
