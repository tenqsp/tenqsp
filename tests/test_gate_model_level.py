# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Phase 1 gate tests through the compiled model, added after the independent gate
audit (2026-10-11): V15 for every 1-D operator x boundary x scheme combination of Phase 1,
V4 with the renewal (birth) condition, V2 at its tolerance, and V16 counted by JAX itself.

Oracles are mathematical: a manufactured solution u*(x) with its source written out by SymPy
from the PDE, the Euler-Lotka stable age distribution, a hand-written transit chain, and JAX's
own compile events.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp

import tenqsp as tq

SLOW = pytest.mark.slow  # the full matrices run in the slow tier (pytest -m slow); the fast
# tier keeps one representative of every operator, boundary and scheme


def _cases(cases, slow_if):
    return [
        pytest.param(
            *c,
            id="-".join(map(str, c)) if isinstance(c, tuple) else str(c),
            marks=(SLOW,) if slow_if(c) else (),
        )
        for c in cases
    ]


# ---------------------------------------------------------------------------- V15 (model)
# u* = 1 + x^2 + 0.5 tanh(3 (x - 0.5)) on [0, 1]; for a zero-flux end u* = 2 + x^2 - 0.6 x^3
# + 0.3 x^4 (u*'(0) = 0). The steady PDE d/dx(v u) - d/dx(D du/dx) = S holds, S from SymPy.

X = sp.Symbol("x", real=True)
U_SMOOTH = 1 + X**2 + sp.Rational(1, 2) * sp.tanh(3 * (X - sp.Rational(1, 2)))
U_NEUMANN = 2 + X**2 - sp.Rational(3, 5) * X**3 + sp.Rational(3, 10) * X**4

VELOCITIES = {
    "+const": sp.Integer(1),
    "+curved": 1 + X / 2,
    "-const": sp.Integer(-1),
    "-curved": -(1 + X / 2),
}
DIFFUSIVITIES = {"const": sp.Rational(1, 10), "variable": (1 + X) / 10}


def _f(e):
    return sp.lambdify(X, e, "numpy")


def _cell_averages(u, faces):
    nodes, weights = np.polynomial.legendre.leggauss(6)
    a, b = faces[:-1, None], faces[1:, None]
    x = 0.5 * (a + b) + 0.5 * (b - a) * nodes
    return (_f(u)(x) * weights).sum(axis=1) / 2


def _axis(m, n, grid):
    if grid == "uniform":
        return m.axis("x", 0, 1, n)
    return m.axis("x", 0, 1, n, spacing="geometric", ratio=3.0 ** (1 / n))  # last/first ~ 3


def _steady(cm, y0):
    """Steady state by Newton's method (the sharp scheme's limiter makes it nonlinear)."""
    y = np.array(y0, dtype=float)
    for _ in range(50):
        f = np.asarray(cm.rhs(0.0, y))
        if np.max(np.abs(f)) < 1e-11 * max(1.0, np.max(np.abs(y))):
            break
        y = y - np.linalg.solve(np.asarray(cm.jacobian(0.0, y)), f)
    return y


def _errors(cm, u, ax):
    exact = _cell_averages(u, np.asarray(ax.faces))
    y = _steady(cm, exact)
    e = np.abs(y - exact)
    return float(np.sum(e * ax.widths)), float(np.max(e))


def advection_case(n, grid, vname, scheme, upstream_bc):
    """Pure advection; the upstream end is an inflow flux or a held (Dirichlet) value, the
    downstream end an outflow."""
    v, u = VELOCITIES[vname], U_SMOOTH
    S = sp.diff(v * u, X)
    up, down = ("lower", "upper") if vname.startswith("+") else ("upper", "lower")
    x_up = 0 if up == "lower" else 1
    m = tq.Model("mms_adv")
    ax = _axis(m, n, grid)
    w = m.field("u", axes=[ax])
    m.advect(w, axis=ax, velocity=str(v), scheme=scheme)
    m.reaction(w, str(S))
    if upstream_bc == "inflow":
        m.inflow(w, axis=ax, at=up, flux=float(abs(v * u).subs(X, x_up)))
    else:
        m.boundary(w, axis=ax, at=up, kind="dirichlet", value=float(u.subs(X, x_up)))
    m.outflow(w, axis=ax, at=down)
    return m.compile(), u, ax


def diffusion_case(n, grid, dname, ends):
    """Diffusion (no advection) with the given (lower, upper) end conditions."""
    D = DIFFUSIVITIES[dname]
    u = U_NEUMANN if "zero_flux" in ends else U_SMOOTH
    S = -sp.diff(D * sp.diff(u, X), X)
    m = tq.Model("mms_diff")
    ax = _axis(m, n, grid)
    w = m.field("u", axes=[ax])
    m.diffuse(w, axis=ax, D=str(D))
    m.reaction(w, str(S))
    h = 3.0
    for side, kind in zip(("lower", "upper"), ends, strict=True):
        x0 = 0 if side == "lower" else 1
        u0, flux = float(u.subs(X, x0)), float((-D * sp.diff(u, X)).subs(X, x0))
        if kind == "dirichlet":
            m.boundary(w, axis=ax, at=side, kind="dirichlet", value=u0)
        elif kind == "robin":  # exchange h (outside - u) into the domain balances the flux
            outside = u0 + flux / h if side == "lower" else u0 - flux / h
            m.boundary(w, axis=ax, at=side, kind="robin", value=outside, h=h)
    return m.compile(), u, ax


def advection_diffusion_case(n, grid, scheme):
    """Advection v = 1 + x/2 and diffusion D = 0.1 together: total inflow flux at the lower
    end, Dirichlet at the upper end."""
    v, D, u = VELOCITIES["+curved"], DIFFUSIVITIES["const"], U_SMOOTH
    S = sp.diff(v * u - D * sp.diff(u, X), X)
    m = tq.Model("mms_advdiff")
    ax = _axis(m, n, grid)
    w = m.field("u", axes=[ax])
    m.advect(w, axis=ax, velocity=str(v), scheme=scheme)
    m.diffuse(w, axis=ax, D=str(D))
    m.reaction(w, str(S))
    m.inflow(w, axis=ax, at="lower", flux=float((v * u - D * sp.diff(u, X)).subs(X, 0)))
    m.boundary(w, axis=ax, at="upper", kind="dirichlet", value=float(u.subs(X, 1)))
    return m.compile(), u, ax


def advection_diffusion_robin_case(n, grid, scheme, robin_at):
    """Advection v = 1 + x/2 and diffusion D = 0.1 with a Robin end (Phase 2): at the inlet
    (lower) the entering flow carries the outside level; at the outlet (upper) the flow leaves
    with the boundary value. The outside levels are chosen so that u* satisfies both ends:
    lower v c + h (c - u) = (v u - D u')(0); upper D u'(1) = h (c - u(1))."""
    v, D, u = VELOCITIES["+curved"], DIFFUSIVITIES["const"], U_SMOOTH
    S = sp.diff(v * u - D * sp.diff(u, X), X)
    h = 3.0
    m = tq.Model("mms_advdiff_robin")
    ax = _axis(m, n, grid)
    w = m.field("u", axes=[ax])
    m.advect(w, axis=ax, velocity=str(v), scheme=scheme)
    m.diffuse(w, axis=ax, D=str(D))
    m.reaction(w, str(S))
    total = v * u - D * sp.diff(u, X)
    if robin_at == "lower":
        v0, u0, F0 = (float(e.subs(X, 0)) for e in (v, u, total))
        m.boundary(w, axis=ax, at="lower", kind="robin", value=(F0 + h * u0) / (v0 + h), h=h)
        m.boundary(w, axis=ax, at="upper", kind="dirichlet", value=float(u.subs(X, 1)))
    else:
        m.inflow(w, axis=ax, at="lower", flux=float(total.subs(X, 0)))
        outside = float((u + D * sp.diff(u, X) / h).subs(X, 1))
        m.boundary(w, axis=ax, at="upper", kind="robin", value=outside, h=h)
    return m.compile(), u, ax


def _orders(build, ns=(20, 40, 80, 160)):
    errs = np.array([_errors(*build(n)) for n in ns])
    orders = np.log2(errs[:-1] / errs[1:])
    return errs, orders[-1]  # finest pair


ADV = [
    (g, v, s, bc)
    for g in ("uniform", "stretched")
    for v in VELOCITIES
    for s in ("erlang", "sharp")
    for bc in ("inflow", "dirichlet")
]


@pytest.mark.parametrize(
    ("grid", "vname", "scheme", "bc"),
    _cases(ADV, lambda c: c[0] == "stretched" or c[1].endswith("curved") and c[2] == "erlang"),
)
def test_v15_advection_through_the_model(grid, vname, scheme, bc):
    """Design order: erlang 1 (L1 and max); sharp 2 in L1, 1 in max (the outflow cell uses
    first-order upwinding, ADR 0004)."""
    errs, (o1, omax) = _orders(lambda n: advection_case(n, grid, vname, scheme, bc))
    if scheme == "erlang":
        assert o1 > 0.85 and omax > 0.85, (errs, o1, omax)
    else:
        assert o1 > 1.85 and omax > 0.85, (errs, o1, omax)


DIFF = [
    (g, d, ends)
    for g in ("uniform", "stretched")
    for d in DIFFUSIVITIES
    for ends in [
        ("dirichlet", "dirichlet"),
        ("robin", "dirichlet"),
        ("dirichlet", "robin"),
        ("zero_flux", "dirichlet"),
    ]
]


@pytest.mark.parametrize(
    ("grid", "dname", "ends"),
    [
        pytest.param(
            g,
            d,
            e,
            id=f"{g}-{d}-{e[0]}-{e[1]}",
            marks=(SLOW,) if (g == "stretched") == (d == "const") else (),
        )
        for g, d, e in DIFF
    ],
)
def test_v15_diffusion_through_the_model(grid, dname, ends):
    """Design order 2 in L1 and max, for every end condition, on uniform and stretched grids."""
    errs, (o1, omax) = _orders(lambda n: diffusion_case(n, grid, dname, ends))
    assert o1 > 1.85 and omax > 1.7, (errs, o1, omax)


@pytest.mark.parametrize(
    ("grid", "scheme"),
    _cases(
        [(g, s) for g in ("uniform", "stretched") for s in ("erlang", "sharp")],
        lambda c: c[0] == "stretched",
    ),
)
def test_v15_advection_diffusion_through_the_model(grid, scheme):
    errs, (o1, _) = _orders(lambda n: advection_diffusion_case(n, grid, scheme))
    assert o1 > (0.85 if scheme == "erlang" else 1.85), (errs, o1)


@pytest.mark.parametrize(
    ("grid", "scheme", "robin_at"),
    _cases(
        [
            (g, s, r)
            for g in ("uniform", "stretched")
            for s in ("erlang", "sharp")
            for r in ("lower", "upper")
        ],
        lambda c: c[0] == "stretched",
    ),
)
def test_v15_advection_diffusion_robin_through_the_model(grid, scheme, robin_at):
    """Robin end together with advection (Phase 2): design order 1 (erlang), 2 (sharp) in L1."""
    errs, (o1, _) = _orders(lambda n: advection_diffusion_robin_case(n, grid, scheme, robin_at))
    assert o1 > (0.85 if scheme == "erlang" else 1.85), (errs, o1)


def chemotaxis_case(n, grid, scheme):
    """Velocity chi grad(S, x) (Phase 2): S relaxes to the profile x + x^2/2 (S' = 1 + x > 0,
    so the flow enters at x = 0 and leaves at x = 1); u* is held steady by a source."""
    chi = sp.Rational(1, 2)
    s_prof = X + X**2 / 2
    v, u = chi * sp.diff(s_prof, X), U_SMOOTH
    m = tq.Model("mms_chemotaxis")
    ax = _axis(m, n, grid)
    S = m.field("S", axes=[ax])
    w = m.field("u", axes=[ax])
    m.reaction(S, str(s_prof) + " - S")
    m.advect(w, axis=ax, velocity=float(chi) * tq.grad(S, ax), scheme=scheme)
    m.reaction(w, str(sp.diff(v * u, X)))
    m.inflow(w, axis=ax, at="lower", flux=float((v * u).subs(X, 0)))
    m.outflow(w, axis=ax, at="upper")
    return m.compile(), u, ax


def _chemotaxis_errors(n, grid, scheme):
    cm, u, ax = chemotaxis_case(n, grid, scheme)
    exact_u = _cell_averages(u, np.asarray(ax.faces))
    c = np.asarray(ax.centers)
    y = _steady(cm, np.concatenate([c + c**2 / 2, exact_u]))
    e = np.abs(cm.unpack(y)["u"] - exact_u)
    return float(np.sum(e * ax.widths)), float(np.max(e))


@pytest.mark.parametrize(
    ("grid", "scheme"),
    _cases(
        [(g, s) for g in ("uniform", "stretched") for s in ("erlang", "sharp")],
        lambda c: c[0] == "stretched" and c[1] == "erlang",
    ),
)
def test_v15_chemotaxis_through_the_model(grid, scheme):
    """grad(S, x) on the faces of uniform and stretched grids: design order as for any
    advection (erlang 1; sharp 2 in L1, 1 in max: the outflow cell, ADR 0004)."""
    ns = (20, 40, 80, 160)
    errs = np.array([_chemotaxis_errors(n, grid, scheme) for n in ns])
    o1, omax = np.log2(errs[:-1] / errs[1:])[-1]
    if scheme == "erlang":
        assert o1 > 0.85 and omax > 0.85, (errs, o1, omax)
    else:
        assert o1 > 1.85 and omax > 0.85, (errs, o1, omax)


@pytest.mark.parametrize("vname", ["+const", "-const"])
def test_sharp_scheme_uses_the_held_value_at_an_inflow_end(vname):
    """With a Dirichlet inflow end the sharp scheme reconstructs the first face from the held
    value. Bounds are the errors of the correct scheme at n = 40 plus 3 % (ignoring the held
    value raised them by 12 % for v > 0 and 6x for v < 0)."""
    l1, _ = _errors(*advection_case(40, "uniform", vname, "sharp", "dirichlet"))
    bound = {"+const": 1.002e-3, "-const": 1.731e-4}[vname] * 1.03  # held value ignored:
    # 1.144e-3 and 1.115e-3
    assert l1 < bound, l1


# ---------------------------------------------------------------------------- V4 renewal


def renewal_model(n, scheme, b, mu=1.0, A=3.0):
    """Age-structured population: births b * (total) enter at age 0, death mu, ages up to A."""
    m = tq.Model("renewal")
    a = m.axis("a", 0, A, n, kind="one_way")
    u = m.field("u", axes=[a])
    birth = m.param("b", b)
    m.advect(u, axis=a, velocity=1.0, scheme=scheme)
    m.inflow(u, axis=a, at="lower", flux=birth * m.integral(u, over=a))
    m.outflow(u, axis=a, at="upper")
    m.hazard(u, mu)
    m.init(u, 1.0)
    return m, a


def _euler_lotka(b, mu, A):
    """r with b * (1 - exp(-(mu + r) A)) / (mu + r) = 1."""
    from scipy.optimize import brentq

    return brentq(lambda r: b * -np.expm1(-(mu + r) * A) / (mu + r) - 1, -mu + 1e-9, 50)


@pytest.mark.parametrize(("scheme", "min_order"), [("erlang", 0.85), ("sharp", 1.6)])
@pytest.mark.parametrize("births", ["replacement", pytest.param("growing", marks=SLOW)])
def test_v4_renewal_gives_the_stable_age_distribution(scheme, min_order, births):
    """McKendrick-von Foerster with renewal: the normalised age profile tends to
    exp(-(mu + r) a), r the Euler-Lotka root (r = 0 at replacement fertility)."""
    mu, A = 1.0, 3.0
    b = mu / -np.expm1(-mu * A) * (1.0 if births == "replacement" else 2.0)
    r = _euler_lotka(b, mu, A)
    errs = []
    for n in (20, 40, 80):
        m, a = renewal_model(n, scheme, b, mu, A)
        sol = m.solve((0, 40), times=[40], rtol=1e-10, atol=1e-12, ledger=False)
        prof = sol["u"][-1] / np.sum(sol["u"][-1] * a.widths)
        f = np.asarray(a.faces)
        exact = -np.diff(np.exp(-(mu + r) * f)) / (mu + r) / a.widths
        exact = exact / np.sum(exact * a.widths)
        errs.append(np.sum(np.abs(prof - exact) * a.widths))
    order = np.log2(errs[-2] / errs[-1])
    assert order > min_order, (errs, order)


# ---------------------------------------------------------------------------- V2


@pytest.mark.parametrize(
    ("N", "v"),
    [(1, 1.0), (2, 0.5), pytest.param(5, 2.5, marks=SLOW), pytest.param(20, 1.0, marks=SLOW)],
)
def test_v2_erlang_age_axis_matches_the_transit_chain_to_1e_10(N, v):
    """N erlang cells over [0, T] at speed v = an N-compartment chain with rate N v / T."""
    T, ka, ke = 2.0, 1.5, 0.2
    m = tq.Model("transit")
    a = m.axis("a", 0, T, N, kind="one_way")
    A, C = m.compartment("A", init=100.0), m.compartment("C")
    Xf = m.field("X", axes=[a])
    m.advect(Xf, axis=a, velocity=v)
    m.inflow(Xf, axis=a, at="lower", flux=ka * A)
    m.outflow(Xf, axis=a, at="upper", to=C)
    m.ode(A, -ka * A)
    m.ode(C, -ke * C)
    k = N * v / T

    def chain(t, x):
        A_, Xs, C_ = x[0], x[1:-1], x[-1]
        into = np.concatenate([[ka * A_], k * Xs[:-1]])
        return np.concatenate([[-ka * A_], into - k * Xs, [k * Xs[-1] - ke * C_]])

    t = np.linspace(0, 12, 13)
    sol = m.solve((0, 12), times=t, rtol=1e-12, atol=1e-14, ledger=False)
    ref = solve_ivp(
        chain, (0, 12), [100.0] + [0.0] * (N + 1), method="Radau", t_eval=t, rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(sol["C"], ref.y[-1], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sol["X"] * (T / N), ref.y[1:-1].T, rtol=1e-10, atol=1e-10)


# V16 counted by JAX's own backend-compile events: in test_v16_compile_once.py.
