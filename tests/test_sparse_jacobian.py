# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Sparse Jacobian from the model structure (V7 in miniature).

The oracle is the dense ``jax.jacfwd`` of the same right-hand side: every entry of the
coloured, integral-lifted sparse Jacobian must equal it. A pattern that missed a dependency
would put that derivative into the wrong entry (columns sharing a colour), so equality at a
random state catches it.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_ledger import MODELS, _ids
from test_v_boundaries_phase2 import biofilm, dispersion, dormancy, robin_advection
from test_v_chemotaxis_fronts import keller_segel

import tenqsp as tq
from tenqsp.examples import phage_latency_model, toy_phage_model


def coupled():
    """Every path by which one variable can enter another's equation: a field in a velocity
    and a diffusivity, a compartment in a transition rate and in boundary values, a mutation
    flux of two fields, a Robin end, an outflow routed into a compartment, a nested integral,
    definitions of definitions, and fields stored in a different axis order than the model's."""
    m = tq.Model("coupled")
    x = m.axis("x", 0, 1, 7, kind="bidirectional")
    c = m.axis("c", labels=["a1", "a2", "a3"])
    a = m.axis("a", 0, 2, 6, kind="one_way")
    N = m.field("N", axes=[x])
    B = m.field("B", axes=[c, x])  # stored (c, x); the model's order is (x, c)
    I = m.field("I", axes=[a, x])  # stored (a, x)
    A, W = m.compartment("A"), m.compartment("W")
    D0, K, chi, k, g, h = m.params(D0=0.05, K=2.0, chi=0.7, k=0.3, g=0.8, h=1.5)
    r = m.param("r", {"a1": 0.2, "a2": 0.1, "a3": 0.05}, over=c)

    sat = m.define("sat", N / (K + N))
    grow = m.define("grow", g * sat * B)  # a definition of a definition
    B_x = m.integral(B, over=c, name="B_x")  # varies with x
    load = m.integral(I * B_x, over=[a, x], name="load")  # nested integral
    # nested, and keeps the axis the inner integral summed over: mix[c] needs B at every c
    mix = m.integral(r[c] * B_x * N, over=x, name="mix")
    M = m.transitions("M", axis=c, rates={"a1->a2": 0.1 * A, "a2->a3": 0.05, "a3->a1": 0.02})

    m.diffuse(N, axis=x, D=D0 * (1 + N / K))  # diffusivity depends on the field itself
    m.reaction(N, -k * N * B_x)
    m.boundary(N, axis=x, at="lower", kind="robin", value=A, h=h)
    m.advect(B, axis=x, velocity=chi * (N - 1), scheme="sharp")  # chemotaxis-like drift
    m.diffuse(B, axis=x, D=0.02)
    m.boundary(B, axis=x, at="lower", kind="dirichlet", value=0.1 * A)
    m.outflow(B, axis=x, at="upper", to=W)
    m.reaction(B, grow - r[c] * B - 0.01 * mix * B)
    m.transition(B, axis=c, matrix=M, of=1e-3 * grow)
    m.transition(B, axis=c, matrix=M, of=B)
    m.advect(I, axis=a, velocity=1, scheme="sharp")
    m.hazard(I, 0.4 + 0.1 * N)
    m.inflow(I, axis=a, at="lower", flux=k * B_x * A)
    m.outflow(I, axis=a, at="upper")
    m.ode(A, -0.1 * A + 1e-2 * load)
    m.ode(W, -0.05 * W)
    m.init(N, 1.0)
    m.init(B, 0.5)
    m.init(I, 0.1)
    m.init(A, 2.0)
    m.init(W, 0.0)
    return m


def renewal_compartment_rate():
    """Pure ODE model (no fields): every row is a dense row."""
    m = tq.Model("odes")
    X, Y = m.compartment("X"), m.compartment("Y")
    a, b = m.params(a=1.2, b=0.4)
    m.ode(X, a * X - b * X * Y)
    m.ode(Y, b * X * Y - Y)
    m.init(X, 2.0)
    m.init(Y, 1.0)
    return m


def robin_inlet():
    return robin_advection(7, "lower", "sharp")


def robin_outlet():
    return robin_advection(7, "upper", "sharp")


def biofilm_bulk():
    return biofilm(6, kind="bulk")


def dormancy_crowded():
    return dormancy(8, g=1.0, d=0.1, E=0.5)


def dispersion_sharp():
    return dispersion(8, 0.17, 2.0)


def keller_segel_small():
    return keller_segel(9, 5.0, eps=0.3)


def chemotaxis_outflow():
    """Sharp chemotaxis with flow out through both ends: the boundary faces' gradients."""
    m = tq.Model("chemotaxis_outflow")
    x = m.axis("x", 0, 1, 8, kind="spatial")
    S, B = m.field("S", axes=[x]), m.field("B", axes=[x])
    m.advect(B, axis=x, velocity=0.7 * tq.grad(S, x) + 0.3 * (x.coord - 0.5) * S, scheme="sharp")
    m.outflow(B, axis=x, at="upper")
    m.outflow(B, axis=x, at="lower")
    m.diffuse(S, axis=x, D=0.1)
    m.reaction(S, B - S)
    m.init(S, 1.0)
    m.init(B, 1.0)
    return m


def chemotaxis_by_class():
    """B(x, s) follows S(x) at a speed per class, crowding through its own total over s."""
    m = tq.Model("chemotaxis_by_class")
    s = m.axis("s", labels=["fast", "slow"])
    x = m.axis("x", 0, 1, 7, kind="spatial")
    S, B = m.field("S", axes=[x]), m.field("B", axes=[s, x])
    chi = m.param("chi", {"fast": 1.0, "slow": 0.2}, over=s)
    B_x = m.integral(B, over=s, name="B_x")
    m.advect(B, axis=x, velocity=chi[s] * tq.grad(S, x) / (1 + B_x), scheme="sharp")
    m.diffuse(B, axis=x, D=0.05)
    m.diffuse(S, axis=x, D=0.2)
    m.reaction(S, B_x - S)
    m.init(S, 1.0)
    m.init(B, 1.0)
    return m


BUILDS = [
    *MODELS,
    phage_latency_model,
    lambda: toy_phage_model(9, 11),
    coupled,
    renewal_compartment_rate,
    robin_inlet,
    robin_outlet,
    biofilm_bulk,
    dormancy_crowded,
    dispersion_sharp,
    keller_segel_small,
    chemotaxis_outflow,
    chemotaxis_by_class,
]


def _random_point(cm, seed: int):
    rng = np.random.default_rng(seed)
    y = np.asarray(cm.initial_state()) + rng.uniform(0.1, 1.0, cm.n_states)
    theta = cm.theta0 * rng.uniform(0.8, 1.25, cm.theta0.size)
    return 0.7, y, theta


# the ledger only adds rows computed by reverse mode; a few models cover every channel kind
WITH_LEDGER = [
    m
    for m in BUILDS
    if getattr(m, "__name__", "")
    in ("transit", "coupled", "toy", "robin_outlet", "biofilm_bulk", "chemotaxis_by_class")
]
CASES = [(b, False) for b in BUILDS] + [(b, True) for b in WITH_LEDGER]


@pytest.mark.parametrize(
    "build, ledger",
    CASES,
    ids=[
        f"{i}{' with ledger' if led else ''}"
        for i, (_, led) in zip(_ids([b for b, _ in CASES]), CASES, strict=True)
    ],
)
def test_sparse_jacobian_equals_dense(build, ledger):
    cm = build().compile()
    t, y, th = _random_point(cm, 7)
    if ledger:
        z = np.concatenate([y, np.zeros(len(cm.ledger_entries))])
        dense = np.asarray(cm._aug_jac(t, z, th, cm.geom))
    else:
        dense = np.asarray(cm.jacobian(t, y, th))
    sparse = cm.sparse_jacobian(t, y, th, ledger=ledger)
    assert sparse.shape == dense.shape
    scale = np.abs(dense).max()
    np.testing.assert_allclose(sparse.toarray(), dense, rtol=0, atol=1e-13 * scale)


def test_coupled_model_exercises_every_path():
    """The stress model really does couple what it claims (no accidental zeros)."""
    cm = coupled().compile()
    t, y, th = _random_point(cm, 3)
    J = np.asarray(cm.jacobian(t, y, th))
    sl = cm.layout.slices
    assert np.abs(J[sl["B"], sl["N"]]).max() > 0  # N in B's velocity and growth
    assert np.abs(J[sl["N"], sl["N"]]).max() > 0
    assert np.abs(J[sl["B"], sl["A"]]).max() > 0  # A in a transition rate and a held value
    assert np.abs(J[sl["I"], sl["B"]]).max() > 0  # inflow k*B_x*A
    assert np.abs(J[sl["A"], sl["I"]]).max() > 0  # nested integral
    assert np.abs(J[sl["W"], sl["B"]]).max() > 0  # routed outflow
    assert cm.n_integrals == 7 + 1 + 3  # B_x has 7 values, load 1, mix 3


def test_colours_do_not_grow_with_the_grid():
    """ADR 0003: lifting integrals keeps the number of forward derivatives bounded."""
    counts = []
    for n_z, n_a in [(10, 10), (30, 40), (60, 80)]:
        S = toy_phage_model(n_z, n_a).compile().sparse_structure
        counts.append((S["n_colours"], S["n_row_colours"]))
    assert len(set(counts)) == 1, counts
    assert counts[0][0] <= 30


def plate(n: int = 80):
    """A 2-axis model of 6,400 cells plus a compartment: diffusion across, drift along, decay,
    outflow at one edge collected in a compartment."""
    m = tq.Model("plate")
    x = m.axis("x", 0, 1, n, kind="bidirectional")
    y = m.axis("y", 0, 1, n, kind="one_way")
    u = m.field("u", axes=[x, y])
    C = m.compartment("C")
    m.diffuse(u, axis=x, D=0.01)
    m.advect(u, axis=y, velocity=0.5)
    m.hazard(u, 0.1 * (1 + x))
    m.outflow(u, axis=y, at="upper", to=C)
    m.ode(C, -0.2 * C)
    m.init(u, total=1.0, shape=tq.normal(x, mean=0.5, sd=0.1) * tq.normal(y, mean=0.2, sd=0.1))
    m.init(C, 0.0)
    return m


def test_solves_above_the_old_dense_limit():
    """Phase 1 refused more than 5,000 states (a dense Jacobian of 6,401 states is 330 MB)."""
    m = plate()
    assert m.compile().n_states == 6_401
    sol = m.solve((0, 3), times=[0, 1.5, 3])
    assert sol.stats["njev"] >= 1
    assert np.all(np.isfinite(sol.y)) and np.all(sol.y > -1e-9)
    assert max(sol.ledger.check().values()) < 1e-10
    tight = m.solve((0, 3), times=[3], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sol.y[-1], tight.y[-1], rtol=0, atol=1e-5 * tight.y[-1].max())


@pytest.mark.slow
def test_toy_phage_model_above_the_old_dense_limit():
    """The phage toy at 60 x 100 (6,121 states) against a tight-tolerance solve."""
    m = toy_phage_model(60, 100)
    assert m.compile().n_states == 6_121
    sol = m.solve((0, 1), times=[1])
    tight = m.solve((0, 1), times=[1], rtol=1e-9, atol=1e-9)
    assert max(sol.ledger.check().values()) < 1e-10
    for name in ("phage", "bacteria"):
        np.testing.assert_allclose(sol[name][-1], tight[name][-1], rtol=1e-6)
