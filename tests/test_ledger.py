# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""V6: conservation ledger.

The oracle for the books is independent of the ledger code: the change in amount computed
directly from the right-hand side (density rates times cell widths, summed), and amounts at
two times computed from the solution itself. Individual channels are also checked against
values written by hand from the model equations.
"""

from __future__ import annotations

import numpy as np
import pytest

import tenqsp as tq
from tenqsp.compile import additive_parts
from tenqsp.examples import phage_latency_model, toy_phage_model

RNG = np.random.default_rng(20261011)


# ---------------------------------------------------------------------------- test models
# Together they use every term kind (reaction, hazard, switch, mutate, rate) and every
# boundary kind (inflow at either end, outflow absorbed and routed, Dirichlet, Robin).


def decay():
    m = tq.Model("decay")
    P = m.compartment("P")
    delta = m.param("delta", 0.3)
    m.ode(P, -delta * P)
    m.dose(P, times=[0.0, 5.0], amounts=[5e8, 1e8])
    return m


def transit(n: int = 5):
    m = tq.Model("transit")
    a = m.axis("a", 0, 2.0, n, kind="one_way")
    A, C = m.compartment("A"), m.compartment("C")
    X = m.field("X", axes=[a])
    ka, ke, D = m.params(ka=1.5, ke=0.2, D=100.0)
    m.advect(X, axis=a, velocity=1.0, scheme="erlang")
    m.inflow(X, axis=a, at="lower", flux=ka * A)
    m.outflow(X, axis=a, at="upper", to=C)
    m.ode(A, -ka * A)
    m.ode(C, -ke * C)
    m.dose(A, times=[0.0, 3.0], amounts=[D, 0.5 * D])
    m.observable("central", C)
    return m


def latency():
    return phage_latency_model(n=20)


def toy():
    return toy_phage_model(n_z=6, n_a=8)


def switch_dirichlet_robin():
    """Three phenotypes switching in a cycle, diffusing on a geometric grid between a held
    value (Dirichlet) and an exchange with a reservoir (Robin), with growth and death."""
    m = tq.Model("switch_bc")
    s = m.axis("s", labels=["A", "B", "C"])
    x = m.axis("x", 0, 1, 8, spacing="geometric", ratio=1.3, kind="spatial")
    F = m.field("F", axes=[x, s])
    q1, q2, r, d = m.params(q1=0.3, q2=0.05, r=0.2, d=0.1)
    Q = m.transitions("Q", axis=s, rates={"A->B": q1, "B->C": q2, "C->A": q1 + q2})
    m.transition(F, axis=s, matrix=Q, of=F)
    m.diffuse(F, axis=x, D=0.05)
    m.reaction(F, r * F * (1 - x.coord))
    m.hazard(F, d)
    m.boundary(F, axis=x, at="lower", kind="dirichlet", value=2.0)
    m.boundary(F, axis=x, at="upper", kind="robin", value=0.5, h=3.0)
    m.init(F, 1.0)
    return m


def two_way():
    """Classes moving up and down a bidirectional axis, leaving at both ends: the lower end
    routes into a compartment W, the upper end absorbs."""
    m = tq.Model("two_way")
    s = m.axis("s", labels=["up", "down"])
    x = m.axis("x", 0, 1, 6)
    B = m.field("B", axes=[s, x])
    W = m.compartment("W")
    c = m.param("c", {"up": 1.0, "down": -2.0}, over=s)
    m.advect(B, axis=x, velocity=c[s], scheme="sharp")
    m.outflow(B, axis=x, at="lower", to=W)
    m.outflow(B, axis=x, at="upper")
    m.ode(W, -0.1 * W)
    m.init(B, tq.normal(x, mean=0.5, sd=0.2))
    return m


def downward():
    """Curved velocity -(1 + x^2) with inflow entering at the upper end, leaving at the lower."""
    m = tq.Model("down")
    x = m.axis("x", 0, 1, 6)
    u = m.field("u", axes=[x])
    m.advect(u, axis=x, velocity=-(1 + x**2))
    m.inflow(u, axis=x, at="upper", flux=2.0)
    m.outflow(u, axis=x, at="lower")
    return m


def held_inflow():
    """Advection in from a held (Dirichlet) value, plus diffusion, out at the upper end."""
    m = tq.Model("held")
    x = m.axis("x", 0, 1, 10, kind="spatial")
    u = m.field("u", axes=[x])
    m.advect(u, axis=x, velocity=2.0)
    m.diffuse(u, axis=x, D=0.02)
    m.boundary(u, axis=x, at="lower", kind="dirichlet", value=3.0)
    m.outflow(u, axis=x, at="upper")
    m.hazard(u, 0.5)
    return m


def mckendrick():
    """Sharp age axis with constant inflow and death (the V4 model)."""
    m = tq.Model("mckendrick")
    a = m.axis("a", 0, 3, 30, kind="one_way")
    u = m.field("u", axes=[a])
    m.advect(u, axis=a, velocity=1.0, scheme="sharp")
    m.inflow(u, axis=a, at="lower", flux=2.0)
    m.outflow(u, axis=a, at="upper")
    m.hazard(u, 0.7)
    return m


def renewal():
    """Births proportional to the population (an integral inside a boundary flux)."""
    m = tq.Model("renewal")
    a = m.axis("a", 0, 3, 20, kind="one_way")
    u = m.field("u", axes=[a])
    b = m.param("b", 1.2)
    m.advect(u, axis=a, velocity=1.0)
    m.inflow(u, axis=a, at="lower", flux=b * m.integral(u, over=a))
    m.outflow(u, axis=a, at="upper")
    m.hazard(u, 1.0)
    m.init(u, 1.0)
    return m


def robin_then_held():
    """Robin exchange at the lower end, held value at the upper end, variable D."""
    m = tq.Model("robin_held")
    x = m.axis("x", 0, 1, 12, kind="spatial")
    u = m.field("u", axes=[x])
    m.diffuse(u, axis=x, D=0.05 * (1 + x))
    m.boundary(u, axis=x, at="lower", kind="robin", value=2.0, h=3.0)
    m.boundary(u, axis=x, at="upper", kind="dirichlet", value=0.5)
    return m


def downward_held_stretched():
    """Downward sharp advection on a stretched grid, entering from a held upper value."""
    m = tq.Model("down_held")
    x = m.axis("x", 0, 1, 15, spacing="geometric", ratio=1.1)
    u = m.field("u", axes=[x])
    m.advect(u, axis=x, velocity=-(1 + x), scheme="sharp")
    m.diffuse(u, axis=x, D=0.01)
    m.boundary(u, axis=x, at="upper", kind="dirichlet", value=1.5)
    m.outflow(u, axis=x, at="lower")
    m.init(u, tq.normal(x, mean=0.3, sd=0.1))
    return m


MODELS = [
    decay,
    transit,
    latency,
    toy,
    switch_dirichlet_robin,
    two_way,
    downward,
    held_inflow,
    mckendrick,
    renewal,
    robin_then_held,
    downward_held_stretched,
]


def _ids(fns):
    return [f.__name__ for f in fns]


# ---------------------------------------------------------------------------- the identity


@pytest.mark.parametrize("build", MODELS, ids=_ids(MODELS))
def test_channels_add_up_to_the_change_in_amount(build):
    """At any state, d(amount)/dt computed from the right-hand side equals the sum of the
    ledger channels, for every field and compartment."""
    cm = build().compile()
    for _ in range(3):
        y = RNG.uniform(0.1, 2.0, cm.n_states) * np.exp(RNG.normal(0, 1, cm.n_states))
        th = cm.theta0 * RNG.uniform(0.5, 1.5, cm.theta0.size)
        d_amount = cm.amounts(np.asarray(cm.rhs(0.7, y, th)))
        books = cm.ledger_rates(0.7, y, th)
        for name, rate in d_amount.items():
            total = sum(books[name].values())
            scale = max(abs(rate), *(abs(v) for v in books[name].values()), 1e-300)
            assert abs(total - rate) <= 1e-12 * scale, (name, total, rate, books[name])


def test_every_channel_kind_is_exercised():
    kinds = set()
    for build in MODELS:
        kinds |= {k for _, k, _ in build().compile().ledger_entries}
    assert kinds == {"reaction", "hazard", "transition", "boundary", "rate", "routed"}
    labels = {lab for b in MODELS for _, _, lab in b().compile().ledger_entries}
    for kind in ("inflow", "outflow (absorbed)", "outflow to", "dirichlet", "robin"):
        assert any(kind in lab for lab in labels), kind


# ---------------------------------------------------------------------------- V6 over time


@pytest.mark.parametrize("build", MODELS, ids=_ids(MODELS))
def test_v6_amounts_match_the_books_over_a_solve(build):
    m = build()
    sol = m.solve((0, 10), rtol=1e-6, atol=1e-6)
    worst = sol.ledger.check()
    assert max(worst.values()) < 1e-10, worst
    # the ledger's amounts are the solution's own
    for name in sol.ledger.names:
        np.testing.assert_array_equal(sol.ledger.amount[name], m.compile().amounts(sol.y)[name])
    # what a routed outflow takes from a field arrives, with the opposite sign, in its target
    L = sol.ledger
    for target, kind, label in m.compile().ledger_entries:
        if kind == "routed":
            field, where = label.removeprefix("outflow from ").split(" at ")
            sent = L.moved[field][f"boundary {where}: outflow to {target}"]
            np.testing.assert_allclose(L.moved[target][label], -sent, rtol=1e-12, atol=1e-300)
            assert np.all(L.moved[target][label] >= 0), "outflow arrives as a gain"


def test_transitions_move_nothing_overall():
    sol = toy().solve((0, 10))
    books = sol.ledger
    mut = books.moved["B"]["mutate along s (M)"]
    assert np.max(np.abs(mut)) < 1e-12 * books.scale("B")


# ---------------------------------------------------------------------------- by hand


def test_decay_books_by_hand():
    """Doses 5e8 at t = 0 and 1e8 at t = 5, decay 0.3: the books at t = 0, 4.99, 5, 10."""
    sol = decay().solve((0, 10), times=[0, 4.99, 5, 10], rtol=1e-10, atol=1e-6)
    L = sol.ledger
    assert L.initial["P"] == 0.0
    np.testing.assert_array_equal(L.moved["P"]["doses"], [5e8, 5e8, 6e8, 6e8])
    t = np.array([0, 4.99, 5, 10])
    exact = 5e8 * np.exp(-0.3 * t) + np.where(t >= 5, 1e8 * np.exp(-0.3 * (t - 5)), 0)
    np.testing.assert_allclose(
        L.moved["P"]["rate: -delta*P"], exact - L.moved["P"]["doses"], rtol=1e-7
    )
    np.testing.assert_allclose(L.rates["P"]["rate: -delta*P"], -0.3 * sol["P"], rtol=1e-14)


def test_books_when_output_starts_later():
    sol = decay().solve((0, 10), times=[3, 6])
    assert sol.ledger.initial["P"] == 0.0
    np.testing.assert_array_equal(sol.ledger.moved["P"]["doses"], [5e8, 6e8])
    assert max(sol.ledger.check().values()) < 1e-10


def test_latency_channels_by_hand():
    """Infection-age example: inflow k*B*P, lysis integral of h*I, outflow v*I(last cell)."""
    m = latency()
    sol = m.solve((0, 6), times=[1.0, 3.0, 6.0], params={"k": 2e-9, "tau": 0.6})
    rates = sol.ledger.rates["I"]
    k, h_max, tau, s = 2e-9, 5.0, 0.6, 0.05
    w = 2.0 / 20
    centres = (np.arange(20) + 0.5) * w
    h = h_max / (1 + np.exp(-(centres - tau) / s))
    I = sol["I"]
    np.testing.assert_allclose(
        rates["boundary age=lower: inflow"], k * sol["B"] * sol["P"], rtol=1e-13
    )
    np.testing.assert_allclose(
        rates["hazard: h_max*sigmoid((age - tau)/s)"], -(I * h).sum(axis=1) * w, rtol=1e-12
    )
    np.testing.assert_allclose(
        rates["boundary age=upper: outflow (absorbed)"], -1.0 * I[:, -1], rtol=1e-13
    )


def test_routed_outflow_appears_on_both_sides():
    sol = transit().solve((0, 12))
    L = sol.ledger
    out = L.moved["X"]["boundary a=upper: outflow to C"]
    into = L.moved["C"]["outflow from X at a=upper"]
    np.testing.assert_allclose(into, -out, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        L.rates["C"]["outflow from X at a=upper"], 1.0 * sol["X"][:, -1], rtol=1e-13
    )
    # depot + transit + central + eliminated = everything dosed
    total = L.amount["A"] + L.amount["X"] + L.amount["C"] - L.moved["C"]["rate: -ke*C"]
    np.testing.assert_allclose(total, L.moved["A"]["doses"], rtol=1e-10)


def test_table_reads_like_an_account():
    text = transit().solve((0, 12)).ledger.table("C")
    assert "amount at the start" in text and "outflow from X at a=upper" in text
    assert "unexplained" in text


def test_ledger_can_be_switched_off():
    m = transit()
    sol = m.solve((0, 12), ledger=False)
    assert sol.ledger is None
    ref = m.solve((0, 12))
    np.testing.assert_allclose(sol["C"], ref["C"], rtol=1e-4, atol=1e-6)


def test_repeated_terms_get_distinct_names():
    m = tq.Model("twice")
    x = m.axis("x", 0, 1, 3)
    F = m.field("F", axes=[x], init=1.0)
    k = m.param("k", 0.1)
    m.hazard(F, k)
    m.hazard(F, k)
    entries = m.compile().ledger_entries
    assert [lab for _, _, lab in entries] == ["hazard: k", "hazard: k #2"]


@pytest.mark.parametrize(
    ("text", "parts"),
    [
        ("g*B*(1 - B/Bmax) - k*B*P", ["g*B*(1 - B/Bmax)", "-k*B*P"]),
        ("beta*lysis - ads_total - delta*P", ["beta*lysis", "-ads_total", "-delta*P"]),
        ("-delta*P", ["-delta*P"]),
        ("a - (b - c)", ["a", "-b", "c"]),
        ("x - -y", ["x", "y"]),
        ("2*(a + b)", ["2*(a + b)"]),
    ],
)
def test_formulas_are_booked_piece_by_piece(text, parts):
    assert additive_parts(text) == parts


def test_statements_agree_across_states():
    """Phage model: bacteria lost to adsorption = infected cells gained; phage made =
    burst size x infected cells that burst."""
    L = latency().solve((0, 24)).ledger
    np.testing.assert_allclose(
        L.moved["B"]["rate: -k*B*P"], -L.moved["I"]["boundary age=lower: inflow"], rtol=1e-12
    )
    np.testing.assert_allclose(
        L.moved["P"]["rate: beta*int_age"],
        -50.0 * L.moved["I"]["hazard: h_max*sigmoid((age - tau)/s)"],
        rtol=1e-12,
    )
