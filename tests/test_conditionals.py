# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Conditionals in formulas (ADR 0006): ``v1 if c1 else v2 if c2 else v3``, ``tq.cases``.

Oracles: values by hand; closed-form solutions of piecewise ODEs (the switch time is known
exactly, so a solve that stops at the switch matches to the solver tolerance, and one that
steps across it does not); finite differences for derivatives.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tenqsp as tq
from tenqsp.expr import FormulaError, parse

TIGHT = {"rtol": 1e-10, "atol": 1e-12}


# --------------------------------------------------------------------------- grammar
@pytest.mark.parametrize(
    "text, message",
    [
        ("k*(x > 1)", "is a condition"),
        ("x if x > 1 & x < 2 else 0", "join conditions with 'and'"),
        ("x if x > 1 | x < 0 else 0", "join conditions with 'or'"),
        ("x if ~(x > 1) else 0", "write 'not'"),
        ("x if x else 0", "After 'if' comes a condition"),
        ("x if True else 0", "not a number"),
        ("x > 1", "is a condition"),
        ("x if x in y else 0", "only < <= > >= == != compare"),
    ],
)
def test_grammar_refusals(text, message):
    import sympy as sp

    names = {n: sp.Symbol(n, real=True) for n in ("x", "y", "k")}
    with pytest.raises(FormulaError, match=message):
        parse(text, names)


def rates(formula: str, values: np.ndarray, **params):
    """Evaluate ``formula`` as a reaction rate of a field F at cell values ``values``."""
    m = tq.Model("c")
    x = m.axis("x", 0, 1, len(values))
    F = m.field("F", axes=[x])
    if params:
        m.params(**params)
    m.reaction(F, formula)
    m.init(F, 1.0)
    return np.asarray(m.compile().rhs(0.0, values))


def test_first_true_condition_wins():
    u = np.array([-1.0, 0.5, 1.5, 2.5, 3.5])
    got = rates("10 if F > 3 else 20 if F > 1 else 30 if F >= 0.5 else 40", u)
    np.testing.assert_array_equal(got, [40, 30, 20, 20, 10])
    got = rates("1 if 0.5 < F <= 2.5 else 0", u)  # chained comparison
    np.testing.assert_array_equal(got, [0, 0, 1, 1, 0])
    got = rates("1 if not (F > 0 and F < 3) or F == 3.5 else 0", u)
    np.testing.assert_array_equal(got, [1, 0, 0, 0, 1])


def test_python_api_builds_the_same_formula():
    m = tq.Model("api")
    x = m.axis("x", 0, 1, 5)
    F = m.field("F", axes=[x])
    k = tq.cases((F > 3, 10), ((F > 1) & ~(F > 2), 20), (tq.eq(F, -1), 50), otherwise=30)
    assert k.text == "10 if F > 3 else 20 if F > 1 and not F > 2 else 50 if F == -1 else 30"
    m.reaction(F, 2 * k + 1)  # a conditional inside arithmetic gets parentheses
    m.init(F, 1.0)
    u = np.array([-1.0, 0.5, 1.5, 2.5, 3.5])
    np.testing.assert_array_equal(m.compile().rhs(0.0, u), 2 * np.array([50, 30, 20, 30, 10]) + 1)


def test_python_api_misuse_is_explained():
    m = tq.Model("api")
    A = m.compartment("A")
    with pytest.raises(TypeError, match="no true/false value yet"):
        bool(A > 1)
    with pytest.raises(TypeError, match="no true/false value yet"):
        0.5 < A <= 1  # noqa: B015 - Python chains with 'and', which needs a truth value
    with pytest.raises(TypeError, match="tq.eq"):
        A == 1  # noqa: B015
    with pytest.raises(TypeError, match="is a condition"):
        (A > 1) * 2
    with pytest.raises(TypeError, match="at least one"):
        tq.cases(otherwise=1)


# --------------------------------------------------------------------------- derivatives
def test_a_branch_not_chosen_cannot_poison_derivatives():
    """log(P) + 1/P is infinite at P = 0, where the other branch is chosen: forward and
    reverse derivatives stay finite and correct."""
    m = tq.Model("guard")
    P, Q = m.compartment("P", init=0.0), m.compartment("Q", init=1.0)
    m.ode(P, "log(P) + 1/P + sqrt(P) if P > 0 else 2*Q")
    m.ode(Q, "-Q")
    cm = m.compile()
    for y in (np.array([0.0, 1.0]), np.array([0.5, 1.0])):
        rev = np.asarray(jax.jacrev(lambda z: cm._rhs_impl(0.0, z, cm.theta0, cm.geom))(y))
        fwd = np.asarray(cm.jacobian(0.0, y))
        assert np.all(np.isfinite(rev)) and np.all(np.isfinite(fwd))
        np.testing.assert_allclose(rev, fwd, rtol=1e-14)
        np.testing.assert_allclose(cm.sparse_jacobian(0.0, y).toarray(), fwd, rtol=1e-14)
    p = 0.5
    assert fwd[0, 0] == pytest.approx(1 / p - 1 / p**2 + 0.5 / np.sqrt(p), rel=1e-14)


# --------------------------------------------------------------------------- time switches
def switched_on():
    m = tq.Model("on")
    A = m.compartment("A", init=0.0)
    m.params(t_on=5.0, kin=1.0)
    m.ode(A, "(kin if t > t_on else 0) - 0.1*A")
    return m


@pytest.mark.scipy_backend
def test_a_time_switch_becomes_a_segment_boundary():
    m = switched_on()
    cm = m.compile()
    assert not cm.switch_events and [str(a) for a, _ in cm.switch_roots] == ["t > t_on"]
    t = np.array([0, 4, 5, 6, 20.0])
    for t_on in (5.0, 7.5):  # a new switch time is a new parameter value: no recompiling
        sol = m.solve((0, 20), times=t, params={"t_on": t_on}, **TIGHT)
        exact = np.where(t > t_on, 10 * (1 - np.exp(-0.1 * (t - t_on))), 0.0)
        np.testing.assert_allclose(sol["A"], exact, atol=1e-8)
        assert sol.stats["segments"] == 2
    assert cm.compilations()["solver rhs (with ledger)"] == 1


def test_a_short_pulse_is_never_stepped_over():
    """A 1e-3 h pulse inside a long quiet run: its two switch times are boundaries."""
    m = tq.Model("pulse")
    A = m.compartment("A", init=0.0)
    m.ode(A, "1000 if 30 < t <= 30.001 else 0")
    sol = m.solve((0, 100), times=[100], **TIGHT)
    assert sol["A"][-1] == pytest.approx(1.0, rel=1e-8)


def test_a_time_condition_without_closed_form_is_located_before_solving():
    """sin(t) > 0 has infinitely many roots. Solver events would miss them (the right-hand
    side is constant on each side, so the solver takes steps longer than pi); sampled and
    polished before the solve, they become segment boundaries."""
    m = tq.Model("days")
    A = m.compartment("A", init=0.0)
    m.ode(A, "1 if sin(t) > 0 else 0")
    cm = m.compile()
    assert [str(a) for a in cm.switch_sampled] == ["sin(t) > 0"] and not cm.switch_events
    np.testing.assert_allclose(cm.switch_times(None, (0, 10)), [np.pi, 2 * np.pi, 3 * np.pi])
    sol = m.solve((0, 2 * np.pi + 1), times=[np.pi, 2 * np.pi + 1], **TIGHT)
    np.testing.assert_allclose(sol["A"], [np.pi, np.pi + 1], rtol=1e-8)


# --------------------------------------------------------------------------- state switches
def test_a_switch_on_a_compartment_stops_exactly():
    """Elimination at 0.5/h above C = 2, 0.1/h below: two exponentials joined at ln(5)/0.5."""
    m = tq.Model("sat")
    C = m.compartment("C", init=10.0)
    m.ode(C, -tq.cases((C > 2, 0.5), otherwise=0.1) * C)
    ts = np.log(5) / 0.5
    t = np.array([0, 1, ts, 5, 20.0])
    sol = m.solve((0, 20), times=t, **TIGHT)
    exact = np.where(t < ts, 10 * np.exp(-0.5 * t), 2 * np.exp(-0.1 * (t - ts)))
    np.testing.assert_allclose(sol["C"], exact, rtol=1e-8)
    assert max(sol.ledger.check().values()) < 1e-12
    assert sol.ledger.moved["C"]["rate: -(0.5 if C > 2 else 0.1)*C"][-1] < 0


def test_a_switch_on_a_total_is_an_event_and_a_cell_switch_is_noted():
    """Growth 1/h slows to 0.3/h once the total passes a quorum of 5 (one crossing):
    B_tot = B0 e^t until ln(5/B0), then 5 e^(0.3 (t - ts)). The per-cell switch never fires
    here; it is only classified and noted."""
    m = tq.Model("crowd")
    x = m.axis("x", 0, 1, 6)
    B = m.field("B", axes=[x])
    B_tot = m.integral(B, over=x, name="B_tot")
    m.diffuse(B, axis=x, D=0.01)
    m.reaction(B, tq.cases((B_tot > 5, 0.3), otherwise=1.0) * B)  # quorum
    m.reaction(B, tq.cases((B > 1e6, -0.1), otherwise=0.0) * B)  # local crowding, per cell
    m.init(B, 2.0)
    cm = m.compile()
    assert [str(a) for a in cm.switch_events] == ["B_tot > 5"]
    assert len(cm.switch_cells) == 1 and str(cm.switch_cells[0]).startswith("B > ")
    notes = [i.message for i in m.validate(raise_on_error=False).issues if i.level == "note"]
    assert any("(varies over ['x']) switches cell by cell" in n for n in notes)
    assert not any("B_tot > 5" in n for n in notes)
    t = np.array([0, 0.5, 1.0, 2.0, 4.0])
    sol = m.solve((0, 4), times=t, **TIGHT)
    ts = np.log(5 / 2)
    exact = np.where(t < ts, 2 * np.exp(t), 5 * np.exp(0.3 * (t - ts)))
    np.testing.assert_allclose(cm.amounts(sol.y)["B"], exact, rtol=1e-8)
    assert max(sol.ledger.check().values()) < 1e-10


def test_sliding_on_a_boundary_is_explained():
    m = tq.Model("chatter")
    X = m.compartment("X", init=1.0)
    m.ode(X, "-1 if X > 0 else 1")
    with pytest.raises(tq.SolverError, match="'X > 0' had just switched at t = 1"):
        m.solve((0, 3))


def test_sparse_jacobian_with_conditionals():
    m = tq.Model("mixed")
    x = m.axis("x", 0, 1, 7)
    B = m.field("B", axes=[x])
    P = m.compartment("P")
    m.advect(B, axis=x, velocity=tq.cases((P > 1, 0.5), otherwise=-0.3), scheme="sharp")
    m.reaction(B, "B*log(B) if B > 0.2 else -B")
    m.ode(P, "1 - P if P < 3 else -P")
    m.init(B, 1.0)
    m.init(P, 1.0)
    cm = m.compile()
    rng = np.random.default_rng(6)
    for _ in range(3):
        y = rng.uniform(0.0, 4.0, cm.n_states)
        np.testing.assert_allclose(
            cm.sparse_jacobian(0.0, y).toarray(), np.asarray(cm.jacobian(0.0, y)), atol=1e-13
        )


def test_ir_round_trip_keeps_the_text():
    m = switched_on()
    ir = m.to_ir()
    assert ir.terms[0].expr == "(kin if t > t_on else 0) - 0.1*A"
    assert tq.Model.from_ir(tq.ModelIR.from_json(ir.to_json())).to_ir() == ir


def test_a_state_switch_starting_exactly_on_its_boundary():
    """C = 0 until a dose at t = 5, and B decays only while C > 0: the condition stands exactly
    on its boundary from the start and must not fire until C leaves it (it used to fire at
    every step, 10,000 events at t = 0). B = e^{0.3*5} e^{-(t - 5)} after the dose."""
    m = tq.Model("drug_on")
    C, B = m.compartment("C", init=0.0), m.compartment("B", init=1.0)
    m.ode(C, "-0.5*C")
    m.ode(B, tq.cases((C > 0, -1.0 * B), otherwise=0.3 * B))
    m.dose(C, times=[5.0], amounts=[1.0])
    sol = m.solve((0, 10), times=[5.0, 10.0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sol["B"], [np.exp(1.5), np.exp(1.5 - 5)], rtol=1e-7)


def test_a_state_at_rest_on_its_boundary():
    """C sits at its equilibrium 1, exactly on 'C > 1': nothing switches, X stays 0."""
    m = tq.Model("at_rest")
    C, X = m.compartment("C", init=1.0), m.compartment("X", init=0.0)
    m.ode(C, "-0.5*(C - 1)")
    m.ode(X, tq.cases((C > 1.0, 1.0), otherwise=0.0))
    sol = m.solve((0, 5), times=[5])
    assert sol["X"][-1] == 0.0
