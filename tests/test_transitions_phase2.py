# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 step 2.5: transition rates that vary across the other axes, joint transitions
across two categorical axes, V28 (Balaban) and the two-phage ``B[r1, r2]`` toy.

Oracles: the matrix exponential (linear constant-coefficient systems), closed forms of
independent cells, and the same model written by hand as a flat ODE system solved with scipy
at tight tolerance. Parameter values are illustrative (they test the mathematics, not a
published fit).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import tenqsp as tq
from tenqsp.examples import toy_phage_model
from tenqsp.ir import IR_VERSION, ModelIR

TIGHT = {"rtol": 1e-10, "atol": 1e-12}


# --------------------------------------------------------------------------- V28
def test_v28_balaban_kill_curve_is_the_matrix_exponential():
    """Balaban 2004 under drug: normal cells n killed at gn, persisters p at gp, switching
    n -> p at a and back at b. Linear: y(t) = expm(A t) y0, and the kill curve is biphasic
    with the two eigenvalues of A as its slopes."""
    gn, gp, a, b = 4.0, 0.05, 1e-3, 1e-2
    m = tq.Model("balaban")
    st = m.axis("state", labels=["n", "p"])
    B = m.field("B", axes=[st])
    m.params(a=a, b=b)
    gamma = m.param("gamma", {"n": gn, "p": gp}, over=st)
    Q = m.transitions("Q", axis=st, rates={"n->p": "a", "p->n": "b"})
    m.transition(B, axis=st, matrix=Q, of=B)
    m.hazard(B, gamma[st])
    m.init(B, 1e6, where={st: "n"})
    m.init(B, 0.0, where={st: "p"})
    t = np.array([0, 0.5, 1, 2, 5, 10, 24.0])
    sol = m.solve((0, 24), times=t, **TIGHT)

    A = np.array([[-gn - a, b], [a, -gp - b]])
    exact = np.array([expm(A * ti) @ [1e6, 0.0] for ti in t])
    np.testing.assert_allclose(sol["B"], exact, rtol=1e-8, atol=1e-6)
    lam = np.sort(np.linalg.eigvals(A).real)
    total = sol["B"].sum(axis=1)
    late = np.log(total[-1] / total[-2]) / (t[-1] - t[-2])  # second phase: slow eigenvalue
    assert late == pytest.approx(lam[1], rel=1e-4)
    first = np.log(total[1] / total[0]) / 0.5  # first phase: the fast eigenvalue
    assert first == pytest.approx(lam[0], rel=1e-2)


# --------------------------------------------------------------------------- integral rate
def test_nielsen_switch_rate_depends_on_the_total():
    """Nielsen 2007: S grows and is killed by the drug; S -> R at
    k_SR = (k_g - k_d)(S + R)/B_max, a rate depending on a total over the switching axis."""
    kg, kd, Bmax, Emax, EC50, hill, ke = 1.5, 0.2, 1e9, 3.0, 1.0, 1.5, 0.3
    m = tq.Model("nielsen")
    st = m.axis("state", labels=["S", "R"])
    B = m.field("B", axes=[st])
    C = m.compartment("C", init=4.0)
    m.params(kg=kg, kd=kd, Bmax=Bmax, Emax=Emax, EC50=EC50, hill=hill, ke=ke)
    m.param("grows", {"S": 1.0, "R": 0.0}, over=st)
    m.integral(B, over=st, name="N")
    m.define("E", "Emax*C^hill/(EC50^hill + C^hill)")
    Q = m.transitions("Q", axis=st, rates={"S->R": "(kg - kd)*N/Bmax"})
    m.reaction(B, "grows[state]*(kg - E)*B - kd*B")
    m.transition(B, axis=st, matrix=Q, of=B)
    m.ode(C, "-ke*C")
    m.init(B, 1e6, where={st: "S"})
    m.init(B, 0.0, where={st: "R"})
    t = np.linspace(0, 48, 9)
    sol = m.solve((0, 48), times=t, **TIGHT)

    def rhs(_, y):
        S, R, c = y
        ksr = (kg - kd) * (S + R) / Bmax
        e = Emax * c**hill / (EC50**hill + c**hill)
        return [kg * S - kd * S - e * S - ksr * S, ksr * S - kd * R, -ke * c]

    ref = solve_ivp(rhs, (0, 48), [1e6, 0, 4.0], t_eval=t, method="Radau", rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(sol["B"], ref.y[:2].T, rtol=1e-7, atol=1e-3)
    assert sol["B"][-1, 1] > 1e7  # resting cells accumulate as the population nears B_max


# --------------------------------------------------------------------------- batched rates
def test_rate_varying_over_another_axis_and_in_time():
    """B(x, s): S -> R at k*x + kc*C, a different rate in every cell of x and changing with
    the compartment C (dC/dt = -C). No transport, so every cell is its own two-state system:
    B_S = B0 exp(-k x t - kc (1 - e^-t))."""
    k, kc = 0.4, 0.7
    m = tq.Model("batched")
    x = m.axis("x", 0, 1, 5)
    s = m.axis("s", labels=["S", "R"])
    B = m.field("B", axes=[x, s])
    C = m.compartment("C", init=1.0)
    m.params(k=k, kc=kc)
    Q = m.transitions("Q", axis=s, rates={"S->R": "k*x + kc*C"})
    m.transition(B, axis=s, matrix=Q, of=B)
    m.ode(C, -C)
    m.init(B, 2.0, where={s: "S"})
    m.init(B, 0.0, where={s: "R"})
    t = np.array([0, 1, 3, 6.0])
    sol = m.solve((0, 6), times=t, **TIGHT)
    xc = np.asarray(x.grid.centers)
    BS = 2.0 * np.exp(-k * xc[None] * t[:, None] - kc * (1 - np.exp(-t[:, None])))
    np.testing.assert_allclose(sol["B"][..., 0], BS, rtol=1e-8)
    np.testing.assert_allclose(sol["B"][..., 1], 2.0 - BS, rtol=1e-8, atol=1e-12)


def test_rate_indexed_by_another_categorical_axis():
    """Organ-indexed exchange PS[organ] between the vascular and tissue spaces (a batched
    generator, one per organ): each organ relaxes to equal amounts at its own rate 2*PS."""
    PS = {"liver": 0.5, "gut": 0.1, "lung": 2.0}
    m = tq.Model("organs")
    organ = m.axis("organ", labels=list(PS))
    sub = m.axis("sub", labels=["vascular", "tissue"])
    A = m.field("A", axes=[organ, sub])
    ps = m.param("PS", PS, over=organ)
    X = m.transitions(
        "X", axis=sub, rates={"vascular->tissue": ps[organ], "tissue->vascular": ps[organ]}
    )
    m.transition(A, axis=sub, matrix=X, of=A)
    m.init(A, 1.0, where={sub: "vascular"})
    m.init(A, 0.0, where={sub: "tissue"})
    t = np.array([0, 0.5, 2.0])
    sol = m.solve((0, 2), times=t, **TIGHT)
    r = np.array(list(PS.values()))
    vascular = 0.5 * (1 + np.exp(-2 * r[None] * t[:, None]))
    np.testing.assert_allclose(sol["A"][..., 0], vascular, rtol=1e-9)
    np.testing.assert_allclose(sol["A"].sum(axis=2), 1.0, rtol=1e-12)


def test_sparse_jacobian_with_field_dependent_rates():
    """Rates that depend on another field cell by cell (a nutrient S(x)) and on a total:
    the sparse Jacobian (pattern from the model) equals the dense one."""
    m = tq.Model("pattern")
    x = m.axis("x", 0, 1, 6)
    s = m.axis("s", labels=["A", "D", "P"])
    B = m.field("B", axes=[x, s])
    S = m.field("S", axes=[x])
    m.integral(B, over=[x, s], name="N")
    Q = m.transitions(
        "Q", axis=s, rates={"A->D": "1/(1 + S)", "D->A": "S/(1 + S)", "D->P": "0.1*N"}
    )
    m.transition(B, axis=s, matrix=Q, of=B)
    m.transition(B, axis=s, matrix=Q, of="0.3*S*B")
    m.diffuse(B, axis=x, D=0.05)
    m.diffuse(S, axis=x, D=0.2)
    m.reaction(S, "-0.1*S*N")
    m.init(B, 1.0)
    m.init(S, 1.0)
    cm = m.compile()
    rng = np.random.default_rng(25)
    for _ in range(3):
        y = rng.uniform(0.1, 3.0, cm.n_states)
        np.testing.assert_allclose(
            cm.sparse_jacobian(0.0, y).toarray(), np.asarray(cm.jacobian(0.0, y)), atol=1e-13
        )


# --------------------------------------------------------------------------- refusals
def issues(m):
    return [str(i) for i in m.validate(raise_on_error=False).issues if i.level == "error"]


def test_a_rate_may_not_vary_along_its_own_axis():
    m = tq.Model("own")
    s = m.axis("s", labels=["S", "R"])
    B = m.field("B", axes=[s])
    Q = m.transitions("Q", axis=s, rates={"S->R": 0.1 * B})
    m.transition(B, axis=s, matrix=Q, of=B)
    m.init(B, 1.0)
    errs = issues(m)
    assert any("varies along ['s'], the axis it moves between" in e for e in errs), errs
    assert sum("varies along" in e for e in errs) == 1  # reported once


def test_a_rate_may_not_vary_over_an_axis_the_field_lacks():
    m = tq.Model("lacks")
    x = m.axis("x", 0, 1, 4)
    s = m.axis("s", labels=["S", "R"])
    B = m.field("B", axes=[s])
    Q = m.transitions("Q", axis=s, rates={"S->R": 0.1 * x})
    m.transition(B, axis=s, matrix=Q, of=B)
    m.init(B, 1.0)
    assert any("varies over ['x']" in e and "apart from ['s']" in e for e in issues(m))


def test_joint_transition_misuse_is_explained():
    m = tq.Model("bad")
    r1 = m.axis("r1", labels=["S", "R"])
    r2 = m.axis("r2", labels=["S", "R"])
    x = m.axis("x", 0, 1, 3)
    with pytest.raises(ValueError, match="one category of each of"):
        m.transitions("J", axis=[r1, r2], rates={"S->R,R": 1.0})
    with pytest.raises(ValueError, match="two or more different"):
        m.transitions("J", axis=[r1, r1], rates={"S,S->R,R": 1.0})
    with pytest.raises(ValueError, match="categorical axis is needed"):
        m.transitions("J", axis=[r1, x], rates={"S,S->R,R": 1.0})
    B = m.field("B", axes=[r1])
    J = m.transitions("J", axis=[r1, r2], rates={"S,S->R,R": 1.0, "S,Q->R,R": 1.0})
    m.transition(B, axis=[r1, r2], matrix=J, of=B)
    m.init(B, 1.0)
    errs = issues(m)
    assert any("axis 'r2' has no category 'Q'" in e for e in errs)
    assert any("needs categorical axes of 'B'" in e for e in errs)


# --------------------------------------------------------------------------- joint
def joint_model():
    m = tq.Model("joint")
    r1 = m.axis("r1", labels=["S", "R"])
    r2 = m.axis("r2", labels=["S", "R"])
    B = m.field("B", axes=[r1, r2])
    M1 = m.transitions("M1", axis=r1, rates={"S->R": 0.2})
    J = m.transitions("J", axis=[r1, r2], rates={"S,S->R,R": 0.05, "R,R->S,S": 0.01})
    m.transition(B, axis=r1, matrix=M1, of=B)
    m.transition(B, axis=[r1, r2], matrix=J, of=B)
    m.init(B, 1.0, where={r1: "S", r2: "S"})
    m.init(B, 0.0, where={r1: "R"})
    m.init(B, 0.0, where={r1: "S", r2: "R"})
    return m


def test_joint_transition_is_the_matrix_exponential():
    """Per-axis switching on r1 plus a joint move (S,S) <-> (R,R): the 4x4 generator written
    by hand (flattened (r1, r2) = SS, SR, RS, RR) gives the exact solution."""
    G = np.zeros((4, 4))
    for o, d, r in [(0, 2, 0.2), (1, 3, 0.2), (0, 3, 0.05), (3, 0, 0.01)]:
        G[d, o] += r
        G[o, o] -= r
    t = np.array([0, 1, 5, 20.0])
    sol = joint_model().solve((0, 20), times=t, **TIGHT)
    exact = np.array([expm(G * ti) @ [1, 0, 0, 0] for ti in t])
    np.testing.assert_allclose(sol["B"].reshape(len(t), 4), exact, rtol=1e-8, atol=1e-13)


def test_joint_transitions_round_trip_and_older_files_load():
    m = joint_model()
    ir = m.to_ir()
    assert ir.ir_version == IR_VERSION
    assert ir.transitions[1].axis == ("r1", "r2")
    assert ir.transitions[1].rates[0][:2] == (("S", "S"), ("R", "R"))
    back = ModelIR.from_json(ir.to_json())
    assert back == ir and tq.Model.from_ir(back).to_ir() == ir
    old = toy_phage_model(n_z=3, n_a=4).to_ir().to_dict()  # a model without joint terms
    old["ir_version"] = "0.3"
    assert (
        ModelIR.from_dict(old).transitions
        == ModelIR.from_dict({**old, "ir_version": "0.4"}).transitions
    )


# --------------------------------------------------------------------------- two phages
def two_phage(mu=1e-6, mux=1e-8):
    """B[r1, r2]: susceptibility to phage 1 and phage 2. Logistic division flux; each phage
    adsorbs only along its own axis; mutation per axis plus a joint double mutation, all on
    the division flux (daughters change category, at the parent's rate)."""
    m = tq.Model("two_phage")
    r1 = m.axis("r1", labels=["S", "R"])
    r2 = m.axis("r2", labels=["S", "R"])
    B = m.field("B", axes=[r1, r2])
    P1, P2 = m.compartment("P1", init=1e4), m.compartment("P2", init=1e4)
    m.params(g=1.0, K=1e9, beta=50.0, dp=0.1, mu=mu, mux=mux)
    k1 = m.param("k1", {"S": 1e-9, "R": 0.0}, over=r1)
    k2 = m.param("k2", {"S": 2e-9, "R": 0.0}, over=r2)
    N = m.integral(B, over=[r1, r2], name="N")
    A1 = m.integral(k1[r1] * B, over=[r1, r2], name="A1")  # adsorbing cells per phage
    A2 = m.integral(k2[r2] * B, over=[r1, r2], name="A2")
    div = m.define("div", "g*B*(1 - N/K)")
    m.reaction(B, div - k1[r1] * P1 * B - k2[r2] * P2 * B)
    for name, axis in [("M1", r1), ("M2", r2)]:
        M = m.transitions(name, axis=axis, rates={"S->R": "mu"})
        m.transition(B, axis=axis, matrix=M, of=div)
    X = m.transitions("X", axis=[r1, r2], rates={"S,S->R,R": "mux"})
    m.transition(B, axis=[r1, r2], matrix=X, of=div)
    m.ode(P1, "(beta - 1)*A1*P1 - dp*P1")
    m.ode(P2, "(beta - 1)*A2*P2 - dp*P2")
    m.init(B, 1e6, where={r1: "S", r2: "S"})
    m.init(B, 0.0, where={r1: "R"})
    m.init(B, 0.0, where={r1: "S", r2: "R"})
    return m, (N, A1, A2)


def two_phage_by_hand(t, mu=1e-6, mux=1e-8):
    """The same model as six hand-written ODEs (SS, SR, RS, RR, P1, P2)."""
    g, K, beta, dp = 1.0, 1e9, 50.0, 0.1
    k1 = np.array([1e-9, 1e-9, 0, 0])  # by r1: SS, SR are S for phage 1
    k2 = np.array([2e-9, 0, 2e-9, 0])

    def rhs(_, y):
        B, P1, P2 = y[:4], y[4], y[5]
        div = g * B * (1 - B.sum() / K)
        dB = div - k1 * P1 * B - k2 * P2 * B
        moves = [(0, 2, mu), (1, 3, mu), (0, 1, mu), (2, 3, mu), (0, 3, mux)]  # r1, r2, joint
        for o, d, r in moves:
            dB[d] += r * div[o]
            dB[o] -= r * div[o]
        A1, A2 = (k1 * B).sum(), (k2 * B).sum()
        return [*dB, (beta - 1) * A1 * P1 - dp * P1, (beta - 1) * A2 * P2 - dp * P2]

    y0 = [1e6, 0, 0, 0, 1e4, 1e4]
    return solve_ivp(rhs, (0, t[-1]), y0, t_eval=t, method="Radau", rtol=1e-12, atol=1e-10).y


def test_two_phage_toy_matches_the_hand_written_system():
    t = np.array([0, 4, 8, 12, 16, 24.0])
    m, _ = two_phage()
    sol = m.solve((0, 24), times=t, **TIGHT)
    ref = two_phage_by_hand(t)
    np.testing.assert_allclose(sol["B"].reshape(len(t), 4), ref[:4].T, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sol["P1"], ref[4], rtol=1e-6)
    np.testing.assert_allclose(sol["P2"], ref[5], rtol=1e-6)
    BRR = sol["B"][-1, 1, 1]
    assert BRR > 1e5  # the double mutant takes over once both phages have bloomed
    assert max(sol.ledger.check().values()) < 1e-8


def test_two_phage_toy_joint_mutation_is_what_seeds_the_double_mutant_early():
    """Without the joint term the double mutant needs two steps (rate ~ mu^2 t^2); with it,
    one (~ mux t): at 2 h the joint term dominates exactly as its rate says."""
    t = np.array([0, 2.0])
    with_x = two_phage(mux=1e-8)[0].solve((0, 2), times=t, **TIGHT)["B"][-1, 1, 1]
    without = two_phage(mux=0.0)[0].solve((0, 2), times=t, **TIGHT)["B"][-1, 1, 1]
    ref = two_phage_by_hand(t, mux=1e-8)[3, -1]
    assert with_x == pytest.approx(ref, rel=1e-6)
    assert with_x > 50 * without


def test_two_phage_sparse_jacobian():
    m, _ = two_phage()
    cm = m.compile()
    rng = np.random.default_rng(2)
    for _ in range(3):
        y = rng.uniform(0.0, 1e6, cm.n_states)
        np.testing.assert_allclose(
            cm.sparse_jacobian(0.0, y).toarray(),
            np.asarray(cm.jacobian(0.0, y)),
            rtol=1e-12,
            atol=1e-9,
        )


def test_transition_says_which_case_through_of():
    """One method for both cases (design decision 2026-10-22): ``of=`` the field itself is
    first-order switching, ``of=`` a production rate moves a share of it; ``of`` is required."""
    m = tq.Model("of")
    s = m.axis("s", labels=["S", "R"])
    o = m.axis("o", labels=["a", "b"])
    B = m.field("B", axes=[s, o], init=1.0)
    C = m.field("C", axes=[s], init=1.0)
    Q = m.transitions("Q", axis=s, rates={"S->R": 0.1})
    m.transition(B, axis=s, matrix=Q, of=B)
    m.transition(B.at(o="a"), axis=s, matrix=Q, of=B.at(o="a"))
    m.transition(B, axis=s, matrix=Q, of="B")
    m.transition(B, axis=s, matrix=Q, of=0.5 * B)
    kinds = [(t.kind, t.expr) for t in m.to_ir().terms]
    assert kinds == [("switch", "0")] * 3 + [("mutate", "0.5*B")]
    with pytest.raises(TypeError):
        m.transition(B, axis=s, matrix=Q)  # no default: the case must be said
    with pytest.raises(ValueError, match="of= is a slice of 'C'"):
        m.transition(B, axis=s, matrix=Q, of=C.at(s="S"))
    assert not hasattr(m, "mutate") and not hasattr(m, "switch")
