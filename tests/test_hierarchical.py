# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Step 2.6 building blocks: parameters over several axes, element weights,
compartment volumes, slices in operators, transfers. Oracles: values and exact solutions by
hand; amounts must be conserved between categories of different size."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import tenqsp as tq
from tenqsp.ir import IR_VERSION, IntegralSpec, ModelIR

TIGHT = {"rtol": 1e-10, "atol": 1e-12}


def errors(m):
    return [str(i) for i in m.validate(raise_on_error=False).issues if i.level == "error"]


# --------------------------------------------------------------------------- parameters
def test_parameter_over_two_axes():
    m = tq.Model("table")
    o = m.axis("o", labels=["a", "b"])
    s = m.axis("s", labels=["x", "y", "z"])
    V = m.param("V", {"a": {"x": 1, "y": 2, "z": 3}, "b": {"x": 4, "y": 5, "z": 6}}, over=[o, s])
    F = m.field("F", axes=[s, o])  # a different axis order than the parameter
    m.reaction(F, V[o, s])
    m.init(F, 0.0)
    cm = m.compile()
    np.testing.assert_array_equal(
        np.reshape(cm.rhs(0, np.zeros(6)), (3, 2)), [[1, 4], [2, 5], [3, 6]]
    )
    th = cm.theta({"V": {"b": {"y": 50}}})
    np.testing.assert_array_equal(
        np.reshape(cm.rhs(0, np.zeros(6), th), (3, 2)), [[1, 4], [2, 50], [3, 6]]
    )
    assert m.param("W", np.arange(6.0), over=[o, s]) is not None  # a flat table, first axis slowest
    with pytest.raises(ValueError, match=r"needs 6 values \(2 \(o\) x 3 \(s\)\)"):
        m.param("U", [1, 2], over=[o, s])
    with pytest.raises(ValueError, match="missing"):
        m.param("U2", {"a": {"x": 1}, "b": {"x": 1, "y": 1, "z": 1}}, over=[o, s])
    with pytest.raises(tq.FormulaError, match=r"write V\[o, s\]"):
        V[s, o]
    m.reaction(F, "V[s, o]")
    assert any("write 'V[o, s]'" in e for e in errors(m))


# --------------------------------------------------------------------------- weights
def two_spaces(V1=1.0, V2=4.0, k=2.0):
    m = tq.Model("spaces")
    s = m.axis("s", labels=["one", "two"])
    V = m.param("V", {"one": V1, "two": V2}, over=s)
    m.params(k=k)
    C = m.field("C", axes=[s], weights=V)
    X = m.transitions("X", axis=s, rates={"one->two": "k", "two->one": "k"})
    m.transition(C, axis=s, matrix=X, of=C)
    m.init(C, 1.0, where={s: "one"})
    m.init(C, 0.0, where={s: "two"})
    return m


def test_weighted_switch_moves_amounts_at_a_clearance():
    """Exchange k*(c1 - c2) between volumes 1 and 4: the concentrations meet at the mean
    weighted by volume (0.2), approaching at rate k(1/V1 + 1/V2); the amount stays 1."""
    V1, V2, k = 1.0, 4.0, 2.0
    t = np.array([0, 0.1, 0.5, 3.0])
    sol = two_spaces(V1, V2, k).solve((0, 3), times=t, **TIGHT)
    lam = k * (1 / V1 + 1 / V2)
    c_eq = V1 / (V1 + V2)
    np.testing.assert_allclose(sol["C"][:, 0], c_eq + (1 - c_eq) * np.exp(-lam * t), rtol=1e-8)
    np.testing.assert_allclose(sol["C"][:, 1], c_eq * (1 - np.exp(-lam * t)), rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(sol.ledger.amount["C"], 1.0, rtol=1e-12)
    assert max(sol.ledger.check().values()) < 1e-12


def test_weighted_mutate_moves_a_fraction_of_the_production():
    """Daughters are a fraction mu of what the origin produces (an amount), so the
    destination's concentration rises by mu*g*c_S*V_S/V_R."""
    m = tq.Model("mut")
    s = m.axis("s", labels=["S", "R"])
    V = m.param("V", {"S": 1.0, "R": 2.0}, over=s)
    g = m.param("g", {"S": 1.0, "R": 0.0}, over=s)
    B = m.field("B", axes=[s], weights=V)
    M = m.transitions("M", axis=s, rates={"S->R": 0.1})
    m.reaction(B, g[s] * B)
    m.transition(B, axis=s, matrix=M, of=g[s] * B)
    m.init(B, 3.0)
    np.testing.assert_allclose(
        m.compile().rhs(0, np.array([3.0, 0.0])), [3.0 * 0.9, 0.1 * 3.0 * 1 / 2]
    )


def test_weighted_initial_total_and_dose_into_one_element():
    m = tq.Model("dose")
    s = m.axis("s", labels=["a", "b"])
    V = m.param("V", {"a": 2.0, "b": 8.0}, over=s)
    C = m.field("C", axes=[s], weights=V)
    P = m.compartment("P", volume="Vp")
    m.params(Vp=5.0)
    m.hazard(C, 0.0)
    m.ode(P, 0 * P)
    m.init(C, total=10.0, shape=1.0)  # an amount, spread at equal concentration
    m.init(P, 0.0)
    m.dose(C.at(s="b"), times=[1], amounts=[16.0])
    m.dose(P, times=[1], amounts=[10.0])
    sol = m.solve((0, 2), times=[0, 2])
    np.testing.assert_allclose(sol["C"][0], [1.0, 1.0])  # 10 spread evenly: 10 / (2 + 8)
    np.testing.assert_allclose(sol["C"][1], [1.0, 1.0 + 16 / 8])
    assert sol["P"][1] == pytest.approx(2.0)  # 10 / 5
    assert sol.ledger.amount["P"][1] == pytest.approx(10.0)
    assert sol.ledger.amount["C"][1] == pytest.approx(26.0)


# --------------------------------------------------------------------------- slices
def slab():
    m = tq.Model("slab")
    o = m.axis("organ", labels=["gut", "liver"])
    s = m.axis("sub", labels=["v", "c"])
    V = m.param("V", {"gut": {"v": 1, "c": 3}, "liver": {"v": 2, "c": 5}}, over=[o, s])
    C = m.field("C", axes=[o, s], weights=V)
    m.init(C, 1.0)
    return m, o, s, C


def test_reaction_and_hazard_on_a_slice_act_only_there():
    m, o, s, C = slab()
    m.reaction(C.at(sub="v"), "2.0")
    m.hazard(C.at(organ="liver", sub="c"), 0.5)
    np.testing.assert_array_equal(
        np.reshape(m.compile().rhs(0, np.ones(4)), (2, 2)), [[2, 0], [2, -0.5]]
    )


def test_integral_and_observable_of_a_slice():
    m, o, s, C = slab()
    m.hazard(C, 0.0)
    vasc = m.integral(C.at(sub="v"), over=o, name="vascular_amount")
    m.observable("vasc", vasc)
    m.observable("liver_c", C.at(organ="liver", sub="c"))
    m.observable("amount", m.integral(C, over=[o, s], name="all"))
    sol = m.solve((0, 1), times=[1])
    assert sol["vasc"].item() == pytest.approx(1 + 2)  # concentration 1 x vascular volumes
    assert sol["liver_c"].item() == pytest.approx(1.0)
    assert sol["amount"].item() == pytest.approx(1 + 3 + 2 + 5)


def test_flow_between_organs_only_in_the_vascular_slice():
    m, o, s, C = slab()
    G = m.transitions("G", axis=o, rates={"gut->liver": 4.0})
    m.transition(C.at(sub="v"), axis=o, matrix=G, of=C.at(sub="v"))
    du = np.reshape(m.compile().rhs(0, np.ones(4)), (2, 2))
    np.testing.assert_allclose(du, [[-4 / 1, 0], [4 / 2, 0]])  # 4 x c_gut,v moved; per volume


# --------------------------------------------------------------------------- transfers
def test_transfers_between_compartments_and_slices():
    """Arterial supply Q[organ] x C_art into each organ's vascular space, venous return into
    C_ven, and elimination from one cell: rates by hand, books balanced on both sides."""
    m, o, s, C = slab()
    m.params(Va=2.0, Vv=4.0, CL=3.0)
    Q = m.param("Q", {"gut": 1.0, "liver": 2.0}, over=o)
    A = m.compartment("A", volume="Va")
    Vn = m.compartment("Vn", volume="Vv")
    m.transfer(A, to=C.at(sub="v"), rate=Q[o])
    m.transfer(C.at(sub="v"), to=Vn, rate=Q[o])
    m.transfer(C.at(organ="liver", sub="c"), to=None, rate="CL")
    m.init(A, 6.0)
    m.init(Vn, 0.0)
    cm = m.compile()
    y = np.array([1.0, 1.0, 2.0, 1.0, 6.0, 0.0])  # C[gut, v], C[gut, c], C[liver, v], ...
    du = np.asarray(cm.rhs(0, y))
    # gut v: +1*6/1 - 1*1/1 ; liver v: (+2*6 - 2*2)/2 ; liver c: -3*1/5
    np.testing.assert_allclose(du[:4], [6 - 1, 0, (12 - 4) / 2, -3 / 5])
    np.testing.assert_allclose(du[4:], [-(1 + 2) * 6 / 2, (1 * 1 + 2 * 2) / 4])
    led = cm.ledger_rates(0, y)
    assert led["A"]["transfer to C.at(sub=v)"] == pytest.approx(-18.0)
    assert led["C"]["transfer from A"] == pytest.approx(18.0)
    assert led["C"]["transfer to elimination"] == pytest.approx(-3.0)
    assert led["Vn"]["transfer from C.at(sub=v)"] == pytest.approx(5.0)
    sol = m.solve((0, 5), times=[5], **TIGHT)
    assert max(sol.ledger.check().values()) < 1e-11
    rng = np.random.default_rng(26)
    y = rng.uniform(0.1, 2.0, cm.n_states)
    np.testing.assert_allclose(
        cm.sparse_jacobian(0.0, y).toarray(), np.asarray(cm.jacobian(0.0, y)), atol=1e-13
    )


def test_slice_to_slice_transfer_element_by_element():
    m = tq.Model("two_fields")
    o = m.axis("organ", labels=["a", "b"])
    s = m.axis("sub", labels=["v", "c"])
    C = m.field("C", axes=[o, s])
    E = m.field("E", axes=[o])  # e.g. an epithelial lining fluid per organ (composition)
    k = m.param("k", {"a": 1.0, "b": 3.0}, over=o)
    m.transfer(C.at(sub="c"), to=E, rate=k[o])
    m.init(C, 2.0)
    m.init(E, 0.0)
    du = np.asarray(m.compile().rhs(0, np.array([2.0, 2.0, 2.0, 2.0, 0.0, 0.0])))
    np.testing.assert_allclose(du, [0, -2, 0, -6, 2, 6])


# --------------------------------------------------------------------------- refusals
def test_refusals_are_explained():
    m, o, s, C = slab()
    x = m.axis("x", 0, 1, 3)
    B = m.field("B", axes=[x])
    P = m.compartment("P")
    E = m.field("E", axes=[s])
    m.init(B, 0.0)
    m.init(P, 0.0)
    m.init(E, 0.0)
    m.transfer(P, to=B, rate=1.0)  # a continuous axis
    m.transfer(C.at(sub="v"), to=E, rate=1.0)  # free axes differ: organ vs sub
    m.transfer(P, to=C.at(sub="v"), rate=x)  # rate over an axis that is not free
    m.transition(
        C.at(organ="gut"),
        axis=o,
        matrix=m.transitions("G", axis=o, rates={"gut->liver": 1}),
        of=C.at(organ="gut"),
    )
    m.integral(C, over=o, name="partial")  # weights vary over sub too
    m.param("Vbad", {"v": 1.0, "c": 0.0}, over=s)
    m.field("D", axes=[s], weights=m["Vbad"])
    m.reaction(C.at(sub="q"), 1.0)
    errs = errors(m)
    expected = [
        "has continuous axes ['x']",
        "leave different axes free (['organ'] and ['sub'])",
        "varies over ['x']",
        "the slice fixes ['organ'], the axis it moves along",
        "integrate over (or fix with a slice) ['sub'] too",
        "must be > 0 everywhere (found 0)",
        "'q' is not a category of 'sub'",
    ]
    for text in expected:
        assert any(text in e for e in errs), (text, errs)
    with pytest.raises(ValueError, match="dose into one element"):
        m.dose(C, times=[0], amounts=[1])
    m.dose(C.at(organ="gut"), times=[0], amounts=[1])
    assert any("fix every axis of 'C' (also ['sub'])" in e for e in errors(m))


def test_round_trip_and_older_files():
    m, o, s, C = slab()
    m.compartment("P", volume="2.0", init=0.0)
    m.reaction(C.at(sub="v"), "-C")
    m.transfer(C.at(sub="c"), to=m["P"], rate=1.0)
    m.integral(C.at(sub="v"), over=o, name="iv")
    m.observable("cv", C.at(organ="gut", sub="v"))
    m.dose(C.at(organ="gut", sub="v"), times=[1], amounts=[1.0])
    ir = m.to_ir()
    assert ir.ir_version == IR_VERSION
    back = ModelIR.from_json(ir.to_json())
    assert back == ir and tq.Model.from_ir(back).to_ir() == ir
    old = two_spaces().to_ir().to_dict()
    for f in old["fields"]:
        f.pop("weights")
    old["ir_version"] = "0.4"
    old.pop("transfers")
    assert ModelIR.from_dict(old).fields[0].weights is None


def test_ir_0_6_slices_an_integrand_formula_and_0_5_files_load():
    """IR 0.6: the value of a formula on a slice (the notation's C[organ=liver] read in a
    formula) is an integral over no axes whose `where` slices a formula of fields."""
    m, o, s, C = slab()
    m.compartment("Q", init=0.0)
    m.ode(m["Q"], "0")
    spec = IntegralSpec("C_gut_v", "C/(V[organ, sub])", (), where=(("organ", "gut"), ("sub", "v")))
    ir = replace(m.to_ir(), integrals=(*m.to_ir().integrals, spec))
    tq.validate(ir)  # accepted
    bad = replace(spec, integrand="2*Q")
    with pytest.raises(tq.ModelError, match="needs a field"):
        tq.validate(replace(m.to_ir(), integrals=(bad,)))
    old = m.to_ir().to_dict()
    old["ir_version"] = "0.5"
    assert ModelIR.from_dict(old) == m.to_ir()
