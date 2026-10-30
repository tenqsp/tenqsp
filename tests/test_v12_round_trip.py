# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""V12: API <-> PDE-notation round trip gives the same model.

`m.to_text()` writes any model in the notation; `tq.Model.from_text` reads it back; the two
descriptions are compared in meaning by `tenqsp.canonical.differences` (definitions inlined,
integrals named by content, the local terms of each place summed into one net rate).

The whole test suite can be run with every compiled model round-tripped:
``pytest --roundtrip`` (tests/conftest.py). Here: the comparison itself is checked to find
real differences, and a few models are round-tripped in the fast tier."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

import tenqsp as tq
from tenqsp.canonical import differences
from tenqsp.examples import phage_latency_model, toy_phage_model, toy_three_axis_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pbpk_models import pbpk  # noqa: E402
from test_gallery_phase2 import friberg  # noqa: E402
from test_transitions_phase2 import two_phage  # noqa: E402

MODELS = {
    "toy phage": toy_phage_model,
    "three axes": toy_three_axis_model,
    "phage latency": phage_latency_model,
    "Friberg": lambda: friberg(dose=1.8),
    "two phages": lambda: two_phage()[0],
    "PBPK": pbpk,
}


@pytest.mark.parametrize("name", list(MODELS))
def test_round_trip_gives_the_same_model(name):
    m = MODELS[name]()
    text = m.to_text()
    back = tq.Model.from_text(text)
    assert differences(m.to_ir(), back.to_ir()) == []
    again = back.to_text()
    assert tq.Model.from_text(again).to_text() == again  # a fixed point after one pass


def test_pbpk_transfers_are_written_as_fluxes():
    text = pbpk().to_text()
    assert "transfer" not in text
    assert "flux J1: Cart -> C[sub=vascular] = Q_art[organ]*Cart" in text
    assert "-sum_organ(J1)/V_art" in text.replace(" ", "")
    assert "d_t(C[organ=liver, sub=cellular]) = -(CLint/Kp_liver)*C/V[organ, sub]" in text


def test_the_comparison_finds_real_differences():
    m = toy_phage_model()
    ir = m.to_ir()
    assert differences(ir, ir) == []

    def changed(**kw):
        return differences(ir, replace(ir, **kw))

    params = tuple(replace(p, value=0.2) if p.name == "delta" else p for p in ir.params)
    assert any("delta" in d for d in changed(params=params))
    terms = tuple(replace(t, scheme="erlang") if t.kind == "advection" else t for t in ir.terms)
    assert changed(terms=terms)  # sharp -> erlang on the age axis
    terms = tuple(t for t in ir.terms if t.kind != "hazard")
    assert any("net rate" in d for d in changed(terms=terms))
    doses = tuple(replace(d, amounts=("2e9",)) for d in ir.doses)
    assert any("doses" in d for d in changed(doses=doses))
    defs = tuple(
        replace(d, expr=d.expr.replace("0.5*z", "0.6*z")) if d.name == "ads" else d
        for d in ir.definitions
    )
    assert changed(definitions=defs)  # inlined into the net rate, an integral, an inflow
    bounds = tuple(replace(b, side="lower") if b.kind == "outflow" else b for b in ir.boundaries)
    assert changed(boundaries=bounds)


def test_the_comparison_ignores_how_a_model_is_written():
    """Same model, other words: a hazard or a reaction, an integral's name, a zero start,
    0.5 or 1/2, a transfer out of the model or a loss term."""
    a = tq.Model("w")
    s = a.axis("s", labels=["S", "R"])
    B = a.field("B", axes=[s], weights="2")
    P = a.compartment("P")
    a.params(k=1.0, g=0.5)
    a.reaction(B, "g*B")
    a.hazard(B, "k*P")
    a.ode(P, a.integral(B, over=s, name="total") * 0.5)
    a.init(B, 1.0)
    a.init(P, 0.0)
    a.transfer(B.at(s="R"), to=None, rate="k")
    b = tq.Model.from_text(
        """
model w
axis s categorical [S, R]
param k=1, g=0.5
field B(s) weights=2
d_t(B) = g*B - k*P*B
d_t(B[s=R]) = -k*B/2
d_t(P) = sum_s(B)/2
B(0) = 1
"""
    )
    assert differences(a.to_ir(), b.to_ir()) == []


def test_slice_values_round_trip():
    m = tq.Model.from_text(
        """
axis organ categorical [gut, "small liver"]
axis sub categorical [v, c]
param V[organ, sub] = {gut: {v: 2, c: 4}, "small liver": {v: 1, c: 0.5}}
field C(organ, sub) weights=V[organ, sub]
d_t(C) = 0
liver = C[organ="small liver", sub=c]
observable effect = liver/(1 + liver)
observable gut_v = C[organ=gut, sub=v]
"""
    )
    back = tq.Model.from_text(m.to_text())
    assert differences(m.to_ir(), back.to_ir()) == []


def test_general_and_density_shapes_round_trip():
    m = tq.Model("shapes")
    x = m.axis("x", 0, 1, 8)
    y = m.axis("y", 0, 1, 4)
    u = m.field("u", axes=[x, y])
    v = m.field("v", axes=[x])
    m.hazard(u, 0.1)
    m.hazard(v, 0.1)
    m.init(u, total=3.0, shape=tq.exp(-x) * tq.normal(y, mean=0.5, sd=0.2))
    m.init(v, tq.normal(x, mean=0.5, sd=0.2))  # a density: the curve itself
    text = m.to_text()
    assert "3*shape(" in text.replace(".0", "") and "density(normal(" in text
    assert differences(m.to_ir(), tq.Model.from_text(text).to_ir()) == []


def test_the_comparison_finds_a_difference_confined_to_one_branch():
    """Random points in [0.3, 1.7] mostly miss z < 0.5; branches are compared one by one."""
    head = "axis z in [0, 1] n=10\nparam w=-1, D0=0.1\nfield p(z)\n"
    a = tq.Model.from_text(head + "d_t(p) = -d_z(((w if z < 0.5 else 2*w) - D0)*p)\n")
    b = tq.Model.from_text(head + "d_t(p) = -d_z((w if z < 0.5 else 2*w - D0)*p)\n")
    assert differences(a.to_ir(), b.to_ir())
    c = tq.Model.from_text(head + "d_t(p) = -d_z((w - D0 if z < 0.5 else 2*w - D0)*p)\n")
    assert differences(a.to_ir(), c.to_ir()) == []
