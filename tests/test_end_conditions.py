# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Boundary conditions as written in papers, and fields read at an end (IR 0.10; decided
2026-10-28).

``m.at(D, z=0, flux=...)``: the flux of D along z at that end equals the formula, in which D
means its value there (``-Dz*dD/dz = alpha*B - beta*D`` at z = 0). ``D(z="lower")`` in any
other equation reads that value. Oracles:

- dormancy entry and exit through one end of a diffusing field, against the same finite
  volumes written out by hand (half-cell balance at the end), and the total conserved;
- infection age with cells entering at l = 0 and bursting at l = 1, against the Erlang
  transit chain (Savic), the burst read as ``b*v*I(l=1)``;
- an exit from the end of one field into another field, element by element along a shared
  axis, conserved cell by cell;
- the new conditions against the older statements they replace (inflow, robin with a
  compartment, dirichlet, zero flux): the same numbers.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import tenqsp as tq
from tenqsp.canonical import differences
from tenqsp.ir import ModelIR
from tenqsp.validate import ModelError

TIGHT = {"rtol": 1e-10, "atol": 1e-13}


# ----------------------------------------------------------------------------- dormancy
def dormancy(n=8, *, new=True):
    """Active cells B enter dormancy at the shallow end of D(z) and wake from there."""
    m = tq.Model("dormancy")
    z = m.axis("z", 0, 1, n)
    a, h, Dz = m.params(a=0.6, h=0.9, Dz=0.05)
    B = m.compartment("B")
    D = m.field("D", axes=[z])
    m.diffuse(D, axis=z, D=Dz)
    m.ode(B, -a * B + h * D(z="lower"))
    m.at(D, z=0, flux=a * B - h * D)
    m.init(B, 1.0)
    m.init(D, 0)
    m.observable("dormant", m.integral(D, over=z, name="D_tot"))
    return m


def dormancy_by_hand(t, n=8, a=0.6, h=0.9, Dz=0.05):
    dz = 1.0 / n
    g = Dz / (0.5 * dz)

    def f(_, y):
        B, u = y[0], y[1:]
        cf = (a * B + g * u[0]) / (g + h)  # the value at z = 0 from the half-cell balance
        J = np.zeros(n + 1)
        J[0] = a * B - h * cf
        J[1:n] = -Dz * (u[1:] - u[:-1]) / dz
        return np.concatenate([[-a * B + h * cf], -(J[1:] - J[:-1]) / dz])

    y0 = np.zeros(n + 1)
    y0[0] = 1.0
    return solve_ivp(f, (0, t[-1]), y0, t_eval=t, method="Radau", rtol=1e-11, atol=1e-14).y


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_entry_and_exit_through_one_end(backend):
    t = np.linspace(0, 10, 21)
    sol = dormancy().solve((0, 10), times=t, backend=backend, **TIGHT)
    ref = dormancy_by_hand(t)
    np.testing.assert_allclose(sol["B"], ref[0], atol=1e-8)
    np.testing.assert_allclose(np.asarray(sol["D"]).T, ref[1:], atol=1e-8)
    total = np.asarray(sol["B"]) + np.asarray(sol["dormant"])
    np.testing.assert_allclose(total, 1.0, atol=1e-9)
    assert max(sol.ledger.check().values()) < 1e-9


@pytest.mark.parametrize("build", ["dormancy", "infection", "hibernation"])
def test_jacobian_with_values_at_the_ends(build):
    cm = globals()[build]().compile()
    y = np.linspace(0.3, 1.1, cm.n_states)
    th = cm.theta()
    dense = np.asarray(cm.jacobian(0.0, y, th))
    sparse = cm.sparse_jacobian(0.0, y, th).toarray()
    np.testing.assert_allclose(sparse, dense, atol=1e-12)


# ----------------------------------------------------------------------------- infection age
def infection(n=10, *, new=True):
    """B is infected at rate k; infected cells age along l at speed v and burst at l = 1,
    releasing b phage each."""
    m = tq.Model("infection")
    l_ = m.axis("l", 0, 1, n, kind="one_way")
    k, v, b = m.params(k=0.8, v=2.0, b=50.0)
    B = m.compartment("B")
    P = m.compartment("P")
    I = m.field("I", axes=[l_])
    m.advect(I, axis=l_, velocity=v)
    m.ode(B, -k * B)
    if new:
        m.at(I, l=0, flux=k * B)
        m.at(I, l=1, outflow=True)
    else:
        m.inflow(I, axis=l_, at="lower", flux=k * B)
        m.outflow(I, axis=l_, at="upper")
    m.ode(P, b * v * I(l="upper"))
    m.init(B, 1.0)
    m.init(P, 0)
    m.init(I, 0)
    return m


def chain_by_hand(t, n=10, k=0.8, v=2.0, b=50.0):
    r = v * n  # cells of width 1/n crossed at speed v: the transit rate

    def f(_, y):
        B, A = y[0], y[1:-1]
        dA = -r * A
        dA[0] += k * B
        dA[1:] += r * A[:-1]
        return np.concatenate([[-k * B], dA, [b * r * A[-1]]])

    y0 = np.zeros(n + 2)
    y0[0] = 1.0
    return solve_ivp(f, (0, t[-1]), y0, t_eval=t, method="Radau", rtol=1e-11, atol=1e-14).y


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_burst_read_at_the_end_of_infection_age(backend):
    t = np.linspace(0, 6, 25)
    sol = infection().solve((0, 6), times=t, backend=backend, **TIGHT)
    ref = chain_by_hand(t)
    np.testing.assert_allclose(sol["P"], ref[-1], rtol=1e-7, atol=1e-8)
    old = infection(new=False).solve((0, 6), times=t, backend=backend, **TIGHT)
    np.testing.assert_allclose(sol["P"], old["P"], rtol=1e-12, atol=1e-12)


def test_the_value_at_an_outflow_end_is_the_edge_cell():
    cm = infection().compile()
    y = np.arange(1.0, cm.n_states + 1.0)
    ev, state = cm._evaluator(0.0, y, cm.theta(), cm.geom)
    np.testing.assert_allclose(
        np.ravel(ev.name("I(l=upper)")), np.ravel(state["I"])[-1], rtol=0, atol=0
    )


# ----------------------------------------------------------------------------- field to field
def hibernation(n_l=4, n_z=6):
    """X(l, z) wakes through its shallow end z = 0 into Y(l), keeping l."""
    m = tq.Model("wake")
    l_ = m.axis("l", 0, 1, n_l)
    z = m.axis("z", 0, 1, n_z)
    h, Dz = m.params(h=1.3, Dz=0.02)
    X = m.field("X", axes=[l_, z])
    Y = m.field("Y", axes=[l_])
    m.diffuse(X, axis=z, D=Dz)
    m.at(X, z=0, flux=-h * X)  # leaves at rate h times its value at z = 0
    m.reaction(Y, h * X(z="lower"))
    m.init(X, (1 + l_) * tq.exp(-3 * z))
    m.init(Y, 0)
    return m


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_an_end_feeds_another_field_element_by_element(backend):
    t = np.linspace(0, 5, 11)
    sol = hibernation().solve((0, 5), times=t, backend=backend, **TIGHT)
    X, Y = np.asarray(sol["X"]), np.asarray(sol["Y"])  # (t, l, z), (t, l)
    per_l = X.sum(axis=2) / 6 + Y  # amount at each l: X over z (widths 1/6) plus Y
    np.testing.assert_allclose(per_l, per_l[:1].repeat(len(t), axis=0), atol=1e-10)
    assert Y[-1].min() > 0.05  # something did move


# ----------------------------------------------------------------------------- old and new
BIOFILM = """
model biofilm
axis z in [0, 1] n=20 kind=spatial
param De=0.1, kr=0.4, h=0.5, V=2, R=1, CL=0.3
field C(z)
compartment M
d_t(C) = d_z(De*d_z(C)) - kr*C
d_t(M) = R - CL*M/V
"""


def test_exchange_written_as_a_flux_equals_robin():
    old = tq.Model.from_text(BIOFILM + "robin C at z=upper: h*(M/V - C) with M\n")
    m = tq.Model("biofilm_new")
    z = m.axis("z", 0, 1, 20, kind="spatial")
    De, kr, h, V, R, CL = m.params(De=0.1, kr=0.4, h=0.5, V=2.0, R=1.0, CL=0.3)
    C = m.field("C", axes=[z])
    M = m.compartment("M")
    m.diffuse(C, axis=z, D=De)
    m.reaction(C, -kr * C)
    m.at(C, z=1, flux=h * (C - M / V))  # leaves at the surface
    m.ode(M, R - CL * M / V + h * (C(z="upper") - M / V))
    m.init(C, 0)
    m.init(M, 0)
    t = np.linspace(0, 20, 41)
    a = old.solve((0, 20), times=t, **TIGHT)
    b = m.solve((0, 20), times=t, **TIGHT)
    np.testing.assert_allclose(b["M"], a["M"], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(b["C"], a["C"], rtol=1e-11, atol=1e-12)


def test_held_value_and_closed_end():
    def build(new):
        m = tq.Model("held")
        z = m.axis("z", 0, 1, 10)
        C = m.field("C", axes=[z])
        m.diffuse(C, axis=z, D=0.2)
        if new:
            m.at(C, z=1, value=2.0)
            m.at(C, z=0, flux=0)
        else:
            m.boundary(C, axis=z, at="upper", kind="dirichlet", value=2.0)
        m.init(C, 0)
        return m

    t = np.linspace(0, 3, 7)
    np.testing.assert_allclose(
        build(True).solve((0, 3), times=t, **TIGHT)["C"],
        build(False).solve((0, 3), times=t, **TIGHT)["C"],
        rtol=1e-12,
        atol=1e-13,
    )


# ----------------------------------------------------------------------------- mistakes
def _model_with(condition, extra=None):
    m = tq.Model("bad")
    z = m.axis("z", 0, 1, 5)
    a = m.param("a", 0.5)
    B = m.compartment("B")
    D = m.field("D", axes=[z])
    E = m.field("E", axes=[z])
    m.diffuse(D, axis=z, D=0.1)
    m.reaction(E, -E)
    m.ode(B, -a * B)
    condition(m, D, E, B, a)
    if extra:
        extra(m, D, E, B, a)
    m.init(B, 1)
    m.init(D, 0)
    m.init(E, 0)
    return m


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        (lambda m, D, E, B, a: m.at(D, z=0, flux=a * D**2), "not linear in D"),
        (lambda m, D, E, B, a: m.at(E, z=0, flux=a * B), "does not move along 'z'"),
        (lambda m, D, E, B, a: m.at(D, z=0, flux=a * E), "varies along 'z'"),
        (lambda m, D, E, B, a: m.init(B, D(z="lower")), "parameters and coordinates only"),
    ],
)
def test_mistakes_are_explained(condition, message):
    with pytest.raises(ModelError, match=message):
        _model_with(condition).compile()


def test_an_end_is_named_by_its_coordinate():
    with pytest.raises(ValueError, match="not an end of 'z'"):
        _model_with(lambda m, D, E, B, a: m.at(D, z=0.5, flux=a * B))
    m = _model_with(lambda m, D, E, B, a: m.at(D, z=0.0, flux=a * B - D))
    assert any(b.kind == "flux" and b.side == "lower" for b in m.to_ir().boundaries)


def test_ends_that_need_each_other_are_refused():
    def both(m, D, E, B, a):
        m.diffuse(E, axis="z", D=0.1)
        m.at(D, z=0, flux=a * E(z="lower") - D)
        m.at(E, z=0, flux=a * D(z="lower") - E)

    with pytest.raises(ModelError, match="depend on each other in a circle"):
        _model_with(both).compile()


# ----------------------------------------------------------------------------- the notation
DORMANCY = """
model dormancy
axis z in [0, 1] n=8
param a=0.6, h=0.9, Dz=0.05
field D(z)
d_t(B) = -a*B + h*D(z=0)
d_t(D) = d_z(Dz*d_z(D))
    at z=0: -Dz*d_z(D) = a*B - h*D
B(0) = 1
observable dormant = int_z(D)
"""


def test_the_notation_reads_conditions_under_their_equation():
    t = np.linspace(0, 10, 21)
    a = tq.Model.from_text(DORMANCY).solve((0, 10), times=t, **TIGHT)
    b = dormancy().solve((0, 10), times=t, **TIGHT)
    np.testing.assert_allclose(a["B"], b["B"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(a["D"], b["D"], rtol=1e-12, atol=1e-14)


def test_written_back_and_read_again():
    m = tq.Model.from_text(DORMANCY)
    text = m.to_text()
    assert "    at z=0: -Dz*d_z(D) = a*B - h*D" in text
    assert differences(m.to_ir(), tq.Model.from_text(text).to_ir()) == []
    assert ModelIR.from_json(m.to_ir().to_json()) == m.to_ir()
    for build in (dormancy, infection, hibernation):  # Python models, printed and read
        ir = build().to_ir()
        assert differences(ir, tq.Model.from_text(build().to_text()).to_ir()) == []


def test_latex_shows_the_condition_and_the_value_at_the_end():
    tex = tq.Model.from_text(DORMANCY).to_latex()
    assert r"D\big|_{z=0}" in tex
    assert r"\text{at } z=0" in tex


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (DORMANCY.replace("-Dz*d_z(D) =", "-2*Dz*d_z(D) ="), "the left side is the flux of D"),
        (DORMANCY.replace("    at z=0", "B(0) = 1\nat z=0"), "belongs to an equation"),
        (DORMANCY.replace("at z=0:", "at z=0.5:"), "not an end of 'z'"),
        (DORMANCY.replace("= a*B - h*D\n", "= a*B - h*D^2\n"), "not linear in D"),
        (DORMANCY.replace("-Dz*d_z(D) = a*B - h*D", "d_z(D) = a*B"), "write a gradient"),
    ],
    ids=["wrong flux", "no equation above", "not an end", "nonlinear", "gradient"],
)
def test_notation_mistakes(text, message):
    with pytest.raises(tq.notation.NotationError, match=message):
        tq.Model.from_text(text)


def test_other_condition_forms():
    held = tq.Model.from_text(
        "axis z in [0, 1] n=10\nfield C(z)\nd_t(C) = d_z(0.2*d_z(C))\n"
        "    at z=1: C = 2\n    at z=0: d_z(C) = 0\n"
    )
    kinds = {b.side: b.kind for b in held.to_ir().boundaries}
    assert kinds == {"upper": "dirichlet", "lower": "zero_flux"}
    age = tq.Model.from_text(
        "axis a in [0, 2] n=4 kind=one_way\nfield N(a)\nd_t(N) = -d_a(N)\n    at a=2: outflow\n"
        "N(0) = 1\n"
    )
    assert [b.kind for b in age.to_ir().boundaries] == ["outflow"]


def test_an_unstated_outflow_end_is_refused_but_older_files_still_load():
    text = "axis a in [0, 2] n=4 kind=one_way\nfield N(a)\nd_t(N) = -d_a(N)\nN(0) = 1\n"
    with pytest.raises(tq.notation.NotationError, match="say what happens there"):
        tq.Model.from_text(text)
    ir = tq.Model.from_text(text.replace("N(0) = 1", "at a=2: outflow\nN(0) = 1")).to_ir()
    d = json.loads(ir.to_json())
    d["ir_version"] = "0.9"
    d["boundaries"] = []  # a 0.9 file relied on the default outflow: it is written in
    assert ModelIR.from_dict(d) == ir
