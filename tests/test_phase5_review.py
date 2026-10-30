# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Phase 5 gate review: each defect found by the independent review, as a test with an exact
oracle (hand calculation or closed form)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import tenqsp as tq
from tenqsp.canonical import differences


def _decay():
    return tq.Model.from_text("param k = 1\ncompartment A\nd_t(A) = -k*A\nA(0) = 1\n", name="d")


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_a_batch_keeps_each_scenarios_own_output_times(backend):
    sim = _decay().simulator(outputs=["A"], backend=backend, rtol=1e-10, atol=1e-12)
    scen = [
        tq.Scenario(params={"k": 1.0}, times=[0.0, 1.0]),
        tq.Scenario(params={"k": 1.0}, times=[0.0, 3.0]),
        tq.Scenario(params={"k": 2.0}, times=[0.0, 3.0]),
    ]
    res = sim.batch(scen)
    got = [float(np.ravel(res[i]["A"])[-1]) for i in range(3)]
    assert got == pytest.approx([math.exp(-1), math.exp(-3), math.exp(-6)], rel=1e-5)


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_a_batch_keeps_each_doses_own_shape(backend):
    m = tq.Model("g")
    z = m.axis("z", 0, 1, 4)
    G = m.field("G", axes=[z])
    m.reaction(G, 0 * G)
    m.init(G, 0)
    sim = m.simulator(outputs=["G"], backend=backend)
    scen = [
        tq.Scenario(name=n, times=[0.0, 1.0], regimen=tq.bolus("G", 1.0, times=[0.5], shape=s))
        for n, s in (("low", tq.point(z, at=0.1)), ("high", tq.point(z, at=0.9)))
    ]
    res = sim.batch(scen)
    low, high = (np.asarray(res[i]["G"])[-1] for i in range(2))
    assert np.argmax(low) == 0 and np.argmax(high) == 3


def test_pre_run_waits_for_a_dose_with_a_lag():
    m = tq.Model("lag")
    A, R = m.compartment("A"), m.compartment("R")
    k, tl = m.params(k=1.0, tlag=10.0)
    m.ode(A, -k * A)
    m.ode(R, k * A)
    m.lag(A, tl)
    m.init(A, 0)
    m.init(R, 0)
    sim = m.simulator(outputs=["R"], rtol=1e-9, atol=1e-11)
    pre = sim.pre_run(tq.Scenario(regimen=tq.bolus("A", 100, times=[5.0])))
    assert pre.reached and float(pre.state["R"]) == pytest.approx(100, rel=1e-7)


def test_pre_run_steady_cycles_with_a_lag_longer_than_the_interval():
    m = tq.Model("lag2")
    A = m.compartment("A")
    k, tl = m.params(k=0.5, tlag=6.0)
    m.ode(A, -k * A)
    m.lag(A, tl)
    m.init(A, 0)
    sim = m.simulator(outputs=["A"], rtol=1e-10, atol=1e-12)
    pre = sim.pre_run(tq.Scenario(regimen=tq.bolus("A", 100, start=0, every=4, n=5)))
    # doses arrive at 2 mod 4: the state at a dosing time is 2 h after an arrival
    exact = 100 * math.exp(-0.5 * 2) / (1 - math.exp(-0.5 * 4))
    assert pre.reached and float(pre.state["A"]) == pytest.approx(exact, rel=1e-6)


def test_pre_run_keeps_the_clock_running():
    m = tq.Model.from_text("param p = 2\ncompartment R\nd_t(R) = p if t < 3 else 0\nR(0) = 0\n")
    sim = m.simulator(outputs=["R"], rtol=1e-9, atol=1e-11)
    assert float(sim.pre_run(tq.Scenario()).state["R"]) == pytest.approx(6, rel=1e-8)
    assert float(sim.pre_run(tq.Scenario(start=1.0)).state["R"]) == pytest.approx(4, rel=1e-8)


def _sealed_channel():
    m = tq.Model("ch")
    z = m.axis("z", 0, 1, 8)
    C = m.field("C", axes=[z])
    m.advect(C, axis=z, velocity=1.0)
    m.at(C, z=0, flux=0)
    m.at(C, z=1, flux=0)
    m.init(C, 1.0)
    m.observable("tot", m.integral(C, over=z, name="tot_"))
    return m


def test_a_flux_condition_at_unit_speed_survives_the_round_trip():
    m = _sealed_channel()
    back = tq.Model.from_text(m.to_text())
    assert differences(m.to_ir(), back.to_ir()) == []
    sol = back.solve((0, 2), times=[0, 2], rtol=1e-10, atol=1e-12)
    assert np.ravel(sol["tot"]) == pytest.approx([1.0, 1.0], abs=1e-9)


def test_conditions_of_a_field_with_only_sliced_equations_are_written():
    m = tq.Model("s")
    s = m.axis("s", labels=["a", "b"])
    z = m.axis("z", 0, 1, 4)
    C = m.field("C", axes=[s, z])
    m.reaction(C.at(s="a"), -0.1 * C)
    m.at(C, z=1, value=1)
    m.init(C, 1)
    assert differences(m.to_ir(), tq.Model.from_text(m.to_text()).to_ir()) == []


@pytest.mark.parametrize("k", [0.5, 0.99, 1.0, 1.01, 2.0])
def test_a_flow_leaving_through_a_flux_condition_carries_the_edge_value(k):
    """Pure flow at speed 1 out through z=1: what leaves is the edge cell's value times 1,
    whatever k in the condition (which only sets the value read there, k*C = 1*edge)."""
    m = tq.Model("o")
    z = m.axis("z", 0, 1, 10)
    C = m.field("C", axes=[z])
    kk = m.param("k", k)
    m.advect(C, axis=z, velocity=1.0)
    m.at(C, z=0, flux=0)
    m.at(C, z=1, flux=kk * C)
    m.init(C, 1.0)
    m.observable("tot", m.integral(C, over=z, name="tot_"))
    m.observable("cend", C(z="upper"))
    sol = m.solve((0, 0.5), times=[0, 0.5], rtol=1e-10, atol=1e-12)
    edge = float(np.asarray(sol["C"])[-1][-1])
    # with zero inflow, 10 upwind cells: the edge is an Erlang(10) survival at t = 0.5
    rate = 10.0
    survive = sum(math.exp(-rate * 0.5) * (rate * 0.5) ** j / math.factorial(j) for j in range(10))
    assert edge == pytest.approx(survive, rel=1e-7)
    assert float(np.ravel(sol["cend"])[-1]) == pytest.approx(edge / k, rel=1e-7)
    assert "carries out the edge cell's value" in " ".join(m.explain().split())


def test_an_integral_over_categories_weighs_end_values_like_the_field():
    m = tq.Model("ex")
    s = m.axis("s", labels=["a", "b"])
    z = m.axis("z", 0, 1, 4)
    w = m.param("w", [1.0, 3.0], over=s)
    h = m.param("h", 0.5)
    D = m.field("D", axes=[s, z], weights=w[s])
    B = m.compartment("B")
    m.diffuse(D, axis=z, D=0.1)
    m.at(D, z=0, flux=-h * D)
    m.ode(B, h * m.integral(D(z="lower"), over=s, name="out"))
    m.init(D, 1.0)
    m.init(B, 0)
    sol = m.solve((0, 2), times=[0, 2], rtol=1e-10, atol=1e-12)
    assert max(sol.ledger.check().values()) < 1e-10
    total_D = float((np.asarray(sol["D"])[-1] * np.array([[1.0], [3.0]]) * 0.25).sum())
    assert total_D + float(np.ravel(sol["B"])[-1]) == pytest.approx(4.0, rel=1e-9)


def test_circles_of_end_values_through_integrals_and_transport_are_refused():
    m = tq.Model("cyc")
    s = m.axis("s", labels=["a", "b"])
    z = m.axis("z", 0, 1, 4)
    D, H = m.field("D", axes=[z]), m.field("H", axes=[s, z])
    m.diffuse(D, axis=z, D=0.1)
    m.diffuse(H, axis=z, D=0.1)
    m.at(D, z=0, flux=0.2 * m.integral(H(z="lower"), over=s, name="Hl") - 0.1 * D)
    m.at(H, z=0, flux=0.3 * D(z="lower") - 0.1 * H)
    m.init(D, 1)
    m.init(H, 1)
    with pytest.raises(tq.ModelError, match="circle"):
        m.validate()
    m = tq.Model("self")
    z = m.axis("z", 0, 1, 4)
    C = m.field("C", axes=[z])
    m.advect(C, axis=z, velocity=0.5 + 0.1 * C(z="upper"))
    m.at(C, z=0, flux=0)
    m.at(C, z=1, outflow=True)
    m.init(C, 1)
    with pytest.raises(tq.ModelError, match="would need itself"):
        m.validate()


def test_explain_compares_amounts_only_between_places_measured_alike():
    m = tq.Model("spread")
    z = m.axis("z", 0, 2, 4)
    k = m.param("k", 0.3)
    C, F = m.compartment("C"), m.field("F", axes=[z])
    m.ode(C, -k * C)
    m.reaction(F, k * C)  # per unit z: F gains 2*k*C in all
    m.init(C, 1)
    m.init(F, 0)
    assert "Amounts balance" not in m.explain()


def test_an_end_is_found_by_its_coordinate_up_to_rounding_only():
    m = tq.Model("geo")
    z = m.axis("z", 0.1, 0.7, 7, spacing="geometric", ratio=1.3)
    assert z.faces[-1] == 0.7
    C, B = m.field("C", axes=[z]), m.compartment("B")
    m.diffuse(C, axis=z, D=0.1)
    m.at(C, z=0.7, flux=0.2 * C)
    m.ode(B, 0.2 * C(z=0.7))
    m.init(C, 1)
    m.init(B, 0)
    m.validate()
    m = tq.Model("tiny")
    x = m.axis("x", 0, 4e-9, 4)
    F = m.field("F", axes=[x])
    m.diffuse(F, axis=x, D=1e-18)
    with pytest.raises(ValueError, match="not an end"):
        m.at(F, x=2e-9, flux=1.0)


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_a_dose_shape_off_its_axis_is_refused_with_a_clear_message(backend):
    m = tq.Model("p")
    z = m.axis("z", 0, 1, 4)
    G = m.field("G", axes=[z])
    m.diffuse(G, axis=z, D=0.1)
    m.init(G, 0)
    m.dose(G, times=[0.5], amounts=3.0, shape=tq.point(z, at=1.5))
    with pytest.raises(ValueError, match="outside the axis z"):
        m.solve((0, 1), times=[1.0], backend=backend)
