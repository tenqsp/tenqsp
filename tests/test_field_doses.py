# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Doses into a field over continuous axes (IR 0.9; decided 2026-10-27): an amount spread by a
shape, as an initial value's total and shape (``dose G at t=0: 100*point(g, at=0)``,
``m.dose(G, ..., amounts=100, shape=tq.point(g, at=0))``, ``tq.bolus("G", 100,
shape=...)``), or a density added to the field.

Oracle: a stomach whose content ages along ``g`` and empties into ``A`` at a constant
hazard ``k``. A bolus D at t_d gives A(t) = D (1 - exp(-k (t - t_d))) for t >= t_d; doses
add up (the model is linear); an infusion of rate R over T gives the integral of that."""

from __future__ import annotations

import json

import numpy as np
import pytest

import tenqsp as tq
from tenqsp.canonical import differences
from tenqsp.ir import IR_VERSION, ModelIR

STOMACH = """
model stomach
axis g in [0, 20] n=40 units=h kind=one_way
param k=0.7
field G(g)
d_t(G) = -d_g(G) - k*G     using sharp
outflow G at g=upper
d_t(A) = k*int_g(G)
"""
TIGHT = {"rtol": 1e-10, "atol": 1e-12}
T = np.linspace(0, 12, 49)


def emptied(times, amounts, t=T, k=0.7):
    return sum(
        np.where(t >= td, a * (1 - np.exp(-k * (t - td))), 0.0)
        for td, a in zip(times, amounts, strict=True)
    )


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_doses_into_the_field_written_in_the_model(backend):
    m = tq.Model.from_text(STOMACH + "dose G at t=0, 3: 100*point(g, at=0), 50*point(g, at=0)\n")
    sol = m.solve((0, 12), times=T, backend=backend, **TIGHT)
    np.testing.assert_allclose(sol["A"], emptied([0, 3], [100, 50]), atol=1e-7)
    assert max(sol.ledger.check().values()) < 1e-9
    assert len(sol.doses) == 2  # the field's cells with no share are not listed


def test_a_dose_at_the_start_is_the_same_as_an_initial_value():
    dosed = tq.Model.from_text(STOMACH + "dose G at t=0: 100*normal(g, mean=2, sd=0.5)\n")
    started = tq.Model.from_text(STOMACH + "G(0) = 100*normal(g, mean=2, sd=0.5)\n")
    a = dosed.solve((0, 12), times=T, **TIGHT)
    b = started.solve((0, 12), times=T, **TIGHT)
    np.testing.assert_allclose(a["G"], b["G"], atol=1e-12)
    np.testing.assert_allclose(a["A"], b["A"], atol=1e-12)


def test_a_density_dose_adds_its_integral():
    """density(5*exp(-g)) adds 5 (1 - e^-20) in all, emptied at once at rate k."""
    m = tq.Model.from_text(STOMACH + "dose G at t=1: density(5*exp(-g))\n")
    sol = m.solve((0, 12), times=T, **TIGHT)
    total = float(np.asarray(sol["A"])[-1] + np.sum(np.asarray(sol["G"])[-1]) * 0.5)
    assert abs(total - 5 * (1 - np.exp(-20))) < 1e-3  # cell averages of exp(-g)


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_a_regimen_into_the_field_replaces_the_model_doses(backend):
    m = tq.Model.from_text(STOMACH)
    reg = tq.bolus("G", 100, times=[0, 2, 4], shape=tq.point("g", at=0))
    reg = reg + tq.infusion("G", rate=10, duration=2, start=6, shape=tq.point("g", at=0))
    sim = m.simulator(outputs=["A"], backend=backend, **TIGHT)
    r = sim(times=T, regimen=reg)
    s = np.linspace(6, 8, 2001)  # the infusion as many small boluses
    inf = np.trapezoid(
        np.where(T[:, None] >= s, 10 * (1 - np.exp(-0.7 * (T[:, None] - s))), 0), s, axis=1
    )
    np.testing.assert_allclose(r["A"], emptied([0, 2, 4], [100] * 3) + inf, atol=1e-5)


def test_a_scenario_with_a_field_dose_saves_and_reads_back(tmp_path):
    sc = tq.Scenario(regimen=tq.bolus("G", 100, times=[0, 12], shape=tq.point("g", at=0)))
    path = tmp_path / "sc.json"
    sc.save(path)
    back = tq.Scenario.load(path)
    assert back.regimen.boluses == sc.regimen.boluses
    assert back.regimen.boluses[0].shape == "point(g, at=0)"


def test_sensitivity_to_a_field_dose_amount_and_time():
    """dA/d(amount) = (1 - e^-k(t - t_d)); dA/d(t_d) = -amount k e^-k(t - t_d), on JAX,
    through sim.function (the dose fills one slot per cell it reaches)."""
    m = tq.Model.from_text(STOMACH)
    sim = m.simulator(outputs=["A"], backend="jax", **TIGHT)
    sc = tq.Scenario(
        regimen=tq.bolus("G", [100, 50], times=[0, 3], shape=tq.point("g", at=0)),
        times=[2.0, 5.0, 8.0],
    )
    s = sim.sensitivity(sc, wrt=["dose 2 amount", "dose 2 time"])
    t = np.array([2.0, 5.0, 8.0])
    np.testing.assert_allclose(
        np.ravel(s["A"]["dose 2 amount"]),
        np.where(t >= 3, 1 - np.exp(-0.7 * (t - 3)), 0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.ravel(s["A"]["dose 2 time"]),
        np.where(t > 3, -50 * 0.7 * np.exp(-0.7 * (t - 3)), 0),
        rtol=1e-4,
        atol=1e-5,
    )


def test_spread_by_a_shape_over_several_cells():
    """A regimen's dose spread by a normal shape reaches the same cells, with the same
    amounts, as the same amount and shape as an initial value."""
    m = tq.Model.from_text(STOMACH)
    cm = m.compile()
    cells, shares = cm.event_cells("G", (), "normal(g, mean=2, sd=0.5)")
    assert abs(shares.sum() - 1) < 1e-12 and len(cells) > 3
    started = tq.Model.from_text(STOMACH + "G(0) = 100*normal(g, mean=2, sd=0.5)\n").compile()
    y0 = np.asarray(started.initial_state())
    amounts = y0 * np.asarray(started.measure())
    np.testing.assert_allclose(100 * shares, amounts[cells], rtol=1e-12)


def test_written_back_as_text_and_ir_version():
    m = tq.Model.from_text(
        STOMACH + "dose G at t=0, 12: 100*point(g, at=0)\ndose G at t=24: density(2*exp(-g))\n"
    )
    text = m.to_text()
    flat = text.replace(" ", "")
    assert "doseGatt=0,12:100*point(g,at=0)" in flat
    assert "doseGatt=24:density(2*exp(-g))" in flat
    ir = m.to_ir()
    assert ir.ir_version == IR_VERSION
    assert differences(ir, tq.Model.from_text(text).to_ir()) == []
    assert ModelIR.from_json(ir.to_json()) == ir
    # a 0.8 file (no dose shapes) still reads
    old = tq.Model.from_text(STOMACH + "dose A at t=0: 1\n").to_ir()
    d = json.loads(old.to_json())
    d["ir_version"] = "0.8"
    for ds in d["doses"]:
        ds.pop("shape", None)
    assert ModelIR.from_dict(d).doses[0].shape is None


def test_mistakes_are_explained():
    with pytest.raises(tq.notation.NotationError, match="needs a shape"):
        tq.Model.from_text(STOMACH + "dose G at t=0: 100\n")
    with pytest.raises(ValueError, match="needs a shape"):
        m = tq.Model.from_text(STOMACH)
        m.dose(m["G"], times=[0], amounts=100)
    m = tq.Model.from_text(STOMACH)
    with pytest.raises(ValueError, match="give a shape"):
        m.simulate(times=[0, 1], regimen=tq.bolus("G", 1, times=[0]))
    with pytest.raises(ValueError, match="a shape is for a field"):
        m.simulate(times=[0, 1], regimen=tq.bolus("A", 1, times=[0], shape=tq.point("g", at=0)))
    with pytest.raises(ValueError, match="no mass on the grid"):
        m.simulate(times=[0, 1], regimen=tq.bolus("G", 1, times=[0], shape="point(g, at=50)"))
    cat = tq.Model.from_text("axis s categorical [S, R]\nfield B(s)\nd_t(B) = 0\n", name="c")
    with pytest.raises(ValueError, match="fix every axis"):
        cat.simulate(times=[0, 1], regimen=tq.bolus("B", 1, times=[0], shape="point(s, at=0)"))
