# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Step 4.1b: regimens, scenarios and the simulator (V44; design decisions
2026-10-24). Expected values come from closed-form PK solutions."""

from __future__ import annotations

import numpy as np
import pytest

import tenqsp as tq

ORAL = """param ka, CL
param V = 30
compartment depot, centr
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
observable conc = centr/V
dose depot at t=0: 100
"""
IV = """param CL = 4, V = 30
compartment A volume=V
d_t(A) = -CL/V*A
observable conc = A
"""
TIGHT = {"rtol": 1e-10, "atol": 1e-12}


def oral(t, doses, ka=1.2, CL=4.0, V=30.0):
    """Superposed one-compartment oral doses: ``doses`` = [(time, amount), ...]."""
    ke, t = CL / V, np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    for td, amt in doses:
        s = np.clip(t - td, 0, None)
        out += amt * ka / (V * (ka - ke)) * (np.exp(-ke * s) - np.exp(-ka * s))
    return out


def iv_infusion(t, rate, start, dur, CL=4.0, V=30.0):
    """A zero-order infusion into a one-compartment model (closed form)."""
    ke, t = CL / V, np.asarray(t, dtype=float)
    on = np.clip(t - start, 0, dur)  # time infused so far
    after = np.clip(t - start - dur, 0, None)
    return rate / CL * (1 - np.exp(-ke * on)) * np.exp(-ke * after)


@pytest.fixture
def oral_model():
    return tq.Model.from_text(ORAL, name="pk")


# ----------------------------------------------------------------------------- regimens


def test_repeated_boluses_match_superposition(oral_model):
    sim = oral_model.simulator(outputs=["conc"], **TIGHT)
    t = np.linspace(0, 96, 193)
    r = sim(
        params={"ka": 1.2, "CL": 4}, regimen=tq.bolus("depot", 100, start=0, every=12, n=7), times=t
    )
    expected = oral(t, [(12 * k, 100) for k in range(7)])
    np.testing.assert_allclose(r["conc"], expected, rtol=1e-7, atol=1e-9 * expected.max())


def test_an_infusion_matches_the_closed_form():
    sim = tq.Model.from_text(IV, name="iv").simulator(**TIGHT)
    t = np.linspace(0, 24, 97)
    r = sim(regimen=tq.infusion("A", rate=50, duration=2, start=3), times=t)
    expected = iv_infusion(t, 50, 3, 2)
    np.testing.assert_allclose(r["conc"], expected, rtol=1e-7, atol=1e-10)
    book = r.solution.ledger
    assert max(book.check().values()) < 1e-9
    np.testing.assert_allclose(book.moved["A"]["infusions"][-1], 100.0)  # rate x duration
    i = int(np.searchsorted(t, 4.0))
    assert book.rates["A"]["infusions"][i] == 50 and book.rates["A"]["infusions"][-1] == 0


def test_two_of_rate_amount_duration_describe_the_same_infusion():
    a = tq.infusion("A", rate=50, duration=2, start=3)
    b = tq.infusion("A", amount=100, duration=2, start=3)
    c = tq.infusion("A", amount=100, rate=50, start=3)
    assert a == b == c
    with pytest.raises(ValueError, match="exactly two of rate, amount and duration"):
        tq.infusion("A", rate=50, amount=100, duration=2)


def test_a_regimen_replaces_the_models_doses_and_an_empty_one_means_none(oral_model):
    sim = oral_model.simulator(outputs=["conc"], **TIGHT)
    t = np.linspace(0, 24, 25)
    p = {"ka": 1.2, "CL": 4}
    model_doses = sim(params=p, times=t)  # not given: the dose written in the model
    np.testing.assert_allclose(model_doses["conc"], oral(t, [(0, 100)]), rtol=1e-7, atol=1e-10)
    moved = sim(params=p, times=t, regimen=tq.bolus("depot", 100, times=[5]))
    np.testing.assert_allclose(moved["conc"], oral(t, [(5, 100)]), rtol=1e-7, atol=1e-10)
    assert np.all(sim(params=p, times=t, regimen=tq.Regimen())["conc"] == 0)
    # m.solve takes the same regimen
    sol = oral_model.solve(
        (0, 24), times=t, params=p, regimen=tq.bolus("depot", 50, times=[0]), **TIGHT
    )
    np.testing.assert_allclose(sol["conc"], oral(t, [(0, 50)]), rtol=1e-7, atol=1e-10)


def test_a_dose_into_one_element_of_a_weighted_field():
    text = """axis s categorical [S, R]
param w[s] = {S: 2, R: 4}
field C(s) weights=w[s]
d_t(C) = 0
"""
    sim = tq.Model.from_text(text, name="w").simulator(outputs=["C"], **TIGHT)
    r = sim(regimen=tq.bolus("C[s=R]", 8, times=[1]), times=[0, 1, 2])
    np.testing.assert_allclose(r["C"], [[0, 0], [0, 2], [0, 2]])  # 8 / w[R] = 2
    same = sim(regimen=tq.bolus("C[s=R]", 8, times=[1]), times=[0, 1, 2])
    assert np.array_equal(same["C"], r["C"])


def test_wrong_targets_and_times_are_explained(oral_model):
    sim = oral_model.simulator(outputs=["conc"])
    p, t = {"ka": 1, "CL": 4}, [0, 1]
    with pytest.raises(KeyError, match="Did you mean 'depot'"):
        sim(params=p, times=t, regimen=tq.bolus("dept", 1, times=[0]))
    with pytest.raises(ValueError, match="before the start"):
        sim(params=p, times=[2, 3], start=2, regimen=tq.bolus("depot", 1, times=[1]))
    with pytest.raises(ValueError, match="Give the output times"):
        sim(params=p)
    with pytest.raises(KeyError, match="not an observable, field or compartment"):
        oral_model.simulator(outputs=["concentration"])
    field = tq.Model.from_text("axis s categorical [S, R]\nfield B(s)\nd_t(B) = 0\n", name="f")
    with pytest.raises(ValueError, match="fix every axis"):
        field.simulate(times=[0, 1], regimen=tq.bolus("B", 1, times=[0]))
    age = tq.Model.from_text("axis a in [0, 1] n=4\nfield N(a)\nd_t(N) = 0\n", name="a")
    with pytest.raises(ValueError, match="continuous axis"):
        age.simulate(times=[0, 1], regimen=tq.bolus("N[a=0]", 1, times=[0]))
    with pytest.raises(ValueError, match="every= and n="):
        tq.bolus("depot", 1, every=12)


def test_a_nonmem_dose_table_reads_as_the_same_regimen():
    table = {
        "ID": [1, 1, 1, 1],
        "TIME": [0, 0, 48, 50],
        "AMT": [100, ".", 120, 0],
        "CMT": [1, 2, 2, 2],
        "RATE": [0, ".", 60, "."],
        "ADDL": [3, ".", 0, "."],
        "II": [12, ".", 0, "."],
        "EVID": [1, 0, 1, 0],
    }
    reg = tq.regimen_from_table(table, targets={1: "depot", 2: "centr"})
    expected = tq.bolus("depot", 100, start=0, every=12, n=4) + tq.infusion(
        "centr", amount=120, rate=60, start=48
    )
    assert reg == expected
    with pytest.raises(ValueError, match="targets="):
        tq.regimen_from_table(table)


# ----------------------------------------------------------------------------- scenarios


def test_values_are_layered_model_then_scenario_then_call(oral_model):
    sim = oral_model.simulator(outputs=["conc"], **TIGHT)
    t = np.linspace(0, 24, 25)
    sc = tq.Scenario(name="base", params={"ka": 1.2, "CL": 5}, times=t)
    r = sim(sc, params={"CL": 4})  # the call changes CL only; ka from the scenario, V default
    assert r.scenario.params == {"ka": 1.2, "CL": 4} and sc.params["CL"] == 5
    np.testing.assert_allclose(r["conc"], oral(t, [(0, 100)], CL=4), rtol=1e-7, atol=1e-10)
    r2 = sim(sc.replace(params={"V": 60}))  # merged: ka, CL kept
    np.testing.assert_allclose(r2["conc"], oral(t, [(0, 100)], CL=5, V=60), rtol=1e-7, atol=1e-10)


def test_a_scenario_saves_and_loads(tmp_path):
    t = [0.0, 12.0, 24.0]
    with_doses = tq.Scenario(
        name="q12h",
        params={"CL": 5, "k": {"S": 1, "R": 2}},
        regimen=tq.bolus("depot", 100, start=0, every=12, n=2)
        + tq.infusion("C[organ=gut, sub=v]", rate=1, duration=2),
        times=t,
        start=0,
    )
    none = tq.Scenario(name="no doses", regimen=tq.Regimen(), times=t)
    model_doses = tq.Scenario(name="model doses", times=t)
    for sc in (with_doses, none, model_doses):
        path = tmp_path / f"{sc.name}.json"
        sc.save(path)
        assert tq.Scenario.load(path) == sc
    assert "regimen" not in model_doses.to_dict() and none.to_dict()["regimen"] == []


def test_simulate_shortcut_matches_the_simulator(oral_model):
    t = [0, 6, 12]
    p = {"ka": 1.2, "CL": 4}
    a = oral_model.simulate(params=p, times=t, outputs=["conc"], **TIGHT)
    b = oral_model.simulator(outputs=["conc"], **TIGHT)(params=p, times=t)
    assert np.array_equal(a["conc"], b["conc"])
    with pytest.raises(KeyError, match="not an output"):
        a["centr"]
    assert "conc" in a.to_xarray() and "centr" not in a.to_xarray()
    with pytest.raises(TypeError, match="Unknown simulator options"):
        oral_model.simulator(atl=1e-8)


# ----------------------------------------------------------------------------- V44


@pytest.mark.scipy_backend
def test_v44_values_doses_and_times_never_rebuild(oral_model):
    """100 runs that change parameter values, dose amounts and times, the number of doses,
    infusions and output times: one build of every function the solver uses."""
    sim = oral_model.simulator(outputs=["conc"])
    rng = np.random.default_rng(44)
    for _ in range(100):
        n = int(rng.integers(1, 5))
        reg = tq.bolus("depot", rng.uniform(10, 200, n), times=np.sort(rng.uniform(0, 24, n)))
        if rng.random() < 0.5:
            reg = reg + tq.infusion(
                "centr",
                rate=rng.uniform(1, 20),
                duration=rng.uniform(0.5, 4),
                start=rng.uniform(0, 20),
            )
        times = np.sort(rng.uniform(0, 48, int(rng.integers(2, 30))))
        sim(params={"ka": rng.uniform(0.5, 2), "CL": rng.uniform(2, 6)}, regimen=reg, times=times)
    counts = oral_model.compile().compilations()
    used = [
        "solver rhs (with ledger)",
        "jacobian: compartment and ledger rows",
        "initial state",
        "observables",
        "ledger",
    ]
    assert all(counts[k] == 1 for k in used), counts
