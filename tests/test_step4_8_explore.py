# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Step 4.8: simulation and exploration (design note docs/design/step4_8_explore.md):
V46 (local sensitivities across a dose and an extinction), grids and batches, pre-runs and
steady states, plots. Oracles: closed forms, central finite differences, single runs."""

from __future__ import annotations

import math
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq
from tenqsp.examples import toy_phage_model

DECAY = "param k = 0.2\ncompartment A\nd_t(A) = -k*A\nA(0) = 10\n"
ORAL = """param ka = 1.0, CL = 5.0, V = 50.0
compartment depot, centr
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
observable conc = centr/V
"""


def oral_conc(t, ka, CL, V, dose=100.0, t_dose=0.0):
    k = CL / V
    s = np.clip(np.asarray(t) - t_dose, 0, None)
    return np.where(
        np.asarray(t) >= t_dose,
        dose * ka / (V * (ka - k)) * (np.exp(-k * s) - np.exp(-ka * s)),
        0.0,
    )


# ------------------------------------------------------------------------------- V46


def test_v46_sensitivity_across_a_dose_and_an_extinction():
    """A = 10 e^{-kt}, +10 at t = 6, extinct when it falls below 1 (t ≈ 18.8 at k = 0.2).
    d A / d k and d A / d (dose amount) against central finite differences, absolute and
    relative, before the dose, between, and after the extinction."""
    m = tq.Model.from_text(DECAY + "extinction A below 1\n", name="v46")
    sim = m.simulator(outputs=["A"], rtol=1e-10, atol=1e-12)
    times = [2.0, 5.0, 7.0, 10.0, 15.0, 25.0]
    sc = tq.Scenario(regimen=tq.bolus("A", 10, times=[6]), times=times)
    s = sim.sensitivity(sc, wrt=["k", "dose 1 amount"])
    rel = sim.sensitivity(sc, wrt=["k", "dose 1 amount"], relative=True)

    def run(k=0.2, amount=10.0):
        sc2 = sc.replace(params={"k": k}, regimen=tq.bolus("A", amount, times=[6]))
        return np.asarray(sim(sc2)["A"]).ravel()

    y = run()
    for name, base, f in (
        ("k", 0.2, lambda v: run(k=v)),
        ("dose 1 amount", 10.0, lambda v: run(amount=v)),
    ):
        h = 1e-5 * base
        fd = (f(base + h) - f(base - h)) / (2 * h)
        ad = np.asarray(s["A"][name]).ravel()
        alive = y > 0
        assert np.allclose(ad[alive], fd[alive], rtol=1e-4, atol=1e-8), name
        assert np.all(ad[~alive] == 0)  # after the extinction: zero, and stays so
        r = np.asarray(rel["A"][name]).ravel()
        assert np.allclose(r[alive], fd[alive] * base / y[alive], rtol=1e-4)
        assert np.all(np.isnan(r[~alive]))
    # closed form before the dose: d/dk 10 e^{-kt} = -10 t e^{-kt}
    assert np.asarray(s["A"]["k"]).ravel()[0] == pytest.approx(-20 * math.exp(-0.4), rel=1e-6)


# ------------------------------------------------------------------------------- grids


def test_a_grid_is_the_cross_product_last_name_fastest():
    r1, r2 = tq.bolus("depot", 100, times=[0]), tq.bolus("depot", 200, times=[0])
    g = tq.grid(params={"CL": [3, 5], "V": [40, 60, 80]}, regimens={"low": r1, "high": r2})
    assert len(g) == 12 and g.columns == ("CL", "V", "regimen")
    assert [lab["V"] for lab in g.labels[:3]] == [40, 40, 60]  # regimen fastest, then V
    assert g[1].name == "CL=3, V=40, high" and g[1].regimen is r2
    assert g[11].params == {"CL": 5, "V": 80}
    base = tq.Scenario(name="trial", params={"ka": 2}, times=[1, 2])
    g2 = tq.grid(params={"CL": [3]}, base=base)
    assert g2[0].params == {"ka": 2, "CL": 3} and g2[0].times == (1.0, 2.0)
    assert g2[0].name == "trial: CL=3"
    assert tq.grid(regimens=[r1, r2])[1].name == "regimen 2"
    with pytest.raises(TypeError, match="list of values"):
        tq.grid(params={"CL": 3})
    with pytest.raises(TypeError, match="tq.Regimen"):
        tq.grid(regimens={"x": 100})


# ------------------------------------------------------------------------------- batches


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_a_batch_gives_each_scenarios_closed_form(backend):
    m = tq.Model.from_text(ORAL, name="oral")
    sim = m.simulator(outputs=["conc"], rtol=1e-9, atol=1e-11, backend=backend)
    times = np.linspace(0.5, 24, 12)
    g = tq.grid(
        params={"CL": [3.0, 8.0], "V": [40.0, 60.0]},
        regimens={
            "100 at 0": tq.bolus("depot", 100, times=[0]),
            "50 at 2": tq.bolus("depot", 50, times=[2]),
        },
    )
    b = sim.batch(g, times=times)
    assert b["conc"].shape == (8, 12) and not b.failed and np.array_equal(b.t, times)
    for i, (sc, lab) in enumerate(zip(b.scenarios, g.labels, strict=True)):
        dose, t_dose = (100.0, 0.0) if lab["regimen"] == "100 at 0" else (50.0, 2.0)
        exact = oral_conc(times, 1.0, sc.params["CL"], sc.params["V"], dose, t_dose)
        assert np.allclose(b["conc"][i], exact, rtol=1e-6, atol=1e-9)
    assert np.allclose(np.asarray(b[3]["conc"]), b["conc"][3], rtol=1e-6)  # b[i]: a full run
    table = b.table()
    assert list(table.columns) == ["scenario", "CL", "V", "regimen", "time", "output", "value"]
    assert len(table) == 8 * 12 and table["CL"].dtype.kind == "f"
    ds = b.to_xarray()
    assert ds["conc"].dims == ("scenario", "time") and list(ds["regimen"].values[:2]) == [
        "100 at 0",
        "50 at 2",
    ]


def test_a_list_of_scenarios_labels_what_varies():
    m = tq.Model.from_text(DECAY, name="decay")
    sim = m.simulator(outputs=["A"])
    b = sim.batch([tq.Scenario(params={"k": k}) for k in (0.1, 0.2, 0.3)], times=[1.0, 2.0])
    assert b.columns == ("k",) and [lab["k"] for lab in b.labels] == [0.1, 0.2, 0.3]
    assert np.allclose(b["A"][:, 1], 10 * np.exp(-2 * np.array([0.1, 0.2, 0.3])), rtol=1e-5)


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_a_failed_scenario_is_nan_and_listed_the_others_run(backend):
    m = tq.Model.from_text(DECAY, name="decay")
    sim = m.simulator(outputs=["A"], backend=backend)
    ok = [tq.Scenario(regimen=tq.bolus("A", 1, times=[t])) for t in (0.5, 0.2)]
    bad = tq.Scenario(name="dose before start", regimen=tq.bolus("A", 1, times=[-1]))
    with pytest.warns(UserWarning, match="1 of 3 scenarios failed"):
        b = sim.batch([ok[0], bad, ok[1]], times=[1.0])
    assert [f[:2] for f in b.failed] == [(1, "dose before start")]
    assert "before the start" in b.failed[0][2]
    assert np.isnan(b["A"][1]).all() and np.isfinite(b["A"][[0, 2]]).all()


def test_a_batch_of_a_field_keeps_its_axes():
    m = toy_phage_model(3, 4)
    sim = m.simulator(outputs=["B", "phage"])
    b = sim.batch(tq.grid(params={"delta": [0.1, 0.2]}), times=[1.0, 2.0])
    assert b["B"].shape == (2, 2, 3, 2) and b["phage"].shape == (2, 2)
    table = b.table()
    rows = table[table.output == "B"]
    assert len(rows) == 2 * 2 * 3 * 2 and set(rows["s"]) == {"S", "R"}
    assert b.to_xarray()["B"].dims == ("scenario", "time", "z", "s")


def test_a_batch_needs_matching_times():
    sim = tq.Model.from_text(DECAY, name="decay").simulator(outputs=["A"])
    with pytest.raises(ValueError, match="same number of output times"):
        sim.batch([tq.Scenario(times=[1, 2]), tq.Scenario(times=[1, 2, 3])])
    with pytest.raises(ValueError, match="n_jobs"):
        sim.batch([tq.Scenario()], times=[1], n_jobs=0)


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_several_processes_give_the_same_numbers(backend):
    sim = tq.Model.from_text(ORAL, name="oral").simulator(outputs=["conc"], backend=backend)
    g = tq.grid(
        params={"CL": [3.0, 4.0, 5.0, 6.0, 7.0]},
        regimens={"q12": tq.bolus("depot", 100, start=0, every=12, n=2)},
    )
    one = sim.batch(g, times=np.linspace(1, 24, 24))
    two = sim.batch(g, times=np.linspace(1, 24, 24), n_jobs=2)
    assert np.array_equal(one["conc"], two["conc"]) and two.columns == ("CL", "regimen")


# ------------------------------------------------------------------------------- pre-runs


def test_a_pre_run_for_a_fixed_time_is_the_solve_to_that_time():
    m = tq.Model.from_text(DECAY, name="decay")
    sim = m.simulator(outputs=["A"], rtol=1e-10, atol=1e-12)
    pre = sim.pre_run(tq.Scenario(), until=5.0)
    assert pre.reached and pre.t == 5.0
    assert float(pre.state["A"]) == pytest.approx(10 * math.exp(-1.0), rel=1e-8)
    r = sim(tq.Scenario(times=[0.0, 5.0]).replace(initial=pre))
    assert np.allclose(np.asarray(r["A"]).ravel(), 10 * np.exp(-np.array([1.0, 2.0])), rtol=1e-7)


def test_steady_without_doses_is_kin_over_kout():
    m = tq.Model.from_text(
        "param kin = 4, kout = 0.5\ncompartment R\nd_t(R) = kin - kout*R\nR(0) = 0\n",
        name="turnover",
    )
    sim = m.simulator(outputs=["R"], rtol=1e-8, atol=1e-10)
    pre = sim.pre_run(until="steady")
    assert pre.reached and float(pre.state["R"]) == pytest.approx(8.0, rel=1e-6)


def test_steady_under_repeated_doses_is_the_pk_steady_state():
    """IV bolus D every tau into A, eliminated at k: the trough at steady state is
    D e^{-k tau} / (1 - e^{-k tau}); the next interval starts and ends there."""
    m = tq.Model.from_text("param k = 0.1\ncompartment A\nd_t(A) = -k*A\n", name="iv")
    sim = m.simulator(outputs=["A"], rtol=1e-10, atol=1e-12)
    D, tau, k = 100.0, 12.0, 0.1
    pre = sim.pre_run(
        tq.Scenario(regimen=tq.bolus("A", D, start=0, every=tau, n=50)), until="steady"
    )
    trough = D * math.exp(-k * tau) / (1 - math.exp(-k * tau))
    assert pre.reached and pre.cycles > 2
    assert float(pre.state["A"]) == pytest.approx(trough, rel=1e-6)
    nxt = tq.Scenario(regimen=tq.bolus("A", D, times=[0]), times=[0.0, tau], initial=pre)
    a = np.asarray(sim(nxt)["A"]).ravel()
    assert a[0] == pytest.approx(trough + D, rel=1e-6) and a[1] == pytest.approx(trough, rel=1e-6)


def test_steady_needs_an_even_regimen_and_says_when_not_reached():
    m = tq.Model.from_text("param k = 0.1\ncompartment A\nd_t(A) = -k*A\n", name="iv")
    sim = m.simulator(outputs=["A"])
    with pytest.raises(ValueError, match="repeats at even intervals"):
        sim.pre_run(tq.Scenario(regimen=tq.bolus("A", 1, times=[0, 5, 7])), until="steady")
    growth = tq.Model.from_text("compartment A\nd_t(A) = 0.1*A\nA(0) = 1\n", name="grow")
    with pytest.warns(UserWarning, match="not steady"):
        pre = growth.simulator(outputs=["A"]).pre_run(until="steady", t_max=50)
    assert not pre.reached


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_a_scenario_starting_from_a_pre_run_on_both_backends_and_in_json(backend, tmp_path):
    m = tq.Model.from_text(DECAY, name="decay")
    sim = m.simulator(outputs=["A"], rtol=1e-10, atol=1e-12, backend=backend)
    pre = sim.pre_run(until=5.0)
    sc = tq.Scenario(times=[1.0], initial=pre)
    assert sc.initial == {"A": pytest.approx(10 * math.exp(-1.0))}
    sc.save(tmp_path / "sc.json")
    back = tq.Scenario.load(tmp_path / "sc.json")
    expect = 10 * math.exp(-1.2)
    assert float(np.asarray(sim(back)["A"]).ravel()[0]) == pytest.approx(expect, rel=1e-7)
    # derivatives: the start is fixed (design decision 4), the run itself is differentiated
    f = sim.function(back, wrt=["k"])
    assert float(f(jnp.array([0.2]))["A"][0]) == pytest.approx(expect, rel=1e-7)
    s = sim.sensitivity(back, wrt=["k"])
    assert float(np.asarray(s["A"]["k"]).ravel()[0]) == pytest.approx(-1.0 * expect, rel=1e-5)
    with pytest.raises(KeyError, match="not a field or compartment"):
        sim(tq.Scenario(times=[1.0], initial={"X": 1}))


# ------------------------------------------------------------------------------- plots


def test_the_plots_draw_and_return_their_axes():
    matplotlib = pytest.importorskip("matplotlib")  # the optional extra tenqsp[plot]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = tq.Model.from_text(ORAL, name="oral")
    sim = m.simulator(outputs=["conc"])
    regs = {
        "q12": tq.bolus("depot", 100, start=0, every=12, n=2),
        "q24": tq.bolus("depot", 200, times=[0]),
    }
    b = sim.batch(tq.grid(params={"CL": [3.0, 6.0]}, regimens=regs), times=np.linspace(0, 24, 25))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warnings from drawing
        ax = tq.plot.trajectories(b, "conc")
        assert len(ax.lines) == 4
        ax = tq.plot.fan(b, "conc", by="regimen")
        assert len(ax.lines) == 2 and len(ax.collections) == 2
        s = sim.sensitivity(
            tq.Scenario(regimen=regs["q12"], times=np.linspace(1, 24, 24)), wrt=["CL", "V"]
        )
        ax = tq.plot.sensitivity(s, "conc")
        assert ax.images[0].get_array().shape == (2, 24)
        r = toy_phage_model(4, 6).simulator(outputs=["I"])(times=[1.0, 2.0])
        ax = tq.plot.snapshot(r, "I", axis="a", times=[1.0, 2.0])
        assert len(ax.lines) == 2 and ax.get_xlabel() == "a (h)"
    with pytest.raises(KeyError, match="varied column"):
        tq.plot.fan(b, "conc", by="V")
    plt.close("all")
