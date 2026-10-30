# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Step 4.4d: lag times and bioavailability of the doses into a place (decided 2026-10-25;
design note docs/design/step4_4d_api.md §1). Oracles: the one-compartment oral PK closed form
with a lag and F (NONMEM's ALAG and F), infusions shifted whole, the ledger closing against
the doses as given, and derivatives against the closed form differentiated by JAX."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq
from tenqsp.backends import jax_replay
from tenqsp.canonical import differences
from tenqsp.ir import IR_VERSION, LagSpec, ModelIR
from tenqsp.validate import ModelError

TIGHT = {"rtol": 1e-10, "atol": 1e-12}
T = np.array([0.25, 0.5, 1, 2, 5, 12.0])
TEXT = """model pk
param ka = 1, k = 0.2, tlag = 0.5, F = 0.8
compartment depot, centr
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - k*centr
dose depot at t=0: 100
lag depot = tlag
bioavailability depot = F
"""


def oral(t, ka=1.0, k=0.2, tlag=0.5, F=0.8, dose=100.0, td=0.0):
    s = jnp.maximum(t - td - tlag, 0.0)
    return F * dose * ka / (ka - k) * (jnp.exp(-k * s) - jnp.exp(-ka * s))


def infused(t, rate, start, dur, ka=1.0, k=0.2, tlag=0.5, F=0.8):
    """centr for a zero-order input F*rate into depot on [start+tlag, start+tlag+dur]:
    superposition of a step up and a step down of the constant-input solution."""

    def step(s):  # response to a unit-rate input switched on at s = 0
        s = np.maximum(s, 0.0)
        return ka / (ka - k) * ((1 - np.exp(-k * s)) / k - (1 - np.exp(-ka * s)) / ka)

    a = start + tlag
    return F * rate * (step(t - a) - step(t - a - dur))


@pytest.fixture(scope="module")
def m():
    return tq.Model.from_text(TEXT)


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_a_lagged_bioavailable_bolus_matches_the_closed_form(m, backend):
    sol = m.solve((0, 12), times=T, backend=backend, **TIGHT)
    assert np.allclose(sol["centr"], oral(T), rtol=1e-8, atol=1e-9)
    assert sol["centr"][0] == 0 and sol["centr"][1] == 0  # nothing before t = 0.5
    # sol.doses: when and how much arrived
    assert sol.doses == [(0.5, "depot", 80.0, 80.0)]


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_a_regimen_is_lagged_and_scaled_too_infusions_shifted_whole(m, backend):
    reg = tq.bolus("depot", 50, times=[1.0]) + tq.infusion("depot", rate=20, duration=2, start=3)
    sol = m.solve((0, 12), times=T, regimen=reg, backend=backend, **TIGHT)
    want = oral(T, dose=50, td=1.0) + infused(T, 20, 3.0, 2.0)
    assert np.allclose(sol["centr"], want, rtol=1e-8, atol=1e-9)
    book = sol.ledger
    assert book.check()["depot"] < 1e-10
    moved = {k: v[-1] for k, v in book.moved["depot"].items()}
    # booked as given; the books close against the regimen
    assert moved["doses"] == pytest.approx(50) and moved["infusions"] == pytest.approx(40)
    assert moved["not absorbed"] == pytest.approx(-0.2 * 90)
    assert book.kinds["depot"]["not absorbed"] == "bioavailability"


@pytest.mark.parametrize("backend", ["scipy", "jax"])
def test_a_dose_arriving_after_the_end_is_not_given(m, backend):
    sol = m.solve((0, 2), times=[1, 2], params={"tlag": 3.0}, backend=backend)
    assert sol.doses == [] and np.all(sol["centr"] == 0) and np.all(sol["depot"] == 0)
    assert "doses" not in sol.ledger.moved["depot"]


def test_values_are_checked_when_run_and_never_rebuild(m):
    with pytest.raises(ModelError, match=r"lag of 'depot': must be >= 0 \(found -1\)"):
        m.solve((0, 5), params={"tlag": -1.0})
    with pytest.raises(ModelError, match="bioavailability of 'depot': must be >= 0"):
        m.solve((0, 5), params={"F": -0.1})
    sim = m.simulator(outputs=["centr"], backend="scipy")
    sim(params={"tlag": 0.3, "F": 0.5}, times=T)
    cm = m.compile()
    before = cm.compilations()
    for x in (0.1, 0.7, 1.4):
        r = sim(params={"tlag": x, "F": x / 2}, times=T)
        assert np.allclose(r["centr"], oral(T, tlag=x, F=x / 2), rtol=1e-5, atol=1e-6)
    assert cm.compilations() == before
    assert before["dose lags and bioavailabilities"] == 1
    sol = m.solve((0, 5), params={"F": 1.3})  # F above 1 is allowed (no warning)
    assert sol.doses[0][2] == pytest.approx(130)


def test_a_lag_on_a_field_applies_to_every_dosed_element():
    m = tq.Model.from_text(
        "param k = 0.3, tl = 2, F = 0.5\n"
        "axis s categorical [S, R]\n"
        "field B(s)\n"
        "d_t(B) = -k*B\n"
        "dose B[s=S] at t=0: 1\n"
        "dose B[s=R] at t=1: 4\n"
        "lag B = tl\n"
        "bioavailability B = F\n",
        name="f",
    )
    for backend in ("scipy", "jax"):
        sol = m.solve((0, 6), times=[2.5, 6], backend=backend, **TIGHT)
        assert [d[:3] for d in sol.doses] == [(2.0, "B", 0.5), (3.0, "B", 2.0)]
        S = np.where(sol.t >= 2, 0.5 * np.exp(-0.3 * (sol.t - 2)), 0)
        R = np.where(sol.t >= 3, 2.0 * np.exp(-0.3 * (sol.t - 3)), 0)
        assert np.allclose(sol["B"], np.stack([S, R], 1), rtol=1e-8, atol=1e-12)


def test_text_python_ir_round_trip_and_migration(m):
    p = tq.Model("pk")
    depot = p.compartment("depot")
    p.param("tlag", 0.5)
    p.param("F", 0.8)
    p.ode(depot, "-depot")
    p.lag(depot, "tlag")
    p.bioavailability("depot", "F")
    assert p.to_ir().lags == (LagSpec("depot", "tlag"),)
    assert "lag depot = tlag\nbioavailability depot = F\n" in p.to_text()
    ir = m.to_ir()
    assert ir.ir_version == IR_VERSION  # 0.8 introduced lags
    assert differences(ir, tq.Model.from_text(m.to_text()).to_ir()) == []
    assert ModelIR.from_json(ir.to_json()) == ir
    # a 0.7 file (no lags) still reads
    d = json.loads(ir.to_json())
    d["ir_version"] = "0.7"
    del d["lags"], d["bioavailabilities"]
    old = ModelIR.from_dict(d)
    assert old.lags == () and old.bioavailabilities == ()
    # a value is not structure; the formula is
    moved = tq.Model.from_text(TEXT.replace("tlag = 0.5", "tlag = 2")).to_ir()
    assert moved.structure_hash() == ir.structure_hash()
    other = tq.Model.from_text(TEXT.replace("lag depot = tlag", "lag depot = 2*tlag")).to_ir()
    assert other.structure_hash() != ir.structure_hash()
    # `lag = …` is still a formula named lag
    f = tq.Model.from_text("param a = 2\ncompartment A\nlag = 3*a\nd_t(A) = -lag*A\n")
    assert f.to_ir().lags == () and f.to_ir().definitions[0].name == "lag"


@pytest.mark.parametrize(
    "line, message",
    [
        ("lag nowhere = tlag", "nowhere"),
        ("lag centr = centr", "uses 'centr'"),
        ("lag depot = t", "uses 't'"),
        ("lag depot[s=S] = tlag", "whole place"),
        ("lag depot = tlag\nlag depot = 1", "already has a lag"),
        ("bioavailability depot", "bioavailability depot = formula"),
    ],
)
def test_mistakes_are_explained(line, message):
    base = TEXT.replace("lag depot = tlag\nbioavailability depot = F\n", "")
    with pytest.raises((tq.NotationError, ModelError), match=message):
        tq.Model.from_text(base + line + "\n")


def test_explain_and_validation_notes(m):
    text = m.explain()
    assert "doses into depot: arrive tlag later, and F of each arrives" in text
    n = tq.Model.from_text(TEXT.replace("lag depot", "lag centr"))
    notes = [i.message for i in n.validate().issues if i.level == "note"]
    assert any("acts on a regimen's doses only" in x for x in notes)


def test_derivatives_through_the_lag_and_bioavailability(m):
    """V8: d centr / d (ka, k, tlag, F) by the replay = the closed form; forward = reverse."""
    cm = m.compile()
    t = T[T != 0.5]  # at the arrival itself the derivative is one-sided (after the dose)
    rec = jax_replay.record(cm, (0, 12), times=t, **TIGHT)
    idx = [cm.params.index[k].start for k in ("ka", "k", "tlag", "F")]
    out = cm.layout.slices["centr"].start
    th = jnp.asarray(rec.run.th)

    def f(x):
        return jax_replay.replay(cm, rec, x)[:, out]

    fwd = np.asarray(jax.jacfwd(f)(th))[:, idx]
    rev = np.asarray(jax.jacrev(f)(th))[:, idx]
    exact = np.asarray(jax.jacfwd(lambda p: oral(t, *p))(jnp.array([1.0, 0.2, 0.5, 0.8])))
    assert np.allclose(fwd, exact, rtol=1e-6, atol=1e-7)
    assert np.allclose(fwd, rev, rtol=1e-10, atol=1e-10)
    assert np.all(fwd[0] == 0)  # before the dose arrives nothing moves
