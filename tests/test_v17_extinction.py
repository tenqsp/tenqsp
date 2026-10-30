# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""V17: extinction semantics.

- an armed group is set to zero at the exact downward crossing of its threshold, as a whole
  (every member, fields and compartments), and stays at zero when nothing feeds it;
- a group that never reached its threshold (mutants seeded from zero) is never touched;
- the noise floor stops round-off "ghosts" from feeding themselves, without blocking sources;
- the ledger books what was removed, so the books still close.

Oracles are closed forms: exponential decay crosses at ln(N0/thr)/k; a population fed at a
constant rate s from 1e-12 reaches 1 at (1 - 1e-12)/s, and then grows as
(1 + s/r) e^{r (t - t1)} - s/r.
"""

from __future__ import annotations

import numpy as np
import pytest

import tenqsp as tq
from tenqsp.ir import IR_VERSION, ModelIR

TIGHT = {"rtol": 1e-10, "atol": 1e-12}


def decaying(k=0.5, N0=100.0, thr=1.0):
    m = tq.Model("decay")
    B = m.compartment("B", init=N0)
    m.ode(B, -k * B)
    m.extinction(B, threshold=thr)
    return m


def test_v17_zeroed_at_the_exact_crossing():
    sol = decaying().solve((0, 20), times=[0, 5, 9, 9.5, 20], **TIGHT)
    (te, name), *_ = sol.extinctions
    assert name == "extinct_B" and len(sol.extinctions) == 1
    assert te == pytest.approx(np.log(100) / 0.5, rel=1e-8)  # as accurate as the solve
    np.testing.assert_allclose(sol["B"][:3], 100 * np.exp(-0.5 * np.array([0, 5, 9])), rtol=1e-8)
    assert sol["B"][3] == 0.0 and sol["B"][4] == 0.0
    assert sol.ledger.moved["B"]["extinction"][-1] == pytest.approx(-1.0, rel=1e-8)
    assert max(sol.ledger.check().values()) < 1e-12
    assert sol.ledger.kinds["B"]["extinction"] == "extinction"


def lineage(n=20):
    """Infected cells I(age) and free phage P form one lineage: the threshold applies to
    their total, and both are zeroed together."""
    m = tq.Model("lineage")
    a = m.axis("age", 0, 1, n, kind="one_way")
    I = m.field("I", axes=[a])
    P = m.compartment("P", init=50.0)
    m.advect(I, axis=a, velocity=1.0)
    m.inflow(I, axis=a, at="lower", flux=0.2 * P)  # P adsorbs and becomes I
    m.outflow(I, axis=a, at="upper", to=P)  # I bursts back into P, one for one
    m.hazard(I, 0.3)
    m.ode(P, -0.2 * P - 0.1 * P)
    m.init(I, 0.0)
    m.extinction([I, P], threshold=0.5, name="phage")
    return m


def test_v17_a_group_spanning_a_field_and_a_compartment():
    m = lineage()
    sol = m.solve((0, 40), times=np.linspace(0, 40, 81), **TIGHT)
    assert [g for _, g in sol.extinctions] == ["phage"]
    te = sol.extinctions[0][0]
    cm = m.compile()
    totals = cm.group_totals(sol.y)[:, 0]
    before, after = sol.t < te, sol.t >= te
    assert totals[before].min() > 0.5 and np.all(totals[after] == 0.0)
    # the total just before the crossing extrapolates to the threshold at te
    i = np.flatnonzero(before)[-1]
    assert totals[i] > 0.5 and totals[i] * np.exp(-0.3 * (te - sol.t[i])) == pytest.approx(
        0.5, rel=0.05
    )
    assert max(sol.ledger.check().values()) < 1e-10
    gone = sol.ledger.moved["I"]["extinction"][-1] + sol.ledger.moved["P"]["extinction"][-1]
    assert gone == pytest.approx(-0.5, rel=1e-6)


def strains(*, kill=3.0):
    """S is killed by a drug that washes out; a division of S gives a resistant R with
    probability 1e-6. R starts at zero: seeded by mutation, it never reaches its threshold
    during the run and must not be wiped. S falls below its threshold and must stay at zero
    after the drug has gone (pbisim let it dip below and regrow)."""
    m = tq.Model("strains")
    s = m.axis("s", labels=["S", "R"])
    B = m.field("B", axes=[s])
    C = m.compartment("C", init=1.0)  # drug
    g = m.param("g", {"S": 1.0, "R": 0.8}, over=s)
    susceptible = m.param("susc", {"S": 1.0, "R": 0.0}, over=s)
    M = m.transitions("M", axis=s, rates={"S->R": 1e-6})
    div = m.define("div", g[s] * B)
    m.reaction(B, div - kill * susceptible[s] * C / (0.05 + C) * B)
    m.transition(B, axis=s, matrix=M, of=div)
    m.ode(C, -0.3 * C)
    m.init(B, 1e3, where={s: "S"})
    m.init(B, 0.0, where={s: "R"})
    return m


def test_v17_mutants_seeded_from_zero_are_never_wiped_and_the_dip_does_not_regrow():
    # without extinction, S dips far below one cell and then regrows: the artefact
    free = strains().solve((0, 30), times=np.linspace(0, 30, 61))
    S_free = free["B"][:, 0]
    assert S_free.min() < 1e-2 and S_free[-1] > 1.0
    # the same strains as two compartments, one group each
    m = tq.Model("two_fields")
    S, R = m.compartment("S", init=1e3), m.compartment("R", init=0.0)
    C = m.compartment("C", init=1.0)
    m.ode(S, S - 1e-6 * S - 3.0 * C / (0.05 + C) * S)
    m.ode(R, 0.8 * R + 1e-6 * S)
    m.ode(C, -0.3 * C)
    m.extinction(S, threshold=1.0)
    m.extinction(R, threshold=1.0)
    sol = m.solve((0, 30), times=np.linspace(0, 30, 61))
    assert [g for _, g in sol.extinctions] == ["extinct_S"]
    te = sol.extinctions[0][0]
    assert np.all(sol["S"][sol.t >= te] == 0.0)  # no regrowth from zero
    # R, seeded from zero, spends hours below its threshold without being wiped, then grows
    later = sol.t > 0
    assert np.all(sol["R"][later] > 0)
    assert np.any((sol["R"] < 1.0) & later) and sol["R"][-1] > 1.0
    assert max(sol.ledger.check().values()) < 1e-10


def test_v17_a_group_can_be_dosed_back_and_go_extinct_again():
    m = tq.Model("again")
    B = m.compartment("B", init=10.0)
    m.ode(B, -1.0 * B)
    m.extinction(B, threshold=1.0)
    m.dose(B, times=[5.0], amounts=[20.0])
    sol = m.solve((0, 12), times=[0, 5, 12], **TIGHT)
    times = [t for t, _ in sol.extinctions]
    assert times == pytest.approx([np.log(10), 5 + np.log(20)], rel=1e-8)
    assert sol.at("B", 5.0) == 20.0 and sol["B"][-1] == 0.0
    assert max(sol.ledger.check().values()) < 1e-12


def ghost(noise_floor: bool, source: float = 0.0):
    m = tq.Model("ghost")
    P = m.compartment("P", init=1e-12)  # round-off left behind by a solver
    m.ode(P, 5.0 * P + source)
    m.extinction(P, threshold=1.0, noise_floor=noise_floor)
    return m


def test_v17_noise_floor_stops_ghosts_but_not_sources():
    t = np.array([0, 5, 10, 11, 12.0])
    grown = ghost(False).solve((0, 12), times=t, **TIGHT)
    assert grown["P"][-1] > 1e10  # a ghost feeding itself: the pbisim cascade
    held = ghost(True).solve((0, 12), times=t, **TIGHT)
    np.testing.assert_array_equal(held["P"], 1e-12)  # sees itself as zero: does not grow
    with pytest.warns(UserWarning, match="fed from outside"):
        fed = ghost(True, source=0.1).solve((0, 12), times=t, **TIGHT)
    # filled at the source rate only, reaches 1 at t = 10, then grows on its own
    np.testing.assert_allclose(fed["P"][:3], 1e-12 + 0.1 * t[:3], rtol=1e-9)
    after = (1 + 0.1 / 5) * np.exp(5 * (t[3:] - 10)) - 0.1 / 5
    np.testing.assert_allclose(fed["P"][3:], after, rtol=1e-6)
    assert fed.extinctions == [] and max(fed.ledger.check().values()) < 1e-10


def test_noise_floor_keeps_observables_and_ledger_honest():
    """Outputs report the state as it is (1e-12), not as the equations see it."""
    m = ghost(True)
    m.observable("phage", m["P"])
    sol = m.solve((0, 1), times=[0, 1])
    np.testing.assert_array_equal(sol["phage"], 1e-12)


@pytest.mark.parametrize("method", ["BDF", "Radau"])
@pytest.mark.parametrize("ledger", [True, False])
def test_events_on_every_path(method, ledger):
    sol = decaying().solve((0, 20), times=[20], method=method, ledger=ledger, **TIGHT)
    assert sol.extinctions[0][0] == pytest.approx(np.log(100) / 0.5, rel=1e-8)
    assert sol["B"][-1] == 0.0


def test_sparse_jacobian_with_a_noise_floor():
    """Below its threshold the group is constant to the equations: its columns vanish."""
    m = tq.Model("floored")
    x = m.axis("x", 0, 1, 5)
    u = m.field("u", axes=[x])
    P = m.compartment("P")
    m.diffuse(u, axis=x, D=0.1)
    m.reaction(u, 2 * u * P)
    m.ode(P, 3 * P - 0.1 * m.integral(u, over=x))
    m.init(u, 1.0)
    m.init(P, 1.0)
    m.extinction([u], threshold=0.5, noise_floor=True)
    cm = m.compile()
    for scale in (1.0, 0.1):  # above, then below the threshold
        y = np.concatenate([scale * np.linspace(0.5, 1.5, 5), [0.7]])
        dense = np.asarray(cm.jacobian(0.0, y))
        np.testing.assert_allclose(cm.sparse_jacobian(0.0, y).toarray(), dense, atol=1e-14)
        if scale < 1:
            assert np.all(dense[:, :5] == 0)


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda m, A, B: (m.extinction(A, threshold=1), m.extinction([A, B], threshold=1)),
         "one extinction group only"),
        (lambda m, A, B: m.extinction(A, threshold=0.0), "threshold must be > 0"),
        (lambda m, A, B: m.extinction([A, A], threshold=1), "listed twice"),
    ],
    ids=["overlap", "threshold", "duplicate"],
)  # fmt: skip
def test_groups_are_checked(build, message):
    m = tq.Model("bad")
    A, B = m.compartment("A", init=1.0), m.compartment("B", init=1.0)
    m.ode(A, -A)
    m.ode(B, -B)
    build(m, A, B)
    with pytest.raises(tq.ModelError, match=message):
        m.compile()


def test_ir_round_trip_and_version_0_1_files():
    m = ghost(True)
    ir = m.to_ir()
    assert ir.extinctions[0].noise_floor is True
    again = ModelIR.from_json(ir.to_json())
    assert again == ir and again.content_hash() == ir.content_hash()
    old = decaying().to_ir().to_dict()
    old["ir_version"] = "0.1"
    for e in old["extinctions"]:
        e.pop("noise_floor")
    loaded = ModelIR.from_dict(old)
    assert loaded.ir_version == IR_VERSION and loaded.extinctions[0].noise_floor is False
    assert loaded.extinctions[0].where == ()


# --------------------------------------------------------------------------- slices
def test_v17_one_group_per_strain_with_slices():
    """B[s] holds both strains in one field; B.at(s="S") and B.at(s="R") are separate groups:
    S is zeroed at its crossing, R (seeded by mutation, never armed early) is left alone."""
    m = strains()
    B, s = m["B"], m["s"]
    for strain in ("S", "R"):
        m.extinction(B.at(s=strain), threshold=1.0)
    sol = m.solve((0, 30), times=np.linspace(0, 30, 61))
    assert [g for _, g in sol.extinctions] == ["extinct_B_s_S"]
    te = sol.extinctions[0][0]
    S, R = sol["B"][:, 0], sol["B"][:, 1]
    assert np.all(S[sol.t >= te] == 0.0) and np.all(R[sol.t > 0] > 0)
    assert R[-1] > 1.0
    assert max(sol.ledger.check().values()) < 1e-10
    assert sol.ledger.moved["B"]["extinction"][-1] < 0
    assert s.type == "categorical"


def test_slices_over_a_spatial_field_and_the_noise_floor():
    """Two classes spread over x: the slice is a class over every x; the floor acts on it."""
    m = tq.Model("sliced")
    c = m.axis("c", labels=["a", "b"])
    x = m.axis("x", 0, 1, 4)
    B = m.field("B", axes=[x, c])
    m.diffuse(B, axis=x, D=0.1)
    m.reaction(B, 5.0 * B)
    m.init(B, 1e-12, where={c: "a"})
    m.init(B, 1.0, where={c: "b"})
    m.extinction(B.at(c="a"), threshold=0.5, noise_floor=True)
    sol = m.solve((0, 2), times=[2], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(sol["B"][-1][:, 0], 1e-12)  # held: a ghost cannot grow
    np.testing.assert_allclose(sol["B"][-1][:, 1], np.exp(10), rtol=1e-6)  # the other class


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda m, B: (m.extinction(B.at(s="S"), threshold=1), m.extinction(B, threshold=1)),
         "overlaps group"),
        (lambda m, B: (m.extinction(B.at(s="S"), threshold=1),
                       m.extinction(B.at(s="S"), threshold=2, name="again")), "overlaps group"),
    ],
    ids=["slice and whole", "same slice twice"],
)  # fmt: skip
def test_slices_may_not_overlap(build, message):
    m = strains()
    build(m, m["B"])
    with pytest.raises(tq.ModelError, match=message):
        m.compile()


def test_slices_are_checked_when_written():
    m = strains()
    with pytest.raises(ValueError, match="not a category"):
        m.extinction(m["B"].at(s="X"), threshold=1)
    with pytest.raises(TypeError, match="Only a field"):
        m["C"].at(s="S")
    two = strains()
    two.extinction(two["B"].at(s="S"), threshold=1)
    two.extinction(two["B"].at(s="R"), threshold=1)  # disjoint: fine
    ir = two.to_ir()
    assert ir.extinctions[0].where == ((("s", "S"),),)
    assert ModelIR.from_json(ir.to_json()) == ir


def test_v17_slice_crossing_time_counts_cell_widths():
    """A class spread over an uneven x grid: the group's amount is the density times the
    cell widths (20 here, whatever the grid), so it crosses 1 at ln(20)/k."""
    m = tq.Model("sliced_decay")
    c = m.axis("c", labels=["a", "b"])
    x = m.axis("x", faces=[0, 0.1, 0.5, 0.6, 1.4, 2.0])
    B = m.field("B", axes=[x, c])
    m.hazard(B, 0.5)
    m.init(B, 10.0)
    m.extinction(B.at(c="a"), threshold=1.0)
    sol = m.solve((0, 10), times=[10], **TIGHT)
    (te, name), *_ = sol.extinctions
    assert name == "extinct_B_c_a" and te == pytest.approx(np.log(20) / 0.5, rel=1e-8)
    assert np.all(sol["B"][-1][:, 0] == 0.0) and np.all(sol["B"][-1][:, 1] > 0)


@pytest.mark.parametrize("k_R", [0.7, 0.7 + 1e-15])
def test_v17_groups_crossing_at_the_same_instant_are_all_zeroed(k_R):
    """Two strains dying at the same rate: scipy reports one event; both must go."""
    m = tq.Model("twins")
    s = m.axis("s", labels=["S", "R"])
    B = m.field("B", axes=[s])
    m.param("k", {"S": 0.7, "R": k_R}, over=s)
    m.hazard(B, "k[s]")
    m.init(B, 100.0)
    for strain in ("S", "R"):
        m.extinction(B.at(s=strain), threshold=1.0)
    sol = m.solve((0, 20), times=[20], **TIGHT)
    assert sorted(g for _, g in sol.extinctions) == ["extinct_B_s_R", "extinct_B_s_S"]
    for te, _ in sol.extinctions:
        assert te == pytest.approx(np.log(100) / 0.7, rel=1e-8)
    np.testing.assert_array_equal(sol["B"][-1], 0.0)
    assert max(sol.ledger.check().values()) < 1e-12


def fed(rate: float):
    m = tq.Model("fed")
    P = m.compartment("P", init=0.0)
    m.ode(P, f"{rate} - P")  # fed from outside the group, cleared at 1/h
    m.extinction(P, threshold=1.0, noise_floor=True)
    return m


def test_noise_floor_with_a_feed_above_the_threshold():
    """Below the threshold the floored group does not see itself, so it fills at the feed
    rate (P = 2t) until it reaches 1 at t = 0.5; from then on it counts and relaxes to the
    feed level: P = 2 - exp(-(t - 0.5))."""
    t = np.array([0.25, 0.5, 1.0, 3.0])
    with pytest.warns(UserWarning, match="fed from outside"):
        sol = fed(2.0).solve((0, 3), times=t, **TIGHT)
    np.testing.assert_allclose(sol["P"], [0.5, 1.0, 2 - np.exp(-0.5), 2 - np.exp(-2.5)], rtol=1e-8)
    assert sol.extinctions == []


def test_noise_floor_with_a_feed_below_the_threshold_does_not_chatter():
    """Fed at 0.5 with clearance 1 (true level 0.5, below the threshold 1): the floored
    group fills to 1 by t = 2, counts, is cleared back below 1 at once and goes extinct; and
    again every 2 h. The floor switches with the group's armed state, never in between, so
    the solver does not chatter at the threshold (it used to fail here)."""
    with pytest.warns(UserWarning, match="fed from outside"):
        sol = fed(0.5).solve((0, 9), times=[9], **TIGHT)
    np.testing.assert_allclose([te for te, _ in sol.extinctions], [2, 4, 6, 8], rtol=1e-8)
    assert sol["P"][-1] == pytest.approx(0.5, rel=1e-8)
    assert max(sol.ledger.check().values()) < 1e-12


def floor_warnings(m) -> list[str]:
    report = m.validate(raise_on_error=False)
    return [i.message for i in report.warnings if i.where.startswith("extinction group")]


def pk_lineage(blood_in_group: bool):
    """Phage at the site fed from blood (the pbisim phage-PK floor bug)."""
    m = tq.Model("pk_lineage")
    P = m.compartment("P", init=0.0)
    Pc = m.compartment("Pc", init=10.0)
    m.transfer(Pc, to=P, rate=0.3)
    m.transfer(P, to=None, rate=1.0)
    m.extinction([P, Pc] if blood_in_group else [P], threshold=1.0, noise_floor=True)
    return m


def infection_lineage(group: str):
    """Infected cells I[age] born at age 0 at k*B*P; P is used up by adsorption."""
    m = tq.Model("lineage")
    a = m.axis("a", 0, 1, 4, kind="one_way")
    I = m.field("I", axes=[a])
    P = m.compartment("P", init=1.0)
    B = m.compartment("B", init=1e6)
    m.advect(I, axis=a, velocity=1.0)
    m.inflow(I, axis=a, at="lower", flux=1e-9 * B * P)
    m.outflow(I, axis=a, at="upper")
    m.ode(P, -1e-9 * B * P)
    m.ode(B, 0.5 * B - 1e-9 * B * P)
    m.init(I, 0.0)
    m.extinction([P, I] if group == "lineage" else [I], threshold=1.0, noise_floor=True)
    return m


def resistant_slice():
    m = strains()
    m.extinction(m["B"].at(s="R"), threshold=1.0, noise_floor=True)
    return m


@pytest.mark.parametrize(
    "build, feed",
    [
        (lambda: fed(0.5), "rate on 'P': 0.5 - P"),
        (lambda: ghost(True, source=0.1), "rate on 'P'"),
        (lambda: pk_lineage(False), "transfer Pc -> P"),
        (lambda: infection_lineage("infected only"), "inflow boundary of 'I'"),
        (resistant_slice, "mutate S -> R into it"),
    ],
    ids=["constant feed", "ghost with a source", "blood outside", "births", "mutation"],
)
def test_a_floored_group_fed_from_outside_is_warned_about(build, feed):
    """Decision 23: below its threshold a floored group does not see its own losses, so a
    feed from outside fills it; the modeller is told (at validation and at compile)."""
    (message,) = floor_warnings(build())
    assert feed in message and "Put the source in the group" in message
    with pytest.warns(UserWarning, match="fed from outside the group"):
        build().compile()


@pytest.mark.parametrize(
    "build",
    [
        lambda: ghost(True),
        lambda: pk_lineage(True),
        lambda: infection_lineage("lineage"),
        lambda: ghost(False, source=0.1),
    ],
    ids=["self-feeding ghost", "blood in the group", "whole lineage", "no floor"],
)
def test_a_group_that_feeds_itself_is_not_warned_about(build):
    """Growth, births from the group's own phage, and a source inside the group are not
    feeds from outside; without a floor there is nothing to warn about."""
    assert floor_warnings(build()) == []
