# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Regressions for the Phase 3 gate review (independent reviewers, 2026-10-23): the notation
reader and writer, the model comparison, explain(), fokker_planck, to_latex() and
diagnostics(sol). Each test is one finding, checked against the mathematics."""

import math

import numpy as np
import pytest

import tenqsp as tq
from tenqsp.canonical import differences
from tenqsp.writer import WriteError


def _flat(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------- reading and writing text


def test_a_flux_into_an_inflow_keeps_its_division_by_the_volume():
    m = tq.Model.from_text(
        "axis a in [0, 1] n=4 kind=one_way\nparam k=0.5, V=10\ncompartment Cc volume=V\n"
        "field I(a)\nflux J: Cc -> I = k*Cc\nd_t(Cc) = -J/V\nd_t(I) = -d_a(I)\n"
        "inflow I at a=lower: J\noutflow I at a=upper\nCc(0) = 1\n"
    )
    assert "d_t(Cc) = -J/V" in m.to_text()
    assert m.solve((0, 1), times=[1.0])["Cc"][-1] == pytest.approx(math.exp(-0.05), rel=1e-6)


def test_a_flux_summed_over_categories_into_an_inflow():
    m = tq.Model.from_text(
        "axis s categorical [S, R]\naxis a in [0, 1] n=4 kind=one_way\n"
        "param k[s] = {S: 1, R: 2}\ncompartment B\nfield I(a, s)\nflux J: B -> I = k[s]*B\n"
        "d_t(B) = -sum_s(J)\nd_t(I) = -d_a(I)\ninflow I at a=lower: J\n"
        "outflow I at a=upper\nB(0) = 1\n"
    )
    sol = m.solve((0, 1), times=[1.0], rtol=1e-10, atol=1e-12)
    assert sol["B"][-1] == pytest.approx(math.exp(-3), rel=1e-7)


def test_a_flux_may_be_used_in_a_place_that_is_not_one_of_its_ends():
    m = tq.Model.from_text(
        "param k=0.1\nflux infect: B -> I = k*B*P\nd_t(B) = -infect\nd_t(I) = infect\n"
        "d_t(P) = -infect\nB(0) = 1\nP(0) = 2\n"
    )
    sol = m.solve((0, 5), times=[5.0])
    assert sol["B"][-1] + sol["I"][-1] == pytest.approx(1.0, rel=1e-9)
    assert sol["P"][-1] - sol["B"][-1] == pytest.approx(1.0, rel=1e-6)  # P - B is constant


@pytest.mark.parametrize(
    ("flux", "expected"),
    [
        ("k0", 10 - 2 * 1.0),  # zero order from an amount
        ("k*A^2", 1 / (1 + 1.0)),  # A' = -A^2: A(1) = 1/2
    ],
)
def test_fluxes_that_are_not_a_rate_times_their_source(flux, expected):
    start = 10 if flux == "k0" else 1
    m = tq.Model.from_text(
        f"param k0=2, k=1\nflux J: A -> B = {flux}\nd_t(A) = -J\nd_t(B) = J\nA(0) = {start}\n"
    )
    sol = m.solve((0, 1), times=[1.0], rtol=1e-9, atol=1e-12)
    assert sol["A"][-1] == pytest.approx(expected, rel=1e-6)
    assert sol["A"][-1] + sol["B"][-1] == pytest.approx(start, rel=1e-9)


def test_a_flux_from_a_slice_may_name_the_slice():
    m = tq.Model.from_text(
        "axis s categorical [S, R]\nparam Q=1\nfield C(s)\ncompartment Cv\n"
        "flux J: C[s=S] -> Cv = Q*C[s=S]\nd_t(C[s=S]) = -J\nd_t(Cv) = J\nC(0) = 1\n"
    )
    sol = m.solve((0, 1), times=[1.0], rtol=1e-10, atol=1e-12)
    assert sol["C"][-1].tolist() == pytest.approx([math.exp(-1), 1.0], rel=1e-6)


def test_a_zero_order_flux_out_of_a_concentration_is_refused():
    with pytest.raises(tq.NotationError, match="rate times what it takes"):
        tq.Model.from_text(
            "param Q=1, V=2\ncompartment C volume=V\ncompartment D\nflux J: C -> D = Q\n"
            "d_t(C) = -J/V\nd_t(D) = J\n"
        )


def test_a_transition_of_another_slice_is_refused():
    with pytest.raises(tq.NotationError, match=r"equation of what moves, d_t\(B\[r=Y\]\)"):
        tq.Model.from_text(
            "axis s categorical [S, R]\naxis r categorical [X, Y]\nparam mu=0.1\nfield B(s, r)\n"
            "transitions M on s: S -> R = mu\nd_t(B) = trans_s(M, B[r=Y])\n"
        )


def test_a_formula_named_like_a_function_leaves_the_function_callable():
    text = "erf = 0.5\nd_t(X) = erf(X) - X\nX(0) = 1\n"
    m = tq.Model.from_text(text)
    api = tq.Model("x")
    X = api.compartment("X")
    api.define("erf", "0.5")
    api.ode(X, "erf(X) - X")
    api.init(X, 1.0)
    a = m.solve((0, 1), times=[1.0])["X"][-1]
    assert a == pytest.approx(api.solve((0, 1), times=[1.0])["X"][-1], rel=1e-9)


def test_a_slice_fixing_one_axis_twice_is_refused():
    for line in ("B[s=S, s=R](0) = 1", "d_t(B[s=S, s=R]) = -B", "observable o = B[s=S, s=R]"):
        with pytest.raises(tq.NotationError, match="fixed twice"):
            tq.Model.from_text(f"axis s categorical [S, R]\nfield B(s)\n{line}\n")


def test_a_transition_rate_may_continue_on_the_next_line():
    m = tq.Model.from_text(
        "axis s categorical [S, R]\nparam mu=0.1, nu=0.2\nfield B(s)\n"
        "transitions M on s: S -> R = max(mu,\n        nu), R -> S = nu\n"
        "d_t(B) = trans_s(M, B)\n"
    )
    assert "S -> R = max(mu, nu)" in m.to_text()


def test_options_are_not_swallowed():
    with pytest.raises(tq.NotationError, match="units are one word"):
        tq.Model.from_text("axis s categorical [S, R]\nparam kk[s] = {S: 1, R: 2} units=1/h, q=2\n")
    with pytest.raises(tq.NotationError, match="gives 1 cells but n=3"):
        tq.Model.from_text("axis q faces=[0, 1] n=3\n")


def test_a_boundary_problem_points_at_the_boundary_line():
    with pytest.raises(tq.NotationError, match=r"line 4: inflow I at a=lower: 2"):
        tq.Model.from_text(
            "axis a in [0, 1] n=4 kind=one_way\nfield I(a)\ninflow I at a=lower: 1\n"
            "inflow I at a=lower: 2\n"
        )


def test_a_conditional_outside_value_of_a_robin_end_reads_back():
    m = tq.Model("r")
    z = m.axis("z", 0, 1, 4)
    C = m.field("C", [z])
    m.params(h=1.0, cin=1.0, cin2=2.0)
    m.boundary(C, axis=z, at="upper", kind="robin", h="h", value="cin if t < 1 else cin2")
    back = tq.Model.from_text(m.to_text())
    assert differences(m.to_ir(), back.to_ir()) == []


def test_two_transports_with_different_schemes_cannot_be_written():
    m = tq.Model("w")
    z = m.axis("z", 0, 1, 4)
    C = m.field("C", [z])
    m.advect(C, axis=z, velocity="1")
    m.advect(C, axis=z, velocity="2", scheme="sharp")
    with pytest.raises(WriteError, match="different schemes"):
        m.to_text()


# ---------------------------------------------------------------- comparing two models


def _cmp(a: str, b: str) -> list:
    head = "axis x in [-1, 1] n=4\nparam v=-1, v1=1, v2=2, D1=0.1, D2=0.2\nfield F(x)\n"
    return differences(tq.Model.from_text(head + a).to_ir(), tq.Model.from_text(head + b).to_ir())


def test_the_comparison_adds_up_repeated_transport_and_diffusion():
    assert _cmp("d_t(F) = -d_x(v1*F) - d_x(v2*F)\n", "d_t(F) = -d_x(v2*F)\n")
    assert _cmp("d_t(F) = d_x(D1*d_x(F)) + d_x(D2*d_x(F))\n", "d_t(F) = d_x(D2*d_x(F))\n")
    assert not _cmp("d_t(F) = -d_x(v1*F) - d_x(v2*F)\n", "d_t(F) = -d_x((v1 + v2)*F)\n")


def test_the_comparison_tells_abs_v_from_v():
    assert _cmp("d_t(F) = -d_x(abs(v)*F)\n", "d_t(F) = -d_x(v*F)\n")


# ---------------------------------------------------------------- explain()


def _explain(text: str) -> str:
    return _flat(tq.Model.from_text(text).explain())


def test_two_transports_on_one_axis_are_one_transit_chain():
    t = _explain(
        "axis a in [0, 2] n=5 kind=one_way\nparam ktr=1\nfield X(a)\ncompartment A\n"
        "compartment C\nd_t(A) = -ktr*A\nd_t(X) = -d_a(X) - d_a(0.5*X)\n"
        "inflow X at a=lower: ktr*A\noutflow X at a=upper to C\nA(0) = 1\n"
    )
    assert "2 transport terms along a; together" in t
    assert "Mean transit time 1.33333" in t and "Mean transit time 2" not in t


def test_a_negative_velocity_per_category_runs_backwards():
    head = (
        "axis a in [0, 2] n=5\naxis s categorical [S, R]\nparam k[s] = {S: 1, R: 2}\n"
        "field X(a, s)\noutflow X at a=lower\n"
    )
    t = _explain(head + "d_t(X) = -d_a((-k[s])*X)\n")
    assert "running towards the lower end" in t and "rate |velocity|/width = 5*k[s]/2" in t
    t = _explain(head + "d_t(X) = -d_a((k[s] - 1.5)*X)\n")
    assert "differs in sign between categories" in t and "Mean transit" not in t


def test_no_transit_time_when_nothing_leaves_the_chain():
    t = _explain("axis a in [0, 2] n=5\nfield X(a)\nd_t(X) = -d_a(X)\n")
    assert "nothing leaves the chain" in t and "Mean transit" not in t


def test_explain_gives_no_sde_where_d_jumps():
    t = _explain(
        "axis x in [-2, 2] n=10\nfield u(x)\n"
        "d_t(u) = -d_x(-x*u) + d_x((0.1 if x < 0 else 0.5)*d_x(u))\n"
    )
    assert "not the density of a single SDE" in t and "sqrt(2 D) dW" not in t


# ---------------------------------------------------------------- fokker_planck


def _fp(D: str) -> str:
    m = tq.Model.from_text("axis x in [-2, 2] n=10\nfield u(x, y)\naxis y in [0, 1] n=2\n")
    m.fokker_planck(m["u"], axis=m["x"], drift="-x", D=D)
    return next(line for line in m.to_text().splitlines() if line.startswith("d_t(u)"))


def test_fokker_planck_accepts_what_it_can_write_exactly():
    assert "0.2469134*x" in _fp("0.1234567*x^2 + 0.1")
    assert "3.141592653589793" in _fp("0.1 + 0.05*erf(x)")
    for D in ("0.1 if t < 1 else 0.2", "0.1*min(1, t)"):  # jumps in time only: slope 0
        assert "d_x((-x)*u)" in _fp(D)


def test_fokker_planck_refuses_what_it_cannot_read():
    with pytest.raises(ValueError, match="cannot read D"):
        _fp("0.1*int_y(u)")


# ---------------------------------------------------------------- to_latex()


def test_latex_divides_transitions_of_weighted_fields_by_the_weight():
    m = tq.Model.from_text(
        "axis s categorical [A, B]\nparam Q=2\nparam V[s] = {A: 1, B: 10}\n"
        "field C(s) weights=V[s]\ntransitions X on s: A -> B = Q\nd_t(C) = trans_s(X, C)\n"
        "C[s=A](0) = 1\n"
    )
    tex = m.to_latex()
    assert r"\frac{1}{V_{B}}\left(Q \, C_{A}\right)" in tex
    rhs = np.asarray(m.compile().rhs(0.0, m.compile().initial_state()))
    assert rhs[:2].tolist() == pytest.approx([-2.0, 0.2])


def test_latex_brackets_sums_powers_of_numbers_and_fractions():
    tex = tq.Model.from_text(
        "axis s categorical [S, R]\nparam g=1, k[s] = {S: 1, R: 2}\nfield B(s)\ncompartment P\n"
        "d_t(B) = -sum_s(B)*k[s]*B - (k[s]/g)^2*B\nd_t(P) = 1e8^2 - 1.5e4^2*P + -g*P - -g\n"
    ).to_latex()
    assert r"-\left(\sum_{s} B\right) \, k_{s} \, B" in tex
    assert r"\left(\frac{k_{s}}{g}\right)^{2}" in tex
    assert r"\left(10^{8}\right)^{2} - \left(1.5 \times 10^{4}\right)^{2}" in tex
    assert "+ -" not in tex and "- -" not in tex


def test_latex_says_a_slice_equation_adds_to_the_field_equation():
    tex = tq.Model.from_text(
        "axis s categorical [S, R]\nparam k=1\nfield B(s)\nd_t(B) = -k*B\nd_t(B[s=S]) = 1\n"
    ).to_latex()
    assert r"\text{(added to the equation of } B\text{)}" in tex


# ---------------------------------------------------------------- diagnostics(sol)


def _issues(m, sol, kind=None):
    return [i.message for i in m.diagnostics(sol).issues if kind is None or i.level == kind]


def test_no_regrowth_warning_for_a_place_refilled_through_another_one():
    m = tq.Model.from_text(
        "axis a in [0, 2] n=5 kind=one_way\nparam ka=1, ke=1\nfield X(a)\ncompartment A\n"
        "compartment C\nd_t(A) = -ka*A\nd_t(X) = -d_a(X)\ninflow X at a=lower: ka*A\n"
        "outflow X at a=upper to C\nd_t(C) = -ke*C\ndose A at t=0 to 72 every 24: 1\n"
    )
    sol = m.solve((0, 96), times=np.linspace(0, 96, 961))
    assert not [w for w in _issues(m, sol, "warning") if "grew back" in w]


def test_a_regrowth_after_a_refilled_dip_is_still_found():
    m = tq.Model.from_text(
        "param g=2, kill=6.5\ncompartment B\n"
        "d_t(B) = (g - kill*(1 if (t < 5 or (t > 10 and t < 15)) else 0) "
        "- g*(1 if t < 10 else 0))*B\nB(0) = 1e6\ndose B at t=10: 1e6\n"
    )
    sol = m.solve((0, 40), times=np.linspace(0, 40, 401), atol=1e-3)
    assert [w for w in _issues(m, sol, "warning") if "grew back" in w]


def test_a_booking_compartment_may_go_negative():
    m = tq.Model.from_text(
        "axis z in [0, 1] n=4\nparam h=1\nfield C(z)\ncompartment W\n"
        "robin C at z=upper: h*(2 - C) with W\nd_t(W) = 0\nC(0) = 1\n"
    )
    sol = m.solve((0, 5), times=np.linspace(0, 5, 11))
    assert float(np.min(sol["W"])) < 0
    assert not [w for w in _issues(m, sol) if w.startswith("went negative")]
