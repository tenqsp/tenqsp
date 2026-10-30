# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Step 3.5: m.explain(), m.diagnostics(), m.fokker_planck, m.to_latex().

Oracles come from the mathematics, not from the code:

- an erlang transport is the transit chain explain() says it is: the cell amounts of a
  solved model equal those of N compartments with the stated rate (matrix exponential);
- fokker_planck's drift correction gives the stationary density of the SDE
  dz = w dt + sqrt(2 D(z)) dW in closed form, and leaving the correction out does not;
- the transitions written out by to_latex are the arrivals minus the departures of §3.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

import tenqsp as tq
from tenqsp.canonical import differences
from tenqsp.examples import toy_phage_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_gallery_phase2 import friberg  # noqa: E402
from test_v12_round_trip import MODELS  # noqa: E402

CHAIN = """
model chain
axis z in [0, {L}] n={n}
param v=2
field T(z)
d_t(T) = -d_z(v*T)
outflow T at z=upper
T(0) = 1*point(z, at=0)
"""


def _claimed(text: str, what: str) -> float:
    m = re.search(rf"{what}[ =]+[^(]*\(= ([0-9.e+-]+)\)", text.replace("\n", " "))
    assert m, f"no '{what}' in:\n{text}"
    return float(m.group(1))


# ------------------------------------------------------------------------- explain


@pytest.mark.parametrize("L,n", [(1, 4), (4, 4), (3, 3)])
def test_erlang_transport_is_the_transit_chain_explain_describes(L, n):
    m = tq.Model.from_text(CHAIN.format(L=L, n=n))
    text = " ".join(m.explain().split())
    assert f"exactly a chain of {n} transit compartments" in text
    k = _claimed(text, r"rate \|velocity\|/width")
    assert k == pytest.approx(2 * n / L)
    assert _claimed(text, "Mean transit time") == pytest.approx(n / k)  # Erlang mean
    assert f"CV 1/sqrt({n}) = {1 / np.sqrt(n):.3g}" in text
    # oracle: dA_i/dt = k A_(i-1) - k A_i, all of the amount in the first compartment
    A = -k * np.eye(n) + k * np.eye(n, k=-1)
    t = np.array([0.3, 1.0, 2.5]) * L / 2
    sol = m.solve((0, t[-1]), times=t, rtol=1e-10, atol=1e-12)
    width = L / n
    for i, ti in enumerate(t):
        expected = scipy.linalg.expm(A * ti) @ np.eye(n)[0]
        np.testing.assert_allclose(sol["T"][i] * width, expected, rtol=1e-6, atol=1e-9)


def test_transit_rate_is_written_with_the_parameters():
    """Notation §6.2: width-1 cells give rate ktr; width-1/3 cells need speed ktr/3."""
    m = tq.Model.from_text(CHAIN.format(L=3, n=3).replace("v=2", "ktr=0.5").replace("v*", "ktr*"))
    text = " ".join(m.explain().split())
    assert "each cell is one transit compartment (density = amount)" in text
    assert "rate |velocity|/width = ktr (= 0.5)" in text
    assert "Mean transit time 3/ktr (= 6)" in text
    text = " ".join(friberg().explain().split())  # speed ktr/3, ktr = 4/mtt, cells 1/3 wide
    assert "of width 1/3 is one transit compartment (amount = density x 1/3)" in text
    assert "rate |velocity|/width = ktr (= 0.04). Mean transit time 3/ktr (= 75)" in text


def test_explain_names_each_scheme_and_transition_source():
    text = " ".join(toy_phage_model().explain().split())
    assert "transport along a, velocity 1 (sharp): MUSCL with the van Leer limiter" in text
    assert "about 10x" in text
    assert (
        "The velocity varies along z (it may change sign), so this is not a fixed transit" in text
    )
    assert "trans_s(M, div) — redistributes the production div" in text
    assert "at the parent's rate" in text
    m = tq.Model("sw")
    s = m.axis("s", labels=["S", "R"])
    B = m.field("B", axes=[s])
    M = m.transitions("M", axis=s, rates={"S->R": 0.1})
    m.transition(B, axis=s, matrix=M, of=B)
    text = " ".join(m.explain().split())
    assert "the cells themselves change category (first-order switching" in text
    assert "moves S -> R at 0.1" in text


def test_explain_lists_what_starts_at_zero():
    text = " ".join(toy_phage_model().explain().split())
    assert "Start at 0 (not given, or given as 0): B[s=R], I, P." in text
    assert "B[s=S]: an amount 1e7 spread like normal(z, mean=0.1, sd=0.08)" in text
    m = tq.Model.from_text(
        """
axis s categorical [S, R, X]
axis z in [0, 1] n=4
field B(s, z)
field u(z)
d_t(B) = 0
d_t(u) = 0
B[s=S](0) = 5*point(z, at=0)
u(0) = density(normal(z, mean=0.5, sd=0.2))
"""
    )
    text = " ".join(m.explain().split())
    assert "B[s=S]: all of 5 in the one cell containing z, at=0." in text
    assert "the curve itself as a density, not scaled to an amount" in text
    assert "Start at 0 (not given, or given as 0): the rest of B." in text


def test_explain_matches_the_same_term_across_equations():
    """Written as a paper writes it, each equation carries its own term; explain() pairs
    them (decided 2026-10-28: no flux declaration required) and shows a conversion factor."""
    hand = tq.Model.from_text(
        """
param Q=2, V1=1, V2=3
d_t(A) = -Q*A/V1
d_t(B) = Q*A/V2 + 0.1*(5/A)^2
A(0) = 1
"""
    )
    text = " ".join(hand.explain().split())
    shared = text.split("Terms shared between equations")[1].split("Boundaries")[0]
    assert "A*Q/V1: A -1, B x (V1/V2)." in shared
    assert "Amounts balance" not in shared  # no volumes declared: TenQSP cannot tell
    with_volumes = tq.Model.from_text(
        """
param Q=2, V1=1, V2=3
compartment A volume=V1
compartment B volume=V2
d_t(A) = -Q*A/V1
d_t(B) = Q*A/V2
A(0) = 1
"""
    )
    assert "Amounts balance" in " ".join(with_volumes.explain().split())
    assert "(5/A)" not in shared  # read inside a power: feedback, not material
    flux = tq.Model.from_text(
        """
param Q=2, V1=1, V2=3
flux J: A -> B = Q*A
d_t(A) = -J
d_t(B) = J
A(0) = 1
"""
    )
    assert "Terms shared between equations" not in flux.explain()
    assert "A -> B: amount Q x A per time" in " ".join(flux.explain().split())


PHAGE = """
param k=1e-9, b=50, w=0.1, g=1, v=2
axis l in [0, 1] n=5 kind=one_way
field I(l)
d_t(B) = g*B - k*B*P
d_t(I) = -d_l(v*I)
    at l=0: v*I = k*B*P
    at l=1: outflow
d_t(P) = b*v*I(l=1) - k*B*P - w*P
B(0) = 1e6
P(0) = 1e7
"""


def test_a_forgotten_partner_term_is_pointed_out():
    good = " ".join(tq.Model.from_text(PHAGE).explain().split())
    assert "B -1, I at l=0 +1, P -1" in good or "B -1, P -1, I at l=0 +1" in good
    assert "P x (b), I leaving at l=1 -1" in good
    assert "Not in the equation of" not in good
    bad = " ".join(tq.Model.from_text(PHAGE.replace("- k*B*P - w*P", "- w*P")).explain().split())
    assert "Not in the equation of P" in bad


def test_explain_lists_every_safeguard_and_noise_floors():
    m = tq.Model.from_text(
        """
param g=1
d_t(P) = g*P
P(0) = 10
extinction lineage: P below 1 with noise_floor
"""
    )
    text = " ".join(m.explain().split())
    assert "No value is clipped, capped or floored" in text
    assert "retried once with Radau" in text
    assert "Noise floors (declared): lineage" in text
    assert "a group that has never reached the threshold" in text


def test_explain_works_on_a_model_with_errors():
    m = tq.Model("bad")
    P = m.compartment("P")
    m.ode(P, "k*P")  # k undefined
    text = m.explain()
    assert "ERRORS" in text and "'k'" in text


@pytest.mark.parametrize("name", list(MODELS))
def test_explain_and_latex_run_on_every_round_trip_model(name):
    m = MODELS[name]()
    assert m.explain().startswith(f"Model '{m.name}'")
    _check_latex(m.to_latex())


# ------------------------------------------------------------------------- Fokker–Planck

FP = """
axis z in [0, 1] n={n}
param w=-1, D0=0.1
field p(z)
p(0) = 1*uniform(z, lower=0, upper=1)
"""


def _stationary(n, *, corrected):
    m = tq.Model.from_text(FP.format(n=n))
    p, z = m["p"], m["z"]
    if corrected:  # sharp, the default
        m.fokker_planck(p, axis=z, drift="w", D="D0*(1 + z)")
    else:  # the drift written as a velocity, no correction
        m.advect(p, axis=z, velocity="w", scheme="sharp")
        m.diffuse(p, axis=z, D="D0*(1 + z)")
    sol = m.solve((0, 40), times=[40.0], rtol=1e-9, atol=1e-12)
    return m, sol["p"][-1]


def _law(n, power):
    """Cell averages of (1+z)^-power on n cells of [0, 1], total 1."""
    faces = np.linspace(0, 1, n + 1)
    F = -((1 + faces) ** (1 - power)) / (power - 1)
    c = np.diff(F) / np.diff(faces)
    return c / (c.sum() / n)


def _l1(x, ref):
    return float(np.abs(x - ref).sum() / ref.sum())


def test_fokker_planck_gives_the_stationary_density_of_the_sde():
    """dz = -dt + sqrt(2 D) dW with D = 0.1 (1 + z), sealed ends: zero flux w p = d(D p)/dz
    gives p = C/D exp(int w/D) = C (1+z)^-11; without the drift correction, (1+z)^-10."""
    m, p80 = _stationary(80, corrected=True)
    _, p160 = _stationary(160, corrected=True)
    e80, e160 = _l1(p80, _law(80, 11)), _l1(p160, _law(160, 11))
    assert e160 < 1e-3 and e80 / e160 > 3.5  # second order (sharp)
    _, naive = _stationary(160, corrected=False)
    assert _l1(naive, _law(160, 10)) < 1e-3  # the uncorrected terms solve another law
    assert _l1(naive, _law(160, 11)) > 5e-2  # ... measurably far from the SDE's
    adv = next(t for t in m.to_ir().terms if t.kind == "advection")
    assert adv.expr.replace(" ", "") == "w-D0" and adv.scheme == "sharp"


def test_fokker_planck_is_the_two_terms_written_by_hand():
    a = tq.Model.from_text(FP.format(n=10))
    a.fokker_planck(a["p"], axis=a["z"], drift="w*z", D="D0*exp(z)")
    b = tq.Model.from_text(
        FP.format(n=10)
        + "d_t(p) = -d_z((w*z - D0*exp(z))*p) + d_z(D0*exp(z)*d_z(p))  using sharp on z\n"
    )
    assert differences(a.to_ir(), b.to_ir()) == []
    assert "Fokker–Planck equation" in a.explain()
    c = tq.Model.from_text(FP.format(n=10))
    c.fokker_planck(c["p"], axis=c["z"], drift="w", D="D0")  # constant D: no correction
    assert next(t for t in c.to_ir().terms if t.kind == "advection").expr == "w"


def test_fokker_planck_refusals():
    m = tq.Model.from_text(FP.format(n=10) + "field S(z)\nd_t(S) = 0\n")
    with pytest.raises(ValueError, match="varies along 'z' through 'S'"):
        m.fokker_planck(m["p"], axis=m["z"], drift="w", D="D0*S")
    with pytest.raises(ValueError, match="jumps or has a kink"):
        m.fokker_planck(m["p"], axis=m["z"], drift="w", D="max(z, 0.5)")
    assert not [t for t in m.to_ir().terms if t.target == "p"]  # nothing half-added


# ------------------------------------------------------------------------- diagnostics


def test_diagnostics_peclet_crossing_time_and_cv():
    m = tq.Model.from_text(
        """
axis z in [0, 1] n=10
param v=1, D=0.001
field u(z)
d_t(u) = -d_z(v*u) + d_z(D*d_z(u))
outflow u at z=upper
u(0) = 1
"""
    )
    rep = str(m.diagnostics())
    assert "crosses a cell in 0.1 time units" in rep
    assert "chain of 10 transit compartments, CV 0.316" in rep
    assert "cell Péclet number |v| x width / D is up to 100" in rep
    assert "the spread you see is mostly the grid's" in rep
    neg = tq.Model.from_text(
        "axis z in [0, 1] n=5\nfield u(z)\nd_t(u) = d_z((0.1 - z)*d_z(u))\nu(0) = 1\n"
    )
    assert any("diffusivity is negative" in i.message for i in neg.diagnostics().warnings)


def test_diagnostics_warns_of_regrowth_from_below_atol():
    text = """
d_t(B) = (-10 if t < 5 else 10)*B + (1e-7 if t > 5 else 0)
B(0) = 1
"""
    m = tq.Model.from_text(text)
    sol = m.solve((0, 10), times=np.linspace(0, 10, 101), atol=1e-6)
    assert sol.options["method"] == "BDF" and sol.options["max_step"] == np.inf
    assert sol.options["rtol"] == 1e-6 and sol.options["atol"] == 1e-6
    rep = m.diagnostics(sol)
    assert any("fell below atol" in i.message and i.where == "B" for i in rep.warnings)
    guarded = tq.Model.from_text(text + "extinction B below 1e-3\n")
    sol = guarded.solve((0, 10), times=np.linspace(0, 10, 101), atol=1e-6)
    assert not any("fell below atol" in i.message for i in guarded.diagnostics(sol).warnings)


def test_diagnostics_notes_a_population_absorbed_below_atol():
    """e^-50 e^50 = 1 exactly, but below atol BDF's undershoot makes zero absorbing (the
    pbisim case): the solve stays at ~0, and diagnostics says so."""
    m = tq.Model.from_text("d_t(B) = (-10 if t < 5 else 10)*B\nB(0) = 1\n")
    sol = m.solve((0, 10), times=np.linspace(0, 10, 101), atol=1e-6)
    assert abs(sol["B"][-1]) < 1e-3  # the exact value is 1
    rep = m.diagnostics(sol)
    assert any("stayed there" in i.message and i.where == "B" for i in rep.notes)


def test_diagnostics_negative_values_and_ledger():
    m = tq.Model.from_text("d_t(P) = -1\nP(0) = 1\n")
    sol = m.solve((0, 3), times=[0, 3.0])
    rep = m.diagnostics(sol)
    assert any("went negative" in i.message for i in rep.warnings)
    assert any(i.where == "ledger" for i in rep.notes)


def test_diagnostics_of_the_toy_model_with_a_solution():
    m = toy_phage_model()
    sol = m.solve((0, 2), times=np.linspace(0, 2, 9), rtol=1e-6, atol=1e-3)
    rep = str(m.diagnostics(sol))
    assert "field 'I', axis 'a': fastest transport crosses a cell in 0.0375" in rep
    assert "over the run that is up to 53.3 cells" in rep
    assert "with sharp the solver follows the front cell by cell" in rep


# ------------------------------------------------------------------------- LaTeX


def _check_latex(tex: str) -> None:
    assert tex.startswith("\\begin{align*}") and tex.rstrip().endswith("\\end{align*}")
    depth = 0
    for i, ch in enumerate(tex):
        if ch == "{" and tex[i - 1] != "\\":
            depth += 1
        elif ch == "}" and tex[i - 1] != "\\":
            depth -= 1
        assert depth >= 0
    assert depth == 0
    assert tex.count("\\left") == tex.count("\\right")
    assert "**" not in tex and "d_t(" not in tex and "__slice__" not in tex


def test_latex_writes_transitions_out_per_category():
    tex = toy_phage_model().to_latex()
    _check_latex(tex)
    assert (
        r"\mathcal{T}^{M}_{s}\!\left[\mathrm{div}\right]_{S} &= \mu \, \mathrm{div}_{R} "
        r"- \mu \, \mathrm{div}_{S}"
    ) in tex
    assert r"\frac{\partial B}{\partial t} &=" in tex
    assert r"\frac{\mathrm{d} P}{\mathrm{d} t} &= \beta \, \mathrm{lysis}" in tex
    assert r"B_{\mathrm{tot}} &= \int_{0}^{1} \sum_{s} B \, \mathrm{d}z" in tex
    assert r"B_{S}(0) &= 10^{7} \, \operatorname{normal}\left(z;\ " in tex


def test_latex_of_joint_transitions_fluxes_and_named_totals():
    tex = MODELS["two phages"]().to_latex()
    assert r"\mathcal{T}^{X}_{r_{1},r_{2}}\!\left[\mathrm{div}\right]_{S,\,S} &= -" in tex
    assert r"k_{1}\left[r_{1}\right] \, P_{1}" in tex  # k1[r1]*P1
    pbpk = MODELS["PBPK"]().to_latex()
    assert r"\left(\mathrm{Cart} \to C_{\mathrm{vascular}}\right)" in pbpk
    assert r"C_{\mathrm{liver},\,\mathrm{cellular}}" in pbpk
    m = tq.Model("auto")  # a total the library named itself is written in place
    z = m.axis("z", 0, 2, 4)
    u = m.field("u", axes=[z])
    P = m.compartment("P")
    m.ode(P, m.integral(u, over=z))
    tex = m.to_latex()
    assert r"\frac{\mathrm{d} P}{\mathrm{d} t} &= \int_{0}^{2} u \, \mathrm{d}z" in tex
    assert "int_z" not in tex


# ------------------------------------------------- found by the independent verification


def test_fokker_planck_brackets_a_conditional_drift_and_refuses_a_jumping_D():
    a = tq.Model.from_text(FP.format(n=10))
    a.fokker_planck(a["p"], axis=a["z"], drift="w if z < 0.5 else 2*w", D="D0*(1 + z)")
    b = tq.Model.from_text(
        FP.format(n=10)
        + "d_t(p) = -d_z(((w if z < 0.5 else 2*w) - D0)*p) + d_z(D0*(1 + z)*d_z(p))"
        + "  using sharp on z\n"
    )
    assert differences(a.to_ir(), b.to_ir()) == []
    t = [0.0, 1.0]
    np.testing.assert_allclose(a.solve((0, 1), times=t)["p"], b.solve((0, 1), times=t)["p"])
    for D in ("D0 if z < 0.5 else 4*D0", "D0*abs(z - 0.5)", "max(D0, z)"):
        with pytest.raises(ValueError, match="jumps or has a kink"):
            a.fokker_planck(a["p"], axis=a["z"], drift="w", D=D)


def test_explain_fokker_planck_drift_keeps_brackets_and_simplifies():
    m = tq.Model.from_text(
        FP.format(n=10) + "d_t(p) = -d_z((w if z < 0.5 else 2*w)*p) + d_z(D0*(1 + z)*d_z(p))\n"
    )
    text = " ".join(m.explain().split())
    assert "dz = ((w if z < 0.5 else 2 * w) + D0) dt" in text
    m = tq.Model.from_text(FP.format(n=10))
    m.fokker_planck(m["p"], axis=m["z"], drift="w", D="D0*(1 + z)")
    assert "dz = (w) dt" in " ".join(m.explain().split())  # w - D0 + D0


def test_explain_negative_zero_and_changing_velocities():
    def transport(v):
        m = tq.Model.from_text(
            f"axis z in [0, 1] n=4\nfield T(z)\nd_t(T) = -d_z(({v})*T)\n"
            "outflow T at z=lower\noutflow T at z=upper\n"
        )
        t = " ".join(m.explain().split())
        return t[t.index("transport") : t.index("Boundaries")]

    back = transport("-1")
    assert "running towards the lower end of z" in back
    assert "rate |velocity|/width = 4. Mean transit time 1," in back
    assert "the velocity is 0, so nothing moves" in transport("0")
    changing = transport("1 + t")
    assert "changes with time" in changing and "hold only while the velocity stays" in changing
    assert "Mean transit time 1/" not in changing
    m = tq.Model.from_text("axis z in [0, 3.14159] n=3\nfield T(z)\nd_t(T) = -d_z(T)\n")
    assert "width 1.0472 is one" in " ".join(m.explain().split())


def test_diagnostics_a_dose_refilling_is_not_regrowth():
    m = tq.Model.from_text("param k=1\nd_t(C) = -k*C\ndose C at t=0, 24: 100\n")
    sol = m.solve((0, 30), times=np.linspace(0, 30, 61), atol=1e-6)
    rep = m.diagnostics(sol)
    assert not any("atol" in i.message for i in rep.issues)


def test_latex_brackets_sums_in_rates_and_escapes_labels():
    m = tq.Model.from_text(
        """
param a=2, b=1
axis s categorical [S, R]
field B(s)
transitions M on s: S -> R = a + b, R -> S = a - b
d_t(B) = trans_s(M, B)
B[s=S](0) = 1
"""
    )
    tex = m.to_latex()
    assert r"&= \left(a - b\right) \, B_{R} - \left(a + b\right) \, B_{S}" in tex
    m = tq.Model.from_text(
        'axis s categorical ["50%", "a&b"]\nfield B(s)\nd_t(B) = -B\nB[s="50%"](0) = 1\n'
    )
    tex = m.to_latex()
    _check_latex(tex)
    assert r"B_{\text{50\%}}(0)" in tex
