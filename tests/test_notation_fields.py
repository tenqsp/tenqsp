# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Phase 3 step 3.3: the notation reader for fields (docs/language/notation.md).

The models of the notation document, read from text, must solve to the same numbers as
the same models built with the Python API (which earlier phases verified against closed
forms and hand-written ODEs); each model here is written as in the document."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

import tenqsp as tq
from tenqsp.examples import phage_latency_model, toy_phage_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pbpk_models import pbpk  # noqa: E402
from test_gallery_phase2 import friberg  # noqa: E402
from test_transitions_phase2 import two_phage  # noqa: E402
from test_v_boundaries_phase2 import biofilm  # noqa: E402

TIGHT = {"rtol": 1e-9, "atol": 1e-9}
DOC = Path(__file__).resolve().parents[1] / "docs" / "language" / "notation.md"


def same(text: str, python: tq.Model, t_end: float, names, agree=1e-7, **solver):
    """Both models solve to the same values of every listed output, within ``agree``
    (``solver`` overrides the tight default tolerances)."""
    t = np.linspace(0, t_end, 7)
    opts = {**TIGHT, **solver}
    a = tq.Model.from_text(text).solve((0, t_end), times=t, **opts)
    b = python.solve((0, t_end), times=t, **opts)
    for n in names:
        scale = np.max(np.abs(b[n]))
        np.testing.assert_allclose(a[n], b[n], rtol=agree, atol=agree * scale, err_msg=n)
    return a


PHAGE_LATENCY = """
model phage_latency
axis a in [0, 2] n=40 units=h kind=one_way
param k=1e-9, beta=50, g=1, Bmax=1e9, delta=0.1, h_max=5, tau=0.8, s=0.05
compartment B
compartment P
field I(a)
h(a) = h_max*sigmoid((a - tau)/s)
flux infect: B -> I = k*B*P
d_t(B) = g*B*(1 - B/Bmax) - infect
d_t(I) = -d_a(I) - h(a)*I          using sharp
inflow  I at a=lower: infect
outflow I at a=upper
d_t(P) = beta*int_a(h(a)*I) - infect - delta*P
B(0) = 1e6
P(0) = 1e7
dose P at t=0: 1e8
observable burst_rate = int_a(h(a)*I)
"""


def test_phage_latency_prd_example():
    same(PHAGE_LATENCY, phage_latency_model(), 6.0, ["B", "P", "I", "burst_rate"])


FRIBERG = """
model friberg
axis z in [0, 3] n=3 kind=one_way
param mtt=100, gamma=0.17, slope=0.5, ke=0.1, circ0=5
ktr = 4/mtt
field Transit(z)
flux mature: Prol -> Transit = ktr*Prol
d_t(Prol)    = ktr*Prol*(1 - slope*Conc)*(circ0/Circ)^gamma - mature
d_t(Transit) = -d_z(ktr*Transit)
inflow  Transit at z=lower: mature
outflow Transit at z=upper to Circ
d_t(Circ)    = -ktr*Circ
d_t(Conc)    = -ke*Conc
Prol(0)    = circ0
Transit(0) = circ0
Circ(0)    = circ0
dose Conc at t=0: 1.8
"""


def test_friberg_with_cells_of_width_one_is_the_published_model():
    same(FRIBERG, friberg(dose=1.8), 600.0, ["Prol", "Circ"])


TOY = """
model phage_toy
axis z in [0, 1] n=30
axis s categorical [S, R]
axis a in [0, 1.5] n=40 units=h kind=one_way
param g_max=1, K=1e9, beta=50, delta=0.1, h_max=5, tau=0.8, s_h=0.05, w_amp=0.5,
      Dz=0.01, mu=1e-6
param k[s] = {S: 1e-9, R: 1e-12}
param cost[s] = {S: 0, R: 0.1}
field B(z, s)
field I(a, z)
compartment P
B_tot     = int_z(sum_s(B))
div       = g_max*(1 - z)*(1 - cost[s])*B*(1 - B_tot/K)
h(a)      = h_max*sigmoid((a - tau)/s_h)
lysis     = int_a(int_z(h*I))
flux infect: B -> I = k[s]*(1 - 0.5*z)*B*P
transitions M on s: S -> R = mu, R -> S = mu
d_t(B) = div - infect + trans_s(M, div) - d_z(w_amp*(0.3 - z)*B) + d_z(Dz*d_z(B))
d_t(I) = -d_a(I) - h*I + d_z(Dz*d_z(I))           using sharp on a
inflow  I at a=lower: sum_s(infect)
outflow I at a=upper
d_t(P) = beta*lysis - int_z(sum_s(infect)) - delta*P
B[s=S](0) = 1e7*normal(z, mean=0.1, sd=0.08)
dose P at t=0: 1e9
observable bacteria = B_tot
observable phage = P
"""


def test_toy_phage_model():
    sol = same(
        TOY,
        toy_phage_model(),
        12.0,
        ["bacteria", "phage", "B", "I"],
        agree=1e-6,
        rtol=1e-7,
        atol=1e-3,
    )  # cells of 1e7: a thousandth of a cell is plenty (tighter takes ~20 s per solve)
    assert max(sol.ledger.check().values()) < 1e-9


TWO_PHAGE = """
model two_phage
axis r1 categorical [S, R]
axis r2 categorical [S, R]
param g=1, K=1e9, beta=50, dp=0.1, mu=1e-6, mux=1e-8
param k1[r1] = {S: 1e-9, R: 0}
param k2[r2] = {S: 2e-9, R: 0}
field B(r1, r2)
N   = sum_r1(sum_r2(B))
A1  = sum_r1(sum_r2(k1[r1]*B))
A2  = sum_r1(sum_r2(k2[r2]*B))
div = g*B*(1 - N/K)
transitions M1 on r1: S -> R = mu
transitions M2 on r2: S -> R = mu
transitions X on (r1, r2): (S, S) -> (R, R) = mux
d_t(B)  = div - k1[r1]*P1*B - k2[r2]*P2*B
          + trans_r1(M1, div) + trans_r2(M2, div) + trans_{r1,r2}(X, div)
d_t(P1) = (beta - 1)*A1*P1 - dp*P1
d_t(P2) = (beta - 1)*A2*P2 - dp*P2
B[r1=S, r2=S](0) = 1e6
P1(0) = 1e4
P2(0) = 1e4
"""


def test_two_phage_joint_transitions():
    same(TWO_PHAGE, two_phage()[0], 24.0, ["B", "P1", "P2"], agree=1e-6)


PBPK = """
model pbpk
axis organ categorical [lung, brain, heart, muscle, adipose, kidney, gut, spleen, liver]
axis sub categorical [vascular, interstitial, cellular]
param V[organ, sub] = {
    lung:    {vascular: 0.06,   interstitial: 0.18,   cellular: 0.96},
    brain:   {vascular: 0.07,   interstitial: 0.21,   cellular: 1.12},
    heart:   {vascular: 0.0165, interstitial: 0.0495, cellular: 0.264},
    muscle:  {vascular: 1.45,   interstitial: 4.35,   cellular: 23.2},
    adipose: {vascular: 0.75,   interstitial: 2.25,   cellular: 12},
    kidney:  {vascular: 0.0155, interstitial: 0.0465, cellular: 0.248},
    gut:     {vascular: 0.06,   interstitial: 0.18,   cellular: 0.96},
    spleen:  {vascular: 0.0095, interstitial: 0.0285, cellular: 0.152},
    liver:   {vascular: 0.09,   interstitial: 0.27,   cellular: 1.44},
}
param Kp[organ]    = {lung: 1, brain: 0.5, heart: 2, muscle: 1.5, adipose: 5, kidney: 2,
                      gut: 2, spleen: 1.5, liver: 3}
param Q_art[organ] = {lung: 0, brain: 45, heart: 15, muscle: 60, adipose: 15, kidney: 70,
                      gut: 65, spleen: 10, liver: 20}
param Q_ven[organ] = {lung: 0, brain: 45, heart: 15, muscle: 60, adipose: 15, kidney: 70,
                      gut: 0, spleen: 0, liver: 95}
param PS1=50, PS2=10, CLint=30, Kp_liver=3, Q_CO=300, R=0, V_art=1.7, V_ven=3.4,
      Q_gut=65, Q_spleen=10
field C(organ, sub) weights=V[organ, sub]
compartment Cart volume=V_art
compartment Cven volume=V_ven
transitions X on sub:
    vascular -> interstitial = PS1
    interstitial -> vascular = PS1
    interstitial -> cellular = PS2
    cellular -> interstitial = PS2/Kp[organ]
transitions portal on organ: gut -> liver = Q_gut, spleen -> liver = Q_spleen
flux J_art:  Cart -> C[sub=vascular]              = Q_art[organ]*Cart
flux J_ven:  C[sub=vascular] -> Cven              = Q_ven[organ]*C
flux J_lung: Cven -> C[organ=lung, sub=vascular]  = Q_CO*Cven
flux J_out:  C[organ=lung, sub=vascular] -> Cart  = Q_CO*C
d_t(C)               = trans_sub(X, C)
d_t(C[sub=vascular]) = trans_organ(portal, C) + (J_art - J_ven)/V[organ, sub]
d_t(C[organ=lung, sub=vascular])  = (J_lung - J_out)/V[organ, sub]
d_t(C[organ=liver, sub=cellular]) = -CLint/Kp_liver*C/V[organ, sub]
d_t(Cart) = (J_out - sum_organ(J_art))/V_art
d_t(Cven) = (sum_organ(J_ven) - J_lung)/V_ven + R/V_ven
d_t(AUC)  = Cart
dose Cven at t=0: 100
"""


def test_whole_body_pbpk_with_fluxes():
    python = pbpk()
    python.dose(python["Cven"], times=[0], amounts=[100.0])
    sol = same(PBPK, python, 24.0, ["C", "Cart", "Cven", "AUC"], agree=1e-6)
    assert max(sol.ledger.check().values()) < 1e-9


def test_pbpk_fluxes_become_the_same_transfers():
    ir = tq.Model.from_text(PBPK).to_ir()
    got = {
        (t.source, t.source_where, t.to, t.to_where, t.rate.replace(" ", "")) for t in ir.transfers
    }
    python = pbpk().to_ir()
    want = {
        (t.source, t.source_where, t.to, t.to_where, t.rate.replace(" ", ""))
        for t in python.transfers
        if t.to is not None  # elimination is a loss term in the notation
    }
    assert got == want


BIOFILM = """
model biofilm
axis z in [0, 1] n=20 kind=spatial
param De=0.1, kr=0.4, h=0.5, V=2, R=1, CL=0.3
field C(z)
compartment M
d_t(C) = d_z(De*d_z(C)) - kr*C
robin C at z=upper: h*(M/V - C) with M
d_t(M) = R - CL*M/V
"""


def test_biofilm_with_the_bulk_liquid():
    same(BIOFILM, biofilm(20, kind="bulk"), 30.0, ["C", "M"])


def test_every_model_in_the_notation_document_reads():
    blocks = re.findall(r"```[a-z]*\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)
    models = [b for b in blocks if b.lstrip().startswith("model ")]
    assert len(models) >= 6
    for b in models:
        tq.Model.from_text(b)  # raises NotationError with the lines if not


# ----------------------------------------------------------------------------- pieces
def test_slices_doses_observables_and_slice_values():
    m = tq.Model.from_text(
        """
axis organ categorical [gut, "small liver"]
axis sub categorical [v, c]
param V[organ, sub] = {gut: {v: 2, c: 4}, "small liver": {v: 1, c: 0.5}}
field C(organ, sub) weights=V[organ, sub]
d_t(C) = 0
dose C[organ=gut, sub=v] at t=0: 10
dose C[organ="small liver", sub=c] at t=0: 3
observable gut_v = C[organ=gut, sub=v]
liver_c = C[organ="small liver", sub=c]
observable twice = 2*liver_c
observable amount = sum_organ(sum_sub(C))
"""
    )
    sol = m.solve((0, 1), times=[1])
    assert sol["gut_v"][0] == pytest.approx(10 / 2)  # amount / volume
    assert sol["twice"][0] == pytest.approx(2 * 3 / 0.5)  # a slice read in a formula
    assert sol["amount"][0] == pytest.approx(13)  # sums of a weighted field are amounts


def test_field_values_at_time_zero():
    m = tq.Model.from_text(
        """
axis z in [0, 1] n=10
axis s categorical [S, R]
field B(z, s)
d_t(B) = 0
B[s=S](0) = 1e7*point(z, at=0)
B[s=R](0) = 2*exp(-z)
"""
    )
    b = m.solve((0, 1), times=[0])["B"][0]
    assert b[0, 0] == pytest.approx(1e8) and np.all(b[1:, 0] == 0)  # 1e7 / width 0.1
    faces = np.linspace(0, 1, 11)  # a density formula is averaged over each cell
    np.testing.assert_allclose(b[:, 1], 2 * -np.diff(np.exp(-faces)) / 0.1, rtol=1e-9)


def test_extinction_groups():
    m = tq.Model.from_text(
        """
axis s categorical [S, R]
field B(s)
d_t(B) = -B
d_t(P) = -P
B(0) = 10
P(0) = 10
extinction B[s=R] below 1
extinction phage: P, B[s=S] below 1 with noise_floor
"""
    )
    groups = {e.name: e for e in m.to_ir().extinctions}
    assert set(groups) == {"extinct_B_s_R", "phage"}
    assert groups["phage"].noise_floor and groups["phage"].group == ("P", "B")


def test_hazards_and_reactions_get_their_own_ledger_lines():
    m = tq.Model.from_text(
        """
axis s categorical [S, R]
field B(s)
d_t(B) = 2*B - 0.5*B - 0.1*P*B - B*B
d_t(P) = 0
P(0) = 1
B(0) = 1
"""
    )
    kinds = [(t.kind, t.expr.replace(" ", "")) for t in m.to_ir().terms if t.target == "B"]
    assert kinds == [
        ("reaction", "2*B"),
        ("hazard", "0.5"),
        ("hazard", "0.1*P"),
        ("reaction", "-(B*B)"),
    ]


def test_using_sharp_chooses_the_scheme_per_axis():
    text = """
axis a in [0, 1] n=5 kind=one_way
axis x in [0, 1] n=5
field F(a, x)
d_t(F) = -d_a(F) - d_x(0.5*F) {suffix}
at a=1: outflow
"""
    schemes = {}
    for suffix in ("", "using sharp", "using sharp on x"):
        ir = tq.Model.from_text(text.format(suffix=suffix)).to_ir()
        schemes[suffix] = {t.axis: t.scheme for t in ir.terms if t.kind == "advection"}
    assert schemes[""] == {"a": "erlang", "x": "erlang"}
    assert schemes["using sharp"] == {"a": "sharp", "x": "sharp"}
    assert schemes["using sharp on x"] == {"a": "erlang", "x": "sharp"}


def test_chemotaxis_velocity_with_grad():
    m = tq.Model.from_text(
        """
axis x in [0, 1] n=20
param chi=0.1
field B(x)
field S(x)
d_t(B) = -d_x(chi*grad_x(S)*B)
d_t(S) = 0
B(0) = 1
S(0) = x
"""
    )
    (adv,) = [t for t in m.to_ir().terms if t.kind == "advection"]
    assert adv.expr.replace(" ", "") == "chi*grad(S,x)"


# ----------------------------------------------------------------------------- refusals
BASE = """
axis a in [0, 1] n=4 kind=one_way
axis s categorical [S, R]
field I(a)
field J(a)
field B(s)
compartment P
d_t(P) = 0
d_t(B) = 0
"""


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("d_t(I) = -v*d_a(I)\nd_t(J) = 0\nv = 1", "conservation form"),
        ("d_t(I) = d_a(I)\nd_t(J) = 0", "written with a minus"),
        ("d_t(I) = -d_a(1*d_a(I))\nd_t(J) = 0", "negative diffusion has no solution"),
        (
            "d_t(I) = -d_a(J)\nd_t(J) = 0",
            r"transport moves I: write -d_a\(v\*I\) \(this one moves J\)",
        ),
        ("d_t(I) = d_a(d_a(I))\nd_t(J) = 0", r"write diffusion as d_a\(D\*d_a\(I\)\)"),
        ("d_t(I) = 0\nd_t(J) = 0\nd_t(I) = 1", "given twice"),
        ("d_t(I) = 0\nd_t(J) = 0\nd_t(Q) = -d_a(Q)", "compartment has no axis 'a'"),
        ("d_t(I) = 0\nd_t(J) = 0\nfield K(a) = 0", r"K\(0\) = 0"),
        ("d_t(I) = 0\nd_t(J) = 0\nh(a) = 2*a\nx = h(2)", "named formula, not a function"),
        (
            "d_t(I) = 0\nd_t(J) = 0\ntransitions M on s: S -> R = 1\n"
            "field G(s)\nd_t(G) = trans_a(M, G)",
            "moves along s",
        ),
        ("d_t(I) = 0\nd_t(J) = 0\nx = B[s=Q]", "no category 'Q'"),
        ("d_t(I) = 0\nd_t(J) = 0\ntransfer P -> B: 1", "flux"),
        ("d_t(I) = -d_a(I) using sharp on s\nd_t(J) = 0", "using sharp on s"),
        ("d_t(I) = 0\nd_t(J) = 0\nI[s=S](0) = 1", "no axis 's'"),
        (
            "d_t(I) = 0\nd_t(J) = 0\nI(0) = 1e7*normal(a, mean=0.5, sd=0.1) + 1",
            "amount times one shape",
        ),
    ],
)
def test_refusals(text, message):
    with pytest.raises(tq.NotationError, match=message):
        tq.Model.from_text(BASE.replace("d_t(B) = 0\n", "d_t(B) = 0\n", 1) + text)


FLUX_BASE = """
axis organ categorical [lung, liver]
param Q=1, V_art=2
param V[organ] = {lung: 1, liver: 3}
field C(organ) weights=V[organ]
compartment Cart volume=V_art
"""


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (  # wrong volume on the arriving side
            "flux J: C -> Cart = Q*C\nd_t(C) = -J/V[organ]\nd_t(Cart) = sum_organ(J)/V",
            r"divide it by Cart's volume, V_art \(not V\)",
        ),
        (  # forgot the sum over organs
            "flux J: C -> Cart = Q*C\nd_t(C) = -J/V[organ]\nd_t(Cart) = J/V_art",
            r"takes sum_organ\(J\) here",
        ),
        (  # arrives nowhere
            "flux J: C -> Cart = Q*C\nd_t(C) = -J/V[organ]\nd_t(Cart) = 0",
            r"arrives in Cart: d_t\(Cart\) needs exactly one term \+J",
        ),
        (  # sign
            "flux J: C -> Cart = Q*C\nd_t(C) = J/V[organ]\nd_t(Cart) = sum_organ(J)/V_art",
            "leaves C",
        ),
        (  # not a rate times the source
            "flux J: C -> Cart = Q\nd_t(C) = -J/V[organ]\nd_t(Cart) = sum_organ(J)/V_art",
            r"rate times what it takes from",
        ),
        (  # not ±J
            "flux J: C -> Cart = Q*C\nd_t(C) = -2*J/V[organ]\nd_t(Cart) = sum_organ(J)/V_art",
            r"appears as -J where it leaves",
        ),
    ],
)
def test_flux_mistakes_are_caught(text, message):
    with pytest.raises(tq.NotationError, match=message):
        tq.Model.from_text(FLUX_BASE + text)


def test_a_balanced_flux_conserves_the_amount():
    m = tq.Model.from_text(
        FLUX_BASE
        + "flux J: C -> Cart = Q*C\nd_t(C) = -J/V[organ]\nd_t(Cart) = sum_organ(J)/V_art\n"
        + "C(0) = 1\n"
    )
    sol = m.solve((0, 5), times=np.linspace(0, 5, 6), rtol=1e-10, atol=1e-12)
    amount = sol["C"] @ np.array([1.0, 3.0]) + 2.0 * sol["Cart"]
    np.testing.assert_allclose(amount, 4.0, rtol=1e-9)
    # each organ empties at Q/V: C_lung = e^-t, C_liver = e^(-t/3)
    np.testing.assert_allclose(sol["C"][:, 0], np.exp(-sol.t), rtol=1e-7)
    np.testing.assert_allclose(sol["C"][:, 1], np.exp(-sol.t / 3), rtol=1e-7)
