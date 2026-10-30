# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""A whole-body PBPK model (permeability-limited small molecule, fu = 1,
B:P = 1) in TenQSP form and written out by hand, for the V32-V35 tests and the step 2.6
hand-test. Values are illustrative (adult-human magnitudes), not a published drug."""

from __future__ import annotations

import numpy as np

import tenqsp as tq

ORGANS = ["lung", "brain", "heart", "muscle", "adipose", "kidney", "gut", "spleen", "liver"]
SUBS = ["vascular", "interstitial", "cellular"]
VOLUME = dict(  # L
    lung=1.2,
    brain=1.4,
    heart=0.33,
    muscle=29.0,
    adipose=15.0,
    kidney=0.31,
    gut=1.2,
    spleen=0.19,
    liver=1.8,
)
FRACTION = dict(vascular=0.05, interstitial=0.15, cellular=0.80)
FLOW = dict(
    brain=45.0, heart=15.0, muscle=60.0, adipose=15.0, kidney=70.0, gut=65.0, spleen=10.0
)  # L/h; hepatic artery below
Q_HA = 20.0
Q_CO = sum(FLOW.values()) + Q_HA  # 300 L/h through the lung
Q_LIVER = Q_HA + FLOW["gut"] + FLOW["spleen"]
KP = dict(
    lung=1.0,
    brain=0.5,
    heart=2.0,
    muscle=1.5,
    adipose=5.0,
    kidney=2.0,
    gut=2.0,
    spleen=1.5,
    liver=3.0,
)
V_ART, V_VEN = 1.7, 3.4
PORTAL = ("gut", "spleen")


def arterial(o):  # flow from the arterial pool into each organ
    return 0.0 if o == "lung" else Q_HA if o == "liver" else FLOW[o]


def venous(o):  # flow from each organ into the venous pool
    return 0.0 if o in ("lung", *PORTAL) else Q_LIVER if o == "liver" else FLOW[o]


def volumes():
    return {o: {s: VOLUME[o] * FRACTION[s] for s in SUBS} for o in ORGANS}


def series_clearance(PS1, PS2, CLint):
    """Hepatic clearance with elimination in liver cells only (series resistance)."""
    G = 1.0 / (1.0 / PS1 + 1.0 / PS2 + 1.0 / CLint)
    return Q_LIVER * G / (Q_LIVER + G)


def pbpk(PS1=50.0, PS2=10.0, CLint=30.0, R=0.0):
    """C[organ, sub] (concentrations, weights = volumes), arterial and venous pools, an
    infusion R (amount/h) into the venous pool, and the arterial AUC as a solver state."""
    m = tq.Model("pbpk")
    organ = m.axis("organ", labels=ORGANS)
    sub = m.axis("sub", labels=SUBS)
    V = m.param("V", volumes(), over=[organ, sub])
    m.params(
        PS1=PS1,
        PS2=PS2,
        CLint=CLint,
        Kp_liver=KP["liver"],
        Q_CO=Q_CO,
        R=R,
        V_art=V_ART,
        V_ven=V_VEN,
    )
    m.param("Kp", KP, over=organ)
    Qa = m.param("Q_art", {o: arterial(o) for o in ORGANS}, over=organ)
    Qv = m.param("Q_ven", {o: venous(o) for o in ORGANS}, over=organ)
    m.params(**{f"Q_{p}": FLOW[p] for p in PORTAL})
    C = m.field("C", axes=[organ, sub], weights=V)
    Cart = m.compartment("Cart", volume="V_art")
    Cven = m.compartment("Cven", volume="V_ven")
    AUC = m.compartment("AUC")
    X = m.transitions(
        "X",
        axis=sub,
        rates={
            "vascular->interstitial": "PS1",
            "interstitial->vascular": "PS1",
            "interstitial->cellular": "PS2",
            "cellular->interstitial": "PS2/Kp[organ]",
        },
    )
    m.transition(C, axis=sub, matrix=X, of=C)
    vasc = C.at(sub="vascular")
    G = m.transitions("portal", axis=organ, rates={f"{p}->liver": f"Q_{p}" for p in PORTAL})
    m.transition(vasc, axis=organ, matrix=G, of=vasc)
    m.transfer(Cart, to=vasc, rate=Qa[organ])
    m.transfer(vasc, to=Cven, rate=Qv[organ])
    lung = C.at(organ="lung", sub="vascular")
    m.transfer(Cven, to=lung, rate="Q_CO")
    m.transfer(lung, to=Cart, rate="Q_CO")
    m.transfer(C.at(organ="liver", sub="cellular"), to=None, rate="CLint/Kp_liver")
    m.ode(Cven, "R/V_ven")
    m.ode(AUC, Cart)
    m.init(C, 0.0)
    m.init(Cart, 0.0)
    m.init(Cven, 0.0)
    m.init(AUC, 0.0)
    return m


def pbpk_by_hand(PS1=50.0, PS2=10.0, CLint=30.0, R=0.0):
    """The same model as 27 + 3 hand-written ODEs, in amounts. State: A[o, s] (row-major),
    then A_art, A_ven, AUC. Returns (rhs, volumes of the 29 amounts)."""
    n = len(ORGANS)
    Vt = np.array([[VOLUME[o] * FRACTION[s] for s in SUBS] for o in ORGANS])
    kp = np.array([KP[o] for o in ORGANS])
    qa = np.array([arterial(o) for o in ORGANS])
    qv = np.array([venous(o) for o in ORGANS])
    li, lu = ORGANS.index("liver"), ORGANS.index("lung")

    def rhs(_, y):
        A = y[: 3 * n].reshape(n, 3)
        c = A / Vt
        ca, cv = y[3 * n] / V_ART, y[3 * n + 1] / V_VEN
        d = np.zeros_like(A)
        x1 = PS1 * (c[:, 0] - c[:, 1])
        x2 = PS2 * (c[:, 1] - c[:, 2] / kp)
        d[:, 0] += -x1 + qa * ca - qv * c[:, 0]
        d[:, 1] += x1 - x2
        d[:, 2] += x2
        for p in PORTAL:
            k = ORGANS.index(p)
            d[k, 0] -= FLOW[p] * c[k, 0]
            d[li, 0] += FLOW[p] * c[k, 0]
        d[lu, 0] += Q_CO * cv - Q_CO * c[lu, 0]
        d[li, 2] -= CLint * c[li, 2] / kp[li]
        dart = Q_CO * c[lu, 0] - qa.sum() * ca
        dven = (qv * c[:, 0]).sum() - Q_CO * cv + R
        return np.concatenate([d.ravel(), [dart, dven, ca]])

    return rhs, np.concatenate([Vt.ravel(), [V_ART, V_VEN, 1.0]])


def perfusion_limited(CLint=30.0):
    """PS -> infinity: each organ one well-stirred tissue, c_v = c_i = c_c/Kp. State: the
    organs' amounts, then A_art, A_ven. Returns (rhs, the vascular concentration per unit
    amount of each organ)."""
    n = len(ORGANS)
    Vt = np.array(
        [
            VOLUME[o]
            * (FRACTION["vascular"] + FRACTION["interstitial"] + FRACTION["cellular"] * KP[o])
            for o in ORGANS
        ]
    )
    qa = np.array([arterial(o) for o in ORGANS])
    qv = np.array([venous(o) for o in ORGANS])
    li, lu = ORGANS.index("liver"), ORGANS.index("lung")

    def rhs(_, y):
        cv_o = y[:n] / Vt
        ca, cv = y[n] / V_ART, y[n + 1] / V_VEN
        d = qa * ca - qv * cv_o
        for p in PORTAL:
            k = ORGANS.index(p)
            d[k] -= FLOW[p] * cv_o[k]
            d[li] += FLOW[p] * cv_o[k]
        d[lu] += Q_CO * cv - Q_CO * cv_o[lu]
        d[li] -= CLint * cv_o[li]
        return np.concatenate([d, [Q_CO * cv_o[lu] - qa.sum() * ca, (qv * cv_o).sum() - Q_CO * cv]])

    return rhs
