# %% [markdown]
# # Tutorial 6: a whole-body PBPK model as one field over organs and spaces
#
# A physiologically based PK (PBPK) model follows a drug through the organs, each fed by its
# own blood flow. In a **permeability-limited** model each organ is further split into
# spaces: the blood in the organ (**vascular**), the fluid between the cells
# (**interstitial**) and the cells themselves (**cellular**), with membranes in between.
# With 9 organs and 3 spaces that is 27 amounts, plus an arterial and a venous blood pool:
# 29 ODEs, usually written out one by one (Jones & Rowland-Yeo 2013, CPT Pharmacometrics
# Syst Pharmacol).
#
# Most of those ODEs say the same thing for a different organ. TenQSP writes the drug as
# **one field `C(organ, sub)`** over two categorical axes: `organ` (lung, brain, ...) and
# `sub` (the three spaces). An exchange between spaces is written once, with rates that
# may depend on the organ; blood flow is written once, acting only on the vascular space;
# the two blood pools are compartments ("hubs") that send blood to every organ and collect
# it back.
#
# **Assumptions** of this version:
#
# 1. a small molecule; blood:plasma ratio 1, so blood and plasma concentrations are equal;
# 2. exchange across membranes by passive permeability: vascular <-> interstitial at PS1,
#    interstitial <-> cellular at PS2 (both L/h, the same in every organ); the cells hold
#    the drug at Kp times the interstitial concentration at equilibrium (Kp per organ, as
#    predicted by Rodgers & Rowland 2006, J Pharm Sci);
# 3. elimination only by metabolism in liver cells, with intrinsic clearance CLint acting
#    on the unbound concentration: the cells' concentration divided by Kp is the plasma
#    concentration in equilibrium with them, and the fraction `fu` of it is unbound. (For
#    brevity, binding is not written into the membrane exchange.)
# 4. the gut and the spleen drain into the liver through the portal vein.
#
# **Values** are illustrative, of adult-human magnitude (organ volumes and flows in the
# range of Brown et al. 1997, Toxicol Ind Health; cardiac output 300 L/h = 5 L/min); they
# are not a particular drug.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import tenqsp as tq

# %% [markdown]
# ## 1. The model
#
# Read it from the top:
#
# - the two axes, then the values. `param Vorg[organ] = {...}` is one value per organ;
#   the volume of each space, `V(organ, sub)`, is the organ's volume times the fraction of
#   it in that space. `weights=V` says that `C` is a **concentration**: amounts are C x V.
#   (The volumes play the role that cell widths play on a continuous axis: sums over the
#   axes, the ledger and doses all work in amounts.)
# - `transitions X on sub:` is the exchange between spaces. Each line is a move with its
#   clearance (L/h): the amount moving per hour is the clearance times the concentration
#   where the drug comes from. Moving vascular -> interstitial at PS1 and back at PS1 gives
#   the familiar PS1 (c_vascular - c_interstitial); the cells release drug at PS2/Kp, so at
#   equilibrium they hold Kp times the interstitial concentration. `PS2/Kp[organ]` means a
#   different rate in each organ: a 3 x 3 exchange per organ, not one shared matrix.
# - `transitions portal on organ:` moves blood from the gut and spleen to the liver, in the
#   vascular space only (it is used in the equation of `C[sub=vascular]`).
# - four **fluxes** move blood between the hubs and the organs: the arterial pool feeds every
#   organ's vascular space at its arterial flow `Q_art[organ]` (the lung gets none: it is fed
#   by the venous pool, through the heart); the organs drain into the venous pool at
#   `Q_ven[organ]` (the gut and spleen drain into the liver instead, so the liver's venous
#   flow is 20 + 65 + 10 = 95 L/h); the venous pool goes through the lung, and the lung
#   into the arterial pool, at the cardiac output.
# - the equations. `d_t(C)` holds for every organ and space; a slice equation such as
#   `d_t(C[sub=vascular])` adds terms where the axes are fixed. Elimination is a loss term
#   in the liver's cells.
# - `AUC` is the area under the arterial concentration, computed by the solver (section 3
#   says why).

# %%
PBPK = """
model pbpk
axis organ categorical [lung, brain, heart, muscle, adipose, kidney, gut, spleen, liver]
axis sub categorical [vascular, interstitial, cellular]

param Vorg[organ]  = {lung: 1.2, brain: 1.4, heart: 0.33, muscle: 29, adipose: 15,
                      kidney: 0.31, gut: 1.2, spleen: 0.19, liver: 1.8}          # L
param frac[sub]    = {vascular: 0.05, interstitial: 0.15, cellular: 0.8}
param Kp[organ]    = {lung: 1, brain: 0.5, heart: 2, muscle: 1.5, adipose: 5, kidney: 2,
                      gut: 2, spleen: 1.5, liver: 3}
param Q_art[organ] = {lung: 0, brain: 45, heart: 15, muscle: 60, adipose: 15, kidney: 70,
                      gut: 65, spleen: 10, liver: 20}                            # L/h
param Q_ven[organ] = {lung: 0, brain: 45, heart: 15, muscle: 60, adipose: 15, kidney: 70,
                      gut: 0, spleen: 0, liver: 95}
param Q_gut=65, Q_spleen=10, Q_CO=300, V_art=1.7, V_ven=3.4
param PS1=50, PS2=10, CLint=30, fu=0.5

V(organ, sub) = Vorg[organ]*frac[sub]
field C(organ, sub) weights=V                    # concentrations (mg/L); amount = C x V
compartment Cart volume=V_art                    # arterial blood
compartment Cven volume=V_ven                    # venous blood

transitions X on sub:                            # exchange across the membranes
    vascular -> interstitial = PS1
    interstitial -> vascular = PS1
    interstitial -> cellular = PS2
    cellular -> interstitial = PS2/Kp[organ]
transitions portal on organ: gut -> liver = Q_gut, spleen -> liver = Q_spleen

flux J_art:  Cart -> C[sub=vascular]              = Q_art[organ]*Cart   # mg/h
flux J_ven:  C[sub=vascular] -> Cven              = Q_ven[organ]*C
flux J_lung: Cven -> C[organ=lung, sub=vascular]  = Q_CO*Cven
flux J_out:  C[organ=lung, sub=vascular] -> Cart  = Q_CO*C

d_t(C)                            = trans_sub(X, C)
d_t(C[sub=vascular])              = trans_organ(portal, C) + (J_art - J_ven)/V
d_t(C[organ=lung, sub=vascular])  = (J_lung - J_out)/V
d_t(C[organ=liver, sub=cellular]) = -fu*CLint/Kp[organ]*C/V        # metabolism

d_t(Cart) = (J_out - sum_organ(J_art))/V_art
d_t(Cven) = (sum_organ(J_ven) - J_lung)/V_ven
d_t(AUC)  = Cart

observable in_tissues = sum_organ(sum_sub(C))    # an amount (mg): sums use the weights
"""
pbpk = tq.Model.from_text(PBPK)
print(pbpk.summary())

ORGANS = ["lung", "brain", "heart", "muscle", "adipose", "kidney", "gut", "spleen", "liver"]
VORG = np.array([1.2, 1.4, 0.33, 29, 15, 0.31, 1.2, 0.19, 1.8])
FRAC = np.array([0.05, 0.15, 0.8])
KP = np.array([1, 0.5, 2, 1.5, 5, 2, 2, 1.5, 3.0])
Q_H = 95.0  # blood flow through the liver, L/h
TIGHT = {"rtol": 1e-10, "atol": 1e-12}

# %% [markdown]
# ## 2. The same model as 29 ODEs
#
# For comparison, here is the model written by hand in amounts, as one would for
# `solve_ivp` (or in NONMEM's `$DES`). Even vectorised with numpy, the bookkeeping that
# the field did for us is now ours: which index is which organ, which flows go where, the
# portal vein, the lung between the pools, and the volumes in every concentration. Written
# out one equation per amount, it is 29 lines that differ only in their indices, and adding
# an organ means touching several of them.


# %%
def pbpk_by_hand(PS1=50.0, PS2=10.0, CLint=30.0, fu=0.5):
    Vt = np.outer(VORG, FRAC)  # (organ, sub) volumes
    q_art = np.array([0, 45, 15, 60, 15, 70, 65, 10, 20.0])
    q_ven = np.array([0, 45, 15, 60, 15, 70, 0, 0, 95.0])
    lung, gut, spleen, liver = 0, 6, 7, 8

    def rhs(_, y):
        A = y[:27].reshape(9, 3)  # amounts, organ x space
        c = A / Vt
        c_art, c_ven = y[27] / 1.7, y[28] / 3.4
        dA = np.zeros((9, 3))
        to_isf = PS1 * (c[:, 0] - c[:, 1])  # vascular -> interstitial
        to_cells = PS2 * (c[:, 1] - c[:, 2] / KP)  # interstitial -> cellular
        dA[:, 0] += q_art * c_art - q_ven * c[:, 0] - to_isf
        dA[:, 1] += to_isf - to_cells
        dA[:, 2] += to_cells
        for organ, q in ((gut, 65.0), (spleen, 10.0)):  # the portal vein
            dA[organ, 0] -= q * c[organ, 0]
            dA[liver, 0] += q * c[organ, 0]
        dA[lung, 0] += 300 * c_ven - 300 * c[lung, 0]
        dA[liver, 2] -= fu * CLint * c[liver, 2] / KP[liver]
        d_art = 300 * c[lung, 0] - q_art.sum() * c_art
        d_ven = (q_ven * c[:, 0]).sum() - 300 * c_ven
        return np.concatenate([dA.ravel(), [d_art, d_ven]])

    return rhs, Vt


t = np.array([0.1, 1, 6, 24, 96.0])
iv = tq.bolus("Cven", 100, times=[0])  # 100 mg IV, into the venous blood
sol = pbpk.solve((0, 96), times=t, regimen=iv, **TIGHT)
rhs, Vt = pbpk_by_hand()
y0 = np.zeros(29)
y0[28] = 100.0
ref = solve_ivp(rhs, (0, 96), y0, t_eval=t, method="Radau", rtol=1e-12, atol=1e-14).y
by_hand = ref[:27].T.reshape(-1, 9, 3) / Vt
diff = np.max(np.abs(np.asarray(sol["C"]) - by_hand) / (np.abs(by_hand) + 1e-6))
print(f"field vs 29 hand-written ODEs: largest relative difference {diff:.1e}")
assert diff < 1e-6
assert np.max(np.abs(np.asarray(sol["Cart"]) / (ref[27] / 1.7) - 1)) < 1e-6

# %% [markdown]
# ## 3. An IV dose: mass balance and AUC = Dose / CL
#
# **Mass balance.** At any time, the drug in the tissues, in the two blood pools and
# eliminated must add up to the dose. The ledger records each term of each equation as an
# amount (here in mg, since `C` has volumes as weights), so the eliminated amount is read
# from it, and `sol.ledger.check()` reports what no term explains (rounding only).
#
# **Clearance.** With elimination only in the liver's cells, the drug reaching the enzymes
# has to cross two membranes in series. As for resistances in series, the conductances add
# as reciprocals: G = 1 / (1/PS1 + 1/PS2 + 1/(fu CLint)), and the liver then clears blood as
# the well-stirred model does with G in place of fu CLint:
#
#     CL_H = Q_H G / (Q_H + G).
#
# As the membranes become very permeable (PS -> infinity), G -> fu CLint and this becomes the
# well-stirred liver model, CL_H = Q_H fu CLint / (Q_H + fu CLint) (Pang & Rowland 1977,
# J Pharmacokinet Biopharm). The lung does not eliminate, so the arterial AUC equals the
# mixed-venous one, and after an IV dose it is Dose / CL_H.


# %%
def hepatic_clearance(PS1=50.0, PS2=10.0, CLint=30.0, fu=0.5):
    G = 1.0 / (1.0 / PS1 + 1.0 / PS2 + 1.0 / (fu * CLint))
    return Q_H * G / (Q_H + G)


CL_H = hepatic_clearance()
times = np.concatenate([np.linspace(0, 2, 81)[1:], np.linspace(2, 600, 300)[1:]])
sol = pbpk.solve((0, 600), times=times, regimen=iv, **TIGHT)

eliminated = -sol.ledger.moved["C"]["hazard: fu * CLint / Kp[organ] / V"]
in_blood = 1.7 * np.asarray(sol["Cart"]) + 3.4 * np.asarray(sol["Cven"])
total = np.asarray(sol["in_tissues"]) + in_blood + eliminated
print(f"tissues + blood + eliminated - dose: at most {np.max(np.abs(total - 100)):.1e} mg")
print(f"ledger: largest unexplained part {max(sol.ledger.check().values()):.1e}")
assert np.max(np.abs(total - 100)) < 1e-6
assert max(sol.ledger.check().values()) < 1e-10

auc = float(sol["AUC"][-1])
print(f"AUC {auc:.6f} mg.h/L; Dose/CL_H {100 / CL_H:.6f} (CL_H = {CL_H:.4f} L/h)")
assert abs(auc / (100 / CL_H) - 1) < 1e-8

# %% [markdown]
# The two agree to 3e-9: by 600 h the arterial concentration has fallen so far that what
# is left of the AUC is below that.
#
# **Why AUC is a solver state.** The same AUC computed afterwards from the saved output
# with the trapezoidal rule depends on the output times, and with these (fine at the start,
# 2 h apart later) it is off by almost one part in a thousand. The equation
# `d_t(AUC) = Cart` lets the solver integrate it with its own error control, at no extra
# cost. The same holds for any exposure metric (Cmax aside): make it a state.

# %%
trapezoid = np.trapezoid(np.concatenate([[0.0], sol["Cart"]]), np.concatenate([[0.0], times]))
print(f"trapezoid on the saved output: {trapezoid:.4f}, off by {trapezoid / auc - 1:.1e}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(times, sol["Cart"], "k", lw=2, zorder=3, label="arterial blood")
for name in ("brain", "muscle", "adipose", "liver"):
    o = ORGANS.index(name)
    amount = np.asarray(sol["C"])[:, o, :] @ (VORG[o] * FRAC)  # mg in the organ
    ax.semilogy(times, amount / VORG[o], label=name)
ax.set_xlim(0, 96)
ax.set_ylim(1e-3, 30)
ax.set_xlabel("time (h)")
ax.set_ylabel("concentration (mg/L of tissue)")
ax.set_title("100 mg IV: blood and whole-organ concentrations")
ax.legend(fontsize="small")
fig.tight_layout()

# %% [markdown]
# The brain (Kp 0.5, high flow) follows the blood quickly; adipose (Kp 5, a large volume and
# a modest flow) fills slowly and then releases the drug, which sets the long terminal phase.
#
# ## 4. An infusion: steady state and tissue:plasma ratios
#
# Under a constant infusion R into the venous blood, the steady-state arterial
# concentration is R / CL_H. In each organ that does not eliminate, nothing is lost at
# steady state, so the concentrations come to equilibrium: vascular = interstitial =
# arterial, and the cells at Kp times that. The whole organ's concentration over plasma is
# then
#
#     Kp_tissue = f_vascular + f_interstitial + f_cellular Kp,
#
# which is what a PBPK user calls the tissue:plasma partition coefficient. In the liver
# the enzymes keep a gradient towards the cells, so its ratio stays below this value.

# %%
infusion = tq.infusion("Cven", rate=10, duration=1000, start=0)  # 10 mg/h
long = np.linspace(0, 800, 401)[1:]
ss = pbpk.solve((0, 800), times=long, regimen=infusion, **TIGHT)
C_end = np.asarray(ss["C"])[-1]  # (organ, sub) at 800 h
c_art = float(ss["Cart"][-1])
print(f"arterial at 800 h {c_art:.8f}; R/CL_H {10 / CL_H:.8f}")
assert abs(c_art / (10 / CL_H) - 1) < 1e-8

kp_tissue = FRAC[0] + FRAC[1] + FRAC[2] * KP
ratio = (np.asarray(ss["C"]) @ FRAC) / np.asarray(ss["Cart"])[:, None]  # organ / plasma
print("organ    cells/vascular   Kp   organ/plasma   Kp_tissue")
for o, name in enumerate(ORGANS):
    cells = C_end[o, 2] / C_end[o, 0]
    print(f"{name:8s} {cells:12.6f} {KP[o]:6.2f} {ratio[-1, o]:12.6f} {kp_tissue[o]:9.4f}")
    if name != "liver":
        assert abs(cells / KP[o] - 1) < 1e-7, name
        assert abs(C_end[o, 1] / C_end[o, 0] - 1) < 1e-7, name
        assert abs(ratio[-1, o] / kp_tissue[o] - 1) < 1e-7, name
liver = ORGANS.index("liver")
assert C_end[liver, 0] > C_end[liver, 1] > C_end[liver, 2] / KP[liver]

fig, ax = plt.subplots(figsize=(7, 4.5))
for o, name in enumerate(ORGANS):
    (line,) = ax.plot(long, ratio[:, o], label=name)
    if name != "liver":
        ax.axhline(kp_tissue[o], color=line.get_color(), ls=":", lw=1)
ax.set_xscale("log")
ax.set_xlabel("time since the start of the infusion (h)")
ax.set_ylabel("organ concentration / arterial concentration")
ax.set_title("10 mg/h infusion: each ratio approaches its Kp_tissue (dotted)")
ax.legend(fontsize="small", ncol=2)
fig.tight_layout()

# %% [markdown]
# ## 5. From permeability-limited to the well-stirred liver
#
# Raising both permeabilities removes the membranes' resistance: the hepatic clearance
# moves from the series value to the well-stirred one, and the simulation follows the
# formula at every step. Values are given when solving, so the model is not rebuilt.
# (Very large PS makes the model very stiff: at PS = 1e4 the heart's blood exchanges with
# its interstitium in milliseconds, and the tight tolerances used here then make long runs.
# The last two lines show the formula only.)

# %%
well_stirred = Q_H * 0.5 * 30 / (Q_H + 0.5 * 30)
print(f"   PS     simulated R/Css   series formula   (well-stirred: {well_stirred:.4f})")
for ps in (10.0, 100.0, 1000.0):
    s = pbpk.solve((0, 800), times=[800], regimen=infusion, params={"PS1": ps, "PS2": ps}, **TIGHT)
    simulated = 10 / float(s["Cart"][-1])
    formula = hepatic_clearance(PS1=ps, PS2=ps)
    print(f"{ps:7.0f} {simulated:16.6f} {formula:16.6f}")
    assert abs(simulated / formula - 1) < 1e-7
for ps in (1e4, 1e5):  # the formula alone: the difference falls tenfold per decade
    print(f"{ps:7.0f} {'':16s} {hepatic_clearance(PS1=ps, PS2=ps):16.6f}")
assert abs(hepatic_clearance(1e5, 1e5) / well_stirred - 1) < 5e-4

# %% [markdown]
# ## What this buys you
#
# - **One equation per kind of process**, not per organ: 29 ODEs become four lines for the
#   tissues and two for the blood pools. Adding an organ is one entry in each table;
#   adding a space (an endosomal space for an antibody, as in Shah & Betts 2012,
#   J Pharmacokinet Pharmacodyn) is one label and its exchange rates.
# - **Fluxes are checked**: each flow leaves one place and arrives in another, so a
#   misplaced volume is an error when the model is read, not a mass leak found later.
# - **Exposure as states and a ledger in amounts**: AUC to solver accuracy, and a mass
#   balance that closes to rounding.
#
# **Limits of this version.** Every organ has the same three spaces (blood has no
# interstitium, and a lung with an epithelial lining fluid or a tumour with depth would be
# separate fields coupled to the slice of their organ); no renal clearance, plasma binding
# in the membrane exchange or blood:plasma ratio; the values are illustrative.
