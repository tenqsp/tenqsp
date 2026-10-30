# %% [markdown]
# # Tutorial 10: how far does an antibiotic get into a biofilm?
#
# A biofilm is a layer of bacteria, tens to hundreds of micrometres thick, stuck to a
# surface (a catheter, a heart valve, the lung's mucus) and bathed in a liquid that carries
# the drug. The drug reaches the deeper cells only by diffusing through the layer, and on the
# way it may be destroyed (by a beta-lactamase) or bound. Stewart (1996, Antimicrob Agents
# Chemother) worked out how deep the drug gets when diffusion and reaction compete. The
# answer is one dimensionless number, the **Thiele modulus**.
#
# The biofilm is a field over **depth** z, from the surface it grows on (z = 0) to the
# liquid (z = L). The liquid can be held at a fixed concentration (sections 1 to 3) or be a
# well-mixed compartment, as in a PK model, that exchanges drug with the biofilm through
# its surface (section 4).
#
# 1. the model: diffusion and reaction in the depth;
# 2. the steady profile, the Thiele modulus and the effectiveness factor;
# 3. how long the drug takes to get in when nothing destroys it;
# 4. the biofilm as a sink for the liquid: coupling to a bulk compartment, and the books.
#
# **Units and values.** Depth in micrometres (um), time in minutes, concentrations in mg/L.
# The values are illustrative, of the order discussed by Stewart: a 200 um biofilm, an
# effective diffusivity De = 1e-6 cm^2/s (6000 um^2/min, a few times below that in water),
# and a reaction rate chosen to give a Thiele modulus of 2.

# %%
import matplotlib.pyplot as plt
import numpy as np

import tenqsp as tq

TIGHT = {"rtol": 1e-10, "atol": 1e-12}
L, DE = 200.0, 6000.0

# %% [markdown]
# ## 1. Diffusion and reaction in the depth
#
# `d_z(De*d_z(C))` is diffusion along z (drug moving from where it is concentrated to where
# it is not, at a speed set by De); `-kr*C` is first-order destruction. The two ends of the
# depth axis need a condition each, written under the equation as in a paper:
#
# - at z = 0 the biofilm sits on a solid surface: nothing crosses (`at z=0: d_z(C) = 0`;
#   an end that is not mentioned is sealed by default, so this line is written here only
#   to be explicit);
# - at z = L the biofilm touches the liquid, held at `Cb` (`at z=200: C = Cb`: the value at
#   the end is fixed).
#
# The axis is cut into 40 **cells** (slices of the depth, 5 um thick; not bacteria). The
# biofilm starts free of drug.

# %%
SLAB = """
model biofilm
axis z in [0, 200] n=40 units=um kind=spatial    # depth: 0 = substratum, 200 = liquid
param De=6000, kr=0.6, Cb=1
field C(z)
d_t(C) = d_z(De*d_z(C)) - kr*C
    at z=0: d_z(C) = 0                            # the substratum: nothing crosses
    at z=200: C = Cb                              # the liquid, held at Cb
"""
slab = tq.Model.from_text(SLAB)
print(slab.explain().split("Starting values")[0])

# %% [markdown]
# ## 2. The steady profile and the Thiele modulus
#
# **The Thiele modulus in plain terms.** Two clocks compete. The drug needs about L^2/De to
# diffuse across the biofilm (here 200^2/6000 = 6.7 min); it survives about 1/kr before it
# is destroyed (1.7 min). The Thiele modulus is phi = L sqrt(kr/De), so phi^2 is the ratio
# of the two times:
#
# - phi well below 1: the drug crosses the biofilm long before it is destroyed, and the
#   whole depth sees nearly Cb;
# - phi well above 1: the drug is destroyed in an outer skin about sqrt(De/kr) thick (L/phi),
#   and the deep cells see almost none of it.
#
# At steady state the profile is C(z)/Cb = cosh(phi z/L) / cosh(phi), and the mean
# concentration in the biofilm, as a fraction of Cb, is the **effectiveness factor**
# eta = tanh(phi)/phi (about 1/phi when phi is large). With kr = 0.6 per min, phi = 2: on
# average the biofilm sees 48 % of the liquid concentration, and its bottom only 27 %.
#
# The model's cells hold averages over each cell, so they are compared with the average of
# the cosh profile over each cell, (sinh at the upper face - sinh at the lower face) /
# (phi/L x cell width) / cosh(phi).

# %%
z_faces = np.linspace(0, L, 41)
z_mid = 0.5 * (z_faces[1:] + z_faces[:-1])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
etas = {}
for phi in (0.5, 2.0, 5.0):
    kr = phi**2 * DE / L**2
    profile = np.asarray(slab.solve((0, 60), times=[60], params={"kr": kr}, **TIGHT)["C"])[-1]
    exact = np.diff(np.sinh(phi * z_faces / L)) / (phi / L) / np.diff(z_faces) / np.cosh(phi)
    err = np.max(np.abs(profile - exact))
    etas[phi] = float(np.mean(profile))
    print(
        f"phi = {phi}: largest difference from the cosh profile {err:.1e}; "
        f"eta = {etas[phi]:.4f} (tanh(phi)/phi = {np.tanh(phi) / phi:.4f})"
    )
    assert err < 3e-3
    assert abs(etas[phi] / (np.tanh(phi) / phi) - 1) < 3e-3
    (dots,) = ax[0].plot(z_mid, profile, "o", ms=3)
    fine = np.linspace(0, L, 200)
    curve = np.cosh(phi * fine / L) / np.cosh(phi)
    ax[0].plot(fine, curve, color=dots.get_color(), label=f"phi = {phi}")
ax[0].set_xlabel("depth z (um; 0 = substratum, 200 = liquid)")
ax[0].set_ylabel("C / Cb")
ax[0].set_title("steady profiles: model (dots) and cosh (lines)")
ax[0].legend(fontsize="small")
phis = np.linspace(0.05, 8, 200)
ax[1].plot(phis, np.tanh(phis) / phis, "k", label="tanh(phi)/phi")
ax[1].plot(list(etas), list(etas.values()), "o", label="model")
ax[1].set_xlabel("Thiele modulus phi")
ax[1].set_ylabel("effectiveness factor (mean C / Cb)")
ax[1].legend(fontsize="small")
fig.tight_layout()

# %% [markdown]
# The differences (at most 0.3 % of Cb, for the steepest profile) are the grid: the scheme
# for diffusion is second order, so halving the cell width divides them by about four.
#
# ## 3. Without reaction: how long the drug takes to get in
#
# If nothing destroys the drug (kr = 0), the biofilm eventually reaches Cb everywhere:
# diffusion alone does not stop an antibiotic, it delays it. The delay is set by L^2/De.
# For a slab sealed on one face, Crank (The Mathematics of Diffusion, 2nd ed. 1975,
# chapter 4) gives the mean concentration as a series:
# mean/Cb = 1 - sum over n >= 0 of 8/((2n+1)^2 pi^2) exp(-(2n+1)^2 pi^2 De t / (4 L^2)).

# %%
t_in = np.linspace(0, 20, 41)
mean_in = np.asarray(slab.solve((0, 20), times=t_in, params={"kr": 0.0}, **TIGHT)["C"]).mean(axis=1)
odd = 2 * np.arange(200)[:, None] + 1  # 1, 3, 5, ...
crank = 1 - np.sum(
    8 / (odd * np.pi) ** 2 * np.exp(-((odd * np.pi) ** 2) * DE * t_in / (4 * L**2)), axis=0
)
print(f"largest difference from Crank's series: {np.max(np.abs(mean_in - crank)):.1e}")
assert np.max(np.abs(mean_in - crank)) < 2e-3
half = float(np.interp(0.5, mean_in, t_in))  # the mean rises steadily, so this inverts it
print(f"half way in after about {half:.1f} min ({half * DE / L**2:.2f} x L^2/De)")

# %% [markdown]
# Half of the final amount is in within about 1.3 minutes (0.2 L^2/De): for a reacting drug
# the delay is a minor point, the destruction (phi) is what keeps the deep cells protected.
#
# ## 4. The biofilm as a sink: coupling to the liquid
#
# In an infection the liquid is not held at a fixed level: it is a compartment fed by the
# dosing and emptied by clearance, and the biofilm takes drug out of it. Here the liquid
# is a flow cell over the biofilm. Everything is per unit area of biofilm surface: `M` is the
# amount of drug in the liquid above it, `V` = 1000 um is the depth of liquid (its volume
# per unit area), so the liquid concentration is M/V. Fresh medium at `Cin` = 10 mg/L flows
# through with clearance `CL` = 50 um/min (5 % of the liquid replaced per minute).
#
# Drug crosses the surface at `h*(M/V - C)` per unit area: a mass-transfer coefficient times
# the difference between the liquid and the biofilm's surface. As in a paper, this is an
# exchange condition under the biofilm's equation, `at z=200: -De*d_z(C) = h*(C - M/V)`: the
# left side is the diffusive flux along z, positive toward the liquid (at this upper end a
# positive flux leaves the biofilm), and in the right side `C` is its value at the surface.
# The same term, with `C(z=200)` for the surface value, is added to the equation of `M`, so
# what enters the biofilm is taken from the liquid. h = 1200 um/min stands for a thin
# unstirred layer of liquid at the surface.

# %%
COUPLED = """
model biofilm_bulk
axis z in [0, 200] n=40 units=um kind=spatial
param De=6000, kr=0.6, h=1200, V=1000, CL=50, Cin=10
field C(z)
compartment M                                     # drug in the liquid (per unit area)
d_t(C) = d_z(De*d_z(C)) - kr*C
    at z=200: -De*d_z(C) = h*(C - M/V)            # exchange through the surface ...
d_t(M) = CL*Cin - CL*M/V + h*(C(z=200) - M/V)     # ... with the liquid; fresh medium, washout
observable Cb = M/V
"""
coupled = tq.Model.from_text(COUPLED)
t = np.linspace(0, 60, 121)
sol = coupled.solve((0, 60), times=t, **TIGHT)
no_biofilm = 10 * (1 - np.exp(-50 / 1000 * t))  # the same flow cell without a biofilm
print(f"liquid at 60 min: {float(sol['Cb'][-1]):.3f} mg/L (without a biofilm {no_biofilm[-1]:.3f})")

# %% [markdown]
# **The books.** The ledger lists, for the biofilm and for the liquid, every way drug
# arrived or left. The amount that entered the biofilm through its surface must equal the
# amount the liquid lost through the same surface, at every time; and the drug that flowed
# in must equal what is in the liquid, plus what is in the biofilm, plus what was destroyed
# there, plus what was washed out.

# %%
print(sol.ledger.table())
moved = sol.ledger.moved
into_biofilm = moved["C"]["boundary z=upper: flux"]
out_of_liquid = moved["M"]["rate: h*(C(z=200) - M/V)"]
print(
    f"largest |entered biofilm + left liquid|: {np.max(np.abs(into_biofilm + out_of_liquid)):.1e}"
)
assert np.max(np.abs(into_biofilm + out_of_liquid)) <= 1e-12 * np.max(into_biofilm)

flowed_in = moved["M"]["rate: CL*Cin"]
accounted = (
    sol.ledger.amount["M"]
    + sol.ledger.amount["C"]
    - moved["C"]["hazard: kr"]  # destroyed (booked as a negative amount moved)
    - moved["M"]["rate: -CL*M/V"]  # washed out
)
print(f"drug in vs accounted for: largest difference {np.max(np.abs(flowed_in - accounted)):.1e}")
assert np.max(np.abs(flowed_in - accounted)) < 1e-9 * flowed_in[-1]

# %% [markdown]
# **Steady state.** At steady state the biofilm takes drug from the liquid as a
# **conductance** G (amount per unit area per minute, per mg/L in the liquid): the surface
# layer (h) and the reacting biofilm (De phi tanh(phi) / L, from the cosh profile) are two
# resistances in series, G = h phi tanh(phi) / (Bi + phi tanh(phi)), with the **Biot number**
# Bi = h L / De comparing the surface layer to the biofilm (here 40: the biofilm is the
# limiting step). The liquid then settles at Cin CL / (CL + G): the biofilm acts as a second
# clearance.

# %%
phi = L * np.sqrt(0.6 / DE)
Bi = 1200 * L / DE
G = 1200 * phi * np.tanh(phi) / (Bi + phi * np.tanh(phi))
steady = coupled.solve((0, 400), times=[400], **TIGHT)
cb = float(np.asarray(steady["Cb"])[-1])
print(
    f"G = {G:.1f} um/min; steady liquid {cb:.4f} mg/L; Cin CL/(CL + G) = {10 * 50 / (50 + G):.4f}"
)
assert abs(cb / (10 * 50 / (50 + G)) - 1) < 1e-3

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(t, sol["Cb"], label="liquid, with biofilm")
ax[0].plot(t, no_biofilm, "k--", label="liquid, no biofilm")
ax[0].plot(t, np.asarray(sol["C"])[:, 0], label="biofilm, at the substratum")
ax[0].set_xlabel("time (min)")
ax[0].set_ylabel("concentration (mg/L)")
ax[0].legend(fontsize="small")
for when in (2, 5, 20, 60):
    i = int(np.argmin(np.abs(t - when)))
    ax[1].plot(z_mid, np.asarray(sol["C"])[i], label=f"t = {when} min")
ax[1].set_xlabel("depth z (um; 0 = substratum, 200 = liquid)")
ax[1].set_ylabel("concentration in the biofilm (mg/L)")
ax[1].legend(fontsize="small")
fig.tight_layout()

# %% [markdown]
# ## What this buys you
#
# - The concentration a bacterium sees depends on how deep it sits. With the depth as an
#   axis, a drug effect can be driven by the local concentration, so that the deep cells are
#   the ones that survive, rather than by one effect-compartment concentration.
# - One number, phi = L sqrt(kr/De), tells whether penetration matters: with phi below
#   about 0.5 the biofilm is nearly uniform and a compartment is enough.
# - The biofilm and the liquid are one model: the liquid level falls because the biofilm
#   takes drug, and the ledger shows how much went where.
#
# **Limits of this version.** First-order destruction (a saturable enzyme, binding or a
# drug that is not destroyed at all behave differently: binding slows penetration but does
# not lower the steady state), a flat biofilm of fixed thickness, and a diffusivity that
# is the same at every depth.
