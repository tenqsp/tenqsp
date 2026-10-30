# %% [markdown]
# # Tutorial 2: where in the gut is a drug absorbed?
#
# Tutorial 1 showed that transit compartments are drug spread along an axis. Here the axes get
# a physiological meaning, and absorption depends on *where* the drug is:
#
# - **Stomach**, a field over the time since the dose was swallowed. Hardly any drug is
#   absorbed there; what matters is when it leaves (gastric emptying), the main cause of the
#   delay before absorption.
# - **Small intestine**, a field over position along it, from 0 (the duodenum) to 1 (the
#   ileocaecal valve). The contents travel along it in about 3.3 h (199 min; Yu & Amidon
#   1996), and absorption is fastest near the start and slower further along.
# - **Colon** and **body** as compartments.
#
# This is the structure of the continuous intestinal transit model in PK-Sim (Willmann et al.
# 2003, 2004), built on the dispersion model of Ni, Ho, Fox, Leuenberger and Higuchi (1980);
# the compartmental absorption and transit (CAT) model of Yu & Amidon (1996) is the same
# intestine cut into 7 compartments.
#
# **Assumptions** of this version:
#
# 1. the absorption rate falls log-linearly along the small intestine,
#    ka(x) = ka_si * ka_ratio^x, from the proximal to the distal value. This is typical for
#    drugs absorbed passively (the absorbing surface shrinks along the gut), not a rule:
#    carriers, efflux and pH-dependent ionisation give other patterns (metoprolol is absorbed
#    faster in the colon than in the ileum);
# 2. the contents move along the intestine at a constant speed;
# 3. the colon is one compartment;
# 4. the drug is in solution: no dissolution, precipitation or metabolism in the gut wall.
#
# **Values.** Absorption rates come from human regional effective permeabilities
# (Dahlgren et al., the same 14 volunteers, x 1e-4 cm/s, jejunum / ileum / colon), with
# ka = 2 Peff / R and R = 1.75 cm:
#
# | | Peff jejunum | ileum | colon | ka proximal (1/h) | distal / proximal | colon (1/h) |
# |---|---|---|---|---|---|---|
# | atenolol | 0.45 | 0.15 | 0.013 | 0.185 | 1/3 | 0.0053 |
# | ketoprofen | 8.85 | 6.53 | 3.37 | 3.64 | 0.74 | 1.39 |
#
# Clearance and volume are illustrative and the same for both drugs, so that every
# difference below comes from absorption.

# %%
import matplotlib.pyplot as plt
import numpy as np

import tenqsp as tq
from tenqsp.examples import oral_absorption_model

t = np.linspace(0, 24, 241)

# %% [markdown]
# ## 1. The model
#
# Read the stomach line as: the drug ages in the stomach (`-d_g(G)`, time passes at speed 1)
# and leaves at the emptying hazard `h_ge(g)`. A Weibull hazard,
# `(beta_ge/eta)(g/eta)^(beta_ge - 1)`, is constant when `beta_ge = 1` (first-order
# emptying, as for a liquid taken fasted) and rises with time when `beta_ge > 1` (a lag, as
# for a tablet waiting for the stomach's housekeeping wave). What leaves the stomach enters
# the small intestine at `x = 0`. That is a boundary condition, written under the intestine's
# equation as in a paper: `at x=0: S/T_si - D_x*d_x(S) = ...`. The left side is the flux
# along `x` that the equation's transport gives (carried at speed 1/T_si, minus what
# disperses back); the right side is what arrives there per hour. What reaches `x = 1` flows
# out (`at x=1: outflow`) into the colon: the colon gains the flux there, `S(x=1)/T_si`, the
# speed times the value at that end.
#
# The dose is written as for a starting value: an amount times a shape saying where along the
# axis it goes. `Dose*point(g, at=0)` puts it all at `g = 0`, just swallowed. (A density, or
# another shape such as `normal(...)`, works the same way.)
#
# The stomach's axis ends at 8 h. Drug still in the stomach at 8 h moves on into the small
# intestine with the emptied drug (`at g=8: outflow` under the stomach's equation, and
# `G(g=8)` in the intestine's inlet: time passes at speed 1, so the flux there is `G` itself),
# so nothing is lost whatever the emptying. The axis length only sets how long the stomach
# model follows the emptying hazard: with these values less than 1e-7 mg of 100 mg is still
# there at 8 h; with first-order emptying of half-time 2 h it would be 6 %, which would then
# all be emptied at 8 h, so the axis should be longer.

# %%
GUT = """
model gut
axis g in [0, 8] n=160 units=h kind=one_way    # time since the dose was swallowed
axis x in [0, 1] n=50 kind=spatial             # position along the small intestine

param Dose=100, eta=0.360674, beta_ge=1, T_si=3.316667, D_x=0
param ka_si=0.185, ka_ratio=0.333333, ka_col=0.0053, k_col=0.041667, CL=8, V=60

field G(g)        # stomach (amount per hour of age)
field S(x)        # small intestine (amount per unit length)

h_ge(g) = beta_ge/eta*(g/eta)^(beta_ge - 1)    # gastric emptying hazard
ka(x) = ka_si*ka_ratio^x                       # absorption rate along the intestine
emptying = int_g(h_ge*G)                       # amount per hour leaving the stomach
absorbed = int_x(ka*S)                         # amount per hour absorbed from the intestine

d_t(G) = -d_g(G) - h_ge*G                                  using sharp
    at g=8: outflow                            # on into the intestine (below)
d_t(S) = -d_x(S/T_si) + d_x(D_x*d_x(S)) - ka*S             using sharp
    at x=0: S/T_si - D_x*d_x(S) = emptying + G(g=8)        # from the stomach
    at x=1: outflow                            # through the ileocaecal valve

d_t(Col) = S(x=1)/T_si - ka_col*Col - k_col*Col     # k_col: leaving with the faeces
d_t(Ac) = absorbed + ka_col*Col - CL/V*Ac

dose G at t=0: Dose*point(g, at=0)             # swallowed: enters the stomach at g = 0
observable conc = Ac/V
"""
gut = tq.Model.from_text(GUT)
print(gut.summary())

# %% [markdown]
# The same model is `tenqsp.examples.oral_absorption_model()`, written with the Python API;
# given the values as rounded in the text above, the two give the same concentrations:

# %%
sol = gut.solve((0, 24), times=t, rtol=1e-10, atol=1e-12)
rounded = {"eta": 0.360674, "T_si": 3.316667, "ka_ratio": 0.333333, "k_col": 0.041667}
same = oral_absorption_model().solve((0, 24), times=t, params=rounded, rtol=1e-10, atol=1e-12)
diff = np.max(np.abs(np.asarray(sol["conc"]) - np.asarray(same["conc"])))
print(f"notation vs Python API: largest difference {diff:.1e}")
assert diff < 1e-7

# %% [markdown]
# ## 2. Gastric emptying sets the delay
#
# A liquid taken fasted empties first order with a half-time of about 15 min
# (`eta = 0.25/ln 2`, `beta_ge = 1`). A tablet is illustrated with a Weibull of scale 1 h and
# shape 1.5: little leaves in the first half hour, then emptying speeds up. The amount left in
# the stomach must be Dose exp(-(t/eta)^beta_ge), the Weibull survival function.
#
# Emptying matters most when the intestine absorbs fast, so this section uses the
# ketoprofen-like values (section 3); for atenolol, slow absorption hides most of the
# difference.

# %%
fast = {"ka_si": 3.64, "ka_ratio": 0.74, "ka_col": 1.39}  # ketoprofen-like
cases = {"liquid, fasted": fast, "tablet (Weibull 1 h, 1.5)": {**fast, "eta": 1.0, "beta_ge": 1.5}}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for label, p in cases.items():
    s = gut.solve((0, 24), times=t, params=p, rtol=1e-8, atol=1e-10)
    eta, b = p.get("eta", 0.360674), p.get("beta_ge", 1.0)
    in_stomach = np.sum(np.asarray(s["G"]) * (8.0 / 160), axis=1)
    expected = 100 * np.exp(-((t / eta) ** b))
    # within 1 % of the dose: the dose starts spread over the first cell (3 min of age);
    # halving the cells halves the difference
    assert np.max(np.abs(in_stomach - expected)) < 1.0, label
    ax[0].plot(t, in_stomach, label=label)
    ax[1].plot(t, s["conc"], label=label)
    print(f"{label}: peak {np.max(s['conc']):.3f} mg/L at {t[np.argmax(s['conc'])]:.1f} h")
ax[0].set_xlim(0, 4)
ax[0].set_ylabel("amount in the stomach (mg)")
ax[1].set_ylabel("plasma concentration (mg/L)")
for a in ax:
    a.set_xlabel("time (h)")
    a.legend(fontsize="small")
fig.tight_layout()

# %% [markdown]
# ## 3. Where the drug is absorbed shapes the curve
#
# Atenolol-like and ketoprofen-like absorption, with the same emptying and disposition.
# The snapshots show the drug along the small intestine at three times: ketoprofen is gone
# before it travels far; atenolol is still being absorbed (slowly) all the way along, and much
# of it reaches the colon, where absorption is slower still.

# %%
drugs = {
    "atenolol-like": {},
    "ketoprofen-like": {"ka_si": 3.64, "ka_ratio": 0.74, "ka_col": 1.39},
}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
results = {}
for k, (label, p) in enumerate(drugs.items()):
    s = gut.solve((0, 24), times=t, params=p, rtol=1e-8, atol=1e-10)
    results[label] = s
    ax[0].plot(t, s["conc"], label=label)
    tq.plot.snapshot(s, "S", axis="x", times=[0.5, 1.5, 3.0], ax=ax[1 + k])
    ax[1 + k].set_title(label)
    ax[1 + k].set_xlabel("position along the small intestine (0 duodenum, 1 ileocaecal valve)")
    ax[1 + k].set_ylabel("drug per unit length (mg)")
ax[0].set_xlabel("time (h)")
ax[0].set_ylabel("plasma concentration (mg/L)")
ax[0].legend(fontsize="small")
fig.tight_layout()

# %% [markdown]
# **How much is absorbed, and where.** (Atenolol's oral bioavailability is reported at about
# 40-50 %, so the atenolol-like values land in the right range.) With nothing cleared from
# the body (`CL = 0`), the body's amount at the end is all that was absorbed; switching off
# colon absorption as well gives the small intestine's share. For a log-linear ka and plug
# flow there is a closed form for the small intestine's fraction:
# F_si = 1 - exp(-ka_si T_si (ka_ratio - 1) / ln(ka_ratio)).

# %%
for label, p in drugs.items():
    total = gut.solve((0, 200), times=[0, 200], params={**p, "CL": 0.0})
    small = gut.solve((0, 200), times=[0, 200], params={**p, "CL": 0.0, "ka_col": 0.0})
    f_total = float(np.asarray(total["Ac"])[-1]) / 100
    f_si = float(np.asarray(small["Ac"])[-1]) / 100
    ka, r = p.get("ka_si", 0.185), p.get("ka_ratio", 0.333333)
    closed = 1 - np.exp(-ka * (199 / 60) * (r - 1) / np.log(r))
    print(
        f"{label}: absorbed {f_total:.1%} in all; small intestine {f_si:.1%} "
        f"(closed form {closed:.1%}), colon {f_total - f_si:.1%}"
    )
    assert abs(f_si - closed) < 0.003

# %% [markdown]
# ## 4. Checks against the published models
#
# Cut the small intestine into 7 cells with the `erlang` scheme (Tutorial 1) and keep ka
# constant: this is exactly the CAT model, whose fraction absorbed is
# 1 - (1 + ka T_si / 7)^-7 (Yu & Amidon 1999). (TenQSP's own tests also check this model
# against plug flow, the dispersion model's closed form and the Weibull emptying.)

# %%
# 7 cells, and the default (erlang) scheme instead of "using sharp" on the intestine
cat_text = GUT.replace("n=50 kind=spatial", "n=7 kind=spatial")
cat_text = cat_text.replace("ka*S             using sharp", "ka*S")
assert cat_text.count("using sharp") == 1  # only the stomach's remains
cat = tq.Model.from_text(cat_text)
s = cat.solve(
    (0, 60), times=[0, 60], params={"ka_ratio": 1.0, "ka_si": 0.5, "ka_col": 0.0, "CL": 0.0}
)
f = float(np.asarray(s["Ac"])[-1]) / 100
yu_amidon = 1 - (1 + 0.5 * (199 / 60) / 7) ** -7
print(f"7 cells: absorbed {f:.6f}; CAT closed form {yu_amidon:.6f}")
assert abs(f - yu_amidon) < 1e-6

# %% [markdown]
# ## 5. Repeated doses
#
# A regimen given at simulation time replaces the dose written in the model. Into a field,
# it takes the same shape as the model's dose: here 100 mg every 12 h for a week, swallowed
# as tablets (the Weibull emptying of section 2), atenolol-like.
#
# The model is linear in the drug, so the concentration under the regimen must be the sum of
# single-dose curves, each shifted to its dose time: a check of the regimen.

# %%
tablet = {"eta": 1.0, "beta_ge": 1.5}
sim = gut.simulator(outputs=["conc"], rtol=1e-10, atol=1e-12)
week = np.linspace(0, 168, 337)
regimen = tq.bolus("G", 100, start=0, every=12, n=14, shape=tq.point("g", at=0))
repeated = np.asarray(sim(params=tablet, times=week, regimen=regimen)["conc"]).ravel()
single = np.asarray(sim(params=tablet, times=week)["conc"]).ravel()  # the model's dose
summed = sum(np.interp(week - 12 * k, week, single, left=0.0) for k in range(14))
print(f"regimen vs sum of single doses: largest difference {np.max(np.abs(repeated - summed)):.1e}")
assert np.max(np.abs(repeated - summed)) < 1e-6

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(week, repeated)
ax.set_xlabel("time (h)")
ax.set_ylabel("plasma concentration (mg/L)")
ax.set_title("100 mg every 12 h, tablets, atenolol-like")
fig.tight_layout()

# %% [markdown]
# ## Limits of this version
#
# - Drug in solution only; no dissolution, precipitation, gut-wall metabolism or transporters.
# - A constant transit speed and one Weibull for emptying (fed-state, multi-phase emptying
#   would need another hazard).
