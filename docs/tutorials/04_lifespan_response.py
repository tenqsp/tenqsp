# %% [markdown]
# # Tutorial 4: cells that live a fixed time
#
# Indirect response models describe a response produced at rate k_in and lost at a first-order
# rate k_out: every cell, whatever its age, has the same chance of being lost in the next hour.
# For cells with a fairly fixed lifespan (red blood cells, platelets, neutrophils in the
# blood) that is not how they are lost. Krzyzanski, Ramakrishnan & Jusko (J Pharmacokinet
# Biopharm 1999) proposed **lifespan-based** indirect response models instead: cells are
# produced at k_in(t) and each lives exactly T_R. The number of cells R(t) then obeys
#
#     dR/dt = k_in(t) - k_in(t - T_R),
#
# a delay differential equation: what leaves now is what was produced T_R ago. NONMEM and
# Monolix need workarounds for the delay (a long chain of transit compartments, or a stored
# history of k_in).
#
# With an axis there is no delay to handle. The cells are spread along their **age**: they
# enter at age 0, all age at the same speed (one day per day), and leave when they reach the
# end of the axis at age T_R. R(t) is the total along the axis.
#
# What you will see:
#
# 1. the model, and its response to a step in k_in: a ramp, then a plateau;
# 2. the `sharp` scheme: a pure delay, checked against the closed form;
# 3. the `erlang` scheme: the same model with a spread of lifespans (also a closed form);
# 4. a drug that stimulates production, checked against the integral of k_in.
#
# **Values** are illustrative: a lifespan of 4 days, production 2 cells per day at baseline.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammainc

import tenqsp as tq

t = np.linspace(0, 8, 401)  # days
kin0, kin1, TR = 2.0, 5.0, 4.0

# %% [markdown]
# ## 1. The model and a step in production
#
# `u(a)` is the density of cells of age `a` (cells per day of age). Production enters at age 0:
# the boundary condition `at a=0: u = kin1`, written under the equation as in a paper (cells
# age at one day per day, so the cells crossing age 0 per day are `u` there). At age T_R the
# cells die: `at a=4: outflow`, what reaches that age leaves, counted in the ledger. At
# baseline, production k_in0 fills every age equally, so the starting density is k_in0 and
# R0 = k_in0 T_R.
#
# Production steps from k_in0 to k_in1 at t = 0. The cells born after the step form a denser
# band that moves along the age axis; the cells dying are still those born before it, at
# the old rate. So R rises linearly, by k_in1 - k_in0 per day, until the first cell born after
# the step reaches T_R; then births and deaths balance again:
#
#     R(t) = k_in0 T_R + (k_in1 - k_in0) min(t, T_R).
#
# The axis is cut into **cells** (pieces of age, here 160 of 0.025 day each); the **scheme**
# is the rule that moves the cells' content from one piece to the next.

# %%
LIFESPAN = """
model lifespan
axis a in [0, 4] n={n} units=day kind=one_way     # age of the cell
param kin0=2, kin1=5
field u(a)                                         # cells per day of age
d_t(u) = -d_a(u){sharp}                            # every cell ages at one day per day
    at a=0: u = kin1                               # production, after the step
    at a=4: outflow                                # cells die at age T_R = 4
u(0) = kin0                                        # the baseline: R0 = kin0*T_R
observable R = int_a(u)
"""


def lifespan(n, scheme):
    return tq.Model.from_text(LIFESPAN.format(n=n, sharp="    using sharp" * (scheme == "sharp")))


exact = kin0 * TR + (kin1 - kin0) * np.minimum(t, TR)
sol = {
    s: lifespan(160, s).solve((0, 8), times=t, rtol=1e-10, atol=1e-12) for s in ("sharp", "erlang")
}

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(t, exact, color="black", lw=3, alpha=0.3, label="closed form")
for s in ("sharp", "erlang"):
    ax[0].plot(t, sol[s]["R"], label=f"{s}, 160 cells")
ax[0].set_xlabel("time (days)")
ax[0].set_ylabel("R, number of cells")
ax[0].legend(fontsize="small")
ax[0].set_title("a step in production at t = 0")
for k, s in enumerate(("sharp", "erlang")):
    tq.plot.snapshot(sol[s], "u", axis="a", times=[0, 1, 2, 3, 3.9], ax=ax[1 + k])
    ax[1 + k].set_ylim(0, 6)
    ax[1 + k].set_xlabel("age (days)")
    ax[1 + k].set_ylabel("cells per day of age")
    ax[1 + k].set_title(f"{s} scheme: the cells by age")
fig.tight_layout()

# %% [markdown]
# With `sharp`, the band of new cells keeps a steep front as it moves: a pure delay. With
# `erlang`, the front spreads as it moves, so some cells reach T_R early and the corners of the
# ramp are rounded.
#
# ## 2. The sharp scheme: a pure delay
#
# The closed form has a corner (a kink) at t = T_R. No grid can follow a corner exactly: the
# `sharp` scheme rounds it over a few cells, and the rounding narrows as the cells are
# refined. The largest error, at the corner, falls by about 40 % each time the cells are
# doubled (slower than on a smooth solution, as it must at a corner); away from it, the ramp
# and the plateau are reproduced to better than 1e-4 cells on 160 cells.

# %%
errors = []
corner = np.abs(t - TR) < 0.4  # 0.4 day either side of the corner
for n in (40, 80, 160):
    s = lifespan(n, "sharp").solve((0, 8), times=t, rtol=1e-10, atol=1e-12)
    err = np.abs(np.asarray(s["R"]) - exact)
    errors.append(err.max())
    away = err[~corner].max()
    print(f"{n:4d} cells: largest error {err.max():.3f} cells; away from the corner {away:.1e}")
rise = (kin1 - kin0) * TR
assert errors[0] > errors[1] > errors[2]
assert errors[-1] < 0.01 * rise  # within 1 % of the rise, even at the corners, on 160 cells
assert away < 1e-4
# a pure delay: R changes in no other way (the plateau is flat and at the right level)
plateau = np.asarray(s["R"])[t > TR + 0.5]
assert np.max(np.abs(plateau - kin1 * TR)) < 1e-3

# %% [markdown]
# ## 3. The erlang scheme: the same model with a spread of lifespans
#
# With the `erlang` scheme each cell of the age axis is a compartment emptying into the next
# at a constant rate, so N cells are a chain of N transit compartments (Tutorial 1). A cell's
# lifespan is then not exactly T_R but gamma (Erlang) distributed, with mean T_R and
# coefficient of variation 1/sqrt(N). That is a lifespan model too, with a spread of
# lifespans, and its step response also has a closed form:
#
#     R(t) = k_in0 T_R + (k_in1 - k_in0) * integral from 0 to t of S(s) ds,
#
# where S(s) = 1 - P(N, N s/T_R) is the fraction of cells still alive at age s (P is the
# regularised incomplete gamma function). The model matches it to solver tolerance on every
# grid: `erlang` is not a poor approximation of the lifespan model, it is exactly the model
# with gamma-distributed lifespans. With few cells it is far from a fixed lifespan:


# %%
def erlang_step_response(n, times):
    c = n / TR
    # integral of P(n, c s) from 0 to t, in closed form
    int_p = times * gammainc(n, c * times) - (n / c) * gammainc(n + 1, c * times)
    return kin0 * TR + (kin1 - kin0) * (times - int_p)


fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, exact, color="black", lw=3, alpha=0.3, label="fixed lifespan (closed form)")
for n in (4, 16, 160):
    s = lifespan(n, "erlang").solve((0, 8), times=t, rtol=1e-10, atol=1e-12)
    diff = np.max(np.abs(np.asarray(s["R"]) - erlang_step_response(n, t)))
    print(
        f"erlang, {n:3d} cells (lifespan CV {1 / np.sqrt(n):.2f}): differs from gamma "
        f"closed form by {diff:.1e}; from fixed lifespan by {np.max(np.abs(s['R'] - exact)):.2f}"
    )
    assert diff < 1e-6
    ax.plot(t, s["R"], label=f"erlang, {n} cells (lifespan CV {1 / np.sqrt(n):.2f})")
ax.set_xlabel("time (days)")
ax.set_ylabel("R, number of cells")
ax.legend(fontsize="small")
fig.tight_layout()

# %% [markdown]
# So the choice between the schemes is a choice of model, not of accuracy: `sharp` for a fixed
# lifespan (Krzyzanski et al. 1999), `erlang` with N cells for gamma-distributed lifespans of
# CV 1/sqrt(N). On 160 cells, `erlang`'s lifespan CV is 8 %, and the corner is rounded over
# more than a day; `sharp` on the same cells rounds it over about 5 hours:

# %%
for s_ in ("sharp", "erlang"):
    off = np.abs(np.asarray(sol[s_]["R"]) - exact) > 0.001 * rise  # off by > 0.1 % of the rise
    width = np.ptp(t[off])
    print(f"{s_:6s}, 160 cells: off the fixed-lifespan response for {width:.2f} days around T_R")
    assert (width < 0.3) if s_ == "sharp" else (width > 1.0)

# %% [markdown]
# ## 4. A drug that stimulates production
#
# Now production depends on a drug, as in Krzyzanski et al.'s stimulation models:
# k_in(t) = k_in0 (1 + S_max C/(SC50 + C)), with C from a one-compartment intravenous bolus,
# C = C0 exp(-k t). Since each cell lives exactly T_R, the cells alive at t are those born
# in the last T_R days:
#
#     R(t) = integral from t - T_R to t of k_in(s) ds,
#
# and for this k_in the integral is a closed form:
# R(t) = k_in0 T_R + (k_in0 S_max/k) ln[(SC50 + C(max(t - T_R, 0)))/(SC50 + C(t))].

# %%
DRUG = """
model lifespan_drug
axis a in [0, 4] n=400 units=day kind=one_way
param kin0=2, Smax=3, SC50=1, k=0.7, V=10
C = Ac/V
field u(a)
d_t(u) = -d_a(u)    using sharp
    at a=0: u = kin0*(1 + Smax*C/(SC50 + C))
    at a=4: outflow
d_t(Ac) = -k*Ac
u(0) = kin0
dose Ac at t=0: 50
observable R = int_a(u)
observable conc = C
"""
td = np.linspace(0, 12, 601)
drug = tq.Model.from_text(DRUG).solve((0, 12), times=td, rtol=1e-10, atol=1e-12)
smax, sc50, k, c0 = 3.0, 1.0, 0.7, 5.0
conc = c0 * np.exp(-k * td)
born_then = c0 * np.exp(-k * np.maximum(td - TR, 0.0))
r_exact = kin0 * TR + kin0 * smax / k * np.log((sc50 + born_then) / (sc50 + conc))
err = np.abs(np.asarray(drug["R"]) - r_exact)
print(f"largest difference from the closed form: {err.max():.3f} cells (peak {r_exact.max():.2f})")
assert np.max(np.abs(np.asarray(drug["conc"]) - conc)) < 1e-6
assert err.max() < 0.01 * (r_exact.max() - kin0 * TR)
assert abs(td[np.argmax(drug["R"])] - TR) < 0.05  # the peak is at T_R

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(td, drug["conc"])
ax[0].set_ylabel("drug concentration (mg/L)")
ax[1].plot(td, r_exact, color="black", lw=3, alpha=0.3, label="closed form")
ax[1].plot(td, drug["R"], label="sharp, 400 cells")
ax[1].axvline(TR, color="grey", ls=":", label="T_R")
ax[1].set_ylabel("R, number of cells")
ax[1].legend(fontsize="small")
for a_ in ax:
    a_.set_xlabel("time (days)")
fig.tight_layout()

# %% [markdown]
# The response rises while the drug is present and peaks exactly at T_R, with a corner: from
# then on the cells dying are those born under the drug, at a higher rate than today's
# production, so R falls, back towards baseline by about 2 T_R. With first-order loss (k_out)
# the peak would come when production fell back to k_out R, without a corner, and the return
# would follow k_out instead of the drug's history.
#
# ## What this buys you
#
# - A lifespan model is an ordinary model: no delay differential equation, no stored history
#   of k_in; the drug, k_in and any feedback are written as in an indirect response model.
# - The scheme is the lifespan distribution: `sharp` for a fixed lifespan, `erlang` with N
#   cells for a gamma distribution of CV 1/sqrt(N). Any other distribution is a hazard along
#   the age axis (`- h(a)*u`, with h(a) the hazard of the lifespan distribution).
# - The age axis is a real coordinate: cells of different ages can respond differently (e.g.
#   an effect only on young cells).
#
# **Limits.** A fixed lifespan has corners that no grid follows exactly; with `sharp` the
# rounding is a few cells wide. Use enough cells that a few cells are short compared to the
# times you care about.
