# %% [markdown]
# # Tutorial 12: drug tolerance in a tumour, a continuum of resistance levels
#
# A tumour exposed to a cytotoxic drug shrinks, and then often grows back while the drug is
# still given. Part of the regrowth need not be due to resistance mutations. Sharma et al.
# (Cell 2010) showed that cancer cell lines contain a small, reversible, **drug-tolerant**
# subpopulation: cells whose expression of stress-response programmes protects them, at the
# cost of dividing more slowly, and whose offspring become sensitive again after a drug
# holiday. Two things move a population toward tolerance under drug:
#
# - **selection**: the drug kills the less tolerant cells faster;
# - **adaptation**: the drug itself pushes surviving cells toward tolerance (stress-induced
#   changes in expression).
#
# In a compartmental model you would declare a sensitive state S and a resistant state R and
# a switching rate between them. But tolerance is a *level*: the expression of the protective
# programme varies continuously from cell to cell, is partly passed on at division, and
# changes a little at random. Lorenzi, Chisholm and Clairambault (Biol Direct 2016) wrote this
# as a population density n(x, t) over a resistance level x (0: the most proliferative
# cells, 1: the most resistant), and Chisholm, Lorenzi, Lorz et al. (Cancer Res 2015; Commun
# Math Sci 2016) added the drug's push as a **drift** along the level, at a speed that grows
# with the drug.
#
# What you will see:
#
# 1. the model in TenQSP's notation, one line per biological process;
# 2. without drug, the population sits at a stable spread of levels, as the paper derives;
# 3. under a constant drug: selection alone, then selection plus the drug's drift;
# 4. the whole distribution reduces *exactly* to three ODEs (size, mean level, spread) when
#    the fitness is quadratic in the level, checked at every time;
# 5. weekly doses with pharmacokinetics, then a drug holiday: the population becomes
#    sensitive again;
# 6. a bounded level, where the three ODEs no longer hold and the continuum is needed.
#
# **Values.** Proliferation, its cost of resistance, the rate of random changes and the
# carrying capacity are those of Lorenzi et al. 2016 (Table 1, per day). The drift speed per
# unit of drug, the drug's levels and its elimination are illustrative.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import tenqsp as tq
from tenqsp.examples import drug_tolerance_model

GAMMA, ETA, BETA, K = 0.66, 0.132, 0.01, 1e8
D = GAMMA / K
TIGHT = {"rtol": 1e-10, "atol": 1e-4}

# %% [markdown]
# ## 1. The model
#
# Cells at level x divide at gamma - eta x^2 per day (fastest at x = 0: resistance costs
# proliferation), die from competition at d rho with rho the tumour size and d = gamma/K, and
# are killed by the drug at c (x - 1)^2 (least at x = 1). Random heritable changes of
# expression spread the levels (diffusion at rate beta), and the drug moves cells toward
# resistance at speed vbar c:
#
#     dn/dt = - d/dx(vbar c n) + beta d2n/dx2 + [gamma - eta x^2 - d rho - c (x - 1)^2] n
#
# The drug level is a compartment `C` (per day, a death rate), so it can be dosed and
# eliminated. The paper solves on [-4, 4]; the ends are sealed (no cell leaves the axis),
# written under the equation as the flux there, drift minus diffusion, equal to 0. The same
# model is `tenqsp.examples.drug_tolerance_model`.

# %%
MODEL = """
model drug_tolerance
axis x in [-4, 4] n=160                           # resistance level

param gamma=0.66, eta=0.132, beta=0.01, K=1e8     # per day
param vbar=0.02                                   # drift per unit of drug
param ke=0, c0=0                                  # drug elimination, level at t = 0
param n0=1e6, x0=0, s0=0.5                        # cells at t = 0: number, mean, sd

field n(x)                                        # cancer cells by resistance level
compartment C                                     # drug level, per day

d = gamma/K
rho = int_x(n)                                    # tumour size

d_t(n) = -d_x(vbar*C*n) + d_x(beta*d_x(n))
         + (gamma - eta*x^2)*n - d*rho*n - C*(x - 1)^2*n      using sharp
    at x=-4: vbar*C*n - beta*d_x(n) = 0
    at x=4:  vbar*C*n - beta*d_x(n) = 0
d_t(C) = -ke*C

n(0) = n0*normal(x, mean=x0, sd=s0)
C(0) = c0

observable size = rho
observable mean_x = int_x(x*n)/rho
observable var_x = int_x(x^2*n)/rho - (int_x(x*n)/rho)^2
"""
tumour = tq.Model.from_text(MODEL)
assert tumour.to_ir() == drug_tolerance_model().to_ir()  # the same model
print(tumour.explain().split("Boundaries")[0])

# %% [markdown]
# `explain()` reads the drift and the diffusion together as a Fokker-Planck equation: each
# cell's level follows dx = vbar C dt + sqrt(2 beta) dW. It also notes that the drug's kill
# term reads `C` but does not use it up; that is intended here (the drug level is not
# depleted by the cells it kills).
#
# ## 2. Without drug: a stable spread of levels
#
# Lorenzi et al. prove (their Theorem 1) that without drug the population settles to a
# Gaussian centred on the most proliferative level, x = 0, with variance sqrt(beta/eta) and
# size (gamma - sqrt(beta eta))/d: selection pulls the levels together, random changes spread
# them, and the balance costs sqrt(beta eta) of growth rate. Start from a few cells away from
# that balance and check where they end up.

# %%
sol0 = tumour.solve((0, 300), times=np.linspace(0, 300, 61), params={"x0": 0.4, "s0": 0.3}, **TIGHT)
rho_eq = (GAMMA - np.sqrt(BETA * ETA)) / D
var_eq = np.sqrt(BETA / ETA)
size0, mean0, var0 = (float(np.asarray(sol0[k])[-1]) for k in ("size", "mean_x", "var_x"))
print(f"size     {size0:.4e}   theorem {rho_eq:.4e}")
print(f"mean     {mean0:.1e}      theorem 0")
print(f"variance {var0:.5f}      theorem {var_eq:.5f}")
assert abs(size0 / rho_eq - 1) < 2e-4
assert abs(mean0) < 1e-4
assert abs(var0 / var_eq - 1) < 2e-3

# %% [markdown]
# The 0.06 % left in the variance is the grid's (160 cells along x); it shrinks as the cells
# are refined (section 5 shows this for the size).
#
# ## 3. A constant drug: selection, then selection plus drift
#
# From now on the tumour starts at that drug-free balance (9.4 x 10^7 cells). A constant drug
# level c changes the fitness to gamma_c - eta_c (x - x_c)^2 - d rho, with eta_c = eta + c,
# x_c = c/eta_c and gamma_c = gamma - eta c/eta_c: the best level moves to x_c, and
# Theorem 2 of the paper gives the new balance. In a pharmacometric reading, the dose that
# brings the size down by a given fraction (the paper's LC levels) follows in closed form.
# Here c = 2 per day:

# %%
START = {"n0": rho_eq, "x0": 0.0, "s0": var_eq**0.5}


def balance(c, vbar):
    """Size, mean and variance at a constant drug level c (Theorem 2 of Lorenzi et al. 2016
    for vbar = 0; the drift terms are derived in section 4)."""
    eta_c, v = ETA + c, vbar * c
    size = (GAMMA - ETA * c / eta_c - np.sqrt(BETA * eta_c) - v**2 / (4 * BETA)) / D
    return size, c / eta_c + v / (2 * np.sqrt(BETA * eta_c)), np.sqrt(BETA / eta_c)


c, VBAR = 2.0, 0.02
print(f"c = {c}: the balance falls to {balance(c, 0)[0] / rho_eq:.0%} of the drug-free size")
t3 = np.linspace(0, 120, 241)
runs = {
    vbar: tumour.solve((0, 120), times=t3, params={**START, "c0": c, "vbar": vbar}, **TIGHT)
    for vbar in (0.0, VBAR)
}
for vbar, s in runs.items():
    size, mean = np.asarray(s["size"]), np.asarray(s["mean_x"])
    exp_size, exp_mean, _ = balance(c, vbar)
    print(
        f"vbar = {vbar}: lowest size {size.min():.2e}; day 5 {size[t3 == 5][0]:.2e}; "
        f"day 120 {size[-1]:.3e} (balance {exp_size:.3e}); "
        f"mean level day 5 {mean[t3 == 5][0]:.2f}, day 120 {mean[-1]:.3f} (balance {exp_mean:.3f})"
    )
    assert abs(size[-1] / exp_size - 1) < 1e-3 and abs(mean[-1] - exp_mean) < 1e-3
early = [np.asarray(s["size"])[t3 == 5][0] for s in runs.values()]
late = [np.asarray(s["size"])[-1] for s in runs.values()]
assert early[1] > early[0] and late[1] < late[0]

# %% [markdown]
# Both runs show the classic picture of an in-vitro experiment with a high dose: a
# bottleneck within two days, then regrowth under the drug as the population moves toward
# resistance. With selection alone, the drug can only favour the more resistant cells that
# random changes have already produced. The drift pushes the survivors toward resistance
# while they are being selected, so the bottleneck is shallower and the regrowth faster.
#
# In the long run the same drift is a cost. At the new balance the mean level overshoots
# the best level x_c by vbar c/(2 sqrt(beta eta_c)), and the size is lower by
# (vbar c)^2/(4 beta d): a population that is always pushed past its optimum grows less.
# Which effect dominates depends on the time scale of the treatment, a question a
# two-state model cannot ask.

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
xc = np.linspace(-4, 4, 161)
xc = 0.5 * (xc[1:] + xc[:-1])
for vbar, s in runs.items():
    label = "selection only" if vbar == 0 else f"selection + drift (vbar = {vbar})"
    ax[0].semilogy(t3, s["size"], label=label)
    ax[1].plot(t3, s["mean_x"], label=label)
for day in (0, 10, 30, 120):
    ax[2].plot(xc, np.asarray(runs[VBAR]["n"])[np.searchsorted(t3, day)], label=f"day {day}")
ax[0].set_ylabel("tumour size (cells)")
ax[1].set_ylabel("mean resistance level")
ax[1].axhline(c / (ETA + c), color="k", ls=":", label="best level x_c")
ax[2].set_xlim(-2, 2.5)
ax[2].set_xlabel("resistance level x")
ax[2].set_ylabel("cells per unit of x")
ax[2].set_title("with drift: the distribution moves")
for a_ in ax[:2]:
    a_.set_xlabel("time (days)")
for a_ in ax:
    a_.legend(fontsize="small")
fig.tight_layout()

# %% [markdown]
# ## 4. Three ODEs, exactly
#
# The fitness is quadratic in x, the drift does not depend on x and the diffusion is
# constant. Then a Gaussian distribution stays Gaussian, and only three numbers change: the
# size rho, the mean level m and the variance s. Writing log n as a quadratic in x and
# matching the powers of x gives, with c = C(t),
#
#     ds/dt   = 2 beta - 2 eta_c s^2
#     dm/dt   = vbar c + 2 eta_c s (x_c - m)
#     drho/dt = rho [gamma_c - eta_c ((m - x_c)^2 + s) - d rho]
#
# This is a model a pharmacometrician can fit in NONMEM or Monolix: the size is the
# observation, the mean level is a latent "tolerance" state that the drug drives, and the
# variance is the heterogeneity on which selection acts (its own ODE, not a fixed
# parameter). The continuum is what tells you that these are the right three equations,
# and what each parameter means. Check them at every time point against the PDE, through
# the bottleneck and the regrowth:


# %%
def three_odes(t, y0, vbar, c_of_t, breaks=()):
    """rho, m, s at the times t; `breaks` are the times where c(t) jumps (doses): the ODEs
    are solved from one to the next, as a solver must across a jump."""

    def rhs(tt, y, hi):
        rho, m, s = y
        cc = c_of_t(min(tt, hi - 1e-9))  # within a segment: the level before the next dose
        eta_c, x_c = ETA + cc, cc / (ETA + cc)
        gamma_c = GAMMA - ETA * cc / eta_c
        return [
            rho * (gamma_c - eta_c * ((m - x_c) ** 2 + s) - D * rho),
            vbar * cc + 2 * eta_c * s * (x_c - m),
            2 * BETA - 2 * eta_c * s * s,
        ]

    edges = sorted({t[0], t[-1], *[b for b in breaks if t[0] < b < t[-1]]})
    out, y = np.empty((3, len(t))), np.asarray(y0, float)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (t >= lo) & (t <= hi)
        seg = solve_ivp(
            rhs, (lo, hi), y, args=(hi,), method="Radau", rtol=1e-12, atol=1e-12,
            dense_output=True,
        )  # fmt: skip
        out[:, sel] = seg.sol(t[sel])
        y = seg.y[:, -1]
    return out


rho3, m3, s3 = three_odes(t3, [rho_eq, 0.0, var_eq], VBAR, lambda tt: c)
size_pde, mean_pde = np.asarray(runs[VBAR]["size"]), np.asarray(runs[VBAR]["mean_x"])
print(f"size: largest relative difference {np.max(np.abs(size_pde / rho3 - 1)):.1e}")
print(f"mean level: largest difference   {np.max(np.abs(mean_pde - m3)):.1e}")
assert np.max(np.abs(size_pde / rho3 - 1)) < 2e-3
assert np.max(np.abs(mean_pde - m3)) < 1e-3

# %% [markdown]
# The remaining differences are the grid's (160 cells along x): section 5 shows them
# shrinking as the cells are refined.
#
# ## 5. Doses, elimination and a drug holiday
#
# Now the drug is given as a dose of 6 every 3.5 days for four weeks (into `C`, eliminated
# at ke = 0.5 per day: a half-life of 1.4 days), then stopped for eight weeks. Doses are
# given at simulation time, as for any TenQSP model; the drug level drives both the kill and
# the drift. After the last dose the drift stops, selection now favours proliferation
# again, and the population drifts back to sensitivity, as the drug-tolerant cells of Sharma
# et al. did after a drug holiday.

# %%
dose_times = np.arange(0, 28, 3.5)
regimen = tq.bolus("C", 6.0, start=0, every=3.5, n=len(dose_times))
t5 = np.linspace(0, 84, 337)
scenario = tq.Scenario(regimen=regimen, params={**START, "ke": 0.5, "vbar": VBAR})
sim = tumour.simulate(scenario, times=t5, **TIGHT)
level = np.asarray(sim.solution["C"])  # states other than the outputs are in sim.solution


def c_of_t(tt):
    past = dose_times[dose_times <= tt + 1e-12]
    return float(np.sum(6.0 * np.exp(-0.5 * (tt - past))))


rho5, m5, s5 = three_odes(t5, [rho_eq, 0.0, var_eq], VBAR, c_of_t, breaks=dose_times)
size5, mean5 = np.asarray(sim["size"]), np.asarray(sim["mean_x"])
print(f"lowest size {size5.min():.2e}; mean level at day 28: {mean5[t5 == 28][0]:.2f}, "
      f"at day 84: {mean5[-1]:.2f}")  # fmt: skip
print("ledger, largest relative unexplained:", f"{max(sim.solution.ledger.check().values()):.1e}")
assert max(sim.solution.ledger.check().values()) < 1e-10
assert mean5[-1] < 0.5 * mean5[t5 == 28][0]  # resensitised during the holiday

# %% [markdown]
# Under the doses the drug level reaches 7 per day, the best level is sharper and the
# distribution narrower (a standard deviation of 0.2, four cells of the grid), so the grid
# shows. The three ODEs are compared with the model on 160 and on 320 cells: the size and
# the mean level converge to them.

# %%
fine = tq.Model.from_text(MODEL.replace("n=160", "n=320")).simulate(scenario, times=t5, **TIGHT)
for label, s in (("160 cells", sim), ("320 cells", fine)):
    print(
        f"{label}: three ODEs vs PDE, size {np.max(np.abs(np.asarray(s['size']) / rho5 - 1)):.1e},"
        f" mean level {np.max(np.abs(np.asarray(s['mean_x']) - m5)):.1e}"
    )
err_coarse = np.max(np.abs(size5 / rho5 - 1))
err_fine = np.max(np.abs(np.asarray(fine["size"]) / rho5 - 1))
assert err_fine < err_coarse / 3 and err_fine < 1.5e-3

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(t5, level)
ax[0].set_ylabel("drug level C (per day)")
ax[1].semilogy(t5, size5, label="PDE")
ax[1].semilogy(t5, rho5, "k:", label="three ODEs")
ax[1].set_ylabel("tumour size (cells)")
ax[2].plot(t5, mean5, label="PDE")
ax[2].plot(t5, m5, "k:", label="three ODEs")
ax[2].set_ylabel("mean resistance level")
for a_ in ax:
    a_.axvspan(28, 84, color="0.9", zorder=0)
    a_.set_xlabel("time (days); shaded: drug holiday")
for a_ in ax[1:]:
    a_.legend(fontsize="small")
fig.tight_layout()

# %% [markdown]
# ## 6. When three ODEs are not enough
#
# The reduction rests on the Gaussian shape: a quadratic fitness, a drift and a diffusion
# that are the same at every level, and a level free to take any value. Chisholm et al.
# (2015) instead bound the levels to [0, 1] with sealed ends: a cell cannot be more
# resistant than fully resistant. Take a strong drift (vbar = 0.3, so 0.6 per day at c = 2)
# and a small, fairly sensitive tumour. On the unbounded level the three ODEs carry the
# mean far past x = 1, where the drug kills again, and the tumour dies out: the cost
# (vbar c)^2/(4 beta) of section 3 now exceeds the growth rate. On the bounded level the
# cells pile up against x = 1, where the drug does not kill, the distribution is no longer
# Gaussian, and the tumour grows. The continuum needs no assumption on the shape; only the
# axis changes. Two checks: the books close (every cell born and killed is accounted for),
# and the grid is fine enough (doubling the cells changes the size by little).

# %%
BOUNDED = (
    MODEL.replace("[-4, 4] n=160", "[0, 1] n={n}")
    .replace("at x=-4:", "at x=0:")
    .replace("at x=4: ", "at x=1:")
)
strong = {"c0": c, "vbar": 0.3, "n0": 1e6, "x0": 0.3, "s0": 0.1}
t6 = np.linspace(0, 60, 121)
bounded = {
    n: tq.Model.from_text(BOUNDED.format(n=n)).solve((0, 60), times=t6, params=strong, **TIGHT)
    for n in (100, 200)
}
b = bounded[200]
rho6, m6, s6 = three_odes(t6, [1e6, 0.3, 0.01], 0.3, lambda tt: c)
change = np.max(np.abs(np.asarray(bounded[100]["size"]) / np.asarray(b["size"]) - 1))
print(f"mean level at day 60: bounded continuum {float(np.asarray(b['mean_x'])[-1]):.3f}, "
      f"three ODEs {m6[-1]:.3f}")  # fmt: skip
print(f"size at day 60: bounded continuum {float(np.asarray(b['size'])[-1]):.3e}, "
      f"three ODEs {rho6[-1]:.3e}")  # fmt: skip
print(f"100 vs 200 cells: size differs by at most {change:.1e}")
print("ledger, largest relative unexplained:", f"{max(b.ledger.check().values()):.1e}")
assert max(b.ledger.check().values()) < 1e-10
assert change < 1e-3
assert m6[-1] > 1 >= float(np.asarray(b["mean_x"])[-1])
assert rho6[-1] < 1 and float(np.asarray(b["size"])[-1]) > 1e7

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
xb = (np.arange(200) + 0.5) / 200
for day in (0, 5, 20, 60):
    ax[0].plot(xb, np.asarray(b["n"])[np.searchsorted(t6, day)], label=f"day {day}")
ax[0].set_xlabel("resistance level x (bounded)")
ax[0].set_ylabel("cells per unit of x")
ax[0].set_title("cells pile up at full resistance")
ax[1].semilogy(t6, b["size"], label="bounded continuum")
ax[1].semilogy(t6, np.where(rho6 > 1, rho6, np.nan), "k:", label="three ODEs (unbounded)")
ax[1].set_xlabel("time (days)")
ax[1].set_ylabel("tumour size (cells)")
for a_ in ax:
    a_.legend(fontsize="small")
fig.tight_layout()

# %% [markdown]
# ## What this buys you
#
# - Selection, random heritable variation and the drug's push toward tolerance are three
#   terms of one equation, each with a meaning a biologist can discuss; the drug enters
#   through a compartment, so any pharmacokinetics and any regimen can drive it.
# - When the fitness is quadratic and the drift uniform, the continuum reduces exactly to
#   three ODEs (size, mean tolerance, heterogeneity) that fit in any NLME tool; the
#   continuum says why these three and how the drug enters each.
# - When it is not (bounded levels, a drift that depends on the level, a fitness with two
#   peaks), the same few lines still solve, with the books kept and a grid that can be
#   refined to show the answer does not depend on it.
#
# **Limits.** The level is one number per cell; Chisholm et al. use two (survival and
# proliferation potential), which is written the same way, as a field over two axes with a
# drift along one of them. The drift speed
# here is illustrative: separating drift from selection needs more than tumour size (e.g.
# a marker of the tolerant state measured over time, or regrowth after a drug holiday).
