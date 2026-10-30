# %% [markdown]
# # Tutorial 7: HIV infection with an infection-age axis
#
# The standard model of HIV dynamics under therapy has three ODEs: target cells `T`,
# productively infected cells `I` and free virus `V` (Perelson et al. 1996, Science). It
# assumes that an infected cell produces virus from the moment it is infected, and dies at
# the same rate whatever its age. Neither is true: a newly infected cell first goes
# through an eclipse (reverse transcription, integration) before it releases virions, its
# production then rises, and its death rate may rise with age as the immune response
# recognises it.
#
# Nelson, Gilchrist, Coombs, Hyman and Perelson (2004, Math Biosci Eng) wrote the infected
# cells as a density `i(a, t)` over the **infection age** `a`, the time since the cell was
# infected, with an age-dependent death rate delta(a) and production p(a); Rong, Feng and
# Perelson (2007, SIAM J Appl Math) analysed it under combination therapy:
#
#     dT/dt = s - d T - (1 - eps_RT) k V T
#     di/dt + di/da = -delta(a) i,        i(0, t) = (1 - eps_RT) k V T
#     dV/dt = (1 - eps_PI) int p(a) i(a, t) da - c V
#
# A reverse-transcriptase inhibitor (RTI, efficacy eps_RT) blocks new infections; a
# protease inhibitor (PI, efficacy eps_PI) makes a fraction of the new virions
# non-infectious. `V` counts infectious virus only (the non-infectious virions are not
# followed).
#
# In TenQSP the infection age is an axis. New infections enter at age 0, cells age at speed
# 1 and die at delta(a) on the way, and the virus is the integral of p(a) i over ages.
#
# **Values** are illustrative, in the range of target-cell-limited models (per mL and per
# day). The production ramp is Nelson et al.'s form, p(a) = p* (1 - exp(-theta (a - a1)))
# after an eclipse a1; the rising death rate delta(a) = delta0 + delta1 a is an
# illustration.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

import tenqsp as tq

# %% [markdown]
# ## 1. The model
#
# The new infections, `(1 - eps_rt)*k*V*T` per day, are written as in a paper: subtracted in
# the equation of `T`, and entering `i` at age 0 by the boundary condition under its equation,
# `at a=0: i = (1 - eps_rt)*k*V*T` (cells age at one day per day, so the cells crossing age 0
# per day are `i` there). The age axis runs to 20 days, cut into 160 cells of 3 hours; with
# these death rates a cell survives 20 days with probability 2e-22, so what leaves at the end
# (`at a=20: outflow`) is negligible.
#
# `using sharp` makes the cells age without spreading along the axis: a cell infected now
# reaches the end of the eclipse after a1 days, not earlier. (The default scheme turns the
# axis into a chain of transit compartments, Tutorial 1, which lets some cells reach
# production early.)
#
# `p0` adds an age-independent production, so that the same model also holds the standard
# model (section 2).

# %%
HIV = """
model hiv_age
axis a in [0, 20] n=160 units=day kind=one_way    # infection age
param s=1e4, d=0.01, k=2.4e-8, c=23                # target cells, infectivity, clearance
param p0=0, pstar=5000, theta=1, a1=0.5            # virus production (copies/cell/day)
param delta0=0.5, delta1=0.2                       # death of infected cells (1/day)
param eps_rt=0, eps_pi=0                           # drug efficacies

delta(a) = delta0 + delta1*a
p(a) = p0 + pstar*max(0, 1 - exp(-theta*(a - a1)))

field i(a) units=cells/mL/day
compartment T units=cells/mL
compartment V units=copies/mL

d_t(T) = s - d*T - (1 - eps_rt)*k*V*T
d_t(i) = -d_a(i) - delta*i                          using sharp
    at a=0: i = (1 - eps_rt)*k*V*T                 # new infections enter at age 0
    at a=20: outflow                               # cells older than 20 days leave
d_t(V) = (1 - eps_pi)*int_a(p*i) - c*V

T(0) = 1e6                                         # s/d: before infection
V(0) = 1e-3
observable I = int_a(i)
"""
hiv = tq.Model.from_text(HIV)
print(hiv.summary())

S0, D0, K, C = 1e4, 0.01, 2.4e-8, 23.0
T0 = S0 / D0
# the virus starts at 1e-3 copies/mL and infected-cell densities are tiny early on, so the
# absolute tolerance must sit far below them
TOL = {"rtol": 1e-10, "atol": 1e-16}

# %% [markdown]
# ## 2. Constant rates: the standard three-ODE model
#
# With delta(a) = delta and p(a) = p for every age, the total I = int i da obeys
# dI/dt = (1 - eps_RT) k V T - delta I, and the virus is produced at p I: the age axis
# carries no information and the model is exactly the standard (T, I, V) model, whatever
# the number of cells. Here it is solved both ways, with both drugs at partial efficacy.
#
# One care: the axis must be long enough that no cell reaches its end. With delta = 1 per
# day a cell survives 20 days with probability 2e-9; that sounds negligible, but after the
# first peak the infected cells fall about 100,000-fold, and the few survivors of the peak
# that leave at 20 days are then a visible part of what remains. The ledger shows how many
# left (`sol.ledger.moved`). So this check uses a 60-day axis (with coarse cells: the totals
# do not depend on them here).

# %%
const = {"p0": 5000.0, "pstar": 0.0, "delta0": 1.0, "delta1": 0.0, "eps_rt": 0.3, "eps_pi": 0.2}
t = np.linspace(0, 60, 121)
text_60 = HIV.replace("in [0, 20] n=160", "in [0, 60] n=120").replace("a=20:", "a=60:")
hiv_60 = tq.Model.from_text(text_60)
sol = hiv_60.solve((0, 60), times=t, params=const, **TOL)
left = sol.ledger.moved["i"]["boundary a=upper: outflow (absorbed)"][-1]
print(f"cells that reached the end of the 60-day axis: {-left:.1e} per mL")


def three_odes(_, x, eps_rt=0.3, eps_pi=0.2, delta=1.0, p=5000.0):
    T, I, V = x
    new = (1 - eps_rt) * K * V * T
    return [S0 - D0 * T - new, new - delta * I, (1 - eps_pi) * p * I - C * V]


ref = solve_ivp(
    three_odes, (0, 60), [T0, 0, 1e-3], t_eval=t, method="Radau", rtol=1e-12, atol=1e-18
)
for j, name in enumerate(["T", "I", "V"]):
    got, want = np.asarray(sol[name]), ref.y[j]
    rel = np.max(np.abs(got - want) / (np.abs(want) + 1e-12 * np.abs(want).max()))
    print(f"{name}: largest relative difference from the three ODEs {rel:.1e}")
    assert rel < 1e-6, name
assert np.max(sol["V"]) > 1e5  # the infection takes off

# %% [markdown]
# ## 3. R0 by quadrature, and the set point it predicts
#
# A cell infected at age 0 survives to age a with probability S(a) = exp(-int_0^a delta),
# so it produces N = int p(a) S(a) da virions over its life (the burst size). Each virion
# infects k T0 / c cells before it is cleared, so
#
#     R0 = (1 - eps_RT)(1 - eps_PI) (k T0 / c) int p(a) S(a) da.
#
# TenQSP computes the integral on its grid, as an observable (`int_a`); a solve over a
# negligible time reads it. scipy's `quad` gives the reference.
#
# R0 also fixes the chronic set point: at steady state every infected cohort follows
# S(a), and the infection must exactly replace itself, so T* = T0 / R0, whatever the shapes
# of delta(a) and p(a). The model is run to its steady state with `pre_run`.

# %%
r0_model = tq.Model.from_text(
    HIV
    + """
observable R0 = (1 - eps_rt)*(1 - eps_pi)*k*(s/d)/c*int_a(p*exp(-(delta0*a + delta1*a^2/2)))
"""
)


def survival(a, delta0=0.5, delta1=0.2):
    return np.exp(-(delta0 * a + delta1 * a * a / 2))


def production(a, pstar=5000.0, theta=1.0, a1=0.5):
    return pstar * max(0.0, 1 - np.exp(-theta * (a - a1)))


burst = quad(lambda a: production(a) * survival(a), 0.5, 40)[0]
for eps_rt, eps_pi in ((0.0, 0.0), (0.3, 0.2)):
    exact = (1 - eps_rt) * (1 - eps_pi) * K * T0 / C * burst
    drugs = {"eps_rt": eps_rt, "eps_pi": eps_pi}
    grid = float(r0_model.solve((0, 1e-6), times=[1e-6], params=drugs, ledger=False)["R0"][-1])
    print(f"eps_RT={eps_rt}, eps_PI={eps_pi}: R0 {grid:.5f} on the grid, {exact:.5f} by quad")
    assert abs(grid / exact - 1) < 2e-3

R0 = K * T0 / C * burst
sim = hiv.simulator(outputs=["T", "V", "I", "i"], **TOL)
set_point = sim.pre_run(tq.Scenario(), until="steady")
T_star = float(set_point.state["T"])
print(f"set point after {set_point.t:g} days: T* = {T_star:.1f}; T0/R0 = {T0 / R0:.1f}")
assert set_point.reached
assert abs(T_star / (T0 / R0) - 1) < 2e-3

# %% [markdown]
# Both agree to about 1e-3, the accuracy of the grid's quadrature: the production ramp
# has a kink at the end of the eclipse, and halving the cells divides the difference by
# about 4.
#
# The age profile at the set point is the survival curve itself, i*(a) = b* S(a) with
# b* = k T* V* the infection rate. The figure shows it with the production p(a): most
# virions come from cells 1-3 days old.

# %%
a_mid = (np.arange(160) + 0.5) * 20 / 160
V_star = float(set_point.state["V"])
b_star = K * T_star * V_star
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(a_mid, set_point.state["i"], label="set point i*(a) (TenQSP)")
ax.plot(a_mid, b_star * survival(a_mid), "k:", label="b* S(a)")
ax.set_xlim(0, 8)
ax.set_xlabel("infection age a (days)")
ax.set_ylabel("infected cells (cells/mL per day of age)")
twin = ax.twinx()
twin.plot(a_mid, [production(x) for x in a_mid], color="tab:red", alpha=0.6)
twin.set_ylabel("production p(a) (copies/cell/day)", color="tab:red")
ax.legend(fontsize="small")
fig.tight_layout()
assert np.max(np.abs(set_point.state["i"] - b_star * survival(a_mid))) < 1e-3 * b_star

# %% [markdown]
# ## 4. The early growth rate is the root of the renewal equation
#
# At the start of infection T stays at T0, so the system is linear: a cohort infected at
# time t - a is still i(a, t) = b(t - a) S(a), and everything grows as e^(r t). Putting
# that into the virus equation gives the characteristic (renewal) equation for r:
#
#     r + c = (1 - eps_RT)(1 - eps_PI) k T0 int p(a) S(a) e^(-r a) da.
#
# The eclipse slows the growth: a virion's offspring appear only a1 days later at the
# earliest. The slope of log V between days 15 and 20 is compared with the root.

# %%


def renewal(r):
    return K * T0 * quad(lambda a: production(a) * survival(a) * np.exp(-r * a), 0.5, 40)[0] - (
        r + C
    )


r = brentq(renewal, -0.4, 20.0)
early = hiv.solve((0, 20), times=[15.0, 20.0], ledger=False, **TOL)
slope = float(np.log(early["V"][1] / early["V"][0]) / 5)
print(f"growth rate: simulated {slope:.5f} per day, root of the renewal equation {r:.5f}")
assert early["T"][-1] > (1 - 1e-4) * T0  # still the linear phase
assert abs(slope / r - 1) < 3e-3
print(f"doubling time {np.log(2) / r:.2f} days")

# %% [markdown]
# The difference, 1.7e-3, is the grid's: with 320 cells it is 4.5e-4 (a quarter, as for R0).

# %% [markdown]
# ## 5. Therapy started at the set point
#
# Starting from the set point (the pre-run's end state), each drug alone and both at 90 %,
# and the two perfect drugs, which have exact answers:
#
# - a **perfect PI** (eps_PI = 1) stops infectious virus production at once, so
#   V(t) = V* e^(-c t);
# - a **perfect RTI** (eps_RT = 1) stops new infections, but the cells already infected
#   carry on along their ages: i(a, t) = b* S(a) for a > t and 0 below. The virus then
#   follows dV/dt = b* int_t p(a) S(a) da - c V, with the "shoulder" seen in patients:
#   virus falls only once the infected cells have aged out of production.

# %%
days = np.linspace(0, 10, 201)
start = tq.Scenario(times=days, initial=set_point)
runs = {
    "RTI 90 %": {"eps_rt": 0.9},
    "PI 90 %": {"eps_pi": 0.9},
    "RTI + PI 90 %": {"eps_rt": 0.9, "eps_pi": 0.9},
    "perfect PI": {"eps_pi": 1.0},
    "perfect RTI": {"eps_rt": 1.0},
}
viral = {name: np.asarray(sim(start, params=p)["V"]).ravel() for name, p in runs.items()}

pi_exact = V_star * np.exp(-C * days)
assert np.max(np.abs(viral["perfect PI"] - pi_exact)) < 1e-6 * V_star

remaining = [
    b_star * quad(lambda a: production(a) * survival(a), max(t_, 0.5), 40)[0] for t_ in days
]


def rti_virus(t_, v):
    return np.interp(t_, days, remaining) - C * v


rti_exact = solve_ivp(
    rti_virus, (0, 10), [V_star], t_eval=days, rtol=1e-10, atol=1e-6, max_step=0.05
).y[0]
rel = np.max(np.abs(viral["perfect RTI"] - rti_exact)) / V_star  # largest near day 1
print(f"perfect RTI: largest difference from the characteristics {rel:.1e} of V*")
assert rel < 1.5e-2

fig, ax = plt.subplots(figsize=(7, 4.5))
for name, v in viral.items():
    ax.semilogy(days, v / V_star, label=name, ls="--" if "perfect" in name else "-")
ax.semilogy(days, rti_exact / V_star, "k:", label="perfect RTI, characteristics")
ax.set_ylim(1e-6, 2)
ax.set_xlabel("days since the start of therapy")
ax.set_ylabel("V / V* (infectious virus)")
ax.legend(fontsize="small")
fig.tight_layout()

# %% [markdown]
# The perfect-RTI run differs from the characteristics by about 1 % of V* around day 1, and
# by less later. The edge between the cells infected before therapy and the empty ages
# below them is a jump that moves along the age axis; the grid smears it over a few cells.
# The difference halves when the cells are halved (2.1 %, 0.95 %, 0.39 % with 80, 160 and
# 320 cells).
#
# The RTI curves stay flat for about a day: the cells infected just before therapy still go
# through their eclipse and production. The standard three-ODE model has no such delay
# unless one is added by hand (Herz et al. 1996, PNAS, added a fixed delay; the age axis
# gives it from the biology).
#
# ## What this buys you
#
# - Eclipse, production ramp and age-dependent death are written as functions of age, not
#   as extra compartments whose number is another modelling choice.
# - The standard model is the special case of constant rates, recovered exactly.
# - R0, the set point and the growth rate come from the same functions p(a) and delta(a) by
#   one integral, and the simulation agrees with each.
#
# **Limits of this version.** Drug efficacies are constants (a PK model driving eps(t)
# would be a few more lines); non-infectious virus, latently infected cells and the
# alternative RTI action of Rong et al. (cells reverting before reverse transcription) are
# not included.
