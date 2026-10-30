# %% [markdown]
# # Tutorial 8: an age-structured population (McKendrick-von Foerster)
#
# McKendrick (1926, Proc Edinb Math Soc) wrote a population as a density n(a, t) over age:
# individuals age at speed 1, die at an age-dependent rate mu(a), and are born at age 0 at a
# rate that is an integral over the parents' ages,
#
#     dn/dt + dn/da = -mu(a) n,        n(0, t) = int beta(a) n(a, t) da.
#
# (von Foerster rederived it in 1959 for cell populations.) Pharmacometricians meet the
# same structure in cell populations whose division or death depends on their age, and in
# lifespan models of red blood cells (Krzyzanski, Ramakrishnan & Jusko 1999,
# J Pharmacokinet Biopharm), where every cell dies at the end of its lifespan. In TenQSP it
# is one field on an age axis, with the births written as a boundary condition at age 0 that
# reads the field itself.
#
# Values are illustrative.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import tenqsp as tq

TIGHT = {"rtol": 1e-10, "atol": 1e-12}

# %% [markdown]
# ## 1. The model
#
# `at a=0: n = int_a(beta*n)`, written under the equation as in a paper, is the birth
# condition n(t, 0) = int beta(a) n(t, a) da: the individuals born per unit time are the
# integral over ages of the birth rate times the density (everyone ages at speed 1, so the
# number crossing age 0 per unit time is n there). Those who reach the end of the axis
# (age 6) leave it (`at a=6: outflow`, counted in the ledger); nobody that old gives birth
# here, so the end of the axis does not change the births.
# `using sharp` moves the population along the axis without spreading it (Tutorial 1).

# %%
POP = """
model mckendrick
axis a in [0, 6] n=240 kind=one_way              # age
param mu0=0.3, mu1=0, b0=0, a1=1, a2=2

mu(a)   = mu0*(1 + mu1*a)                         # death rate
beta(a) = b0 if a > a1 and a < a2 else 0          # birth rate: only between ages a1 and a2

field n(a)
d_t(n) = -d_a(n) - mu*n                           using sharp
    at a=0: n = int_a(beta*n)                    # births
    at a=6: outflow                              # what reaches age 6 leaves

n(0) = exp(-((a - 1.5)/0.3)^2)
observable N = int_a(n)
"""
pop = tq.Model.from_text(POP)
faces = np.linspace(0, 6, 241)
mid, width = 0.5 * (faces[1:] + faces[:-1]), np.diff(faces)

# %% [markdown]
# ## 2. No births: deaths only, and the characteristics
#
# Without births and with a constant death rate, the total falls as N0 e^(-mu t), whatever
# the ages. With a death rate that rises with age, mu(a) = mu0 (1 + mu1 a), each cohort
# moves along a **characteristic** a = a0 + t and loses members as it goes: with survival
# S(a) = exp(-int_0^a mu), the density is n(a, t) = n0(a - t) S(a) / S(a - t). The figure
# compares the profiles with this formula, averaged over each cell as TenQSP stores them.

# %%
t = np.linspace(0, 3, 7)
flat = pop.solve((0, 3), times=t, **TIGHT)
decay = np.asarray(flat["N"]) / flat["N"][0]
gap = np.max(np.abs(decay - np.exp(-0.3 * t)))
print(f"no births, mu constant: largest difference from exp(-mu t) {gap:.1e}")
assert gap < 1e-9


def S(a, mu0=0.3, mu1=1.0):
    return np.exp(-mu0 * (a + mu1 * a * a / 2))


def characteristic(a, t_):
    return np.exp(-((((a - t_) - 1.5) / 0.3) ** 2)) * S(a) / S(a - t_)


def cell_average(f, n=240):  # the exact density averaged over each of n cells
    faces_n = np.linspace(0, 6, n + 1)
    x, w = np.polynomial.legendre.leggauss(8)
    pts = 0.5 * (faces_n[1:] + faces_n[:-1])[:, None] + 0.5 * np.diff(faces_n)[:, None] * x
    return (f(pts) * w).sum(axis=1) / 2


ageing = pop.solve((0, 3), times=[1.0, 2.0, 3.0], params={"mu1": 1.0}, **TIGHT)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(mid, cell_average(lambda a: characteristic(a, 0.0)), color="grey", label="t = 0")
for k, t_ in enumerate([1.0, 2.0, 3.0]):
    ax.plot(mid, ageing["n"][k], label=f"t = {t_:g}")
    ax.plot(mid, cell_average(lambda a, t_=t_: characteristic(a, t_)), "k:", lw=1)
ax.set_xlabel("age a")
ax.set_ylabel("density n(a, t)")
ax.set_title("deaths rising with age: the cohort moves and shrinks (dotted: exact)")
ax.legend(fontsize="small")
fig.tight_layout()

# %% [markdown]
# The curves are hard to tell apart. The difference is the grid's: the `sharp` scheme is
# second order, so halving the cells should divide the difference by about 4. Measured at
# t = 3, as a fraction of the population (`cells=` reads the same model with another
# number of cells):

# %%
errors = []
for n in (120, 240, 480):
    fine = tq.Model.from_text(POP, cells={"a": n})
    end = fine.solve((0, 3), times=[3.0], params={"mu1": 1.0}, ledger=False, **TIGHT)
    exact = cell_average(lambda a: characteristic(a, 3.0), n)
    errors.append(np.sum(np.abs(end["n"][-1] - exact)) / np.sum(exact))
    print(f"{n} cells: L1 difference from the characteristics {errors[-1]:.1e}")
ratios = np.array(errors[:-1]) / np.array(errors[1:])
print("ratios", np.round(ratios, 2))
assert np.all(ratios > 3) and errors[-1] < 1e-2

# %% [markdown]
# ## 3. Births: the growth rate and the stable age distribution
#
# With births at rate b0 between ages a1 and a2 and a constant death rate, the population
# forgets its starting ages and settles into a **stable age distribution** that keeps its
# shape while the total grows (or shrinks) at a rate r. Both come from the
# **Euler-Lotka equation**: the offspring a newborn will have over its life, each
# discounted by e^(-r a) for the growth of the population in the meantime, add up to one,
#
#     int beta(a) S(a) e^(-r a) da = 1,
#
# and the stable distribution is proportional to e^(-r a) S(a). Here
# int_a1^a2 b0 e^(-(mu + r) a) da = 1 is solved for r by root finding.

# %%
b0, mu, a1, a2 = 2.5, 0.3, 1.0, 2.0
r = brentq(
    lambda r_: b0 * (np.exp(-(mu + r_) * a1) - np.exp(-(mu + r_) * a2)) / (mu + r_) - 1,
    -mu + 1e-9,
    20.0,
)
births = {"b0": b0, "mu0": mu, "a1": a1, "a2": a2}
grown = pop.solve((0, 60), times=[58.0, 60.0], params=births, ledger=False, **TIGHT)
r_sim = float(np.log(grown["N"][1] / grown["N"][0]) / 2)
print(f"growth rate: simulated {r_sim:.6f}, Euler-Lotka root {r:.6f}")
assert abs(r_sim / r - 1) < 1e-5

profile = grown["n"][-1] / np.sum(grown["n"][-1] * width)
stable = -np.diff(np.exp(-(mu + r) * faces)) / (mu + r) / width  # e^(-(mu + r) a), cell averages
stable /= np.sum(stable * width)
l1 = np.sum(np.abs(profile - stable) * width)
print(f"age profile vs the stable age distribution: L1 difference {l1:.1e}")
assert l1 < 1e-4

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(mid, profile, label="TenQSP at t = 60")
ax.plot(mid, stable, "k:", label="e^(-ra) S(a), r from Euler-Lotka")
ax.axvspan(a1, a2, color="grey", alpha=0.15, label="fertile ages")
ax.set_xlabel("age a")
ax.set_ylabel("share of the population per unit age")
ax.legend(fontsize="small")
fig.tight_layout()

# %% [markdown]
# ## 4. Red blood cells: a lifespan instead of a death rate
#
# In the lifespan model of erythropoiesis, cells are produced at a rate kin (driven, for
# instance, by erythropoietin), never die by chance, and are removed when they reach their
# lifespan TR: the age axis ends at TR and the outflow there (`at a=120: outflow`) is the
# removal. The births are now a source from outside, not an integral of the field
# (`at a=0: R = kin`). Starting from no cells, the count rises linearly, kin t, until the
# first cells reach TR, and then stays at kin TR: the drug or disease effect on production
# shows up with a delay of exactly one lifespan, which is what these models are for.

# %%
RBC = tq.Model.from_text("""
model rbc_lifespan
axis a in [0, 120] n=120 units=day kind=one_way  # cell age; lifespan TR = 120 days
param kin=2                                       # production (illustrative units per day)
field R(a)
d_t(R) = -d_a(R)                                  using sharp
    at a=0: R = kin
    at a=120: outflow
observable total = int_a(R)
""")
days = np.array([30.0, 60.0, 90.0, 200.0])
rbc = RBC.solve((0, 200), times=days, **TIGHT)
expected = 2 * np.minimum(days, 120)  # kin x min(t, TR)
print("total red cells:", np.asarray(rbc["total"]).ravel(), "expected", expected)
assert np.max(np.abs(np.asarray(rbc["total"]).ravel() / expected - 1)) < 1e-8

# %% [markdown]
# ## What this buys you
#
# - Births that depend on the whole population (an integral over ages) are one line under
#   the equation;
#   the solver handles the coupling between age 0 and every other age.
# - Ageing, survival and removal at a fixed age are the same axis, so a lifespan model, a
#   cell-cycle model and a population model are written the same way.
# - Exact results (characteristics, Euler-Lotka, lifespan delay) check the numbers.
#
# **Limits.** Age is the only structure here; size or maturation that progress at their own
# speed need a velocity other than 1 (Tutorial 1), and random lifespans need a removal
# hazard rather than a fixed end.
