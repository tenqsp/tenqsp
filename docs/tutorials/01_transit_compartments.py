# %% [markdown]
# # Tutorial 1: from transit compartments to an age axis
#
# Pharmacometricians describe a delay in absorption with **transit compartments** (Savic et
# al. 2007): the dose passes through a chain of compartments before it reaches the absorption
# compartment. This tutorial shows that such a chain *is* a structured model: one quantity,
# the drug in transit, spread along an axis (time since the dose, as a fraction of the mean
# transit time). TenQSP writes it that way, and you then choose how the delay behaves instead
# of choosing a number of compartments.
#
# What you will see:
#
# 1. the transit model you know, written in TenQSP's notation;
# 2. the same model as a field on an axis, giving the same numbers to 9 digits;
# 3. what the number of compartments really controls (the spread of the delay), and how to
#    get a sharp delay instead;
# 4. Savic's closed-form input as a check.
#
# Every section ends with a check against a known result, so you can see each step is right.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammainc

import tenqsp as tq

t = np.linspace(0, 12, 121)

# %% [markdown]
# ## 1. The transit model you know
#
# Three transit compartments between the dose and the absorption compartment `Aa`, each
# emptying at k_tr = (n + 1)/MTT with n = 3, then first-order absorption into the central
# compartment. Written as in NONMEM's `$DES` or a paper: each movement of drug from one place
# to another is the same term, subtracted where it leaves and added where it arrives
# (`ktr*A0` leaves `A0` and enters `A1`). `m.explain()` matches these terms across the
# equations ("Terms shared between equations", with whether the amounts balance) and points
# out a term that reads a place but is missing from that place's own equation (`ka*Aa` added
# to `Ac` but not taken from `Aa`), so a typo that loses or creates drug is visible.

# %%
chain = tq.Model.from_text("""
model transit_chain
param MTT=2, ka=1.2, CL=3, V=10
ktr = 4/MTT
d_t(A0) = -ktr*A0
d_t(A1) = ktr*A0 - ktr*A1
d_t(A2) = ktr*A1 - ktr*A2
d_t(A3) = ktr*A2 - ktr*A3
d_t(Aa) = ktr*A3 - ka*Aa
d_t(Ac) = ka*Aa - CL/V*Ac
dose A0 at t=0: 100
observable conc = Ac/V
""")
sol_chain = chain.solve((0, 12), times=t, rtol=1e-10, atol=1e-12)
print(
    "peak concentration",
    float(np.max(sol_chain["conc"])),
    "at t =",
    t[np.argmax(sol_chain["conc"])],
)

# %% [markdown]
# ## 2. The same model as a field on an axis
#
# Now the drug in transit is one quantity, `u(z)`, along an axis `z` from 0 (just dosed) to 1
# (reaching `Aa`). It moves along `z` at speed 1/MTT, so it takes MTT on average to cross:
# `d_t(u) = -d_z(u/MTT)`. Nothing enters at `z = 0` (an end that is not mentioned is sealed),
# and at `z = 1` the drug flows out (`at z=1: outflow`, written under the equation of `u`) into
# `Aa`: the amount leaving per hour is the flux there, the speed times the value at that end,
# `u(z=1)/MTT`, which is the term `Aa` gains.
#
# The axis is cut into 4 cells. TenQSP's default way of moving material between cells
# (the `erlang` scheme: each cell passes its content on to the next at a rate proportional to
# what it holds) makes each cell exactly a transit compartment: 4 cells are A0 to A3 above.
# The dose starts in the first cell (a dose statement into a cell of a continuous axis is not
# available yet, so it is the starting value).

# %%
axis_model = tq.Model.from_text("""
model transit_axis
axis z in [0, 1] n=4 kind=one_way
param MTT=2, ka=1.2, CL=3, V=10
field u(z)
d_t(u) = -d_z(u/MTT)
    at z=1: outflow
d_t(Aa) = u(z=1)/MTT - ka*Aa
d_t(Ac) = ka*Aa - CL/V*Ac
u(0) = 100*point(z, at=0.125)
observable conc = Ac/V
""")
sol_axis = axis_model.solve((0, 12), times=t, rtol=1e-10, atol=1e-12)

difference = np.max(np.abs(np.asarray(sol_axis["conc"]) - np.asarray(sol_chain["conc"])))
print(f"largest difference in concentration: {difference:.1e}")
assert difference < 1e-9

# %% [markdown]
# `m.explain()` lists, in words, what TenQSP built from the notation, including the numerical
# choices (here, the scheme and the cell size), what happens at each end of the axis, and the
# terms it matched: `u(z=1)/MTT` leaves `u` at `z = 1` and is gained by `Aa`.

# %%
print(axis_model.explain())

# %% [markdown]
# ## 3. What the number of compartments controls
#
# With the axis, the mean transit time is MTT whatever the number of cells. The number of
# cells N only sets how *spread out* the arrival is: the time to cross is Erlang
# (gamma) distributed with mean MTT and coefficient of variation 1/sqrt(N). Choosing n
# transit compartments is choosing that spread.
#
# To see the arrival itself, absorption is switched off (`ka = 0`), so `Aa` collects the drug
# as it arrives. The fraction arrived by time t is the gamma distribution function,
# P(N, N t / MTT). With the `sharp` scheme (`using sharp` in the notation), the axis carries
# the drug along without spreading it, and refining the grid approaches a pure delay of MTT.


# %%
def arrival(n, scheme="erlang"):
    text = f"""
model transit_n
axis z in [0, 1] n={n} kind=one_way
param MTT=2
field u(z)
d_t(u) = -d_z(u/MTT){"    using sharp" if scheme == "sharp" else ""}
    at z=1: outflow
d_t(Aa) = u(z=1)/MTT
u(0) = 100*point(z, at={0.5 / n})
"""
    return np.asarray(tq.Model.from_text(text).solve((0, 6), times=t[t <= 6])["Aa"])


ts = t[t <= 6]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for n in (1, 4, 16, 64):
    a = arrival(n)
    exact = 100 * gammainc(n, n * ts / 2.0)
    assert np.max(np.abs(a - exact)) < 1e-4, n
    ax[0].plot(ts, a, label=f"{n} cells (= {n - 1} transit cpts)")
for n in (16, 64, 256):
    ax[1].plot(ts, arrival(n, "sharp"), label=f"sharp, {n} cells")
ax[1].axvline(2.0, color="grey", ls=":", label="MTT")
for a_ in ax:
    a_.set_xlabel("time (h)")
    a_.set_ylabel("amount arrived in Aa")
    a_.legend(fontsize="small")
ax[0].set_title("erlang scheme: N cells = Erlang delay, CV 1/sqrt(N)")
ax[1].set_title("sharp scheme: approaches a pure delay")
fig.tight_layout()

# %% [markdown]
# Each erlang curve above matched the gamma distribution function to 1e-4 (of a dose of 100).
#
# ## 4. Savic's closed-form input
#
# Savic et al. derived the rate at which the chain delivers drug to `Aa`:
# Dose k_tr (k_tr t)^n e^(-k_tr t) / n!. With 4 cells (n = 3) and `ka = 0`, the amount
# arrived is its integral, the gamma distribution function used above; here it is checked as
# a rate, from the model's own arrival curve.

# %%
n, mtt = 3, 2.0
ktr = (n + 1) / mtt
a = arrival(n + 1)
rate = np.gradient(a, ts)
savic = 100 * ktr * (ktr * ts) ** n * np.exp(-ktr * ts) / np.prod(np.arange(1, n + 1))
inner = slice(1, -1)  # the numerical derivative is least accurate at the ends
print(f"largest difference from Savic's input: {np.max(np.abs(rate - savic)[inner]):.2f} per h")
assert np.max(np.abs(rate - savic)[inner]) < 0.01 * savic.max()

# %% [markdown]
# ## What this buys you
#
# - The delay's mean (MTT) and its spread are separate choices: the spread is set by the
#   cells and the scheme, not by adding parameters.
# - The axis is a real coordinate, so rates can depend on it. Tutorial 2 uses this to make
#   absorption depend on *where* in the gut the drug is.
# - The same notation, axis, and field carry over to maturation (Friberg's myelosuppression
#   model), cell lifespans and infection age.
