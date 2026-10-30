# %% [markdown]
# # Tutorial 3: myelosuppression as a maturation axis
#
# The semi-mechanistic myelosuppression model of Friberg et al. (J Clin Oncol 2002) is the
# standard model of chemotherapy-induced neutropenia. Neutrophil precursors proliferate in a
# compartment `Prol`, mature through three transit compartments, and appear in the blood as
# circulating neutrophils `Circ`. A feedback term, (Circ0/Circ)^gamma, speeds up
# proliferation when neutrophils are low (the effect of G-CSF), and the drug kills
# proliferating cells.
#
# The three transit compartments are a stand-in for *maturation*, a process every cell goes
# through in about the same time. As in Tutorial 1, TenQSP writes them as cells spread along an
# axis, here the fraction of maturation completed (0 = just left `Prol`, 1 = released into the
# blood). The feedback loop stays as it was: `Prol` and `Circ` are ordinary compartments, and
# the axis sits between them.
#
# What you will see:
#
# 1. the published model written out by hand with scipy;
# 2. the same model in TenQSP, with maturation as an axis: the same numbers to 9 digits;
# 3. without drug, the baseline is a steady state;
# 4. the mean transit time is MTT whatever the axis is cut into, and only the spread changes;
# 5. what a less spread maturation does to the neutrophil nadir.
#
# **Values** are illustrative. MTT = 100 h and gamma = 0.17 lie within the ranges Friberg et
# al. report across drugs (MTT about 90-135 h, gamma about 0.16-0.2); the baseline (5 x 10^9
# cells/L), the drug's pharmacokinetics and its slope are made up for the example.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import tenqsp as tq

t = np.linspace(0, 600, 601)  # hours
P = {"MTT": 100.0, "gamma": 0.17, "circ0": 5.0, "slope": 0.5, "CL": 1.0, "V": 50.0}
DOSE = 90.0  # mg, an intravenous bolus: 1.8 mg/L at t = 0

# %% [markdown]
# ## 1. The published model
#
# With n = 3 transit compartments, k_tr = (n + 1)/MTT, and (as in the paper) the
# proliferation rate and the neutrophils' elimination rate both equal to k_tr:
#
#     dProl/dt  = k_tr Prol (1 - E_drug) (Circ0/Circ)^gamma - k_tr Prol
#     dT1/dt    = k_tr Prol - k_tr T1
#     dT2/dt    = k_tr T1 - k_tr T2
#     dT3/dt    = k_tr T2 - k_tr T3
#     dCirc/dt  = k_tr T3 - k_tr Circ
#
# The drug is a one-compartment intravenous bolus, C = Dose/V exp(-CL/V t), with a linear
# effect E_drug = slope x C. Every compartment starts at Circ0, the baseline. Written out by
# hand:


# %%
def friberg_by_hand(MTT, gamma, circ0, slope, CL, V):
    ktr = 4 / MTT

    def rhs(_, x):
        prol, t1, t2, t3, circ, ac = x
        e_drug = slope * ac / V
        return [
            ktr * prol * (1 - e_drug) * (circ0 / circ) ** gamma - ktr * prol,
            ktr * (prol - t1),
            ktr * (t1 - t2),
            ktr * (t2 - t3),
            ktr * (t3 - circ),
            -CL / V * ac,
        ]

    x0 = [circ0, circ0, circ0, circ0, circ0, DOSE]
    return solve_ivp(rhs, (0, 600), x0, t_eval=t, method="Radau", rtol=1e-12, atol=1e-12).y


by_hand = friberg_by_hand(**P)
print(f"nadir {by_hand[4].min():.3f} x 10^9/L at {t[np.argmin(by_hand[4])]:.0f} h")

# %% [markdown]
# ## 2. The same model with maturation as an axis
#
# The cells maturing are one quantity, `Transit(z)`, along an axis `z` from 0 to 1: the
# fraction of maturation done. They move along it at speed 1/T_mat, so maturation takes T_mat
# on average. In Friberg's model a cell spends on average 1/k_tr in `Prol` before it leaves,
# and 3/k_tr in transit, so T_mat = 3/k_tr = 3/4 MTT.
#
# The cells leaving `Prol` (`ktr*Prol` per hour) enter the axis at `z = 0`. As in a paper,
# that is a boundary condition under the equation of `Transit`: `at z=0: Transit/T_mat =
# ktr*Prol`, where the left side is the flux along `z` (the speed times the density) and the
# same term `ktr*Prol` is subtracted in the equation of `Prol`. The cells reaching `z = 1`
# flow out (`at z=1: outflow`) into `Circ`, which gains the flux there, `Transit(z=1)/T_mat`
# (the speed times the density at that end). `m.explain()` matches each of these terms with
# its partner. With the axis cut into 3 **cells** (the pieces the axis is divided
# into for the computation) and TenQSP's default **scheme** (its rule for moving material from
# one cell to the next), `erlang`, each cell passes its content on at a rate proportional to
# what it holds: each cell is exactly one transit compartment, as in Tutorial 1.
#
# The starting value of `Transit` is a density (cells per unit of `z`). In the steady state
# the cells flow along the axis at k_tr Circ0 per hour, so the density is that flux divided by
# the speed, k_tr Circ0 T_mat (= 3 Circ0); each of the three cells, 1/3 wide, then holds Circ0,
# as each transit compartment does.


# %%
def friberg_text(n=3, scheme="erlang"):
    sharp = "    using sharp" if scheme == "sharp" else ""
    return f"""
model friberg
axis z in [0, 1] n={n} kind=one_way      # fraction of maturation completed
param MTT=100, gamma=0.17, circ0=5, slope=0.5, CL=1, V=50

ktr = 4/MTT
T_mat = 3/ktr                           # time spent maturing (3/4 of MTT)
Conc = Ac/V
E_drug = slope*Conc

field Transit(z)                        # maturing cells, per unit of z

d_t(Prol) = ktr*Prol*(1 - E_drug)*(circ0/Circ)^gamma - ktr*Prol
d_t(Transit) = -d_z(Transit/T_mat){sharp}
    at z=0: Transit/T_mat = ktr*Prol    # cells leaving Prol start maturing
    at z=1: outflow                     # mature cells leave into the blood
d_t(Circ) = Transit(z=1)/T_mat - ktr*Circ
d_t(Ac) = -CL/V*Ac

Prol(0) = circ0
Transit(0) = ktr*circ0*T_mat
Circ(0) = circ0
dose Ac at t=0: {DOSE}
observable ANC = Circ
"""


friberg = tq.Model.from_text(friberg_text())
sol = friberg.solve((0, 600), times=t, rtol=1e-12, atol=1e-12)

diff_circ = np.max(np.abs(np.asarray(sol["Circ"]) - by_hand[4]))
diff_prol = np.max(np.abs(np.asarray(sol["Prol"]) - by_hand[0]))
# each cell is 1/3 wide: its amount (the transit compartment) is density x 1/3
diff_transit = np.max(np.abs(np.asarray(sol["Transit"]) / 3 - by_hand[1:4].T))
print(f"largest differences: Circ {diff_circ:.1e}, Prol {diff_prol:.1e}, T1-T3 {diff_transit:.1e}")
assert max(diff_circ, diff_prol, diff_transit) < 1e-9 * P["circ0"]
assert np.min(sol["Circ"]) < 0.5 * P["circ0"]  # a real nadir, not a trivial run

# %% [markdown]
# ## 3. Without drug, the baseline is a steady state
#
# With no drug effect (slope = 0), every compartment at Circ0 must stay there: proliferation
# balances the cells leaving `Prol`, the feedback is 1, and each transit compartment passes on
# what it receives. Two checks: the rates of change at the starting values, before the dose,
# are zero (`m.compile()` gives the model's right-hand side, and `initial_state()` the starting
# values without the doses), and a long run with slope = 0 stays at Circ0.

# %%
compiled = friberg.compile()
y0 = np.asarray(compiled.initial_state())  # the dose is not in it: no drug yet
residual = np.max(np.abs(np.asarray(compiled.rhs(0.0, y0))))
flat = friberg.solve((0, 2000), times=[500, 2000], params={"slope": 0.0}, rtol=1e-10, atol=1e-12)
drift = np.max(np.abs(np.asarray(flat["Circ"]) - P["circ0"]))
print(f"rates of change at baseline: {residual:.1e}; Circ after 2000 h differs by {drift:.1e}")
assert residual < 1e-12
assert drift < 1e-9

# %% [markdown]
# ## 4. The mean transit time stays MTT; only the spread changes
#
# Three transit compartments make the maturation time Erlang-distributed with a coefficient of
# variation (CV) of 1/sqrt(3) = 58 %: some cells mature in a third of the mean time, others
# take twice as long. That spread is part of the model, and with an axis it becomes a choice
# separate from the mean:
#
# - more `erlang` cells: the same mean, a smaller CV (1/sqrt(N) for N cells);
# - the `sharp` scheme (`using sharp`): the axis carries cells along without spreading them,
#   so refining it approaches a fixed maturation time.
#
# To check the mean, follow a cohort of cells. Put one unit in `Prol` with proliferation
# switched off, let it mature and count, in `W`, the amount not yet in the blood integrated
# over time: at the end `W` is the mean time from `Prol` to the blood, which Friberg et al.
# call MTT. It must be MTT for every axis.

# %%
COHORT = """
model cohort
axis z in [0, 1] n={n} kind=one_way
param MTT=100
ktr = 4/MTT
T_mat = 3/ktr
field Transit(z)
d_t(Prol) = -ktr*Prol
d_t(Transit) = -d_z(Transit/T_mat){sharp}
    at z=0: Transit/T_mat = ktr*Prol
    at z=1: outflow
d_t(Circ) = Transit(z=1)/T_mat
d_t(W) = Prol + int_z(Transit)          # not yet in the blood, integrated over time
Prol(0) = 1
"""
sharp_errors = []
for n, scheme in [(3, "erlang"), (30, "erlang"), (100, "sharp"), (200, "sharp"), (400, "sharp")]:
    text = COHORT.format(n=n, sharp="    using sharp" if scheme == "sharp" else "")
    c = tq.Model.from_text(text).solve((0, 3000), times=[3000], rtol=1e-10, atol=1e-13)
    mean_time = float(np.asarray(c["W"])[-1])
    print(f"{n:4d} cells, {scheme:6s}: mean time from Prol to blood {mean_time:.4f} h")
    assert abs(float(np.asarray(c["Circ"])[-1]) - 1) < 1e-9  # the whole cohort arrived
    if scheme == "erlang":
        assert abs(mean_time - P["MTT"]) < 1e-6  # exact: each cell is a transit compartment
    else:
        sharp_errors.append(abs(mean_time - P["MTT"]))
# sharp: within 0.1 % on 100 cells, and closer each time the cells are doubled
assert sharp_errors[0] < 1e-3 * P["MTT"]
assert sharp_errors[0] > 2 * sharp_errors[1] > 4 * sharp_errors[2]

# %% [markdown]
# With `erlang` cells the mean is exact on every grid (each cell is a transit compartment of
# mean time T_mat/N). The `sharp` scheme gets the mean slightly short (by 0.07 h of 100 on 100
# cells), and the error more than halves each time the cells are doubled.

# %% [markdown]
# ## 5. What the spread does to the nadir
#
# The same drug, the same MTT and gamma, and three maturation axes. Each one starts at a
# steady state (the starting density is the same, k_tr Circ0 T_mat, whatever the cells), so
# all three are checked for that before the drug is given.
#
# With less spread, the cells killed in `Prol` are missed in the blood all at once instead of
# being smeared over a wide range of maturation times: the nadir is deeper and comes earlier.
# Here the published 3-compartment version places it about 10 h later and 0.25 x 10^9/L
# higher than a fixed maturation time would. Which is right is a question for the data (a
# well-sampled nadir can tell them apart); the axis lets you ask it without changing MTT.
#
# The right panel follows the maturing cells along the axis (sharp scheme): the gap made by
# the drug in `Prol` travels along maturation and reaches the blood about T_mat = 75 h later.

# %%
variants = [(3, "erlang"), (30, "erlang"), (200, "sharp")]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
nadirs = {}
for n, scheme in variants:
    model = tq.Model.from_text(friberg_text(n, scheme))
    cm = model.compile()
    assert np.max(np.abs(np.asarray(cm.rhs(0.0, np.asarray(cm.initial_state()))))) < 1e-12
    s = model.solve((0, 600), times=t, rtol=1e-8, atol=1e-10)
    anc = np.asarray(s["ANC"])
    nadirs[(n, scheme)] = (anc.min(), t[np.argmin(anc)])
    cv = "0" if scheme == "sharp" else f"{1 / np.sqrt(n):.2f}"
    ax[0].plot(t, anc, label=f"{scheme}, {n} cells (maturation CV {cv})")
    print(f"{scheme:6s} {n:4d} cells: nadir {anc.min():.2f} x 10^9/L at {t[np.argmin(anc)]:.0f} h")
    if scheme == "sharp":
        tq.plot.snapshot(s, "Transit", axis="z", times=[0, 50, 100, 150, 200], ax=ax[1])
ax[0].axhline(P["circ0"], color="grey", ls=":", lw=1)
ax[0].set_xlabel("time (h)")
ax[0].set_ylabel("circulating neutrophils (10^9/L)")
ax[0].legend(fontsize="small")
ax[0].set_title("same MTT, different spread of maturation times")
ax[1].set_xlabel("fraction of maturation completed")
ax[1].set_ylabel("maturing cells (10^9/L per unit of z)")
ax[1].set_title("maturing cells, sharp scheme")
fig.tight_layout()

# the published model agrees with section 1, and less spread gives a deeper, earlier nadir
assert abs(nadirs[(3, "erlang")][0] - by_hand[4].min()) < 1e-6
depth = [nadirs[v][0] for v in variants]
timing = [nadirs[v][1] for v in variants]
assert depth[0] > depth[1] > depth[2] and timing[0] > timing[1] >= timing[2]

# %% [markdown]
# ## What this buys you
#
# - Friberg's model is unchanged where it matters: the feedback and the drug act on
#   compartments, as in the paper, and with 3 `erlang` cells the numbers are the published
#   model's.
# - The mean maturation time (MTT) and its spread are separate: the spread is set by the
#   number of cells and the scheme, not by adding transit compartments and re-estimating
#   k_tr.
# - The axis is a real coordinate, so a drug or a growth factor can act on cells at a given
#   stage of maturation (a rate that depends on `z`), which a chain of compartments can only
#   approximate.
#
# **Limits.** The drug effect here is linear and acts only on `Prol`; an Emax effect or a
# second dosing cycle are one line each (`E_drug = Emax*Conc/(EC50 + Conc)`,
# `dose Ac at t=0 to 504 every 504: 90`). The sharp scheme converges to a fixed maturation
# time as the cells are refined; with a few cells it still has some spread of its own.
