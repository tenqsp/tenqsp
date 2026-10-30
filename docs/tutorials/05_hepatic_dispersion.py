# %% [markdown]
# # Tutorial 5: the dispersion model of hepatic elimination
#
# Pharmacometricians predict hepatic clearance from intrinsic clearance with one of two
# models:
#
# - the **well-stirred** model: the liver is one well-mixed compartment, and the
#   availability (the fraction of drug escaping the liver on one pass) is
#   F = 1/(1 + R_N);
# - the **parallel-tube** model: blood flows through the sinusoids without mixing, and the
#   drug is eliminated along the way, F = exp(-R_N).
#
# Here R_N = fu CL_int/Q is the *efficiency number* (unbound intrinsic clearance over hepatic
# blood flow), and hepatic clearance is CL_H = Q (1 - F). The two models are the extremes of
# one picture: blood flows along the sinusoids and also mixes along them, because the
# sinusoids differ in length and speed. Roberts & Rowland (J Pharmacokinet Biopharm 1986) wrote
# that picture as the **dispersion model**: the drug concentration C along the sinusoid moves
# with the blood, spreads by dispersion and is eliminated,
#
#     dC/dθ = -dC/dZ + D_N d²C/dZ² - R_N C,
#
# with Z the position along the liver from 0 (inlet) to 1 (outlet) and θ time in units of the
# liver's transit time (V/Q). The *dispersion number* D_N measures the mixing: D_N -> 0 gives
# the parallel-tube model, D_N -> infinity the well-stirred one.
#
# This is a field along one spatial axis, which TenQSP writes as it is printed. What you will
# see:
#
# 1. the model, with the Danckwerts boundary conditions;
# 2. its steady state against the closed-form availability F(D_N, R_N), as the cells are
#    refined;
# 3. the two limits, a third classical model (tanks in series), and what it means for
#    predicted clearance;
# 4. a bolus through the liver: its outflow curve, and the mean and variance of the transit
#    time.
#
# **Values.** D_N = 0.17 is of the order Roberts & Rowland reported for the perfused rat
# liver; R_N = 2 (an extraction ratio near 0.8) and the hepatic blood flow of 90 L/h are
# illustrative.

# %%
import matplotlib.pyplot as plt
import numpy as np

import tenqsp as tq


def availability(DN, RN):
    """Roberts & Rowland (1986): F = C(1)/C_in for the dispersion model, Danckwerts ends."""
    a = np.sqrt(1 + 4 * RN * DN)
    return (
        4
        * a
        / ((1 + a) ** 2 * np.exp((a - 1) / (2 * DN)) - (1 - a) ** 2 * np.exp(-(a + 1) / (2 * DN)))
    )


# %% [markdown]
# ## 1. The model
#
# The equation has three terms: transport with the blood (`-d_Z(C)`, speed 1 in these units),
# dispersion (`d_Z(DN*d_Z(C))`) and elimination (`-RN*C`). The axis is cut into **cells**
# (here 80 pieces of liver, each 1/80 long), and the `sharp` **scheme** moves the drug between
# them: unlike the default `erlang` scheme, it adds almost no spreading of its own, so the
# mixing in the model is D_N's.
#
# **The two ends.** They are written under the equation, as in a paper. At the inlet, the
# blood brings drug at concentration C_in, and all of it enters: the flux crossing the inlet,
# carried and dispersed, equals what the blood brings (`at Z=0: C - DN*d_Z(C) = Cin`; the left
# side is the flux along `Z` that the equation's two transport terms give). At the outlet, the
# drug leaves with the blood and nothing disperses back in from the hepatic vein
# (`at Z=1: outflow`), here into a compartment `Out` that collects it: `Out` gains the flux
# there, `C(Z=1)` (speed 1 times the concentration at the outlet). These are the Danckwerts
# ("closed") boundary conditions of Roberts & Rowland: C - D_N dC/dZ = C_in at the inlet and
# dC/dZ = 0 at the outlet.
#
# At steady state the drug leaving per unit time, divided by what enters (C_in = 1, times a
# flow of 1), is the availability F.

# %%
LIVER = """
model liver
axis Z in [0, 1] n={n} kind=spatial          # position along the sinusoid
param DN=0.17, RN=2, Cin=1
field C(Z)
d_t(C) = -d_Z(C) + d_Z(DN*d_Z(C)) - RN*C{sharp}
    at Z=0: C - DN*d_Z(C) = Cin              # what the blood brings in
    at Z=1: outflow                          # what leaves with the blood
d_t(Out) = C(Z=1)                            # collects the drug escaping the liver
"""


def liver(n, scheme="sharp", dispersion=True):
    text = LIVER.format(n=n, sharp="    using sharp" if scheme == "sharp" else "")
    if not dispersion:  # no mixing at all: drop the dispersion term, and its part of the flux
        text = text.replace(" + d_Z(DN*d_Z(C))", "").replace("C - DN*d_Z(C) = Cin", "C = Cin")
    return tq.Model.from_text(text)


def steady_availability(model, params=None, t_end=40.0):
    """Run to steady state; the drug leaving between t_end and t_end + 1 is F."""
    sol = model.solve(
        (0, t_end + 1), times=[t_end, t_end + 1], params=params, rtol=1e-11, atol=1e-14
    )
    out = np.asarray(sol["Out"])
    return float(out[1] - out[0])


print(liver(80).diagnostics())

# %% [markdown]
# `diagnostics()` reports the **cell Péclet number**, the cell length times the speed divided
# by D_N: it says whether the cells are fine enough to resolve the dispersion (below 2 is
# comfortable).
#
# ## 2. The availability against the closed form
#
# Roberts & Rowland give the steady-state availability in closed form: with a = sqrt(1 + 4 R_N
# D_N),
#
#     F = 4a / [(1 + a)² exp((a - 1)/(2 D_N)) - (1 - a)² exp(-(a + 1)/(2 D_N))].
#
# The model's F approaches it as the cells are refined: with `sharp`, the error falls about
# four-fold or more each time the cells are doubled (second order). With the `erlang` scheme it
# only halves (first order): erlang moves drug between cells as a chain of well-mixed tanks,
# which adds mixing of its own (about half a cell's length of extra D_N), and that biases F
# upwards.

# %%
F_exact = availability(0.17, 2.0)
print(f"closed form: F = {F_exact:.6f}, extraction ratio {1 - F_exact:.4f}")
errors = {"sharp": [], "erlang": []}
for n in (20, 40, 80, 160):
    for scheme in errors:
        errors[scheme].append(steady_availability(liver(n, scheme)) - F_exact)
    print(
        f"{n:4d} cells: error sharp {errors['sharp'][-1]:+.1e}, erlang {errors['erlang'][-1]:+.1e}"
    )
sharp, erlang = np.abs(errors["sharp"]), np.abs(errors["erlang"])
assert np.all(sharp[:-1] / sharp[1:] > 3.5)  # second order (or better, on coarse grids)
assert np.all(np.abs(erlang[:-1] / erlang[1:] - 2) < 0.1)  # first order
assert sharp[-1] < 1e-6
assert np.all(np.asarray(errors["erlang"]) > 0)  # extra mixing: always too much drug escapes

# %% [markdown]
# ## 3. The two limits, tanks in series, and clearance
#
# For little mixing (D_N = 0.001) the model must give the parallel-tube F = exp(-R_N); for
# a lot (D_N = 1000), the well-stirred F = 1/(1 + R_N). (Little mixing needs fine cells: the
# cell Péclet number 1/(n D_N) is kept near 2.)
#
# The extra mixing of the `erlang` scheme is itself a classical model: with no dispersion
# term, N `erlang` cells are N well-stirred tanks in series, F = (1 + R_N/N)^-N (the
# tanks-in-series model of hepatic elimination). The model gives it to solver tolerance on any
# number of cells.

# %%
RN = 2.0
limits = {
    "parallel tube": (steady_availability(liver(400), {"DN": 1e-3, "RN": RN}), np.exp(-RN)),
    "well stirred": (steady_availability(liver(100), {"DN": 1e3, "RN": RN}), 1 / (1 + RN)),
}
for name, (model_F, limit) in limits.items():
    print(f"{name}: model {model_F:.4f}, limit {limit:.4f}")
    assert abs(model_F / limit - 1) < 0.01
for n in (1, 3, 10):
    tanks = steady_availability(liver(n, "erlang", dispersion=False), {"RN": RN})
    print(f"{n:2d} tanks in series: F = {tanks:.6f}, closed form {(1 + RN / n) ** -n:.6f}")
    assert abs(tanks - (1 + RN / n) ** -n) < 1e-8

# %% [markdown]
# In a pharmacometrician's terms: with a hepatic blood flow Q, R_N = fu CL_int/Q and
# CL_H = Q (1 - F). The right panel shows predicted hepatic clearance against unbound
# intrinsic clearance, for Q = 90 L/h: the models agree for poorly extracted drugs
# (fu CL_int << Q, all give CL_H ≈ fu CL_int) and differ for highly extracted ones, where the
# dispersion model lies between the two extremes. The points are the TenQSP model.

# %%
Q = 90.0  # L/h, illustrative
dn_grid = np.logspace(-3, 3, 61)
dn_points = {1e-2: 200, 0.1: 80, 0.17: 80, 1.0: 80, 10.0: 80}  # D_N: cells
fu_clint = np.logspace(0, 3.3, 60)  # L/h
fu_clint_points = [10.0, 45.0, 180.0, 720.0]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogx(dn_grid, [availability(d, RN) for d in dn_grid], label="closed form")
model_points = [steady_availability(liver(n), {"DN": d, "RN": RN}) for d, n in dn_points.items()]
ax[0].semilogx(list(dn_points), model_points, "o", label="TenQSP (sharp)")
ax[0].axhline(np.exp(-RN), color="grey", ls="--", lw=1, label="parallel tube, exp(-R_N)")
ax[0].axhline(1 / (1 + RN), color="grey", ls=":", lw=1, label="well stirred, 1/(1 + R_N)")
ax[0].set_xlabel("dispersion number D_N")
ax[0].set_ylabel("availability F")
ax[0].set_title(f"R_N = {RN:g}")
ax[0].legend(fontsize="small")
for d, f in zip(dn_points, model_points, strict=True):
    assert abs(f - availability(d, RN)) < 1e-4

r = fu_clint / Q
ax[1].loglog(fu_clint, Q * r / (1 + r), ls=":", label="well stirred")
ax[1].loglog(fu_clint, Q * (1 - availability(0.17, r)), label="dispersion, D_N = 0.17")
ax[1].loglog(fu_clint, Q * (1 - np.exp(-r)), ls="--", label="parallel tube")
cl_points = [Q * (1 - steady_availability(liver(80), {"RN": x / Q})) for x in fu_clint_points]
ax[1].loglog(fu_clint_points, cl_points, "o", color="C1", label="TenQSP, D_N = 0.17")
ax[1].axhline(Q, color="grey", lw=1)
ax[1].set_xlabel("unbound intrinsic clearance fu CL_int (L/h)")
ax[1].set_ylabel("hepatic clearance CL_H (L/h)")
ax[1].legend(fontsize="small")
fig.tight_layout()
for x, cl in zip(fu_clint_points, cl_points, strict=True):
    assert abs(cl - Q * (1 - availability(0.17, x / Q))) < 1e-4 * Q

# %% [markdown]
# ## 4. A bolus through the liver
#
# Roberts & Rowland estimated D_N from **outflow curves**: a bolus injected into the portal
# vein of a perfused liver, and the concentration leaving it measured over time. In the model,
# the bolus starts in a compartment `Src` that empties quickly into the inlet (rate k_f = 20,
# so in about 1/20 of a transit time).
#
# Two checks, from the moments of the time the drug spends in the liver:
#
# - **Mean.** Without elimination, the mean transit time is exactly 1 (one V/Q), whatever
#   D_N: mixing spreads the transit times but does not change their mean. `W1` integrates the
#   drug still inside over time, which gives that mean (plus 1/k_f spent in `Src`).
# - **Variance.** For a tube closed at both ends (the Danckwerts conditions), the variance of
#   the transit time is 2 D_N - 2 D_N² (1 - exp(-1/D_N)), a standard result of chemical
#   engineering. This is the relation used to read D_N off an outflow curve. `W2` integrates
#   2 t times the drug inside, which gives the second moment.
#
# With elimination, the bolus loses the same fraction as a constant infusion does at steady
# state, so the drug that comes out must be F x dose, and what came out plus what was
# eliminated must be the dose (from the solution's **ledger**, TenQSP's record of every amount
# moved by each term of the model).

# %%
BOLUS = """
model liver_bolus
axis Z in [0, 1] n=100 kind=spatial
param DN=0.17, RN=0, kf=20
field C(Z)
d_t(Src) = -kf*Src
d_t(C) = -d_Z(C) + d_Z(DN*d_Z(C)) - RN*C    using sharp
    at Z=0: C - DN*d_Z(C) = kf*Src
    at Z=1: outflow
d_t(Out) = C(Z=1)
inside = Src + int_Z(C)                     # drug not yet out
d_t(W1) = inside                            # -> mean time to come out
d_t(W2) = 2*t*inside                        # -> second moment
Src(0) = 1
"""
bolus = tq.Model.from_text(BOLUS)
theta = np.linspace(0, 4, 401)
fig, ax = plt.subplots(figsize=(7, 4))
kf = 20.0
for dn in (0.05, 0.17, 0.5):
    s = bolus.solve((0, 40), times=np.append(theta, 40), params={"DN": dn}, rtol=1e-10, atol=1e-13)
    w1, w2 = float(np.asarray(s["W1"])[-1]), float(np.asarray(s["W2"])[-1])
    mean = w1 - 1 / kf  # the time spent in Src is exponential with mean 1/kf ...
    var = (w2 - w1**2) - 1 / kf**2  # ... and variance 1/kf^2; times add, so variances add
    var_exact = 2 * dn - 2 * dn**2 * (1 - np.exp(-1 / dn))
    print(
        f"D_N = {dn}: mean transit time {mean:.5f} (exact 1), "
        f"variance {var:.5f} (exact {var_exact:.5f})"
    )
    assert abs(float(np.asarray(s["Out"])[-1]) - 1) < 1e-9  # everything came out
    assert abs(mean - 1) < 1e-4
    assert abs(var - var_exact) < 1e-3 * var_exact
    rate_out = np.gradient(np.asarray(s["Out"])[:-1], theta)
    ax.plot(theta, rate_out, label=f"D_N = {dn}")
ax.set_xlabel("time (in transit times, V/Q)")
ax.set_ylabel("rate of drug leaving the liver (fraction of dose)")
ax.set_title("outflow curve after a bolus, no elimination")
ax.legend(fontsize="small")
fig.tight_layout()

eliminated_run = bolus.solve((0, 40), times=[40], params={"RN": 2.0}, rtol=1e-10, atol=1e-13)
came_out = float(np.asarray(eliminated_run["Out"])[-1])
moved = eliminated_run.ledger.moved["C"]
eliminated = -float(np.asarray(moved["hazard: RN"])[-1])
print(f"with R_N = 2: came out {came_out:.5f} (F = {F_exact:.5f}); eliminated {eliminated:.5f}")
assert abs(came_out - F_exact) < 1e-4
assert abs(came_out + eliminated - 1) < 1e-9

# %% [markdown]
# Why mixing raises F: elimination is first order at the same rate everywhere, so a molecule
# that spends a time T in the liver escapes with probability exp(-R_N T), and F is the average
# of exp(-R_N T) over the outflow curve. All curves have the same mean, T = 1, but the average
# of exp(-R_N T) grows with the spread around it (the drug escaping early gains more than the
# drug staying late loses). So F is lowest, exp(-R_N), with no spread (parallel tube), and
# rises with D_N towards the well-stirred 1/(1 + R_N) (section 3).
#
# ## What this buys you
#
# - The dispersion model is written as Roberts & Rowland print it; the well-stirred,
#   parallel-tube and tanks-in-series models are its limits and its `erlang` version, so one
#   model covers the clearance models in use.
# - The liver is now an axis, so anything can vary along it: zonation of enzymes (CL_int
#   higher near the outlet), oxygen, or a drug's own gradient feeding back on its elimination.
# - A model with the liver as a dispersion organ fits into a larger model (a PBPK model, a
#   gut model as in Tutorial 2) as an ordinary part.
#
# **Limits.** This is the one-dimensional model, with linear elimination and the drug in
# blood only (no binding kinetics, no uptake into hepatocytes, which Roberts & Rowland's later
# papers add). With `sharp`, little mixing needs fine cells (cell Péclet number near 2 or less);
# `diagnostics()` reports it.
