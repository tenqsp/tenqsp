# %% [markdown]
# # Tutorial 11: how fast does a phage plaque grow?
#
# A drop of phage on a lawn of bacteria makes a **plaque**: a clear disc that widens as
# phage infect the cells at its edge, burst out of them and diffuse to the next cells. Yin
# and McCaskill (1992, Biophys J) wrote this as a reaction-diffusion model; Fort and Méndez
# (2002, Phys Rev Lett) and Gourley and Kuang (2004, SIAM J Appl Math) showed that the
# **latent period** (the time from infection to lysis) slows the front markedly.
#
# Here the infected cells carry an **infection age** a, the time since they were infected,
# as in Tutorial 7: they age at speed 1 and lyse at a hazard h(a) that rises near the
# latent period tau. Space is a second axis x (micrometres across the lawn), so the infected
# cells are a field over (a, x). The age axis replaces the delay of the delay-differential
# models: any distribution of lysis times can be written, a fixed delay being one limit.
#
# 1. the model, written in TenQSP's notation;
# 2. without space: the growth rate of the phage, from the lysis-time distribution;
# 3. without a latent period: Fisher's front speed 2 sqrt(D r);
# 4. with a latent period, the linear theory of the front speed, checked with a shallow
#    start;
# 5. a steep start: the front runs somewhat faster than the minimal speed of the linear
#    theory, and why.
#
# **Values** are illustrative (of the order used in the plaque literature): lawn 1e9 cells
# per mL, adsorption rate constant 1e-8 mL/h, burst size 50, latent period about 0.5 h,
# phage diffusivity 400 um^2/h in the soft agar. The lawn does not grow (a stationary lawn).
#
# This script runs in two to three minutes: sections 4 and 5 solve a model of 9,600 + 480
# states three times.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq, minimize_scalar

import tenqsp as tq

# %% [markdown]
# ## 1. The model
#
# - `B(x)`: uninfected bacteria. They are infected at `k*B*P`, written as in a paper in
#   each equation it changes: lost from `B`, used up from `P` (one phage per infection), and
#   entering the infected cells at age 0.
# - `I(a, x)`: infected cells, by infection age and position. `-d_a(I)` is ageing; `-h*I`
#   is lysis at the hazard h(a). Under its equation, `at a=0: I = k*B*P` is the boundary
#   condition by which the new infections enter the age axis (cells age at speed 1, so the
#   cells crossing age 0 per hour are `I` there), and `at a=1.5: outflow` lets the few cells
#   that reach the end of the age axis leave (counted in the ledger).
# - `P(x)`: free phage. They diffuse (`d_x(D*d_x(P))`), are released at `beta` per lysed
#   cell (`lysis = int_a(h*I)`, a field over x: the lysis rate at each position), are used
#   up when they adsorb (one phage per infection) and decay at `delta`.
#
# The phage start as `P0*exp(-lam0*x)`: many at the left end, falling off to the right.
# Both ends of the lawn are sealed, so the plaque spreads from x = 0 to the right. The same
# model is `tenqsp.examples.plaque_model()`, written with the Python API.

# %%
PLAQUE = """
model plaque
axis x in [0, 600] n=240 units=um kind=bidirectional    # position across the lawn
axis a in [0, 1.5] n=40 units=h kind=one_way            # infection age
param D=400, k=1e-8, B0=1e9, beta=50, delta=0.05, h_max=30, tau=0.5, s=0.03,
      P0=1e6, lam0=1
field B(x)
field P(x)
field I(a, x)
h(a) = h_max*sigmoid((a - tau)/s)                       # lysis hazard
lysis = int_a(h*I)                                      # cells lysing, at each x
d_t(B) = -k*B*P
d_t(I) = -d_a(I) - h*I          using sharp
    at a=0: I = k*B*P                                   # infected cells start at age 0
    at a=1.5: outflow
d_t(P) = d_x(D*d_x(P)) + beta*lysis - k*B*P - delta*P
B(0) = B0
P(0) = P0*exp(-lam0*x)
observable phage = int_x(P)
"""
V = dict(D=400.0, k=1e-8, B0=1e9, beta=50.0, delta=0.05, h_max=30.0, tau=0.5, s=0.03)

# %% [markdown]
# ## 2. Without space: how fast do phage multiply?
#
# Ahead of the plaque the lawn is untouched (B = B0) and the phage are few. There the model
# is linear, and the phage grow exponentially at a rate r. An infection at time t - a lyses
# at time t with probability density f(a) = h(a) exp(-(integral of h from 0 to a)), the
# **lysis-time distribution**. Counting the phage gives the characteristic equation
# (Euler-Lotka in demography, the same equation that gives a population's growth rate from
# its life table):
#
#     r = -k B0 - delta + beta k B0 f^(r),    f^(s) = integral of f(a) exp(-s a) da
#
# (lost by adsorption and decay; gained from the bursts of cells infected earlier, discounted
# by how much the phage have grown since). f^ is computed by quadrature on a fine age grid.


# %%
def lysis_transform(p, a_max=1.5):
    """s -> f^(s), the Laplace transform of the lysis-time density (to the end of the axis)."""
    a = np.linspace(0.0, a_max, 30001)
    h = p["h_max"] / (1 + np.exp(-(a - p["tau"]) / p["s"]))
    f = h * np.exp(-cumulative_trapezoid(h, a, initial=0.0))
    return lambda s: np.trapezoid(f * np.exp(-s * a), a)


def growth_rate(p):
    F, kB = lysis_transform(p), p["k"] * p["B0"]
    return brentq(lambda r: r + kB + p["delta"] - p["beta"] * kB * F(r), 1e-6, 100.0)


r = growth_rate(V)
print(f"root of the characteristic equation: r = {r:.4f} per h")

# %% [markdown]
# To see this rate in the model, start with a uniform, vanishingly small phage titre
# (P0 = 1e-30, lam0 = 0): every position is then the same well-mixed culture, and the lawn
# is depleted by less than 1e-6. `cells=` solves the same text on another grid: 2 cells in
# x are enough here, and 80 in age make the lysis-time distribution sharper. The first
# bursts come in step; they take a few latent periods to spread out, so the rate is
# measured late.

# %%
mixed = tq.Model.from_text(PLAQUE, cells={"x": 2, "a": 80})
t = np.linspace(0.0, 9.0, 361)
sol = mixed.solve((0, 9), times=t, params={"P0": 1e-30, "lam0": 0.0}, rtol=1e-10, atol=1e-45)
late = t >= 7
rate = np.polyfit(t[late], np.log(np.asarray(sol["phage"])[late]), 1)[0]
print(f"model: phage grow at {rate:.4f} per h ({rate / r - 1:+.2%})")
assert abs(rate / r - 1) < 3e-3
assert max(sol.ledger.check().values()) < 1e-9

# %% [markdown]
# ## 3. Without a latent period: Fisher's front
#
# If cells lyse as soon as they are infected (no `I` at all), each adsorption returns
# `beta - 1` phage at once, the phage grow at r = (beta - 1) k B0 - delta at the leading
# edge, and the equation for P is Fisher's. Its front moves at c* = 2 sqrt(D r) (Fisher
# 1937; Kolmogorov, Petrovsky and Piskunov 1937), whatever the start, provided the start
# falls off steeply enough. This front is **pulled**: its speed is set entirely at the
# leading edge, where the phage are few.
#
# For a clear front, burst size 2 and fast decay (delta = 5 per h) here: r = 5 per h.

# %%
NO_LATENCY = """
model plaque_no_latency
axis x in [0, 600] n=240 units=um kind=bidirectional
param D=400, k=1e-8, B0=1e9, beta=2, delta=5, P0=1e6, lam0=1
field B(x)
field P(x)
infect = k*B*P
d_t(B) = -infect
d_t(P) = d_x(D*d_x(P)) + beta*infect - infect - delta*P
B(0) = B0
P(0) = P0*exp(-lam0*x)
"""
x = (np.arange(240) + 0.5) * 600 / 240  # cell centres, um


def front_speed(sol, t_from, x_to=500.0):
    """Speed of the point where the lawn has fallen to half, from t_from until the front
    reaches x_to (fitted straight line)."""
    B = np.asarray(sol["B"]) / 1e9
    front = np.array([x[np.where(row < 0.5)[0].max()] if (row < 0.5).any() else 0 for row in B])
    sel = (sol.t >= t_from) & (front < x_to) & (front > 0)
    return np.polyfit(sol.t[sel], front[sel], 1)[0], front


fisher = tq.Model.from_text(NO_LATENCY)
sol_f = fisher.solve((0, 8), times=np.linspace(0, 8, 81), rtol=1e-6)
c_fisher, _ = front_speed(sol_f, 3.0)
c_star_f = 2 * np.sqrt(400 * ((2 - 1) * 1e-8 * 1e9 - 5))
print(f"front speed {c_fisher:.1f} um/h; 2 sqrt(D r) = {c_star_f:.1f} um/h")
assert abs(c_fisher / c_star_f - 1) < 0.03

# %% [markdown]
# The front moves at 2 sqrt(D r) to within 0.5 % (the rest is the 2.5 um cells and the
# fitting window).
#
# ## 4. With a latent period: the linear theory of the front speed
#
# At the leading edge the phage fall off like exp(-lambda (x - c t)) for some decay rate
# lambda. Putting this into the linearized model gives, for each lambda, the speed c at
# which such an edge moves (Fort and Méndez 2002):
#
#     lambda c = D lambda^2 - k B0 - delta + beta k B0 f^(lambda c)
#
# the characteristic equation of section 2 with r replaced by lambda c - D lambda^2. The
# delay enters through f^: the bursts arrive late, by which time the edge has moved on. The
# speed c(lambda) has a minimum c* at some lambda*. A start that falls off **more slowly**
# than lambda* (lam0 < lambda*) keeps its own decay rate and moves at c(lam0); a steeper
# start should settle at c*.


# %%
def speed(p, lam):
    F, kB = lysis_transform(p), p["k"] * p["B0"]
    return brentq(
        lambda c: lam * c - p["D"] * lam**2 + kB + p["delta"] - p["beta"] * kB * F(lam * c),
        1e-9,
        1e6,
    )


def minimal_speed(p):
    res = minimize_scalar(lambda lam: speed(p, lam), bounds=(1e-4, 2.0), method="bounded")
    return res.fun, res.x


c_star, lam_star = minimal_speed(V)
no_delay = 2 * np.sqrt(V["D"] * ((V["beta"] - 1) * V["k"] * V["B0"] - V["delta"]))
print(f"c* = {c_star:.1f} um/h at lambda* = {lam_star:.3f} per um")
print(f"(the same phage without a latent period: 2 sqrt(D r) = {no_delay:.0f} um/h)")

plaque = tq.Model.from_text(PLAQUE).compile()  # compiled once, solved three times
t_s = np.linspace(0, 10, 101)
sol_s = plaque.solve((0, 10), times=t_s, params={"lam0": 0.1}, rtol=1e-6)
c_shallow, front_s = front_speed(sol_s, 4.0)
print(f"start exp(-0.1 x): front speed {c_shallow:.1f} um/h; c(0.1) = {speed(V, 0.1):.1f} um/h")
assert abs(c_shallow / speed(V, 0.1) - 1) < 0.01
assert max(sol_s.ledger.check().values()) < 1e-9

# %% [markdown]
# The latent period of half an hour cuts the minimal speed about eighteen-fold, from 885 to
# 49 um/h: during the latent period the infected cells stay put while the front would
# otherwise have moved on.
#
# ## 5. A steep start: faster than c*
#
# The default start, exp(-x) with lam0 = 1 per um, is much steeper than lambda* (about 0.19
# per um), so the linear theory predicts c*. Two phage decay rates are compared: the
# default (delta = 0.05 per h, phage live about 20 h) and fast decay (3 per h).

# %%
t_p = np.linspace(0, 12, 121)
pushed = {}
for delta in (0.05, 3.0):
    s = plaque.solve((0, 12), times=t_p, params={"delta": delta}, rtol=1e-6)
    c, front = front_speed(s, 4.0)
    cs, _ = minimal_speed(dict(V, delta=delta))
    pushed[delta] = (s, front)
    print(f"delta = {delta}: front speed {c:.1f} um/h, c* = {cs:.1f} um/h ({c / cs - 1:+.0%})")
    assert cs * 0.97 <= c <= cs * 1.2
    assert max(s.ledger.check().values()) < 1e-9

# %% [markdown]
# Both fronts run faster than c*: by about 14 % when phage live long, by about 7 % when they
# decay fast. This is not a transient: a front set by its leading edge approaches its speed
# from *below*, slowly (Bramson 1983, Mem Amer Math Soc), never from above. Nor is it the
# grid, which reproduced c(0.1)
# to 1 % in section 4 (though a steep edge, about 5 um long here, is less well resolved by
# 2.5 um cells than a shallow one).
#
# The reason is what the linear theory leaves out. It assumes the lawn is intact (B = B0)
# wherever there are phage, so that every phage is soon adsorbed. Behind the front the lawn
# is gone: the phage released there are no longer used up, live for 1/delta, and diffuse
# forward in large numbers into the leading edge. The front is then partly **pushed** from
# behind instead of only pulled by its edge, and moves faster than c*. The faster the phage
# decay, the smaller this reservoir and the closer the speed to c*, which is what the two
# runs show.

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
s05, front05 = pushed[0.05]
for when in (2, 4, 6, 8):
    i = int(np.argmin(np.abs(t_p - when)))
    (line,) = ax[0].plot(x, np.asarray(s05["B"])[i] / 1e9, label=f"lawn, t = {when} h")
    ax[0].plot(x, np.asarray(s05["P"])[i] / np.max(s05["P"]), ":", color=line.get_color())
ax[0].set_xlabel("position x (um)")
ax[0].set_ylabel("lawn / B0 (solid); phage / its maximum (dotted)")
ax[0].set_title("steep start, delta = 0.05 per h")
ax[0].legend(fontsize="small")
ax[1].plot(t_s, front_s, label="shallow start, exp(-0.1 x)")
for delta, (_, front) in pushed.items():
    ax[1].plot(t_p, front, label=f"steep start, delta = {delta}")
ax[1].plot(t_p, c_star * (t_p - 4) + front05[40], "k:", label="slope c* (delta = 0.05)")
ax[1].set_xlabel("time (h)")
ax[1].set_ylabel("front position (um; lawn at half)")
ax[1].set_ylim(0, 600)
ax[1].legend(fontsize="small")
fig.tight_layout()

# %% [markdown]
# ## What this buys you
#
# - The latent period is a distribution over infection age, not a fixed delay: the same
#   model covers a sharp latent period, a spread one, and (by removing the age axis) none.
#   The checks above use only the lysis-time distribution f(a).
# - Space and infection age are two axes of one field `I(a, x)`; the lysis rate at each
#   position is `int_a(h*I)`, and the ledger keeps the books of every phage and cell.
# - The linear theory gives the growth rate and the speeds that a model must reproduce; the
#   model shows where the theory stops (a pushed front).
#
# **Limits.** A one-dimensional lawn (a real plaque is a disc, whose edge moves at nearly
# the same speed once it is wide); a lawn that does not grow; no phage decay inside the agar
# other than first order. The 240 x 40 grid is coarse for the steep edge; a convergence study
# (`cells=`) would cost several minutes.
