# %% [markdown]
# # Tutorial 9: persisters, from two states to a dormancy axis
#
# When an antibiotic kills a bacterial culture, the number of live cells often falls fast at
# first and then much more slowly: a **biphasic kill curve**. The few cells that survive the
# first phase are not resistant (their offspring are killed as fast as the original culture);
# they were *dormant*, growing little or not at all, when the drug arrived. Balaban et al.
# (2004, Science) watched single cells and described them as a **phenotypic switch**: normal
# cells n turn into persisters p at a rate a and back at a rate b. Nielsen et al. (2007,
# Antimicrob Agents Chemother) used the same idea in a pharmacometric model of time-kill
# curves: a growing, drug-sensitive state S and a resting, insensitive state R.
#
# Both are models over a **categorical axis** (a list of states). Dormancy, though, is a
# matter of degree: cells slow down gradually, and the deeper they are the longer they take
# to wake up. This tutorial goes from the two states to a **continuous dormancy level**
# along which cells drift and spread, and growth and kill depend on the level:
#
# 1. Balaban's two-state switch under drug, checked against its exact solution;
# 2. Nielsen-style time-kill curves, where the switching rate depends on the total;
# 3. a continuous dormancy level (a Fokker-Planck equation), its mass balance and steady
#    state;
# 4. two wells on the dormancy level, and when they behave as two states;
# 5. growth and kill depending on the level: a biphasic kill curve without declaring two
#    states.
#
# All values are illustrative (chosen to show the behaviour, not fitted to data).

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

import tenqsp as tq

TIGHT = {"rtol": 1e-10, "atol": 1e-12}

# %% [markdown]
# ## 1. Balaban's switch: two states on a categorical axis
#
# The cells are one field `B` over the axis `state`, with two categories, `n` (normal) and
# `p` (persister). Under the drug, normal cells are killed at `gamma[n]` per hour and
# persisters at the much lower `gamma[p]`. The switching is declared once, as
# `transitions Q on state: n -> p = a, p -> n = b`, and applied with `trans_state(Q, B)`:
# TenQSP adds what arrives in each state and subtracts what leaves it, so switching can
# neither create nor lose cells. (In NONMEM you would write the two equations and the four
# switching terms by hand.)

# %%
balaban = tq.Model.from_text("""
model balaban
axis state categorical [n, p]
param a=1e-3, b=1e-2
param gamma[state] = {n: 4, p: 0.05}
field B(state)
transitions Q on state: n -> p = a, p -> n = b
d_t(B) = trans_state(Q, B) - gamma[state]*B
B[state=n](0) = 1e6
observable total = sum_state(B)
""")
t = np.linspace(0, 24, 97)
sol_b = balaban.solve((0, 24), times=t, **TIGHT)

# %% [markdown]
# The model is linear with constant rates, so its exact solution is a matrix exponential:
# B(t) = exp(A t) B(0), with A the 2 x 2 matrix of the rates. The total is a sum of two
# exponentials, and the two slopes of log(total) are the two eigenvalues of A: the fast one
# (about -gamma[n]) is the killing of the normal cells; the slow one (about -(gamma[p] + b))
# is the persisters, killed slowly and waking up into a drug that kills them.

# %%
A = np.array([[-4.0 - 1e-3, 1e-2], [1e-3, -0.05 - 1e-2]])
exact = np.array([expm(A * ti) @ [1e6, 0.0] for ti in t])
rel = np.max(np.abs(np.asarray(sol_b["B"]) - exact) / np.maximum(exact, 1e-3))
print(f"largest relative difference from the matrix exponential: {rel:.1e}")
assert rel < 1e-7

lam = np.sort(np.linalg.eigvals(A).real)
total = np.asarray(sol_b["total"])
fast = np.log(total[2] / total[0]) / (t[2] - t[0])  # the first half hour
slow = np.polyfit(t[t >= 10], np.log(total[t >= 10]), 1)[0]
print(f"slopes of log(total): first {fast:.3f}, late {slow:.4f} per h")
print(f"eigenvalues of A:          {lam[0]:.3f},      {lam[1]:.4f} per h")
assert abs(fast / lam[0] - 1) < 0.01
assert abs(slow / lam[1] - 1) < 1e-6

# %% [markdown]
# ## 2. Nielsen's resting state: a switching rate that depends on the total
#
# In Nielsen's model the growing state S doubles, dies at `kd` and is killed by the drug
# with an Emax model; the resting state R does neither growth nor drug, and only dies. Cells
# move from S to R faster as the culture fills up: k_SR = (kg - kd) N / Bmax, where N = S + R
# is the total. In the notation, `N = sum_state(B)` is that total, and the rate of a
# transition may be any formula, here one of N. This is a static time-kill experiment: the
# concentration `C` is a parameter, set when solving.

# %%
nielsen = tq.Model.from_text("""
model nielsen
axis state categorical [S, R]
param kg=1.5, kd=0.2, Bmax=1e9, Emax=3, EC50=1, hill=1.5, C=0
param grows[state] = {S: 1, R: 0}
field B(state)
N = sum_state(B)
E = Emax*C^hill/(EC50^hill + C^hill)
transitions Q on state: S -> R = (kg - kd)*N/Bmax
d_t(B) = grows[state]*(kg - E)*B - kd*B + trans_state(Q, B)
B[state=S](0) = 1e6
observable total = N
""")
concentrations = [0, 0.5, 1, 2, 8]
t_kill = np.linspace(0, 24, 49)
kill_curves = {
    c: nielsen.solve((0, 24), times=t_kill, params={"C": c}, **TIGHT) for c in concentrations
}

# %% [markdown]
# The check is the same model written out as three ODEs (S, R and nothing else) and solved
# with scipy at tight tolerance:


# %%
def nielsen_by_hand(c, kg=1.5, kd=0.2, Bmax=1e9, Emax=3.0, EC50=1.0, hill=1.5):
    E = Emax * c**hill / (EC50**hill + c**hill)

    def rhs(_, y):
        S, R = y
        ksr = (kg - kd) * (S + R) / Bmax
        return [(kg - E) * S - kd * S - ksr * S, ksr * S - kd * R]

    ref = solve_ivp(rhs, (0, 24), [1e6, 0.0], t_eval=t_kill, method="Radau", rtol=1e-12)
    return ref.y.T


for c, s in kill_curves.items():
    ref = nielsen_by_hand(c)
    diff = np.max(np.abs(np.asarray(s["B"]) - ref) / np.maximum(ref, 1.0))
    assert diff < 1e-6, (c, diff)
print("all five time-kill curves match the hand-written ODEs to 1e-6")
resting = np.asarray(kill_curves[0]["B"])[-1]
print(f"no drug, 24 h: {resting[0]:.2e} growing and {resting[1]:.2e} resting cells")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(t, np.asarray(sol_b["B"])[:, 0], label="normal")
ax[0].semilogy(t, np.asarray(sol_b["B"])[:, 1], label="persisters")
ax[0].semilogy(t, total, "k--", label="total")
ax[0].set_ylim(1e-2, 2e6)
ax[0].set_title("Balaban switch under drug: biphasic kill")
for c, s in kill_curves.items():
    ax[1].semilogy(t_kill, s["total"], label=f"C = {c} x EC50")
ax[1].set_title("Nielsen-style time-kill curves")
for a_ in ax:
    a_.set_xlabel("time (h)")
    a_.set_ylabel("cells per mL")
    a_.legend(fontsize="small")
fig.tight_layout()

# %% [markdown]
# ## 3. A continuous dormancy level
#
# Now replace the two states by a level xi from 0 (fully active) to 1 (deeply dormant). Each
# cell's level wanders: it is pulled toward a typical level xi0 at a rate theta (a
# **drift**), and jitters at random, more strongly deeper down (a **diffusion** D(xi) =
# D0 (1 + xi)). For one cell this is a stochastic differential equation,
# d xi = -theta (xi - xi0) dt + sqrt(2 D(xi)) dW. For the population it is the
# **Fokker-Planck equation** for the density B(xi):
#
#     d_t(B) = -d_xi((drift - dD/dxi)*B) + d_xi(D*d_xi(B))
#
# The term `- dD/dxi` (here `- D0`) is easy to forget: where cells jitter more, they also
# leave faster, and without it the population would pile up where D is small. Both ends are
# sealed (TenQSP's default): no cell leaves the axis.
#
# The axis is cut into **cells** (small intervals of xi; not bacteria), and the equation is
# solved for the number of bacteria in each; `using sharp` chooses the more accurate of
# TenQSP's two ways of moving them between cells (see Tutorial 1). With growth, death and
# drug set to 0 (their defaults below), the model only moves bacteria along the axis.

# %%
DORMANCY = """
model dormancy
axis xi in [0, 1] n=80 kind=bidirectional
param theta=2, xi0=0.3, D0=0.05
param g=0, d=0, E=0, K=1e9
field B(xi)
D(xi) = D0*(1 + xi)
drift(xi) = -theta*(xi - xi0)
N = int_xi(B)
d_t(B) = -d_xi((drift - D0)*B) + d_xi(D*d_xi(B))
         + g*(1 - xi)*(1 - N/K)*B - d*B - E*(1 - xi)^2*B     using sharp
B(0) = 1e6*normal(xi, mean=0.8, sd=0.05)
observable total = N
"""
dormancy = tq.Model.from_text(DORMANCY)
t_d = np.array([0, 0.25, 1, 3, 40.0])
sol_d = dormancy.solve((0, 40), times=t_d, rtol=1e-11, atol=1e-8)

# %% [markdown]
# Two checks. **Mass**: nothing is born or dies, so the total stays 1e6 to round-off (the
# amount that leaves one cell is exactly the amount that enters the next). **Steady state**:
# with sealed ends the flux vanishes everywhere at steady state, which gives
# B(xi) proportional to exp(integral of drift/D) / D, here
# (1 + xi)^(theta (1 + xi0)/D0 - 1) exp(-theta xi / D0). The model's cells are compared
# with the averages of that curve over each cell.

# %%
totals = np.asarray(sol_d["total"])
print(f"largest relative change in the total: {np.max(np.abs(totals / 1e6 - 1)):.1e}")
assert np.max(np.abs(totals / 1e6 - 1)) < 1e-12

theta, xi0, D0, n_xi = 2.0, 0.3, 0.05, 80
power = theta * (1 + xi0) / D0 - 1


def steady(x):
    peak = power * np.log1p(xi0) - theta * xi0 / D0  # divided out, to keep numbers modest
    return np.exp(power * np.log1p(x) - theta * x / D0 - peak)


faces = np.linspace(0, 1, n_xi + 1)
cells = np.array([quad(steady, lo, hi)[0] for lo, hi in zip(faces[:-1], faces[1:], strict=True)])
exact_ss = 1e6 * cells / cells.sum() / np.diff(faces)
err = np.max(np.abs(np.asarray(sol_d["B"])[-1] - exact_ss)) / exact_ss.max()
print(f"steady state: largest difference {err:.1e} of the peak (80 cells)")
assert err < 5e-3

# %% [markdown]
# The 0.2 % that remains is the grid: with 160 cells it is four times smaller (the
# accuracy of the `sharp` scheme is second order in the cell width).
#
# ## 4. Two wells on the dormancy level
#
# A single pull toward one typical level gives one population, with a spread. Persisters
# need two: most cells near an active level, a few near a deep one, and a barrier between
# them that cells cross only rarely. Write the drift as the downhill slope of a landscape
# U(xi), drift = -dU/dxi, with two valleys: one at xi near 0.14 (active) and a shallower
# one near 0.84 (dormant). D is now constant, so the correction term is 0. The steady state
# is then proportional to exp(-U/D): deep valleys hold most cells.
#
# The population starts in the active valley, distributed as it would be at equilibrium
# there (`shape(...)` spreads the amount like any formula, here exp(-U/D) cut at the
# barrier). Section 5 adds growth and kill; the parameters `g` and `E` are 0 until then.

# %%
WELLS = """
model wells
axis xi in [0, 1] n=200 kind=bidirectional
param H=0.05, w=0.35, tilt=0.04, D=0.01, g=0, E=0
field B(xi)
y(xi) = (xi - 0.5)/w
U(xi) = H*(1 - y^2)^2 + tilt*xi                   # the landscape
drift(xi) = 4*H*y*(1 - y^2)/w - tilt              # -dU/dxi
active(xi) = sigmoid((0.5 - xi)/0.03)             # 1 in the active valley, 0 in the dormant
d_t(B) = -d_xi(drift*B) + d_xi(D*d_xi(B)) + g*active*B - E*active*B     using sharp
B(0) = 1*shape((1 if xi < 0.5 else 0)*exp(-U/D))
observable awake = int_xi((1 if xi < 0.5 else 0)*B)
observable total = int_xi(B)
"""
wells = tq.Model.from_text(WELLS)
t_w = np.linspace(0, 200, 41)
sol_w = wells.solve((0, 200), times=t_w, rtol=1e-10, atol=1e-14)

# %% [markdown]
# If cells settle within a valley much faster than they cross the barrier, the masses in
# the two halves should behave like Balaban's two states, with switching rates a (active to
# dormant) and b (back). Both come from the continuous model, not from a fit:
#
# - at equilibrium a fraction pi_d = a/(a + b) is dormant: the share of exp(-U/D) above the
#   barrier (a quadrature);
# - the halves relax toward equilibrium at the rate a + b, the slowest rate of the
#   Fokker-Planck equation. That rate is computed here independently of TenQSP: the same
#   equation written on 1000 cells with plain finite differences (`fokker_planck_matrix`
#   below), and its eigenvalue closest to 0 that is not 0.


# %%
H, w, tilt, D = 0.05, 0.35, 0.04, 0.01


def landscape(x):
    return H * (1 - ((x - 0.5) / w) ** 2) ** 2 + tilt * x


def fokker_planck_matrix(n, net_rate=lambda x: 0.0 * x):
    """The wells model on n cells: flux at each inner face = drift x (mean of the two
    cells) - D x (difference) / width; sealed ends; plus a net growth rate per cell."""
    f = np.linspace(0, 1, n + 1)
    x, h = 0.5 * (f[1:] + f[:-1]), 1.0 / n
    y = (f[1:-1] - 0.5) / w
    v = 4 * H * y * (1 - y**2) / w - tilt
    left, right = (v / 2 + D / h) / h, (v / 2 - D / h) / h  # face flux = left B_i + right B_i+1
    main = np.zeros(n)
    main[:-1] -= left
    main[1:] += right
    return diags([main + net_rate(x), -right, left], [0, 1, -1], format="csc")


def leading_rates(n, net_rate=lambda x: 0.0 * x, k=2):
    vals = eigs(fokker_planck_matrix(n, net_rate), k=k, sigma=1e-6, return_eigenvectors=False)
    return np.sort(vals.real)[::-1]


Z = quad(lambda x: np.exp(-landscape(x) / D), 0, 1, epsrel=1e-12)[0]
pi_d = quad(lambda x: np.exp(-landscape(x) / D), 0.5, 1, epsrel=1e-12)[0] / Z
a_plus_b = -leading_rates(1000)[1]
a, b = pi_d * a_plus_b, (1 - pi_d) * a_plus_b
print(f"dormant at equilibrium {pi_d:.4f}; a + b = {a_plus_b:.5f} per h")
print(f"two-state rates from the landscape: a = {a:.2e}, b = {b:.2e} per h")

switch = np.array([[-a, b], [a, -b]])
two_state = np.array([(expm(switch * ti) @ [1.0, 0.0])[0] for ti in t_w])
awake = np.asarray(sol_w["awake"])
print(f"awake fraction at 200 h: continuous {awake[-1]:.4f}, two states {two_state[-1]:.4f}")
assert np.max(np.abs(awake - two_state)) < 1e-3

# %% [markdown]
# The halves follow the two-state model to within 0.06 % of the cells over 200 h (and
# closer on a finer grid: 0.2 % with 100 cells). This is the sense in which a two-state
# model is a limit of the continuous one: valleys that are deep compared with D (here the
# barrier is 3.7 to 6.5 times D), so that cells forget where in a valley they came from before
# they cross.
#
# ## 5. Growth and kill along the level
#
# Now let active cells grow (g = 1 per h) and the drug kill them (E = 4 per h), both through
# `active(xi)`, while dormant cells neither grow nor are killed. No category is declared,
# yet the kill curve is biphasic: the active valley is emptied within hours, then the
# population falls at the pace at which dormant cells climb out of their valley into the
# drug.
#
# Checks: the **books close** (the ledger accounts for every cell born and killed); and the
# late slope of log(total) is the leading eigenvalue of the same equation with growth and
# kill, from the independent finite-difference matrix above.

# %%
t_k = np.concatenate([np.linspace(0, 10, 41), np.linspace(15, 200, 38)])  # the first phase is fast
sol_k = wells.solve((0, 200), times=t_k, params={"g": 1.0, "E": 4.0}, rtol=1e-10, atol=1e-16)
print("ledger, largest relative unexplained:", max(sol_k.ledger.check().values()))
assert max(sol_k.ledger.check().values()) < 1e-10

total_k = np.asarray(sol_k["total"])
late = t_k >= 100
slope = np.polyfit(t_k[late], np.log(total_k[late]), 1)[0]
expected = leading_rates(1000, lambda x: (1.0 - 4.0) / (1 + np.exp(-(0.5 - x) / 0.03)), k=1)[0]
print(f"late slope {slope:.5f} per h; leading eigenvalue {expected:.5f} per h")
assert abs(slope / expected - 1) < 0.01

# the two-state model with the switching rates of section 4
two = np.array([[1.0 - 4.0 - a, b], [a, -b]])
slow_two = np.sort(np.linalg.eigvals(two).real)[-1]
print(f"two-state model with a, b of section 4: late slope {slow_two:.5f} per h")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
xc = 0.5 * (faces[1:] + faces[:-1])
for ti, row in zip(t_d, np.asarray(sol_d["B"]), strict=True):
    ax[0].plot(xc, row, label=f"t = {ti:g} h")
ax[0].plot(xc, exact_ss, "k:", label="exact steady state")
ax[0].set_xlabel("dormancy level xi")
ax[0].set_ylabel("cells per unit of xi")
ax[0].set_title("one typical level: drift and spread")
ax[1].plot(t_w, awake, label="continuous level, xi < 0.5")
ax[1].plot(t_w, two_state, "k--", label="two states, a and b from the landscape")
ax[1].set_xlabel("time (h)")
ax[1].set_ylabel("fraction in the active valley")
ax[1].set_title("two valleys, no drug")
ax[2].semilogy(t_k, total_k, label="continuous level")
ax[2].semilogy(
    t_k, total_k[late][0] * np.exp(expected * (t_k - 100)), "k:", label="leading eigenvalue"
)
ax[2].set_xlabel("time (h)")
ax[2].set_ylabel("cells (start = 1)")
ax[2].set_title("two valleys, growth and kill when active")
for a_ in ax:
    a_.legend(fontsize="small")
fig.tight_layout()

# %% [markdown]
# The last line printed is worth a moment. A two-state model using the switching rates
# measured *without* the drug predicts a slower second phase than the continuous model
# shows. Under the drug, a dormant cell need not climb all the way over the barrier: it is
# killed as soon as it comes near the active side, and half of the cells that reach the
# top of the barrier would otherwise have fallen back. The wake-up rate seen in a kill curve
# is therefore not the spontaneous switching rate. A two-state model fitted to the kill
# curve absorbs this into b; the continuous level explains where the difference comes
# from.
#
# ## What this buys you
#
# - Two states, a switching rate that depends on the total and a drug acting on one state
#   are a few lines of notation; `trans_state` keeps the books of the switching.
# - The same population on a continuous level carries more biology: how deep a cell is
#   decides how fast it grows, how much drug it feels and how long it takes to wake up, and
#   the biphasic kill curve emerges instead of being declared.
# - The two-state model is recovered when the dormant and active levels are separated by a
#   barrier high compared with the jitter, with rates that follow from the landscape.
#
# **Limits.** The landscape and the jitter here are illustrative; measuring them needs
# single-cell data (lag times, regrowth after drug), not population kill curves alone. The
# dormancy level is a single number per cell; real dormancy has several causes (ATP, ppGpp,
# toxin-antitoxin systems) that this lumps together.
