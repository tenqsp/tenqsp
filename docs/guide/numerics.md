# Numerical choices

A model in TenQSP is a set of equations; solving it needs a few choices that the equations
do not make for you: how finely each axis is cut, how transport is carried from cell to
cell, how accurate each step must be, which solver runs it, and what to do with a population
that falls to a fraction of a cell. This chapter explains each choice in plain words, when it
matters, and how to check it.

Most models run well with the defaults: the `erlang` scheme, `rtol = atol = 1e-6`, and
scipy's BDF. The sections below say when not to rely on them.

## Finite volumes in plain words

A continuous axis (age, depth, progress through the gut) is cut into **cells**. Each cell
holds the average of the field over the cell: the amount in the cell divided by its width.
The boundaries between cells are called **faces**.

Transport and diffusion move material across faces. TenQSP computes one **flux** per face,
the amount crossing it per unit time, and takes it from the cell on one side and gives it to
the cell on the other. Whatever leaves one cell arrives in its neighbour, so the books
balance by construction: no amount is created or lost by the discretisation, only by the
terms you wrote (losses, births, boundaries). This is why transport must be written in
conservation form, `-d_a(v*I)`, and why the [ledger](explain.md#the-ledger-where-every-amount-came-from)
closes to rounding.

After this step the model is a large system of ordinary differential equations, one per cell
of each field plus one per compartment, which an ODE solver integrates in time, as NONMEM or
Monolix integrate a compartmental model.

## Grids

An axis is cut into `n` cells of equal width unless you say otherwise. Two other forms put
cells where they are needed:

| Notation | Python | Cells |
|---|---|---|
| `axis x in [0, 1] n=30` | `m.axis("x", 0, 1, 30)` | 30 cells of equal width |
| `axis x in [0, 1] n=8 spacing=geometric ratio=1.3` | `m.axis("x", 0, 1, 8, spacing="geometric", ratio=1.3)` | each cell 1.3 times as wide as the one before: fine near 0, coarse near 1 (a ratio below 1 refines toward the upper end) |
| `axis x faces=[0, 0.05, 0.1, 0.2, 0.4, 0.7, 1]` | `m.axis("x", faces=[0, 0.05, 0.1, 0.2, 0.4, 0.7, 1])` | the faces given: 6 cells |

Fine cells belong where the solution changes fast: near the surface of a biofilm where a
drug enters, or at the young end of an age axis where newly infected cells arrive.

```python
import numpy as np
import tenqsp as tq

m = tq.Model("grids")
x = m.axis("x", 0, 1, 8, spacing="geometric", ratio=1.3)
print(x.widths.round(3))        # [0.042 0.054 0.071 0.092 0.12 0.156 0.202 0.263]
print(x.faces.round(3))
```

### A convergence study

How fine is fine enough? Solve on two or three grids and compare what you care about. With a
model in a `.tq` file, `tq.load(..., cells=)` changes the number of cells of an axis without
editing the file, so the study needs one model file.

The model: a tablet's drug moves through the gut along progress `z` (0 just swallowed, 1 at
the absorption site) at speed 1/MTT, with the `sharp` scheme (below).

=== "Notation"

    ```tq
    model transit
    axis z in [0, 1] n=20 kind=one_way
    param MTT=1.5
    field G(z)
    d_t(G) = -d_z(G/MTT)  using sharp
        at z=1: outflow
    d_t(absorbed) = G(z=1)/MTT
    dose G at t=0: 100*point(z, at=0)
    observable arrival = G(z=1)/MTT
    ```

=== "Python"

    ```python
    m = tq.Model("transit")
    z = m.axis("z", 0, 1, 20, kind="one_way")
    MTT = m.param("MTT", 1.5)
    G = m.field("G", axes=[z])
    absorbed = m.compartment("absorbed")
    m.advect(G, axis=z, velocity=1 / MTT, scheme="sharp")
    m.at(G, z=1, outflow=True)
    m.ode(absorbed, G(z=1) / MTT)
    m.dose(G, times=[0], amounts=100, shape=tq.point(z, at=0))
    m.observable("arrival", G(z=1) / MTT)
    ```

The arrival rate at `z = 1` is the distribution of transit times. Its mean should be MTT and,
for transport without spread, its standard deviation should go to 0 as the grid is refined:

```python
from pathlib import Path

Path("transit.tq").write_text(m.to_text())
t = np.linspace(0, 4, 401)


def transit_times(model):
    sol = model.solve((0, 4), times=t)
    w = sol["arrival"] / 100 * (t[1] - t[0])          # fraction arriving in each interval
    mean = np.sum(t * w) / np.sum(w)
    sd = np.sqrt(np.sum((t - mean) ** 2 * w) / np.sum(w))
    return mean, sd


for model in [m, tq.load("transit.tq", cells={"z": 40})]:
    mean, sd = transit_times(model)
    print(f"sharp, {model['z'].n:3d} cells: mean {mean:.3f}, sd {sd:.3f}")
```

```text
sharp,  20 cells: mean 1.494, sd 0.178
sharp,  40 cells: mean 1.496, sd 0.106
```

The mean is right on both grids, and the spread shrinks as the grid is refined (it is 0.303
on 10 cells and 0.063 on 80). That spread is numerical: if a sharp delay is what you mean,
use enough cells that the spread no longer matters for your outputs (here, the time course of
`absorbed`).

## Schemes: `erlang` and `sharp`

Transport along an axis is carried by one of two schemes.

**`erlang`** (the default) is first-order upwind: each cell passes its content to the next at
rate |velocity| ÷ cell width. That is *exactly* a chain of transit compartments, one per cell,
so `m.explain()` describes it as one:

```python
erlang = tq.Model.from_text(m.to_text().replace("  using sharp on z", ""))
print(erlang.explain().split("Boundaries")[0])
```

```text
    transport along z, velocity 1 / MTT (erlang): first-order upwind on 20 cells, which is
      exactly a chain of 20 transit compartments: each cell of width 1/20 is one transit
      compartment (amount = density x 1/20), passing to the next at rate |velocity|/width =
      20/MTT (= 13.3). Mean transit time MTT (= 1.5), CV 1/sqrt(20) = 0.224. The spread of
      this chain is the model's variability, so refining the grid changes the model; it is
      not a convergence parameter (use sharp for transport without spread).
```

```python
Path("transit_erlang.tq").write_text(erlang.to_text())
mean, sd = transit_times(tq.load("transit_erlang.tq", cells={"z": 40}))
print(f"erlang, 40 cells: mean {mean:.3f}, sd {sd:.3f}, CV {sd / mean:.3f}")
```

```text
erlang, 40 cells: mean 1.500, sd 0.237, CV 0.158
```

The CV is 1/√n, as for an Erlang distribution with n stages: 0.224 on 20 cells, 0.158 on 40
(and 0.316 on 10). So with `erlang` **the number of
cells is part of the model**, as the number of transit compartments is in a Friberg or a
Savic absorption model: refining the grid does not converge to anything, it changes the
spread of the delay. That spread is often what you want: it is the between-cell variability
of maturation times, of gut transit, of lifespans. Choose n for that spread (or estimate it),
and write the axis with cells of width 1 so each cell is one compartment
([Tutorial 1](../tutorials/01_transit_compartments.ipynb)).

**`sharp`** (`using sharp` at the end of the equation, `scheme="sharp"` in Python) is a
second-order scheme (MUSCL with the van Leer limiter): a straight line in each cell, its slope
limited so no new peaks or negative values appear. Fronts keep their shape, and refining the
grid converges to transport without spread, as the study above shows. Use it when the axis is
a clock and the delay should be (nearly) exact: infection age with a latent period, time
since a dose, a plaque front.

`sharp` costs more: the solver follows the front from cell to cell, so its steps are at most
about one cell crossing time (`m.diagnostics()` reports it). Measured in
[the benchmarks](../benchmarks.md) (scipy BDF, 24 h): the toy phage model with 30 × 40
cells took 7.25 s with `sharp` against 0.67 s with `erlang`; the 40-cell phage latency model
0.63 s against 0.22 s.

| | `erlang` | `sharp` |
|---|---|---|
| what it is | a chain of transit compartments, exactly | transport, with the grid's spread shrinking as cells are added |
| refining the grid | changes the model (CV 1/√n) | converges |
| cost | low | higher: steps about one cell crossing time |
| use for | transit chains, maturation, any delay whose spread is part of the model | delays that should be sharp, fronts |

On a field with several axes, `using sharp on a` chooses `sharp` along `a` only.

**With diffusion.** When an equation has both transport and diffusion along one axis,
`erlang` adds its own spread, about |v| × width ÷ 2, on top of the diffusivity `D`. The
**cell Péclet number** |v| × width ÷ D measures the ratio; above 2, the grid's spread is larger
than `D`'s, and `m.diagnostics()` says so. Use more cells or `sharp` if `D` is meant to set the
spread. `m.fokker_planck` uses `sharp` by default for this reason.

## Tolerances

The solver chooses its own time steps, so that the error it makes in each step stays below
about `atol + rtol × |value|` for every state:

- **`rtol`** (relative tolerance, default 1e-6): the error allowed as a fraction of the value.
  1e-6 is six significant digits.
- **`atol`** (absolute tolerance, default 1e-6): the error allowed in absolute terms, in the
  units of the state. It sets the size below which a value is not resolved: a population of
  1e-7 cells per mL, at `atol = 1e-6`, is noise to the solver.

Both are given to `solve`, `simulator` or `simulate`: `m.solve((0, 24), rtol=1e-8,
atol=1e-9)`. `atol` may also be one number per state.

`atol` matters most for populations that fall to very small numbers and might grow back:
bacteria under an antibiotic, a resistant strain seeded by mutation. Here bacteria are killed
at rate 4 for 10 h while they grow at rate 1, then grow back. At t = 10 the exact count is
1e6·e⁻³⁰ ≈ 9.4e-8, and at t = 40 it is back to exactly 1e6:

```python
kill = tq.Model.from_text("""
model kill
param g=1, kmax=4, tstop=10
k_kill = kmax if t < tstop else 0
d_t(B) = g*B - k_kill*B
B(0) = 1e6
""")
t = np.linspace(0, 40, 81)
for atol in [1e-6, 1e-9]:
    sol = kill.solve((0, 40), times=t, atol=atol)
    print(f"atol {atol:g}: B(10) = {sol['B'][20]:.3g}, B(40) = {sol['B'][-1]:.3g}")
```

```text
atol 1e-06: B(10) = 1.36e-07, B(40) = -9.82e-09
atol 1e-09: B(10) = 9.39e-08, B(40) = 1.03e+06
```

At the default `atol` the population below 1e-6 is lost in the noise, and the run says the
bacteria never come back. At `atol = 1e-9` the same model regrows to 1.03e6, close to the
exact 1e6 (and within 0.03 % of it at `atol = 1e-12`). The first run is not a solver
failure: it simply did not resolve what happened below `atol`.
`m.diagnostics(sol)` points it out ("its total fell below atol … and stayed there"). There are
two honest fixes:

- if a fraction of a cell is meaningless (it is: 9.4e-8 bacteria per mL is no bacterium),
  declare an [extinction group](#extinction-groups-and-noise-floors), and the population is set
  to 0 when it falls below one cell;
- if small numbers are real (a large volume, a concentration), set `atol` well below the
  smallest value that matters.

**Large doses.** A dose that raises a state by a large amount leaves rounding errors of about
1e-14 times the rise in the states it feeds. If `atol` is below that, the solver's error
estimate for those states is noise, and BDF fails or survives by chance. For a 1e11 phage
bolus that is 1e-3. `m.diagnostics(sol)` warns about it and names the `atol` to use, and a
failed run says so in its message ([Troubleshooting](troubleshooting.md)).

**Too tight.** A tolerance near the limit of rounding for a fast process can make the solver
crawl with tiny steps. Both solvers stop after `max_steps` steps between two doses or events
(default 100,000) with a message instead of crawling on; loosen `rtol` to about 1e-8.

## The two solvers

Both solvers are BDF (backward differentiation formulas), the standard method for stiff
systems: systems with processes on very different time scales, such as fast adsorption and
slow growth, or many small cells. Both give the same answers (they agree to about 1e-13 on the
benchmark models) and take the same steps.

```python
pk = tq.Model.from_text("""
model pk
param ka=1, CL=4, V=30
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
dose depot at t=0 to 72 every 12: 100
observable conc = centr/V
""")
times = np.linspace(0, 96, 97)
a = pk.solve((0, 96), times=times)                      # scipy (default)
b = pk.solve((0, 96), times=times, backend="jax")       # the JAX BDF
print(np.max(np.abs(a["conc"] - b["conc"])) / np.max(a["conc"]))
```

### scipy BDF (the default)

- scipy's BDF with TenQSP's sparse Jacobian (below). Fast for one run at any size, up to
  10^5 states ([benchmarks](../benchmarks.md): 100,489 states, 24 h, in about 230 s).
- Doses split the run into segments. Each segment starts afresh at the post-dose state, with a
  first step never below 1e-8 of the segment's length.
- If BDF fails on a segment, that segment is retried once with Radau, an implicit Runge–Kutta
  method; the retry is reported in `sol.messages` and by `m.diagnostics(sol)`.
  `method="Radau"` uses Radau throughout.
- `max_steps` (default 100,000) per segment.

### JAX BDF (`backend="jax"`)

- TenQSP's own port of scipy's BDF, written in JAX: the same steps, inside one compiled
  program. It is the solver behind [gradients](gradients.md) and batched runs.
- **Compiled once** per model structure (1–15 s in the benchmarks). After that, new
  parameter values, dose amounts and times, and output times never rebuild it. Small models
  then run faster than on scipy (2–8× up to roughly 250 states, once compiled).
- **Two tiers**, chosen by size. Up to 250 states (500 when JAX runs on a GPU) it solves its
  linear systems with a dense LU inside JAX. Above, it calls SuperLU, the sparse LU scipy uses
  (the "sparse tier"); a solve there takes 1.5 to 2.6 times as long as on scipy
  ([Performance](../background/performance.md)).
- **No Radau fallback.** A failing run raises `tq.SolverError` saying where it failed and
  suggesting `backend="scipy"`.
- **Slots.** JAX needs fixed array sizes, so the number of doses, infusions and output times is
  rounded up to a power of two (at least 8). A run with 9 doses after runs with 8 builds once
  more, then not again.

Use scipy for single runs, large models and models with steep fronts; use JAX for many runs of
a small model (scans, populations), for derivatives, and for batches.

## The sparse Jacobian

An implicit solver such as BDF needs the **Jacobian**: how fast each state's rate of change
responds to each state. In a structured model almost all of it is zero: a cell is affected by
its own state, its neighbours along each axis, its categories, and the compartments it reads.
TenQSP derives this pattern from the model's equations and computes the non-zero entries
exactly (with JAX derivatives, not finite differences), grouping columns that never touch the
same rows so one pass computes many of them. The number of passes does not grow with the
grid. The solver then factorises the sparse matrix with a sparse LU.

`m.compile().summary()` shows the pattern's size, the passes and the memory:

```python
print(m.compile().summary().splitlines()[-2])
```

```text
  Jacobian: sparse, 40 non-zeros of 21 x 21; 5 forward and 2 reverse passes + 1 for the
  compartments (2 integral values lifted); one evaluation 1.0 ms
```

You never write or check it yourself. It is what lets a model of 10^5 states fit in memory: a
dense Jacobian of that size would need 80 GB. Compartments that read whole fields (totals,
integrals) depend on many states; their rows are computed exactly by reverse-mode
derivatives, one pass each.

## Extinction groups and noise floors

### Extinction groups

An ODE population never reaches zero: under killing it falls to 1e-3, 1e-8, 1e-20 cells, and
whatever is left grows back as soon as the killing stops. Biologically, a fraction of a
bacterium is no bacterium; numerically, below `atol` the solver does not resolve it anyway
(see [Tolerances](#tolerances)). An **extinction group** makes the choice explicit: when the
group's total falls below a threshold, every member is set to 0.

```tq
extinction B below 1                              # B, a compartment or a whole field
extinction B[s=R] below 1                         # one strain of a field B(z, s)
extinction lineage1: P1, I1 below 1               # a named group of several members
```

In Python: `m.extinction(B, threshold=1)`, `m.extinction(B.at(s="R"), threshold=1)`,
`m.extinction([P1, I1], threshold=1, name="lineage1")`.

- The **total** is the sum of the members' amounts (densities times cell widths, plus
  compartment values), so the members must be counted in the same units.
- A group is **armed** once its total has reached the threshold. An armed group falling
  below it is set to zero at the exact crossing, and the solve restarts there.
- A group that has **never reached** the threshold is never touched. Mutants seeded from zero
  can spend hours below one cell and still grow.
- One group per strain: each strain dies on its own.
- What was removed is booked in the ledger under `extinction`, and `sol.extinctions` lists
  `(time, group)` for each.

```python
kill_ext = tq.Model.from_text(kill.to_text() + "\nextinction B below 1\n")
sol = kill_ext.solve((0, 40), times=t)
print(sol.extinctions)               # [(4.605..., 'extinct_B')]
print(sol["B"][-1])                  # 0.0
print(sol.ledger.table("B"))
```

The bacteria cross one cell at t = ln(1e6)/3 ≈ 4.6 h and stay at 0: the answer no longer
depends on `atol`.

### Ghost populations and the noise floor

An implicit solver solves a system of equations at every step, to within rounding. A state
that should be exactly 0 can come out as 1e-12 or 1e-15: a **ghost**. Most ghosts do nothing.
But a ghost that feeds itself grows like a real population: a ghost phage infects a bacterium,
the infected cell bursts into fifty ghost phage, and so on. From 1e-12, a few hours of
exponential growth make a real epidemic, in an arm that was never given any phage.

The model below starts with a ghost of 1e-12 phage and nothing else:

```python
ghost_text = """
model ghost
param g=1, K=1e9, k=1e-9, beta=50, lam=2, delta=0.1
d_t(B) = g*B*(1 - B/K) - k*B*P
d_t(I) = k*B*P - lam*I
d_t(P) = beta*lam*I - k*B*P - delta*P
B(0) = 1e6
P(0) = 1e-12
"""
t48 = np.linspace(0, 48, 97)
for extra in ["", "extinction lineage: P, I below 1",
              "extinction lineage: P, I below 1 with noise_floor"]:
    sol = tq.Model.from_text(ghost_text + extra).solve((0, 48), times=t48)
    print(f"{extra or 'no group':52s} B(48) = {sol['B'][-1]:9.3g}, max P = {sol['P'].max():.3g}")
```

```text
no group                                             B(48) =  2.64e-08, max P = 4.52e+10
extinction lineage: P, I below 1                     B(48) = -1.18e-08, max P = 4.52e+10
extinction lineage: P, I below 1 with noise_floor    B(48) =     1e+09, max P = 1e-12
```

Without a floor the ghost grows to 4.5e10 phage and wipes out the bacteria. An extinction
group alone does not help: the ghost never falls below 1, it grows through it, and from then
on it counts as a real population.

With `with noise_floor` (`noise_floor=True` in Python), while the group's total is below the
threshold **every equation sees the whole group as zero**. The ghost cannot infect, so it
cannot grow by feeding itself, and the bacteria grow to their capacity. Outputs and the ledger
still report the state as it is (here 1e-12): nothing is clipped.

Sources from outside the group still add to it, and below the threshold the group does not
see its own losses either. So a group fed from outside (phage infused from blood into a site)
fills up at the feed rate until it reaches the threshold; a feed too weak to hold it there
gives repeated fill-and-extinction cycles. Validation warns about such a feed: put the source
in the group (the blood compartment of a phage lineage), or leave out the noise floor.

```tq
extinction lineage1: P1, I1 below 1 with noise_floor    # phage 1 and the cells it infected
```

A phage lineage is its free phage and the cells it has infected, since each makes the other.
`m.explain()` lists every group and floor; extinction groups and noise floors work the same on
both backends.

## Threads

The sparse LU that both solvers use (SuperLU) calls BLAS. With OpenBLAS's default of one
thread per core, its idle threads spin and slow the whole machine for no gain. TenQSP
therefore runs BLAS on one thread while it factorises and solves, and restores your setting
afterwards. `tenqsp._threads.BLAS_THREADS = None` turns this off.

For several runs at once, use several processes (`sim.batch(..., n_jobs=8)`, see
[Simulation and exploration](simulation.md)) rather than threads.

## Measured speeds

Every timing quoted in this guide comes from [the benchmarks](../benchmarks.md), with the
machine and versions they were measured on. Some landmarks (scipy BDF, `erlang`, 24 h unless
stated):

| model | states | solve |
|---|---:|---:|
| transit chain, 5 cells | 7 | 0.04 s |
| phage latency, 40 age cells | 42 | 0.22 s (`sharp`: 0.63 s) |
| whole-body PBPK, 9 organs × 3 spaces | 30 | 0.09 s |
| toy phage, 30 × 40 cells | 1,261 | 0.67 s (`sharp`: 7.25 s) |
| toy phage, 100 × 100 cells, 72 h | 10,201 | 11.8 s |
| toy phage, 316 × 316 cells | 100,489 | 230 s, 0.77 GB |

The first solve of a model also builds it (derives the Jacobian's pattern and compiles the
right-hand side): in the benchmarks this added about half a second for small models and a
few seconds at 10^4 states.
