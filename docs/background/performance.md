# Performance

How long will my model take? This page gives measured answers: one solve at sizes from a few
states to 10^5, on each solver; what a gradient costs; how many parameter sets a batch runs
per second on a CPU and on a GPU; how much memory it all needs; and what compiling costs. It
ends with advice on which solver to use when.

## How the numbers were measured

- **Machine:** a Linux workstation with a 64-core CPU, JAX on the CPU; the GPU runs on an
  NVIDIA RTX A6000. Python 3.12, JAX 0.11.2, scipy 1.18.1, numpy 2.5.
- **Settings:** `rtol = atol = 1e-6`, the ledger on, unless stated. Times are wall-clock
  seconds, the best of three runs after a first one; the first run, which also builds or
  compiles the model, is shown apart.
- **Models:** the **toy phage model** (bacteria and phage, with infected cells on a grid of
  depth × infection age, "30 z × 40 a" meaning 30 × 40 cells, plus compartments), with the
  `erlang` scheme unless stated, 24 hours; and models from the gallery (a transit chain,
  Friberg myelosuppression, whole-body PBPK, HIV infection age).
- Each case ran in its own process, one after another. Some runs shared the machine with
  other work; where that was so, times may be inflated by up to a few tens of percent.

Your times will differ with your machine, but the ratios (between sizes, schemes, solvers)
carry over.

## One solve, scipy (the default)

| Model | States | Solve (s) | First solve, with building (s) |
|---|---:|---:|---:|
| transit chain, 5 cells | 7 | 0.04 | 0.7 |
| Friberg myelosuppression, 600 h | 6 | 0.03 | 0.9 |
| whole-body PBPK, 9 organs × 3 spaces | 30 | 0.09 | 1.3 |
| phage latency, 40 age cells | 42 | 0.22 | 0.9 |
| HIV infection age, 60 cells, 60 days (`rtol` 1e-8) | 62 | 0.31 | 1.3 |
| three-axis toy, 24 a × 16 z × 2 genotypes, 12 h | 802 | 2.7 | 4.8 |
| toy phage 30 z × 40 a | 1,261 | 0.67 | 2.0 |
| toy phage 60 z × 60 a | 3,721 | 1.9 | 3.8 |
| toy phage 100 z × 100 a, **72 h** | 10,201 | 11.8 | 14.9 |

Larger grids of the toy phage model (24 h):

| States | Solve (s) | Peak memory (GB) | One Jacobian (ms) | Jacobian passes |
|---:|---:|---:|---:|---:|
| 1,261 | 0.77 | 0.43 | 1.8 | 10 |
| 3,721 | 2.2 | 0.45 | 3.2 | 10 |
| 10,201 | 10.6 | 0.49 | 8.6 | 10 |
| 29,241 | 41 | 0.55 | 16 | 10 |
| 100,489 | 230 | 0.77 | 47 | 10 |

What this says:

- **Small models are fast, and their time is mostly overhead.** One call of the right-hand
  side costs about 0.2 ms at every size up to 10^4 states; it is the cost of calling into JAX,
  not of the arithmetic.
- **Time grows a little faster than the number of states.** 10^5 states solve in under four
  minutes in less than 1 GB. At that size most of the time goes into the sparse LU
  factorisations.
- **The Jacobian is cheap and its cost does not explode.** The number of passes stays at 10
  at every size, and the solver needed only a handful of Jacobians per run.
- **Building** (deriving the Jacobian's pattern and compiling) adds about half a second to a
  few seconds to the first solve of a model, once.
- **The ledger costs nothing measurable.** The toy model with `sharp` took 17.3 s with it and
  18.5 s without, within run-to-run noise.
- **Writing and reading the notation, `explain()` and the other reports take milliseconds**,
  whatever the grid size.

## `sharp` against `erlang`

The `sharp` scheme keeps fronts sharp, and pays for it with more steps and more Jacobians
(see [The numerical method](method.md#sharp-muscl-with-the-van-leer-limiter)):

| Model | `erlang` | `sharp` | Ratio |
|---|---:|---:|---:|
| phage latency, 40 age cells (42 states) | 0.22 s, 4 Jacobians | 0.63 s, 48 Jacobians | 2.9 |
| toy phage 30 z × 40 a (1,261 states) | 0.67 s, 3 Jacobians | 7.25 s, 374 Jacobians | 10.8 |

A steep travelling front is the hardest case: phage plaque fronts on 240 × 40 cells (10,080
states, 10 h) took 38.5 s on scipy and about 19,500 right-hand-side calls, against 2,249 calls
for the smooth 10^4-state toy model over 72 h. Use `sharp` where a sharp delay is part of the
question, and `erlang` elsewhere ([Numerical choices](../guide/numerics.md#schemes-erlang-and-sharp)).

## One solve, JAX

The JAX solver takes the same steps as scipy and gives the same answers (to 1e-13 or better on
these models). What differs is the cost. Up to 250 states it factorises with a dense LU inside
JAX (the **dense tier**); above, with the same sparse LU as scipy (the **sparse tier**).

| Model | States | scipy (s) | JAX, dense tier (s) | JAX, sparse tier (s) | First JAX solve, with compiling (s) |
|---|---:|---:|---:|---:|---:|
| transit chain, 5 cells | 7 | 0.047 | 0.006 | | 1.1 |
| phage latency, 40 age cells | 42 | 0.23 | 0.061 | | 1.4 |
| toy phage 10 z × 12 a | 141 | 0.20 | 0.13 | 0.52 | 2.2 |
| toy phage 16 z × 20 a | 353 | 0.32 | 0.61 | 0.72 | 2.8 |
| toy phage 22 z × 30 a | 705 | 0.49 | 3.2 | 0.99 | 3.2 |
| toy phage 24 z × 38 a | 961 | 0.62 | 8.2 | 1.27 | 3.4 |
| toy phage 40 z × 60 a | 2,481 | 1.40 | | 3.05 | 5.6 |
| toy phage 60 z × 94 a | 5,761 | 3.59 | | 7.30 | 9.7 |
| toy phage 80 z × 125 a | 10,161 | 8.97 | | 13.5 | 15.7 |

(The first-solve column is for the tier the size picks. The 7- and 42-state rows come from an
earlier run than the others; the solver did not change between them.)

- **Small models run 1.5 to 8 times faster on JAX** once compiled (the smaller, the larger
  the gain): there is no Python between the steps.
- **The dense tier loses badly above a few hundred states**: a dense LU's cost grows with the
  cube of the size. At 961 states it is 13 times slower than scipy. That is why the switch to
  the sparse tier is at 250 states.
- **The sparse tier costs 1.5 to 2.6 times scipy's time**: the extra is the crossing between
  JAX and the host for each factorisation.
- **Compiling costs 1 to 16 s**, once per model structure. For a single run that is more than
  the run; it pays off over many runs.
- A steep front makes the crossings add up: the 10,080-state plaque-front model took 75 s on
  JAX against 38.5 s on scipy.

## Gradients

A gradient or a table of sensitivities comes from `sim.function` and `sim.sensitivity`
([Gradients and sensitivities](../guide/gradients.md)). Measured with every parameter in
`wrt` and every state as an output, JAX solver:

| Model | States | Parameters | Forward solve (s) | `f(x)` (s) | `jax.jacfwd` (s) | `jax.grad` (s) | Gradient ÷ forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| two-compartment oral PK, 2 doses | 3 | 5 | 0.016 | 0.034 | 0.037 | 0.065 | 4.0 |
| Friberg myelosuppression, at rest | 6 | 5 | 0.003 | 0.011 | 0.016 | 0.022 | 7.2 |
| phage latency, 40 age cells | 42 | 8 | 0.036 | 0.074 | 0.14 | 0.15 | 4.3 |
| toy phage 10 z × 12 a | 141 | 12 | 0.060 | 0.12 | 0.58 | 0.51 | 8.6 |
| toy phage 16 z × 20 a | 353 | 12 | 0.42 | 0.81 | 3.6 | 2.8 | 6.8 |

- **One call `f(x)` costs about two forward solves**: the run, then its replay (more on a
  model at rest, where the steps are chosen for the derivative).
- **A reverse-mode gradient costs 4 to 9 forward solves, whatever the number of
  parameters.** `jax.jacfwd` grows with the number of parameters; with a few parameters on a
  small model it is the cheaper of the two.
- **Steering the steps for the derivative** costs 12 to 18 % more steps on models whose
  derivatives move with the solution. On the Friberg model at rest it raised the steps from 15
  to 89, and made the derivative right (it was 6 % off without).
- Forward and reverse mode agreed to 1e-15 or better.

On the sparse tier, larger models (the gradient of a sum of squares of log phage at 12 times,
12 parameters):

| States | scipy forward (s) | `jax.grad` (s) | Gradient ÷ scipy forward | Building `f` and compiling the gradient (s) | Peak memory (GB) |
|---:|---:|---:|---:|---:|---:|
| 141 | 0.20 | 1.4 | 7.3 | 14 | 1.0 |
| 961 | 0.62 | 6.0 | 9.7 | 20 | 2.0 |
| 2,481 | 1.40 | 13.9 | 9.9 | 31 | 2.4 |
| 5,761 | 3.59 | 34 | 9.5 | 56 | 4.2 |
| 10,161 | 8.97 | 66 | 7.4 | 97 | 6.7 |

- **A gradient of a 10^4-state model takes about a minute**, 5 times the JAX forward solve
  and 7 times scipy's.
- **Memory is what limits gradients on large models.** The run's record (every step, kept for
  the replay) grows with the number of states and steps: 6.7 GB at 10,161 states. The record has
  room for twice the steps of the first run above 2,000 states, four times below.
- **Compiling the gradient** takes from a quarter of a minute to a minute and a half,
  growing with size, once per model structure.

## Batches: many parameter sets in one call

`jax.vmap` over a function from `sim.function` runs many parameter sets (virtual patients,
a sweep) in one call. Toy phage model, every parameter varied by ±20 %:

**On a CPU.**

| States | What | Members | One call per member (s each) | Batch (s per member) |
|---:|---|---:|---:|---:|
| 141 | forward | 64 | 0.15 (dense) | 0.15 (dense), 0.16 (sparse) |
| 353 | forward | 64 | 0.64 (dense) | 0.29 (sparse) |
| 505 | forward | 64 | 0.95 (sparse) | 0.40 (sparse) |
| 505 | forward | 1,000 | 0.95 (sparse) | 0.33 (sparse): 333 s in all, 4.1 GB |
| 141 | with derivatives | 64 | 0.15 (dense) | 0.31 (sparse) |
| 353 | with derivatives | 64 | 1.14 (dense) | 0.63 (sparse) |

- **A batch runs on one core.** On the dense tier it costs about what one call per member
  costs; on the sparse tier, forward runs were 2.4 to 3.5 times faster than one sparse call per member,
  and above 250 states it beats the dense batch by 2.7 to 5 times.
- **Spread over cores** (`jax.shard_map`), 64 members on 64 cores ran 7 times faster than
  one call per member (141 states, with derivatives: 1.3 s for all 64).
- For runs that do not need JAX, `sim.batch(..., n_jobs=...)` spreads scenarios over
  processes on the scipy solver ([Simulation and exploration](../guide/simulation.md)).

**On a GPU** (RTX A6000, forward runs unless stated):

| States | Members | Tier | Batch (s per member) | Batch (s) |
|---:|---:|---|---:|---:|
| 141 | 64 | dense | 0.019 | 1.2 |
| 141 | 1,024 | dense | 0.011 | 11.3 |
| 353 | 64 | dense | 0.077 | 4.9 |
| 505 | 1,000 | dense | 0.155 | 155 |
| 601 | 1,000 | dense | 0.253 | 253 |
| 913 | 64 | sparse | 0.389 | 24.9 |
| 141, with derivatives | 64 | dense | 0.046 | 2.9 |
| 353, with derivatives | 64 | dense | 0.176 | 11.2 |

- **A GPU pays off for batches of small and middle-sized models.** At 141 states it ran a
  member in 0.011–0.019 s against 0.15 s on one CPU core; at 505 states, 1,000 members took
  155 s against 333 s on the CPU.
- **On a GPU the dense tier wins up to about 550 states** (64 members; about 650 for 1,000),
  so there the switch to the sparse tier is at 500 states. Above that the sparse tier's
  factorisations run on the CPU anyway, and the GPU's advantage shrinks.
- **Memory with derivatives is the record of every member**: 20 GB for 64 members at 353
  states on the CPU's sparse tier (10 GB on the dense tier). Without derivatives
  (`derivatives=False`) the same batches need 1 to 4 GB.
- No member of any batch failed.

## Compiling

| What | Once per model structure |
|---|---|
| scipy: deriving the Jacobian's pattern and compiling | about 0.5 s for small models, a few seconds at 10^4 states |
| JAX solve, dense tier | 1 to 9 s |
| JAX solve, sparse tier | 2.5 to 16 s (grows with size) |
| `f`, `jax.jacfwd`, `jax.grad` on small models | 2.5 to 10 s each |
| building `f` and compiling its gradient, sparse tier | 14 s at 141 states to 97 s at 10,161 |

New parameter values, dose amounts and times, and output times never recompile: a hundred
solves of the transit model with new parameters and dose amounts took 4.6 s in total. On the
JAX solver the number of doses and output times is rounded up to a power of two, so a run with
9 doses after runs with 8 compiles once more ([Numerical choices](../guide/numerics.md#the-two-solvers)).

### Many subjects

`sim.function(scenario, wrt=...)` is built for the scenario's *shape*: how many doses go into
each target, and how many output times there are. A new scenario of the same shape reuses the
build; a new shape builds again, at the costs in the table above. In a population dataset,
where subjects have different dosing histories and sampling times, that can dominate.

The remedy is to give every subject the same shape: pad its regimen with doses of amount 0,
and its output times with times whose outputs are not used. A dose of amount 0 or an extra
output time changes nothing in the model, but it adds a point where the solver stops, so its
steps change. The results therefore agree **to the solver's tolerance, not to round-off**: in
the test of this page's claim (a two-dose oral model padded with two empty doses and two extra
times), by about 0.1 x `rtol`, values and derivatives alike, on both solvers. Choose `rtol`
with that in mind.

## Which solver when

- **One run, or a few, of any size: scipy** (the default). It is the fastest single run above
  about 250 states, it scales to 10^5 states in under 1 GB, and it has the Radau fallback.
- **Many runs of a small model (below about 250 states): JAX.** Once compiled, it is up to 8
  times faster per run; with dozens of runs or more the compilation is paid back.
- **Derivatives: JAX** (`sim.function`, `sim.sensitivity`). Expect a gradient to cost 4 to 10
  forward solves, and budget memory on large models: several GB at 10^4 states.
- **Populations and sweeps:** without derivatives, `derivatives=False`; on a CPU, batch on the
  sparse tier above 250 states, or spread members over cores. `sim.batch(n_jobs=...)` on
  scipy is the simple choice when no JAX function is needed.
- **When a GPU helps:** large batches (hundreds of members or more) of models up to a few
  hundred states. It does not help a single run, a large model (above about 500 states the
  factorisations go back to the CPU), or a model with steep fronts.
- **Steep fronts (`sharp`, plaques): scipy.** They need many steps, and every step on the
  JAX sparse tier crosses to the host.
- **The slow part is usually the scheme or the tolerance, not the solver.** `sharp` costs 3 to
  11 times `erlang`; a tolerance near rounding makes the solver crawl. See
  [Troubleshooting](../guide/troubleshooting.md#the-run-is-slow).
