# Benchmarks

*A developer record: measurements taken at the end of each phase of development, with the
machine and versions used. The "targets" are those of the project's original design
specification.*

Re-measured at the end of each phase. Run alone on an idle machine:
`python benchmarks/phase1.py`, `python benchmarks/phase2_scale.py`. Times are wall-clock seconds on the development machine (64 cores,
CPU JAX).

## Phase 1 gate (2026-10-11, commit after `3e88501`)

scipy BDF, `rtol = atol = 1e-6`, t = 0 → 24 h, ledger on. "Solve" is the best of 3 solves
after the first, which includes compilation. Dense `jacfwd` Jacobian handed to scipy as a
sparse matrix (Phase 1).

| model | states | ledger channels | first solve incl. compile (s) | solve (s) | RHS call (ms) | RHS evals | Jacobians | LU | ledger unexplained |
|---|---|---|---|---|---|---|---|---|---|
| transit chain, 5 erlang cells | 7 | 5 | 0.4 | 0.04 | 0.446 | 386 | 1 | 37 | 8e-16 |
| phage latency, 40 age cells, sharp | 42 | 8 | 1.4 | 0.71 | 0.212 | 5,179 | 50 | 344 | 1e-15 |
| phage latency, 40 age cells, erlang | 42 | 8 | 0.7 | 0.18 | 0.208 | 1,509 | 4 | 125 | 2e-15 |
| toy phage 30 z x 40 a, sharp | 1,261 | 9 | 17.6 | 15.80 | 0.274 | 6,411 | 383 | 923 | 1e-15 |
| toy phage 30 z x 40 a, erlang | 1,261 | 9 | 1.7 | 0.76 | 0.330 | 1,314 | 2 | 108 | 4e-16 |
| toy phage 60 z x 60 a, erlang | 3,721 | 9 | 3.5 | 2.34 | 0.292 | 1,434 | 3 | 127 | 8e-16 |

Python 3.12.15, JAX 0.11.2, scipy 1.18.1, numpy 2.5.3.

Reading the table:

- **One right-hand-side call costs ~0.2–0.4 ms at every size here.** That is JAX dispatch
  overhead, not arithmetic. Small models are dominated by it.
- **`sharp` vs `erlang`: about 20× slower on the toy model** (15.8 s vs 0.76 s) and 4× on the
  phage model. The sharp front of newly infected cells moves one age cell every 0.04 h. BDF
  follows it with small steps and frequent Jacobians (383 vs 2). A smooth limiter only halved
  the cost and Radau was slower (39 s). The cost is the price of an undispersed age profile.
- **The ledger costs nothing measurable.** On the toy model, sharp, it took 17.3 s with the
  ledger and 18.5 s without, which is within run-to-run noise.
- **Compile once:** 100 solves of the transit model with new parameters and dose amounts take
  4.6 s in total and compile each function once.
- Once, during the 3,721-state run, scipy printed `RuntimeWarning: invalid value encountered
  in subtract` from inside BDF. Cause (found later): scipy allocates its BDF difference table
  with `np.empty` and, on the first step, subtracts a row it has not written yet. Whether
  numpy warns depends on what was left in that memory, which is why the warning depended on
  what had run earlier in the process. The row is overwritten before use, so results are
  unaffected. The backend now silences exactly this message from exactly that module.

Phase 1 limit: the dense Jacobian was refused above 5,000 states. Lifted by the sparse
Jacobian of step 2.1 (below).

## Phase 2 step 2.1: sparse Jacobian at 10^3 – 10^5 states (2026-10-12)

`benchmarks/phase2_scale.py`: the toy phage model with the `erlang` scheme (as in ADR 0003),
scipy BDF, `rtol = atol = 1e-6`, t = 0 → 24 h, ledger on. Each size ran in its own process,
one after another, on an otherwise idle machine. The Jacobian is built from the model's
structure (coloured forward passes, integrals lifted, compartment and ledger rows by reverse
mode).

| states | forward passes | reverse passes | Jacobian non-zeros | structure (s) | one Jacobian (ms) | solve (s) | RHS evals | Jacobians | LU | peak memory (GB) | ledger unexplained |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,261 | 10 | 3 | 11,305 | 0.19 | 3.0 | 0.74 | 1,314 | 2 | 108 | 0.44 | 2e-15 |
| 3,721 | 10 | 3 | 36,567 | 0.23 | 3.6 | 2.0 | 1,434 | 3 | 127 | 0.45 | 1e-15 |
| 10,201 | 10 | 3 | 101,341 | 0.37 | 6.5 | 9.3 | 1,782 | 4 | 160 | 0.49 | 1e-15 |
| 29,241 | 10 | 3 | 290,949 | 0.77 | 19 | 41.7 | 2,197 | 3 | 195 | 0.57 | 7e-16 |
| 100,489 | 10 | 3 | 1,002,817 | 2.45 | 58 | 225 | 2,700 | 3 | 245 | 0.87 | 2e-15 |

Reverse passes: 3 for the integrals (`dq/dy`, rows with disjoint support share a pass), plus
one per compartment and per ledger channel (10 here). "Structure" is the one-off cost of
deriving the patterns and colourings, on the first solve.

Reading the table:

- **The number of passes does not grow with the grid** (10 forward at every size), as ADR 0003
  required. One Jacobian at 10^5 states takes 58 ms, and the solver needed only 3.
- **10^5 states solve in under 4 minutes using 0.87 GB.** ADR 0003's hand-written spike took
  171 s and 2.4 GB for a similar model. As there, the sparse LU factorisations dominate.
- The targets ask for a 72 h simulation of 10^4 states in under 30 s. A 24 h simulation takes 9.3 s
  here.
- The first solve of a new model now compiles two Jacobian functions (about 1 s together)
  instead of one dense `jacfwd`.

## Phase 2 gate (2026-10-21, commit after `3189fec`)

Each case in its own process, one after another (`benchmarks/phase2_gate.py k`,
`benchmarks/phase1.py`, `benchmarks/phase2_scale.py n_z n_a`). scipy BDF, `rtol = atol = 1e-6`
unless stated, ledger on; sparse Jacobian from the model structure. The machine was not fully
idle: two other processes (an IDE kernel server and Monolix) used about 1.4 of 64
cores throughout.

**Performance targets**

| target | measured | |
|---|---|---|
| RHS evaluation, 10^4 states, jitted, CPU: < 1 ms | 0.23 ms (10,201 states) | met |
| Jacobian colours for 3-point stencils + compartments: <= ~30, independent of grid size | 10 at every size from 1,261 to 100,489 states | met |
| 72 h simulation, 10^4 states, scipy BDF: < 30 s | 11.8 s (10,201 states, `erlang`) | met |
| >= 10^5 states on the scipy sparse path | 100,489 states, 24 h in 230 s, 0.77 GB | met |

**Phase 2 models** ("solve" is the best of 3 after the first, which compiles; colours =
forward passes for one Jacobian)

| model | states | colours | first solve incl. compile (s) | solve (s) | RHS call (ms) | RHS evals | Jacobians | LU | peak memory (GB) | ledger unexplained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| target: toy phage 100 z x 100 a, erlang, 72 h | 10,201 | 10 | 14.9 | 11.83 | 0.229 | 2,249 | 8 | 194 | 0.55 | 2e-15 |
| toy phage 30 z x 40 a, sharp, 24 h | 1,261 | 16 | 9.8 | 7.32 | 0.260 | 6,277 | 374 | 886 | 0.47 | 3e-15 |
| three-axis toy I[a, z, s] 24 a x 16 z x 2, 12 h | 802 | 19 | 4.8 | 2.69 | 0.207 | 3,155 | 117 | 355 | 0.47 | 2e-15 |
| two-phage toy B[r1, r2], 24 h | 6 | 7 | 1.2 | 0.19 | 0.162 | 1,830 | 5 | 128 | 0.39 | 9e-16 |
| whole-body PBPK 9 organs x 3 spaces, bolus, 24 h | 30 | 29 | 1.3 | 0.09 | 0.169 | 579 | 1 | 52 | 0.42 | 3e-16 |
| HIV infection age, 60 cells, 60 days (rtol 1e-8) | 62 | 5 | 1.3 | 0.31 | 0.170 | 2,625 | 2 | 181 | 0.38 | 2e-15 |
| Friberg myelosuppression, 600 h | 6 | 4 | 0.9 | 0.03 | 0.218 | 264 | 1 | 24 | 0.37 | 6e-16 |

**Phase 1 models, re-measured**

| model | states | ledger channels | first solve incl. compile (s) | solve (s) | RHS call (ms) | RHS evals | Jacobians | LU | ledger unexplained |
|---|---|---|---|---|---|---|---|---|---|
| transit chain, 5 erlang cells | 7 | 5 | 0.7 | 0.04 | 0.348 | 399 | 1 | 38 | 1e-15 |
| phage latency, 40 age cells, sharp | 42 | 8 | 1.8 | 0.63 | 0.184 | 5,281 | 48 | 346 | 1e-15 |
| phage latency, 40 age cells, erlang | 42 | 8 | 0.9 | 0.22 | 0.166 | 1,515 | 4 | 125 | 2e-15 |
| toy phage 30 z x 40 a, sharp | 1,261 | 9 | 9.7 | 7.25 | 0.227 | 6,277 | 374 | 886 | 3e-15 |
| toy phage 30 z x 40 a, erlang | 1,261 | 9 | 2.0 | 0.67 | 0.234 | 1,338 | 3 | 109 | 3e-15 |
| toy phage 60 z x 60 a, erlang | 3,721 | 9 | 3.8 | 1.85 | 0.199 | 1,444 | 3 | 128 | 2e-15 |

**Scale (toy phage, erlang, 24 h)**

| states | colours | reverse passes | Jacobian non-zeros | structure (s) | one Jacobian (ms) | solve (s) | RHS evals | Jacobians | LU | peak memory (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,261 | 10 | 3 | 11,245 | 0.19 | 1.8 | 0.77 | 1,338 | 3 | 109 | 0.43 |
| 3,721 | 10 | 3 | 36,447 | 0.24 | 3.2 | 2.2 | 1,444 | 3 | 128 | 0.45 |
| 10,201 | 10 | 3 | 101,141 | 0.37 | 8.6 | 10.6 | 1,827 | 6 | 162 | 0.49 |
| 29,241 | 10 | 3 | 290,609 | 0.80 | 16 | 41.2 | 2,211 | 3 | 196 | 0.55 |
| 100,489 | 10 | 3 | 1,002,185 | 2.43 | 47 | 230 | 2,731 | 3 | 246 | 0.77 |

Python 3.12.15, JAX 0.11.2, scipy 1.18.1, numpy 2.5.3.

Reading the tables:

- **Every performance target is met on the scipy path.** (The differentiable tier is Phase 4.)
- **The `sharp` toy model halved since Phase 1** (15.8 s -> 7.3 s, same 374 Jacobians):
  each Jacobian is now coloured and sparse instead of a dense `jacfwd`. `sharp` is still ~10x
  `erlang` on this model, for the reason given under Phase 1.
- **Colours stay at 10 for the 2-D toy at every size.** The three-axis toy needs 19 (age,
  depth and genotype couple within a cell), the PBPK model 29 of its 30 states: the arterial
  and venous pools touch every organ, so almost every column is coupled to the hubs. At 30
  states that costs nothing.
- Solve times at 10^3 – 10^5 states agree with step 2.1 to within run-to-run noise; the
  review fixes (event handling, noise-floor slots) added no measurable cost.

## Phase 3 gate (2026-10-23)

`python benchmarks/phase3_gate.py k`, one case per process, one after another. The machine was
**not idle**: other processes (an IDE kernel server, Monolix) kept the load average at
20–50, so times may be inflated. As a reference, the target case re-measured in the same
conditions: 10,201 states, 72 h, `erlang`: **12.2 s** (Phase 2 gate: 11.8 s), 2,249 RHS
evaluations, as before.

Times in milliseconds (best of 3). "lines" is the length of `to_text()`; "compare" is
`canonical.differences` between the Python model and the model read back from its text;
"gap" is the largest relative difference between the two models' solutions; "RHS evals"
compares the Python model with the text model.

| model | lines | write | read | compare | differences | explain | diagnostics | LaTeX | gap | RHS evals |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| toy phage 30 z x 40 a, sharp, 24 h | 47 | 0 | 6 | 6 | none | 5 | 3 | 1 | 0 | 6,277 / 6,277 |
| toy phage 100 z x 100 a, sharp, 1 h | 47 | 0 | 6 | 7 | none | 5 | 4 | 1 | 0 | 4,760 / 4,760 |
| three-axis toy I[a, z, s], 12 h | 52 | 0 | 9 | 7 | none | 8 | 3 | 1 | 0 | 3,155 / 3,155 |
| whole-body PBPK 9 organs x 3 spaces, 24 h | 61 | 0 | 8 | 2 | none | 4 | 1 | 2 | 5e-14 | 579 / 579 |
| Friberg myelosuppression, 600 h | 29 | 0 | 2 | 2 | none | 4 | 3 | 1 | 0 | 264 / 264 |

Python 3.12.15, JAX 0.11.2, scipy 1.18.1, numpy 2.5.3.

Reading the table:

- **A model read from text is the same model.** The solutions are bit-identical with the same
  number of RHS evaluations. PBPK is the exception, at 5e-14: rounding at the last digits,
  likely from a formula written in another but equivalent order (not investigated further).
- **The notation and the reports cost milliseconds**, against seconds for a solve. Reading
  doesn't grow with grid size (the 100 x 100 toy reads as fast as the 30 x 40), because the
  text has the same number of lines.

## Phase 4 step 4.3: the JAX BDF, dense tier (2026-10-24, commit after `7684b35`)

`python benchmarks/step4_3_jax.py`, one case per process, one after another, on an idle
machine (load average below 1 at the start). The `erlang` scheme, `rtol = atol = 1e-6`,
t = 0 → 24 h, 25 output times, ledger on. "JAX solve" and "scipy solve" are the best of 3
after the first. "JAX first solve" includes tracing and compiling the program; it is paid
once per model structure and slot size (15 runs with new
values and doses, no new build). "Largest difference" is the largest difference between the
two backends' outputs, relative to the largest value of each variable.

| model | states | JAX first solve incl. compile (s) | JAX solve (s) | scipy solve (s) | LU JAX / scipy | Jacobians JAX / scipy | largest difference | ledger unexplained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| transit chain, 5 erlang cells | 7 | 1.1 | 0.006 | 0.047 | 38 / 38 | 1 / 1 | 8e-16 | 1e-15 |
| phage latency, 40 age cells | 42 | 1.4 | 0.061 | 0.227 | 125 / 125 | 4 / 4 | 3e-14 | 2e-15 |
| toy phage 10 z x 12 a | 141 | 2.2 | 0.094 | 0.198 | 70 / 70 | 3 / 3 | 3e-15 | 2e-15 |
| toy phage 16 z x 20 a | 353 | 2.4 | 0.403 | 0.292 | 83 / 83 | 3 / 3 | 3e-14 | 2e-15 |
| toy phage 22 z x 30 a | 705 | 4.6 | 2.44 | 0.48 | 98 / 98 | 3 / 3 | 1e-13 | 1e-15 |
| toy phage 24 z x 38 a | 961 | 9.3 | 7.51 | 0.61 | 106 / 106 | 2 / 2 | 1e-13 | 2e-15 |

Python 3.12.15, JAX 0.11.2, scipy 1.18.1, numpy 2.5.3; CPU JAX.

Reading the table:

- **The two backends do the same work.** The LU and Jacobian counts are identical in every
  case, and the answers agree to 1e-13 or better: the JAX BDF is a port of scipy's, step for
  step.
- **Small models run 2–8× faster on JAX** once compiled (no Python between steps), up to
  roughly 250 states. Above that the dense LU takes over: its cost grows with the cube of the
  number of states, scipy's sparse LU roughly linearly. At 961 states JAX is 12× slower.
- **Compiling costs 1–9 s** per model structure, once. For a single run of a small model that
  is more than the run; it pays off over many runs (parameter scans, and the gradients and
  batches of steps 4.4 and 4.6, which is what the JAX backend is for).
- At 705 and 961 states XLA's dense LU spreads over all 64 cores (load average ~40, from the
  run itself). A hand-test overlapped the first four cases, so they were re-run alone; the
  numbers shown are the re-runs, which agreed with the first pass within noise.
- This is why the dense tier stops at 1,000 states (at the time; since step 4.5 the switch is
  at 250 states, 500 on a GPU). The sparse tier (step 4.5) is meant to close the gap above a
  few hundred states.

## Phase 4 step 4.4: what a derivative costs (2026-10-25, commit after `a548fe1`)

`python benchmarks/step4_4_gradients.py`, one case per process, one after another, on an
idle machine. Every parameter in `wrt`, the model's own doses, every state as output,
`rtol = atol = 1e-6`. Times are the best of 3 after a first call (which compiles; shown
apart). "forward" is one JAX solve without the ledger (steps for the solution only);
"f(x)" one call of `sim.function`: the run with steps steered by the derivative too
(design note `step4_4d_api.md` §6.3), then its replay; "jacfwd" every output by every
parameter (`sim.sensitivity`); "grad" the reverse-mode gradient of a sum of squares, an
estimator's objective. "forward = reverse": the gradient from jacfwd against grad.

| model | states | params | forward (s) | steps plain / steered | f(x) (s) | jacfwd (s) | grad (s) | grad / forward | compile f, jacfwd, grad (s) | forward = reverse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| two-compartment oral PK, 2 doses | 3 | 5 | 0.0123 | 226 / 265 | 0.0470 | 0.0705 | 0.0888 | 7.2 | 3.9, 2.5, 2.6 | 5e-16 |
| Friberg myelosuppression (at rest) | 6 | 5 | 0.0028 | 15 / 89 | 0.0106 | 0.0221 | 0.0315 | 11.2 | 4.7, 3.3, 3.2 | 4e-16 |
| phage latency, 40 age cells | 42 | 8 | 0.0347 | 702 / 841 | 0.1268 | 0.3593 | 0.5607 | 16.1 | 5.3, 3.8, 3.9 | 5e-16 |
| toy phage 10 z x 12 a | 141 | 12 | 0.0541 | 376 / 422 | 0.2442 | 1.6226 | 1.5230 | 28.1 | 7.6, 6.8, 6.6 | 2e-16 |
| toy phage 16 z x 20 a | 353 | 12 | 0.3971 | 456 / 512 | 0.8773 | 7.7230 | 5.6998 | 14.4 | 8.8, 13.1, 10.9 | 6e-16 |

Python 3.12.15, JAX 0.11.2, numpy 2.5.3; x86_64, CPU JAX.

Reading the table:

- **Steering costs 12–18 % more steps** on models whose derivatives move with the solution,
  and adds steps only where the derivative needs them: the Friberg model at rest goes from
  15 to 89 steps (and its derivative from 6 % off to right).
- **A gradient costs 7–28 forward solves** (reverse mode), whatever the number of
  parameters; **jacfwd** grows with the number of parameters and is the cheaper of the two
  for a few parameters on small models. Forward and reverse agree to round-off.
- **Where the time goes:** the forward run reuses one LU factorisation over many steps (70
  LUs for ~400 steps in the toy phage model, table 4.3); the replay differentiates every
  step with its own Jacobian and LU (exact derivatives need the Jacobian at each step).
  Using the run's LU with a few refinement iterations instead would bring the gradient
  closer to the forward cost; a candidate for later (no change in results beyond
  round-off).
- **Compiling** f, jacfwd and grad takes 2.5–13 s each, once per model structure.
- **Re-measured after step 4.5** (the replay no longer walks the record's unused rows; same
  results): see the 4.5 section below. A gradient now costs 4–9 forward solves.
- **Batches (`jax.vmap`) are not faster yet on CPU:** in the hand-test, 200 patients took
  about 2x the time of 200 single calls (the batch steps in lockstep, and both branches of
  every choice inside the solver run). Making batches fast is step 4.6.

## Phase 4 step 4.5: the sparse tier (2026-10-25)

`python benchmarks/step4_5_sparse.py`, one case per process, one after another (load
average 6 at the start, from other users' processes). Toy phage model, `erlang` scheme,
`rtol = atol = 1e-6`, t = 0 → 24 h, 25 output times. "JAX dense" and "JAX sparse" force each
tier; "grad" is the reverse-mode gradient of a sum of squares of log P at 12 times, every
parameter (12), through `sim.function` on the sparse tier; "peak memory" is the process's
largest resident size. Design note `docs/design/step4_5_sparse_tier.md`.

| model | states | scipy (s) | JAX dense (s) | JAX sparse (s) | sparse / scipy | first sparse solve incl. compile (s) | f(x) (s) | grad (s) | grad / scipy | build f + compile grad (s) | peak memory (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| toy phage 10 z x 12 a | 141 | 0.203 | 0.108 | 0.320 | 1.58 | 2.2 | 0.580 | 1.176 | 5.8 | 12.9 | 1.0 |
| toy phage 16 z x 20 a | 353 | 0.324 | 0.415 | 0.460 | 1.42 | 2.4 | 0.802 | 2.895 | 8.9 | 13.8 | 1.4 |
| toy phage 22 z x 30 a | 705 | 0.481 | 2.276 | 0.644 | 1.34 | 2.6 | 1.085 | 4.361 | 9.1 | 16.8 | 2.0 |
| toy phage 24 z x 38 a | 961 | 0.576 | 6.154 | 0.804 | 1.39 | 2.9 | 1.345 | 5.414 | 9.4 | 18.1 | 2.5 |
| toy phage 40 z x 60 a | 2,481 | 1.360 | – | 1.680 | 1.24 | 3.9 | 2.749 | 12.741 | 9.4 | 26.1 | 5.2 |
| toy phage 60 z x 94 a | 5,761 | 3.412 | – | 6.377 | 1.87 | 8.7 | 10.442 | 32.994 | 9.7 | 54.1 | 11.1 |
| toy phage 80 z x 125 a | 10,161 | 8.936 | – | 12.038 | 1.35 | 14.4 | 20.142 | 64.313 | 7.2 | 94.5 | 20.4 |

Python 3.12.15, JAX 0.11.2; x86_64, CPU JAX.

Reading the table:

- **The sparse tier takes the same steps as scipy** (the same matrix and factorisation) and
  costs 1.2–1.9x scipy's time: the extra is JAX's host-call overhead (~100 µs per solve) and
  the Jacobian's parts computed in JAX. The dense tier wins below ~300 states and loses
  badly above (6.2 s against 0.80 s at 961 states): hence `SPARSE_FROM_STATES` = 250.
- **Compiling is cheap on the sparse tier** (2–15 s, against 9 s for the dense tier at 961
  states): no dense LU to compile.
- **A gradient costs 6–10 scipy forward solves** at every size, 64 s at 10,161 states.
- **Memory** limits gradients on large models: about 2 GB per 1,000 states here (the record
  of ~700 steps with room for 4x as many, and the replay's reverse pass). Design note §5.
  After the design decision (room 2x above 2,000 states): 3.8 GB at 2,481 states and
  14.9 GB at 10,161 (gradients 12.1 s and 62 s, as before): about 1.5 GB per 1,000 states.

Step 4.4's benchmark re-measured with the replay skipping the record's unused rows (the
353-state case now runs on the sparse tier):

| model | states | params | forward (s) | steps plain / steered | f(x) (s) | jacfwd (s) | grad (s) | grad / forward | compile f, jacfwd, grad (s) | forward = reverse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| two-compartment oral PK, 2 doses | 3 | 5 | 0.0163 | 226 / 265 | 0.0340 | 0.0374 | 0.0649 | 4.0 | 3.9, 2.5, 2.5 | 5e-16 |
| Friberg myelosuppression (at rest) | 6 | 5 | 0.0030 | 15 / 89 | 0.0107 | 0.0164 | 0.0216 | 7.2 | 4.5, 3.3, 3.3 | 2e-16 |
| phage latency, 40 age cells | 42 | 8 | 0.0361 | 702 / 841 | 0.0735 | 0.1418 | 0.1537 | 4.3 | 5.1, 3.7, 3.6 | 1e-15 |
| toy phage 10 z x 12 a | 141 | 12 | 0.0596 | 376 / 422 | 0.1227 | 0.5844 | 0.5121 | 8.6 | 7.7, 6.3, 5.9 | 4e-16 |
| toy phage 16 z x 20 a | 353 | 12 | 0.4181 | 456 / 514 | 0.8122 | 3.6069 | 2.8422 | 6.8 | 9.2, 10.0, 8.4 | 2e-16 |

## Phase 4 step 4.6 and the Phase 4 gate (2026-10-27, commit after `d13568c`)

### Parameter sets in one call (`jax.vmap`)

`python benchmarks/step4_6_batches.py` (CPU) and `... gpu` (an RTX A6000, in a separate
environment with CUDA JAX), one case per process, one after another; load average 1.4 at
the start. Toy phage model, `erlang`, `rtol = atol = 1e-6`, 0 → 24 h, P at 12 times, every
parameter (12) varied by ±20% (log-normal). "forward": the solve alone (what
`sim.function(..., derivatives=False)` runs); "function": `sim.function` with derivatives
(the run recorded, then replayed). Design note `docs/design/step4_6_batches.md`.

| device | states | tier | what | members | cores | one call per member (s) | batch per member (s) | batch (s) | first batch incl. compile (s) | failed | peak memory (GB) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | 141 | dense | forward | 64 | 1 | 0.150 | 0.149 | 9.5 | 12.6 | 0 | 0.7 |
| cpu | 141 | sparse | forward | 64 | 1 | 0.557 | 0.158 | 10.1 | 13.5 | 0 | 0.8 |
| cpu | 353 | dense | forward | 64 | 1 | 0.643 | 0.772 | 49.4 | 52.3 | 0 | 1.2 |
| cpu | 353 | sparse | forward | 64 | 1 | 0.810 | 0.290 | 18.6 | 22.1 | 0 | 1.1 |
| cpu | 505 | dense | forward | 64 | 1 | 1.328 | 2.023 | 129.4 | 132.5 | 0 | 1.6 |
| cpu | 505 | sparse | forward | 64 | 1 | 0.950 | 0.395 | 25.3 | 29.5 | 0 | 1.2 |
| cpu | 141 | dense | function | 64 | 1 | 0.147 | 0.342 | 21.9 | 28.6 | 0 | 4.6 |
| cpu | 141 | sparse | function | 64 | 1 | 0.822 | 0.305 | 19.5 | 26.6 | 0 | 4.7 |
| cpu | 353 | dense | function | 64 | 1 | 1.142 | 1.737 | 111.2 | 116.2 | 0 | 10.3 |
| cpu | 353 | sparse | function | 64 | 1 | 1.180 | 0.630 | 40.3 | 47.7 | 0 | 20.0 |
| cpu | 141 | dense | function | 64 | 64 | 0.140 | 0.021 | 1.3 | 10.5 | 0 | 3.5 |
| cpu | 505 | sparse | forward | 1,000 | 1 | 0.948 | 0.333 | 333.4 | 337.3 | 0 | 4.1 |
| gpu | 141 | dense | forward | 64 | – | – | 0.019 | 1.2 | 6.3 | 0 | – |
| gpu | 141 | dense | forward | 1,024 | – | – | 0.011 | 11.3 | – | 0 | – |
| gpu | 353 | dense | forward | 64 | – | – | 0.077 | 4.9 | 10.3 | 0 | – |
| gpu | 353 | sparse | forward | 64 | – | – | 0.162 | 10.4 | 15.3 | 0 | – |
| gpu | 505 | dense | forward | 64 | – | – | 0.182 | 11.6 | – | 0 | 0.8 |
| gpu | 505 | sparse | forward | 64 | – | – | 0.216 | 13.8 | – | 0 | 0.1 |
| gpu | 505 | dense | forward | 1,000 | – | – | 0.155 | 155 | – | 0 | – |
| gpu | 601 | dense | forward | 1,000 | – | – | 0.253 | 253 | – | 0 | 17.7 |
| gpu | 601 | sparse | forward | 1,000 | – | – | 0.301 | 301 | – | 0 | 0.5 |
| gpu | 913 | dense | forward | 64 | – | – | 0.951 | 60.9 | – | 0 | 2.6 |
| gpu | 913 | sparse | forward | 64 | – | – | 0.389 | 24.9 | – | 0 | 0.1 |
| gpu | 141 | dense | function | 64 | – | – | 0.046 | 2.9 | 12.3 | 0 | – |
| gpu | 353 | dense | function | 64 | – | – | 0.176 | 11.2 | 20.9 | 0 | – |

GPU rows: measured 2026-10-26 (before the step 4.7 fixes, which touch only the sparse
tier's host side); "–": not recorded. GPU memory is the GPU's peak.

Reading the table:

- **On a CPU a batch runs on one core.** On the dense tier it costs about what one call per
  member costs; on the sparse tier it is 2.4–3.5x faster (one host call per batch). Above
  250 states the sparse batch beats the dense one 2.7–5x: the CPU threshold stays at 250.
- **Several cores** (`jax.shard_map`, recipe in `docs/simulation.md`): 64 members on 64
  cores, 7x faster than one call per member.
- **On a GPU the dense tier wins up to ~550 states** (64 members; ~650 for 1,000), hence
  `SPARSE_FROM_STATES_GPU` = 500.
- **Memory with derivatives** is the record: 20 GB for 64 members at 353 states. Without
  (`derivatives=False`): 1–4 GB.

### Step 4.5 re-measured after the gate fixes

`python benchmarks/step4_5_sparse.py` (columns as in the step 4.5 section above), load
average 1.5:

| model | states | scipy (s) | JAX dense (s) | JAX sparse (s) | sparse / scipy | first sparse solve incl. compile (s) | f(x) (s) | grad (s) | grad / scipy | build f + compile grad (s) | peak memory (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| toy phage 10 z x 12 a | 141 | 0.195 | 0.125 | 0.516 | 2.64 | 2.5 | 0.813 | 1.427 | 7.3 | 14.3 | 1.0 |
| toy phage 16 z x 20 a | 353 | 0.320 | 0.609 | 0.716 | 2.24 | 2.8 | 1.173 | 3.154 | 9.9 | 15.7 | 1.3 |
| toy phage 22 z x 30 a | 705 | 0.485 | 3.172 | 0.990 | 2.04 | 3.2 | 1.586 | 4.823 | 10.0 | 18.7 | 1.7 |
| toy phage 24 z x 38 a | 961 | 0.620 | 8.195 | 1.271 | 2.05 | 3.4 | 1.942 | 5.985 | 9.7 | 19.8 | 2.0 |
| toy phage 40 z x 60 a | 2,481 | 1.399 | – | 3.046 | 2.18 | 5.6 | 4.584 | 13.916 | 9.9 | 30.8 | 2.4 |
| toy phage 60 z x 94 a | 5,761 | 3.587 | – | 7.295 | 2.03 | 9.7 | 10.630 | 34.032 | 9.5 | 55.9 | 4.2 |
| toy phage 80 z x 125 a | 10,161 | 8.972 | – | 13.479 | 1.50 | 15.7 | 21.226 | 66.305 | 7.4 | 97.2 | 6.7 |

- **Memory fell by more than half on large models** (10,161 states: 6.7 GB against 14.9;
  2,481: 2.4 against 3.8). Much of the step 4.5 figure was the leak (factorisations never
  returned, step 4.7).
- **One sparse solve is 1.1–1.6x slower than at step 4.5** (961 states: 1.27 s against
  0.80): the factorisations now run on their own thread, whose core this machine keeps at
  a low clock (`ondemand` governor). Gradients barely moved (66 s against 64 at 10,161
  states). The dense tier, whose code did not change, also measured 15–33% slower in this
  run, so part of the difference is this machine's run-to-run variation.

### Performance targets, differentiable rows (the Phase 4 gate)

| target | measured | |
|---|---|---|
| Differentiable solve, dense tier: ≤ 5x scipy at 1k states | dense tier 13x at 961 states (8.2 s vs 0.62 s); the tier the size picks (sparse) 2.0x (1.27 s) | met by the automatic tier |
| Gradient, sparse tier, 10⁴ states: ≤ 10x the forward solve (amended 2026-10-26) | 66 s vs 13.5 s: 4.9x the JAX forward solve (7.4x scipy's) | met |
| 1,000 sets at 500 states without rebuilding, CPU and GPU | CPU: 333 s, 4.1 GB (505 states, sparse, forward, one core); GPU: 155 s (505 states, dense) | reported |

The forward rows of §10 (RHS evaluation, colours, the 72 h scipy run, the largest model) were
re-measured at the Phase 2 and 3 gates; Phase 4 did not change the scipy path.

## Phase 5 gate (2026-10-29, commit after `a3892bf`)

`python benchmarks/phase5_gate.py k`, one case per process, one after another. The machine
was not idle (an IDE kernel and Monolix; load average about 6 on 64 cores), so
times may be slightly inflated.

Seconds. "build" compiles the model; scipy is the default backend (BDF, sparse Jacobian);
JAX is timed twice, the first call including its compilation; "gap" is the largest
difference between the two backends' outputs relative to each output's size, at 25 output
times.

| model | states | build | scipy | RHS evals | JAX, 1st | JAX, 2nd (tier) | gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| target: toy phage 100 z x 100 a, erlang, 72 h | 10,201 | 0.01 | 11.81 | 2,249 | 18.25 | 15.92 (sparse) | 5e-13 |
| plaque fronts 240 x x 40 a, 10 h, rtol 1e-6 | 10,080 | 0.11 | 38.50 | 19,496 | 76.59 | 74.62 (sparse) | 3e-05 |
| oral absorption 160 g x 50 x, 24 h | 212 | 0.14 | 3.22 | 4,832 | 4.06 | 1.59 (dense) | 1e-05 |
| drug tolerance 160 x, constant drug, 60 d | 161 | 0.07 | 1.37 | 704 | 1.78 | 0.15 (dense) | 9e-15 |

Python 3.12.15, JAX 0.11.2, scipy 1.18.1, numpy 2.5.3.

- **The target case is unchanged** (11.8 s; Phase 2 gate 11.8 s, Phase 3 12.2 s): the
  Phase 5 features (end values, flux conditions, doses into fields) cost nothing when a
  model does not use them.
- **The backends agree** to their tolerances (plaque and oral absorption at the default or
  1e-6 tolerances; 1e-13 to 1e-15 where the solution is smooth).
- **Plaque fronts are twice as slow on JAX** as on scipy at 10^4 states: a steep travelling
  front needs ~20,000 RHS evaluations, and each sparse-tier step crosses to the host for its
  LU. scipy remains the default for forward runs; the JAX backend is for gradients and
  batches, as designed (ADR 0009).

### Tutorials

`python benchmarks/phase5_gate.py tutorials`: each tutorial run as a script in its own
process (as `tests/test_tutorials.py` does), seconds.

| tutorial | s |
|---|---:|
| 01 transit compartments | 11 |
| 02 where in the gut | 250 |
| 03 myelosuppression | 44 |
| 04 lifespan response | 22 |
| 05 hepatic dispersion | 36 |
| 06 whole-body PBPK | 9 |
| 07 infection age | 41 |
| 08 age-structured population | 38 |
| 09 persisters and dormancy | 16 |
| 10 biofilm penetration | 5 |
| 11 phage plaque fronts | 125 |
| 12 drug tolerance continuum | 31 |

All twelve run. For a workshop, 02 (about 4 minutes: about fifteen solves of the 212-state model, most at tolerances of 1e-8 to 1e-10)
and 11 (2 minutes, fronts at 10^4 states) are the ones to precompute or shorten.

## Forward or reverse derivatives (2026-10-31)

`python benchmarks/forward_or_reverse.py`, one case per process, one after another. The
machine was not idle (other users' processes), so each time is good to about ±20 %.

A transit chain of n compartments, the n split into p groups that share an exit rate (p
parameters, all in `wrt`); rtol 1e-6, 12 output times, one subject. Milliseconds; in
brackets, the derivative's cost in plain solves (`derivatives=False`).

| States | Names | Solve | `jax.grad`, forward | `jax.grad`, reverse | `jax.jacfwd`, forward | `jax.jacfwd`, reverse |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 5 | 7 | **17** (2.4) | 25 (3.5) | **16** | 23 |
| 10 | 10 | 10 | **18** | 30 | **19** | 27 |
| 20 | 20 | 10 | **32** | 47 | **32** | 95 |
| 40 | 40 | 20 | **112** | 160 | **107** | 274 |
| 100 | 5 | 48 | **124** (2.6) | 643 (13) | **123** | 536 |
| 100 | 10 | 135 | **141** | 665 | **142** | 665 |
| 100 | 20 | 51 | **202** | 668 | **214** | 786 |
| 100 | 50 | 53 | 749 | **665** | **712** | 1,355 |
| 100 | 100 | 53 | 1,619 | **727** | **1,954** | 4,693 |
| 400 | 10 | 1,684 | **4,081** | 14,723 | **4,075** | 12,062 |
| 400 | 50 | 1,641 | **8,541** | 15,069 | **8,534** | 16,413 |

- **Reverse costs 3.5 to 13 solves for a gradient, flat in the number of names**; forward
  costs 2–3 solves and a fraction of a solve more for each name (about a third of a solve at
  100 states, a fifteenth at 400).
- **For a gradient they cross at 40–50 names** at 100 states; at 400 states forward was still
  1.8 times faster with 50.
- **For a whole table of derivatives (`jax.jacfwd`) forward was faster in every case**,
  2.4 times at 100 names: reverse makes one backward pass per output time.
- So `derivatives="auto"` picks forward up to 40 names, reverse above.
- **Memory.** With 12 names on the 61-state toy phage model, a batch of 4 (`jax.vmap`) needed
  3.5 MB of working memory forward and 840 MB reverse (XLA's compile-time estimate); without
  derivatives, 1 MB.
