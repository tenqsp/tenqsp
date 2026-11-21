# ADR 0010 — Stacked batches: the scenarios of one build as one system

- **Status:** Accepted by the project lead, 2026-11-21.
- **Amends:** ADR 0009, decisions on batches (every batch one `jax.vmap` call).
- **Code:** `src/tenqsp/backends/jax_bdf.py` (`units=`: the error norm of a stack),
  `src/tenqsp/backends/jax_backend.py` (`_stacked_program`, `stacked_linear`,
  `stacked_unsupported`), `src/tenqsp/gradients.py` (`FunctionBatch(batch=)`,
  `_stacked_impl`), `src/tenqsp/simulate.py` (`sim.function(..., batch=)`).

## Context

`sim.function([sc1, sc2, ...])` runs the scenarios of one build in one `jax.vmap` call: each
scenario takes its own steps, so the batched solver carries per-member step sizes, orders and
Newton iterations, and every operation becomes a gather, a scatter or a batched product of
small matrices. For small models the cost is those operations, not the arithmetic: profiled
on a 6-state model (Friberg's myelosuppression model) with 20 scenarios, a batched BDF step
ran about 590 XLA operations of about 1 µs each, and a run of the 20 scenarios with 4
forward sensitivities took 82–91 ms, against about 4 ms for one scenario alone. Population
fitting (tenqsp-fit) runs such batches thousands of times.

## Options considered

1. Keep `vmap`, cut the per-step operations (fewer gathers, fused updates): measured gains of
   1.2–1.7x, not the 5–10x needed.
2. **Stack the scenarios into one system**: one state vector of all scenarios, one sequence
   of steps, the iteration matrix as one block per scenario. Every step is then a few large
   operations.
3. A compiled solver outside JAX (SUNDIALS CVODES through CasADi), per scenario: fast per
   run, but a second code path, and derivatives outside JAX.

## Measurements

Friberg's model (6 states), 20 scenarios, one run with 4 forward sensitivities, this
machine (CPU):

| | `vmap` | stacked |
|---|---:|---:|
| one run | 82 ms | 12.7 ms |
| one run and its derivatives (JVP) | 91 ms | 12.7 ms |
| compile | 4.6 s | 3.2 s |

Values agree with the `vmap` batch to 3e-6 (relative; rtol 1e-6), derivatives to 2e-6. A
population fit of the same model (tenqsp-fit, FOCE-I, 20 patients) went from 203 s to 43 s
(29 s with tenqsp-fit's compilation cache), with the same estimates to the digits printed.

With many states per scenario, dense blocks lose: in a prototype with a maturation chain of
N cells (N + 3 states), the stacked run with dense blocks was faster than `vmap` at 33 and 103
states and slower at 303 (13.0 s against 5.0 s), where a sparse stacked system (one SuperLU
factorisation of the block-diagonal matrix for all scenarios) took 2.0 s.

## Decision

1. **`batch=` on `sim.function([...])`**: `"vmap"` (the default) or `"stacked"`, reported as
   `fs.batch`; `"stacked"` where a condition below fails raises with the reason; `batch=` on a
   single scenario raises. (Revised before release, 2026-11-21: the first form of this record
   made `"auto"`, stacked wherever the conditions hold, the default. Measured after: on a
   pooled fit of 6 arms whose doses span 9 decades (8 states), the shared run took 996 steps
   against at most 460 for one arm, and the fit 2 times as long; on a model with a chain of
   cells and 6 such arms, stacking was slower from 33 states on and even at 13. How
   much the scenarios differ, which sets the shared steps, is not known before the run, so
   stacking is asked for, not chosen.)
2. **The stacked run**: one `bdf` call per segment on the stacked state, with the right-hand
   side and its Jacobian mapped over scenarios; the iteration matrix as dense blocks, never
   assembled (an explicit inverse per block up to 16 states, an LU per block above).
3. **The error norm of the least easy scenario** (`bdf(units=)`): every norm (the state, the
   Newton correction, the forward sensitivities, the first step) is the largest of the
   scenarios' root-mean-square norms, so no scenario is solved less accurately than alone;
   each sensitivity's floor (`SENS_FLOOR`) is taken per scenario.
4. **Doses and times**: the run stops at the union of the scenarios' dose and infusion times
   (each scenario's doses applied to its own block) and writes every scenario's state at the
   union of their output times; each scenario's outputs are picked from it. Forward
   sensitivities jump at each scenario's own moving boundaries, as in a single run.
5. **When stacking applies**: two or more scenarios; derivatives `"forward"` or `False`
   (reverse derivatives replay a record per run: not stacked); no events (extinction
   thresholds, conditions on the state), no conditions on time, no noise floors; the dense
   tier (at most `SPARSE_FROM_STATES` states); the same start time; and doses at no more
   than `STACK_TIMES_FACTOR` (4) times as many distinct times as the scenario with the most,
   plus 4 (otherwise the shared run restarts far more often than any one scenario would).
6. **A failed run**: a stacked run that fails gives NaN for every scenario of the call;
   `fs.status(X)` then reruns them one by one (`vmap`) to say which failed and why. (A
   fallback inside the compiled function, rerunning one by one where the stacked run fails,
   was measured: it made a population fit four times slower, from compiling both paths at
   every call site.)
7. **No change to single runs** (`sim.function(scenario)`, `model.solve`, `sim.batch`).

## Not in this record

- A **sparse stacked tier** (decision tests: the threshold between dense blocks and one
  SuperLU of the block-diagonal system, measured between 33 and 103 states per scenario);
  until then models above the dense tier stay on `vmap`.
- Stacked **reverse derivatives** (a record of units × states × steps), stacked **events**
  (per-scenario terminal events inside one run), conditions on time, and `sim.batch`.
- A compiled solver outside JAX: not needed for the measured cases.

## Consequences

- **A scenario's results depend slightly on the others in its batch** (within the solver's
  tolerances): the shared steps are those the least easy scenario needs. The values are
  also slightly less smooth in a scenario's own dose times (moving one dose moves a boundary
  of the shared run); derivatives are unaffected. `batch="vmap"` gives each scenario its own
  steps.
- One failing scenario gives NaN for the whole call (decision 6).
- `fs.steps(X)` gives the shared run's count for every scenario.
- Stacked batches are much faster for small models, and so are fits built on them.
