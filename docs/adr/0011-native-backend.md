# ADR 0011 — A native CPU backend: the JAX programs translated to numba

- **Status:** Accepted by the project lead, 2026-11-22 (option chosen: numba).
- **Related:** ADR 0001 (the JAX BDF), ADR 0010 (stacked batches).
- **Code:** `src/tenqsp/backends/native/` (`translate.py`, `bdf_template.py`, `program.py`),
  `src/tenqsp/gradients.py` (`FunctionBatch(backend=)`), `src/tenqsp/simulate.py`.

## Context

Population and pooled fits (tenqsp-fit) run a batch of scenarios thousands of times, each
with forward sensitivities. On the CPU, TenQSP's JAX BDF takes the same steps as scipy's (and
as an ODEPACK solver on the same problem), but each step is about 350 XLA operations plus the
Newton iterations, each a separately executed kernel: 70–90 µs per step for an 8-state model,
where a compiled step takes a few µs. The cost is fixed per operation, not arithmetic (the
derivatives come nearly free; 6 scenarios cost little more than 2). Tracing the solver's JAX
program also takes seconds for every fit.

## Options considered

1. Fewer, larger XLA operations per step: at most 2–3x.
2. A C solver (SUNDIALS CVODES): proven, but a C compiler on every user's machine.
3. **numba**: pip-installable on every platform, no compiler needed; the solver ported to
   numba, the model's functions generated from its JAX programs.

## Decision

1. **The model's functions are translated, not written twice.** `translate.source` turns a
   JAX function's jaxpr into Python source for numba: the right-hand side, its Jacobian
   (`jax.jacfwd`) and its derivative along P directions (`jax.vmap` of `jax.jvp`). Constants
   are folded at translation (the geometry, indices built from literals); gathers and
   scatter-adds with constant indices become index maps. Every model feature that reaches the
   right-hand side (axes, categories, operators, formulas) is covered by the same translation.
2. **Two code styles.** A small kernel (every array at most 2048 entries, at most 40 000
   operations) is straight-line scalar code: arrays are symbolic at translation, structural
   operations rearrange entries, and each elementwise result is one line per entry. A larger
   kernel works on flat buffers, structural operations as integer maps and elementwise chains
   as one loop. (Array code with numpy broadcasting took 12 minutes to compile for an 8-state
   model; scalar code compiles in seconds and runs without making arrays.)
3. **The solver is a port of `jax_bdf.bdf`** and of the JAX program's segment loop (doses at
   segment starts, infusions constant on a segment, the jumps of the derivatives where a
   boundary moves, output times at the end after its doses): the same predictor, Newton
   iteration, sensitivity corrector, error scale, order and step choice, so the two backends
   take the same steps up to round-off. Not ported: events, conditions on time, noise floors,
   the step record (reverse derivatives), stacked units, the sparse tier.
4. **What stays in JAX:** laying out a scenario's inputs from `x`, the model's doses, lag and
   bioavailability, the initial state and their derivatives along each entry of `x`, and the
   outputs read from the states. The solve is a `jax.pure_callback` inside a
   `jax.custom_jvp` (its tangent is the sensitivities times the tangent of `x`, a linear map,
   so reverse mode transposes it); under `jax.vmap` the callback gets the whole batch
   (`vmap_method="broadcast_all"`), and numba runs the scenarios in parallel on the cores.
5. **One module per model structure** (`ModelIR.structure_hash`, the number of directions, the
   solver's and translator's sources), written to `~/.cache/tenqsp/native` (or
   `TENQSP_NATIVE_CACHE`; owner-only permissions), so numba caches the compiled solver beside
   it: the first build of a model compiles it, later ones load it.
6. **Opt in:** `sim.function([...], backend="native")` for a list of scenarios with forward or
   no derivatives; refused, with the reason, elsewhere. numba is an optional dependency.
   tenqsp-fit chooses it where it applies.

## Measurements

On this machine (CPU), one batch with 4 forward sensitivities:

| | JAX | native |
|---|---:|---:|
| oral PK, 3 scenarios, ~330 steps each | 45 ms | 4.7 ms |
| a population fit of Friberg's model (20 patients; tenqsp-fit), whole fit, one start | 44 s | 9 s |

Step counts identical to the JAX backend's; values within 1e-12 (relative), forward
derivatives within 1e-11, on the tests' models.

## Consequences

- A first fit of a model compiles its native solver (tens of seconds); later fits load it.
- Results agree with the JAX backend to round-off, not bit for bit.
- Not yet native: models with events or conditions on time, reverse derivatives, the sparse
  tier; they stay on the JAX backend.
