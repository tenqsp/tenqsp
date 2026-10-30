# Step 4.5: the JAX backend's sparse tier

Status: **accepted** by the project lead 2026-10-25: all three §5 recommendations (threshold
automatic at 250, no option; record room 2x above 2,000 states; no scipy
forward-sensitivity fallback). Built and measured. With the smaller room, a gradient's
peak memory fell from 5.2 to 3.8 GB at 2,481 states and from 20.4 to 14.9 GB at 10,161
(-27 %; the rest does not depend on the room), at the same speed. The route
follows ADR 0001 (decision 3): the TenQSP BDF in JAX, with SuperLU on the host for the
linear algebra above a few hundred states. Nothing in the public API changes.

## 1. What it does

Until now the JAX backend refused models above 1,000 states: it factorised the iteration
matrix `I - c J` as a dense matrix, whose cost grows with the cube of the number of states
(7 s per solve at 961 states, against 0.6 s on scipy). The sparse tier factorises the same
matrix with SuperLU, the sparse LU scipy uses, so the JAX backend now runs models of any
size, and `sim.function`, `sim.sensitivity`, `jax.grad`, `jax.jacfwd` and `jax.vmap` work on
them.

- **Automatic by size:** dense up to `SPARSE_FROM_STATES` = 250 states, sparse above
  (§5, decision 1). `jax_backend.tier(cm)` says which.
- **The same steps as scipy.** The Jacobian's parts are computed in JAX (the same coloured
  passes as the scipy backend), then assembled and factorised on the host exactly as scipy's
  BDF does with the scipy backend's Jacobian. Same matrix, same factorisation, same
  decisions: on the toy phage model at 2,481 states both backends make the same number of
  Jacobians and LU factorisations and agree to 1e-10 (`test_step4_5_sparse.py`).
- **Derivatives:** each replayed step solves with its own matrix
  (`jax.lax.custom_linear_solve`): factorised and solved on the host; reverse mode solves
  with the transposed factors. They equal the dense tier's to 1e-10 and finite differences
  of whole solves; forward = reverse.
- A singular or non-finite matrix gives NaN, as a dense LU does, so the solver treats it as
  a failed Newton iteration (a fresh Jacobian, then a smaller step) or reports the failure;
  it never raises from inside the program.

## 2. How (for the record)

- `jax_bdf.bdf(linear=)`: the integrator's linear algebra is pluggable
  (`LinearSolver(factor, solve, empty)`; the dense LU is `jax_bdf.DENSE`).
- `CompiledModel.jacobian_pieces` (in JAX, fixed shapes) and `assemble_sparse` (on the
  host); `sparse_jacobian` (the scipy backend's) now uses the same assembly.
- `backends/jax_sparse.py`: in the run, a factorisation lives on the host and the program
  holds an integer handle to it (`jax.pure_callback`); it is reused over many steps as scipy
  does, and freed once unused (least recently used of the last 64). In the replay, one
  host call factorises and solves (several right-hand sides at once for `jacfwd`).
- Under `jax.vmap` the host calls run one member after another (correct, not faster; 4.6).

## 3. Measured

`python benchmarks/step4_5_sparse.py` (table in `docs/benchmarks.md`, step 4.5). Toy phage
model, erlang scheme, rtol = atol = 1e-6, 0 → 24 h. In short:

| states | scipy | JAX dense | JAX sparse | gradient, 12 parameters (sparse) | peak memory |
|---:|---:|---:|---:|---:|---:|
| 141 | 0.20 s | **0.11 s** | 0.32 s | 1.2 s | 1.0 GB |
| 353 | **0.32 s** | 0.42 s | 0.46 s | 2.9 s | 1.4 GB |
| 961 | **0.58 s** | 6.2 s | 0.80 s | 5.4 s | 2.5 GB |
| 2,481 | **1.36 s** | – | 1.68 s | 12.7 s | 5.2 GB |
| 5,761 | **3.41 s** | – | 6.38 s | 33.0 s | 11.1 GB |
| 10,161 | **8.94 s** | – | 12.0 s | 64.3 s | 20.4 GB |

- Forward solves on the sparse tier cost 1.2–1.9x scipy's at every size; the dense tier is
  faster below ~300 states and much slower above, hence the threshold of 250.
- A gradient (reverse mode, every parameter) costs 6–10 scipy forward solves.
- **Memory** is what limits gradients on large models (§5, decision 2).

## 4. Found while building

1. **Host calls cost ~100 µs each** (JAX's `pure_callback`, measured with a function that
   does nothing). A solve is a host call, so on a tiny model the sparse tier is 5–10x
   slower than the dense one; above a few hundred states the solve itself dominates. A
   compiled extension (JAX's FFI) would remove most of it; not needed now.
2. **The replay walked every row of the record**, including the room left for more steps
   (with `RECORD_ROOM` = 4 and rounding to a power of two, 4–8 rows per step taken): each
   unused row computed a Jacobian (and on the sparse tier made a host call), then was
   thrown away. Unused rows are now skipped. This sped up gradients on both tiers
   (dense: from 7–28 to 3–9 forward solves; sparse: from 22–49 to 7–11), with identical
   results. A run that fails before its first step now replays nothing, so `sim.function`
   marks its derivatives NaN explicitly.
3. **The record was copied once more before the replay** (to keep unused rows harmless);
   now done row by row: XLA's working memory for a gradient −19 %.
4. **The test summary's counts were inflated** (tests/conftest.py): pytest-xdist workers
   also run the summary hook, and each added the other workers' result files to its own
   counters before saving them. The counts of replayed runs and gradient comparisons
   reported since step 4.4a (e.g. "9,880 runs recorded and replayed", "4,746 runs agree")
   were too high; pass/fail and the worst differences were not affected (each check
   asserts in its own test; the worst values are maxima). Fixed.
5. `pytest --backend=jax --tier=sparse` puts every JAX solve on the sparse tier, so the
   whole suite (V9, `--replay`, `--gradients`) also checks the sparse path; with 4x the
   per-test timeout (finding 1).

## 5. Decisions for the project lead

1. **Tier threshold: automatic, at 250 states, no user option** (recommended). Users never
   choose a linear solver; the threshold is a measured constant, revisited in 4.6. The
   alternative is a solver option (`linear="dense" | "sparse"`), an API addition
   with little use.
2. **Memory of gradients on large models.** The record keeps room for 4x the steps of the
   run at `f.x0` (so `f` works far from x0 before giving "too many steps for the record"),
   rounded up to a power of two. Memory grows with states x room. Options:
   - **A.** Keep it as it is, and document the memory (about 2 GB per 1,000 states, for a
     run of ~700 steps).
   - **B.** Less room on large models (2x above 2,000 states): about half the memory;
     an optimizer that moves far from x0 makes a new `sim.function` sooner (it gets NaN
     with that reason, as now). **Recommended.**
   - **C.** B, plus not keeping the full difference array per step (only what each step
     needs): a further cut, a few days' work; later if a real model needs it.
3. **The scipy forward-sensitivity fallback** (ADR 0001 decision 4: "for very large
   sizes, forward sensitivities on the scipy path, for a handful of parameters"):
   **recommended: not built now.** The sparse tier gives exact gradients at 10,000 states
   (64 s, 20 GB, for 12 parameters). Forward sensitivities on scipy integrate (1 + p) copies of the system:
   at 10,000 states and 12 parameters that is 130,000 states, which scipy solves in about
   3 minutes (171 s at 100,000 states, ADR 0003), without `jit` or `grad`. Build it only if a
   real model needs more states than the sparse tier's memory allows.
