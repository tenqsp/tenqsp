# ADR 0001 — Stiff integration and linear-solve strategy

- **Status:** Accepted by the project lead, 2026-10-10 (Phase 0.5 spike (a), 2026-10-09).
- **Spike code and raw data:** `spikes/linear_solver/` (`results.jsonl`, `summarize.py`).

## Context

PRD v0.4 §6.3 asks for a differentiable stiff backend that scales past the dense-Jacobian
regime, tiered by model size. The plan flagged this as the single high-risk item that sets
the v1.0 schedule. The spike measures the candidate options on a structured phage model
before any library code depends on the choice.

**Test model** (`toy_model.py`, hand-written, flux-form finite volumes):
- `B[z, s]`: dormancy level `z`, a bidirectional axis with sign-changing drift and diffusion,
  times susceptibility `s ∈ {S, R}`. Logistic growth with a `B_tot` integral, mutation on the
  parent division flux, and adsorption.
- `I[a, z]`: infection age `a`, one-way, with renewal inflow and absorbed outflow; z-diffusion.
- `P`: compartment, coupled through integrals of lysis and adsorption.

Sizes: 961, 3,721, 10,201 and 29,241 states. Run over t ∈ [0, 24] h with rtol 1e-6, atol 1e-3.
The accuracy reference is scipy BDF at rtol 1e-9 / atol 1e-6. The machine is the development
box (64 cores, CPU JAX 0.11.2, diffrax 0.7.2, lineax 0.1.1, scipy 1.18.1). Sizes ran
concurrently, so wall times carry roughly ±30% contention noise. Ratios within one size are
reliable.

## Options measured

| Tag | Integrator | Linear solve |
|---|---|---|
| `scipy` | scipy BDF (variable order) | SuperLU on a sparse Jacobian built by colored jvps + vjp for dense rows |
| `dense` | diffrax Kvaerno | lineax dense LU (default `AutoLinearSolver`) |
| `gmres` | diffrax Kvaerno | lineax GMRES on the linearised operator |
| `jgmres` | diffrax Kvaerno | pure-JAX GMRES on the colored-extracted sparse matrix, Jacobi-preconditioned |
| `splu` | diffrax Kvaerno | colored extraction → host SuperLU via `jax.pure_callback` |

## Results (forward solve, wall time after compilation)

> **Correction (2026-10-10, ADR 0003):** these runs were executed concurrently and the wall
> times are inflated. Re-run sequentially, scipy BDF takes 7.7 s at 10,201 states (not 70 s),
> 33 s at 29,241 (not 46 s), and 171 s at 99,856. The ranking and the decision below are
> unaffected.

| states | colors | scipy BDF | diffrax+splu (Kvaerno4) | diffrax+jgmres | diffrax dense | max rel. err (diffrax) |
|---:|---:|---:|---:|---:|---:|---:|
| 961 | 61 | **0.40 s** | 8.7 s (622 steps) | 20.9 s | 137 s (Kvaerno5) | 1.1e-5 |
| 3,721 | 121 | **2.1 s** | 39.5 s (784 steps) | 241 s | >1,800 s (timed out) | 1.2e-5 |
| 10,201 | 201 | **70 s** | 105 s (924 steps) | 618 s | not run | 1.2e-5 |
| 29,241 | 341 | **46 s** | 682 s (1,107 steps) | >1,800 s (timed out) | not run | 1.3e-5 |

The scipy working-tolerance runs matched the reference to 3e-5–1e-4 relative error.

Kvaerno variants at 961 states (splu): Kvaerno3 took 1,502 steps (19 s), Kvaerno4 622 steps
(8.7 s), and Kvaerno5 1,028 steps with 476 rejected (16 s). Kvaerno4 is the best SDIRK here.

**Gradients** (d log P(24) / d log k_S, reverse mode through diffrax at 961 states):

| path | gradient | central FD | rel. err | wall |
|---|---:|---:|---:|---:|
| dense (default) | −1.342057 | −1.342057 | 6e-8 | 389 s |
| splu forward + splu implicit adjoint (shimmed) | −1.342057 | −1.342057 | 6e-8 | 28.5 s |

## Findings

1. **Variable-order BDF beats SDIRK on this problem class by ~5–25×**, mostly because it
   takes far fewer steps. The phage dynamics have long smooth phases where high-order BDF
   takes large steps. diffrax has no BDF, so every diffrax configuration lost on step count
   before linear-solve cost was even counted.
2. **Dense Jacobians fail at small sizes, not only large ones.** Materialising the dense
   Jacobian costs `n` jvps every time it is formed. At 961 states this alone made `dense`
   15× slower than `splu`.
3. **Colored compressed Jacobians work**: 61–341 jvps per Jacobian instead of 961–29,241. But
   the **color count grows with the grid** (≈ 2·n_z + 1), because the logistic `B_tot`
   integral makes every `B` row dense over all `B` columns. The PRD target ("≤ ~30 colors
   independent of grid size", §10) is only reachable if **integral couplings are lifted**:
   introduce each integral as an auxiliary state/algebraic variable, or solve a bordered /
   low-rank system with Sherman–Morrison–Woodbury. The IR knows every integral explicitly,
   so the compiler can do this automatically. Single dense rows (the `P` equation) are
   handled well by one vjp each.
4. **diffrax's Newton loop cannot carry a lineax iterative solver's state.** The lineax
   GMRES state holds the linearised operator, which carries a closed jaxpr, and fails at
   trace time with `TypeError`. Krylov solvers work only when the solver extracts plain
   arrays first, as `jgmres` does.
5. **Pure-JAX Jacobi-GMRES is correct but slow** (2–6× slower than splu). It needs a stronger
   preconditioner, which the stiff reaction coupling makes hard.
6. **Host SuperLU through `pure_callback` works inside diffrax**, including reverse-mode
   gradients via `trans='T'`. It is not GPU-capable, and `vmap` runs sequentially.
7. **Gradients always use a dense solve by default.** `optimistix.root_find`'s
   `ImplicitAdjoint` hard-defaults to a dense `AutoLinearSolver`, and diffrax passes no
   `adjoint` through. Shimming it to the sparse solver gave an identical gradient 13.6×
   faster. The library needs its own integrator, or a supported hook upstream, to do this
   without monkey-patching.
8. **Factorization caching needs ownership.** The spike's LRU cache of 1,024 SuperLU objects
   reached 75 GB of RSS at 29k states. A real implementation keeps only the live
   factorization(s).

## Decision

1. **Forward simulation: scipy BDF + colored sparse Jacobian is the default backend** for all
   sizes, not only the "reference" backend. It is fastest at every size tested and robust.
   This moves the colored Jacobian into Phase 2.
2. **Lift integral couplings in the compiler**, so the color count stays bounded as grids
   are refined. Re-measure against the PRD §10 target.
3. **Differentiable backend: write a TenQSP-owned implicit integrator in JAX** — a
   variable-order BDF/NDF (the scipy algorithm), with a pluggable linear solver and adjoint,
   instead of using diffrax's SDIRK solvers. Linear-solver tiers:
   - dense LU when `n ≲ 1–2k` (GPU-capable, `vmap`-able);
   - colored extraction + host SuperLU (CPU) above that, which is what was measured here;
   - pure-JAX preconditioned Krylov kept as an experimental option only.

   Gradients use the same sparse solver on the transposed system. diffrax stays an optional
   backend for non-stiff and small models, and as a cross-check (V9).
4. **Fallback for the differentiable tier at very large sizes:** forward sensitivities on the
   scipy path, for a handful of parameters.

   > **Amended 2026-10-25 (step 4.5, project lead decision):** not built. The sparse tier (host
   > SuperLU) gives exact gradients at 10,000 states (64 s, 20 GB for 12 parameters);
   > forward sensitivities on scipy would integrate (1 + p) copies of the system, slower and
   > without `jit`/`grad`. Revisit only if a real model needs more states than the sparse
   > tier's memory allows. Design note `docs/design/step4_5_sparse_tier.md` §5.

## Consequences

- The PRD §6.3 tiering stands, but the differentiable tier is a **custom BDF in JAX**, not
  diffrax configuration. Estimated effort is 3–4 weeks inside Phase 4: integrator, error
  control, event restarts, adjoint, and tests against scipy. That fits within the plan's
  3–8-week Phase 4 band, toward its upper half.
- PRD §10 color target: valid only with integral lifting (finding 3). Revise the target text.
- CPU host SuperLU is not differentiable through JAX transforms by itself. Gradients go
  through the implicit-function rule (as in the spike), so correctness depends on the
  transpose-solve path. That path gets a V8 test at each tier.
- GPU: dense tier only, for v1.0.
- Open questions for the project lead: accept the custom-BDF route and its cost; or keep diffrax
  SDIRK, accept 5–25× slower differentiable solves, and maintain a monkey-patch for sparse
  gradients.
