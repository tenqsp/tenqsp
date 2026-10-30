# ADR 0009 — Phase 4: the differentiable backend, its tiers, derivatives and batches

- **Status:** Accepted by the project lead at the Phase 4 gate, 2026-10-27. Each decision below was
  already taken by the project lead in the design note it cites; this ADR collects them in one place.
- **Code:** `src/tenqsp/backends/jax_bdf.py` (the integrator), `jax_backend.py` (segments,
  events, the tier), `jax_sparse.py` (the sparse tier), `jax_replay.py` (the replay),
  `src/tenqsp/gradients.py` (`sim.function`, `sim.sensitivity`), `src/tenqsp/simulate.py`,
  `src/tenqsp/regimen.py`. Design notes: `docs/design/step4_3_jax_bdf.md`,
  `step4_4_gradients.md`, `step4_4d_api.md`, `step4_5_sparse_tier.md`, `step4_6_batches.md`.
  Measurements: `docs/benchmarks.md` (steps 4.3–4.6).

## Context

ADR 0001 chose scipy's BDF as the default forward solver and a TenQSP-owned BDF in JAX for
derivatives. Phase 4 built that JAX solver, its derivatives, a sparse tier for large models
and batches over parameter sets. This ADR records the numerical choices underneath, each of
which decides numbers a user gets or what they cost.

## Decisions

1. **The JAX solver is scipy's BDF, rewritten in JAX** (step 4.3): variable order 1–5, the
   same error control, Newton iteration and step-size rules, restarted at every dose,
   infusion edge and event, so the two backends take comparable steps (V9, `pytest --backend=jax`:
   the test suite's solves run on the JAX solver and pass against the same oracles). No Radau fallback on the JAX path: a failing run
   raises `SolverError` (or gives NaN inside `sim.function`) and suggests `backend="scipy"`.
2. **Derivatives come from recording a run and replaying it** (step 4.4,
   discretize-then-differentiate): the run's steps are fixed, and the replay's arithmetic
   carries the derivative. Dose times, lag times, infusion edges and the times of switch
   and extinction events move with the parameters (implicit function theorem for state
   events; PRD §6.3 and V8 amended 2026-10-25). The run's steps are chosen for one
   direction of the derivative as well as the solution (project lead decision B, step 4.4d §6.3).
   The record has room for 4x the steps of the run at `f.x0` (2x above 2,000 states, project lead
   decision 2026-10-25); an `x` needing more gives NaN with its status.
3. **Two linear-algebra tiers, chosen by size** (step 4.5; project lead decisions 2026-10-25 and
   2026-10-26): a dense LU in JAX up to 250 states on a CPU, 500 on a GPU; above, SuperLU on
   the host called from JAX, with the same matrix and factorisation as scipy's BDF. No user
   option; no scipy forward-sensitivity fallback.
4. **Batches (`jax.vmap`) run their members in lockstep** (step 4.6): work that only some
   members need runs when any member needs it (`jax_bdf.when`); a done member's state is
   kept without copying its record (`jax_bdf.while_running`); on the sparse tier one host
   call takes the whole batch. Several CPU cores: documented recipe (`jax.shard_map`), an
   option in step 4.8.
5. **`sim.function(..., derivatives=False)`** (step 4.6, project lead decision 1): the solve alone,
   no record or replay, steps for the solution only; differentiating it raises.
6. **Host factorisations are counted exactly and live on one Python thread** (step 4.7).
   Each is freed when its member replaces it and when its integration ends
   (`LinearSolver.release`); nothing is evicted, so runs in several Python threads at once
   never lose each other's (the step 4.5 "last 64, least recently used" room did, found by
   the gate review). They are made and freed on one Python thread: with
   scipy 1.18 a factorisation made in one JAX host call and freed in another never returns
   its memory (~2 MB at 500 states; 57 GB for 1,000 members at 505 states before the fix).
   Cost on this machine (idle cores at a low clock, `ondemand` governor): a single sparse
   run 1.1–1.6x slower; batches unchanged; none when the process runs on one core.
7. **Targets**: the 10⁴-state gradient ≤ 10 forward solves
   (measured 5.2x).

## Consequences

- Derivatives are those of the computed solution (exact for the steps taken). `pytest
  --backend=jax --gradients` checks them against central finite differences on 185 runs of
  the test suite's models (largest difference 7.5e-6 of the outputs' size; runs above 200
  states, or that fail at tight tolerances, are not checked). A gradient costs 4–10
  forward solves.
- Memory for derivatives grows with steps x states (about 1.5 GB per 1,000 states for a
  gradient on large models); populations without derivatives need a few MB per member.
- The sparse tier depends on a host callback: `jit` and `vmap` work, but not export to
  another device, and on a GPU its linear algebra runs on the CPU.
- Revisit decision 6 if scipy fixes the thread-state leak (then make and free on any thread).
