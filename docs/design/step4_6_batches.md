# Step 4.6: parameter sets in one call (`jax.vmap`)

Status: **accepted** (project lead decisions 2026-10-26, §6). The fixes (§1) and decisions 1–2 are built and tested.
Measured on the toy phage model (erlang, 0–24 h, rtol = atol = 1e-6), CPU: 64 cores (one
member at a time uses one), GPU: RTX A6000 (double precision, in a separate environment
with CUDA JAX; the project's environment stays CPU-only). Scripts:
`benchmarks/step4_6_batches.py`, the exploratory runs in this note.

## 1. What was wrong, and what changed (no change to the API)

Until now, `jax.vmap` over `sim.function` gave the right answers but took **7–12x the time
of one call per member** (16 members at 61–353 states). Three causes:

1. **The step record was copied at every step.** JAX runs a batched loop until every member
   is done, and keeps a done member's state by copying the *whole* state at every
   iteration, including the step record (rows x orders x states: hundreds of MB). Now the
   loop's test is "some member is still running", done members keep their state by the
   same copy *except* the record, and their writes to it are dropped
   (`jax_bdf.while_running`). 16 members at 61 states: 11.8 s -> 1.9 s.
2. **Every "only if needed" became "always".** Under `vmap` a branch (a fresh Jacobian, an
   LU, locating an event, the replay skipping unused record rows) is computed for every
   member at every step and the result picked afterwards. Now the work runs when *some*
   member needs it, and is skipped when none does (`jax_bdf.when`).
3. **The sparse tier, under `vmap`:** one host call per member per Newton iteration, and
   the host kept only the last 64 factorisations: a batch of more than ~32 members lost
   factorisations still in use (NaN solves, then steps shrinking without end: the old code
   had not finished an 8-member batch after 10 minutes with that room set to 2). Now one
   host call takes the whole batch, only the members that need a new factorisation get
   one, and a factorisation is freed when its member replaces it. A batch of 100 at 61
   states: 3x faster than one call per member (25 s vs 78 s).

The members of a batch still run **in lockstep**: each step of the batch costs what the
most demanding member needs (when one member needs a fresh LU, every member's is
recomputed on the dense tier). That is what `vmap` is; it pays off where one call has much
fixed cost per step (host calls, small arrays) or where the hardware is wide (a GPU).

Tests: `tests/test_step4_6_batches.py` (a batch equals its members and the closed form,
across doses and an extinction; derivatives of a batch; nested batches; a sparse batch
beyond the host's room; `when` does no work no member needs; a batch costs at most ~5x its
members' calls, slow tier).

## 2. Measured: time per member (forward solve; 64 members unless stated)

| states | CPU, dense, one by one | CPU, dense, batch | CPU, sparse, one by one | CPU, sparse, batch | GPU, dense, batch | GPU, sparse, batch |
|---:|---:|---:|---:|---:|---:|---:|
| 141 | 0.16 s | 0.15 s | 0.57 s | 0.16 s | 0.019 s (0.011 s for 1,024) | 0.097 s |
| 353 | 0.65 s | 0.80 s | 0.78 s | **0.30 s** | 0.077 s (0.058 s for 1,000) | 0.16 s |
| 505 | 1.37 s | 2.14 s | 0.87 s | **0.40 s** | 0.155 s (1,000 members) | – |
| 961 | – | – | ~0.8 s (step 4.5) | – | 1.32 s (100 members) | – |

With derivatives (`sim.function`, 64 members), per member:

| states | CPU, dense, one by one | CPU, dense, batch | CPU, sparse, one by one | CPU, sparse, batch | GPU, dense, batch |
|---:|---:|---:|---:|---:|---:|
| 141 | 0.16 s | 0.35 s | 0.60 s | 0.32 s | 0.046 s |
| 353 | 1.13 s | 1.76 s | 0.82 s | 0.65 s | 0.18 s |

Memory of a batch of 64 on the CPU: forward 0.7 GB (141 states, dense) to 4.7 GB (505,
sparse); with derivatives 4.6 GB (141, dense) to **22 GB (353, sparse)**: each member
carries its step record for the replay.

Several CPU cores (`jax.shard_map` over CPU devices, `XLA_FLAGS=--xla_force_host_platform_device_count=64`;
JAX only, no TenQSP API): 64 members at 141 states with derivatives, one per core: 1.6 s,
against 8.6 s one by one (5.4x).

## 3. The project lead's question: does the dense tier win for populations near the threshold?

**On the CPU, no.** Batched, the sparse tier is as fast as the dense tier at 141 states and
2.7x (353) to 5x (505) faster above the threshold: batching removes most of the host calls'
fixed cost (§4.5 finding 1), while the dense tier's batch recomputes every member's LU
whenever one member needs it. The threshold of 250 stays right for batches on the CPU.

**On a GPU, yes, up to several hundred states.** The dense tier runs on the GPU; the sparse
tier's host calls go back to the CPU at every Newton iteration. On the A6000 the dense
tier's batch beats every CPU option up to ~500 states (8–14x at 141, 4–5x at 353, 2.6x at
505) and loses at 961 (the A6000 computes in double precision at 1/32 of its single
precision speed). So on a GPU the threshold should be higher: somewhere between 505 and 961
on this card (§5, decision 2; measured in §6: 500).

## 4. PRD §10, the differentiable rows (re-measured; table in `docs/benchmarks.md`)

> Re-measured at the Phase 4 gate after the step 4.7 fixes: `docs/benchmarks.md`, "PRD §10,
> differentiable rows" (gradient 4.9x; 1,000 sets on the CPU 333 s, 4.1 GB).

| target | measured | |
|---|---|---|
| Differentiable solve, dense tier: ≤ 5x scipy at 1k states | dense tier 10.7x at 961 states (6.2 s vs 0.58 s); the tier the size picks (sparse) 1.4x (0.80 s) | met by the automatic tier |
| Gradient, sparse tier, 10^4 states: ≤ 3x the forward solve | 62 s vs 12.0 s (5.2x the JAX forward solve, 7.0x scipy's) | **not met** |
| 1,000 sets at 500 states without rebuilding, CPU and GPU | GPU: 155 s (505 states, dense, forward); CPU: §5 | needs decision 1 |

## 5. Decisions for the project lead

1. **A forward-only JAX function, for populations** (an API addition). Today the
   only JAX function is `sim.function`, which records every run for its derivatives: per
   member the record is 21 MB at 141 states, 104 MB at 353 and 149 MB at 505 (room for 4x
   the steps of the run at `f.x0`), so 1,000 members at 505 states need ~150 GB; and it
   costs 2–3 forward solves. A simulation of a population needs only the outputs.
   - **A.** `sim.function(scenario, wrt=..., derivatives=False)`: the same function
     (`f(x)`, `f.names`, `f.x0`, `jit`, `vmap`), no record, no replay; `jax.grad` of it raises
     a clear error. **Recommended**: one name to learn, and 4.8's batched scenarios build
     on it.
   - **B.** A separate `sim.forward(scenario, wrt=...)`.
   - **C.** Nothing new: populations through `sim.function` in chunks that fit in memory.
2. **The tier on a GPU.** Your rule stays automatic. Recommended: on a GPU, dense up to a
   higher threshold, measured as on the CPU (between 505 and 961 on the A6000; I would
   measure 600–900 and set it), and the CPU threshold of 250 unchanged. Alternative: one
   threshold everywhere (the GPU then gives up a 2.6–5x gain from 250 to ~500 states).
3. **Several CPU cores.** On the CPU, one batch runs on one core; populations gain most from
   spreading members over cores (5.4x on 64 cores here). Recommended: document the JAX
   recipe now (above), and let 4.8's batched scenarios do it for the user (an option there,
   shown to you with the 4.8 sketch).
4. **The gradient target at 10^4 states** (≤ 3x the forward solve; the PRD allows adjusting
   it after step 4.3). Measured 5.2x. Recommended: set it to ≤ 10 forward solves (what step
   4.5 measured, 6–10x, across sizes), and keep the step 4.5 option C (keep only what each
   step needs in the record) for when a real model needs better.

## 6. Project lead decisions (2026-10-26)

1. **A**: `sim.function(scenario, wrt=..., derivatives=False)`. Built: no record, no replay,
   steps chosen for the solution alone; `jax.grad`/`jacfwd`/`jacrev` of it raise. Compiled
   memory per member: 0.3 MB against 202 MB (61 states), 1.3 MB against 464 MB (141 states);
   16 members at 141 states 5x faster than with derivatives. Default `True`: existing code
   unchanged. Tests: `tests/test_step4_6_batches.py` (closed forms on both tiers, the error,
   memory).
   - When derivatives are needed (the project lead's question): fitting with a gradient-based
     optimizer, local sensitivity and identifiability, gradient-based NLME or MCMC. These
     usually differentiate one parameter set per call, so the record is that of one run
     (tens to hundreds of MB), not of a population; the ~150 GB figure is 1,000 members
     *with* derivatives in one batch.
2. **Accepted**: the tier stays automatic, with a GPU threshold above the CPU's 250 (one
   device per run, so one threshold applies). Measured on the A6000 (forward, seconds per
   member; `benchmarks/step4_6_batches.py gpu`):

   | states | members | dense | sparse |
   |---:|---:|---:|---:|
   | 353 | 64 | **0.077** | 0.16 |
   | 505 | 64 | **0.18** | 0.22 |
   | 601 | 64 | 0.32 | **0.25** |
   | 601 | 1,000 | **0.25** | 0.30 |
   | 705 | 64 | 0.49 | **0.29** |
   | 793 | 64 | 0.66 | **0.34** |
   | 913 | 64 | 0.95 | **0.39** |

   The tiers cross at ~550 states (64 members) to ~650 (1,000), within ~20% of each
   other there; the gain is below ~450 (2x at 353). Set: `SPARSE_FROM_STATES_GPU` = 500
   (`jax_backend.tier`, by `jax.default_backend()`). The sparse tier on a GPU machine is
   also faster than on the CPU alone (0.25 s per member at 601 states against 0.40 at 505),
   the rest of each step running on the GPU.
3. **Accepted**: the multi-core recipe documented (`docs/simulation.md`); an option offered
   with step 4.8's batched scenarios.
4. **Accepted**: PRD §10, the 10⁴-state gradient target is ≤ 10 forward solves (amended).
   Option C of step 4.5 (keep only what each step needs in the record) is kept for later.
