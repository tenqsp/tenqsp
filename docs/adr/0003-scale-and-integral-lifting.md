# ADR 0003 — scipy BDF at 10⁵ states, and lifting integrals out of the Jacobian

- **Status:** Accepted by the project lead, 2026-10-10 (Phase 0.5 spike (c), 2026-10-10).
- **Spike code and raw data:** `spikes/scale/` (`run_c.py`, `toy_model_lifted.py`,
  `breakdown_c.py`, `results.jsonl`, `handtest_c.py`).

## Context

ADR 0001 made scipy BDF with a colored sparse Jacobian the default forward backend. It also
found that integral couplings (`B_tot` in logistic crowding, total lysis and adsorption in
the phage equation) make whole rows dense. That made the color count, and so the cost of one
Jacobian, grow with the grid (≈ 2·n_z + 1). Open questions:

1. Does scipy BDF still work at the PRD's 10⁵-state target (§10)?
2. Can integrals be "lifted" so the color count stays bounded, as the PRD requires?

## Method: lifting

The RHS is written as `f(y) = g(y, q)` with `q = W·y`, the vector of integrals the model uses.
All three in the toy model are linear in the state. Then

    J = ∂g/∂y  +  (∂g/∂q) · W

- `∂g/∂y` is local (each cell sees itself and its stencil neighbours), so the color count is
  bounded.
- `(∂g/∂q)·W` has rank equal to the number of integrals (3 here). It costs 3 jvps plus
  3 vjps, independent of grid size.

The IR declares every integral explicitly (`int_<axis>(...)`), so the compiler can do this
automatically. `handtest_c.py` shows it on a 4-compartment example worked by hand.

## Results

Same toy model as ADR 0001: 24 h, rtol 1e-6, atol 1e-3. Runs were **sequential** (no
concurrent load).

| states | mode | colors | 1 Jacobian | Jacobian nnz | BDF solve | nfev / njev / nlu | peak RSS |
|---:|---|---:|---:|---:|---:|---|---:|
| 961 | unlifted | 61 | 0.7 ms | 8.2k | 0.4 s | 865 / 3 / 76 | 0.5 GB |
| 961 | **lifted** | **8** | 0.8 ms | 8.2k | 0.4 s | same | 0.5 GB |
| 10,201 | unlifted | 201 | 10.5 ms | 90k | 7.7 s | 1376 / 6 / 124 | 0.8 GB |
| 10,201 | **lifted** | **8** | 3.3 ms | 90k | 7.6 s | same | 0.5 GB |
| 29,241 | unlifted | 341 | 40 ms | 261k | 32.6 s | 1654 / 3 / 150 | 1.8 GB |
| 29,241 | **lifted** | **8** | 6.7 ms | 261k | 34.3 s | same | 0.9 GB |
| 99,856 | unlifted | 631 | 342 ms | 894k | 167.5 s | 2010 / 3 / 188 | 7.9 GB |
| 99,856 | **lifted** | **8** | 19.5 ms | 894k | 170.9 s | same | 2.4 GB |

- **Correctness.** The lifted Jacobian equals the dense `jax.jacfwd` oracle to 2e-15 relative
  (checked at 961 states). Lifted and unlifted trajectories agree to 2e-13 at 961 states and
  3e-10 at 99,856.
- **Where the time goes at 99,856 states** (`breakdown_c.py`):
  - one RHS evaluation: 0.64 ms;
  - one sparse LU of `I − γJ`: 0.63 s (fill-in 0.9M → 6.0M nonzeros);
  - one LU solve: 12 ms.

  The 188 factorizations take about 118 s of the 171 s, roughly 70%.
- Pattern detection by brute-force probing (a spike crutch; the library derives the pattern
  from the IR) took 323 s at 10⁵ states. It is excluded from the solve time.

## Findings

1. **scipy BDF meets the 10⁵-state target:** about 3 minutes for a 24 h simulation and
   2.4 GB of memory, on one CPU process.
2. **Lifting holds the color count at 8 regardless of grid size** (vs. 631 unlifted at 10⁵).
   One Jacobian becomes 17× faster and peak memory 3.3× lower.
3. **Total solve time is unchanged**, because scipy BDF reuses its Jacobian (only 3–6
   evaluations per run). The Jacobian is not the bottleneck for forward simulation.
   Lifting matters for memory, and for any setting that re-evaluates Jacobians often:
   gradients, sensitivities, and the custom JAX BDF of ADR 0001.
4. **The bottleneck at 10⁵ states is sparse LU factorization**, partly from fill-in. Part of
   that fill-in is the dense `B×B` block the rank-3 correction creates (2·n_z = 630 squared).
5. **Correction to ADR 0001.** Its spike-(a) timings at 10k and 29k states (70 s and 46 s)
   were inflated by concurrent runs. Sequential runs give 7.7 s and 33 s. The ranking and the
   decision in ADR 0001 are unaffected; diffrax figures from that spike are probably also
   inflated, by a similar factor.

## Decision

1. **The compiler lifts every integral** into an explicit auxiliary quantity `q`. The Jacobian
   is assembled as a local colored part plus a low-rank integral part. This replaces the
   "dense rows recovered by vjp" approach, which stays only as a fallback for couplings the
   IR cannot classify.
2. **Keep scipy BDF for forward simulation up to 10⁵ states.** PRD §10's "≥ 10⁵ states on the
   scipy sparse path" is confirmed.
3. **Performance work for large models, in the custom JAX BDF (Phase 4), should target the
   factorization:**
   - solve with `I − γ·J_local` plus a Woodbury correction for the integral part, so the
     dense block never enters the LU;
   - reuse factorizations across steps;
   - try alternative orderings.

   Not needed for v1.0 forward simulation.
4. **Every benchmark from now on runs sequentially** on an otherwise idle machine, and
   records that it did.

## Consequences

- The IR must mark which terms depend on integrals (it already declares them; PRD §9
  `IntegralSpec`).
- Nonlinear integrals (e.g. `int_a(h(a)*I^2)`) still lift the same way: `W` becomes the
  Jacobian of `q` with respect to `y`, computed by vjp, one per integral.
- PRD §10 colour target "≤ ~30, independent of grid size" is met (8 here).

## Project lead decisions (2026-10-10)

**10⁵ states is the v1.0 target.** The project lead's expected use:
- axes of 5–10 bins (latency, dormancy);
- ≤5 phages, i.e. ≤5 binary susceptibility axes;
- organs as a categorical axis (PBPK-PD);
- biofilm sites;
- <5 nutrient sources.

State count is the product of a field's axis sizes:

| Scenario | B | I (age × dormancy × organ × genotype × infecting phage) | Total |
|---|---:|---:|---:|
| Typical: 3 phages, dormancy 5, 10 organs, one biofilm site | ~800 | ~6,000 | ~7k |
| Heavy: 5 phages, dormancy 10, 15 organs | 4,800 | 240,000 | ~250k |
| Naive: as heavy, with a 20-layer biofilm axis on every organ | 96,000 | 4.8M | ~5M |

To keep realistic models within 10⁵, the design must never force a full tensor product where
one is not needed. Requirements:

1. **State-count report (Phase 1).** `summary()` / `explain()` give the state count per field
   and the axis product driving it, and warn above 10⁵.
2. **Fields on part of a structure (Phase 2).** A structure that exists only at some sites
   (e.g. a biofilm depth axis at one implant) is modelled as its own field, coupled by
   exchange terms. It is not an axis on every organ. Make this convenient and documented.
3. **Reachable-genotype axis (option).** A flattened categorical axis listing only the reachable
   genotypes, as an alternative to the full 2ⁿ product of per-phage axes (PRD §3 already
   allows a flattened axis).

Models well beyond 10⁵ states are a Phase 4 concern: the custom JAX BDF, Woodbury solves, and
faster factorization.

