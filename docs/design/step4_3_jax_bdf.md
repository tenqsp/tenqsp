# Step 4.3: a JAX BDF integrator (dense tier). Design note

Status: **accepted** by the project lead 2026-10-24 (decisions 1–3 in §9 agreed as written). Follows ADR 0001
(decision 3) and PRD v0.6 §6.3. Gradients are step 4.4 and the sparse tier is step 4.5, but
this step must leave room for both, so they are sketched here too.

## 1. What it is for

The scipy backend stays the default for plain simulation. ADR 0001 found it the fastest at
every size, and it is robust: Radau fallback, events and the ledger. The JAX backend is for
the things scipy cannot do:

- **gradients** of outputs with respect to parameters (step 4.4; the estimation library,
  local sensitivity in 4.8);
- **batches**: `vmap` over parameter sets, regimens and scenarios, and the GPU for small
  models (step 4.6);
- **the inside of a larger JAX program**, such as an optimizer that is itself jitted.

So 4.3 must give the **same answers as scipy** (V9). It must also be built so that `jit`,
`vmap` and differentiation pass through it.

## 2. The algorithm: scipy's BDF, written in JAX

We copy the algorithm of `scipy.integrate.BDF` (Shampine & Reichelt 1997, the MATLAB
`ode15s` method), rather than inventing one:

- variable order 1–5, with the NDF coefficients (κ) scipy uses;
- quasi-constant step size, with the solution history held as a difference array `D` of
  shape (order + 3, n); a step-size change rescales `D` (scipy's `change_D`);
- simplified Newton: the iteration matrix `I − c·h·J` is factorised once and reused, with
  at most 4 iterations and scipy's convergence-rate test;
- the Jacobian is recomputed only after a Newton failure. The factorisation is redone when
  `h` or the order changes;
- error control and order selection as in scipy (error estimates at orders k−1, k and k+1;
  safety factor 0.9; change factor between 0.2 and 10).

Matching scipy step for step has two benefits. It is a strong test (V45: same step counts
and errors, up to round-off). It also means the two backends make the same accuracy
trade-offs, so users never have to wonder which one is "right".

**Inside JAX.** The integrator is one `lax.while_loop` over steps. All its state is fixed-
shape arrays: t, h, order, D, the LU factors, the Jacobian, the event bookkeeping and the
counters. There is no Python in the loop, so it compiles once per model structure.

**The Jacobian** is built from the existing coloured sparse Jacobian: the same jvps as the
scipy path (10–20 instead of n), densified for the dense tier. The dense-tier LU uses
`jax.scipy.linalg.lu_factor` / `lu_solve`. The linear solver is one small interface
(`factor(M) → F`, `solve(F, b)`, `solve_transposed(F, b)`), so 4.5 can plug in the host
SuperLU.

## 3. Doses, conditions and events (everything the scipy backend does)

| Feature | scipy backend (today) | JAX backend |
|---|---|---|
| Boluses | segment boundaries; jump at the boundary | same. Times and amounts are arrays in fixed **slots** (unused slots at +∞), so regimens up to the capacity share one build |
| Infusions | constant input between boundaries | same, in slots (start, stop, index, rate) |
| Conditions on time (ADR 0006) | switch times on the host (closed-form roots; sampled conditions polished with `brentq`) | computed **in JAX**, since under `grad`/`vmap` the parameters are not numbers. Closed-form roots are lambdified with `jnp`. Sampled conditions use the same grid, sign changes, and a fixed number of bisection steps to round-off. They fill slots as well |
| State-dependent switches | terminal scipy events, direction-aware | root-finding on the step's interpolating polynomial (as scipy does), with Illinois/regula falsi to round-off. On a root the step is cut there and the integrator restarts at order 1. The direction bookkeeping is carried in the loop |
| Extinction groups (V17) | terminal events; zero the group at the exact crossing; arming; noise-floor slots | the same rules, as events in the loop; armed flags and floor slots carried |
| Ledger (§5.8) | quadrature variables with infinite atol | identical construction and tolerance rescaling |
| Output times | `t_eval` | interpolation at the requested times as steps pass them (scipy's dense output). The times are an array in a padded slot of fixed capacity |
| Failure | Radau retry, then `SolverError` | no Radau inside JAX. A run returns a **status** (ok / step size underflow / too many steps / too many events). A single run raises `SolverError`, naming the segment and suggesting `backend="scipy"`. In a batch, a failed member gives NaN outputs and its status, and the others are unaffected |

**Capacities.** JAX needs fixed array sizes, so the backend uses fixed slots for doses,
infusions, switch times and output times. The slot sizes are rounded up to powers of two (8,
16, 32, …). A run that needs more slots causes one rebuild at the next size. In practice a
model is built at most a handful of times. `summary()` reports the capacities.

## 4. What gradients will mean (step 4.4; fixed now so 4.3 records what 4.4 needs)

- **Forward mode** (`jax.jvp` / `jacfwd`) works through `lax.while_loop` directly. It suits
  a few parameters.
- **Reverse mode** (`grad`, for many parameters) cannot differentiate a `while_loop`. 4.3
  therefore records the **step history** in fixed buffers: the accepted (t, h, order) and
  the states. With this history, 4.4 can run the **discrete adjoint** backwards, one step at
  a time, with the transposed solve. The result is the exact gradient of the computed
  solution, with no second accuracy question. The buffers are sized by `max_steps`
  (default 100,000 steps); for the dense tier (≤ ~2,000 states) the states take at most a
  few hundred MB. 4.4 may switch to checkpointing if that is too much.
- The step-size and order choices are treated as constants when differentiating (standard
  practice; the alternative differentiates the error controller and gives noisy
  gradients).
- Events: the derivative is taken with the event time held fixed, and
  `explain()` will say so.
- Test V8: gradients against central finite differences (rel. error < 1e-4) with tight
  tolerances, including across a dose and an extinction event (V46 in 4.8).

## 5. How it is chosen (API)

- `m.solve(..., backend="jax")` and `m.simulator(..., backend="jax")`, explicit for forward
  runs. The default stays `"scipy"`.
- The gradient and batch functions of 4.4–4.6 (e.g. `sim.gradient(...)`,
  `sim.batch(...)`) use the JAX backend by themselves, since scipy cannot do them.
- The same options as scipy: `method="BDF"` (the only one), `rtol`, `atol`, `max_step`,
  `ledger`, plus `max_steps`.

## 6. Tests

- **V45, order and agreement with scipy.**
  - On linear test problems (`y' = A y` with known eigenvalues, stiff and non-stiff), each
    fixed order 1–5 reaches its convergence order.
  - With adaptive order, the number of accepted steps is within 5% of scipy BDF, and so is
    the error against the exact solution.
- **V9, backend agreement.** Every V1–V6 model and every gallery model is solved by both
  backends. Outputs agree to within 10 × rtol (relative), the ledger closes to the same
  accuracy, and extinctions happen at the same times (to the event tolerance).
- **Events.**
  - A switch at a closed-form time and a sampled one, against scipy.
  - A state-dependent switch with a known crossing time.
  - Extinction at an exact threshold crossing (V17 cases).
  - A large bolus (V14).
- **Compile once.**
  - 100 parameter sets, regimens up to the capacity, and output times up to the capacity
    give one build.
  - The next capacity gives exactly one more.
- **Under `jit` and `vmap`**: a smoke test here; the full batch tests are in 4.6.
- Optional: run the whole fast suite a second time with `backend="jax"` forced
  (`pytest --backend=jax`), like `--roundtrip`. This catches anything the targeted tests
  miss. Runtime TBD.

## 7. Risks and what we do about them

1. **Compile time.** The integrator, RHS, Jacobian and events all go into one jitted
   program. I expect 5–30 s per model structure, and will measure and report it. If it is
   too slow, the pieces can be compiled separately and called from the loop.
2. **Per-step overhead for tiny models.** `while_loop` iterations are cheap, but each one
   does a dense LU. For 2–50 states this should beat scipy's Python step loop. For ~2,000
   states the LU (≈ 2.7 GFLOP) dominates. 4.6 measures both.
3. **Matching scipy exactly** is fiddly: order selection, the Newton tolerance, the step
   after a restart. The first sub-step is a close port with V45, before any events are
   added.
4. **The size of the event machinery.** It is the biggest part of the scipy backend
   (≈ 300 lines), and porting it into a jittable loop is most of the work. Sub-steps
   isolate it (below).

## 8. Sub-steps (each with tests, committed separately)

- **4.3a** BDF core: fixed span, no events, dense LU, interpolated outputs. V45 against
  scipy and exact solutions.
- **4.3b** Segments: boluses, infusions and output times in slots; the ledger; capacities
  and compile-once tests.
- **4.3c** Conditions on time computed in JAX (closed-form roots and sampled), then
  state-dependent switches as in-step root finding.
- **4.3d** Extinction groups and noise floors; statuses and failure reporting;
  `backend="jax"` in `solve`/`simulator`. V9 over all V1–V6 and gallery models.
- **4.3e** Hand-test, benchmarks (compile time; run time against scipy at several sizes),
  docs.

Estimate: two to three weeks of work at the pace of Phases 2–3. The rest of the 3–4 weeks
in ADR 0001 is 4.4 (gradients) and 4.5 (sparse tier).

## 9. Decisions for the project lead

1. **Backend choice**: scipy stays the default for plain runs, and JAX is chosen with
   `backend="jax"` or used automatically by gradient and batch functions. Agree?
2. **Failure behaviour**: no automatic Radau fallback inside JAX. A single run raises an
   error that suggests `backend="scipy"`; a batch marks failed members (NaN + status)
   instead of stopping. Agree?
3. **Capacities**: fixed slots rounded up to powers of two, with one rebuild when a run
   needs more. Agree?
