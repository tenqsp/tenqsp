# Step 4.4: gradients through the JAX solver. Design note

Status: **accepted** by the project lead 2026-10-25 (decisions 1–3 in §8 agreed as written; PRD v0.6
§6.3 and V8 amended accordingly). Follows PRD v0.6 §6.3,
V8, and the sketch in the step 4.3 design note (§4), which this note revises where 4.3 taught
us something.

## 1. What it is for

A gradient answers "if this parameter moves a little, how much does each output move?".
Three users need it:

- **estimation** (the sister library `tenqsp-fit`): an optimizer needs the gradient of an
  objective (e.g. -2 log-likelihood) with respect to all estimated parameters, many times;
- **local sensitivity** (step 4.8): d(output)/d(parameter) over time, for a few parameters,
  often normalised (d log y / d log θ);
- **anything else written in JAX** that calls a TenQSP model inside it (a jitted optimizer,
  an optimal-design criterion).

The gradient has to be **right** (V8: against finite differences and closed forms) and
**cheap**.

## 2. Which derivative: of the computed solution

There are two choices. *Optimise-then-discretise* solves a second ODE (the continuous adjoint)
with its own tolerances, and its gradient is only approximately the gradient of what the
solver computed; optimizers then see small inconsistencies between the objective and its
gradient. *Discretise-then-optimise* differentiates the solver's own arithmetic: the gradient
is exact for the numbers the solver produced. We use the second (as planned in 4.3, and as
diffrax, Stan and most estimation tools do).

One simplification, as planned: the step sizes and orders the solver chose are held fixed
when differentiating. Differentiating the error controller itself gives noisy, useless
gradients. This is standard practice.

## 3. How: record the run, then replay it differentiably

4.3's solver is a `while_loop` that decides its own steps. JAX can push forward-mode
derivatives through such a loop, but not reverse mode (needed for many parameters). Writing
the adjoint of every part by hand (BDF steps, order changes, dense output, doses, events) would
be a large amount of error-prone code. Instead:

1. **Forward run (4.3, unchanged in substance).** It now also records, for every accepted
   step: the step's position, its order, the difference array `D` it started from, the
   step-size changes applied to `D` before and after it, and its Newton correction. It also
   records each segment: where it starts, its first step, the parameter slots in force,
   and whether it ended at its boundary or at an event (with the groups zeroed there).
2. **Replay.** A second function walks the recorded steps with `lax.scan` (a fixed-length
   loop, which JAX can differentiate in both directions). It does no error control, no step
   rejection and no Newton iteration: the solution values are taken from the record, while
   the derivatives flow through the replayed arithmetic. What it adds is the **derivative
   of each step**: a BDF step solves an implicit equation
   G(y_new; history, θ) = 0, and its derivative comes from the implicit function theorem,
   one linear solve with the matrix I − c·h·J at that step. This is wrapped as a custom JAX
   derivative of one step, so JAX assembles the rest (forward mode, reverse mode, `jit`,
   `vmap`) by itself. Reverse mode through the scan *is* the discrete adjoint, derived by JAX
   rather than by hand.

Costs, per gradient:

- the forward run (as today);
- the replay: one Jacobian and one LU **per step** (the forward run reuses one LU across many
  steps; exact derivatives need the Jacobian at each step). For the toy phage model at 961
  states that is ~450 LUs against 106, so I expect a gradient to cost 2-5× a forward solve on
  the dense tier, close to the PRD's 3×. If it is too slow, the stale LU of the forward run
  plus a few corrections gives the same answer to round-off; that is an optimisation for
  later, measured first.
- Reverse mode, in addition, stores each step's `D` (≤ 8 × n numbers). At 1,000 states and
  1,000 steps that is 64 MB. If a run is too long for memory, JAX's checkpointing
  (`jax.checkpoint` every k steps) trades memory for recomputation; we add it only if needed.
- The record has a fixed size: the number of steps, rounded up to a power of two, as with
  dose slots. A run needing more steps than ever before builds the replay once more.

**Found in 4.4a (2026-10-25).** Taking the values from the record matters. A first version
recorded only each step's Newton correction and rebuilt the values from it. That is
unstable: each step then extrapolates from the previous ones, and at orders above 1 this
amplifies round-off (0.4 off a peak of 100 after ~400 order-5 steps). Re-solving each step
by Newton is stable, but agrees with the run only to its Newton tolerance, and solutions that
grow from tiny values (a ghost phage, a seeded mutant) amplify even that (up to 300
tolerances in the test suite). With the difference array recorded, the replay equals the run
to round-off on every JAX solve of the test suite (`pytest --backend=jax --replay`).
Memory: 8 numbers per state per step, the same that reverse mode keeps anyway.

## 4. Doses, switches and events: moving boundaries

This is the one place where I propose to change the PRD.

PRD v0.6 §6.3 says gradients across an event are taken "with the event time held fixed".
For time-defined boundaries this gives **wrong gradients**, and for some parameters the
gradient would be exactly zero:

- `kin if t > t_on else 0`: with the switch time held fixed, d(output)/d(t_on) = 0. An
  optimizer could never estimate `t_on`.
- a dose lag time (NONMEM's ALAG), an infusion duration, a dose time: the same problem.
- the saturable switch `C > 2`: when the crossing happens depends on the parameters. Holding
  it fixed gives the wrong slope after the crossing.

The exact derivative is cheap to get, so I propose to **let boundaries move**:

- **Time boundaries** (doses, infusion starts and stops, switch times of conditions on time):
  the replay records each step's position *as a fraction of its segment*, so when a segment's
  ends move with θ, its steps stretch with them. Differentiating through that stretching gives
  the exact derivative with respect to dose times, lag times, durations and `t_on`.
- **State events** (a condition on a state switching; an extinction crossing): the event
  time t* solves g(y(t*), θ) = 0. Its derivative follows from the same implicit function
  theorem: dt*/dθ = -(∂g/∂θ + ∂g/∂y · dy/dθ) / (dg/dt). It is a few lines, wrapped like the
  step. After an extinction the group is zero for every θ nearby, so its derivative there is
  zero, as it should be.
- **What stays structural** (no derivative, and changing it may rebuild): how many doses
  there are, which compartment they go to, and the *order* in which boundaries and events
  happen. A parameter change big enough to reorder them is a different model run, and the
  gradient is only valid on one side. Finite parameter changes are fine; the gradient simply
  describes the current order.
- Conditions whose switching is not smooth at all (`abs`, `max`, `min` inside the
  right-hand side, without a switch event) are differentiated as JAX does: one side's slope
  at the kink.

## 5. What you can differentiate

- every parameter value (including parameters used in initial values, volumes, weights);
- dose amounts and dose times of a regimen; infusion rates, amounts and durations;
- initial values given at simulation time.

Not in 4.4: derivatives with respect to grid sizes or axis ranges (structural), and the
scipy backend (it cannot differentiate; models above 1,000 states wait for 4.5 and its
forward-sensitivity fallback).

## 6. API sketch (project lead approval needed for anything public)

```python
sim = m.simulator(outputs=["conc"], backend="jax", rtol=1e-8)

# 1. a plain JAX function, for tenqsp-fit and any JAX code
f = sim.function(scenario, wrt=["CL", "V", "ka"])   # f(x) -> dict of output arrays
x0 = f.values                                       # current values of CL, V, ka
jax.grad(lambda x: loss(f(x)["conc"], data))(x0)    # the caller builds the objective
jax.jacfwd(f)(x0)                                   # or any JAX transform, jit, vmap

# 2. a ready-made sensitivity table (the basis of 4.8's local sensitivity)
s = sim.sensitivity(scenario, wrt=["CL", "V"], relative=False)
s["conc"]["CL"]      # d conc / d CL at each output time, shape (n_t,)
```

- `wrt` names parameters, or regimen quantities such as `"dose[0].amount"`,
  `"dose[0].time"`, `"infusion[0].duration"` (exact spelling to be settled).
- `sensitivity` chooses forward mode (few parameters, the usual case for sensitivity) and
  `function` leaves the choice to the caller.
- A failed run (status not OK) returns NaN outputs and NaN gradients, with the status, so an
  optimizer can back off rather than crash. A single `sensitivity` call raises SolverError
  as `solve` does.
- The ledger is off in these functions (it is not differentiated); `solve` keeps it.

## 7. Tests (V8)

- **Closed forms**: one-compartment oral PK, d conc / d(ka, CL, V, dose amount, dose time);
  the `t_on` switch; the saturable `C > 2` switch (derivative through the crossing time);
  extinction time. Relative error < 1e-6 at tight tolerances.
- **Finite differences**: field models (the toy phage model, transit chains, the three-axis
  toy) against central differences at rtol 1e-10, relative error < 1e-4.
- **Forward = reverse**: forward- and reverse-mode gradients agree to 1e-10 (same discrete
  derivative, two routes).
- **The replay reproduces the run**: replayed outputs equal the forward run's to round-off.
- **Compile once**: 100 gradient calls with new values: one build of the replay.
- **A sweep like `--jacobian`**: `pytest --gradients` checks, for every model a test
  compiles (up to a size limit), the gradient of one output against finite differences.
  Runtime to be measured; in CI only on the full-matrix runs if slow.
- **Cost**: gradient time against forward time at several sizes, in `docs/benchmarks.md`.

## 8. Decisions for the project lead

1. **Derivatives move with event and dose times** (§4), instead of the PRD's "event time held
   fixed". This changes PRD §6.3 and the V8 wording. Recommended: without it, `t_on`, lag
   times and dose times cannot be estimated.
2. **The API in §6**: `sim.function(...)` (a JAX function for the fit library and other JAX
   code) and `sim.sensitivity(...)` (a table of derivatives). Recommended: both. The plots
   and normalised sensitivity reports stay in 4.8.
3. **Failed runs give NaN** in `function` (so a batch or an optimizer can carry on), and
   raise in `sensitivity`, as in 4.3. Recommended.

## 9. Sub-steps (each with tests, committed separately)

- **4.4a** Record the run (steps, orders, `D`, segment ends, events), and a replay that
  reproduces the forward outputs. No derivatives yet.
- **4.4b** The custom derivative of one BDF step; forward and reverse mode on runs without
  doses or events; V8 against the closed-form PK model with the model's own dose at t = 0.
- **4.4c** Moving boundaries: doses, infusions, switch times on time (stretching segments);
  state events and extinctions (implicit event-time derivative). Closed-form tests.
- **4.4d** The API (`sim.function`, `sim.sensitivity`), NaN on failure, compile-once;
  `pytest --gradients`.
- **4.4e** Hand-test, benchmarks (gradient cost against the forward solve), docs.

Estimate: two to three weeks.

## 10. Risks

1. **The record and the forward run must agree exactly** on where every step, segment and
   event sits, or the replay differentiates a slightly different run. 4.4a's test (replay =
   run to round-off) guards this before any derivative is written.
2. **Cost of one LU per step** (§3). Measured in 4.4e; the stale-LU refinement is the
   fallback.
3. **Finite differences are a noisy oracle** for adaptive solvers: a parameter nudge can
   change the steps the solver takes. Hence closed forms first, tight tolerances, and
   forward = reverse as an independent check.
4. **Memory for long runs** in reverse mode (§3): checkpointing if needed.
