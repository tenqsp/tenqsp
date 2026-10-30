# Step 4.4d: lag, bioavailability, and the gradient API. Sketch

Status: **accepted** by the project lead 2026-10-25. `lag` and `bioavailability` approved as
written in §1; §2–§4 (gradient API, `wrt=` names) and the three recommendations in §5
agreed as written.

## 1. Lag time and bioavailability (approved)

```
param tlag = 0.5, F = 0.8
lag depot = tlag                 # every dose into depot arrives tlag later
bioavailability depot = F        # and only the fraction F of it arrives
```

Python: `m.lag(depot, "tlag")`, `m.bioavailability(depot, "F")`.

- They apply to **every** dose into that target: the model's own doses and those of a
  regimen given at simulation time, boluses and infusions. A bolus of 100 at t = 0 arrives
  as 80 at t = 0.5. An infusion is shifted by the lag (start and stop) and its rate scaled
  by F, as in NONMEM.
- They are formulas of parameters (not of states or t), like a dose amount. The lag must
  be ≥ 0 and F ≥ 0; a value outside that is an error when the model is run (like a
  negative volume). F > 1 is allowed (it happens with saturable first-pass), without a
  warning.
- Changing their *values* never rebuilds; they are ordinary parameters. So `tlag` and `F`
  can be estimated, with exact gradients (4.4c: dose times move with the parameters).
- A dose whose arrival falls after the end of the run is not given. The ledger books each
  dose when and as it arrives (amount × F), and the part lost to F as
  "not absorbed" (so the books still close against what the regimen gave). `sol.doses`
  lists arrival time and amount arrived.
- `explain()` says: "doses into depot arrive tlag later, and F of each arrives".
- Targets: a compartment, or a field (then every dosed element of it).
- The model's text writer and reader handle both; IR version bump with a migration test.

## 2. `sim.function`: a JAX function, for the fitting library and other JAX code

```python
sim = m.simulator(outputs=["conc"], backend="jax", rtol=1e-8)
f = sim.function(scenario, wrt=["CL", "V", "ka", "tlag"])

f.names      # ("CL", "V", "ka", "tlag")      what each entry of x is
f.x0         # their values in the scenario: array([4., 30., 1.2, 0.5])
y = f(x)     # {"conc": array (n_t,)}           x: a 1-D array in f.names order

loss = lambda x: jnp.sum((f(x)["conc"] - data) ** 2)
jax.grad(loss)(f.x0)          # any JAX transform works: grad, jacfwd, jit, vmap
```

- **What f does at a new x**: it runs the solver at x (no derivative through its step
  choices), then replays that run to give the derivatives (4.4a–c). So f is a real
  function of x, not just "derivatives at one point", and it is pure JAX: it can sit
  inside `jax.jit`, `jax.grad`, `jax.vmap` and a JAX optimizer.
- **Fixed by the scenario** (a new scenario makes a new f): the outputs, output times,
  number of doses and which compartments they go into, and the things not in `wrt`.
- **Room for steps**: f records up to a fixed number of solver steps, set from the run at
  `f.x0` with room to spare (×4, a power of two). An x that needs more steps gives NaN with
  status "too many steps for the record" (a new `sim.function(...)` at that x makes room).
- **Failures**: a run that fails gives NaN outputs (and NaN gradients) instead of an error,
  so an optimizer can step back and a batch carries on (project lead decision 2026-10-25). The
  status: `f.status(x)` (`"ok"`, otherwise the reason in words; as built, 2026-10-25).

## 3. `sim.sensitivity`: a table of derivatives

```python
s = sim.sensitivity(scenario, wrt=["CL", "V"])                 # d conc / d CL, ...
s = sim.sensitivity(scenario, wrt=["CL", "V"], relative=True)  # d log conc / d log CL

s.t                  # output times
s["conc"]["CL"]      # shape (n_t,), or (n_t, *axes) for an output over axes
s.values["conc"]     # the outputs themselves
s.table()            # a pandas DataFrame: one row per time, one column per (output, name)
```

- `relative=True` gives "% change per % change" (dy/dθ × θ/y). Where the output is 0 it
  is undefined and given as NaN.
- It runs once at the scenario's values, forward mode (cheapest for a few parameters). A
  failing run raises `SolverError`, as `solve` does (project lead decision 2026-10-25).
- Plots of sensitivities come in 4.8.

## 4. The `wrt=` names

| name | means | example |
|---|---|---|
| a parameter | its value | `"CL"`, `"tlag"`, `"F"` |
| a parameter over a categorical axis | all its entries, one per label | `"k"` → `k[S]`, `k[R]` |
| one entry of it | that entry | `"k[R]"` |
| a dose of the regimen: amount or time | the n-th bolus, in time order | `"dose 1 amount"`, `"dose 1 time"` |
| an infusion of the regimen | its rate, amount, start or duration | `"infusion 1 duration"` |

- Dose and infusion names refer to a regimen given in the scenario (as built: a scenario
  using the model's own `dose` lines refuses them, and says to give the doses as a regimen
  or to differentiate the parameters the model's doses use).
- Doses and infusions are counted **from 1**, in time order, as in a dosing table
  (`print(scenario.regimen)` shows the numbers). The regimen's own dose numbers are only
  needed for regimen-specific questions ("what if the second dose were later?"); a lag or
  F, which act on all doses, are parameters.
- Changing one infusion quantity holds others fixed: `duration` keeps the rate (the stop
  moves), `rate` and `amount` keep the duration, `start` keeps the duration (both ends
  move).
- Not differentiable (structural): grid sizes and axis ranges, the number of doses, which
  compartment a dose goes into. Asking for one of these, or a name that does not exist,
  is an error naming the closest valid names.

## 5. Decisions (project lead, 2026-10-25: all three as recommended)

1. **Dose numbering from 1, in time order** (`"dose 1 time"`), as in a dosing table, or
   Python-style from 0 (`"dose[0].time"`). Recommended: from 1, with the words written out
   as above: it reads like a dosing record, and the people using it are modelers.
2. **`f(x)` takes a flat array** in `f.names` order (what optimizers use), or a dict by
   name. Recommended: the array, with `f.unpack(x)` → dict for reading.
3. **F lost before absorption in the ledger** booked as its own channel ("not absorbed"),
   so the books close against the doses as given. Recommended (otherwise the ledger shows
   only what arrived and the regimen's totals do not reconcile).

## 6. Found while building (2026-10-25)

1. **An event after a moving event** (fixed). When a segment's start moves (an earlier
   event or a lagged dose) its steps shift with it, and the replay read the state at the
   recorded event time from the step's interpolant. On the step that straddles the event,
   the interpolant's slope blends both sides of the kink, so the event time took 0.77 of its
   true derivative (two switches in a row, `C > MIC` then `C > MIC/2`). The state is now
   read at the event time moved with the segment, so the shift enters only through the
   true slope. Test: `test_jax_gradients.py::test_an_event_after_a_moving_event`. Found by
   `pytest --gradients`.
2. **An output time on a moving boundary is a kink.** There the derivative is one-sided
   (after the boundary, as outputs are reported after the doses there); central differences
   average the two sides. Not a fault; the sweep leaves such times out.
3. **Steps chosen for the solution, not for its derivative** (decided 2026-10-25: option
   B, below; done). The
   derivative is exact for the computed solution, but the solver chooses its steps (and
   order) to keep the *solution* accurate. A solution at rest takes huge first-order steps:
   the Friberg model started exactly at its steady state with fixed numbers runs 500 h in
   15 steps, and d Circ(500) / d circ0 comes out 1.021 against 1.092 (6 %). With
   `max_step=1` it is 1.0917. The same happens in practice for a model started at a
   baseline from data (an indirect response with R(0) fixed and kin estimated). Likewise a
   solution that is itself inaccurate (values far below atol, then regrowing) has an
   equally inaccurate derivative (0.3 % in a test with atol 1e-3). The sweep counts these
   apart ("agree only with finer steps").

   Options put to the project lead: A, document it and leave `max_step` to the user; **B, let the
   derivative steer the steps** (chosen); C, check afterwards and refine. As built
   (`jax_bdf.bdf(sens=)`, `jax_backend._program(steer=True)`):
   - the run carries the derivative `s` along one direction of the `wrt` numbers (each
     moved in proportion to its value, 1 where it is 0, with fixed random signs and
     weights in [0.5, 1.5], so none cancels another), with its own difference array,
     rescaled and updated with the state's;
   - after each converged Newton solve, the derivative's corrector (linear) is solved with
     the same factorised matrix and iterated as Newton is; if it does not converge, that
     counts as a Newton failure (fresh Jacobian, then a halved step);
   - its error, with the state's `rtol` and `atol`, enters the error test and the order
     choice as the larger of the two norms (CVODES with `errconS`);
   - across a dose or infusion boundary it gets the dose's change and the jump of the
     right-hand side times how fast the boundary moves; across an event it is kept as it
     is (zero for a group zeroed there). It only chooses steps; the derivatives themselves
     still come from the replay, exactly.
   Applies to `sim.function`, `sim.sensitivity` and `jax_replay.record(direction=)` (the
   `--gradients` sweep). Ordinary runs (`sim(...)`) are unchanged. Result: the Friberg
   example gives 1.09186 at default tolerances (reference 1.09186 with `max_step = 1`);
   cost on an oral PK model +17 % steps; 4.4e measures it properly.
   Tests: `tests/test_step4_4d_steer.py` (a response at its baseline in closed form, the
   Friberg example, the carried derivative against the replay's across doses, lags, dose
   times and infusions).
