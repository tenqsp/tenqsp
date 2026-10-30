# Step 4.8: simulation and exploration — API sketch

Status: **accepted** (project lead, 2026-10-27: "go with these", all five recommendations in §5) and
**built**: `src/tenqsp/explore.py`, `src/tenqsp/plot.py`, `Scenario(initial=)`,
`Simulator.batch`/`pre_run`; tests `tests/test_step4_8_explore.py` (V46 included), hand-test
user guide `docs/simulation.md`.

## 0. What §8.3 asks for, and where it stands

| §8.3 item | state |
|---|---|
| Regimens at simulation time (bolus, infusion, `every=`, `n=`; target required) | *built*, step 4.1 |
| Local sensitivity, optionally d log y / d log θ | *built*, step 4.4 (`sim.sensitivity`); V46 still needs its own test (across a dose and an extinction) |
| Batched scenarios: parameter grid × regimens in one call, one labelled result | §1 |
| Steady state / pre-run (G13) | §2 |
| Plots (optional matplotlib) | §3 |

Virtual populations (random effects) and trial simulation stay out (estimation library,
§15.8); a population drawn by the user is just a list of scenarios (§1).

## 1. Batched scenarios

```python
reg = {"100 q12h": tq.bolus("depot", 100, start=0, every=12, n=7),
       "200 q24h": tq.bolus("depot", 200, start=0, every=24, n=4)}
grid = tq.grid(params={"CL": [3, 5, 8], "V": [40, 60]}, regimens=reg)   # 3 x 2 x 2 = 12
b = sim.batch(grid, times=np.linspace(0, 96, 193))

b["conc"]            # array (12, 193): scenario x time (x field axes, if any)
b.scenarios          # the 12 tq.Scenario, each named, e.g. "CL=3, V=40, 100 q12h"
b.table()            # pandas, long: one row per scenario x time x output, with columns
                     #   scenario, CL, V, regimen, time, output, value (NONMEM-like)
b.to_xarray()        # dims (scenario, time, ...), coords CL, V, regimen on scenario
b[3]                 # the 4th scenario's SimResult (as sim(...) gives)
b.failed             # scenarios whose run failed, with the reason (the rest still run)

sim.batch([sc1, sc2, sc3])          # any list of scenarios also works (a drawn population)
sim.batch(grid, n_jobs=8)           # several CPU cores (scipy: processes; JAX: §1.2)
```

### 1.1 What it does

- A grid is the full cross product, in the order given (last name varies fastest). Values
  a grid does not set come from the simulator's scenario and the model, as for `sim(...)`.
  Per-category values work as everywhere (`{"k": [{"S": 1}, {"S": 2}]}`).
- Every scenario of a batch must share the output times (one array). Scenarios given as a
  list may have their own times only if they all have the same number of times; otherwise
  `batch` says so (use one `sim(...)` per scenario).
- One failed scenario does not stop the batch: its outputs are NaN, `b.failed` lists it
  with the solver's message. (Decision 3.)

### 1.2 How it runs (the same results whichever way)

- `backend="scipy"` (the default): one solve per scenario; `n_jobs > 1` runs them in that
  many processes.
- `backend="jax"`: scenarios with the same shape (number of doses, infusions) are run as
  one `jax.vmap` call without derivatives (step 4.6); `n_jobs > 1` spreads each call over
  that many cores (`jax.shard_map`, the step 4.6 recipe, done for you).

## 2. Steady state and pre-runs (G13)

```python
pre = sim.pre_run(sc, until="steady")        # run until nothing changes any more
pre = sim.pre_run(sc, until=24 * 14)         # or for a fixed time (two weeks)
pre.state                                    # the end state, by name (like sol.unpack)
pre.t, pre.reached                           # how long it ran; whether steady was reached

sc2 = sc.replace(initial=pre)                # the next scenario starts from it
r = sim(sc2)                                 # t = 0 is the end of the pre-run
```

- "Steady" without doses: every state's rate of change is below its tolerance
  (|dy/dt| ≤ atol + rtol·|y|, the solver's own rtol/atol) for one doubling of the run
  time; it gives up at `t_max` (default 1e6 time units) and says so (`pre.reached` False,
  with a warning).
- "Steady" under a repeated regimen (a scenario with `every=` doses): two consecutive
  dosing intervals agree to the tolerance at the same point in the interval, i.e.
  pharmacokinetic steady state; the end state is taken just before the next dose.
- `Scenario` gains a field `initial=` (a pre-run or a mapping of state values), saved in
  its JSON. Doses in `sc2` are at times relative to its own start.

## 3. Plots (optional: `pip install tenqsp[plot]`)

```python
tq.plot.trajectories(r_or_b, "conc", log=True)          # one line per scenario
tq.plot.fan(b, "conc", by="regimen", q=(5, 50, 95))      # median and band per group
tq.plot.sensitivity(s, "conc", relative=True)            # heat map: parameter x time
tq.plot.snapshot(r, "B", axis="age", times=[0, 6, 24])   # a field along one axis
```

Each returns the matplotlib `Axes` (pass `ax=` to draw into your own), with axis labels
from the model's units. Nothing else (styles, themes) is set.

## 4. V46

A test with closed forms: a one-compartment model with a dose at t = 6 and an extinction,
`sim.sensitivity` against central finite differences (rel. error < 1e-4), absolute and
relative, before and after both.

## 5. Decisions for the project lead

1. **The batch entry point.** Recommended: `sim.batch(scenarios)` with `tq.grid(...)` to
   build the cross product (a grid is just a list of scenarios, so drawn populations use the
   same call). Alternative: one `sim.sweep(params=..., regimens=...)` that only does grids.
2. **Several cores now.** Recommended: `n_jobs=` on `batch` from the start (step 4.6 left
   it to here; pbisim's `ClinicalTrial(n_jobs=)` is the same idea). Alternative: one core
   only for now.
3. **A failed scenario in a batch.** Recommended: NaN outputs, listed in `b.failed`, the
   rest run (a sweep over a wide range will meet some failures). Alternative: stop at the
   first failure (as `sim(...)` does).
4. **Pre-runs and derivatives.** Recommended: a pre-run's end state is a fixed starting
   point: `sim.function` and `sim.sensitivity` do not differentiate through the pre-run
   (d/dθ of the starting state is taken as 0, and the docs say so). Making it
   differentiable means running the pre-run inside the JAX function at every x; possible
   later if estimation needs it.
5. **The long table's columns.** Recommended: `scenario, <each varied parameter>,
   regimen, time, output, value` (one row per scenario × time × output, as NONMEM/Monolix
   output tables), field outputs with one column per axis. Alternative: wide (one column
   per output).

## 6. As built (differences from the sketch)

- `b[i]` runs scenario `i` again for its full `SimResult`: a batch keeps only the outputs,
  so its memory stays small whatever the number of scenarios.
- `n_jobs` starts fresh processes (`spawn`) on both backends: JAX does not survive `fork`,
  and the CPU device count for `jax.shard_map` can only be set before JAX starts. Each
  process builds the model once (a few seconds). Each process gets one BLAS and one XLA
  thread (2026-10-27: with a thread per core each, idle BLAS threads spun and big solves
  slowed 10x). Measured: 80 scenarios of the phage toy, 4 processes, 3.9x faster than one
  (2.8x before that fix).
- On `backend="jax"`, a group that cannot run as one `vmap` call (nothing varies, a
  list-valued parameter) runs one scenario at a time instead.
- A pre-run under repeated dosing runs one dosing interval at a time; `pre.cycles` says
  how many. Its end is the trough (the next dose not yet given).
- To start from a given state, both solvers, the replay and `sim.function` take a starting
  state (NaN: the model's own value); nothing is rebuilt.
