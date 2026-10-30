# Simulation and exploration

A model is built once and then run many times, with different parameter values, regimens
and output times. Nothing is rebuilt when only those change. This chapter covers the ways to
run a model: one solve, a named **scenario**, a **grid** of scenarios run as a batch, a
**pre-run** to a steady state, and the plots that go with them.

The examples use a one-compartment oral PK model, so the page runs in seconds. Everything
works the same for a model with fields and axes.

=== "Notation"

    ```tq
    model oral_pk
    param ka, CL                    # no value: given when solving or simulating
    param V = 30
    d_t(depot) = -ka*depot
    d_t(centr) = ka*depot - CL/V*centr
    dose depot at t=0 to 72 every 12: 100
    observable conc = centr/V
    ```

=== "Python"

    ```python
    import numpy as np
    import tenqsp as tq

    m = tq.Model("oral_pk")
    ka = m.param("ka")
    CL = m.param("CL")
    V = m.param("V", 30)
    depot = m.compartment("depot")
    centr = m.compartment("centr")
    m.ode(depot, -ka * depot)
    m.ode(centr, ka * depot - CL / V * centr)
    m.dose(depot, times=[0, 12, 24, 36, 48, 60, 72], amounts=100)
    m.observable("conc", centr / V)
    ```

## One solve: `m.solve`

```python
t = np.linspace(0, 96, 97)
sol = m.solve((0, 96), times=t, params={"ka": 1.2, "CL": 4}, rtol=1e-6, atol=1e-6)
print(sol)
```

`m.solve((t0, t1), ...)` integrates from `t0` to `t1` and reports the state at `times`. The
options:

| Option | Means |
|---|---|
| `times=` | the output times |
| `params=` | parameter values by name, e.g. `{"CL": 4}` |
| `regimen=` | a regimen replacing the model's doses ([Doses and regimens](doses.md)) |
| `rtol=`, `atol=` | relative and absolute tolerances (both 1e-6 by default) |
| `max_step=` | the longest step the solver may take |
| `backend=` | `"scipy"` (default) or `"jax"` (below) |

### The solution

| | Gives |
|---|---|
| `sol["conc"]` | an observable or compartment: one value per output time, shape `(n_t,)` |
| `sol["G"]` | a field: shape `(n_t, *its axis sizes)`, in the field's axis order |
| `sol.t` | the output times |
| `sol.at("conc", 12)` | the value at one output time |
| `sol.coords` | per axis: cell centres (continuous) or labels (categorical), units, cell faces |
| `sol.doses`, `sol.infusions` | what was given ([Doses and regimens](doses.md)) |
| `sol.ledger` | the books of every field and compartment ([Reading a model back](explain.md)) |
| `sol.stats`, `sol.messages` | solver counters and notes |
| `sol.to_xarray()` | everything as an `xarray.Dataset` (needs `xarray`) |

```python
print(sol["conc"].shape, sol.at("conc", 12))
ds = sol.to_xarray()
print(list(ds.data_vars))                  # ['depot', 'centr', 'conc']
```

At a dose time the state is reported just after the dose.

## Values are layered

The values a run uses come in layers, each replacing only the names it gives:

1. the model's own values (`V = 30`);
2. the scenario's `params` (below);
3. `params=` in the call.

A parameter declared without a value must be given in one of the layers. If it is not, the
run stops with `tq.MissingValues`, which names every missing one:

```python
try:
    m.solve((0, 24), times=[0, 24])
except tq.MissingValues as e:
    print(e)
```

A misspelt name is an error that suggests the closest one:

```python
try:
    m.solve((0, 24), times=[0, 24], params={"kaa": 1.2, "CL": 4})
except KeyError as e:
    print(e)                               # ... Did you mean 'ka'?
```

## Scenarios and the simulator

A **simulator** is the model ready to run many times, with the outputs and solver settings
fixed once, so every run is comparable. A **scenario** says what to run: values, regimen,
output times, a start time, under a name.

```python
sim = m.simulator(outputs=["conc"])        # also rtol=, atol=, max_step=, backend=
reg = tq.bolus("depot", 200, start=0, every=12, n=8)
sc = tq.Scenario(name="200 q12h", params={"ka": 1.2, "CL": 5}, regimen=reg, times=t)

r = sim(sc)                                # a SimResult
print(r, r["conc"].max())
r2 = sim(sc, params={"CL": 3})             # the call's values win; sc is unchanged
print(r2.scenario.params)                  # {'ka': 1.2, 'CL': 3}: every layer merged
```

| `tq.Scenario(...)` | Means |
|---|---|
| `params=` | values by name |
| `regimen=` | a `tq.Regimen`; left out, the model's own doses; `tq.Regimen()`, none |
| `times=` | the output times |
| `start=` | when the run starts (default 0); it ends at the last output time |
| `initial=` | the starting state: a pre-run (below) or values by name |
| `name=` | a label, used in tables and plots |

A `SimResult` gives the outputs (`r["conc"]`, `r.t`, `r.to_xarray()`), `r.scenario` (what
was run, with every layer of values merged) and `r.solution`, the full solution with every
state, the ledger and the solver's messages. Outputs can be observables, fields or
compartments; by default every observable.

Scenarios are values: `replace` makes a changed copy, and a scenario saves to readable JSON
(diffable under version control):

```python
sc_heavy = sc.replace(name="heavy", params={"V": 60})   # params merged: ka, CL, V
print(sc_heavy.params)
sc.save("q12h.json")
print(tq.Scenario.load("q12h.json") == sc)              # True
```

For a single run there is a shortcut, `m.simulate(sc)`, which is the same as
`m.simulator()(sc)`.

## Many scenarios: `tq.grid` and `sim.batch`

`tq.grid` makes every combination of the values given, as scenarios; `sim.batch` runs a list
of scenarios and collects the outputs:

```python
regs = {"100 q12h": tq.bolus("depot", 100, start=0, every=12, n=8),
        "200 q24h": tq.bolus("depot", 200, start=0, every=24, n=4)}
grid = tq.grid(params={"CL": [3, 6]}, regimens=regs, base=tq.Scenario(params={"ka": 1.2}))
print([s.name for s in grid])              # 2 x 2 = 4 scenarios, regimens varying fastest
b = sim.batch(grid, times=t)

print(b["conc"].shape)                     # (4, 97): scenario x time (x a field's axes)
print(b.table().head())                    # long table, one row per scenario x time x output
```

- The last name varies fastest (regimens last). Scenario names say what each one is
  (`"CL=3, 100 q12h"`). `base=` gives a scenario every grid point starts from (its times,
  start and other values). A per-category value is a dict: `{"k": [{"S": 1}, {"S": 2}]}`.
- `b.table()` (pandas) has the columns `scenario`, the varied values, `time`, `output`,
  `value`. `b.to_xarray()` has dims `(scenario, time, ...)` with the varied values along
  `scenario`.
- `b[i]` runs scenario `i` again and returns its full `SimResult` (the same numbers; a batch
  keeps only the outputs, to keep its memory small).
- Any list of scenarios works too, e.g. a population you drew yourself:
  `sim.batch([sc1, sc2, sc3], times=t)`. Every scenario needs the same number of output
  times; `times=` sets them for all.

A scenario that fails does not stop the batch. Its outputs are NaN, it is listed in
`b.failed` with the reason, and a warning says how many failed:

```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    b2 = sim.batch([sc, sc.replace(name="broken", params={"ka": float("nan")})])
print(b2.failed)                           # [(1, 'broken', "'ka' must be a finite number, not nan.")]
```

To use several processes, give `n_jobs`. Each process builds the model once (a few
seconds), so it pays off for long batches:

```py
b = sim.batch(grid, times=t, n_jobs=8)     # the same numbers, in 8 processes
```

## Pre-runs and steady state

Often a simulation should start from a state the system has reached already: a patient at
steady state on maintenance dosing, a tumour that has grown for two weeks, a bacterial
culture in stationary phase. A **pre-run** runs ahead and keeps the end state:

```python
maint = tq.Scenario(params={"ka": 1.2, "CL": 4}, regimen=reg)
pre = sim.pre_run(maint, until="steady")
print(pre)                                 # how long, steady or not, dosing intervals
print(pre.state)                           # the end state by name
```

- `until="steady"` with a regimen that repeats at even intervals: one interval after
  another, until two in a row end in the same state within the tolerances. That is the
  pharmacokinetic steady state, and `pre.state` is the trough, just before the next dose.
- With no doses, or a single dose: the doses are given, then the run goes on until every
  rate of change is within the tolerances (|dy/dt| ≤ atol + rtol·|y|). Several doses that do
  not repeat at even intervals are refused: give a time instead.
- `until=24 * 14` runs for a fixed time (two weeks).
- It gives up at `t_max` (default 1e6 time units) with a warning and `pre.reached = False`.

Start a scenario from it with `initial=pre`. The new scenario starts at its own t = 0 from
that state. Here, a new dose at steady state brings the concentration back to the same
trough 12 hours later:

```python
start_ss = tq.Scenario(params={"ka": 1.2, "CL": 4}, initial=pre,
                       regimen=tq.bolus("depot", 200, times=[0]), times=[0, 6, 12])
r_ss = sim(start_ss)
print(r_ss["conc"])                        # trough, 6 h after the dose, trough again
assert np.isclose(r_ss["conc"][0], r_ss["conc"][2], rtol=1e-4)
```

Starting values can also be given by name, `initial={"centr": 50.0}`; states not named keep
the model's own starting values. `initial=` is saved in the scenario's JSON.

!!! note
    The starting state from a pre-run is fixed: derivatives
    ([Gradients and sensitivities](gradients.md)) of a scenario that starts from it do not
    flow back through the pre-run.

## Plots

With `pip install "tenqsp[plot]"`:

```python
tq.plot.trajectories(b, "conc", log=True)              # one line per scenario (or one SimResult)
tq.plot.fan(b, "conc", by="regimen", q=(5, 50, 95))    # median and 5-95 % band per group
```

| Function | Draws |
|---|---|
| `tq.plot.trajectories(result, "conc", log=False)` | one line for a `SimResult`, one per scenario for a batch |
| `tq.plot.fan(batch, "conc", by="regimen", q=(5, 50, 95))` | the median as a line and the outer percentiles as a band, per group of a varied column |
| `tq.plot.sensitivity(s, "conc")` | a heat map of derivatives, parameter × time ([Gradients and sensitivities](gradients.md)) |
| `tq.plot.snapshot(result, "G", axis="z", times=[1, 6, 24])` | a field along one axis at a few times ([A first model](../start/first-model.md)) |

Each draws on `ax=` (or a new figure) and returns the matplotlib `Axes`; no style is set, so
your own matplotlib settings apply.

## Two solvers

Every solve, simulator and scenario runs on one of two solvers. Both use BDF (a standard
method for stiff equations, the kind where fast and slow processes mix) and give the same
answers.

- **scipy** (the default): scipy's BDF with TenQSP's exact sparse Jacobian. Fast for one
  run at any size. If a segment between two doses fails, it retries that segment with Radau
  and says so in `sol.messages`.
- **jax**: TenQSP's own BDF written in JAX. It is compiled once per model structure (1–15 s), after which new values, doses and output times never rebuild it. Small models
  then run faster than on scipy. Derivatives ([Gradients and sensitivities](gradients.md))
  always use it.

```python
sol_jax = m.solve((0, 96), times=t, params={"ka": 1.2, "CL": 4}, backend="jax")
print(np.max(np.abs(sol_jax["conc"] - sol["conc"])))   # round-off
```

`m.simulator(outputs=[...], backend="jax")` runs every scenario on it; in a batch, scenarios
of the same shape (same doses into the same places) then run together as one call. How the
two differ in speed, memory and failure handling is in [Numerical choices](numerics.md).
