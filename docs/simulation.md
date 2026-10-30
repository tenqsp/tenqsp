# Simulation: values, regimens and scenarios

A model is built once and then run many times with different parameter values, regimens and
output times. Nothing is rebuilt when only those change.

## Values

```
param ka, CL        # declared without a value: given when solving or simulating
param V = 30        # a default
```

Values are layered. The model's own values come first, then the scenario's `params`, then
`params=` in the call. Each layer replaces only the names it gives. A missing value is an
error that names it (`tq.MissingValues`). A misspelt name is an error that suggests the
closest one.

## Regimens

```python
reg = (tq.bolus("depot", 100, start=0, every=12, n=7)          # or times=[0, 12, ...]
       + tq.infusion("centr", rate=50, duration=2, start=48))  # two of rate, amount, duration
tq.bolus("C[organ=gut, sub=vascular]", 5, times=[0])           # one element of a field
tq.bolus("G", 100, start=0, every=12, n=7,                     # a field over continuous axes:
         shape=tq.point("g", at=0))                            #   where the amount goes
tq.regimen_from_table(df, targets={1: "depot", 2: "centr"})    # NONMEM TIME/AMT/CMT/RATE/DUR/ADDL/II/EVID
```

How a regimen relates to the model's own doses:

- If you don't give a regimen, the `dose` lines written in the model are used.
- A regimen you give **replaces** the model's doses; it never adds to them.
- `tq.Regimen()` means no doses at all.

Amounts are amounts. Into a place with a volume, or a weighted field, the value rises by
amount / volume.

Into a field over continuous axes (a stomach over time since swallowing, a tissue over
depth), a dose needs a **shape**, as an initial value does: `tq.point("g", at=0)` puts the
whole amount in the cell containing `g = 0`; `tq.normal("z", mean=1, sd=0.2)` spreads it, and
the cells receive exactly the amount. Category axes are fixed in the target
(`"B[s=R]"`). An infusion takes `shape=` the same way. Derivatives with respect to a dose's
amount and time work as for any dose (`"dose 1 amount"`); the shape itself is held fixed. An infusion is a constant rate between its start and stop. The ledger books
it as rate × time under the channel `infusions`.

### Lag times and bioavailability

```
lag depot = tlag                 # m.lag(depot, "tlag")
bioavailability depot = F        # m.bioavailability(depot, "F")
```

These belong to the model, not the regimen. They act on **every** dose into the place: the
model's own and a regimen's, boluses and infusions, as NONMEM's ALAG and F do (with one
difference, for infusions, below).

- A bolus of 100 at t = 0, with `tlag = 0.5` and `F = 0.8`, arrives as 80 at t = 0.5.
- An infusion is shifted whole (start and stop move by the lag), and its rate is scaled by F:
  its duration stays. NONMEM does this for an infusion given by its duration (DUR), but for
  one given by its rate (RATE) it keeps the rate and shortens the infusion to F·AMT/RATE.
  The amount that arrives is the same; the time course differs. `regimen_from_table`
  follows TenQSP's rule for both.
- A dose whose arrival falls after the end of the run is not given.
- `sol.doses` lists each dose as it arrived: time and amount.
- The ledger books each dose as given, under `doses` or `infusions`, and the part that did
  not arrive as `not absorbed`. So the books still close against the regimen.
- Both are ordinary parameters. Changing their values never rebuilds the model, and both
  can be estimated, with exact gradients.
- A lag or F below 0 is an error when the model is run. F above 1 is allowed.

## Scenarios and the simulator

```python
sim = m.simulator(outputs=["conc"], rtol=1e-8)   # solver settings fixed here
sc = tq.Scenario(name="200 q12h", params={"CL": 5}, regimen=reg, times=np.linspace(0, 96, 193))
r = sim(sc)                       # r["conc"], r.t, r.to_xarray(), r.solution (everything)
r2 = sim(sc, params={"CL": 3})    # the call's values win; sc is unchanged
sc.replace(params={"V": 60})      # a copy; params merged
sc.save("q12h.json"); tq.Scenario.load("q12h.json")
m.simulate(sc)                    # one-line shortcut
```

A run starts at t = 0 (or `start=`) and ends at the last output time. `r.scenario` records
what was run, with every layer of values merged.

## Many scenarios: `tq.grid` and `sim.batch`

```python
regs = {"100 q12h": tq.bolus("depot", 100, start=0, every=12, n=7),
        "200 q24h": tq.bolus("depot", 200, start=0, every=24, n=4)}
grid = tq.grid(params={"CL": [3, 5, 8], "V": [40, 60]}, regimens=regs)   # 3 x 2 x 2 = 12
b = sim.batch(grid, times=np.linspace(0, 96, 193))

b["conc"]        # (12, 193): scenario x time (x the field's axes)
b.table()        # pandas, one row per scenario x time x output:
                 #   scenario, CL, V, regimen, time, output, value
b.to_xarray()    # dims (scenario, time, ...); CL, V, regimen along scenario
b[3]             # scenario 3 run again: its full SimResult (same numbers)
b.failed         # [(index, name, reason)] of scenarios that failed (their outputs NaN)

sim.batch([sc1, sc2, sc3], times=...)   # any list, e.g. a population you drew yourself
sim.batch(grid, times=..., n_jobs=8)    # in 8 processes
```

- A grid is every combination, the last name varying fastest (regimens last). Scenario
  names say what each one is (`"CL=3, V=40, 100 q12h"`). `base=` gives a scenario the grid
  starts from (its times, start and other values). A per-category value is a dict:
  `{"k": [{"S": 1}, {"S": 2}]}`.
- For a plain list, the table's columns are the parameters whose values differ.
- Every scenario needs the same number of output times; `times=` sets them for all.
- A scenario that fails does not stop the batch: its outputs are NaN, it is listed in
  `b.failed`, and a warning says how many failed.
- The numbers are the same however it runs. On `backend="scipy"` each scenario is one solve.
  On `backend="jax"`, scenarios of the same shape (same doses into the same places) run as
  one `jax.vmap` call without derivatives. `n_jobs` starts that many fresh processes
  (each builds the model once, a few seconds).

## Pre-runs and steady state

```python
pre = sim.pre_run(sc, until="steady")        # or until=24 * 14 (a fixed time)
pre.state, pre.t, pre.reached, pre.cycles    # the end state by name; how long; steady?
r = sim(sc2.replace(initial=pre))            # sc2 starts from that state, at its own t = 0
tq.Scenario(..., initial={"R": 8.0})         # or give starting values by name yourself
```

- `until="steady"` with a regimen that repeats at even intervals (`every=`): one interval
  after another, until two in a row end in the same state within the tolerances. That is
  the pharmacokinetic steady state; `pre` holds the trough, just before the next dose.
- With no doses, or a single dose time: the doses are given (and have arrived, lags
  included), then the run goes on in doubling stretches until every rate of change is
  within the tolerances (|dy/dt| ≤ atol + rtol·|y|). Several doses that do not repeat at
  even intervals are refused: give `until=<time>`.
- It stops at `t_max` (default 1e6 time units) with a warning and `pre.reached = False`.
- `initial=` is saved in the scenario's JSON. States it does not name keep the model's
  initial values.
- The starting state is fixed: `sim.function` and `sim.sensitivity` on a scenario that
  starts from a pre-run do not differentiate through the pre-run.

## Plots (`pip install tenqsp[plot]`)

```python
tq.plot.trajectories(b, "conc", log=True)              # one line per scenario (or a SimResult)
tq.plot.fan(b, "conc", by="regimen", q=(5, 50, 95))    # median and band per group
tq.plot.sensitivity(s, "conc")                         # heat map, parameter x time
tq.plot.snapshot(r, "I", axis="a", times=[1, 6, 24])   # a field along one axis
```

Each draws on `ax=` (or a new figure) and returns the matplotlib `Axes`; labels come from
the model's names and units, and no style is set.

## Derivatives: `sim.function` and `sim.sensitivity`

Both use the JAX solver whatever the simulator's `backend`. Derivatives are those of the
computed solution: the run is recorded and replayed, and dose times, lag times and switch
times move with the parameters (design notes `docs/design/step4_4_gradients.md`,
`docs/design/step4_4d_api.md`).

```python
sim = m.simulator(outputs=["conc"], rtol=1e-8)
sc = tq.Scenario(regimen=tq.bolus("depot", 100, times=[0, 12]), times=[1, 2, 4, 8, 24])

# a table of derivatives, for looking at
s = sim.sensitivity(sc, wrt=["CL", "V"])                 # d conc / d CL, d conc / d V
s = sim.sensitivity(sc, wrt=["CL", "V"], relative=True)  # % change per % change
s["conc"]["CL"]; s.values["conc"]; s.table()

# a JAX function, for estimation (tenqsp-fit) and any JAX code
f = sim.function(sc, wrt=["CL", "V", "ka", "tlag"])
f.names          # what each entry of x is
f.x0             # their values in the scenario
f(x)             # {"conc": array}; x is a list of numbers in f.names order
jax.grad(lambda x: jnp.sum((f(x)["conc"] - data) ** 2))(f.x0)
```

What `wrt=` takes:

| name | means |
|---|---|
| `"CL"` | a parameter |
| `"k"` | every entry of a per-category parameter (`k[S]`, `k[R]`) |
| `"k[R]"` | one entry |
| `"dose 2 time"`, `"dose 2 amount"` | the 2nd bolus of the scenario's regimen, in time order |
| `"infusion 1 rate"` (`amount`, `duration`, `start`) | the 1st infusion of the regimen |

- Changing an infusion's duration keeps its rate (the stop moves), unless its amount is in
  `wrt` too (then the amount is kept). Its rate or amount keep the duration. Its start keeps
  the duration, so both ends move.
- At each new `x`, `f` solves the model again and then replays that run for the
  derivatives. So `f(x)` is the simulation at `x`, not an extrapolation from `f.x0`.
  `jax.jit`, `jax.grad`, `jax.jacfwd` and `jax.vmap` all work on it.
- The solver chooses its steps to keep the derivatives accurate as well as the solution.
  This matters when the solution sits still but its derivative does not, for example a
  response started at its baseline while `kin` is estimated: steps chosen for the solution
  alone would be far too long for the derivative. The run carries the derivative along
  one direction that mixes every `wrt` number and checks its error at every step (as
  CVODES does with `errconS`). This costs extra steps only where the derivative needs them.
- A run that fails gives NaN outputs and NaN derivatives instead of an error, so an
  optimizer can step back. `f.status(x)` says why.
- The record has room for 4× the steps of the run at `f.x0` (2× on models above 2,000
  states, to halve the memory a gradient needs there). An `x` that needs more gives
  NaN, with the status "too many steps for the record"; make a new `sim.function` there.
- `sim.sensitivity` raises `SolverError` when the run fails, as `solve` does.
- Changing values never rebuilds. A new scenario of the same shape (same outputs, number of
  output times, doses and `wrt` names) shares the build.
- Cost (measured, `docs/benchmarks.md`, steps 4.4 and 4.5): one `f(x)` costs 2–3 forward
  solves; a reverse-mode gradient (`jax.grad`) 4–10, whatever the number of parameters; `jax.jacfwd`
  grows with the number of parameters. Building `f` and compiling `jax.grad` or
  `jax.jacfwd` of it take a few seconds each, once per model structure.
- `jax.vmap(f)` over many parameter sets gives what one call per set gives (to the
  tolerance). The sets run in lockstep, so each step costs what the most demanding set
  needs. On a CPU a batch is about as fast as a loop on the dense tier (up to 250 states)
  and 1.3–3x faster on the sparse tier (above); on a GPU a batch of the dense tier is 4–14x
  faster than the best on the CPU up to ~350 states. Each member carries its own record:
  about 20 MB at 150 states, 100 MB at 350 (design note `docs/design/step4_6_batches.md`).

### Simulating without derivatives: `derivatives=False`

For a virtual population, a sweep or anything else that only needs the outputs:

```python
g = sim.function(sc, wrt=["CL", "V"], derivatives=False)
pop = jax.vmap(g)(X)   # X: one row per patient, in g.names order
```

- The same function as above (`g.names`, `g.x0`, `g(x)`, `g.status(x)`, `jax.jit`,
  `jax.vmap`), but the runs are not recorded or replayed, and the steps are chosen for
  the solution alone. So it needs a few MB per member instead of tens to hundreds, and is
  2–5x faster.
- The outputs equal those of a function with derivatives to the solver's tolerance (the
  two take different steps).
- `jax.grad`, `jax.jacfwd` or `jax.jacrev` of it raise an error saying so; make the
  function without `derivatives=False` to differentiate.

### Several CPU cores

On a CPU a batch runs on one core. The simplest way to use several is
`sim.batch(scenarios, n_jobs=4)` (above). To spread one `jax.vmap` call's members over
several cores instead (5.4x on 64 cores for 64 members), tell JAX how many cores to use **before it is imported**,
then split the batch with `jax.shard_map`. The number of members must be a multiple of
the number of cores:

```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"  # before import jax

import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

mesh = Mesh(np.array(jax.devices()), ("p",))
run = jax.jit(jax.shard_map(jax.vmap(g), mesh=mesh, in_specs=P("p"),
                            out_specs=P("p"), check_vma=False))
pop = run(X)   # X: (a multiple of 16) rows
```

**BLAS threads.** The sparse LU that both solvers use (SuperLU) calls BLAS. With OpenBLAS's
default of one thread per core, the idle threads spin: on a 64-core machine one 9,600-state
solve used 3,400 CPU-seconds in 54 s. TenQSP therefore runs BLAS on one thread while it
factors and solves (34 s, 38 CPU-seconds), and restores your setting afterwards.
`tenqsp._threads.BLAS_THREADS = None` turns this off.

## Backends

Every solve, simulator and scenario runs on one of two solvers. Both use BDF and give the
same answers (they agree to about 1e-13 on the benchmark models).

```python
m.solve((0, 24), backend="scipy")              # default
m.solve((0, 24), backend="jax")                # the JAX BDF
sim = m.simulator(outputs=["conc"], backend="jax")
```

- **scipy** (default): scipy's BDF with TenQSP's sparse Jacobian. Fast for one run at any
  size, up to 10^5 states. If a segment fails it retries with Radau. Both solvers stop
  after `max_steps` steps (default 100,000) between two doses or events, with a message:
  a run that needs that many is crawling, usually because the tolerances are close to
  rounding for a fast process (e.g. rtol = 1e-10 with exchange rates of 1e6 per hour);
  loosen rtol to 1e-8 or so.
- **jax**: TenQSP's own port of scipy's BDF, written in JAX. It is compiled once per model
  structure (1–15 s), after which new parameter values, dose amounts and times, and output
  times never rebuild it. Small models (up to a few hundred states) then run faster than on
  scipy. It is the solver that gradients (`sim.function`, `sim.sensitivity`) and batched
  runs use.
  - Any size. Up to 250 states (500 when JAX runs on a GPU) it solves its linear systems
    with a dense LU inside JAX; above that with SuperLU, the sparse LU scipy uses, called
    from JAX (the "sparse tier"). The choice is automatic. On the sparse tier a solve takes 1.3–1.9x as long
    as on scipy, and takes the same steps. Gradients of large models need memory: see
    `docs/benchmarks.md`.
  - No Radau fallback: a failing run raises a `SolverError` that says where it failed and
    suggests `backend="scipy"`.
  - Doses, infusions, conditions (on time and on states), extinction groups and noise floors
    all work as on scipy, and the ledger is the same.
  - Slots: the number of doses, infusions and output times is rounded up to a power of two
    (8, 16, 32, ...), so a run with 9 doses after runs with 8 builds once more, then not again.

Measured timings are in `docs/benchmarks.md` (steps 4.3, 4.4 and 4.5).

### Large doses and atol

A dose that raises a state by a large amount (a 1e10 phage bolus) leaves rounding errors of
about 1e-14 x the dose in the states it feeds. If `atol` is below that, the solver's error
estimate for those states is noise, and BDF fails or survives by chance, on either backend.
Measured on a phage model, plain BDF succeeded every time once `atol` >= 1e-14 x the dose
(1e-4 for 1e10) and as rarely as 0 in 8 below it. So:

- `m.diagnostics(sol)` warns when a run's `atol` is below this level for any dose, and names
  the `atol` to use;
- when a run fails (JAX) or a segment is retried with Radau (scipy), the message says so too.

Each solution lists the doses it gave: `sol.doses`, as `(time, target, amount, rise)`, where
`rise` is how much the dose raised the state's value.
