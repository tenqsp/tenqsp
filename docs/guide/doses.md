# Doses and regimens

A dose is something added to the model at a given time: a tablet into the gut, a bolus into
plasma, an inoculum of bacteria, a batch of new cells at age 0. TenQSP has two ways to give
doses. **Doses written in the model** belong to the model, like its parameters' default
values: the model runs with them when nothing else is said. A **regimen given at
simulation** is a separate value (`tq.bolus`, `tq.infusion`, or a NONMEM-style dose table)
that you pass to a run; it replaces the model's doses for that run. Use the first for a
model that always comes with the same dosing, the second for comparing regimens.

Lag times and bioavailability (NONMEM's ALAG and F) belong to the model and act on both.

## Doses written in the model

A `dose` line names the place, the times and the amount. A regular schedule is written as a
range, like NONMEM's ADDL and II:

=== "Notation"

    ```tq
    model oral_pk
    param ka=1.2, CL=4, V=30
    d_t(depot) = -ka*depot
    d_t(centr) = ka*depot - CL/V*centr
    dose depot at t=0 to 72 every 12: 100       # seven doses, 0, 12, ..., 72
    observable conc = centr/V
    ```

=== "Python"

    ```python
    import numpy as np
    import tenqsp as tq

    m = tq.Model("oral_pk")
    ka, CL, V = m.params(ka=1.2, CL=4, V=30)
    depot = m.compartment("depot")
    centr = m.compartment("centr")
    m.ode(depot, -ka * depot)
    m.ode(centr, ka * depot - CL / V * centr)
    m.dose(depot, times=[0, 12, 24, 36, 48, 60, 72], amounts=100)
    m.observable("conc", centr / V)
    ```

```python
oral = m
sol = oral.solve((0, 96), times=np.linspace(0, 96, 193))
print(sol["conc"].max())
```

In `t=0 to 72 every 12` the last dose falls on the end time; `0 to 70 every 12` is refused
with a message, so a range never silently drops or adds a dose. In Python the times are
written out. Several `dose` lines into the same place add up.

### The forms of a dose line

| Notation | Python | Meaning |
|---|---|---|
| `dose P at t=0, 24: 1e8` | `m.dose(P, times=[0, 24], amounts=1e8)` | the same amount at each time |
| `dose P at t=0, 24: 1e8, 5e7` | `amounts=[1e8, 5e7]` | one amount per time |
| `dose P at t=0 to 144 every 24: 1e8` | the seven times written out | a regular schedule |
| `dose C[organ=gut] at t=0: 100` | `m.dose(C.at(organ="gut"), ...)` | one element of a field over categories |
| `dose G at t=0: 100*point(z, at=0)` | `amounts=100, shape=tq.point(z, at=0)` | into a field over continuous axes: an amount and where it goes |
| `dose B at t=0: density(1e6*exp(-z))` | `density=1e6*tq.exp(-z)` | a density added to the field |

### Into one element of a field

A field over categories (organs, susceptibility classes) is dosed one element at a time:
fix every categorical axis in brackets. Here a gut-and-liver model gets two different doses
into the gut, and a regular dose into plasma:

=== "Notation"

    ```tq
    model two_organs
    axis organ categorical [gut, liver]
    param V[organ] = {gut: 2, liver: 1.5}
    param k=0.3, ke=0.2
    field C(organ) weights=V[organ]             # concentrations; amount = C x V
    compartment plasma volume=5
    d_t(C) = -k*C
    d_t(plasma) = -ke*plasma
    dose C[organ=gut] at t=0, 24: 100, 50       # one amount per time
    dose plasma at t=0 to 48 every 24: 10
    ```

=== "Python"

    ```python
    m = tq.Model("two_organs")
    organ = m.axis("organ", labels=["gut", "liver"])
    Vo = m.param("V", {"gut": 2, "liver": 1.5}, over=organ)
    k, ke = m.params(k=0.3, ke=0.2)
    C = m.field("C", axes=[organ], weights=Vo)
    plasma = m.compartment("plasma", volume=5)
    m.reaction(C, -k * C)
    m.ode(plasma, -ke * plasma)
    m.dose(C.at(organ="gut"), times=[0, 24], amounts=[100, 50])
    m.dose(plasma, times=[0, 24, 48], amounts=10)
    ```

**Amounts are amounts.** Into a place with a volume, or a field with weights, the value
rises by amount / volume: 100 into a gut of volume 2 raises its concentration by 50, and 10
into a plasma of volume 5 raises it by 2.

```python
sol = m.solve((0, 30), times=[0, 1, 24, 30])
print(sol["C"][0], sol["plasma"][0])      # [50. 0.] 2.0: just after the doses at t = 0
```

At a dose time the solution reports the state just *after* the dose.

### Into a field over continuous axes

A field over a continuous axis (cells by age, drug by progress through the gut, bacteria by
depth in a biofilm) needs to know **where along the axis** a dose goes. It is written as a
starting value is (see [Starting values](initial.md)):

- an **amount times a shape**: `1e6*normal(a, mean=1, sd=0.5)` spreads 1e6 like a normal
  curve, cut to the grid and scaled so the cells receive exactly 1e6;
  `100*point(z, at=0)` puts all 100 in the cell that contains `z = 0`;
- a **density**: `density(1e5*exp(-a))` adds 1e5·e^-a per unit of `a` at every `a`
  (here about 1e5 in total on [0, 10]).

A cohort model of maturing cells, given new cohorts by age:

=== "Notation"

    ```tq
    model maturation
    axis a in [0, 10] n=20 units=d kind=one_way      # cell age, days
    param kdeath=0.1
    field N(a)
    d_t(N) = -d_a(N) - kdeath*N
        at a=10: outflow
    dose N at t=0, 5: 1e6*normal(a, mean=1, sd=0.5)  # an amount and its shape
    dose N at t=2: density(1e5*exp(-a))              # a density added
    observable total = int_a(N)
    ```

=== "Python"

    ```python
    m = tq.Model("maturation")
    a = m.axis("a", 0, 10, 20, units="d", kind="one_way")
    kdeath = m.param("kdeath", 0.1)
    N = m.field("N", axes=[a])
    m.advect(N, axis=a, velocity=1)
    m.reaction(N, -kdeath * N)
    m.at(N, a=10, outflow=True)
    m.dose(N, times=[0, 5], amounts=1e6, shape=tq.normal(a, mean=1, sd=0.5))
    m.dose(N, times=[2], density=1e5 * tq.exp(-a))
    m.observable("total", m.integral(N, over=a))
    ```

```python
maturation = m
sol = maturation.solve((0, 6), times=[0, 2, 5, 6])
print(sol["total"][0])                    # 1e6: exactly the amount
```

A bare number into such a field (`dose N at t=0: 1e6`) is refused, with a message
suggesting `1e6*point(a, at=0)` or `density(1e6)`: TenQSP never guesses where a dose goes.

## Regimens given at simulation

To compare dosing schemes without editing the model, build a **regimen** and pass it to a
run. A regimen is a list of boluses and infusions:

```python
reg = (tq.bolus("depot", 200, start=0, every=24, n=4)              # 0, 24, 48, 72
       + tq.infusion("centr", rate=50, duration=2, start=48))      # a loading infusion
print(reg)
```

- `tq.bolus(target, amount, times=[...])`, or `start=`, `every=` and `n=` (`n` is the
  number of doses). `amount` is one number for all times or one per time.
- `tq.infusion(target, ...)` is a constant rate between its start and stop: give **two** of
  `rate`, `amount` and `duration`, as NONMEM's RATE and AMT. It repeats with `times=` (the
  starts) or `start=`, `every=`, `n=`.
- `+` combines regimens; the events are kept in time order.
- `tq.Regimen()` is a regimen with no doses at all.

Give it to `m.solve(..., regimen=reg)`, to a simulator call, or to a scenario
([Simulation and exploration](simulation.md)):

```python
t = np.linspace(0, 96, 193)
r = oral.simulate(regimen=reg, times=t)
print(len(r.solution.doses))              # 4: the model's own 100 q12h are gone
none = oral.simulate(regimen=tq.Regimen(), times=t)
print(none["conc"].max())                 # 0.0: no doses at all
```

!!! warning "A regimen replaces the model's doses; it never adds to them"
    With no regimen, the `dose` lines of the model are used. A regimen you give replaces
    **all** of them for that run. To keep a model dose and add another, write both in the
    regimen.

### Targets

The target is a name as text, a model handle, or one element of a field:

| Target | Means |
|---|---|
| `"depot"` or `depot` | a compartment |
| `"C[organ=gut]"` or `C.at(organ="gut")` | one element of a field over categories (every categorical axis fixed) |
| `"C[organ=gut, sub=vascular]"` | the same, with two axes (a PBPK organ's vascular space) |
| `"N"` with `shape=tq.point("a", at=0)` | a field over continuous axes: `shape=` says where the amount goes |

```python
r = maturation.simulate(regimen=tq.bolus("N", 1e6, times=[0], shape=tq.point("a", at=0)),
                        times=[0, 1])
print(r["total"])                         # [1e6, 1e6*exp(-0.1)]
```

A shape is any of the named shapes (`tq.point`, `tq.normal`, ...); the cells receive exactly
the amount. An infusion takes `shape=` the same way and spreads its rate.

### From a NONMEM dose table

`tq.regimen_from_table` reads dose records in NONMEM's columns, from a pandas DataFrame or
a dictionary of columns:

```python
import pandas as pd

df = pd.DataFrame({
    "ID":   [1,   1,   1],
    "TIME": [0,   0,   48],
    "AMT":  [100, 0,   500],
    "CMT":  [1,   2,   2],
    "RATE": [0,   ".", 250],
    "ADDL": [3,   0,   0],
    "II":   [12,  0,   0],
    "EVID": [1,   0,   1],
    "DV":   [".", 0.5, "."],
})
reg_nm = tq.regimen_from_table(df, targets={1: "depot", 2: "centr"})
print(reg_nm)
```

This gives four boluses of 100 into `depot` (0, 12, 24, 36: one dose plus ADDL = 3 more,
every II = 12) and an infusion of 500 into `centr` at 250 per hour from t = 48, so for 2
hours. The observation row (EVID 0) is skipped.

| Column | Read as |
|---|---|
| `TIME`, `AMT` | when, and how much; rows with AMT missing or 0 are not doses |
| `CMT` | a place name, or a number mapped by `targets={1: "depot", ...}` |
| `RATE` | 0 or missing: a bolus; above 0: an infusion of AMT at that rate |
| `DUR` | an infusion of AMT over DUR (give RATE or DUR, not both) |
| `ADDL`, `II` | ADDL more doses, every II |
| `EVID` | 1 is a dose; 0 and 2 are skipped |

Column names are case-insensitive, and `.` or an empty cell counts as missing. Other columns
(`DV`, covariates) are not read. Some records are refused with a message rather than read
differently from NONMEM: resets (EVID 3 or 4), steady-state doses (SS), a rate modelled in
the model (RATE = -1 or -2), and a table with several subjects (read it with
`tq.regimens_from_table`, below).

A table with several subjects gives one regimen per subject with `tq.regimens_from_table`,
a dictionary from ID to regimen, in the order the subjects first appear. Each subject's rows
are read by the same rules; a subject with only observation rows gets an empty regimen.

```python
df2 = pd.DataFrame({
    "ID":   [1,   1,   2,   2,   3],
    "TIME": [0,   12,  0,   24,  0],
    "AMT":  [100, 100, 200, 0,   0],
    "CMT":  [1,   1,   1,   2,   2],
    "EVID": [1,   1,   1,   0,   0],
})
regimens = tq.regimens_from_table(df2, targets={1: "depot", 2: "centr"})
print(list(regimens))        # [1, 2, 3]
print(regimens[2])           # one bolus of 200 into depot at 0; subject 3 has none
```

## Lag time and bioavailability

```tq
lag depot = tlag                 # every dose into depot arrives tlag later (NONMEM's ALAG)
bioavailability depot = F        # only the fraction F of it arrives (NONMEM's F)
```

In Python, `m.lag(depot, "tlag")` and `m.bioavailability(depot, "F")`. Both belong to the
model, not to the regimen, and act on **every** dose into the place: the model's own and a
regimen's, boluses and infusions.

=== "Notation"

    ```tq
    model oral_lag
    param ka=1.2, CL=4, V=30, tlag=0.5, F=0.8
    d_t(depot) = -ka*depot
    d_t(centr) = ka*depot - CL/V*centr
    lag depot = tlag
    bioavailability depot = F
    dose depot at t=0: 100
    observable conc = centr/V
    ```

=== "Python"

    ```python
    m = tq.Model("oral_lag")
    ka, CL, V, tlag, F = m.params(ka=1.2, CL=4, V=30, tlag=0.5, F=0.8)
    depot = m.compartment("depot")
    centr = m.compartment("centr")
    m.ode(depot, -ka * depot)
    m.ode(centr, ka * depot - CL / V * centr)
    m.lag(depot, "tlag")
    m.bioavailability(depot, "F")
    m.dose(depot, times=[0], amounts=100)
    m.observable("conc", centr / V)
    ```

A bolus of 100 at t = 0 arrives as F × 100 = 80 at t = 0.5. After that the depot empties
at rate ka, so depot(t) = 80 e^{-1.2 (t - 0.5)}:

```python
sol = m.solve((0, 2), times=[0, 0.25, 0.5, 0.75, 1, 2])
print(sol["depot"][:3])                    # [0, 0, 80]: nothing before t = 0.5
print(sol.doses)                           # [(0.5, 'depot', 80.0, 80.0)]
assert np.isclose(sol["depot"][3], 80 * np.exp(-1.2 * 0.25), rtol=1e-5)
print(sol.ledger.table("depot"))           # doses +100, not absorbed -20
```

An **infusion** is shifted whole (start and stop move by the lag) and its **rate** is
scaled by F, so its duration stays. An infusion of 50 per hour for 2 hours from t = 1 runs
at 40 per hour from 1.5 to 3.5:

```python
r = m.simulate(regimen=tq.infusion("depot", rate=50, duration=2, start=1),
               times=np.linspace(0, 10, 11))
print(r.solution.infusions)                # [(1.5, 3.5, 'depot', 40.0)]
```

!!! note "One difference from NONMEM"
    NONMEM does the same for an infusion given by its duration (DUR). For one given by its
    rate (RATE) it keeps the rate and shortens the infusion to F·AMT/RATE. The amount that
    arrives is the same; the time course differs. `regimen_from_table` follows TenQSP's rule
    for both.

More about lags and F:

- A dose whose arrival falls after the end of the run is not given.
- The ledger books each dose as given, under `doses` or `infusions`, and the part that did
  not arrive as `not absorbed`, so the books still close against the regimen.
- `tlag` and `F` are ordinary parameters: they can be formulas of parameters, changing
  them never rebuilds the model, and both can be differentiated
  ([Gradients and sensitivities](gradients.md)).
- A lag or F below 0 is an error when the model is run. F above 1 is allowed.
- One lag and one bioavailability per place.

## What was given: `sol.doses` and `sol.infusions`

Every solution lists the doses as they arrived, lags and F applied:

- `sol.doses`: `(time, target, amount, rise)` for each bolus, where `rise` is how much it
  raised the state's value (the amount divided by the volume, weight or cell width). A dose
  into a field over continuous axes gives one entry per cell it reached.
- `sol.infusions`: `(start, stop, target, rate)` for each infusion, the stop clipped to the
  end of the run.

## How doses are solved

Doses split a run into **segments**: the solver integrates up to the dose time, the dose is
added to the state, and integration starts again from the new state with a fresh step size.
So a dose is exact (not smeared over a short pulse), and a dose at an output time shows in
that output. Infusion starts and stops are segment ends too, so the solver never steps over
a change of rate.

### Large doses and `atol`

`atol`, the absolute tolerance, is the size below which the solver stops caring about a
state's error. A dose that raises a state by a large amount (a 1e10 inoculum) leaves
rounding errors of about 1e-14 × the dose in the states it feeds. If `atol` is below that,
the solver's error estimate for those states is noise, and the solve fails or succeeds by
chance. The rule is:

$$ \text{atol} \ge 10^{-14} \times \text{the largest dose} $$

`m.diagnostics(sol)` checks it and names the `atol` to use; when a run fails or a segment is
retried, the message says so too. Here an inoculum of 1e10 bacteria into a culture killed by
a drug, at the default `atol` of 1e-6:

```python
culture = tq.Model.from_text("""
model inoculum
param g=1, Bmax=1e11, kill=2, ke=0.3
d_t(B) = g*B*(1 - B/Bmax) - kill*A/(1 + A)*B
d_t(A) = -ke*A
A(0) = 10
dose B at t=2: 1e10
""")
sol = culture.solve((0, 24), times=np.linspace(0, 24, 25))
print(culture.diagnostics(sol))           # warns: use atol=1e-04 or larger
sol = culture.solve((0, 24), times=np.linspace(0, 24, 25), atol=1e-4)
```

More on tolerances in [Numerical choices](numerics.md).
