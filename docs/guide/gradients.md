# Gradients and sensitivities

A **sensitivity** says how much an output moves when a parameter moves: how the
concentration at 8 h changes with clearance, how the bacterial count at 24 h changes with
the dose. TenQSP computes these derivatives exactly (to the solver's tolerance), not by
re-running the model with nudged values. Use them to see which parameters matter where, to
plan sampling times, and to fit a model to data with a gradient-based optimizer.

There are two ways in:

- `sim.sensitivity(scenario, wrt=[...])`: a table of derivatives at the scenario's values,
  for looking at;
- `sim.function(scenario, wrt=[...])`: the simulation as a JAX function of those numbers,
  for estimation and any JAX code (`jax.grad`, `jax.jacfwd`, `jax.vmap`).

Both start from a simulator and a scenario ([Simulation and exploration](simulation.md)).
The examples use a one-compartment model with an intravenous bolus, because its solution
is known in closed form and the derivatives can be checked by hand.

=== "Notation"

    ```tq
    model iv_bolus
    param CL=4, V=30
    d_t(A) = -CL/V*A                # amount in the body
    observable conc = A/V
    ```

=== "Python"

    ```python
    import numpy as np
    import tenqsp as tq

    m = tq.Model("iv_bolus")
    CL, V = m.params(CL=4, V=30)
    A = m.compartment("A")
    m.ode(A, -CL / V * A)
    m.observable("conc", A / V)
    ```

## A table of derivatives: `sim.sensitivity`

```python
sim = m.simulator(outputs=["conc"], rtol=1e-8, atol=1e-10)
times = [0.5, 1, 2, 4, 8, 12, 24]
sc = tq.Scenario(regimen=tq.bolus("A", 100, times=[0]), times=times)

s = sim.sensitivity(sc, wrt=["CL", "V"])
print(s.table())             # one row per output time, one column per (output, name)
```

`s["conc"]["CL"]` is d conc / d CL at every output time, `s.values["conc"]` the outputs
themselves, `s.t` the times. An output that varies over the model's axes (a field) has one
derivative per cell, in `s[output]`; `s.table()` holds the outputs with one value per time.

**Checking it by hand.** After a bolus D, the concentration is
\( C(t) = \frac{D}{V} e^{-CL\,t/V} \), so

$$ \frac{\partial C}{\partial CL} = -\frac{t}{V}\, C(t). $$

```python
t = np.array(times)
C = 100 / 30 * np.exp(-4 / 30 * t)
exact = -t / 30 * C
print(np.max(np.abs(s["conc"]["CL"] - exact) / np.abs(exact)))   # ~5e-7
assert np.allclose(s["conc"]["CL"], exact, rtol=1e-5)
```

The agreement is set by the tolerances: the derivative is that of the computed solution.

**Relative sensitivities.** With `relative=True` each entry is
\( \frac{\partial y}{\partial \theta} \frac{\theta}{y} \): the percent change in the output
per percent change in the parameter, which compares parameters of different units. Here it
is \( -CL\,t/V \), a straight line in time:

```python
sr = sim.sensitivity(sc, wrt=["CL", "V"], relative=True)
print(sr["conc"]["CL"])      # -0.067, -0.13, ..., -3.2
assert np.allclose(sr["conc"]["CL"], -4 / 30 * t, rtol=1e-5)
```

Where the output is 0 the relative sensitivity is NaN. `tq.plot.sensitivity(s, "conc")`
draws the table as a heat map, parameter × time. A run that fails raises `SolverError`, as
`m.solve` does.

## What `wrt=` takes

| Name | Means |
|---|---|
| `"CL"` | a parameter |
| `"k"` | every entry of a per-category parameter (`k[S]`, `k[R]`) |
| `"k[R]"` | one entry |
| `"tlag"`, `"F"` | a lag time or bioavailability: ordinary parameters ([Doses and regimens](doses.md)) |
| `"dose 2 time"`, `"dose 2 amount"` | the 2nd bolus of the scenario's regimen, counted from 1 in time order |
| `"infusion 1 rate"` (or `amount`, `duration`, `start`) | the 1st infusion of the regimen |

- Dose numbers refer to the **scenario's regimen** (`print(sc.regimen)` lists them). To
  differentiate with respect to a dose, give the doses as a regimen; with the model's own
  `dose` lines, differentiate the parameters they use.
- Changing an infusion's duration keeps its rate (the stop moves), unless its amount is in
  `wrt` too (then the amount is kept). Its rate or amount keep the duration. Its start keeps
  the duration, so both ends move. Give an infusion's rate or its amount, not both.
- An axis, a field, a compartment, a named formula or an observable is not something to
  differentiate; the error says which it is and suggests the closest valid name.
- Dose times, lag times and the times at which a condition on time switches move with the
  parameters, so their derivatives are exact too.

## The simulation as a function: `sim.function`

For estimation, and for anything else written in JAX, `sim.function` gives the simulation as
a plain function of a vector of numbers:

```python
import jax
import jax.numpy as jnp

f = sim.function(sc, wrt=["CL", "V"])
print(f.names)               # ('CL', 'V'): what each entry of x is
print(f.x0)                  # [ 4. 30.]: their values in the scenario
y = f(jnp.array([5.0, 25.0]))
print(y["conc"])             # {output: array}, as the simulator gives
```

`f(x)` solves the model again at each new `x`: it is the simulation there, not an
extrapolation from `f.x0`. `jax.jit`, `jax.grad`, `jax.jacfwd` and `jax.vmap` all work on it:

```python
J = jax.jacfwd(f)(f.x0)["conc"]        # (7 times, 2 parameters)
assert np.allclose(J[:, 0], exact, rtol=1e-5)
```

```py
X = jnp.array([[3.0, 30.0], [4.0, 30.0], [5.0, 30.0]])
Y = jax.vmap(f)(X)["conc"]             # (3, 7): one row per parameter set
```

### A small fit

Synthetic data from `CL = 5`, `V = 25`, fitted from the scenario's values with
`scipy.optimize.least_squares`, using the exact Jacobian:

```python
from scipy.optimize import least_squares

data = np.asarray(f(jnp.array([5.0, 25.0]))["conc"])

def residuals(x):
    return np.asarray(f(jnp.asarray(x))["conc"]) - data

def jacobian(x):
    return np.asarray(jax.jacfwd(f)(jnp.asarray(x))["conc"])

fit = least_squares(residuals, f.x0, jac=jacobian, bounds=([0.1, 1.0], [100.0, 1000.0]))
print(fit.x, fit.nfev)                 # [ 5. 25.]
assert np.allclose(fit.x, [5.0, 25.0], rtol=1e-4)
```

The bounds keep the optimizer away from values where the model makes no sense (a volume of
0). The same works with `jax.grad` of an objective, for optimizers that take a gradient:

```python
def sse(x):
    return jnp.sum((f(x)["conc"] - data) ** 2)

print(jax.grad(sse)(f.x0))             # d SSE / d (CL, V) at the start
```

Building `f` and compiling `jax.grad` or `jax.jacfwd` of it take a few seconds each, once per
model structure; after that, new values reuse the build.

## How the derivatives are computed

**Record and replay.** At each `x`, `f` runs the model once, keeping a record of every step
it took, then replays that record to get the derivatives. The derivatives are therefore those
of the computed solution, step for step: consistent with the outputs, which is what an
optimizer needs.

**Steps chosen for the derivative too.** The solver chooses its steps to keep the
derivatives accurate as well as the solution. This matters when the solution sits still but
its derivative does not: a response started at its baseline while its production rate is
estimated, say. Steps chosen for the solution alone would be far too long for the
derivative. The extra steps are taken only where the derivative needs them.

**Failed runs give NaN.** A run that fails gives NaN outputs and NaN derivatives instead of an
error, so an optimizer can step back. `f.status(x)` says `"ok"` or why the run failed:

```python
print(f.status(f.x0))                  # ok
```

**The record has room** for 4 times the steps of the run at `f.x0` (2 times on models above
2,000 states, to halve the memory a gradient needs there). An `x` that needs more gives NaN,
with the status "too many steps for the record (make a new sim.function(...) at this x)":
build a new function at values closer to the ones you need.

**Pre-runs are fixed.** A scenario that starts from a pre-run
([Simulation and exploration](simulation.md)) starts from a fixed state: the derivatives do
not flow back through the pre-run.

## Simulating populations: `derivatives=False`

For a virtual population, a sweep or anything else that only needs the outputs, make the
function without derivatives:

```py
g = sim.function(sc, wrt=["CL", "V"], derivatives=False)
pop = jax.vmap(g)(X)                   # X: one row per patient, in g.names order
```

It is the same function (`g.names`, `g.x0`, `g(x)`, `g.status(x)`, `jax.jit`, `jax.vmap`),
but the runs are not recorded or replayed and the steps are chosen for the solution alone.
So it needs a few MB per member instead of tens to hundreds, and is 2–5 times faster. Its
outputs equal those of a function with derivatives to the solver's tolerance. `jax.grad`
or `jax.jacfwd` of it raise an error saying so.

## What it costs

Measured on the benchmark models (CPU):

- one call `f(x)` costs about 2 to 4 forward solves (the run, then its replay; steps chosen
  for the derivative add a few);
- a reverse-mode gradient (`jax.grad`) costs 4–10 forward solves, whatever the number of
  parameters;
- `jax.jacfwd` grows with the number of parameters, and is the cheaper of the two for a few
  parameters on small models;
- `jax.vmap(f)` over many parameter sets gives what one call per set gives. The sets run in
  lockstep, so each step costs what the most demanding set needs, and each member carries
  its own record of the run, so memory grows with the batch.

Measured times and memory, by model size, CPU and GPU, are on the
[Performance](../background/performance.md) page.

`sim.function` is meant for estimation (tenqsp-fit) and for any JAX code of your own.
