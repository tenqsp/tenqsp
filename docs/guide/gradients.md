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

**Other tolerances for one call.** `f` runs at the simulator's `rtol` and `atol`. To check that
they are tight enough, run the same function at tighter ones: `f(x, rtol=..., atol=...)`
(keyword only; `atol` is one number or one per state). The tolerances are inputs of the
compiled program, so this needs no new build and no new compile, and the numbers are those of
a simulator built at those tolerances. `f.status` and a function of several scenarios take
them too.

```python
y_tight = f(f.x0, rtol=1e-10, atol=1e-12)["conc"]
print(np.max(np.abs(f(f.x0)["conc"] - y_tight)))  # what the simulator's tolerances cost
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

### Many scenarios at once

For a population (one scenario per subject, each with its own doses and sampling times),
give `sim.function` a list of scenarios. The result is one JAX function of a table of
values, one row per scenario, and every scenario runs in one `jax.vmap` call of one build:

```python
subjects = [
    tq.Scenario(name="1", regimen=tq.bolus("A", 100, times=[0]), times=[0.5, 1, 2, 4, 8]),
    tq.Scenario(name="2", regimen=tq.bolus("A", 80, times=[0, 12]), times=[1, 6, 12.5, 24]),
    tq.Scenario(name="3", params={"V": 25}, regimen=tq.bolus("A", 120, times=[0]),
                times=[2, 24]),
]
fs = sim.function(subjects, wrt=["CL", "V"])
print(fs.x0)                 # [[ 4. 30.] [ 4. 30.] [ 4. 25.]]: each scenario's values
Y = fs(fs.x0)["conc"]        # (3 scenarios, 5 times)
print(fs.n_times)            # [5 4 2]: the rest of each row is NaN
assert np.isclose(Y[1, 2], 80 / 30 * (np.exp(-4 / 30 * 12.5) + np.exp(-4 / 30 * 0.5)))
assert np.isnan(Y[2, 2])
```

`fs.t` holds each scenario's times and `fs.mask` says which entries are real. An objective
over all subjects masks the rest, and its gradient has one row per subject:

```python
obs = np.where(fs.mask, 1.0, np.nan)          # measurements, in the same layout

def sse_all(X):
    r = jnp.where(fs.mask, fs(X)["conc"] - obs, 0.0)
    return jnp.sum(r**2)

G = jax.grad(sse_all)(jnp.asarray(fs.x0))     # (3, 2): d SSE / d (CL, V) per subject
print(fs.status(fs.x0))                       # ('ok', 'ok', 'ok')
```

The scenarios may differ in their number of doses, infusions and output times: each is
padded to the largest among them with slots the solver never reaches (at t = +inf), so the
padding changes no value and no derivative, and every list of scenarios of one model shares
one build. Names in `wrt` are those of a single scenario's function; a dose number
(`"dose 2 amount"`) must exist in every scenario. Results agree with each scenario's own
function to within the solver's tolerance: running in one `jax.vmap` call rounds
differently, which can move a solver step.

## How the derivatives are computed

`f` gets its derivatives in one of two ways, chosen by `derivatives=`:

```python
f = sim.function(sc, wrt=["CL", "V"])                          # "auto", the default
print(f.derivatives)                   # forward
fr = sim.function(sc, wrt=["CL", "V"], derivatives="reverse")  # always reverse
assert np.allclose(jax.jacfwd(f)(f.x0)["conc"], jax.jacfwd(fr)(fr.x0)["conc"], rtol=1e-6)
```

**Forward** (forward sensitivities). At each `x`, the run carries the derivative of every
state with respect to every name in `wrt` alongside the solution, as Stan, nlmixr2 and
CVODES do. The cost grows with the number of names; nothing is kept beyond the run, so it
needs little memory. It gives the whole table of derivatives (every output time × every
name), which is what Levenberg–Marquardt, Gauss–Newton or FOCE-type fits use.

**Reverse** (record and replay). At each `x`, `f` runs the model once, keeping a record of
every step it took, then goes back through that record to get the derivatives. A gradient
of one number (`jax.grad` of an objective) then costs the same whatever the number of names,
but the record takes memory, and a whole table (`jax.jacfwd`) costs more than forward.

**"auto"** picks forward up to 40 names in `wrt` and reverse above. Measured on the CPU
([Performance](../background/performance.md)):

- forward was faster for every table of derivatives (`jax.jacfwd`), up to 100 names;
- for a gradient (`jax.grad`), forward was faster up to about 40–50 names; above that, reverse;
- with 12 names on a 61-state model, a batch of 4 needed 3.5 MB forward and 840 MB reverse.

Set `derivatives=` yourself when you know better: `"forward"` for a least-squares fit with
many parameters, `"reverse"` for a gradient-based sampler (NUTS) with many.

Either way the derivatives are those of the computed solution, consistent with the outputs,
which is what an optimizer needs. The two agree to the solver's tolerance.

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

**How much work a run takes.** `f.steps(x)` is the most steps the solver takes for `f(x)`
between two doses or events. The simulator's `max_steps` (100,000 by default) is a limit
on this number; a run that reaches it fails with the status "too many steps". The count
is that of `f`'s own run, whose steps also follow the derivatives it carries, and it is
given also when the run fails. Values far outside a model's range can make a run take
far more steps than usual (a fast clearance here):

```python
print(f.steps(f.x0))                         # 117
print(f.steps(np.array([400.0, 30.0])))      # 416
```

A program that calls `f` at many values chosen by an optimizer can then set `max_steps`
from the work at reasonable values, say 50 times `f.steps(f.x0)`, so that a run at an
extreme value fails quickly instead of taking minutes. A function of several scenarios
gives one count per scenario: `fs.steps(X)`.

**The record has room** (reverse only) for 4 times the steps of the run at `f.x0` (2 times on
models above 2,000 states, to halve the memory a gradient needs there). An `x`, or a much tighter
`rtol` given to one call, that needs more gives NaN, with the status "too many steps for the
record (make a new sim.function(...) at this x, or at this tolerance; ...)": build a new
function at values closer to the ones you need, or from a simulator at that tolerance.
Forward derivatives keep no record, so they have no such limit.

**Pre-runs are fixed.** A scenario that starts from a pre-run
([Simulation and exploration](simulation.md)) starts from a fixed state: the derivatives do
not flow back through the pre-run.

**Second derivatives.** `jax.hessian(loss)`, where `loss` is a number made from `f`'s
outputs, gives the matrix of its second derivatives (for standard errors, or a
Newton-type method):

```python
loss = lambda x: jnp.sum(f(x)["conc"] ** 2)
H = jax.hessian(loss)(f.x0)
```

It works with both kinds of derivatives, with these limits; each raises an error that says
so:

- **models with extinction groups or switches**: not available. The derivative of an
  event's time is exact to first order only, so a second derivative through it would be
  slightly wrong. Take central differences of `jax.grad(loss)` instead.
- **forward derivatives**: differentiate the gradient forward, with `jax.hessian` or
  `jax.jacfwd(jax.grad(loss))`, not `jax.jacrev(jax.grad(loss))`.
- **large models** (those solved with sparse linear algebra): only with
  `derivatives="reverse"`.

## Simulating populations: `derivatives=False`

For a virtual population, a sweep or anything else that only needs the outputs, make the
function without derivatives:

```py
g = sim.function(sc, wrt=["CL", "V"], derivatives=False)
pop = jax.vmap(g)(X)                   # X: one row per patient, in g.names order
```

It is the same function (`g.names`, `g.x0`, `g(x)`, `g.status(x)`, `jax.jit`, `jax.vmap`),
but the runs are not recorded or replayed and the steps are chosen for the solution alone.
So it needs a few MB per member, and is 2–5 times faster than reverse derivatives. Its
outputs equal those of a function with derivatives to the solver's tolerance. `jax.grad`
or `jax.jacfwd` of it raise an error saying so.

## What it costs

Measured on the benchmark models (CPU):

- forward: one call `f(x)` with its derivatives costs about 2–3 forward solves, plus a
  fraction of a solve for each name in `wrt` (a third of a solve each on a 100-state model,
  less on larger ones); `jax.grad` and `jax.jacfwd` cost the same;
- reverse: one call `f(x)` costs about 2 to 4 forward solves (the run, then its replay); a
  gradient (`jax.grad`) costs 4–13 forward solves, whatever the number of names; `jax.jacfwd`
  grows with the number of names;
- `jax.vmap(f)` over many parameter sets gives what one call per set gives. The sets run in
  lockstep, so each step costs what the most demanding set needs. With reverse, each member
  carries its own record of the run, so memory grows with the batch.

Measured times and memory, by model size, CPU and GPU, are on the
[Performance](../background/performance.md) page.

`sim.function` is meant for estimation (tenqsp-fit) and for any JAX code of your own.
