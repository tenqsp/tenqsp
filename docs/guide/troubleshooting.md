# Troubleshooting

This chapter is organised by what you see: a message, a run that fails, a result that looks
wrong, a run that is slow. For each, the message as TenQSP prints it (shortened where
marked), what it means, and what to do. Every example on this page runs; the messages are
copied from its output.

Two habits catch most problems before they become symptoms: read `m.explain()` once for every
new model, and look at `m.diagnostics(sol)` after the first run
([Reading a model back](explain.md)).

```python
import numpy as np
import tenqsp as tq


def show(action):
    """Run action() and print the message of the error it raises."""
    try:
        action()
        print("no error")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
```

## The model text is refused

`tq.Model.from_text` and `tq.load` read the whole text before they complain, then raise one
`tq.NotationError` that lists every problem with its line number. Fix them from the top: a
later problem is sometimes a consequence of an earlier one.

```python
show(lambda: tq.Model.from_text("""
model gut
axis a in [0, 1] n=10 kind=one_way
param v=1
field I(a)
d_t(I) = -v*d_a(I)
    at a=1: outflow
"""))
```

```text
NotationError: The model text has 1 problem(s):

line 6: d_t(I) = -v*d_a(I)
the transport term must be in conservation form: write -d_a(v*I), not -v*d_a(I) (TenQSP
keeps every cell's books balanced only in this form).
```

The forms that are refused, with the message (shortened) and what to write instead:

| Written | Message | Instead |
|---|---|---|
| `d_t(I) = -v*d_a(I)` | the transport term must be in conservation form | `-d_a(v*I)` |
| `d_t(I) = d_a(v*I)` | transport is written with a minus, -d_a(v*I); to move toward lower a, give a negative velocity | `-d_a(v*I)`, or `-d_a(-v*I)` to move down |
| `d_t(F) = -d_x(D*d_x(F))` | diffusion is written +d_x(D*d_x(F)); a negative diffusion has no solution | `+d_x(D*d_x(F))` |
| `h(2)` where `h(a) = …` | 'h(2)': h is a named formula, not a function; write h(a) (or just h) | `h(a)` or `h` |
| `d_t(P) = d_a(I)` (P a compartment) | a compartment has no axis 'a'; to use a field's total, integrate it: int_a(…) | `int_a(I)` |
| `d_t(B)` twice | d_t(B) is given twice (also line 4); write one equation per field or slice, with all its terms | one equation |
| `compartment P = 1e7` | a declaration has no value; the value at time 0 is written P(0) = 1e7 | `P(0) = 1e7` |
| `dose B at t=0 to 140 every 24: …` | from 0 to 140 is not a whole number of steps of 24; the last dose must fall on the end time | `0 to 144 every 24` |
| a misspelt name in a formula, `Cl/V` | Unknown name 'Cl' at column 12 of 'ka*depot - Cl/V*centr' | the declared name |

Each message names the line, and for a formula points at the term. The next two sections show
two refusals that are about balance rather than spelling.

### "say what happens there": a one_way axis with no outflow

```python
show(lambda: tq.Model.from_text("""
model gut
axis z in [0, 1] n=10 kind=one_way
field G(z)
d_t(G) = -d_z(G)
"""))
```

```text
NotationError: The model text has 1 problem(s):

line 4: field G(z)
the flow carries G to the upper end of 'z'; say what happens there: 'at z=<end>: outflow' (it
leaves the model, recorded in the ledger; read it elsewhere as G(z=upper)), or a condition.
```

**What it means.** Along a `one_way` axis (age, progress through the gut) the flow carries
material to the upper end, and the model does not say what happens to it there. TenQSP will
not let it disappear unnoticed.

**What to do.** Write the end condition under the equation, usually `at z=1: outflow`, and
read what leaves where it goes: `d_t(depot) = G(z=1)/MTT` for drug that reaches the absorption
site. A model built in Python gets the same message as a `tq.ModelError` when it is compiled
or solved: add `m.at(G, z=1, outflow=True)`. See [Boundary conditions](boundaries.md).

### "so that what leaves one place arrives in the other": a flux that does not balance

```python
show(lambda: tq.Model.from_text("""
model two_spaces
param Q=5, V1=2, V2=4
compartment A volume=V1
compartment B volume=V2
flux J: A -> B = Q*A
d_t(A) = -J/V1
d_t(B) = J/V1
"""))
```

```text
NotationError: The model text has 1 problem(s):

line 8: d_t(B) = J/V1
flux J is an amount per time; in the equation of B divide it by B's volume, V2 (not V1), so
that what leaves one place arrives in the other.
```

**What it means.** A `flux` is an amount per time. In the equation of a concentration it must
be divided by that place's own volume, or drug is created or lost on the way.

**What to do.** Divide by the volume of the place whose equation it is: `d_t(B) = J/V2`. If
the flux arrives nowhere (`B`'s equation has no `+J`), the message says that instead:
"flux J arrives in B: d_t(B) needs exactly one term +J."

### A term that is missing from one equation

The same process written in several equations, without a named flux, is not an error when
one place is missing it: some models leave phage out of the adsorption balance on purpose.
`m.explain()` points it out ("Not in the equation of P: if it takes part (used up or
produced), the term is missing there"); see
[Reading a model back](explain.md#terms-shared-between-equations).

## A value or a name is not found

### `tq.MissingValues`

```python
pk = tq.Model.from_text("""
model pk
param ka, CL, V=30
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
dose depot at t=0 to 72 every 12: 100
observable conc = centr/V
""")
show(lambda: pk.solve((0, 96)))
```

```text
MissingValues: No value for ka, CL: declared without a value in the model, so give them when
solving or simulating, e.g. params={'ka': ..., 'CL': ...}.
```

**What it means.** `param ka, CL` declares parameters without a value, as you would for
parameters that are estimated or that differ between scenarios. TenQSP never guesses one.

**What to do.** Give them in the call (`params={"ka": 1.2, "CL": 4}`), in a scenario, or as
defaults in the model (`param ka=1.2`). `m.diagnostics()` without a solution needs them too,
since velocities and diffusivities may depend on them: pass a solution,
`m.diagnostics(m.solve(..., params={...}))`.

### A misspelt parameter

```python
show(lambda: pk.solve((0, 96), params={"kaa": 1.2, "CL": 4}))
```

```text
KeyError: "The model has no parameter 'kaa'. Did you mean 'ka'?"
```

A value given for a name the model does not have is an error, never ignored: otherwise the
run would silently use the default. The suggestion is the closest name. Check the case of
the letters too: `Cl` for `CL` is refused as well.

### A misspelt keyword argument

```python
show(lambda: pk.solve((0, 96), params={"ka": 1.2, "CL": 4}, rtoll=1e-8))
```

```text
TypeError: solve() got an unexpected keyword argument 'rtoll'
```

This is Python's own message: every option of `solve` and of the model-building calls is a
named argument, so a misspelt one is refused. `m.simulator` lists its options when it refuses
one ("Unknown simulator options ['rtoll']; they are ['atol', 'backend', …]"). The options of
`solve` are `times`,
`params`, `theta`, `method`, `rtol`, `atol`, `max_step`, `max_steps`, `ledger`, `regimen`,
`initial` and `backend`. `method` takes `"BDF"` or `"Radau"`; anything else is refused with a
`ValueError` naming them.

### An output that does not exist

```python
sol = pk.solve((0, 96), params={"ka": 1.2, "CL": 4}, times=np.linspace(0, 96, 97))
show(lambda: sol["concentration"])
show(lambda: sol.at("conc", 3.3))
show(lambda: pk.simulator(outputs=["concentration"]))
```

```text
KeyError: "No field, compartment or observable named 'concentration'; there are ['centr', 'conc', 'depot']."
KeyError: 't = 3.3 is not an output time; request it with times=[...].'
KeyError: "'concentration' is not an observable, field or compartment of the model. Observables: ['conc']; fields and compartments: ['depot', 'centr']."
```

## The run fails

### "BDF failed … retried with Radau", and doses too large for `atol`

A run that completes may still carry a warning. Here a bolus of 1e11 phage at t = 2 h, with
`atol = 1e-6`:

```python
bolus_text = """
model bolus
axis age in [0, 1] n=10 units=h kind=one_way
param k=1e-7, beta=100, g=1, K=1e9, delta=0.1, h_max=30, tau=0.5, s=0.03
field I(age)
h(age) = h_max*sigmoid((age - tau)/s)
d_t(B) = g*B*(1 - B/K) - k*B*P
d_t(I) = -d_age(I) - h*I   using sharp
    at age=0: I = k*B*P
    at age=1: outflow
d_t(P) = beta*int_age(h*I) - k*B*P - delta*P
B(0) = 1e8
dose P at t=2: 1e11
"""
bolus = tq.Model.from_text(bolus_text)
sol = bolus.solve((0, 3), times=[3], rtol=1e-8, atol=1e-6, backend="scipy")
print(sol.messages)
print(bolus.diagnostics(sol))
```

```text
["BDF failed on [2, 3] (Required step size is less than spacing between numbers.); retried
with Radau. The dose into 'P' at t = 2 raised its value by 1e+11; with atol below about 1e-03
its rounding errors can make BDF fail. Try atol=1e-03 or larger."]
...
[warning] P: the dose at t = 2 raised its value by 1e+11. That leaves rounding errors of about
1e-03 in the states it feeds, above atol (1e-06): the solver's error estimate is then noise,
and BDF fails or survives by chance (a slightly different rtol can change the outcome). Use
atol=1e-03 or larger.
```

**What it means.** A dose that raises a state by a large amount leaves rounding errors of
about 1e-14 times the rise in the states it feeds. With `atol` below that, the solver's error
estimate for those states is noise. BDF failed on the segment after the dose; the scipy
backend retried it with Radau, which completed it.

**What to do.** Take the `atol` the message gives (here 1e-3: a thousandth of a phage per mL
is far below anything that matters next to 1e11). The run then needs no retry:

```python
sol = bolus.solve((0, 3), times=[3], rtol=1e-8, atol=1e-3)
print(sol.messages)          # []
```

### `SolverError` from the JAX backend

The JAX backend has no Radau to fall back on. The same run fails there, with the same advice:

```python
show(lambda: bolus.solve((0, 3), times=[3], rtol=1e-8, atol=1e-6, backend="jax"))
```

```text
SolverError: The JAX solver failed at t = 2 (required step size is less than spacing between
numbers). The dose into 'P' at t = 2 raised its value by 1e+11; with atol below about 1e-03
its rounding errors can make BDF fail. Try atol=1e-03 or larger. The scipy backend retries a
failing segment with Radau: try backend='scipy'.
```

**What to do.** First what the message says about `atol`. If the run still fails, try
`backend="scipy"`, which retries a failing segment with Radau. A `SolverError` from the scipy
backend means Radau failed as well; its message says between which times.

A failure just after a condition switches (`k = k1 if C > MIC else k2`) has its own advice in
the message: if the model pushes itself straight back across the condition, no step size
works. Replace the switch by a smooth form (with `sigmoid`), or model what happens on the
boundary.

Inside `sim.function` (gradients) a failed run gives NaN instead of an error, so an
optimiser can step back; `f.status(x)` says why. See
[Gradients and sensitivities](gradients.md).

### "The solver took more than … steps": the run is crawling

Both backends stop after `max_steps` steps between two doses or events (100,000 by default),
rather than crawling on. To see the message, lower the limit:

```python
show(lambda: pk.solve((0, 96), params={"ka": 1.2, "CL": 4}, max_steps=20))
```

```text
SolverError: The solver took more than 20 steps between t = 0 and t = 12 and reached only
t = 0.0955461 (last step 0.017): it is crawling, not failing. The usual cause is a tolerance
near the limit of rounding for a fast process (here rtol = 1e-06, atol = 1e-06): loosen rtol
(e.g. 1e-8) or atol, or allow more steps (max_steps=...).
```

**What to do.** With the default limit this means the steps have become tiny. As the message
says, the usual cause is a tolerance near the limit of rounding for a fast process (fast
exchange between spaces, fast adsorption): loosen `rtol` to about 1e-8, or `atol`. If the
model really needs that many steps, raise `max_steps`.

## The result looks wrong

### A population dips below `atol` and regrows (or stays at 0)

Bacteria are killed for 10 h, then the drug stops. Two killing rates (a third follows in the
next section):

```python
def kill_model(kmax):
    return tq.Model.from_text(f"""
model kill
param g=1, kmax={kmax}, tstop=10
k_kill = kmax if t < tstop else 0
d_t(B) = g*B - k_kill*B
B(0) = 1e6
""")


t = np.linspace(0, 40, 81)
for kmax in [3.8, 4.0]:
    m = kill_model(kmax)
    sol = m.solve((0, 40), times=t)
    exact = 1e6 * np.exp(-(kmax - 1) * 10 + 30)
    print(f"kmax {kmax}: B(40) = {sol['B'][-1]:.3g}, exact {exact:.3g}")
    print(m.diagnostics(sol))
```

```text
kmax 3.8: B(40) = 1.55e+07, exact 7.39e+06
[warning] B: its total fell below atol (1e-06) at t ≈ 10 and grew back to 1.6e-05 by
t ≈ 12.5. Below atol the solver does not resolve a population, so this regrowth may be a
numerical artefact. Declare an extinction threshold (m.extinction) if it should die out, or
tighten atol if the remnant is real.
...
kmax 4.0: B(40) = -9.82e-09, exact 1e+06
[note] B: its total fell below atol (1e-06) at t ≈ 9.5 and stayed there. Below atol a
population is not resolved, so this solve does not decide whether it would have grown back.
If that matters, make it an explicit choice: declare an extinction threshold (m.extinction),
or tighten atol.
```

**What it means.** At its lowest the population was below `atol`, where the solver does not
resolve it: the first run regrew to twice the exact value, the second did not regrow at all
although the exact solution returns to 1e6. Neither is reliable.

**What to do.** Decide what a fraction of a cell means in your model:

- It is no cell (bacteria per mL in a patient): declare an extinction group,
  `extinction B below 1`. The population is set to 0 when it falls below one cell, at the
  exact crossing, and the result no longer depends on `atol`.
- It is real (a concentration, or a large volume): set `atol` well below the smallest value
  that matters. With `atol=1e-12` both runs above come within 0.03 % of the exact value
  (within 4 % at `atol=1e-9`).

See [Numerical choices](numerics.md#extinction-groups-and-noise-floors).

### Negative values

```python
m = kill_model(3.9)
sol = m.solve((0, 40), times=t)
print(f"B(40) = {sol['B'][-1]:.3g}")
print(m.diagnostics(sol).warnings[0])
```

```text
B(40) = -2.18e+06
[warning] B: went negative (down to -2.18e+06, beyond -10 x atol = -1e-05): the solver's
tolerance allowed an undershoot. Tighten atol/rtol, or check for a negative rate or
diffusivity.
```

**What it means.** TenQSP never clips a value: a negative value the solver makes is kept, and
reported once it is below −10 × `atol`. Here the population undershot to a tiny negative
number while it was below `atol`; once the killing stopped, it "grew" at rate 1, away from 0,
to −2e6. A negative population that grows is a symptom of this kind: the equations are
fine, the solve did not resolve the small numbers.

**What to do.** As above: an extinction group if small numbers mean nothing, a smaller `atol`
if they are real. If the values go negative while they are well above `atol`, look for a
negative rate or diffusivity in the model: `m.diagnostics()` warns about a negative
diffusivity, and about a negative velocity on a `one_way` axis.

### A population appears from nothing

An arm that was never given phage shows an epidemic; a strain that was never seeded grows. A
solver can leave round-off, such as 1e-12, in states that should be 0, and a population that feeds
itself (phage that infect and burst) grows from it. Declare the lineage as an extinction group
with a noise floor, `extinction lineage: P, I below 1 with noise_floor`; see
[Numerical choices](numerics.md#ghost-populations-and-the-noise-floor).

### The books do not close

`m.diagnostics(sol)` reports the largest unexplained amount in the ledger. Around 1e-15 it is
rounding (a note). Above 1e-6 it is a warning, "the books do not close; tighten rtol/atol":
look at `sol.ledger.table(name)` for the field or compartment named, and tighten the
tolerances.

## The run is slow

### Solves are slow with `sharp`

```python
for name, scheme in [("erlang", ""), ("sharp", "  using sharp")]:
    m = tq.Model.from_text(f"""
model transit
axis z in [0, 1] n=20 kind=one_way
param MTT=1.5
field G(z)
d_t(G) = -d_z(G/MTT){scheme}
    at z=1: outflow
dose G at t=0: 100*point(z, at=0)
""")
    sol = m.solve((0, 4), times=np.linspace(0, 4, 41))
    print(name, {k: sol.stats[k] for k in ("nfev", "njev", "nlu")})
```

```text
erlang {'nfev': 565, 'njev': 1, 'nlu': 51}
sharp {'nfev': 2069, 'njev': 19, 'nlu': 144}
```

**What it means.** With `sharp`, the solver follows the front from cell to cell, so its steps
are at most about one cell crossing time (`m.diagnostics()` gives it), and it needs more
right-hand-side evaluations (`nfev`), Jacobians (`njev`) and LU factorisations (`nlu`). On the
toy phage model of [the benchmarks](../benchmarks.md) (1,261 states, 24 h) that is 7.25 s
against 0.67 s with `erlang`.

**What to do.** Use `sharp` only on the axes where the delay must be sharp (`using sharp on a`
for one axis of several), and check how many cells the outputs you care about really need
([a convergence study](numerics.md#a-convergence-study)). If the spread of a transit chain is
acceptable, or is the model's own variability, `erlang` is the faster choice, and an exact
transit chain.

### The JAX backend compiles again

The JAX backend compiles a model once, then runs new values without rebuilding. But the
number of doses, infusions and output times sits in slots rounded up to a power of two (at
least 8). A run that needs more slots builds once more:

```python
import time

reg8 = tq.bolus("depot", 100, start=0, every=12, n=8)
reg9 = tq.bolus("depot", 100, start=0, every=12, n=9)
times = np.linspace(0, 120, 11)
values = {"ka": 1.2, "CL": 4}
for label, reg in [("8 doses", reg8), ("8 doses again", reg8), ("9 doses", reg9)]:
    start = time.perf_counter()
    pk.solve((0, 120), times=times, params=values, regimen=reg, backend="jax")
    print(f"{label}: {time.perf_counter() - start:.2f} s")
```

```text
8 doses: 0.96 s
8 doses again: 0.04 s
9 doses: 0.96 s
```

(Times from one machine; the pattern is what matters.) The 9th dose needs 16 slots, so the
program is built again; any run with 9 to 16 doses then reuses it. The same holds for output
times: 11 and 12 output times share a build, 20 need the next one.

**What to do.** For a scan or a population, keep the number of doses and of output times the
same in every run, or at least within one power of two (9 to 16 share a build, as do 17 to
32). Changing parameter values, dose amounts and dose times never rebuilds.

### A large model is slow or runs out of memory

`m.summary()` says how many state variables a grid gives, before you build anything, and
`m.compile().summary()` the memory of the Jacobian and its LU factors
([Reading a model back](explain.md#msummary-how-big-is-it)). The scipy backend handles 10^5
states in a few minutes; above about 250 states the JAX backend switches to its sparse tier
automatically. Gradients of large models need much more memory than solves: see
[Gradients and sensitivities](gradients.md).
