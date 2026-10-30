# Reading a model back

A model that runs is not yet a model you trust. Before you look at the numbers you want to
know what TenQSP made of what you wrote: which terms it recognised, which numerical scheme
carries each one, what happens at each end of each axis, what starts at zero. After a run
you want to know whether the solver resolved what you care about, and whether every amount
is accounted for.

This chapter goes through the tools that answer those questions: `m.explain()` (the model in
words), `m.diagnostics()` (accuracy guidance, and checks of a solution), the **ledger** (the
books of every field and compartment), `m.summary()` and `m.validate()`, and the ways to
write a model back out: `m.to_text()`, `m.to_latex()`, `m.to_ir()` and saving to a file.
None of them changes the model.

## The example

An infection-age model of bacteria and phage (as in
[Tutorial 7](../tutorials/07_infection_age.ipynb)): bacteria `B`,
free phage `P`, and infected cells `I` spread over infection age `a`. Cells are infected at
rate `k*B*P`, enter at age 0, and lyse with a hazard `h(a)` that rises near age `tau`,
releasing `beta` phage each. An extinction group removes the bacteria if they fall below one
cell.

=== "Notation"

    ```tq
    model phage_latency
    axis a in [0, 2] n=40 units=h kind=one_way        # infection age
    param k=1e-9, beta=50, g=1, Bmax=1e9, delta=0.1, h_max=5, tau=0.8, s=0.05
    compartment B                                     # uninfected bacteria
    compartment P                                     # free phage
    field I(a)                                        # infected cells by infection age
    h(a) = h_max*sigmoid((a - tau)/s)                 # lysis hazard, rises near age tau
    d_t(B) = g*B*(1 - B/Bmax) - k*B*P
    d_t(I) = -d_a(I) - h(a)*I          using sharp
        at a=0: I = k*B*P                             # newly infected cells enter at age 0
        at a=2: outflow
    d_t(P) = beta*int_a(h(a)*I) - k*B*P - delta*P
    B(0) = 1e6
    P(0) = 1e7
    dose P at t=0: 1e8
    observable burst_rate = int_a(h(a)*I)
    extinction B below 1
    ```

=== "Python"

    ```python
    import numpy as np
    import tenqsp as tq

    m = tq.Model("phage_latency")
    a = m.axis("a", 0, 2, 40, units="h", kind="one_way")
    k, beta, g, Bmax, delta, h_max, tau, s = m.params(
        k=1e-9, beta=50, g=1, Bmax=1e9, delta=0.1, h_max=5, tau=0.8, s=0.05
    )
    B = m.compartment("B")
    P = m.compartment("P")
    I = m.field("I", axes=[a])
    h = m.define("h", h_max * tq.sigmoid((a - tau) / s))
    m.ode(B, g * B * (1 - B / Bmax) - k * B * P)
    m.advect(I, axis=a, velocity=1, scheme="sharp")
    m.hazard(I, h)
    m.at(I, a=0, value=k * B * P)
    m.at(I, a=2, outflow=True)
    m.ode(P, beta * m.integral(h * I, over=a) - k * B * P - delta * P)
    m.init(B, 1e6)
    m.init(P, 1e7)
    m.dose(P, times=[0], amounts=1e8)
    m.observable("burst_rate", m.integral(h * I, over=a))
    m.extinction(B, threshold=1)
    ```

```python
phage = m
sol = phage.solve((0, 24), times=np.linspace(0, 24, 49))
```

## `m.explain()`: the model in words

```python
print(phage.explain())
```

`explain()` reads the model only; it does not solve it and needs no parameter values beyond
those in the model. It returns plain text, in sections. They are described here in the order
they appear.

### Size

The first lines count the state variables: one per cell of each field, one per compartment.

```text
Model 'phage_latency': 42 state variables
  field I: 40 = 40 (a)
  compartments: 2 (B, P)
```

### Equations

Each field and compartment, term by term: what the term means and which scheme carries it.

```text
Equations
  I (over a)
    I: - h*I — a loss at the per-capita rate h (its own ledger line).
    transport along a, velocity 1 (sharp): MUSCL with the van Leer limiter: a straight line
      in each cell, its slope limited so no new peaks or negative values appear. Fronts keep
      their shape and refining the grid converges to transport without spread. ...
  B (a compartment)
    d_t(B) = g*B*(1 - B/Bmax) - k*B*P — as written.
```

What you can read here:

- **Losses.** A term `-r*F` is recognised as a loss at per-capita rate `r` and gets its own
  line in the ledger, so a solution can say how many cells lysed.
- **Transport.** With the default `erlang` scheme the transport is described as the chain of
  transit compartments it is: how many, the rate from one to the next (|velocity| ÷ cell
  width), the mean transit time and the CV. With `sharp`, the scheme and its cost. See
  [Numerical choices](numerics.md) for when to use which.
- **Diffusion**, and transport plus diffusion on the same axis read as a Fokker–Planck
  equation: the single-cell process `dx = v dt + sqrt(2 D) dW` it describes.
- **Transitions**: whether they act on the cells themselves (switching) or redistribute a
  production at the parent's rate (mutation at division). See
  [Categories and transitions](transitions.md).
- Anything else is added cell by cell, "as written".

### Terms shared between equations

The same process often appears in several equations: adsorption `k*B*P` removes bacteria,
removes phage and creates infected cells at age 0. TenQSP matches such terms across the
equations and boundary conditions and lists every place with its sign, or its factor for a
conversion (here `beta` phage per lysed cell):

```text
Terms shared between equations (the same term, matched by TenQSP)
  I*h_max/(1 + exp(-(a - tau)/s)): I -1, P x (beta).
  B*P*k: B -1, P -1, I at a=0 +1.
```

A term that is proportional to a place but missing from that place's own equation is
pointed out. Here the phage are not used up by adsorption: `k*B*P` leaves `B` and enters
`I`, but `P`'s equation has no `- k*B*P`.

```python
missing = tq.Model.from_text("""
model missing_term
param k=1e-9, beta=50, g=1, delta=0.1, lam=2
d_t(B) = g*B - k*B*P
d_t(I) = k*B*P - lam*I
d_t(P) = beta*lam*I - delta*P
B(0) = 1e6
P(0) = 1e7
""")
print(missing.explain())
```

```text
Terms shared between equations (the same term, matched by TenQSP)
  B*P*k: B -1, I +1. Amounts balance: what one place loses, the others gain. Not in the
    equation of P: if it takes part (used up or produced), the term is missing there.
  I*lam: I -1, P x (beta).
```

This is a note, not an error: some models leave the phage out of the balance on purpose.
If it is a mistake, it is the kind that runs without complaint and gives the wrong burst.

### Boundaries

Every end of every continuous axis, including the ones you did not mention, so a default is
never silent:

```text
Boundaries
  I at a=lower (a = 0): held at k*B*P.
  I at a=upper (a = 2): what the flow carries there goes out, leaving the model.
```

An end with no condition is sealed: "sealed, nothing crosses (the default)". See
[Boundary conditions](boundaries.md).

### Starting values

The values given at time 0, then everything that starts at 0, as in NONMEM and Monolix.
A forgotten starting value shows up here.

```text
Starting values
  B = 1e6
  P = 1e7
  Start at 0 (not given, or given as 0): I.
```

### Doses

The model's own doses (a regimen given at simulation replaces them; see
[Doses and regimens](doses.md)):

```text
Doses
  P: 1e8 at t = 0
  Each dose is added at once (a bolus) and the solver restarts there with a fresh step size.
```

### Extinction groups

Each group, its members, its threshold and whether it has a noise floor. What they do, and
when you need one, is in [Numerical choices](numerics.md#extinction-groups-and-noise-floors).

```text
Extinction groups
  extinct_B: B. When their total falls below 1 (after having been above it), all of them are
    set to 0 at that moment and the solve restarts; a group that has never reached the
    threshold (e.g. mutants seeded from 0) is never touched.
```

### Numerical safeguards

The complete list of what the solver does beyond integrating your equations. There are no
hidden clamps: no value is ever clipped, capped or floored to keep a run going.

```text
Numerical safeguards (the complete list)
  No value is clipped, capped or floored: negative values, if the solver makes any, are
  kept and reported by diagnostics(sol).
  Solver: BDF unless solve(method=...) says otherwise; a segment where BDF fails is retried
  once with Radau (said in sol.messages).
  Each segment (between doses) starts with a fresh step, never below 1e-08 x its length.
```

Noise floors you declared and the rule used at ends with a flux condition are added to this
list when the model has them.

## `m.diagnostics()`: accuracy guidance

`m.diagnostics()` looks at every transported axis and reports, per field and axis:

| Quantity | What it tells you |
|---|---|
| **cell crossing time**: the cell width ÷ the fastest velocity | with `sharp`, the solver follows a front from cell to cell, so its steps are at most about this long; it is what makes `sharp` slower |
| **CV of the transit chain** (`erlang`, uniform cells, a constant velocity) | the spread that the chain of n cells gives, 1/√n |
| **cell Péclet number**: \( |v| \times \text{width} / D \) (transport and diffusion on one axis) | above 2, first-order upwind (`erlang`) adds more spread than \(D\) itself, so the spread you see is mostly the grid's |

```python
print(phage.diagnostics())
```

```text
[note] field 'I', axis 'a': fastest transport crosses a cell in 0.05 time units (|v| up to 1,
cells 0.05 wide); with sharp the solver follows the front cell by cell, so its steps are at
most about this long.
```

The Péclet number needs transport and diffusion on the same axis. A drug carried through a
tissue at speed `v` while it diffuses and is eliminated:

=== "Notation"

    ```tq
    model tissue
    axis x in [0, 1] n=20
    param v=0.5, D=0.01, kel=0.2
    field C(x)
    d_t(C) = -d_x(v*C) + d_x(D*d_x(C)) - kel*C
        at x=1: outflow
    dose C at t=0: 10*point(x, at=0)
    ```

=== "Python"

    ```python
    m = tq.Model("tissue")
    x = m.axis("x", 0, 1, 20)
    v, D, kel = m.params(v=0.5, D=0.01, kel=0.2)
    C = m.field("C", axes=[x])
    m.advect(C, axis=x, velocity=v)
    m.diffuse(C, axis=x, D=D)
    m.hazard(C, kel)
    m.at(C, x=1, outflow=True)
    m.dose(C, times=[0], amounts=10, shape=tq.point(x, at=0))
    ```

```python
tissue = m
print(tissue.diagnostics())
```

```text
[note] field 'C', axis 'x': fastest transport crosses a cell in 0.1 time units (|v| up to
0.5, cells 0.05 wide); erlang: a chain of 20 transit compartments, CV 0.224.
[note] field 'C', axis 'x': cell Péclet number |v| x width / D is up to 2.5: transport
dominates diffusion within a cell: first-order upwind (erlang) then adds more spread
(|v| x width / 2) than D itself, so the spread you see is mostly the grid's. Use more cells
or sharp if D is meant to set it.
```

Here the grid's own spread, 0.5 × 0.05 / 2 = 0.0125, is larger than `D` = 0.01: more than
half of the spreading you would see comes from the grid, not from the model. Twice as many
cells, or `using sharp`, brings it below `D`.

A model whose parameters have no values yet (`param ka`) cannot be diagnosed before a solve,
since velocities and diffusivities may depend on them: pass a solution instead.

## `m.diagnostics(sol)`: checks of a solution

With a solution, `diagnostics` adds what only a run can show:

| Check | Level | Meaning |
|---|---|---|
| solver messages | warning | a segment where BDF failed and was retried with Radau |
| a dose larger than `atol` can carry | warning | a dose that raised a state by more than about `atol` / 1e-14 leaves rounding errors above `atol`; the message names the `atol` to use ([Numerical choices](numerics.md#tolerances)) |
| negative values | warning | a field or compartment below −10 × `atol` |
| fell below `atol` and grew back | warning | below `atol` the solver does not resolve a population, so the regrowth may be an artefact |
| fell below `atol` and stayed there | note | the run did not decide whether it would have grown back |
| ledger closure | note, or warning above 1e-6 | the largest unexplained amount, relative |

The two checks on populations below `atol` skip anything covered by an extinction group
whose threshold is at least `atol`: there the choice has been made explicitly.

```python
print(phage.diagnostics(sol))
```

```text
[note] field 'I', axis 'a': fastest transport crosses a cell in 0.05 time units (|v| up to 1,
cells 0.05 wide); over the run that is up to 480 cells; with sharp the solver follows the front
cell by cell, so its steps are at most about this long.
[note] I: its total fell below atol (1e-06) at t ≈ 12 and stayed there. Below atol a
population is not resolved, so this solve does not decide whether it would have grown back.
If that matters, make it an explicit choice: declare an extinction threshold (m.extinction),
or tighten atol.
[note] ledger: largest unexplained amount, relative: 1.5e-15
```

The bacteria went extinct (their group removed them at the crossing), the phage kept
nothing to infect, and the infected cells drained out. The note on `I` says the solve did
not decide whether a remnant below `atol` would matter; here it cannot, since nothing is
left to infect.

The result is a report, the same kind `m.validate()` returns, so it can also be read in code:

```python
report = phage.diagnostics(sol)
print(report.ok, len(report.warnings), len(report.notes))
for issue in report.notes:
    print(issue.level, issue.where)
```

[Troubleshooting](troubleshooting.md) shows each warning on a run that triggers it, and what
to do.

## The ledger: where every amount came from

Every solution carries a **ledger** (unless the solve was run with `ledger=False`). For each
field and compartment it books the amount at the start, what each term, boundary, transition,
dose and extinction added or removed, and the amount at the end. The amount of a field is its
density times the cell widths (and its weights, for concentrations), summed.

```python
print(sol.ledger.names)
print(sol.ledger.table("P"))
```

```text
('I', 'B', 'P')
P: from t = 0 (start) to t = 24
  amount at the start                                     1e+07
  rate: beta * int_a                               +1.42215e+10
  rate: -k * B * P                                 -2.85138e+08
  rate: -delta * P                                 -1.13506e+10
  doses                                                  +1e+08
  amount at the end                                 2.69577e+09
  unexplained                                         -1.29e-05  (largest, relative: 1.5e-15)
```

**Unexplained** is the amount at the end minus the start and every channel: what the books
fail to account for. TenQSP integrates the channels alongside the state, so it is rounding,
not truncation error: around 1e-15 of the amounts involved. A large value means the
tolerances are too loose for the run, or a term was booked where it does not act.

| | |
|---|---|
| `sol.ledger.names` | the fields and compartments booked |
| `sol.ledger.table(name)` | the books of one (or, with no name, all) as text |
| `sol.ledger.check()` | per name, the largest unexplained amount relative to the amounts involved |
| `sol.ledger.unexplained(name)` | the unexplained amount at each output time |
| `sol.ledger.moved[name][channel]` | cumulative amount moved in (+) or out (−) since the start, at each output time |
| `sol.ledger.rates[name][channel]` | the rate of each channel at each output time |
| `sol.ledger.kinds[name][channel]` | reaction, hazard, transition, boundary, rate, dose, infusion, bioavailability (`not absorbed`) or extinction |

```python
B_books = sol.ledger.moved["B"]
print(list(B_books))                     # the channels of B, extinction among them
assert max(sol.ledger.check().values()) < 1e-10
print(sol.extinctions)                   # (time, group) of each extinction
```

The ledger answers questions a plot cannot: how many cells were infected over the run (the
adsorption channel of `B`), how much drug left through the outflow of a gut model, how much
an extinction removed. A dose given with a bioavailability is booked as given, and the part
that did not arrive as `not absorbed`, so the books close against the regimen.

## `m.summary()`: how big is it

```python
print(phage.summary())
```

```text
Model 'phage_latency': 42 state variables
  field I: 40 = 40 (a)
  compartments: 2 (B, P)
```

`m.summary()` counts state variables per field and compartment, and which axes drive the
count, without building anything: useful before you refine a grid. Above 100,000 state
variables it adds a warning with ways to make the model smaller. For the memory the solver
will need, ask the built model: `m.compile().summary()` compiles the model, times one
right-hand-side call and one Jacobian, and gives the size of the state, of the sparse
Jacobian, and of the LU factors of the matrix the solver factorises to take its steps.

```python
print(phage.compile().summary())
```

```text
Compiled model 'phage_latency': 42 state variables
  field I: 40 = 40 (a)
  compartments: 2 (B, P)
  parameters: 8 numbers in theta
  compile: right-hand side 0.14 s, Jacobian 0.22 s
  one right-hand-side call: 0.249 ms
  Jacobian: sparse, 125 non-zeros of 42 x 42; 7 forward and 1 reverse passes + 2 for the
  compartments (1 integral values lifted); one evaluation 1.7 ms
  memory: state 0.00034 MB; Jacobian 0.0017 MB sparse (a dense one would need 0.0141 MB); its
  LU factors (the solver's I - hJ) 0.011 MB
```

The times are those of one machine and vary from run to run.

## `m.validate()`: every problem at once

`m.validate()` checks the whole model and lists every error, warning and note in one
report. It raises `tq.ModelError` when there are errors (`raise_on_error=False` returns the
report instead). Solving and compiling validate first, so you rarely need to call it; it is
useful while a model built in Python is still incomplete.

```python
draft = tq.Model("draft")
z = draft.axis("z", 0, 1, 10, kind="one_way")
G = draft.field("G", axes=[z])
draft.advect(G, axis=z, velocity=1)
print(draft.validate(raise_on_error=False))
```

```text
[warning] field 'G', axis 'z': no inflow given at the lower end of this one_way axis, so
nothing enters there (inflow 0). Add an inflow condition if, for example, newly infected
cells should enter at age 0.
[error] field 'G', axis 'z': the flow carries G to the upper end of 'z'; say what happens
there: 'at z=<end>: outflow' (it leaves the model, recorded in the ledger; read it elsewhere
as G(z=upper)), or a condition.
[note] field 'G': no initial condition given; starts at 0.
```

A model read from text is validated as it is read: `tq.Model.from_text` raises a
`tq.NotationError` that lists every problem with its line.

## Writing a model out

### `m.to_text()`

`m.to_text()` writes any model as notation, also one built in Python. Reading the text back
gives the same model: `tenqsp.canonical.differences` compares two models and lists what
differs (nothing, here).

```python
from tenqsp.canonical import differences

text = tissue.to_text()
print(text)
again = tq.Model.from_text(text)
print(differences(tissue.to_ir(), again.to_ir()))      # []
```

```text
model tissue

axis x in [0, 1] n=20

param v = 0.5
param D = 0.01
param kel = 0.2

field C(x)

d_t(C) = -d_x(v*C) + d_x(D*d_x(C)) - kel*C
    at x=1: outflow

dose C at t=0: 10*point(x, at=0)
```

This is the easiest way to show a colleague a model built in a Python loop, or to check that
a Python model says what you meant.

### `m.to_latex()`

The equations as a paper shows them, an amsmath `align*` block: equations, named formulas,
fluxes, boundaries and starting values, with every transition written out category by
category. Doses and outputs are a protocol, not equations, and are left out.

```python
print(tissue.to_latex())
```

The equation line of that block, as MathJax shows it (the block goes on with the boundary at
`x = 1`):

$$
\begin{align*}
\frac{\partial C}{\partial t} &= -\frac{\partial}{\partial x}\left(v \, C\right) + \frac{\partial}{\partial x}\left(D \, \frac{\partial C}{\partial x}\right) - \mathrm{kel} \, C
\end{align*}
$$

### `m.to_ir()`, saving and loading

`m.to_ir()` is the complete model description as plain data: what every other tool here
reads. It saves to JSON and loads back. A model can equally be kept as notation in a `.tq`
file, which is easier to read and to compare between versions:

```python
from pathlib import Path

Path("tissue.tq").write_text(tissue.to_text())
m1 = tq.load("tissue.tq")                          # the same model
m2 = tq.load("tissue.tq", cells={"x": 40})         # the same model on a finer grid

Path("tissue.json").write_text(tissue.to_ir().to_json())
m3 = tq.Model.from_ir(tq.ModelIR.from_json(Path("tissue.json").read_text()))

print(differences(tissue.to_ir(), m1.to_ir()), differences(tissue.to_ir(), m3.to_ir()))
print(m2.summary())
```

`tq.load` reads `.tq` files only. `cells=` changes the number of cells of continuous axes and
`params=` replaces default values, so one file serves a whole convergence study or a set of
parameter tables. A model read from a file or from JSON is an ordinary `tq.Model`; objects
are available by name (`m1["C"]`), and Python calls may add to it.
