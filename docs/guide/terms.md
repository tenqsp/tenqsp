# Reactions, transport and diffusion

The equation of a field, `d_t(F) = …`, is written as in a paper: a sum of terms. TenQSP
splits the right-hand side into its terms (the parts joined by `+` and `-`) and recognises
each one: transport along an axis, diffusion, a loss, a transition between categories, or a
reaction. What it recognises decides how the term is computed, how it appears in the
ledger, and how `explain()` describes it. This chapter goes through the kinds of terms, the
two transport schemes, and the forms TenQSP refuses.

## How an equation is read

| Term | Read as | Python |
|---|---|---|
| `-d_a(v*F)` | **transport** of `F` along `a` at velocity `v` | `m.advect(F, axis=a, velocity=v)` |
| `-d_a(F)` | transport at velocity 1 | `m.advect(F, axis=a, velocity=1)` |
| `d_x(D*d_x(F))` | **diffusion** with diffusivity `D` | `m.diffuse(F, axis=x, D=D)` |
| both of the above | a **Fokker–Planck** equation (cells that drift and jitter) | `m.fokker_planck(F, axis=x, drift=w, D=D)` |
| `-r*F`, with `r` free of `F` | a **loss** at per-capita rate `r`, its own line in the ledger | `m.hazard(F, r)` |
| `trans_s(M, X)` | **transitions** between the categories of `s` | `m.transition(F, axis=s, matrix=M, of=X)` ([next chapter](transitions.md)) |
| anything else | a **reaction**, added cell by cell (births, sources, interactions) | `m.reaction(F, expr)` |
| `d_t(P) = …` for a compartment | the whole right-hand side, as an ODE | `m.ode(P, rhs)` |

The order of the terms does not matter, and the Python calls can be made in any order. In
Python each call adds one term; several `m.reaction` calls on the same field add up.

## Transport

Transport moves a field along a continuous axis: cells ageing, maturing, drifting toward
tolerance, drug moving down the gut. It is written in **conservation form**,
`-d_a(v*F)`: the rate of change is minus the slope of the flux `v*F`. In this form what leaves
one cell is exactly what enters the next, so the books balance cell by cell.

- The **velocity** is what multiplies `F` inside the derivative. It may be a number, a
  parameter, a formula of the axis coordinate (`w*(0.3 - z)`), of time, or of the state.
- A velocity may be **negative** (movement toward lower values) or change sign along the
  axis, on a `bidirectional` or `spatial` axis. To move toward lower `a`, write
  `-d_a(-v*I)`. On a `one_way` axis the velocity must stay ≥ 0.
- The flux at each face between cells is taken from the side the flow comes from
  (**upwinding**): the value cannot become negative because of the transport. This is
  automatic, whatever the sign of the velocity.
- A velocity may depend on the transported field itself (in `-d_x(0.3*u*u)` the velocity
  is `0.3*u`) or on another field. **Chemotaxis**, cells moving up the gradient of a
  nutrient `S`, uses `grad_x(S)`, the slope of `S` along `x`. It is allowed only inside a
  velocity along `x`.

Bacteria that grow on a nutrient, swim up its gradient and spread at random, with the
nutrient held at 1 at the surface (`x = 1`) and diffusing into the medium:

=== "Notation"

    ```tq
    model chemotaxis
    axis x in [0, 1] n=40 units=mm kind=spatial
    param chi=0.02, Db=1e-3, Ds=0.01, g=1, ks=0.5, uptake=1e-8
    field B(x)                            # bacteria
    field S(x)                            # nutrient
    d_t(B) = -d_x(chi*grad_x(S)*B) + d_x(Db*d_x(B)) + g*S/(ks + S)*B
    d_t(S) = d_x(Ds*d_x(S)) - uptake*B*S
        at x=1: S = 1
    B(0) = 1e6*exp(-x/0.1)
    ```

=== "Python"

    ```python
    import numpy as np
    import tenqsp as tq

    m = tq.Model("chemotaxis")
    x = m.axis("x", 0, 1, 40, units="mm", kind="spatial")
    chi, Db, Ds, g, ks, uptake = m.params(chi=0.02, Db=1e-3, Ds=0.01, g=1, ks=0.5,
                                          uptake=1e-8)
    B = m.field("B", axes=[x])
    S = m.field("S", axes=[x])
    m.advect(B, axis=x, velocity=chi * tq.grad(S, x))
    m.diffuse(B, axis=x, D=Db)
    m.reaction(B, g * S / (ks + S) * B)
    m.diffuse(S, axis=x, D=Ds)
    m.hazard(S, uptake * B)
    m.at(S, x=1, value=1)
    m.init(B, 1e6 * tq.exp(-x / 0.1))
    ```

In Python the slope is `tq.grad(S, x)`; the notation also accepts `grad(S, x)`. The ends
not mentioned are sealed: nothing crosses them (see [Boundary conditions](boundaries.md)).

## Transport schemes: erlang or sharp

On a grid, transport has to be approximated. TenQSP offers two ways, chosen per equation.

**`erlang` (the default)** is first-order upwinding. On a uniform grid of N cells it is
*exactly* a chain of N transit compartments: each cell passes its content to the next at
rate |velocity| ÷ cell width. The time to cross the axis therefore follows an Erlang
(gamma) distribution with mean L/v and coefficient of variation (CV) **1/√N**. This spread
is part of the model, as in a NONMEM transit model: the number of cells sets the variability
of the delay, so refining the grid changes the model. It is not a convergence parameter.

**`sharp`** is a second-order scheme (MUSCL with the van Leer limiter): a straight line in
each cell, its slope limited so that no new peaks or negative values appear. Fronts keep
their shape, and refining the grid converges to transport without spread: a delay of almost
exactly L/v. Write it at the end of the equation, `using sharp`, or `using sharp on a` to
choose it for one axis of a field that moves along several. In Python:
`m.advect(F, axis=a, velocity=v, scheme="sharp")`.

A dose travelling through five transit cells, then twenty, with each scheme (mean transit
time 2):

```python
TRANSIT = """
model transit
axis z in [0, 1] n={n} kind=one_way
param MTT=2
field G(z)
d_t(G) = -d_z(G/MTT)  {scheme}
    at z=1: outflow
d_t(out) = G(z=1)/MTT
dose G at t=0: 100*point(z, at=0)
"""

t = np.linspace(0, 6, 601)
for scheme in ("", "using sharp"):
    for n in (5, 20):
        tr = tq.Model.from_text(TRANSIT.format(n=n, scheme=scheme))
        sol = tr.solve((0, 6), times=t)
        rate = np.gradient(sol["out"], t)            # arrival rate at the end
        mean = np.trapezoid(t * rate, t) / sol["out"][-1]
        sd = np.sqrt(np.trapezoid((t - mean) ** 2 * rate, t) / sol["out"][-1])
        print(f"{scheme or 'erlang':12s} n={n:2d}  mean {mean:.2f}  CV {sd / mean:.2f}")
```

The `erlang` CVs are 0.44 and 0.22, i.e. 1/√5 and 1/√20; the `sharp` ones are about 0.35 and
0.12, and keep falling as the grid is refined (0.04 with 80 cells). `explain()` says which
scheme carries each transport and, for `erlang`, the chain it amounts to:

```python
print(tq.Model.from_text(TRANSIT.format(n=5, scheme="")).explain().split("\n\n")[1])
```

**Which to choose.**

- `erlang` when the delay is meant to be variable: transit compartments, maturation with
  individual variation (the CV is then chosen through N), or when speed matters most.
- `sharp` when the axis is a clock or a position and the timing should be exact: infection
  age with a latent period, a lifespan, a front moving through tissue.
- `sharp` costs more: the solver follows the front from cell to cell with small steps. On
  the measured models it took about 3× as long on a 40-cell infection-age model (0.63 s
  against 0.22 s) and about 10× on a 30 × 40 phage model (7.3 s against 0.67 s). See
  [Numerical choices](numerics.md).

[Tutorial 1](../tutorials/_build/01_transit_compartments.ipynb) shows the transit-chain
reading in detail.

## Diffusion

`d_x(D*d_x(F))` spreads a field along `x` with diffusivity `D`: drug diffusing into a
biofilm, cells moving at random, random heritable changes of a phenotype. It is always
written with a plus sign. `D` may vary along the axis (`D0*(1 + x)`); at a face between
two cells TenQSP uses the harmonic mean of the two cells' values, so a cell with `D = 0`
blocks the flux. Ends not mentioned are sealed. In Python: `m.diffuse(F, axis=x, D=D)`.

## Transport plus diffusion: a Fokker–Planck equation

Single cells whose state `z` drifts at speed `w` and jitters at random,
\( dz = w\,dt + \sqrt{2D}\,dW \) (a stochastic differential equation in Itô form), have a
density that obeys

$$
\partial_t F = -\partial_z\big((w - \partial_z D)\,F\big) + \partial_z\big(D\,\partial_z F\big).
$$

The transport runs at the drift *minus the slope of D*. When `D` is the same everywhere the
correction is 0 and the equation is transport plus diffusion as written. `m.fokker_planck`
writes the two terms from the drift and `D`, working out the slope; here `D = D0*(1 + z)`, so
the velocity becomes `w - D0`:

=== "Notation"

    ```tq
    model tolerance
    axis z in [0, 1] n=30                  # tolerance level
    param w=0.2, D0=0.01, g=0.5, kd=0.3
    field N(z)
    d_t(N) = -d_z((w - D0)*N) + d_z(D0*(1 + z)*d_z(N)) + g*(1 - z)*N - kd*N   using sharp
    N(0) = 1e6*normal(z, mean=0.2, sd=0.05)
    ```

=== "Python"

    ```python
    m = tq.Model("tolerance")
    z = m.axis("z", 0, 1, 30)
    w, D0, g, kd = m.params(w=0.2, D0=0.01, g=0.5, kd=0.3)
    N = m.field("N", axes=[z])
    m.fokker_planck(N, axis=z, drift=w, D=D0 * (1 + z))
    m.reaction(N, g * (1 - z) * N)
    m.hazard(N, kd)
    m.init(N, total=1e6, shape=tq.normal(z, mean=0.2, sd=0.05))
    ```

`m.fokker_planck` uses the `sharp` scheme by default: `D` is meant to set the spread, and
`erlang` would add the grid's own spread (about |w| × cell width ÷ 2) to it. `explain()`
reads any equation with both terms back as the single-cell process it describes
("single cells follow dz = (w) dt + sqrt(2 D) dW"). `D` may vary through the coordinate; if
it depends on a field, write the two terms yourself with `grad`.
[Tutorial 12](../tutorials/_build/12_drug_tolerance_continuum.ipynb) uses this for a tumour's
continuum of drug tolerance.

## Losses

A term `-r*F` whose rate `r` does not contain `F` is a **loss** at per-capita rate `r`:
natural death, killing by a drug, elimination. `-B/tau` is a loss at rate `1/tau`, and
`-(kd + kc)*B` one at rate `kd + kc`. Each loss term gets its own line in the
[ledger](explain.md), so a solution can say how many cells died of each cause:

=== "Notation"

    ```tq
    model kill
    axis z in [0, 1] n=10                 # dormancy level
    param g=1, kd=0.1, kc=2, ke=0.3
    field B(z)
    d_t(B) = g*(1 - z)*B - kd*B - kc*(1 - z)*C*B
    d_t(C) = -ke*C
    B(0) = 1e6
    C(0) = 1
    ```

=== "Python"

    ```python
    m = tq.Model("kill")
    z = m.axis("z", 0, 1, 10)
    g, kd, kc, ke = m.params(g=1, kd=0.1, kc=2, ke=0.3)
    B = m.field("B", axes=[z])
    C = m.compartment("C")
    m.reaction(B, g * (1 - z) * B)
    m.hazard(B, kd)                       # natural death
    m.hazard(B, kc * (1 - z) * C)         # killed by the drug, less when dormant
    m.ode(C, -ke * C)
    m.init(B, 1e6)
    m.init(C, 1)
    ```

```python
sol = m.solve((0, 10), times=[0, 10])
print(sol.ledger.table("B"))          # one line for growth, one for each cause of death
```

The model is the same whichever way the equation is written; `-kd*B - kc*(1 - z)*C*B` and
`-(kd + kc*(1 - z)*C)*B` only differ in how the ledger splits the loss.

## Reactions

Every other term is a **reaction**: added to the field cell by cell, as written. Births
(`g*B*(1 - N/K)`), sources (`src`), interactions with other fields and compartments
(`-k*B*P` is a loss, but `k*B*P` gained by another field is a reaction there), and losses
whose rate depends on the field itself (`-k*B*B`, crowding). Reactions have no derivative
along an axis, so they never move material between cells; to move material, use transport,
diffusion or a transition.

## Compartment equations

A compartment has one equation, `d_t(P) = …`, with the whole right-hand side; in Python,
one `m.ode(P, rhs)` call. It may read fields through totals (`int_a(h*I)`), a field at an
end (`G(z=1)`), slices and other compartments. `d_` operators are refused there, since a
compartment has no axes.

When the same process changes several places, it is written in each equation, as in a paper:
infection `k*B*P` leaves `B` and, at age 0, enters `I`; adsorption uses up phage. `explain()`
matches the terms across equations and points out a place where one is missing. See
[Reading a model back](explain.md).

## Conditionals

A rate that switches, a drug effect above a threshold, a different rate by phase: a formula
can choose between values. The first true case wins and there is always a final value:

=== "Notation"

    ```tq
    model kill_switch
    param MIC=1, k1=2, k2=0.5, k3=0, g=0.8, ke=0.2
    k = k1   if C > MIC
        k2   if C > MIC/2
        k3   otherwise
    d_t(C) = -ke*C
    d_t(B) = g*B - k*B
    C(0) = 8
    B(0) = 1e6
    ```

=== "Python"

    ```python
    m = tq.Model("kill_switch")
    MIC, k1, k2, k3, g, ke = m.params(MIC=1, k1=2, k2=0.5, k3=0, g=0.8, ke=0.2)
    C = m.compartment("C")
    B = m.compartment("B")
    k = m.define("k", tq.cases((C > MIC, k1), (C > MIC / 2, k2), otherwise=k3))
    m.ode(C, -ke * C)
    m.ode(B, g * B - k * B)
    m.init(C, 8)
    m.init(B, 1e6)
    ```

The notation has three spellings of the same conditional, which give the same model:

```tq
k = k1 if C > MIC else k2 if C > MIC/2 else k3      # in one line

k = k1   if C > MIC                                 # the cases brace of a paper
    k2   if C > MIC/2
    k3   otherwise

if C > MIC:                                         # a block, as in NONMEM or Monolix
    k = k1
elif C > MIC/2:
    k = k2
else:
    k = k3
```

In a block every branch must set the same names; each name becomes one conditional formula.
A block only names formulas: equations (`d_t`) are never inside one. A condition is not a
number: `k*(C > MIC)` is refused; write `k if C > MIC else 0`.

The solver handles the switch: when the condition depends on time and parameters it restarts
exactly at the switch time, as at a dose; when it depends on a compartment or a total (here
`C`), it stops exactly at the crossing and restarts. The formula language
(functions, conditions, `and`/`or`/`not`) is the same everywhere in a model.

## What is refused

TenQSP refuses a term it cannot read safely, with the line number and a correction. The
common cases:

| Written | Message (shortened) | Write instead |
|---|---|---|
| `d_t(I) = -v*d_a(I)` | the transport term must be in conservation form | `-d_a(v*I)` |
| `d_t(I) = d_a(v*I)` | transport is written with a minus; to move toward lower `a`, give a negative velocity | `-d_a(-v*I)` |
| `d_t(F) = -d_x(D*d_x(F))` | diffusion is written `+d_x(D*d_x(F))`; a negative diffusion has no solution | `+d_x(D*d_x(F))` |
| `-d_a(v*J)` in the equation of `I` | in the equation of `I`, transport moves `I` | `-d_a(v*I)` |
| `d_x(d_x(F))` | write diffusion as `d_x(D*d_x(F))`, with the diffusivity `D` | `d_x(D*d_x(F))` |
| `d_a(I)` in a compartment's equation | a compartment has no axis `a`; to use a field's total, integrate it | `int_a(…)` |
| `d_t(F)` given twice | write one equation per field or slice, with all its terms | one equation |
| `h(2)` | `h` is a named formula, not a function | `h(a)` or `h` |
| `compartment P = 1e7` | a declaration has no value | `P(0) = 1e7` |
| `-h*(x > 0.5)*F` | a condition is not a number | `-(h if x > 0.5 else 0)*F` |

A full message points at the term:

```python
try:
    tq.Model.from_text("""
model bad
axis a in [0, 2] n=10 kind=one_way
param v=1, h=0.2
field I(a)
d_t(I) = -v*d_a(I) - h*I
    at a=2: outflow
""")
except Exception as e:
    print(e)
```
