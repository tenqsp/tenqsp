# Starting values

Every model needs a state at time 0: the inoculum of bacteria, the phage titre, the tumour's
age distribution, a drug level already in the body. This chapter shows how to give it. For a
compartment it is one number. For a field there is a choice that matters: is the number you
write a **density** (cells per unit of the axis) or an **amount** (cells in total) spread with
a given **shape**?

## One form: `X(0) = …`

In the notation a starting value is written as in a paper, `X(0) = value`; `(0)` always means
time 0. In Python it is `m.init(X, value)`.

=== "Notation"

    ```tq
    model start
    axis z in [0, 1] n=20                         # dormancy depth
    axis s categorical [S, R]                     # genotype
    field B(z, s)                                 # bacteria
    d_t(B) = 0.5*(1 - z)*B
    d_t(P) = -0.1*P                               # free phage
    B[s=S](0) = 1e7*normal(z, mean=0.1, sd=0.08)  # B[s=R] starts at 0: not given
    P(0) = 1e7
    ```

=== "Python"

    ```python
    import numpy as np
    import tenqsp as tq

    m = tq.Model("start")
    z = m.axis("z", 0, 1, 20)
    s = m.axis("s", labels=["S", "R"])
    B = m.field("B", axes=[z, s])
    P = m.compartment("P")
    m.reaction(B, 0.5*(1 - z)*B)
    m.ode(P, -0.1*P)
    m.init(B, total=1e7, shape=tq.normal(z, mean=0.1, sd=0.08), where={s: "S"})
    m.init(P, 1e7)
    ```

A declaration never carries a value (`compartment P = 1e7` is refused, with a message that
says to write `P(0) = 1e7`): a value written there could be read as "fixed at this value".
In Python, `m.compartment("P", init=1e7)` and `m.field("B", axes=[z], init=...)` are
shortcuts for `m.init`.

## Anything not given starts at 0

As in NONMEM and Monolix, a compartment, a field or the part of a field that is not given
starts at 0. It is not a warning. `m.explain()` lists what starts at 0, so a forgotten value
is visible:

```python
e = m.explain()
print(e[e.index("Starting values"):e.index("Numerical safeguards")])
```

```text
Starting values
  B[s=S]: an amount 1e7 spread like normal(z, mean=0.1, sd=0.08), scaled so the cells hold
    exactly that amount.
  P = 1e7
  Start at 0 (not given, or given as 0): the rest of B.
```

## A density, or an amount and a shape

**A value is a density, unless it contains a shape.**

- `B(0) = 1e7*exp(-z)` is a density: the formula *is* the value, cells per unit of `z`, at
  each depth. The total is whatever its integral is.
- `B(0) = 1e7*normal(z, mean=0.1, sd=0.08)` is an **amount** times a **shape**: the shape is
  integrated over each cell, cut to the axis and scaled so the cells hold exactly 1e7 in
  total, whatever the grid. A normal curve centred near 0 loses part of its tail below
  `z = 0`; that part is put back in, so the inoculum is still 1e7.
- `B(0) = 1e7*shape(exp(-z))` spreads an amount like **any formula** of the axes. Here: 1e7
  cells, distributed like an exponential cut to [0, 1].
- `B(0) = 1e7*density(normal(z, mean=0.1, sd=0.08))` uses a named curve **itself** as a
  density, not scaled to an amount. It differs from `1e7*normal(…)` only by the part of the
  curve the axis cuts off: here the part below `z = 0` is lost, not put back in (the total
  is 1e7 × Φ(0.1/0.08) = 8.94e6).

Side by side, on an axis `z` in [0, 1]:

| written | means | at z = 0 | at z = 1 | total on the axis |
|---|---|---|---|---|
| `B(0) = 1e7*exp(-z)` | the formula **is** the value (a density) | 1e7 | 1e7·e⁻¹ = 3.7e6 | ∫ = 6.3e6 |
| `B(0) = 1e7*shape(exp(-z))` | 1e7 **spread** like exp(−z) | 1.58e7 | 5.8e6 | exactly 1e7 |
| `B(0) = 1e7*normal(z, mean=0.1, sd=0.08)` | 1e7 spread like a normal (named shapes are always spread) | | | exactly 1e7 |
| `B(0) = 1e7*density(normal(z, mean=0.1, sd=0.08))` | the normal curve **is** the value | | | 1e7 × the part of the curve inside [0, 1] |

The numbers, computed on a fine grid (200 cells):

```python
for written in ["1e7*exp(-z)", "1e7*shape(exp(-z))", "1e7*normal(z, mean=0.1, sd=0.08)",
                "1e7*density(normal(z, mean=0.1, sd=0.08))"]:
    mz = tq.Model.from_text(f"""
model compare
axis z in [0, 1] n=200
field B(z)
d_t(B) = 0*B
B(0) = {written}
""")
    sol = mz.solve((0, 1), times=[0])
    b = sol["B"][0]                                   # the density in each cell at t = 0
    width = np.diff(sol.coords["z"]["faces"])
    print(f"{written:43s} first cell {b[0]:.3g}  last cell {b[-1]:.3g}  total {b @ width:.4g}")
```

```text
1e7*exp(-z)                                 first cell 9.98e+06  last cell 3.69e+06  total 6.321e+06
1e7*shape(exp(-z))                          first cell 1.58e+07  last cell 5.83e+06  total 1e+07
1e7*normal(z, mean=0.1, sd=0.08)            first cell 2.65e+07  last cell 0  total 1e+07
1e7*density(normal(z, mean=0.1, sd=0.08))   first cell 2.37e+07  last cell 0  total 8.944e+06
```

**Values are stored as cell averages.** "1e7 at z = 0" is the average of the curve over the
first cell: 9.98e6 here, with cells 0.005 wide; 9.5e6 with 10 cells. The total, though, is
exact on any grid.

When to use which: an inoculum is usually known as a count (CFU per mL, a phage titre), so
give an **amount and a shape**: the count is then exact, and refining the grid does not
change it. Write a **density** when the formula itself is what you know, for example a
steady age profile from an analysis.

### The shapes

The first argument is the axis; the rest are keywords (one of each pair):

| Shape | Keywords |
|---|---|
| `normal` | `mean` with `sd`, `var` or `cv` |
| `lognormal` | `meanlog`, `median` or `mean`, with `sdlog` or `cv` |
| `gamma` | `shape` with `rate` or `scale`; or `mean` with `sd`, `var` or `cv` |
| `exponential` | `rate` or `mean` |
| `beta` (on [0, 1]) | `alpha` and `beta`; or `mean` with `concentration`, `var` or `sd` |
| `uniform` | optional `lower`, `upper` (default: the whole axis) |
| `point` | `at`: everything in the cell containing it |
| `shape(formula)` | any formula of the axes, e.g. `shape(exp(-z))` |

Shapes on several axes multiply: `1e7*normal(x, mean=0.5, sd=0.1)*normal(y, mean=0.2,
sd=0.1)` puts 1e7 in total on the (x, y) plane.

In Python, the shape goes in `shape=` and the amount in `total=`. Named shapes are
`tq.normal`, `tq.gamma`, `tq.point`, …; any other formula of the axes is passed as it is,
`shape=tq.exp(-z)`, and becomes `shape(exp(-z))`.

### `point`: everything in one cell

`point(z, at=0)` puts the whole amount in the one cell that contains `z = 0`, the first.
The density there is the amount divided by that cell's width: the amount is exact, but it is
spread over one cell, so a finer grid makes the start sharper. `at=` the upper end uses the
last cell. This is how cells enter an age axis at age 0, or a dose of drug enters the gut at
the start of its transit ([A first model](../start/first-model.md)).

```python
m1 = tq.Model("one_cell")
a = m1.axis("a", 0, 2, 20)
I = m1.field("I", axes=[a])
m1.reaction(I, 0*I)
m1.init(I, total=1e6, shape=tq.point(a, at=0))
b = m1.solve((0, 1), times=[0])["I"][0]
print(np.flatnonzero(b), b[0])          # [0] 1e7: 1e6 cells in a cell 0.1 wide
```

A point exactly on a face between two cells goes in the cell above it (within rounding:
`at=0.3` on a grid of 10 cells on [0, 1] is on the face between the third and fourth cells,
and goes in the fourth).

## Parts of a field: slices

`B[s=S](0) = …` gives the starting value of one part of a field: the genotype `S`, over all
its other axes. Parts that are not given start at 0. Several parts may be given, each once:

```tq
B[s=S](0) = 9.9e6*normal(z, mean=0.1, sd=0.08)   # 99 % susceptible
B[s=R](0) = 1e5*normal(z, mean=0.1, sd=0.08)     # 1 % resistant
```

Giving the same part twice, or `B(0)` together with `B[s=S](0)`, is an error that names both
lines:

```text
line 9: B(0) = 1
the value at time 0 of this part of B is already given (line 7).
```

In Python, `where=` picks the part, and `split=` shares an amount (or a value) between
categories; the fractions must add up to 1. These two give the same model:

=== "Notation"

    ```tq
    model mixed
    axis z in [0, 1] n=20
    axis s categorical [S, R]
    field B(z, s)
    d_t(B) = 0.5*(1 - z)*B
    B[s=S](0) = 9.9e6*shape(exp(-z))
    B[s=R](0) = 1e5*shape(exp(-z))
    ```

=== "Python"

    ```python
    m = tq.Model("mixed")
    z = m.axis("z", 0, 1, 20)
    s = m.axis("s", labels=["S", "R"])
    B = m.field("B", axes=[z, s])
    m.reaction(B, 0.5*(1 - z)*B)
    m.init(B, total=1e7, shape=tq.exp(-z), split={s: {"S": 0.99, "R": 0.01}})
    ```

```python
sol = m.solve((0, 1), times=[0])
width = np.diff(sol.coords["z"]["faces"])
print(width @ sol["B"][0])               # [9.9e6, 1e5]: amount per genotype
```

## `m.init` and the notation

| Notation | Python |
|---|---|
| `P(0) = 1e7` | `m.init(P, 1e7)` |
| `B(0) = 1e7*exp(-z)` (a density) | `m.init(B, 1e7*tq.exp(-z))` |
| `B(0) = 1e7*normal(z, mean=0.1, sd=0.08)` | `m.init(B, total=1e7, shape=tq.normal(z, mean=0.1, sd=0.08))` |
| `B(0) = 1e7*shape(exp(-z))` | `m.init(B, total=1e7, shape=tq.exp(-z))` |
| `B(0) = 1e7*point(z, at=0)` | `m.init(B, total=1e7, shape=tq.point(z, at=0))` |
| `B(0) = 1e7*density(normal(z, …))` | `m.init(B, 1e7*tq.normal(z, …))` (a value: a shape in it is the curve itself) |
| `B[s=S](0) = …` | `m.init(B, ..., where={s: "S"})` |
| `B[s=S](0) = 0.99*…`, `B[s=R](0) = 0.01*…` | `m.init(B, ..., split={s: {"S": 0.99, "R": 0.01}})` |

`m.init` takes either a value or both `total=` and `shape=`, never a mix; anything else is
refused with an example of the right call.

## Starting from another state

The values above are the model's own. Two ways to start elsewhere without editing the model:

- **A pre-run.** Run the model until it settles (a steady state, or a fixed time) and start
  the next run from where it ended: the bacteria in stationary phase before treatment, a
  drug at its steady-state trough before a change of dose.
- **Values by name.** `tq.Scenario(..., initial={"P": 3e6})` replaces the starting values it
  names; the others keep the model's. A field takes an array, one value (a density) per cell.

Example: cells enter a maturation chain of five stages at rate `kin`, die at rate `kout`
on the way, and the mature ones collect in `M`. With cells of width 1, each cell is one
transit stage and its density is its amount.

```python
m = tq.Model.from_text("""
model maturation
axis a in [0, 5] n=5 kind=one_way
param kin=10, kout=0.5
field N(a)
d_t(N) = -d_a(N) - kout*N
    at a=0: 1*N = kin
    at a=5: outflow
d_t(M) = N(a=5) - kout*M
""")
sim = m.simulator(outputs=["M", "N"])
sc = tq.Scenario(name="from empty", times=np.linspace(0, 10, 11))
pre = sim.pre_run(sc, until="steady")
print(pre.reached, pre.state["N"])         # True, 10/1.5, 10/1.5**2, ...: each stage keeps 1/1.5
r = sim(sc.replace(initial=pre))           # starts at the steady state
print(r["M"][:3])                          # stays at 10/1.5**5/0.5 = 2.634

r2 = sim(tq.Scenario(name="given", times=np.linspace(0, 10, 11),
                     initial={"N": np.zeros(5), "M": 3.0}))
print(r2["M"][:3])                         # starts at 3, then falls
```

The [Simulation and exploration](simulation.md) chapter covers pre-runs in full: how a steady
state is detected under a repeating regimen, the time limit, and what is saved with the
scenario.

## See also

- [Doses and regimens](doses.md): adding material later, with the same amount × shape form.
- [Totals, integrals and ends](integrals.md): reading totals and parts of a field.
