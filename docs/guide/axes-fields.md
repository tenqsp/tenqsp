# Axes, fields and compartments

A TenQSP model is built from a few kinds of parts: **axes** that say what a population is
structured by (age, maturation, depth, genotype, organ), **fields** that hold a quantity
spread over those axes, ordinary **compartments**, **parameters**, **named formulas** and
**observables**. This chapter takes each in turn, with the notation and the Python API side
by side, and ends with model files.

The equations that move things between these parts are the subject of the next chapters:
[Reactions, transport and diffusion](terms.md) and [Categories and transitions](transitions.md).

## The parts at a glance

| Part | Notation | Python | Example |
|---|---|---|---|
| continuous axis | `axis a in [0, 10] n=20` | `m.axis("a", 0, 10, 20)` | age, maturation, depth, dormancy level |
| categorical axis | `axis s categorical [S, R]` | `m.axis("s", labels=["S", "R"])` | genotype, phenotype, organ, tissue space |
| field | `field N(a)` | `m.field("N", axes=[a])` | cells by age, drug by organ |
| compartment | `compartment P` (or just its equation) | `m.compartment("P")` | free phage, drug in plasma |
| parameter | `param ke=0.2` | `m.params(ke=0.2)` | rates, volumes, potencies |
| named formula | `h(a) = h_max*sigmoid((a - tau)/s)` | `m.define("h", ...)` | a hazard, a division rate |
| observable | `observable conc = centr/V` | `m.observable("conc", centr/V)` | what you compare with data |

## Continuous axes

A continuous axis is a range of values cut into cells. `axis a in [0, 10] n=20` is the
interval from 0 to 10 cut into 20 cells of width 0.5. A field over `a` then holds one value
per cell. More cells give a finer description; the [numerical choices](numerics.md) chapter
says how many are enough.

Each continuous axis has a **kind**, which tells TenQSP how material may move along it:

| kind | Meaning | What TenQSP checks |
|---|---|---|
| `bidirectional` (the default) | material may move either way (a drift whose sign changes, diffusion) | nothing extra |
| `one_way` | material only moves toward larger values: age, time since infection, maturation, progress through the gut | the velocity must stay ≥ 0; the upper end must be stated (usually `outflow`); a warning if nothing enters at the lower end (a note if the field is filled by doses or a starting value) |
| `spatial` | a position: depth in a biofilm, distance in tissue | treated like `bidirectional`; the kind documents what the axis is |

`units=` records the units of the coordinate (`units=h`, `units=mm`); they are kept with
the model (`m.to_text()` writes them back) and in the solution's coordinates,
`sol.coords["a"]["units"]`. TenQSP does not convert or check units.

A model of maturing cells shows a `one_way` axis in use. Cells enter at age 0 at rate
`kin`, age at speed `v`, die at rate `kd`, and leave when they reach age 10:

=== "Notation"

    ```tq
    model maturation
    axis a in [0, 10] n=20 units=d kind=one_way   # age, days
    param kin=100, v=1
    param kd = 0.05 units=1/d
    field N(a) units=cells
    d_t(N) = -d_a(v*N) - kd*N
        at a=0: v*N = kin
        at a=10: outflow
    observable total = int_a(N)
    ```

=== "Python"

    ```python
    import numpy as np
    import tenqsp as tq

    m = tq.Model("maturation")
    a = m.axis("a", 0, 10, 20, units="d", kind="one_way")
    kin, v = m.params(kin=100, v=1)
    kd = m.param("kd", 0.05, units="1/d")
    N = m.field("N", axes=[a], units="cells")
    m.advect(N, axis=a, velocity=v)
    m.hazard(N, kd)
    m.at(N, a=0, flux=kin)
    m.at(N, a=10, outflow=True)
    m.observable("total", m.integral(N, over=a))
    ```

The two lines under the equation are its [boundary conditions](boundaries.md): what enters
at age 0 and what happens at age 10.

**Cells of unequal width.** Two other ways to cut an axis:

- `faces=` lists the cell boundaries yourself, e.g. fine cells near a surface:
  `axis x faces=[0, 0.1, 0.3, 0.6, 1]` (Python: `m.axis("x", faces=[0, 0.1, 0.3, 0.6, 1])`).
- `spacing=geometric ratio=1.2` makes each cell 1.2 times wider than the one before, so the
  cells are finest at the lower end.

```python
g = tq.Model("grids")
x = g.axis("x", faces=[0, 0.1, 0.3, 0.6, 1])
r = g.axis("r", 0, 1, 10, spacing="geometric", ratio=1.2)
print(x.widths)                       # [0.1 0.2 0.3 0.4]
print(r.widths[1] / r.widths[0])      # 1.2 (up to rounding)
print(r.centers[:3])                  # the cell centres, finest near 0
```

In notation the geometric axis is `axis r in [0, 1] n=10 spacing=geometric ratio=1.2`.
`m.to_text()` writes an axis with uneven cells as its list of faces.

## Categorical axes

A categorical axis is a list of named categories: `axis s categorical [S, R]`. There is no
width and no order; a field over `s` has one value per category. Use one for anything that
comes in kinds: genotypes (susceptible and resistant), phenotypes (active and dormant),
organs, tissue spaces (vascular, interstitial, cellular). A label with spaces is written in
quotes, `["wild type", R]`.

Categories become useful with parameters that differ between them and with
[transitions](transitions.md) between them. A sensitive and a resistant strain under a drug
whose level `C0` is not fixed in the model:

=== "Notation"

    ```tq
    model two_strains
    axis s categorical [S, R]
    param g[s] = {S: 1, R: 0.8}           # growth, 1/h
    param kill[s] = {S: 2, R: 0.1}        # kill per unit of drug
    param K = 1e9
    param C0                              # drug level: given when solving
    field B(s) units=CFU/mL
    N = sum_s(B)
    d_t(B) = g[s]*B*(1 - N/K) - kill[s]*C0*B
    B[s=S](0) = 1e6
    B[s=R](0) = 1e2
    observable cfu = N
    ```

=== "Python"

    ```python
    m = tq.Model("two_strains")
    s = m.axis("s", labels=["S", "R"])
    g = m.param("g", {"S": 1, "R": 0.8}, over=s)
    kill = m.param("kill", {"S": 2, "R": 0.1}, over=s)
    K = m.param("K", 1e9)
    C0 = m.params("C0")
    B = m.field("B", axes=[s], units="CFU/mL")
    N = m.integral(B, over=s, name="N")
    m.reaction(B, g[s] * B * (1 - N / K))
    m.hazard(B, kill[s] * C0)
    m.init(B, 1e6, where={s: "S"})
    m.init(B, 1e2, where={s: "R"})
    m.observable("cfu", N)
    ```

`g[s]` is the growth rate of each category; in the equation of `B` it takes the value of the
category being updated. `sum_s(B)` adds up the categories (see
[Totals, integrals and ends](integrals.md)).

`C0` is declared without a value, so it must be given when solving. Leaving it out is an
error that names it, never a guess:

```python
sol = m.solve((0, 10), times=[0, 5, 10], params={"C0": 1})
print(sol["cfu"])                     # the sensitive strain is killed, the resistant one grows
try:
    m.solve((0, 10))
except Exception as e:
    print(e)                          # No value for C0: declared without a value ...
```

## Fields: densities and amounts

`field N(a)` declares a quantity spread over the axes in brackets. A field may have several
axes, continuous and categorical mixed: `field B(z, s)` is bacteria by dormancy level and
genotype, `field C(organ, sub)` is drug by organ and tissue space.

The value of a field in a cell is a **density**: an amount per unit of the continuous axes
(cells per day of age, cells per mm of depth). The **amount** in a cell is its value times
the cell's width. This is what makes the result independent of the grid: refine the grid and
the values stay comparable, while each cell holds less.

```python
d = tq.Model.from_text("""
model density
axis z in [0, 2] n=4
field B(z)
d_t(B) = -0.1*B
B(0) = 100
""")
sol = d.solve((0, 1), times=[0, 1])
width = np.diff(sol.coords["z"]["faces"])
print(sol["B"][0])                    # [100. 100. 100. 100.]: the density in each cell
print(width)                          # [0.5 0.5 0.5 0.5]
print((sol["B"][0] * width).sum())    # 200.0: the amount, density x width, added up
```

`B(0) = 100` is a density of 100 per unit of `z` on an axis 2 long, so the field holds 200 in
all. The ledger (`sol.ledger`) and every total (`int_z(B)`) count amounts. On a categorical
axis a category has no width: the value is the amount (or, with weights below, a
concentration). [Starting values](initial.md) shows how to give an amount and a shape instead
of a density.

**A transit chain with cells of width 1.** If an axis runs from 0 to N in N cells, each cell
has width 1, so density and amount are the same number and each cell is one transit
compartment. This is the recommended way to write a NONMEM-style transit chain
([Tutorial 3](../tutorials/03_myelosuppression.ipynb)).

### Fields with weights: concentrations

In a PBPK model a field over organs and tissue spaces holds a **concentration**, and each
category has a volume. `weights=` gives each category that size, as width does on a
continuous axis:

=== "Notation"

    ```tq
    model two_organs
    axis organ categorical [liver, muscle]
    param V[organ] = {liver: 1.5, muscle: 25}      # L
    param kel[organ] = {liver: 0.5, muscle: 0.1}   # 1/h
    field C(organ) weights=V[organ] units=mg/L
    d_t(C) = -kel[organ]*C
    C(0) = 1
    observable amount = sum_organ(C)
    ```

=== "Python"

    ```python
    m = tq.Model("two_organs")
    organ = m.axis("organ", labels=["liver", "muscle"])
    V = m.param("V", {"liver": 1.5, "muscle": 25}, over=organ)
    kel = m.param("kel", {"liver": 0.5, "muscle": 0.1}, over=organ)
    C = m.field("C", axes=[organ], weights=V[organ], units="mg/L")
    m.hazard(C, kel[organ])
    m.init(C, 1)
    m.observable("amount", m.integral(C, over=organ))
    ```

```python
sol = m.solve((0, 1), times=[0, 1])
print(sol["C"][0])                    # [1. 1.]: concentrations
print(sol["amount"][0])               # 26.5 = 1 x 1.5 + 1 x 25, an amount
```

With weights, `sum_organ(C)` adds up amounts (concentration × volume), and rates between
categories are clearances or flows (volume per time): see the PBPK example in
[Categories and transitions](transitions.md#pbpk-exchange-between-tissue-spaces).

## Compartments

A compartment is a single well-mixed quantity: free phage, drug in plasma, a nutrient. It
needs no declaration: its equation `d_t(P) = …` declares it. Declare it only to give it
units or a volume:

=== "Notation"

    ```tq
    model one_cpt
    param V=30, CL=4
    compartment Cc volume=V units=mg/L
    d_t(Cc) = -CL/V*Cc
    dose Cc at t=0: 300
    observable conc = Cc
    ```

=== "Python"

    ```python
    m = tq.Model("one_cpt")
    V, CL = m.params(V=30, CL=4)
    Cc = m.compartment("Cc", volume=V, units="mg/L")
    m.ode(Cc, -CL / V * Cc)
    m.dose(Cc, times=[0], amounts=300)
    m.observable("conc", Cc)
    ```

With `volume=`, the compartment holds a concentration: a dose of 300 (an amount) raises it by
300/V, and the ledger counts amounts (value × volume).

```python
sol = m.solve((0, 5), times=[0, 5])
print(sol["conc"])                    # [10.  5.13]: 300/30 at the start
print(sol.ledger.table("Cc"))         # the dose of 300 and the amount cleared
```

A declaration never carries a value: `compartment P = 1e7` is refused, because it could be
read as "fixed at 1e7". The value at time 0 is written `P(0) = 1e7`.

## Parameters

| What | Notation | Python |
|---|---|---|
| several values | `param ka=1.2, CL=4, V=30` | `ka, CL, V = m.params(ka=1.2, CL=4, V=30)` |
| with units (one per line) | `param kd = 0.05 units=1/d` | `m.param("kd", 0.05, units="1/d")` |
| declared without a value | `param ka, CL` | `ka, CL = m.params("ka", "CL")` |
| one value per category | `param k[s] = {S: 1e-9, R: 1e-12}` | `m.param("k", {"S": 1e-9, "R": 1e-12}, over=s)` |
| a table over two axes | `param V[organ, sub] = {liver: {vascular: 0.09, …}, …}` | `m.param("V", {...}, over=[organ, sub])` |
| a table declared without values | `param k[s]` | `m.param("k", over=s)` |

A value written in the model is a **default**. Values given when solving or simulating,
`m.solve(..., params={"ka": 1.5})`, replace it. A parameter declared without a value must be
given there; a missing value is an error that names it.

**Tables.** A parameter over several categorical axes is written in nested braces, one level
per axis, in the order of the axes in the brackets:

```tq
param V[organ, sub] = {
    liver:  {vascular: 0.09, interstitial: 0.27, cellular: 1.44},
    muscle: {vascular: 1.45, interstitial: 4.35, cellular: 23.2},
}
```

Labels may come in any order; a missing or unknown label is an error that names it. In a
formula the table is used with its axes, `V[organ, sub]`; inside the equation of a field over
those axes it takes each cell's value.

**Tables kept in files.** `tq.table` turns a long table (one row per value: a column per
axis and a value column) into these nested braces. Values given this way replace the ones
in the model, or fill a table declared without values:

```python
import pandas as pd

df = pd.DataFrame({"organ": ["liver", "muscle"], "kel": [0.5, 0.1]})
kel_table = tq.table(df, axes=["organ"], value="kel")
print(kel_table)                      # {'liver': 0.5, 'muscle': 0.1}

tissues = tq.Model.from_text("""
model tissues
axis organ categorical [liver, muscle]
param V[organ] = {liver: 1.5, muscle: 25}
param kel[organ]                      # declared without values
field C(organ) weights=V[organ]
d_t(C) = -kel[organ]*C
C(0) = 1
observable amount = sum_organ(C)
""")
sol = tissues.solve((0, 1), times=[0, 1], params={"kel": kel_table})
print(sol["amount"])
```

With two axes, `tq.table(pd.read_csv("volumes.csv"), axes=["organ", "sub"], value="V")`
gives the nested form `{"liver": {"vascular": ...}, ...}`. A row given twice is an error.

## Named formulas

A formula used more than once, or one that deserves a name, is named in its own line:

```tq
h(a) = h_max*sigmoid((a - tau)/s)      # lysis hazard, rises near age tau
div  = g*B*(1 - N/K)                   # division rate
```

The brackets in `h(a)` document what the formula varies over; they are not function
arguments. Use it as `h(a)` or `h`; `h(2)` is refused. In Python, `m.define` names a formula
and returns a handle to use in others:

```python
lm = tq.Model("lysis")
age = lm.axis("a", 0, 2, 40, units="h", kind="one_way")
h_max, tau, s_h = lm.params(h_max=5, tau=0.8, s_h=0.05)
h = lm.define("h", h_max * tq.sigmoid((age - tau) / s_h))
print(h)                              # h: the stored equations stay short
```

Named formulas are formulas: they are evaluated for every cell at once and never hold
statements. A formula may choose between values with a condition; see
[Reactions, transport and diffusion](terms.md#conditionals).

## Observables

`observable name = formula` names an output: what you plot or compare with data. It is any
formula of the model's quantities: a compartment, a concentration `centr/V`, a total
`int_a(N)`, a slice `C[organ=liver, sub=cellular]`. Each observable appears in the solution
under its name, `sol["conc"]`.

## Model files

A model written in notation can live in a plain text file with the extension `.tq`, read with
`tq.load`. Two options make one file serve many runs:

- `cells={"a": 80}` changes the number of cells of a continuous axis, so a convergence study
  needs one file;
- `params={...}` replaces default values (a table from `tq.table` included).

```python
with open("maturation.tq", "w") as f:
    f.write("""
model maturation
axis a in [0, 10] n=20 units=d kind=one_way
param kin=100, v=1
param kd = 0.05 units=1/d
field N(a) units=cells
d_t(N) = -d_a(v*N) - kd*N
    at a=0: v*N = kin
    at a=10: outflow
observable total = int_a(N)
""")

for n in (20, 80):
    mm = tq.load("maturation.tq", cells={"a": n}, params={"kd": 0.1})
    sol = mm.solve((0, 50), times=[50])
    print(n, sol["total"])            # about 623 and 630
```

Both are close to the exact 632 (kin/kd × (1 − e^−1)). The small difference is not an error
of the solver: by default each cell of a transported axis is one transit compartment, so the
grid sets how much the ages spread (see [the transport schemes](terms.md#transport-schemes-erlang-or-sharp)).

The model is named by its `model` line, or else by the file name. A model read from a file
is an ordinary `tq.Model`: Python calls may add to it, and `m.to_text()` writes any model,
also one built in Python, back as notation.
