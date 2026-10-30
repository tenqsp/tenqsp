# Totals, integrals and ends

A field holds a density: bacteria per unit of dormancy depth, infected cells per hour of
infection age, drug per litre of tissue. Most of what a model *does* with a field goes
through a total: crowding depends on the total number of bacteria, phage are released by all
lysing cells whatever their age, a PK output is the amount in the body. This chapter shows
how to write those totals, how to read a field at one end of an axis or in one category, and
how to make outputs from them.

## Integrals and sums

Inside any formula:

- `int_z(expr)` **integrates** over the continuous axis `z`: the sum over its cells of
  value × cell width.
- `sum_s(expr)` **sums** over the categorical axis `s` (genotypes, organs).
- They **nest**: `int_z(sum_s(B))` is the total over both axes.

The result varies over the axes that are left. If `B` is a field over `(z, s)`,
`int_z(sum_s(B))` is one number, `int_z(B)` has one value per genotype, and `sum_s(B)` is a
profile over `z`.

Example: two genotypes, susceptible `S` and resistant `R`, spread over a dormancy depth `z`
(cells deeper in dormancy grow more slowly; the resistant ones pay a fitness cost). All of
them compete for the same carrying capacity, through their total.

=== "Notation"

    ```tq
    model growth
    axis z in [0, 1] n=20                      # dormancy depth
    axis s categorical [S, R]                  # genotype
    param g=1, K=1e9
    param cost[s] = {S: 0, R: 0.2}
    field B(z, s)
    B_tot = int_z(sum_s(B))                    # all bacteria: one number
    d_t(B) = g*(1 - z)*(1 - cost[s])*B*(1 - B_tot/K)
    B[s=S](0) = 1e6*uniform(z)
    B[s=R](0) = 1e4*uniform(z)
    observable total = B_tot
    observable by_type = int_z(B)              # one value per genotype
    observable profile = sum_s(B)              # one value per cell of z
    ```

=== "Python"

    ```python
    import numpy as np
    import tenqsp as tq

    m = tq.Model("growth")
    z = m.axis("z", 0, 1, 20)
    s = m.axis("s", labels=["S", "R"])
    g, K = m.params(g=1, K=1e9)
    cost = m.param("cost", {"S": 0, "R": 0.2}, over=s)
    B = m.field("B", axes=[z, s])
    B_tot = m.integral(B, over=[z, s], name="B_tot")
    m.reaction(B, g*(1 - z)*(1 - cost[s])*B*(1 - B_tot/K))
    m.init(B, total=1e6, shape=tq.uniform(z), where={s: "S"})
    m.init(B, total=1e4, shape=tq.uniform(z), where={s: "R"})
    m.observable("total", B_tot)
    m.observable("by_type", m.integral(B, over=z))
    m.observable("profile", m.integral(B, over=s))
    ```

`B_tot = int_z(sum_s(B))` is a **named total**: it is computed once per evaluation and
used wherever its name appears. In Python, `m.integral(expr, over=[...], name=...)` makes
one; `over=` lists the axes, continuous or categorical alike. An integral written inline
without a name gets one made up (`int_z`, `int_s`), and the same integral written twice is
stored once. `m.to_text()` shows them:

```python
print(m.to_text())
```

```text
B_tot = int_z(sum_s(B))
int_z = int_z(B)
int_s = sum_s(B)
```

The integrand can be any formula of the field, not only the field itself: the lysis rate
summed over infection ages, `int_a(h(a)*I)`; the mean position of a population,
`int_x(x*B)/int_x(B)`; the bacteria that adsorb phage, `sum_s(k[s]*B)`.

## Outputs over totals, and outputs over axes

An observable may be one number at each time or vary over the axes its formula leaves
free. `sol.observable_axes` says which:

```python
sol = m.solve((0, 10), times=np.linspace(0, 10, 11))
print(sol.observable_axes)            # {'by_type': ('s',), 'profile': ('z',), 'total': ()}
print(sol["total"].shape)             # (11,)      one per output time
print(sol["by_type"].shape)           # (11, 2)    time x genotype
print(sol["profile"].shape)           # (11, 20)   time x cell of z
print(sol["by_type"][0])              # [1e6, 1e4]: the amounts given at time 0
```

The axes come in the field's order; `sol.coords["s"]["values"]` gives the labels and
`sol.coords["z"]["values"]` the cell centres.

## Weighted fields: totals are amounts

In a PBPK model a field holds **concentrations**, and each category has a volume. Declared
with `weights=`, the field knows it: `int_`/`sum_` then add up **amounts**, value × weight,
just as integrals over a continuous axis multiply by the cell width.

=== "Notation"

    ```tq
    model tissues
    axis organ categorical [gut, liver, muscle]
    param V[organ] = {gut: 1, liver: 2, muscle: 25}       # L
    param kel[organ] = {gut: 0, liver: 0.5, muscle: 0}    # elimination in the liver, 1/h
    field C(organ) weights=V[organ]                       # concentrations, mg/L
    d_t(C) = -kel[organ]*C
    C(0) = 1
    observable amount = sum_organ(C)                      # mg: sum of C x V
    ```

=== "Python"

    ```python
    m = tq.Model("tissues")
    organ = m.axis("organ", labels=["gut", "liver", "muscle"])
    V = m.param("V", {"gut": 1, "liver": 2, "muscle": 25}, over=organ)
    kel = m.param("kel", {"gut": 0, "liver": 0.5, "muscle": 0}, over=organ)
    C = m.field("C", axes=[organ], weights=V)
    m.hazard(C, kel[organ])
    m.init(C, 1)
    m.observable("amount", m.integral(C, over=organ))
    ```

Every organ starts at 1 mg/L, so the body holds 1 + 2 + 25 = 28 mg; only the liver's 2 mg
are eliminated:

```python
sol = m.solve((0, 4), times=[0, 1, 2, 4])
print(sol["amount"])                                  # 28, then 26 + 2 exp(-0.5 t)
assert np.allclose(sol["amount"], 26 + 2*np.exp(-0.5*sol.t), rtol=1e-5)
```

## One category in a formula: slices

A **slice** `C[organ=liver]` is the field with some categorical axes fixed. Read in a
formula, it is the field's **value** there, a concentration, not an amount: what a drug
effect in the liver should be driven by. With several axes, `C[organ=liver, sub=cellular]`
fixes both, and a slice that fixes only some axes still varies over the others.

```tq
model liver_effect
axis organ categorical [gut, liver, muscle]
param V[organ] = {gut: 1, liver: 2, muscle: 25}
param kel[organ] = {gut: 0, liver: 0.5, muscle: 0}
param Emax=1, EC50=0.5, kE=1
field C(organ) weights=V[organ]
d_t(C) = -kel[organ]*C
d_t(E) = kE*(Emax*C[organ=liver]/(EC50 + C[organ=liver]) - E)   # effect driven by the liver
C(0) = 1
observable C_liver = C[organ=liver]
```

!!! note
    Reading a slice inside a formula is part of the notation; the Python API has no
    equivalent yet (`C.at(organ="liver")` is accepted by `m.observable`, `m.integral`,
    `m.dose` and the other calls that take a part of a field, but not inside a formula).
    Build such a model from text, or add the equation to a text model:

```python
m = tq.Model.from_text("""
model liver_effect
axis organ categorical [gut, liver, muscle]
param V[organ] = {gut: 1, liver: 2, muscle: 25}
param kel[organ] = {gut: 0, liver: 0.5, muscle: 0}
param Emax=1, EC50=0.5, kE=1
field C(organ) weights=V[organ]
d_t(C) = -kel[organ]*C
d_t(E) = kE*(Emax*C[organ=liver]/(EC50 + C[organ=liver]) - E)
C(0) = 1
observable C_liver = C[organ=liver]
""")
sol = m.solve((0, 4), times=[0, 1, 2, 4])
print(sol["C_liver"])                                 # exp(-0.5 t): a concentration
assert np.allclose(sol["C_liver"], np.exp(-0.5*sol.t), rtol=1e-5)
```

`m.observable("C_liver", C.at(organ="liver"))` gives the same output in Python.

## A field read at an end

`D(z=0)` is the value of the field `D` at the lower end of `z`, the way a paper writes
D(t, 0); `I(a=upper)` is the value at the upper end of `a`. In Python the same is written
`D(z=0)`, `I(a="upper")`. It may appear in any formula, usually the equation of a
compartment fed by that end.

The value read is **the one the boundary condition at that end gives**:

| condition at that end | `F(z=end)` is |
|---|---|
| a flux condition, `at z=0: -Dz*d_z(D) = alpha*B - beta*D` | the value the condition sees at the end |
| a held value, `at z=1: C = 1` | the value held |
| `outflow` | the edge value the flow carries out |

This is why the books balance. With an outflow at age 4 and speed 1, what leaves the field
per unit time is speed × `I(a=4)`; a compartment whose equation adds `I(a=4)` gains exactly
that, not an approximation of it. Infected cells that finish their latent period and lyse,
cells that finish maturing and enter the circulation: the end value is the hand-over.

```python
m = tq.Model.from_text("""
model release
axis a in [0, 4] n=40 kind=one_way
param r=2, h=0.5
field I(a)
d_t(I) = -d_a(I) - h*I  using sharp
    at a=0: I = r
    at a=4: outflow
d_t(L) = I(a=upper)
""")
sol = m.solve((0, 30), times=[0, 30])
print(sol.ledger.table("I"))      # boundary a=upper: outflow (absorbed)   -7.04537
print(sol.ledger.table("L"))      # rate: I(a=upper)                       +7.04537
```

[Boundary conditions](boundaries.md) shows the same with an exchange at a flux condition,
where `D(z=0)` in the equation of `B` returns to `B` exactly what leaves `D`.

## Slopes: `grad_x(S)`, only inside a velocity

Bacteria that swim up a gradient of an attractant (chemotaxis) move at a velocity
proportional to its slope. `grad_x(S)` is the slope of the field `S` along `x` (also
written `grad(S, x)`; in Python `tq.grad(S, x)`):

=== "Notation"

    ```tq
    model chemotaxis
    axis x in [0, 1] n=40
    param chi=0.05, Db=0.002, DS=0.01
    field B(x)                                 # bacteria
    field S(x)                                 # attractant
    d_t(B) = -d_x(chi*grad_x(S)*B) + d_x(Db*d_x(B))  using sharp
    d_t(S) = d_x(DS*d_x(S))
    B(0) = 1
    S(0) = x                                   # more attractant to the right
    observable B_mean_x = int_x(x*B)/int_x(B)  # mean position of the bacteria
    ```

=== "Python"

    ```python
    m = tq.Model("chemotaxis")
    x = m.axis("x", 0, 1, 40)
    chi, Db, DS = m.params(chi=0.05, Db=0.002, DS=0.01)
    B = m.field("B", axes=[x])
    S = m.field("S", axes=[x])
    m.advect(B, axis=x, velocity=chi*tq.grad(S, x), scheme="sharp")
    m.diffuse(B, axis=x, D=Db)
    m.diffuse(S, axis=x, D=DS)
    m.init(B, 1)
    m.init(S, x)
    m.observable("B_mean_x", m.integral(x*B, over=x)/m.integral(B, over=x))
    ```

```python
sol = m.solve((0, 5), times=[0, 5])
print(sol["B_mean_x"])            # 0.5 at the start, then the bacteria move right
```

The slope is taken at the faces between cells, where velocities live, so `grad_x` is
accepted **only inside a velocity along `x`**. Anywhere else it is refused with a message
that says so.

## See also

- [Boundary conditions](boundaries.md): what happens at the ends that `F(z=0)` reads.
- [Starting values](initial.md): amounts and densities at time 0.
- [Tutorial 6](../tutorials/_build/06_whole_body_pbpk.ipynb): weighted fields, slices and
  fluxes in a whole-body PBPK model.
