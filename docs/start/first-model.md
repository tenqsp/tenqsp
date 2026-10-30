# A first model

Drug taken by mouth does not reach the absorption site at once: it travels down the gut
first. A pharmacometric model usually describes that delay with a chain of transit
compartments. Here the drug in transit is instead one quantity, `G`, spread along an axis
`z` that measures **progress through the gut**: 0 just swallowed, 1 at the absorption site.
It moves along `z` at speed 1/MTT, so it takes MTT on average to arrive.

## The model

A model can be written as text, the way equations look in a paper, or built step by step in
Python. Both give the same model; pick the tab you prefer (the choice applies to every page).

=== "Notation"

    ```tq
    model delayed_pk
    axis z in [0, 1] n=20 kind=one_way      # progress through the gut
    param MTT=1.5, ka=2, CL=4, V=30
    field G(z)                               # drug in transit, by progress
    d_t(G) = -d_z(G/MTT)  using sharp        # moves along z at speed 1/MTT
        at z=1: outflow
    d_t(depot) = G(z=1)/MTT - ka*depot
    d_t(centr) = ka*depot - CL/V*centr
    dose G at t=0 to 48 every 12: 100*point(z, at=0)
    observable conc = centr/V
    ```

=== "Python"

    ```python
    import numpy as np
    import tenqsp as tq

    m = tq.Model("delayed_pk")
    z = m.axis("z", 0, 1, 20, kind="one_way")
    MTT, ka, CL, V = m.params(MTT=1.5, ka=2, CL=4, V=30)
    G = m.field("G", axes=[z])
    depot = m.compartment("depot")
    centr = m.compartment("centr")
    m.advect(G, axis=z, velocity=1 / MTT, scheme="sharp")
    m.at(G, z=1, outflow=True)
    m.ode(depot, G(z=1) / MTT - ka * depot)
    m.ode(centr, ka * depot - CL / V * centr)
    m.dose(G, times=[0, 12, 24, 36, 48], amounts=100, shape=tq.point(z, at=0))
    m.observable("conc", centr / V)
    ```

Line by line:

- `axis z in [0, 1] n=20 kind=one_way` declares a continuous axis cut into 20 cells.
  `one_way` says material only moves toward larger `z`, as through the gut.
- `field G(z)` is a quantity spread over `z` (a density: amount per unit of `z`).
  `depot` and `centr` are ordinary compartments; they need no declaration.
- `d_t(G) = -d_z(G/MTT)` is transport along `z` at speed 1/MTT, written in conservation
  form as in a paper. `using sharp` keeps the front sharp (a delay of almost exactly MTT);
  without it each cell acts as one transit compartment (see the
  [user guide](../guide/terms.md)).
- `at z=1: outflow`, written under the equation of `G`, says what happens at that end: what
  arrives there leaves `G`. The flux there is speed times value, `G(z=1)/MTT`, and the depot
  gains exactly that.
- `dose G at t=0 to 48 every 12: …` gives 100 every 12 hours, all of it at `z = 0`.
- `observable conc = centr/V` names an output.

## Solve it

```python
m = tq.Model.from_text("""
model delayed_pk
axis z in [0, 1] n=20 kind=one_way
param MTT=1.5, ka=2, CL=4, V=30
field G(z)
d_t(G) = -d_z(G/MTT)  using sharp
    at z=1: outflow
d_t(depot) = G(z=1)/MTT - ka*depot
d_t(centr) = ka*depot - CL/V*centr
dose G at t=0 to 48 every 12: 100*point(z, at=0)
observable conc = centr/V
""")

t = np.linspace(0, 60, 241)
sol = m.solve((0, 60), times=t)

print(sol["conc"].shape)          # (241,): one value per output time
print(sol["G"].shape)             # (241, 20): a field, one value per cell
peak = np.argmax(sol["conc"])
print(f"peak {sol['conc'][peak]:.2f} at t = {t[peak]}")
```

`sol["G"]` holds the field at every output time and cell; `sol.coords["z"]["values"]` gives
the cell centres. With `pip install "tenqsp[plot]"`:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(sol.t, sol["conc"])
ax.set(xlabel="time (h)", ylabel="concentration")
tq.plot.snapshot(sol, "G", axis="z", times=[0.5, 1.0, 1.5])   # the drug moving down the gut
```

## Ask TenQSP what it built

`m.explain()` describes the model in words: what each term means, which numerical scheme
carries it, what happens at every end of every axis, what starts at zero, the doses, and the
complete list of numerical safeguards. It also matches terms shared between equations:
here `G(z=1)/MTT` leaves `G` and enters `depot`, and `ka*depot` leaves `depot` and enters
`centr`, so the amounts balance.

```python
print(m.explain())
```

Every solution carries a **ledger**: for each field and compartment, what was there at the
start and the end, and what each term, boundary and dose added or removed. "Unexplained" is
what the books fail to account for, which is rounding only.

```python
print(sol.ledger.table("centr"))
assert max(sol.ledger.check().values()) < 1e-10
```

`m.to_latex()` gives the equations for an article, and `m.to_text()` writes any model, also
one built in Python, as notation.

```python
print(m.to_latex())
print(m.to_text())
```

## Where next

- The [user guide](../guide/index.md) takes each part of a model in turn.
- [Tutorial 1](../tutorials/01_transit_compartments.ipynb) shows that this axis *is*
  a chain of transit compartments, and what the number of compartments really controls.
