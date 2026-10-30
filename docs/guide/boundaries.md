# Boundary conditions

A continuous axis has two ends, and every field that lives on it needs to know what happens
there. Does drug enter a biofilm at its surface? Do newly infected cells enter at infection
age 0? Do cells that reach the end of a maturation axis leave, or pile up? This chapter shows
how to say it, the way a paper does: one line per end, right under the field's equation.

Categorical axes (genotypes, organs) have no ends; only continuous axes do.

## What an end is

An axis `z in [0, 1]` has a **lower end** at `z = 0` and an **upper end** at `z = 1`. A
field over `z` can gain or lose material only there, apart from its reactions and losses.
What crosses an end is a **flux**: an amount per unit time, moving along the axis.

TenQSP uses one sign convention, the one of a paper: **a flux is positive toward larger
coordinates**. At the lower end a positive flux enters the field; at the upper end a
positive flux leaves it.

There are five things an end can do:

| At the end | Written under the equation | Python |
|---|---|---|
| a given flux crosses | `at z=0: -Dz*d_z(D) = alpha*B - beta*D` | `m.at(D, z=0, flux=alpha*B - beta*D)` |
| cells enter at age 0 (renewal) | `at a=0: I = r` (speed 1) or `at a=0: v*I = r` | `m.at(I, a=0, value=r)`, `m.at(I, a=0, flux=r)` |
| the field is held at a value | `at z=1: C = 1` | `m.at(C, z=1, value=1)` |
| nothing crosses (the default) | `at z=0: d_z(C) = 0`, or nothing at all | `m.at(C, z=0, flux=0)`, or nothing |
| what the flow brings leaves | `at a=2: outflow` | `m.at(I, a=2, outflow=True)` |

An end is named by its coordinate (`z=0`, `a=2`) or as `z=lower`, `z=upper`; in Python,
`z=0` or `z="lower"`. A coordinate that is not an end of the axis is refused.

## A held value: drug penetrating a biofilm

A biofilm of depth 1 sits on a sealed surface at `z = 0`; its top, `z = 1`, touches liquid
that holds the drug at a fixed level 1. Inside, the drug diffuses and is used up at rate
`kr`. Two conditions: nothing crosses the sealed side (the default, so nothing to write),
and the drug is held at 1 at the top.

=== "Notation"

    ```tq
    model penetration
    axis z in [0, 1] n=40                  # depth: sealed at 0, liquid at 1
    param De=0.01, kr=1
    field C(z)                             # drug in the biofilm
    d_t(C) = d_z(De*d_z(C)) - kr*C
        at z=1: C = 1                      # held by the liquid
    ```

=== "Python"

    ```python
    import numpy as np
    import tenqsp as tq

    m = tq.Model("penetration")
    z = m.axis("z", 0, 1, 40)
    De, kr = m.params(De=0.01, kr=1)
    C = m.field("C", axes=[z])
    m.diffuse(C, axis=z, D=De)
    m.hazard(C, kr)
    m.at(C, z=1, value=1)
    ```

After a while the profile stops changing. That steady profile is known in closed form: with
the penetration depth \( \lambda = \sqrt{D_e / k_r} \),

$$ C(z) = \frac{\cosh(z/\lambda)}{\cosh(1/\lambda)} . $$

Here \( \lambda = 0.1 \): the drug reaches only about a tenth of the biofilm.

```python
sol = m.solve((0, 20), times=[0, 20])
zc = sol.coords["z"]["values"]                     # cell centres
lam = np.sqrt(0.01 / 1)
exact = np.cosh(zc / lam) / np.cosh(1 / lam)
print(f"largest difference: {np.max(np.abs(sol['C'][-1] - exact)):.4f}")   # 0.0065
```

The largest difference, 0.0065, is at the top, where the profile is steepest. It falls about
fourfold each time the cells are halved (0.0018 with 80 cells, 0.00047 with 160): the
second-order accuracy of the diffusion scheme. To hold the drug at 1, the liquid supplies
whatever the biofilm consumes; the ledger shows how much:

```python
print(sol.ledger.table("C"))       # "boundary z=upper: dirichlet" is what entered at the top
```

`m.explain()` lists every end, the defaults included:

```python
e = m.explain()
print(e[e.index("Boundaries"):e.index("Starting values")])
```

```text
Boundaries
  C at z=lower (z = 0): sealed, nothing crosses (the default).
  C at z=upper (z = 1): held at 1.
```

## A flux condition

When what crosses an end depends on the state, write the flux:

```tq
at z=0: <the flux of the field along z> = <formula>
```

- **The left side** is the flux as the field's equation transports it: `v*I` for a flow at
  velocity `v`, `-Dz*d_z(D)` for diffusion with diffusivity `Dz`, `v*D - Dz*d_z(D)` for
  both. TenQSP checks it against the equation and, if it differs, says what it should be:

    ```text
    line 7: at z=0: -2*Dz*d_z(D) = k
    the left side is the flux of D along z, from its equation: -Dz*d_z(D) (not -2*Dz*d_z(D)).
    ```

- **On the right**, the field's own name means its value at that end. The formula must be
  **linear** in that value: each term proportional to it or free of it, such as an
  exchange `h*(C - Cout)`. `k*D^2` at the end is refused, with a message that says why.
  Other fields are read at the end the same way, `I(z=0)`.
- **The sign** follows the convention above: at the lower end, positive means entering.

Example: bacteria `B` (a compartment) fall dormant at rate `alpha`. Dormant cells `D` are
spread over a depth of dormancy `z`; they enter at `z = 0`, drift deeper by diffusion, and
wake up at rate `beta` from the shallowest level. One condition at `z = 0` says both:

=== "Notation"

    ```tq
    model dormancy
    axis z in [0, 1] n=20                     # depth of dormancy
    param alpha=0.2, beta=0.5, Dz=0.01
    field D(z)                                # dormant cells
    d_t(D) = d_z(Dz*d_z(D))
        at z=0: -Dz*d_z(D) = alpha*B - beta*D # in by falling dormant, out by waking
    d_t(B) = -alpha*B + beta*D(z=0)           # B gains what leaves D
    B(0) = 1e6
    ```

=== "Python"

    ```python
    m = tq.Model("dormancy")
    z = m.axis("z", 0, 1, 20)
    alpha, beta, Dz = m.params(alpha=0.2, beta=0.5, Dz=0.01)
    D = m.field("D", axes=[z])
    B = m.compartment("B")
    m.diffuse(D, axis=z, D=Dz)
    m.at(D, z=0, flux=alpha*B - beta*D)
    m.ode(B, -alpha*B + beta*D(z=0))
    m.init(B, 1e6)
    ```

The equation of `B` reads `D(z=0)`: the value of `D` at that end, the same value the
condition uses (see [Totals, integrals and ends](integrals.md)). So what leaves `D` is exactly
what `B` gains, and the total number of cells stays 1e6:

```python
sol = m.solve((0, 48), times=np.linspace(0, 48, 97))
width = np.diff(sol.coords["z"]["faces"])
total = sol["B"] + sol["D"] @ width            # amount in D = density x cell width
print(f"total from {total.min():.6g} to {total.max():.6g}")
assert np.allclose(total, 1e6, rtol=1e-12)
```

`m.explain()` matches the terms written in two places ("Terms shared between equations"):
`B*alpha` leaves `B` and enters `D` at `z = 0`; `D(z=lower)*beta` does the reverse.

!!! note "A flow that leaves through an end with a flux condition"
    Where the flow leaves through an end, it carries out the value of the edge cell
    (upwinding, the rule that a flow carries what is upstream of it). There the condition
    sets the value read at the end (and, with diffusion, the slope), not how much the flow
    carries out. For a flow that only leaves, write `outflow` instead.

## Cells entering at age 0: renewal

Along an **age** axis (time since infection, since division, since entering a maturation
pathway), cells do not diffuse in; they are *born* at age 0. The condition at `a = 0` says
how many arrive per unit time. This is the renewal condition of age-structured models
(McKendrick–von Foerster).

Along an age axis the speed is 1 (age grows as fast as time), so the flux at `a = 0` is the
value there, `1*I`, and the condition can be written either way:

- `at a=0: I = k*B*P` holds the value at age 0 at `k*B*P`; at speed 1 that *is* a flux of
  `k*B*P`. This is how a paper writes it, I(t, 0) = kBP.
- `at a=0: v*I = k*B*P` is the flux form, for any speed `v`. With speed 1, write `1*I`,
  because `I = …` alone is the held value of the previous line. In Python this is
  `m.at(I, a=0, flux=k*B*P)`.

The two give the same numbers at speed 1. At another speed only the flux form says "this
many per unit time".

A small example with a closed form. Cells enter at a constant rate `r = 2` per hour, die at
rate `h = 0.5`, and leave when they reach age 4 (the upper end). The axis is `one_way`:
material only moves toward larger ages.

=== "Notation"

    ```tq
    model renewal
    axis a in [0, 4] n=40 kind=one_way
    param r=2, h=0.5
    field I(a)                               # cells by age
    d_t(I) = -d_a(I) - h*I  using sharp
        at a=0: I = r                        # r cells per hour enter at age 0
        at a=4: outflow                      # the oldest leave
    d_t(L) = I(a=4)                          # ... and are counted here
    ```

=== "Python"

    ```python
    m = tq.Model("renewal")
    a = m.axis("a", 0, 4, 40, kind="one_way")
    r, h = m.params(r=2, h=0.5)
    I = m.field("I", axes=[a])
    L = m.compartment("L")
    m.advect(I, axis=a, velocity=1, scheme="sharp")
    m.hazard(I, h)
    m.at(I, a=0, value=r)
    m.at(I, a=4, outflow=True)
    m.ode(L, I(a=4))
    ```

At steady state the age profile is \( I(a) = r\,e^{-h a} \), so the number of cells is
\( (r/h)(1 - e^{-4h}) = 3.4587 \) and they leave at age 4 at the rate
\( r\,e^{-4h} = 0.2707 \) per hour.

```python
sol = m.solve((0, 30), times=[0, 30])
width = np.diff(sol.coords["a"]["faces"])
n_cells = sol["I"][-1] @ width
print(f"cells: {n_cells:.4f} (closed form {2 / 0.5 * (1 - np.exp(-2)):.4f})")
print(sol.ledger.table("I"))
print(sol.ledger.table("L"))
```

The ledger of `I` shows 60 entering at age 0 (2 per hour for 30 hours), what died, and what
left at age 4; the ledger of `L` shows it gained exactly what left.

## Outflow

`at a=4: outflow` says: whatever the flow carries to this end leaves the field. It is
recorded in the ledger (`boundary a=upper: outflow (absorbed)`), and another equation may
use it by reading the field there, as `L` does above with `I(a=4)` (or `I(a=upper)`): the
cells that finish maturing and enter the circulation, the phage released at the end of the
latent period, the drug that reaches the absorption site.

Along an axis declared `kind=one_way`, the end the flow reaches **must** be stated. TenQSP
refuses a model that leaves it unsaid, because the material would otherwise leave
unnoticed:

```text
line 5: field I(a)
the flow carries I to the upper end of 'a'; say what happens there: 'at a=<end>: outflow'
(it leaves the model, recorded in the ledger; read it elsewhere as I(a=upper)), or a condition.
```

## No flux

An end that is not mentioned is **sealed**: nothing crosses. This is the right choice for a
closed side (the sealed base of a biofilm, the top of a dormancy scale that cells cannot
pass). It can also be written out, `at z=0: d_z(C) = 0` or `at z=0: -De*d_z(C) = 0`, or
`m.at(C, z=0, flux=0)`. The only exception is the end a `one_way` flow reaches (above).

## Exchange with a compartment

A biofilm in a stirred liquid exchanges drug with it at its surface. The liquid is a
compartment `M` (an amount, in volume `V`); the exchange at the surface is
`h*(C - M/V)`: positive, so leaving the biofilm, when the biofilm is richer than the liquid.
It is written in both places, as in a paper: as the flux condition at the surface, and as a
term in the liquid's equation, with `C(z=1)` the value at the surface.

=== "Notation"

    ```tq
    model biofilm
    axis z in [0, 1] n=20 kind=spatial               # depth: sealed at 0, liquid at 1
    param De=0.1, kr=0.4, h=0.5, V=2, R=1, CL=0.3
    field C(z)                                       # drug in the biofilm
    compartment M                                    # amount in the liquid
    d_t(C) = d_z(De*d_z(C)) - kr*C
        at z=1: -De*d_z(C) = h*(C - M/V)             # leaves through the surface
    d_t(M) = R - CL*M/V + h*(C(z=1) - M/V)           # ... into the liquid
    ```

=== "Python"

    ```python
    m = tq.Model("biofilm")
    z = m.axis("z", 0, 1, 20, kind="spatial")
    De, kr, h, V, R, CL = m.params(De=0.1, kr=0.4, h=0.5, V=2, R=1, CL=0.3)
    C = m.field("C", axes=[z])
    M = m.compartment("M")
    m.diffuse(C, axis=z, D=De)
    m.hazard(C, kr)
    m.at(C, z=1, flux=h*(C - M/V))
    m.ode(M, R - CL*M/V + h*(C(z=1) - M/V))
    ```

Drug is infused into the liquid at rate `R`, cleared from it, and enters the biofilm, where
it is used up. What crosses the surface balances exactly:

```python
sol = m.solve((0, 24), times=np.linspace(0, 24, 49))
print(sol.ledger.table("C"))
print(sol.ledger.table("M"))
assert max(sol.ledger.check().values()) < 1e-12
```

```text
C: from t = 0 (start) to t = 24
  ...
  boundary z=upper: flux                               +6.38964
M: from t = 0 (start) to t = 24
  ...
  rate: h*(C(z=1.0) - M/V)                             -6.38964
```

The biofilm gained 6.39 through its surface (the flux was negative: toward smaller `z`,
into the biofilm), and the liquid lost the same 6.39.

!!! tip "Write the exchange in both places"
    If the term is missing from `M`'s equation, the drug that enters the biofilm comes
    from nowhere. `m.explain()` matches the term across the condition and the equation
    ("Terms shared between equations") and points out a place where it is missing.

## Older statements

Models written before conditions moved under their equation used separate statements.
They are still read, so older models keep working, and `m.to_text()` writes the form of
this chapter wherever it says the same thing.

| Older statement | The same, written under the equation | Older Python |
|---|---|---|
| `inflow I at a=lower: r` | `at a=0: 1*I = r` | `m.inflow(I, axis=a, at="lower", flux=r)` |
| `outflow I at a=upper` | `at a=4: outflow` | `m.outflow(I, axis=a, at="upper")` |
| `outflow I at a=upper to L` | `at a=4: outflow`, and `+ I(a=4)` in `L`'s equation | `m.outflow(..., to=L)` |
| `dirichlet C at z=upper: 1` | `at z=1: C = 1` | `m.boundary(C, ..., kind="dirichlet", value=1)` |
| `robin C at z=upper: h*(M/V - C) with M` | the exchange of the previous section | `m.boundary(..., kind="robin", value=M/V, h=h, to=M)` |
| `zero_flux C at z=lower` | nothing (the default) | `m.boundary(..., kind="zero_flux")` |

## See also

- [Totals, integrals and ends](integrals.md): reading a field at an end in any formula.
- [Tutorial 7](../tutorials/07_infection_age.ipynb): HIV-infected cells by infection
  age, with new infections entering at age 0.
- [Tutorial 10](../tutorials/10_biofilm_penetration.ipynb): drug penetration into a
  biofilm.
