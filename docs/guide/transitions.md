# Categories and transitions

Many populations come in kinds: susceptible and resistant bacteria, active and dormant cells,
the same drug in different organs and tissue spaces. A **categorical axis** lists the kinds,
and **transitions** move material between them: cells switching phenotype, daughters that
carry a mutation, drug crossing from the vascular to the interstitial space. This chapter
shows how to declare transitions, the two ways they act (switching and production), joint
transitions on two axes, slices, and PBPK exchange written the same way.

## Categorical axes

`axis s categorical [S, R]` declares the categories; a field over `s` has one value per
category (see [Axes, fields and compartments](axes-fields.md#categorical-axes)). Typical
uses:

| Axis | Categories | Transitions between them |
|---|---|---|
| genotype | `[S, R]`, or one axis per resistance locus | mutation at division |
| phenotype | `[active, dormant]` | switching, possibly drug-induced |
| organ | `[lung, liver, gut, …]` | blood flow (e.g. the portal vein) |
| tissue space | `[vascular, interstitial, cellular]` | permeability-limited exchange |

## Declaring transitions

A `transitions` statement names a set of moves between the categories of one axis, each with
its rate:

```tq
transitions M on s: S -> R = mu, R -> S = mu

transitions Q on s:                       # one move per line
    active -> dormant = a
    dormant -> active = b
```

In Python, `m.transitions` returns the set, to use in a transition term:

```python
import numpy as np
import tenqsp as tq

demo = tq.Model("demo")
s = demo.axis("s", labels=["S", "R"])
mu = demo.param("mu", 1e-6)
M = demo.transitions("M", axis=s, rates={"S->R": mu, "R->S": mu})
```

**Only the moves are declared.** What leaves the origin is derived from them (the "diagonal"
of the rate matrix), so a transition can neither create nor lose material: everything that
arrives in `R` left `S`. Declaring `S -> S` is refused for that reason. Written out, the
transition term in category `d` is

$$
\sum_{o \ne d} M_{d,o}\,X_o \;-\; \Big(\sum_{d' \ne d} M_{d',d}\Big)\,X_d ,
$$

what arrives from the other categories minus what leaves.

A rate may be any formula of parameters, coordinates of the field's other axes, fields,
compartments and totals: a switching rate that grows with depth, `a*(1 + 4*x)`, or one
induced by a drug, `a*C/(EC50 + C)`. It may not vary along the axis it moves between
(`a*B` on `s` is refused): the rate is that of leaving one category for another. To depend
on the whole population, use a total such as `sum_s(B)`.

## Switching: the cells themselves change category

`trans_s(Q, B)`, with the field itself as second argument, moves cells between categories at
rate `Q` per cell: phenotypic switching, persister formation, exchange between tissue spaces.
In Python: `m.transition(B, axis=s, matrix=Q, of=B)`.

Bacteria in a biofilm, by depth `x` and phenotype: active cells grow and are killed, dormant
cells do neither; cells enter dormancy faster deeper in the biofilm and wake up at a constant
rate:

=== "Notation"

    ```tq
    model persisters
    axis x in [0, 1] n=20 units=mm kind=spatial        # depth in a biofilm
    axis s categorical [active, dormant]
    param g=1, kill=3, a=0.05, b=0.01, D=0.001
    field B(x, s)
    transitions Q on s: active -> dormant = a*(1 + 4*x), dormant -> active = b
    d_t(B) = trans_s(Q, B) + d_x(D*d_x(B))
    d_t(B[s=active]) = g*B - kill*B
    B(0) = 1e6
    observable dormant = int_x(B[s=dormant])
    observable total = int_x(sum_s(B))
    ```

=== "Python"

    ```python
    m = tq.Model("persisters")
    x = m.axis("x", 0, 1, 20, units="mm", kind="spatial")
    s = m.axis("s", labels=["active", "dormant"])
    g, kill, a, b, D = m.params(g=1, kill=3, a=0.05, b=0.01, D=0.001)
    B = m.field("B", axes=[x, s])
    Q = m.transitions("Q", axis=s, rates={"active->dormant": a * (1 + 4 * x),
                                          "dormant->active": b})
    m.transition(B, axis=s, matrix=Q, of=B)
    m.diffuse(B, axis=x, D=D)
    m.reaction(B.at(s="active"), g * B)
    m.hazard(B.at(s="active"), kill)
    m.init(B, 1e6)
    m.observable("dormant", m.integral(B.at(s="dormant"), over=x))
    m.observable("total", m.integral(B, over=[x, s]))
    ```

```python
sol = m.solve((0, 10), times=[0, 5, 10])
print(sol["total"], sol["dormant"])   # the active cells are killed; the dormant ones remain
print(sol.ledger.table("B"))          # the switch line nets to 0: nothing created or lost
```

The second equation, `d_t(B[s=active]) = …`, is an **equation on a slice**: its terms apply
only where `s = active`, added to the field's own equation. In Python the same is done by
passing a slice, `B.at(s="active")`, to `m.reaction` or `m.hazard`. In a formula, a slice
`B[s=dormant]` reads the field there.
[Tutorial 9](../tutorials/09_persisters_and_dormancy.ipynb) builds this kind of model
from a two-state switch up to a continuous dormancy axis.

## Production: daughters that carry a mutation

`trans_s(M, div)`, with a **rate of production** as second argument, redistributes part of
that production between categories. With `div = g[s]*B` the division rate, a fraction `mu`
of the daughters of `S` cells are `R`. In Python:
`m.transition(B, axis=s, matrix=M, of=div)`. The production itself is added as usual (the
term `div`); the transition only moves the daughters.

=== "Notation"

    ```tq
    model mutation
    axis s categorical [S, R]
    param mu = 1e-6
    param g[s] = {S: 1, R: 0.5}            # resistance costs half the growth rate
    field B(s)
    div = g[s]*B
    transitions M on s: S -> R = mu
    d_t(B) = div + trans_s(M, div)
    B[s=S](0) = 1e6
    ```

=== "Python"

    ```python
    m = tq.Model("mutation")
    s = m.axis("s", labels=["S", "R"])
    mu = m.param("mu", 1e-6)
    g = m.param("g", {"S": 1, "R": 0.5}, over=s)
    B = m.field("B", axes=[s])
    div = m.define("div", g[s] * B)
    M = m.transitions("M", axis=s, rates={"S->R": mu})
    m.reaction(B, div)
    m.transition(B, axis=s, matrix=M, of=div)
    m.init(B, 1e6, where={s: "S"})
    ```

**Why this matters.** Mutants arise when *sensitive* cells divide, so they are supplied at
the **parent's** division rate, `mu*g[S]*B[S]`, whatever their own growth rate. A costly
mutant (here growing at half the rate) is still supplied at the full rate of its parents.
Applying `mu` to the mutant's own growth rate would under-supply it by exactly the cost.
With `of=div` the supply is right by construction; the result matches the closed-form
solution of the two ODEs:

```python
t = np.array([0.0, 5.0, 10.0])
sol = m.solve((0, 10), times=t, rtol=1e-10, atol=1e-10)
gS, gR, mu_, S0 = 1.0, 0.5, 1e-6, 1e6
exact = mu_ * S0 * gS / (gS * (1 - mu_) - gR) * (np.exp(gS * (1 - mu_) * t) - np.exp(gR * t))
print(sol["B"][:, 1])                 # resistant cells
print(exact)                          # the same, to 8 digits
```

`of=` has no default in Python, so the two cases, switching (`of=B`) and production
(`of=div`), cannot be confused. `explain()` says which one each transition is:

```python
print(m.explain().split("\n\n")[1])
```

## Transitions written out

`m.to_latex()` writes each transition term out category by category, so the equations of an
article come from the model itself:

```python
print(m.to_latex())
```

For the mutation model this gives \( \mathcal{T}^{M}_{s}[\mathrm{div}]_S = -\mu\,\mathrm{div}_S \)
and \( \mathcal{T}^{M}_{s}[\mathrm{div}]_R = \mu\,\mathrm{div}_S \).

## Joint transitions on two axes

When a population is structured by two categorical axes, transitions can act on one axis or
on both at once. Bacteria facing two phages carry one susceptibility axis per phage,
`B[r1, r2]`. A mutation on one locus moves cells along that axis; a single mutation that
gives resistance to both (a shared receptor) moves them along both at once, a **joint
transition**, written `trans_{r1,r2}(X, div)`:

=== "Notation"

    ```tq
    model two_phage
    axis r1 categorical [S, R]                        # susceptibility to phage 1
    axis r2 categorical [S, R]                        # susceptibility to phage 2
    param g=1, K=1e9, beta=50, dp=0.1, mu=1e-6, mux=1e-8
    param k1[r1] = {S: 1e-9, R: 0}
    param k2[r2] = {S: 2e-9, R: 0}

    field B(r1, r2)

    N   = sum_r1(sum_r2(B))
    A1  = sum_r1(sum_r2(k1[r1]*B))                    # adsorbing cells, phage 1
    A2  = sum_r1(sum_r2(k2[r2]*B))
    div = g*B*(1 - N/K)

    transitions M1 on r1: S -> R = mu
    transitions M2 on r2: S -> R = mu
    transitions X on (r1, r2): (S, S) -> (R, R) = mux # one mutation, resistant to both

    d_t(B)  = div - k1[r1]*P1*B - k2[r2]*P2*B
              + trans_r1(M1, div) + trans_r2(M2, div) + trans_{r1,r2}(X, div)
    d_t(P1) = (beta - 1)*A1*P1 - dp*P1
    d_t(P2) = (beta - 1)*A2*P2 - dp*P2

    B[r1=S, r2=S](0) = 1e6                            # the other genotypes start at 0
    P1(0) = 1e4
    P2(0) = 1e4
    ```

=== "Python"

    ```python
    m = tq.Model("two_phage")
    r1 = m.axis("r1", labels=["S", "R"])
    r2 = m.axis("r2", labels=["S", "R"])
    g, K, beta, dp, mu, mux = m.params(g=1, K=1e9, beta=50, dp=0.1, mu=1e-6, mux=1e-8)
    k1 = m.param("k1", {"S": 1e-9, "R": 0}, over=r1)
    k2 = m.param("k2", {"S": 2e-9, "R": 0}, over=r2)
    B = m.field("B", axes=[r1, r2])
    P1 = m.compartment("P1")
    P2 = m.compartment("P2")
    N = m.integral(B, over=[r1, r2], name="N")
    A1 = m.integral(k1[r1] * B, over=[r1, r2], name="A1")
    A2 = m.integral(k2[r2] * B, over=[r1, r2], name="A2")
    div = m.define("div", g * B * (1 - N / K))
    M1 = m.transitions("M1", axis=r1, rates={"S->R": mu})
    M2 = m.transitions("M2", axis=r2, rates={"S->R": mu})
    X = m.transitions("X", axis=[r1, r2], rates={"S,S->R,R": mux})
    m.reaction(B, div)
    m.hazard(B, k1[r1] * P1)
    m.hazard(B, k2[r2] * P2)
    m.transition(B, axis=r1, matrix=M1, of=div)
    m.transition(B, axis=r2, matrix=M2, of=div)
    m.transition(B, axis=[r1, r2], matrix=X, of=div)
    m.ode(P1, (beta - 1) * A1 * P1 - dp * P1)
    m.ode(P2, (beta - 1) * A2 * P2 - dp * P2)
    m.init(B, 1e6, where={r1: "S", r2: "S"})
    m.init(P1, 1e4)
    m.init(P2, 1e4)
    ```

- `trans_r1(M1, div)` moves cells along `r1` only, in every category of `r2`: a cell already
  resistant to phage 2 can still become resistant to phage 1.
- In the joint form each category is a pair: `(S, S) -> (R, R)`. In Python the pair is
  written `"S,S->R,R"`, with the axes in the order given to `axis=[r1, r2]`.
- `trans_s` is short for `trans_{s}`.
- Per-category parameters over different axes combine freely: `k1[r1]` varies over `r1`,
  `k2[r2]` over `r2`, and in the equation of `B[r1, r2]` each takes the value of its own
  category.

```python
sol = m.solve((0, 24), times=[0, 12, 24])
print(sol["B"][-1])                   # [[S,S  S,R], [R,S  R,R]] at 24 h: the double resistant
```

## PBPK: exchange between tissue spaces

In a PBPK model the drug in each organ is split between vascular, interstitial and cellular
spaces. With the field holding a concentration and each category weighted by its volume
(`weights=V[organ, sub]`), the same transition statements describe permeability-limited
exchange. The rates are then **clearances** (volume per time): a move at rate `PS1` carries
`PS1 × concentration` of the origin, an amount per time, and each space's concentration
changes by that amount divided by its own volume. A rate may vary over another axis: the
uptake back from cells, `PS2/Kp[organ]`, depends on the organ's partition coefficient.

A gut–liver example (the portal vein carries drug from the gut to the liver, which
eliminates it; the full nine-organ model, with arterial and venous blood, is in
[Tutorial 6](../tutorials/06_whole_body_pbpk.ipynb)):

=== "Notation"

    ```tq
    model gut_liver
    axis organ categorical [gut, liver]
    axis sub categorical [vascular, interstitial, cellular]

    param V[organ, sub] = {                       # volumes, L
        gut:   {vascular: 0.06, interstitial: 0.18, cellular: 0.96},
        liver: {vascular: 0.09, interstitial: 0.27, cellular: 1.44},
    }
    param Kp[organ] = {gut: 2, liver: 3}
    param PS1=50, PS2=10, Q_gut=65, CLint=30      # L/h

    field C(organ, sub) weights=V[organ, sub]     # concentrations; amount = C x V

    transitions X on sub:                         # exchange between the three spaces
        vascular -> interstitial = PS1
        interstitial -> vascular = PS1
        interstitial -> cellular = PS2
        cellular -> interstitial = PS2/Kp[organ]
    transitions portal on organ: gut -> liver = Q_gut

    d_t(C)               = trans_sub(X, C)
    d_t(C[sub=vascular]) = trans_organ(portal, C)
    d_t(C[organ=liver, sub=cellular]) = -CLint/Kp[organ]*C/V[organ, sub]

    dose C[organ=gut, sub=vascular] at t=0: 100
    observable amount = sum_organ(sum_sub(C))
    observable C_liver = C[organ=liver, sub=cellular]
    ```

=== "Python"

    ```python
    m = tq.Model("gut_liver")
    organ = m.axis("organ", labels=["gut", "liver"])
    sub = m.axis("sub", labels=["vascular", "interstitial", "cellular"])
    V = m.param("V", {"gut": {"vascular": 0.06, "interstitial": 0.18, "cellular": 0.96},
                      "liver": {"vascular": 0.09, "interstitial": 0.27, "cellular": 1.44}},
                over=[organ, sub])
    Kp = m.param("Kp", {"gut": 2, "liver": 3}, over=organ)
    PS1, PS2, Q_gut, CLint = m.params(PS1=50, PS2=10, Q_gut=65, CLint=30)
    C = m.field("C", axes=[organ, sub], weights=V[organ, sub])
    X = m.transitions("X", axis=sub, rates={
        "vascular->interstitial": PS1, "interstitial->vascular": PS1,
        "interstitial->cellular": PS2, "cellular->interstitial": PS2 / Kp[organ]})
    portal = m.transitions("portal", axis=organ, rates={"gut->liver": Q_gut})
    m.transition(C, axis=sub, matrix=X, of=C)
    vascular = C.at(sub="vascular")
    m.transition(vascular, axis=organ, matrix=portal, of=vascular)
    m.hazard(C.at(organ="liver", sub="cellular"), CLint / Kp[organ] / V[organ, sub])
    m.dose(C.at(organ="gut", sub="vascular"), times=[0], amounts=100)
    m.observable("amount", m.integral(C, over=[organ, sub]))
    m.observable("C_liver", C.at(organ="liver", sub="cellular"))
    ```

- `trans_sub(X, C)` acts in every organ; the portal flow acts only on the vascular slice,
  `d_t(C[sub=vascular]) = trans_organ(portal, C)` (in Python, a transition on
  `C.at(sub="vascular")`).
- The dose of 100 is an amount: in the gut's vascular space (0.06 L) it starts as a
  concentration of 100/0.06.
- The ledger counts amounts. The two transitions move drug but net to 0; only the hepatic
  elimination removes it:

```python
sol = m.solve((0, 4), times=[0, 0.5, 1, 4])
print(sol["C"][0])                    # the starting concentrations, by organ and space
print(sol.ledger.table("C"))
```

## From Python

| Notation | Python |
|---|---|
| `transitions M on s: S -> R = mu` | `M = m.transitions("M", axis=s, rates={"S->R": mu})` |
| `transitions X on (r1, r2): (S, S) -> (R, R) = mux` | `m.transitions("X", axis=[r1, r2], rates={"S,S->R,R": mux})` |
| `trans_s(Q, B)` (switching) | `m.transition(B, axis=s, matrix=Q, of=B)` |
| `trans_s(M, div)` (production) | `m.transition(B, axis=s, matrix=M, of=div)` |
| `trans_{r1,r2}(X, div)` | `m.transition(B, axis=[r1, r2], matrix=X, of=div)` |
| `d_t(C[sub=vascular]) = trans_organ(portal, C)` | `m.transition(C.at(sub="vascular"), axis=organ, matrix=portal, of=C.at(sub="vascular"))` |

`m.transitions` also accepts the moves as `(from, to, rate)` triples. Transitions are only
between categories: along a continuous axis, use transport or diffusion
([Reactions, transport and diffusion](terms.md)).
