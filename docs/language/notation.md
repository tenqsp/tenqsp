# The TenQSP model language (PDE notation) — proposal

*Status: **approved by the project lead, 2026-10-22** (Phase 3, step 3.1); implemented in steps
3.2–3.5. The decisions are listed in §8. Read since step 3.3: every statement of §2 and
every term of §3; `m.to_text()` since step 3.4; `m.to_latex()`, `m.explain()`,
`m.diagnostics()` and `m.fokker_planck` since step 3.5 (§7). Boundary conditions written
as in a paper, under their equation, and fields read at an end, since IR 0.10 (project lead,
2026-10-28; §2.4).*

A model is written as text, the way equations look in a paper. The text is read into the
same model description (the IR) that the Python API builds, so a model written either way
compiles, solves and saves identically. Formulas inside the text use the formula language
([formulas.md](formulas.md)) unchanged.

```python
m = tq.Model.from_text(text)          # or tq.load("phage.tq")
m.solve((0, 24), times=...)           # solver settings stay here, not in the text
print(m.to_text())                    # any model, also one built in Python, as text
print(m.to_latex())                   # the equations as a paper shows them
```

---

## 1. A first example

The phage infection-age model of PRD §8.1, complete:

```
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
    at a=2: outflow                               # older ones leave (in the ledger)
d_t(P) = beta*int_a(h(a)*I) - k*B*P - delta*P     # one phage used up per infection

B(0) = 1e6                                        # I starts at 0: not given
P(0) = 1e7

dose P at t=0: 1e8
observable burst_rate = int_a(h(a)*I)
```

Reading it:

- One statement per line; `#` starts a comment. A statement continues on the next line when
  that line is indented, starts with an operator (`+ - * /`), follows a line ending with a
  comma, or a bracket is still open.
- Lines starting with a keyword (`axis`, `param`, `field`, `flux`, …) declare things;
  `d_t(X) = …` is the equation of `X`; `X(0) = …` is its value at time 0; `name = …` (or
  `name(axes) = …`) names a formula.
- A boundary condition stands right under its field's equation, as in a paper:
  `at a=0: I = k*B*P` is the renewal condition I(t, 0) = kBP (speed 1 along `a`, so the
  flux at that end is I itself); `at a=2: outflow` says what reaches the oldest age leaves
  the model (§2.4). The same term `k*B*P` is written in each equation it changes;
  `m.explain()` lists where it appears and points out a place that is missing it.
- The order of statements does not matter, except that each name is declared once.

## 2. Statements

| Statement | Example | Python API equivalent |
|---|---|---|
| model name | `model pbpk` (optional; default: the file name) | `tq.Model("pbpk")` |
| continuous axis | `axis a in [0, 2] n=40 units=h kind=one_way` | `m.axis("a", 0, 2, 40, units="h", kind="one_way")` |
| | `axis x in [0, 1] n=30 spacing=geometric ratio=1.1` | `spacing=, ratio=` |
| | `axis x faces=[0, 0.1, 0.5, 1]` | `faces=` |
| categorical axis | `axis s categorical [S, R]` (labels with spaces in quotes: `["wild type", R]`) | `m.axis("s", labels=[...])` |
| parameters | `param k=1e-9, beta=50` | `m.params(k=1e-9, beta=50)` |
| | `param k = 1e-9 units=mL/PFU/h` (units: one parameter per line) | `m.param("k", 1e-9, units=...)` |
| one value per category | `param k[s] = {S: 1e-9, R: 1e-12}` | `m.param("k", {...}, over=s)` |
| declared without a value (given when solving or simulating, `params={"ka": 1.2}`; a missing value is an error) | `param ka, CL` · `param k[s]` | `m.params("ka", "CL")` · `m.param("k", over=s)` |
| a table over several axes | `param V[organ, sub] = {lung: {vascular: 0.06, …}, …}` (§2.2) | `over=[organ, sub]` |
| field | `field I(a, z) units=cells` | `m.field("I", axes=[a, z])` |
| field with weights | `field C(organ, sub) weights=V[organ, sub]` | `weights=V` |
| compartment | `compartment P units=PFU` | `m.compartment("P")` |
| | `compartment Cart volume=V_art` | `volume=` |
| value at time 0 | `P(0) = 1e7`, `B[s=S](0) = 1e7*point(z, at=0)` (§2.1) | `m.init(...)` |
| named formula | `h(a) = h_max*sigmoid((a - tau)/s)` or `div = g*B*(1 - N/K)` | `m.define(...)` |
| named total | `B_tot = int_z(sum_s(B))` | `m.integral(B, over=[z, s], name="B_tot")` |
| equation | `d_t(B) = …` | `m.ode`, `m.reaction`, `m.advect`, … (§3) |
| equation on a slice | `d_t(C[sub=vascular]) = trans_organ(portal, C)` | the same calls with `C.at(sub="vascular")` |
| flux (material moving between places) | `flux J_ven: C[sub=vascular] -> Cven = Q_ven[organ]*C` (§2.3) | `m.transfer(...)` |
| transitions | `transitions M on s: S -> R = mu, R -> S = mu` | `m.transitions("M", axis=s, rates={...})` |
| joint transitions | `transitions X on (r1, r2): (S, S) -> (R, R) = mux` | `axis=[r1, r2]` |
| boundary condition (under the field's equation, §2.4) | `at a=0: v*I = k*B*P` (the flux along the axis there; cells entering age 0) | `m.at(I, a=0, flux=k*B*P)` |
| | `at z=0: -Dz*d_z(D) = alpha*B - beta*D` (inside, `D` is its value at that end) | `m.at(D, z=0, flux=alpha*B - beta*D)` |
| | `at z=1: C = 1` (held at a value) | `m.at(C, z=1, value=1)` |
| | `at a=2: outflow` (what the flow brings there leaves) | `m.at(I, a=2, outflow=True)` |
| | `at z=0: d_z(C) = 0` (nothing crosses: the default) | `m.at(C, z=0, flux=0)` |
| a field at an end, in any formula | `D(z=0)`, `I(a=upper)` (as a paper's D(t, 0)) | `D(z=0)`, `I(a="upper")` |
| older end statements (still read) | `inflow I at a=lower: …`, `outflow I at a=upper [to L]`, `dirichlet C at z=upper: 1`, `robin C at z=upper: h*(M/V - C) [with M]`, `zero_flux C at z=lower` | `m.inflow`, `m.outflow`, `m.boundary` |
| doses | `dose P at t=0, 24: 1e8` | `m.dose(P, times=[0, 24], amounts=1e8)` |
| | `dose P at t=0, 24: 1e8, 5e7` (one amount per time) | `amounts=[1e8, 5e7]` |
| | `dose P at t=0 to 144 every 24: 1e8` (a regimen, like NONMEM's II/ADDL; the last dose falls on the end time, so `0 to 140 every 24` is refused) | the seven times written out |
| | `dose C[organ=gut, sub=vascular] at t=0: 100` (every axis fixed) | `m.dose(C.at(...), ...)` |
| | `dose G at t=0, 12: 100*point(g, at=0)` (into a field over continuous axes: an amount times a shape, as `G(0) = …`; categorical axes fixed) | `m.dose(G, times=[0, 12], amounts=100, shape=tq.point(g, at=0))` |
| | `dose B at t=0: density(1e6*exp(-z))` (a density added to the field; a bare number is refused: say where it goes) | `m.dose(B, times=[0], density=1e6*tq.exp(-z))` |
| lag time (NONMEM's ALAG) | `lag depot = tlag`: every dose into `depot` arrives `tlag` later (an infusion is shifted whole) | `m.lag(depot, "tlag")` |
| bioavailability (NONMEM's F) | `bioavailability depot = F`: only the fraction `F` of each dose arrives (bolus amounts and infusion rates scaled) | `m.bioavailability(depot, "F")` |
| output | `observable phage = P` / `observable C_liver = C[organ=liver, sub=vascular]` | `m.observable(...)` |
| extinction group | `extinction B[s=R] below 1` | `m.extinction(B.at(s="R"), threshold=1)` |
| | `extinction lineage1: P1, I1 below 1 with noise_floor` | `name="lineage1", noise_floor=True` |

**Slices** are written `C[organ=liver, sub=vascular]`: the field `C` where those axes are
fixed. Square brackets with `=` are a slice; `k[s]` without `=` is a parameter value per
category, as in formulas. A slice may be read in any formula (the Python API has no
equivalent yet) (e.g. a drug effect driven by
`C[organ=liver, sub=cellular]`); in a slice equation, `V[organ, sub]` takes the fixed label
of `sub`.

Declarations (`compartment`, `field`) never carry a value: a value written there could be
read as "fixed at this value". A compartment that needs no units or volume need not be
declared at all; its equation `d_t(X) = …` declares it.

### 2.1 Values at time 0

There is one form, as in a paper: **`X(0) = value`**. `(0)` always means
time 0.

```
P(0) = 1e7                                        # a compartment
I(0) = 0                                          # a field: density 0 everywhere
B(0) = 1e7*exp(-z)                                # a field: a density formula of its axes
B[s=S](0) = 1e7*normal(z, mean=0.1, sd=0.08)      # a part of a field: an amount and a shape
B[s=S](0) = 1e7*point(z, at=0)                    # all 1e7 in the cell at z=0
```

- **Anything not given starts at 0**, as in NONMEM and Monolix: a compartment, a whole
  field, or the parts of a field not covered by a slice (above, `B[s=R]`). This is not a
  warning; `explain()` lists what starts at 0, so a forgotten value is visible.
- **A value is a density, unless it contains a shape.** `1e7*exp(-z)` is a density (cells
  per unit of `z`). `1e7*normal(z, …)` is an **amount** times a **shape**: the shape is cut
  to the grid and scaled so the cells hold exactly 1e7 in total, whatever the grid. Shapes:
  `normal`, `lognormal`, `gamma`, `exponential`, `beta`, `uniform`, `point`
  ([formulas.md](formulas.md) §3).
- **Any other shape**: `shape(formula)` scales any formula of the axes to the amount,
  `B(0) = 1e7*shape(exp(-z))`. Shapes on several axes multiply:
  `1e7*normal(x, mean=0.5, sd=0.1)*normal(y, mean=0.2, sd=0.1)`.
- **A density that uses a shape function**: `density(normal(z, mean=0.1, sd=0.08))` is the
  curve itself (cells per unit of `z`), not scaled to an amount; it differs from
  `1*normal(…)` only by the part of the curve the axis cuts off.

**A density or an amount — the difference, side by side** (axis `z` in [0, 1]):

| written | means | at z=0 | at z=1 | total on the axis |
|---|---|---|---|---|
| `B(0) = 1e7*exp(-z)` | the formula **is** the value (a density) | 1e7 | 1e7·e⁻¹ | ∫ = 6.3e6 |
| `B(0) = 1e7*shape(exp(-z))` | 1e7 **spread** like exp(−z) (an exponential cut to [0, 1]) | 1.58e7 | 5.8e6 | exactly 1e7 |
| `B(0) = 1e7*normal(z, …)` | 1e7 spread like a normal (named shapes are always spread) | | | exactly 1e7 |
| `B(0) = 1e7*density(normal(z, …))` | the normal curve **is** the value | | | 1e7 × the part of the curve inside [0, 1] |

So `density(…)` is only needed around a named shape: a formula with no shape in it is
already a density. Values are stored as cell averages, so "1e7 at z=0" is the average of
the curve over the first cell.
- **`point(z, at=0)`** puts the whole amount in the one cell that contains `z=0`. The
  density there is the amount divided by that cell's width, so the amount is exact but it
  is spread over one cell: a finer grid makes the start sharper. On a face between two
  cells, the cell above it is used; `at=` the upper end uses the last cell.
- One value per part: giving `B[s=S](0)` twice, or `B(0)` together with `B[s=S](0)`, is
  an error that names both lines.

### 2.2 Parameter tables

A parameter that varies over categorical axes is written in braces, one level per axis, in
the order of the axes in the brackets:

```
param k[s] = {S: 1e-9, R: 1e-12}

param V[organ, sub] = {
    lung:  {vascular: 0.06, interstitial: 0.18, cellular: 0.96},
    brain: {vascular: 0.07, interstitial: 0.21, cellular: 1.12},
    ...
}

param fu[drug, organ, sub] = {
    A: {lung: {vascular: 1, interstitial: 0.8, cellular: 0.5}, ...},
    B: {lung: {...}, ...},
}
```

Labels may be in any order; a missing or unknown label is an error that names it. This is
the same shape the Python API takes (`m.param("V", {...}, over=[organ, sub])`).

Tables kept in files stay in Python: `tq.table(df, axes=["organ", "sub"], value="V")`
turns a long pandas table (one row per value) into these nested braces, and values are
given when the model is read:

```python
V = tq.table(pd.read_csv("volumes.csv"), axes=["organ", "sub"], value="V")
m = tq.load("pbpk.tq", params={"V": V})          # replaces the default in the file
```

### 2.3 Fluxes: material moving between places

When material leaves one place and arrives in another (blood flow, uptake, a bacterium
becoming an infected cell), give the flow a name with `flux`, say where it goes, and use
the name in both equations: with a minus where it leaves, with a plus where it arrives.

```
flux J_out: C[organ=lung, sub=vascular] -> Cart = Q_CO*C   # amount/h

d_t(C[organ=lung, sub=vascular]) = ... - J_out/V[organ, sub]
d_t(Cart)                        = ... + J_out/V_art
```

`flux name: from -> to = amount per time`. The two ends are compartments or slices of
fields; inside the expression the field of a slice end (`C` above) means its value on that
slice. A flux is an **amount per time**: in the equation of a concentration it is divided
by the volume, as in a paper; in the equation of an amount (or of cells in a field without
weights) it is used as it is. Usually it is a rate times its source (`k*A`, `Vmax*A/(Km + A)`);
a flux that does not read its source (zero order, `= k0`) may leave a place without a volume,
but out of a concentration it is refused, since it is almost always `Q` written for `Q*C`.

**What the library checks.** For every flux, the amount that leaves `from` must equal the
amount that arrives in `to`, after the volumes:

- the equation of `from` has the flux with a minus, the equation of `to` with a plus (or, at
  the end with fewer axes, summed over the others: `sum_organ(J_art)` when one compartment
  sends to every organ);
- an inflow boundary counts as arriving in `to` (`inflow I at a=lower: infect`);
- anywhere else the name is an ordinary value and may be used freely: phage used up by
  adsorption is `d_t(P) = … - infect` (a loss of `P`, not part of the flux).

A mismatch is an error that names the flux and both lines:

```
line 31: flux J_out leaves C[organ=lung, sub=vascular] as J_out but arrives in Cart as
         J_out*V_art/V_ven (line 38 divides by V_ven); divide by V_art there.
```

Inside the library a balanced flux becomes the same transfer that the Python API's
`m.transfer` makes, so a model written either way saves identically, and `to_text()`
writes transfers as fluxes.

**Fluxes are optional.** A flow may also be written as a paper writes it, the same term
in each equation it changes (`- Q_CO*C[…]/V[organ, sub]` in one, `+ Q_CO*C[…]/V_art` in
the other). It gives the same results, and the ledger closes. `m.explain()` then matches
the term across the equations and boundary conditions ("Terms shared between equations":
each place with its sign, or its factor for a conversion such as b phage per lysed cell;
with declared volumes, whether the amounts balance), and points out a term proportional to
a place that is missing from that place's own equation (phage not used up by adsorption).
A process with more than two participants, such as adsorption B + P → I, can only be
written this way: an arrow has two ends.

**A flux into an inflow** (a bacterium entering the infected cells at age 0): the inflow
counts as the place it arrives in, `inflow I at a=lower: infect`. The library keeps `-infect`
in the source's equation and the inflow as written; there is no transfer in between. In
the form of §2.4 this is simply `at a=0: I = k*B*P` under the equation of `I`.

### 2.4 Boundary conditions

A field's axis has two ends, and what happens there is part of the equation, as in a
paper: written right under the field's equation, one line per end that needs one.

```
d_t(D) = d_z(Dz*d_z(D)) - kD*P*D
    at z=0: -Dz*d_z(D) = alpha*(1 - frac)*B - beta*frac*D
d_t(I) = -d_l(frac/tau*I) - mu*I
    at l=0: frac/tau*I = k*P*B
    at l=1: outflow
d_t(C) = d_z(De*d_z(C)) - kr*C
    at z=1: C = 1
```

- **A flux condition**, `at z=0: <flux> = <formula>`. The left side is the flux of the
  field along the axis as its equation's transport gives it: `v*I` for a flow at velocity
  `v`, `-Dz*d_z(D)` for diffusion, `v*D - Dz*d_z(D)` for both. TenQSP checks it against the
  equation and, if it differs, says what it should be. The flux is positive toward larger
  coordinates, as in a paper: at the lower end a positive flux enters, at the upper end it
  leaves. On the right, the field's own name means its value at that end (`beta*frac*D` is
  D(t, 0)), and the formula must be linear in it: each term proportional to it or free of
  it, e.g. an exchange `h*(C - Cout)`. Other fields are read at that end the same way,
  `I(z=0)`. Where the flow *leaves* through the end, it carries out the value of the last
  cell (upwind), as a flow must: there the condition sets the value read at the end (and,
  with diffusion, the gradient), not how much the flow carries out. For a flow that only
  leaves, write `outflow`. A speed of exactly 1 is written `1*C`, since `C = …` alone
  would be a held value.
- **A held value**, `at z=1: C = 1` (C(t, 1) = 1).
- **No flux**, `at z=0: d_z(C) = 0` (or the flux `= 0`). This is the default for any end
  not mentioned, except the next case.
- **Outflow**, `at a=2: outflow`: what the flow carries to this end leaves the model. The
  ledger records it, and another equation may use it by reading the field there, e.g. the
  phage released by lysis, `b*v*I(l=1)`. Along an axis declared `kind=one_way`, the end
  the flow reaches must be stated; TenQSP refuses a model that leaves it unsaid, because
  the material would otherwise leave unnoticed. (Models saved as IR before 0.10, which
  relied on a silent outflow there, get it written in when they are read. A text file
  carries no version, so one written before then is refused with a message naming the
  line to add, `at a=2: outflow`.)

An end can be named by its coordinate (`z=0`, `a=2`) or as `z=lower`, `z=upper`.

**A field read at an end**, `D(z=0)` in any formula, is the value the condition at that
end gives: the level the exchange there sees, the value held, or the edge value carried
out by an outflow. So what crosses and what another equation reads are the same number,
and the books balance exactly:

```
d_t(B) = g*frac*B - alpha*(1 - frac)*B + beta*frac*D(z=0)    # B gains what leaves D
```

The older statements (`inflow`, `outflow … to L`, `dirichlet`, `robin … with M`,
`zero_flux`, §2) are still read, so older models keep working; `to_text()` writes the
form above wherever it says the same thing.

**Elimination** (material leaving the model) is a loss term, not a flux:
`d_t(C[organ=liver, sub=cellular]) = -CLint/Kp_liver*C/V[organ, sub]` (a clearance
removes CL x concentration per time, so the concentration falls at CL/V).

## 3. How an equation is read

`d_t(F) = right-hand side` is split into its terms (the parts joined by `+` and `-`), and
each term is recognised:

| Term | Read as | API |
|---|---|---|
| `-d_a(v*F)` | **transport** of `F` along `a` at velocity `v` (conservation form; the sign of `v` may change; upwinding is automatic). The velocity is what multiplies `F`; it may depend on fields, `F` included (Burgers `-d_x(0.3*u*u)`) | `m.advect(F, axis=a, velocity=v)` |
| `-d_a(F)` | transport at velocity 1 | `velocity=1` |
| `d_x(D*d_x(F))` | **diffusion** with diffusivity `D` | `m.diffuse(F, axis=x, D=D)` |
| `-r*F` (`r` does not contain `F`) | a **loss** at per-capita rate `r` (the ledger shows it as a loss) | `m.hazard(F, r)` |
| `trans_s(M, X)` | **transitions** along the categorical axis `s`, with rates `M`, of the quantity `X` | `m.transition(F, axis=s, matrix=M, of=X)` |
| anything else | added cell by cell (births, sources, interactions) | `m.reaction(F, ...)` |

**Transitions.** `trans_s(M, X)` is named like the other operators along an axis (`d_a`
differentiates along `a`, `int_a` integrates along it). It moves `X` between the categories
of `s` with the rates declared in `transitions M on s: …`:

- `X` is the field itself, `trans_s(Q, B)`: cells **switch** category at rate `Q` per cell
  (phenotypic switching, exchange between tissue spaces). Python:
  `m.transition(B, axis=s, matrix=Q, of=B)`.
- `X` is a rate of production, `trans_s(M, div)`: a fraction `M` of it ends up in another
  category, e.g. daughters of divisions that carry a mutation, at the **parent's** division
  rate. Python: `m.transition(B, axis=s, matrix=M, of=div)`. (`of=` has no default, so the
  two cases cannot be confused.)
- Joint transitions on two axes are written `trans_{r1,r2}(X, div)`; `trans_s` is short for
  `trans_{s}`.

Only the moves are declared (`S -> R = mu`); what leaves the origin is derived, so a
transition can neither create nor lose cells. Written out, `trans_s(M, X)` in category `d`
is

    sum over o ≠ d of M[d, o]·X[o]   −   (sum over d' ≠ d of M[d', d])·X[d]

(what arrives from the other categories minus what leaves). `m.to_latex()` prints every
transition term in this expanded form, so the equations of an article come from the model
itself.

Inside any formula:

- `int_a(expr)` integrates over a continuous axis, `sum_s(expr)` sums over a categorical
  one; nest them for several axes: `int_z(sum_s(B))`. Each becomes a named total (the same
  integral written twice is stored once).
  For a field with weights (concentrations, e.g. PBPK), `int_`/`sum_` add up **amounts**
  (value × weight), as the Python API's `m.integral` does: `sum_organ(sum_sub(C))` is the
  amount of drug in the tissues. A slice read in a formula, `C[organ=liver, sub=cellular]`,
  is the field's value there (a concentration).
- `grad_x(S)` is the slope of `S` along `x` (only inside a velocity along `x`, e.g.
  chemotaxis `-d_x(chi*grad_x(S)*B)`); `grad(S, x)` also works.
- A named formula declared with its axes, `h(a) = …`, may be used as `h(a)` or `h`. The
  brackets document what it varies over; they are not function arguments (`h(2)` is
  refused).

**Scheme.** Transport uses the `erlang` scheme (exactly a transit-compartment chain) unless
the equation ends with `using sharp` (sharper fronts on the same grid), or `using sharp on a`
to choose per axis.

**Transport plus diffusion is a Fokker–Planck equation.** Single cells that drift and
jitter along an axis, `dz = w dt + sqrt(2 D) dW` (an SDE in Itô form), have a density that
obeys `d_t(F) = -d_z((w - dD/dz)*F) + d_z(D*d_z(F))`: transport at the drift *minus the
slope of D*, plus diffusion. When D is the same everywhere the correction is 0. Between-cell
noise is thus already a TenQSP axis; `explain()` says which single-cell process an equation
with both terms describes, and `m.fokker_planck(F, axis=z, drift=w, D=D)` (§7) writes the two
terms from the drift and D, working out the slope of D. Its scheme is `sharp` by default:
here D sets the spread, and `erlang` would add the grid's own (about |w| x width / 2).

**Compartments** take one equation each, `d_t(P) = …`, with the whole right-hand side
(`d_` operators are refused there: a compartment has no axes).

**Material that moves between places** is written as a paper writes it, the same term in
each equation (and boundary condition) it changes; `m.explain()` matches the terms and
points out a missing one. A named `flux` (§2.3) is an optional, explicit form for a move
between exactly two places.

**Losses.** Each term of the form `-r*F` gets its own line in the ledger, so a solution can
say how many cells died and how many were infected. The user writes the same equation
either way.

## 4. Conditionals

Three spellings, all giving the same model (ADR 0006: the first true case wins; there is
always a final value):

```
k = k1 if C > MIC else k2                 # in one line (the formula language)

k = k1   if C > MIC                       # the cases brace of a paper
    k2   if C > MIC/2
    k3   otherwise

if C > MIC:                               # a block, for code ported from NONMEM/Monolix
    k = k1
    E = Emax
elif C > MIC/2:
    k = k2
    E = Emax/2
else:
    k = k3
    E = 0
```

In a block every branch must set the same names; each name becomes one conditional
formula. A block only names formulas; equations (`d_t`) are never inside one.

## 5. What is refused, with the message

| Written | Message (shortened) | Instead |
|---|---|---|
| `d_t(I) = -v*d_a(I)` | transport must be in conservation form | `-d_a(v*I)` |
| `d_t(I) = d_a(v*I)` | transport is written with a minus, `-d_a(v*I)`; to move toward lower `a`, give a negative velocity | `-d_a(-v*I)` |
| `-d_x(D*d_x(F))` | diffusion is written `+d_x(D*d_x(F))`; a negative diffusion has no solution | — |
| `-d_a(v*J)` in the equation of `I` | transport moves `I`: write `-d_a(v*I)` | — |
| `d_a(d_a(I))` | write diffusion as `d_a(D*d_a(I))` | — |
| `d_a(I)` in a compartment equation | a compartment has no axis `a` | `int_a(...)` |
| `d_t(I)` given twice | one equation per field (slices may add their own) | — |
| `flux J: A -> B = …` but `B`'s equation has no `+J` | the flux leaves `A` but arrives nowhere; add `+ J` to `B`'s equation | — |
| a flux whose amounts do not match | names the flux, both lines and the factor between them | fix the volume |
| `h(2)` | `h` is a named formula over `a`, not a function | — |
| `compartment P = 1e7` | a declaration has no value; the value at time 0 is written `P(0) = 1e7` | `P(0) = 1e7` |
| `trans_z(M, B)` with `z` continuous | transitions are between categories; along a continuous axis use transport or diffusion | `-d_z(...)` |
| `trans_s(M, B)` with `M` declared on `r` | `M` moves along `r`, not `s` | `trans_r(M, B)` |
| `at z=0: -2*Dz*d_z(D) = …` | the left side is the flux of `D` along `z`, from its equation: `-Dz*d_z(D)` | the flux as the equation gives it |
| `at z=0: …` with no equation above it | a boundary condition belongs to an equation: write it right below `d_t(F) = …` | — |
| `at z=0: -Dz*d_z(D) = h*D^2` | not linear in `D` (its value at this end) | terms proportional to `D` or free of it |
| a `one_way` axis whose last end is not mentioned | the flow carries `I` to the upper end of `a`; say what happens there | `at a=2: outflow` |
| `flux J: P -> I` used as `-int_z(J)` | a flux joins two places cell by cell; a process with more participants is written without an arrow | the term in each equation |

Every message gives the line number and points at the term, e.g.

```
line 14: d_t(I) = -v*d_a(I) - h*I
                    ^^^^^^^^^^^
the transport term must be in conservation form: write -d_a(v*I), not -v*d_a(I)
(TenQSP keeps every cell's books balanced only in this form).
```

## 6. Worked examples

Each of these is a model of the test suite. `tests/test_notation_fields.py` checks that each
solves to the numbers of its Python version; §6.3–6.5 are written exactly as their Python
versions and give the same model (V12). The others differ in wording only: §1 calls the age
axis `a`, §6.2 uses cells of width 1, §6.6 names its numbers as parameters.

### 6.1 One-compartment PK

```
param ka=1.2, CL=4, V=30

d_t(depot)  = -ka*depot
d_t(centr)  = ka*depot - CL/V*centr
C2 = centr/V

dose depot at t=0 to 72 every 12: 100
observable conc = C2
```

Both compartments are declared by their equations and start at 0 (not given).

With an absorption lag and a bioavailability, add two lines (both are formulas of parameters,
>= 0, and apply to every dose into `depot`, a regimen's too):

```
param tlag=0.5, F=0.8
lag depot = tlag
bioavailability depot = F
```

### 6.2 Friberg myelosuppression: transit compartments as an age axis

```
model friberg
axis z in [0, 3] n=3 kind=one_way                 # maturation: 3 transit compartments
param mtt=100, gamma=0.17, slope=0.5, ke=0.1, circ0=5
ktr = 4/mtt

field Transit(z)

flux mature: Prol -> Transit = ktr*Prol          # into the first transit compartment

d_t(Prol)    = ktr*Prol*(1 - slope*Conc)*(circ0/Circ)^gamma - mature
d_t(Transit) = -d_z(ktr*Transit)
inflow  Transit at z=lower: mature
outflow Transit at z=upper to Circ
d_t(Circ)    = -ktr*Circ
d_t(Conc)    = -ke*Conc

Prol(0)    = circ0
Transit(0) = circ0
Circ(0)    = circ0
```

The axis runs from 0 to 3 in 3 cells, so **each cell has width 1, is one transit
compartment, and holds density × width = its amount**. The numbers are then the ones of the
ODE model: every compartment starts at `circ0`, and the speed `ktr` is the rate from one
compartment to the next. This is the recommended way to write a transit chain.

The same model with `axis z in [0, 1] n=3` also works but uses other numbers: the cells are
1/3 wide, so the same amounts need density `3*circ0`, and the same rate ktr out of each cell
needs speed `ktr/3` (rate = speed ÷ cell width). Whatever the axis, `explain()` translates
back:

```
transport along z, velocity ktr (erlang): first-order upwind on 3 cells, which is
  exactly a chain of 3 transit compartments: each cell is one transit compartment
  (density = amount), passing to the next at rate |velocity|/width = ktr (= 0.04). Mean
  transit time 3/ktr (= 75), CV 1/sqrt(3) = 0.577. The spread of this chain is the
  model's variability, so refining the grid changes the model; it is not a convergence
  parameter (use sharp for transport without spread).
```

### 6.3 The toy phage model (`tenqsp.examples.toy_phage_model`)

```
model phage_toy
axis z in [0, 1] n=30                             # dormancy level
axis s categorical [S, R]                         # susceptibility
axis a in [0, 1.5] n=40 units=h kind=one_way      # infection age

param g_max=1, K=1e9, beta=50, delta=0.1, h_max=5, tau=0.8, s_h=0.05, w_amp=0.5,
      Dz=0.01, mu=1e-6
param k[s] = {S: 1e-9, R: 1e-12}
param cost[s] = {S: 0, R: 0.1}

field B(z, s)                                     # bacteria
field I(a, z)                                     # infected cells
compartment P                                     # free phage

B_tot     = int_z(sum_s(B))
div       = g_max*(1 - z)*(1 - cost[s])*B*(1 - B_tot/K)
h(a)      = h_max*sigmoid((a - tau)/s_h)
lysis     = int_a(int_z(h*I))
flux infect: B -> I = k[s]*(1 - 0.5*z)*B*P        # adsorption: bacteria become infected

transitions M on s: S -> R = mu, R -> S = mu

d_t(B) = div - infect + trans_s(M, div) - d_z(w_amp*(0.3 - z)*B) + d_z(Dz*d_z(B))
d_t(I) = -d_a(I) - h*I + d_z(Dz*d_z(I))           using sharp on a
inflow  I at a=lower: sum_s(infect)               # all genotypes enter at age 0
outflow I at a=upper
d_t(P) = beta*lysis - int_z(sum_s(infect)) - delta*P # phage used up by adsorption

B[s=S](0) = 1e7*normal(z, mean=0.1, sd=0.08)      # B[s=R], I and P start at 0

dose P at t=0: 1e9
observable bacteria = B_tot
observable phage = P
```

### 6.4 Two phages, one susceptibility axis each (`B[r1, r2]`)

```
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

### 6.5 Whole-body PBPK: tables, slices, fluxes

```
model pbpk
axis organ categorical [lung, brain, heart, muscle, adipose, kidney, gut, spleen, liver]
axis sub categorical [vascular, interstitial, cellular]

param V[organ, sub] = {                           # volumes, L
    lung:    {vascular: 0.06,   interstitial: 0.18,   cellular: 0.96},
    brain:   {vascular: 0.07,   interstitial: 0.21,   cellular: 1.12},
    heart:   {vascular: 0.0165, interstitial: 0.0495, cellular: 0.264},
    muscle:  {vascular: 1.45,   interstitial: 4.35,   cellular: 23.2},
    adipose: {vascular: 0.75,   interstitial: 2.25,   cellular: 12},
    kidney:  {vascular: 0.0155, interstitial: 0.0465, cellular: 0.248},
    gut:     {vascular: 0.06,   interstitial: 0.18,   cellular: 0.96},
    spleen:  {vascular: 0.0095, interstitial: 0.0285, cellular: 0.152},
    liver:   {vascular: 0.09,   interstitial: 0.27,   cellular: 1.44},
}
param Kp[organ]    = {lung: 1, brain: 0.5, heart: 2, muscle: 1.5, adipose: 5, kidney: 2,
                      gut: 2, spleen: 1.5, liver: 3}
param Q_art[organ] = {lung: 0, brain: 45, heart: 15, muscle: 60, adipose: 15, kidney: 70,
                      gut: 65, spleen: 10, liver: 20}         # L/h
param Q_ven[organ] = {lung: 0, brain: 45, heart: 15, muscle: 60, adipose: 15, kidney: 70,
                      gut: 0, spleen: 0, liver: 95}
param PS1=50, PS2=10, CLint=30, Kp_liver=3, Q_CO=300, R=0, V_art=1.7, V_ven=3.4,
      Q_gut=65, Q_spleen=10

field C(organ, sub) weights=V[organ, sub]        # concentrations; amount = C x V
compartment Cart volume=V_art
compartment Cven volume=V_ven

transitions X on sub:                            # exchange between the three spaces
    vascular -> interstitial = PS1
    interstitial -> vascular = PS1
    interstitial -> cellular = PS2
    cellular -> interstitial = PS2/Kp[organ]
transitions portal on organ: gut -> liver = Q_gut, spleen -> liver = Q_spleen

flux J_art:  Cart -> C[sub=vascular]              = Q_art[organ]*Cart   # amount/h
flux J_ven:  C[sub=vascular] -> Cven              = Q_ven[organ]*C
flux J_lung: Cven -> C[organ=lung, sub=vascular]  = Q_CO*Cven
flux J_out:  C[organ=lung, sub=vascular] -> Cart  = Q_CO*C

d_t(C)               = trans_sub(X, C)
d_t(C[sub=vascular]) = trans_organ(portal, C) + (J_art - J_ven)/V[organ, sub]
d_t(C[organ=lung, sub=vascular])  = (J_lung - J_out)/V[organ, sub]
d_t(C[organ=liver, sub=cellular]) = -CLint/Kp_liver*C/V[organ, sub]  # hepatic elimination

d_t(Cart) = (J_out - sum_organ(J_art))/V_art
d_t(Cven) = (sum_organ(J_ven) - J_lung)/V_ven + R/V_ven        # R: infusion
d_t(AUC)  = Cart
```

Everything starts at 0 (nothing is given); the drug arrives by a dose or the infusion `R`.
Each equation shows everything that enters and leaves its place, as in a paper, and the four
fluxes are checked: e.g. what `J_out` takes from the lung (÷ the lung's vascular volume)
must arrive in `Cart` (÷ `V_art`).

### 6.6 Biofilm with the bulk liquid: exchange with a compartment

```
model biofilm
axis z in [0, 1] n=20 kind=spatial               # depth: sealed at 0, liquid at 1
param De=0.1, kr=0.4, h=0.5, V=2, R=1, CL=0.3

field C(z)                                       # drug in the biofilm
compartment M                                    # amount in the liquid

d_t(C) = d_z(De*d_z(C)) - kr*C
    at z=1: -De*d_z(C) = h*(C - M/V)             # leaves through the surface
d_t(M) = R - CL*M/V + h*(C(z=1) - M/V)           # ... into the liquid
```

The exchange is written in both places, as in a paper; `C(z=1)` is the value at the
surface that the condition gives, so what leaves the biofilm is exactly what the liquid
gains. (The older `robin C at z=upper: h*(M/V - C) with M` says the same in one line.)

### 6.7 An extinction group with a noise floor

```
extinction phage: P, I below 1 with noise_floor
```

## 7. From Python

```python
m = tq.Model.from_text(text)                     # read; errors list every problem with lines
m = tq.load("phage.tq")                          # from a file (".tq", plain text)
m = tq.load("phage.tq", cells={"a": 80})         # same model, a finer grid on axis a
m = tq.load("pbpk.tq", params={"V": tq.table(df, axes=["organ", "sub"], value="V")})
text = m.to_text()                               # write any model as text (V12: reads back
                                                 # to the same model)
tex = m.to_latex()                               # the equations, transitions written out
print(m.explain())                               # what each term means, which scheme
print(m.diagnostics())                           # accuracy guidance (Péclet, cell crossing)
print(m.diagnostics(sol))                        # ... and checks of a solution
m.fokker_planck(F, axis=z, drift=w, D=D)         # an SDE's density: the two terms (§3)
m.at(D, z=0, flux=alpha*B - beta*D)              # a boundary condition (§2.4)
m.ode(B, -alpha*B + beta*D(z=0))                 # ... and D read at that end
```

`cells=` changes the number of cells of continuous axes, so a convergence study needs one
model file. `params=` replaces default values written in the file. A model read from text
is an ordinary `tq.Model`: Python calls may add to it.

**What the library tells you.**

- `m.explain()` reads the model (no solve) and says, term by term, what it means and which
  scheme carries it: an `erlang` transport as the transit chain it is (number of
  compartments, rate = |velocity| ÷ cell width, mean transit time, CV; a velocity that
  changes with time or the state gets the rate only), `sharp` with its cost,
  diffusion, the Fokker–Planck reading of transport plus diffusion, whether a transition
  acts on the cells (`of=` the field) or redistributes a production at the parent's rate;
  then every boundary (defaults included), the starting values and what starts at 0, doses,
  extinction groups, and the complete list of numerical safeguards (no value is ever
  clipped or capped). The same term written in several equations and boundary conditions
  is matched ("Terms shared between equations": each place with its sign or factor, and
  whether the amounts balance when volumes are declared); a term proportional to a place
  that is missing from that place's own equation is pointed out.
- `m.diagnostics()` gives accuracy guidance per transported axis: the cell Péclet number
  |v| × width ÷ D (above 2, `erlang`'s own spread exceeds D), the time to cross a cell (with
  `sharp` the solver's steps are at most about that long) and the transit chain's CV.
  `m.diagnostics(sol)` adds checks of a solution: a population whose total fell below `atol`
  and grew back (a warning: below `atol` the solver does not resolve it, so the regrowth may
  be an artefact) or stayed there (a note: the solve did not decide whether it would have
  grown back), values below −10 × `atol`, segments retried with Radau, and how well the
  ledger closed. It returns the same kind of report as `m.validate()`.
- `m.to_latex()` typesets the equations of `m.to_text()` (amsmath `align*`), with every
  transition written out category by category, then named formulas, fluxes, boundaries and
  starting values. Doses and outputs are a protocol, not equations, and are left out.

## 8. Decisions and open questions

Decided by the project lead (2026-10-22):

| Topic | Decision |
|---|---|
| Values at time 0 | one form, `X(0) = …`; declarations carry no value; anything not given starts at 0 (§2.1) |
| Transit chains | written with cells of width 1; `explain()` translates back to transit rates (§6.2) |
| Transitions | `trans_s(M, X)`; joint ones `trans_{r1,r2}(X, div)`; `m.to_latex()` writes the sum out (§3) |
| Python API | `m.mutate` and `m.switch` become one `m.transition(F, axis=, matrix=, of=)` |
| Parameter tables | nested braces at every depth; files read in Python with `tq.table` (§2.2) |
| Sharp scheme | `using sharp [on a]` at the end of the equation |
| Slices | `C[organ=liver, sub=vascular]`, readable in any formula |
| Moving material | named `flux name: from -> to = …` with its ends declared, checked for balance; recommended, not required; no `transfer` statement (§2.3) |
| Losses | each `-r*F` term its own ledger line |
| Elimination | a loss term; no `-> out` |
| Operators | Python's own only (no `<-`, no `eq`/`ne`, no `d/dt`); no rxode2-specific syntax |
| Doses | `dose P at t=0 to 144 every 24: 1e8`; several regimens are several lines and add up |
| Files | `.tq` |
| Starting values with any shape | `shape(formula)` spreads an amount like any formula; `density(named shape)` uses a named curve itself as a density (§2.1; project lead 2026-10-23) |
| An SDE's density | `m.fokker_planck(F, axis=, drift=, D=, scheme=)` writes velocity `drift − dD/dz` and diffusion `D`; `sharp` by default (project lead 2026-10-23) |
| Parameter values | a parameter may be declared without a value (`param ka`); values written in the model are defaults; values given when solving or simulating replace them; a missing value is an error naming it, never a guess (project lead 2026-10-24) |
| Doses in the model | allowed, as the model's default regimen; a regimen given at simulation replaces them, never adds to them (project lead 2026-10-24; step 4.1) |
| A total lost below `atol` | `m.diagnostics(sol)` notes a total that fell below `atol` and stayed there, besides the PRD's warning for one that grew back (project lead 2026-10-23) |

Decided by the project lead (2026-10-28):

| Topic | Decision |
|---|---|
| Boundary conditions | written as in a paper, under the field's equation: `at z=0: <flux> = …`, `at z=1: C = …`, `at a=2: outflow`; the left side checked against the equation's transport (§2.4) |
| A field at an end | `D(z=0)` readable in any formula, the value the condition at that end gives |
| Outflow along a one_way axis | must be stated; nothing leaves the model unnoticed |
| Material moving between places | as in a paper, the same term in each equation; nothing compulsory; `explain()` matches the terms and points out a missing one; `flux` stays as an optional explicit form |
| Older end statements | still read; `to_text()` writes the new form where it says the same |

No questions are open; the notation is approved for implementation (steps 3.2–3.4).
