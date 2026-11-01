# Notation reference

This page describes the TenQSP model language completely, in the order of a language
reference: how a model text is cut into statements, every statement, the operators and
functions that formulas may use, the names you cannot use, and what the reader refuses. It is
meant for looking things up. The [user guide](../guide/index.md) teaches the same language by
task, with longer explanations; each entry here links to the chapter that explains it.

Every entry gives the **syntax** (a template: `[ … ]` is optional, `…` repeats, words in
*italics* stand for something you write), the **meaning**, a short **example**, the
equivalent **Python** call, and the **errors** the reader gives for the common mistakes. The
messages are copied from TenQSP; each is printed under the line it is about,
`line 4: d_t(I) = -v*d_a(I) - h*I`, which is left out here. Two complete models close the
page.

## Lexical rules

### Statements and lines

A model text is a sequence of **statements**, normally one per line. Blank lines are ignored.

A line **continues** the statement above it when

| Rule | Example |
|---|---|
| it is indented | `param g=1, K=1e9,` / `      beta=50` |
| it starts with an operator `+ - * /` | `d_t(B) = div - infect` / `+ trans_s(M, div)` |
| the line above ends with a comma | `param k=4/100,` / `j=2` |
| a bracket `( [ {` is still open | a parameter table in braces over several lines |
| it is the `elif …:` or `else:` of an `if` block | see [`if` blocks](#if-elif-else-blocks) |

The one exception is a boundary condition, `at z=0: …`: it is always a statement of its own,
even when indented, and belongs to the equation above it
([Boundary condition](#boundary-condition-at)).

### Comments

`#` starts a comment that runs to the end of the line, except inside quotes
(`["wild type", R]`).

```tq
param k=1e-9            # adsorption rate, mL/PFU/h
```

### Names

A name is ASCII letters, digits and `_`, not starting with a digit: `B`, `k_on`, `Q_art`,
`lambda_`. Names are case-sensitive and unique across the model: an axis, a parameter, a field,
a compartment, a named formula, a set of transitions, an observable and an extinction group
cannot share a name. Some names are reserved; see [Reserved words](#reserved-words-and-names).

### Numbers and labels

Numbers are written `3`, `0.5`, `1e-9`, `2.5E+3`. Where a statement needs a number (a
parameter value, an axis range, a dose time, a threshold) it may be arithmetic of numbers,
`4/100` or `-1e-9`, but not a parameter. `inf`, `nan`, `True` and `False` are not numbers.

Labels of categories are names (`S`, `vascular`) or text in quotes (`"wild type"`); a label
in quotes is written in quotes wherever it is used: `B[s="wild type"]`.

### Order of statements

The order does not matter: a parameter may be used before the line that declares it. Three
things are tied to their place: a boundary condition belongs to the equation right above it
(other conditions of the same equation may come between), the lines of an `if` block belong
together, and so do the lines of a cases brace.

### Files

A model may be kept in a plain text file with the extension `.tq` and read with
`tq.load("model.tq")`. The model is named by its `model` line, or else by the file name.
`cells={"a": 80}` changes the number of cells of a continuous axis, `params={...}` replaces
values written in the file. `tq.Model.from_text(text)` reads a string the same way, with
`name=` for a text without a `model` line. See [Model files](../guide/axes-fields.md#model-files).

Errors:

```text
tq.load reads .tq model files, not 'x.txt'.
cells= names ['q'], which are not continuous axes of the model text.
params= names ['zz'], which are not parameters of the model text.
```

Every problem of a text is reported at once, as one `tq.NotationError` listing each problem
with its line.

**A fitting part after the model.** A line `fit` or `fit` *name* ends the model: TenQSP reads
the statements above it and nothing from that line on. The rest of the file is a fitting part
(data, estimated parameters, error models) for a fitting program such as tenqsp-fit to read,
so one file serves both: `tq.load` gives the model to simulate, the fitting program the whole
problem. `fit` followed by `=` is an ordinary named formula, not this line.

```tq
model theoph
param ka=1.6, CL=0.04, ke=0.086
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - ke*centr
observable conc = centr*ke/CL

fit theoph_pop          # TenQSP stops reading here
...
```

!!! note "New in 0.7.0 (not yet released)"
    With 0.6.1 from PyPI, a `fit` line is an unknown statement.

## Statements

### model

**Syntax** `model` *name*

**Meaning** The name of the model. Optional: without it the model is named after the file
(`tq.load`) or the `name=` given to `from_text`, else `model`.

**Example**

```tq
model phage_toy
```

**Python** `m = tq.Model("phage_toy")`

**Errors**

```text
write the model name as one word: model my_model.
the model is named twice.
```

### axis: continuous

**Syntax**

```text
axis name in [lower, upper] n=cells [kind=…] [spacing=geometric ratio=r] [units=u]
axis name faces=[f0, f1, …, fn] [kind=…] [units=u]
```

**Meaning** A range cut into `n` cells (or into the cells between the given `faces`). The
options:

| Option | Values | Meaning |
|---|---|---|
| `n` | whole number ≥ 1 | number of cells |
| `kind` | `bidirectional` (default), `one_way`, `spatial` | how material may move: `one_way` only toward larger values (age, maturation; its upper end must be stated), `spatial` documents a position |
| `spacing` | `uniform` (default), `geometric` | geometric cells grow by the factor `ratio` from one cell to the next |
| `ratio` | number | with `spacing=geometric` |
| `faces` | `[numbers]` | the cell boundaries themselves, instead of `in [ ] n=` |
| `units` | one word, `h`, `mm`, `1/h` | the units of the coordinate; write it last |

See [Continuous axes](../guide/axes-fields.md#continuous-axes) and
[Grids](../guide/numerics.md#grids).

**Example**

```tq
axis a in [0, 2] n=40 units=h kind=one_way
axis x in [0, 1] n=30 spacing=geometric ratio=1.1
axis d faces=[0, 0.1, 0.5, 1]
```

**Python** `m.axis("a", 0, 2, 40, units="h", kind="one_way")`;
`m.axis("x", 0, 1, 30, spacing="geometric", ratio=1.1)`; `m.axis("d", faces=[0, 0.1, 0.5, 1])`

**Errors**

```text
axis 'a': give the number of cells, n=….
axis 'a': n is a number of cells (a whole number >= 1).
axis 'a': the range is [lower, upper].
axis 'a': write 'axis a in [lower, upper] n=cells', 'axis a faces=[...]' or 'axis a categorical [A, B]'.
unknown option 'knd='; did you mean 'kind='? The options here are n=, units=, kind=, spacing=, ratio=, faces=.
Axis 'a': kind must be one of one_way, bidirectional, spatial, not 'oneway'.
Axis 'a': spacing must be 'uniform' or 'geometric', not 'log'.
Axis 'a': spacing='geometric' needs ratio=.
axis 'a': faces=[...] gives 2 cells but n=3; leave out n= (the faces decide it).
axis 'a': give faces=[...] or 'in [lower, upper] n=…'.
units=h,: units are one word such as 1/h or mL/h; put other options before units=, or give them on their own line.
```

### axis: categorical

**Syntax** `axis` *name* `categorical [`*label*`, …] [units=u]`

**Meaning** A set of categories: genotypes, phenotypes, organs, tissue spaces. Material
moves between them only through [transitions](#transitions). See
[Categorical axes](../guide/axes-fields.md#categorical-axes).

**Example**

```tq
axis s categorical [S, R]
axis g categorical ["wild type", R]
```

**Python** `m.axis("s", labels=["S", "R"])`

**Errors**

```text
axis 's': write the labels in square brackets, e.g. [S, R].
Axis 's': labels must be unique, got ('S', 'S').
```

### param

**Syntax**

```text
param name=value, name=value, …
param name = value units=u
param name, name, …
param name[axis] = {label: value, …}
param name[axis1, axis2] = {label1: {label2: value, …}, …}
param name[axis]
```

**Meaning** Parameters: values that do not change in time. Changing one never recompiles.

- A value is a number (or arithmetic of numbers). A value computed from other parameters is
  a [named formula](#named-formula), not a parameter.
- `units=` is for one parameter per line.
- **Without a value** (`param ka, CL`) the parameter is declared only; its value must be
  given when solving or simulating, `params={"ka": 1.2}`. A missing value is an error that
  names it, never a guess. Values written in the text are defaults that `params=` replaces.
- **One value per category**: `k[s] = {S: 1e-9, R: 1e-12}`. Every label once, in any order.
- **A table over several axes**: nested braces, one level per axis in the order of the
  brackets. Used in formulas as `V[organ, sub]`.

See [Parameters](../guide/axes-fields.md#parameters).

**Example**

```tq
param ka=1.2, CL=4, V=30
param k_ads = 1e-9 units=mL/PFU/h
param F
param k[s] = {S: 1e-9, R: 1e-12}
param V[organ, sub] = {
    liver: {vascular: 0.09, tissue: 1.5},
    gut:   {vascular: 0.06, tissue: 1.0},
}
```

**Python** `m.params(ka=1.2, CL=4, V=30)`; `m.param("k_ads", 1e-9, units="mL/PFU/h")`;
`m.params("F")`; `m.param("k", {"S": 1e-9, "R": 1e-12}, over=s)`;
`m.param("V", {...}, over=[organ, sub])`; `m.param("k", over=s)` (declared only). A long
table in a file becomes nested braces with `tq.table(df, axes=["organ", "sub"], value="V")`.

**Errors**

```text
'k=': write name=value (or name[axis] = {…}), or the name alone.
parameter 'k' must be a number, not '2*j'. To compute a value from other parameters, name a formula instead (e.g. 'ktr = 4/mtt').
parameter 'k': The number 1e309 is not finite at column 1 of '1e999'.
units= is for one parameter per line: param k = 1e-9 units=mL/h.
'k' has a table of values; say which axes it is over: k[s] = {…}.
'k[s]' needs one value per category, in braces: k[s] = {…}.
parameter 'k' over 's' needs one value for each of ['S', 'R']; missing ['R'].
parameter 'k' over 's' needs one value for each of ['S', 'R']; unknown ['X'].
parameter 'k': label 'S' is given twice.
The name 'k' is already used by a parameter; every name must be unique.
```

At a solve, a parameter declared without a value and not given:

```text
No value for ka: declared without a value in the model, so give them when solving or simulating, e.g. params={'ka': ...}.
```

### field

**Syntax** `field` *name*`(`*axis*`, …) [weights=`*formula*`] [units=u]`

**Meaning** A quantity spread over one or more axes: a density per unit of each continuous
axis, one value per category of each categorical axis. With `weights=` the field is a
concentration and its amount is value × weight (PBPK: `weights=V[organ, sub]`); totals and
fluxes then count amounts. A declaration never carries a value. See
[Fields](../guide/axes-fields.md#fields-densities-and-amounts).

**Example**

```tq
field I(a, z) units=cells
field C(organ, sub) weights=V[organ, sub] units=mg/L
```

**Python** `m.field("I", axes=[a, z], units="cells")`;
`m.field("C", axes=[organ, sub], weights=V, units="mg/L")`

**Errors**

```text
write a field with its axes: field I(a, z).
a declaration has no value; the value at time 0 is written B(0) = 1.
field 'B': model 'model' has nothing named 'y'.
```

### compartment

**Syntax** `compartment` *name* `[volume=`*formula*`] [units=u]`, or
`compartment` *name*`, `*name*`, …`

**Meaning** One number that changes in time. A compartment needs no declaration: its
equation `d_t(X) = …` declares it. Declare it to give it `units=` or a `volume=` (then its
value is a concentration, and [fluxes](#flux) into it are divided by the volume). Options are
for one compartment per line. See [Compartments](../guide/axes-fields.md#compartments).

**Example**

```tq
compartment P units=PFU
compartment Cart volume=V_art
```

**Python** `m.compartment("P", units="PFU")`; `m.compartment("Cart", volume=V_art)`

**Errors**

```text
a declaration has no value; the value at time 0 is written P(0) = 1e7.
units= and volume= are for one compartment per line.
write compartment names: compartment P (or compartment A, B).
```

### Named formula

**Syntax** *name* `=` *formula*, or *name*`(`*axis*`, …) =` *formula*

**Meaning** Names a formula, used by its name in any other formula as if written out. The
brackets on the left list the axes it varies over; they document it and are not function
arguments: use it as `h(a)` (the same axes, in the same order) or `h`. A formula may also be
a [conditional](#conditional-expressions) written over several lines
([cases brace](#cases-brace)) or set in an [`if` block](#if-elif-else-blocks). A formula
that is just a slice, `C_liver = C[organ=liver, sub=tissue]`, names that value. See
[Named formulas](../guide/axes-fields.md#named-formulas).

**Example**

```tq
ktr = 4/mtt
h(a) = h_max*sigmoid((a - tau)/s)
div  = g*B*(1 - N/K)
```

**Python** `m.define("ktr", 4/mtt)`; `h = m.define("h", h_max*tq.sigmoid((a - tau)/s))`

**Errors**

```text
'h(2)': h is a named formula, not a function; write h(a) (or just h).
'f(z, a)': f is a named formula, not a function; write f(a, z) (or just f).
'f(x)': the brackets list the axes the formula varies over, and 'x' is not an axis.
The name 'x' is already used by a definition; every name must be unique.
'd_z(…)' stands for a whole term of a field's equation, as in d_t(F) = … - d_a(v*F) or + trans_s(M, F); it cannot be part of a formula.
```

### Named total

**Syntax** *name* `=` `int_`*axis*`(…)` or `sum_`*axis*`(…)`, nested for several axes

**Meaning** A formula that is one integral (or sum) names that total: `B_tot` below is the
number of bacteria, varying over no axis. For a field with weights, the total is an amount.
See [Integrals and sums](../guide/integrals.md#integrals-and-sums).

**Example**

```tq
B_tot = int_z(sum_s(B))
A_body = sum_organ(sum_sub(C))
```

**Python** `B_tot = m.integral(B, over=[z, s], name="B_tot")`

**Errors** as for [`int_` and `sum_`](#int_axis-and-sum_axis).

### Equation `d_t`

**Syntax** `d_t(`*X*`) =` *right-hand side* `[using sharp [on` *axis*`, …]]`

**Meaning** The equation of a compartment or a field: how fast it changes. One per
compartment or field. For a compartment the right-hand side is any formula of its rate. For
a field the right-hand side is split into its terms (the parts joined by `+` and `-`) and
each is recognised: transport, diffusion, a loss, transitions or a reaction (the
[table under operators](#d_axis-transport-and-diffusion)). Boundary conditions are written
under it. See [How an equation is read](../guide/terms.md#how-an-equation-is-read).

**Example**

```tq
d_t(centr) = ka*depot - CL/V*centr
d_t(I) = -d_a(I) - h*I + d_z(Dz*d_z(I))            using sharp on a
```

**Python** `m.ode(centr, ka*depot - CL/V*centr)`; for a field, one call per term:
`m.advect(I, axis=a, velocity=1, scheme="sharp")`, `m.hazard(I, h)`,
`m.diffuse(I, axis=z, D=Dz)`.

**Errors**

```text
d_t(A) is given twice (also line 1); write one equation per field or slice, with all its terms.
d_t(A) = … needs a right-hand side.
d_t(…) appears on the right-hand side; write the rates themselves.
a compartment has no axis 'a'; to use a field's total, integrate it: int_a(…).
'A' is a parameter; give a compartment, a field or a slice of a field.
an equation is written d_t(X) = …, not d/dt(X).
```

and those of the terms ([transport](#d_axis-transport-and-diffusion),
[transitions](#trans_axis-and-trans_ab)).

### Equation on a slice

**Syntax** `d_t(`*F*`[`*axis*`=`*label*`, …]) =` *right-hand side*

**Meaning** Terms that act only on part of a field over categorical axes: the portal vein
only in the vascular space, hepatic clearance only in liver tissue. A slice equation adds to
the field's own equation; the field may have both. Inside it, a table parameter such as
`V[organ, sub]` takes the fixed label of the sliced axis. Transport and diffusion act on a
whole field and are refused here. See
[PBPK: exchange between tissue spaces](../guide/transitions.md#pbpk-exchange-between-tissue-spaces).

**Example**

```tq
d_t(C)                          = trans_sub(X, C)
d_t(C[sub=vascular])            = trans_organ(portal, C) + (J_art - J_ven)/V[organ, sub]
d_t(C[organ=liver, sub=tissue]) = -CLint*C/V[organ, sub]
```

**Python** the same calls with the slice, `C.at(sub="vascular")`:
`m.hazard(C.at(organ="liver", sub="tissue"), CLint/V[organ, sub])`

**Errors**

```text
transport and diffusion act on a whole field; write them in d_t(B).
a slice fixes categorical axes; 'z' is continuous.
axis 's' has no category 'X' (it has ['S', 'R']).
```

### Value at time 0

**Syntax** *X*`(0) =` *value*, or *F*`[`*axis*`=`*label*`, …](0) =` *value*

**Meaning** The starting value. `(0)` always means time 0. Anything not given starts at 0,
as in NONMEM and Monolix. For a field the value is:

| Value | Meaning |
|---|---|
| a formula with no shape, `1e7*exp(-z)` | a **density**, as written |
| an amount times a shape, `1e7*normal(z, mean=0.1, sd=0.08)` | the amount, **spread** like the shape; the cells hold exactly 1e7 |
| an amount times `shape(`*formula*`)` | the amount spread like any formula of the axes |
| `density(`*formula with a shape*`)`, `1e7*density(normal(z, …))` | the named curve itself as a density (not scaled to an amount) |
| shapes on several axes multiplied | the product is the shape |

A compartment takes a number or a formula of parameters. See
[Starting values](../guide/initial.md).

**Example**

```tq
P(0) = 1e7
B(0) = 1e7*exp(-z)
B[s=S](0) = 1e7*normal(z, mean=0.1, sd=0.08)
B[s=R](0) = 1e3*point(z, at=0)
```

**Python** `m.init(P, 1e7)`; `m.init(B, 1e7*tq.exp(-z))`;
`m.init(B, total=1e7, shape=tq.normal(z, mean=0.1, sd=0.08), where={s: "S"})`;
`m.init(B, total=1e7, shape=tq.exp(-z))` for `1e7*shape(exp(-z))`

**Errors**

```text
a value at time 0 needs a value: X(0) = value.
the value at time 0 of this part of A is already given (line 2).
a shape spreads an amount over an axis; a compartment has none.
write an amount times one shape, e.g. 1e7*normal(z, mean=0.1, sd=0.08), 1e7*point(z, at=0) or 1e7*shape(exp(-z)).
density(…) multiplies the rest of the value, e.g. 1e7*density(normal(z, mean=0.1, sd=0.08)).
shape(…) takes one formula of the axes, e.g. shape(exp(-z)).
a value at time 0 is written X(0) = value.
```

The last is the answer to `init A = 1`. Giving `B(0)` and `B[s=S](0)` together is the
"already given" error.

### flux

**Syntax** `flux` *name*`:` *from* `->` *to* `=` *amount per time*

**Meaning** Material leaving one place and arriving in another (blood flow, uptake). The two
ends are compartments or slices of fields. Use the name with a minus in the equation of
*from* and with a plus in that of *to*, divided by the volume (or weights) where the place
has one, summed (`sum_s(J)`) where the place has fewer axes. TenQSP checks that what leaves
arrives. Anywhere else the name is an ordinary value. Fluxes are optional: the same term
written in each equation gives the same model, and a process with more than two places
(B + P → I) can only be written that way. See
[Categories and transitions](../guide/transitions.md#pbpk-exchange-between-tissue-spaces) and
[Terms shared between equations](../guide/explain.md#terms-shared-between-equations).

**Example**

```tq
flux J_art: Cart -> C[sub=vascular] = Q_art[organ]*Cart
d_t(Cart) = (sum_organ(J_ven) - sum_organ(J_art))/V_art
```

**Python** `m.transfer(Cart, to=C.at(sub="vascular"), rate=Q_art[organ])` (the rate is the
flux divided by its source); the `±J` terms are not written in Python.

**Errors**

```text
write a flux as 'flux J: from -> to = amount per time', e.g. flux infect: B -> I = k*B*P.
flux J arrives in B: d_t(B) needs exactly one term +J.
flux J leaves A: the equation d_t(A) needs exactly one term -J.
flux J is an amount per time; in the equation of B divide it by B's volume, V2 (not V1), so that what leaves one place arrives in the other.
flux J: A has no volume or weights; use J without dividing.
flux J varies over ['s']; P varies over nothing, so it takes sum_s(J) here.
the flux J moves material in and out of this place, so it appears as -J where it leaves and +J where it arrives (divided by the volume, or summed over axes: sum_s(J)), not as '-(2 * J)'.
the flux J joins two places cell by cell, so it cannot be integrated over 'z' here: … A process with more participants, like B + P -> I, is written without an arrow: …
a flux out of A (a concentration) is a rate times what it takes from: write J = rate*A. …
flux 'J' leaves and arrives in the same place.
flux 'J' is declared twice.
'B' is not declared; give a compartment, a field or a slice of a field.
material moving between places is a flux: flux J: A -> B = rate*A, then - J in the equation of A and + J in the equation of B.
```

The last is the answer to a `transfer` statement, which does not exist.

### transitions

**Syntax**

```text
transitions name on axis: FROM -> TO = rate, FROM -> TO = rate, …
transitions name on axis:
    FROM -> TO = rate
    …
transitions name on (axis1, axis2): (A1, A2) -> (B1, B2) = rate
```

**Meaning** Declares the moves between the categories of a categorical axis (or, jointly,
of several), with their rates. Only the moves are declared: what leaves the origin is
derived, so transitions neither create nor lose material. They act through the term
[`trans_s(M, X)`](#trans_axis-and-trans_ab) in an equation. Rates are formulas
(`PS/Kp[organ]`). See [Declaring transitions](../guide/transitions.md#declaring-transitions)
and [Joint transitions](../guide/transitions.md#joint-transitions-on-two-axes).

**Example**

```tq
transitions M on s: S -> R = mu, R -> S = mu
transitions X on (r1, r2): (S, S) -> (R, R) = mux
```

**Python** `M = m.transitions("M", axis=s, rates={"S->R": mu, "R->S": mu})`;
`m.transitions("X", axis=[r1, r2], rates={"S,S->R,R": mux})`

**Errors**

```text
write transitions as 'transitions M on s: S -> R = mu, R -> S = nu' (moves on one line separated by commas, or one per indented line).
'S to R': write each move as FROM -> TO = rate.
the move S -> R is given twice.
axis 's' has no category 'X' (it has ['S', 'R']).
'S': on (s, r) a category is a label per axis, e.g. (S, S).
transitions 'M': 'z' is a continuous axis; a categorical axis is needed.
```

### Boundary condition `at`

**Syntax**, written indented right under the equation `d_t(F) = …` of a whole field:

```text
at axis=end: flux of F along axis = formula
at axis=end: F = formula
at axis=end: d_axis(F) = 0
at axis=end: outflow
```

*end* is the coordinate of the end (`z=0`, `a=2`) or `lower` / `upper`.

**Meaning** What happens at one end of one axis of the field, as a paper writes it:

| Form | Meaning | Python |
|---|---|---|
| `at z=0: -Dz*d_z(D) = alpha*B - beta*D` | a **flux condition**: the left side is the flux of `F` along the axis as its equation's transport gives it (`v*I` for a flow, `-Dz*d_z(D)` for diffusion, `v*D - Dz*d_z(D)` for both); positive toward larger coordinates, so at the lower end a positive flux enters. On the right, `F` is its value at that end; the formula must be linear in it | `m.at(D, z=0, flux=alpha*B - beta*D)` |
| `at a=0: v*I = k*B*P` | renewal: cells entering at age 0 (a flux condition on a flow) | `m.at(I, a=0, flux=k*B*P)` |
| `at z=1: C = 1` | a **held value** | `m.at(C, z=1, value=1)` |
| `at z=0: d_z(C) = 0` (or the flux `= 0`) | **no flux**: the default for an end not mentioned | `m.at(C, z=0, flux=0)` |
| `at a=2: outflow` | what the flow carries to this end leaves the model (in the ledger); required at the end a `one_way` axis flows to | `m.at(I, a=2, outflow=True)` |

A speed of exactly 1 is written `1*I = …` in a flux condition, since `I = …` is a held
value. Another formula reads the field at an end as [`F(x=0)`](#a-field-at-an-end).
See [Boundary conditions](../guide/boundaries.md).

**Example**

```tq
d_t(D) = d_z(Dz*d_z(D)) - kD*D
    at z=0: -Dz*d_z(D) = alpha*B - beta*D
d_t(I) = -d_l(v*I) - mu*I
    at l=0: v*I = k*P*B
    at l=upper: outflow
```

**Errors**

```text
a boundary condition 'at z=…: …' belongs to an equation: write it right below d_t(F) = … (other conditions of the same equation may come between).
a boundary condition is for a field's equation (the ends of its axes); 'A' is not a whole field.
the left side is the flux of D along z, from its equation: -Dz*d_z(D) (not -2*Dz*d_z(D)).
'h*D^2' is not linear in D (its value at this end): a boundary condition is written as a flux equal to terms that are each proportional to D or free of it, e.g. -D*d_z(D) = h*(Cout - D).
D does not move along z (no -d_z(v*D) or d_z(D*d_z(D)) in its equation), so it has no flux there to set.
write a gradient condition as the flux it gives: -Dz*d_z(D) = …
d_z(D) = … fixes the gradient, which is a flux only for a field that diffuses without flowing along z; write the flux: D - Dz*d_z(D) = …
a condition is an equation: 'at z=0: <flux of D> = …', 'D = <value>', or 'outflow'.
'a' is not an axis of D.
at: z=0.5 is not an end of 'z' (its ends are 0 and 1).
this end already has a boundary condition.
```

and, for a `one_way` axis whose upper end is not stated (reported at the field's
declaration):

```text
the flow carries I to the upper end of 'a'; say what happens there: 'at a=<end>: outflow' (it leaves the model, recorded in the ledger; read it elsewhere as I(a=upper)), or a condition.
```

### Older end statements: inflow, outflow, dirichlet, robin, zero_flux

**Syntax**

```text
inflow F at axis=lower|upper: formula
outflow F at axis=lower|upper [to L]
dirichlet F at axis=lower|upper: formula [with M]
robin F at axis=lower|upper: h*(outside - F) [with M]
zero_flux F at axis=lower|upper
```

**Meaning** The one-line form of boundary conditions, still read so older models keep
working; `m.to_text()` writes the `at` form wherever it says the same. These stand anywhere
in the text, and the end is `lower` or `upper` only. `inflow` is a flux entering at that end;
`outflow … to L` sends what leaves into compartment `L`; `robin` is an exchange with an
outside level, `with M` taking it from (and giving it to) compartment `M`. See
[Older statements](../guide/boundaries.md#older-statements).

**Example**

```tq
inflow  I at a=lower: k*B*P
outflow I at a=upper to L
dirichlet C at z=upper: 1
robin E at z=upper: h*(M/V - E) with M
zero_flux C at z=lower
```

**Python** `m.inflow(I, axis=a, at="lower", flux=k*B*P)`;
`m.outflow(I, axis=a, at="upper", to=L)`;
`m.boundary(C, axis=z, at="upper", kind="dirichlet", value=1)`;
`m.boundary(E, axis=z, at="upper", kind="robin", value=M/V, h=h, to=M)`;
`m.boundary(C, axis=z, at="lower", kind="zero_flux")`

**Errors**

```text
write a boundary as e.g. 'inflow I at a=lower: k*B*P', 'outflow I at a=upper [to L]', 'dirichlet C at z=upper: 1', 'robin C at z=upper: h*(M/V - C) [with M]' or 'zero_flux C at z=lower'.
'A' is not a field; boundaries are ends of a field's axis.
write the exchange as h*(outside - C): a coefficient times the outside level minus C.
an inflow is not taken from a compartment with 'with'; to take it from another place, write it as a flux.
a zero_flux end takes nothing more.
```

### dose

**Syntax**

```text
dose X at t=t1, t2, …: amount
dose X at t=t1, t2, …: amount1, amount2, …
dose X at t=start to end every interval: amount
dose F[axis=label, …] at t=…: amount
dose F at t=…: amount*shape
dose F at t=…: density(formula)
```

**Meaning** Bolus doses written in the model: its default regimen. A regimen given at
simulation replaces these, never adds to them. Times are numbers ≥ 0; amounts are formulas
(a parameter such as `D0` is fine), one for all times or one per time. `start to end every
interval` is a regimen like NONMEM's II/ADDL; the last dose falls on `end`, so the span must
be a whole number of intervals. Several `dose` lines add up.

- Into a **compartment**: the amount (with a volume, the value rises by amount ÷ volume).
- Into **one element** of a field: a slice that fixes every axis.
- Into a field over **continuous** axes (categorical axes fixed by a slice): an amount times a
  shape, as for a [value at time 0](#value-at-time-0); or `density(…)`, a density added
  to the field. A bare number is refused: say where it goes.

See [Doses written in the model](../guide/doses.md#doses-written-in-the-model).

**Example**

```tq
dose P at t=0, 24: 1e8
dose P at t=0, 24: 1e8, 5e7
dose depot at t=0 to 144 every 24: 100
dose C[organ=gut, sub=vascular] at t=0: 100
dose G at t=0, 12: 100*point(g, at=0)
dose B at t=0: density(1e6*exp(-z))
```

**Python** `m.dose(P, times=[0, 24], amounts=1e8)`; `amounts=[1e8, 5e7]`; the regimen's
times written out, `times=[0, 24, 48, 72, 96, 120, 144]`;
`m.dose(C.at(organ="gut", sub="vascular"), times=[0], amounts=100)`;
`m.dose(G, times=[0, 12], amounts=100, shape=tq.point(g, at=0))`;
`m.dose(B, times=[0], density=1e6*tq.exp(-z))`

**Errors**

```text
write a dose as 'dose P at t=0, 24: 1e8' or 'dose P at t=0 to 144 every 24: 1e8'.
from 0 to 140 is not a whole number of steps of 24; the last dose must fall on the end time.
the interval after 'every' must be > 0.
3 dose time(s) but 2 amounts: give one amount for all, or one per time.
a dose into the field B needs a shape saying where the amount goes along ['z'], e.g. 1e6*point(z, at=0); or density(1e6) to add 1e6 per unit of z everywhere.
a dose into B goes into one category of s: name it, e.g. dose B[s=S] at …
the doses of one line share one shape (or are all densities); write a line per shape.
dose times must be >= 0.
'Q' is not declared; give a compartment, a field or a slice of a field.
```

### lag

**Syntax** `lag` *place* `=` *formula*

**Meaning** Every dose into the place (a compartment, or a whole field) arrives *formula*
later, as NONMEM's ALAG; an infusion is shifted whole. The formula uses parameters only and
must be ≥ 0. It applies to the model's own doses and to a regimen's. See
[Lag time and bioavailability](../guide/doses.md#lag-time-and-bioavailability).

**Example**

```tq
param tlag=0.5
lag depot = tlag
```

**Python** `m.lag(depot, "tlag")`

**Errors**

```text
write it as 'lag depot = formula' (a place, then '=' and a formula of parameters), e.g. 'lag depot = tlag' or 'bioavailability depot = F'.
a lag applies to a whole place; write its name only.
a lag uses 'P' (a compartment); it may use parameters and coordinates only.
lag: 'P' already has a lag; give one per place.
```

### bioavailability

**Syntax** `bioavailability` *place* `=` *formula*

**Meaning** Only the fraction *formula* of every dose into the place arrives, as NONMEM's
F: bolus amounts and infusion rates are scaled; the ledger books the rest as "not
absorbed". A formula of parameters, ≥ 0 (above 1 is allowed).

**Example**

```tq
param F=0.8
bioavailability depot = F
```

**Python** `m.bioavailability(depot, "F")`

**Errors** as for `lag`, and

```text
bioavailability: 'P' already has a bioavailability; give one per place.
```

### observable

**Syntax** `observable` *name* `=` *formula*

**Meaning** A named output, `sol["name"]`: any formula of the model's quantities, a total,
or a slice of a field. An observable of a field without a total varies over the field's axes.
See [Observables](../guide/axes-fields.md#observables).

**Example**

```tq
observable conc = centr/V
observable bacteria = int_z(sum_s(B))
observable C_liver = C[organ=liver, sub=tissue]
```

**Python** `m.observable("conc", centr/V)`;
`m.observable("C_liver", C.at(organ="liver", sub="tissue"))`

**Errors**

```text
write an output as 'observable name = formula'.
The name 'A' is already used by a compartment; every name must be unique.
```

### extinction

**Syntax** `extinction [`*name*`:]` *member*`, … below` *threshold* `[with noise_floor]`

**Meaning** A lineage group (compartments, fields, slices of fields) that is set to zero
when its total falls below the threshold after having been above it. A group that never
reached the threshold (mutants seeded from zero) is never touched. Without a name, the group
is named from its members (`extinct_P`, `extinct_B_s_R`). `with noise_floor`: while below
the threshold, every equation sees the group as zero, so round-off "ghost" amounts cannot
grow by feeding themselves. The threshold is a number. See
[Extinction groups and noise floors](../guide/numerics.md#extinction-groups-and-noise-floors).

**Example**

```tq
extinction B[s=R] below 1
extinction phage: P, I below 1 with noise_floor
```

**Python** `m.extinction(B.at(s="R"), threshold=1)`;
`m.extinction([P, I], threshold=1, name="phage", noise_floor=True)`

**Errors**

```text
write an extinction group as 'extinction [name:] P, I below 1 [with noise_floor]'.
threshold must be a number, not 'th'. …
The name 'p' is already used by an extinction group; every name must be unique.
```

```text
line 2: extinction P below -1
threshold must be > 0.
```

### if, elif, else blocks

**Syntax**

```text
if condition:
    name = formula
    …
elif condition:
    name = formula
    …
else:
    name = formula
    …
```

**Meaning** Conditional named formulas, for code ported from NONMEM or Monolix. The first
true condition wins; `else:` is required, so every name always has a value. Every branch
sets the same names, each once; each name becomes one conditional formula. A block only
names formulas: equations are never inside one, and blocks do not nest (join conditions with
`and`). See [Conditionals](../guide/terms.md#conditionals).

**Example**

```tq
if C > MIC:
    k = k1
    E = Emax
elif C > MIC/2:
    k = k2
    E = Emax/2
else:
    k = k3
    E = 0
```

**Python** `k = m.define("k", tq.cases((C > MIC, k1), (C > MIC/2, k2), otherwise=k3))`, and
the same for `E`.

**Errors**

```text
an if block ends with 'else:' (the values when no condition holds), so that every name always has a value.
every branch must set the same names: the first sets ['k'], the branch else sets ['j'].
line 2: put each assignment on its own indented line.
line 3: an if block inside another is not read; join the conditions with 'and' instead.
line 3: an if block only names formulas (name = value); equations d_t(X) = … are never inside one.
line 4: 'k' is set twice in one branch.
'elif' without an 'if' block above it.
```

### Cases brace

**Syntax**

```text
name = value if condition
       value if condition
       …
       value otherwise
```

**Meaning** A conditional formula written as a paper's cases brace. The cases are tried in
order and the first true one wins; the last line is `value otherwise`. The same formula as
`k1 if C > MIC else k2 if C > MIC/2 else k3`.

**Example**

```tq
k = k1   if C > MIC
    k2   if C > MIC/2
    k3   otherwise
```

**Python** `m.define("k", tq.cases((C > MIC, k1), (C > MIC/2, k2), otherwise=k3))`

**Errors**

```text
the last case must be 'value otherwise' (a value when no condition holds).
'2 otherwise': each case is 'value if condition', and the last one 'value otherwise'.
```

### using sharp

**Syntax** at the end of a field's equation: `using sharp`, or `using sharp on` *axis*`, …`

**Meaning** Transport uses the `erlang` scheme by default: each cell is one transit
compartment, so the grid sets the spread. `using sharp` switches every transport term of the
equation to the `sharp` scheme (sharper fronts on the same grid); `using sharp on a` only
the transport along `a`. See [Transport schemes](../guide/terms.md#transport-schemes-erlang-or-sharp).

**Example**

```tq
d_t(G) = -d_z(G/MTT)  using sharp
d_t(I) = -d_a(I) - d_x(0.1*I)  using sharp on a, x
```

**Python** `m.advect(G, axis=z, velocity=1/MTT, scheme="sharp")`

**Errors**

```text
a compartment has no axis to move along: 'using sharp' is for fields.
'using sharp' is for an equation with a transport term -d_a(v*F).
'using sharp on z': this equation moves I along ['a'].
```

## Operators

### `d_t`: the time derivative

`d_t(X)` appears only on the left of an [equation](#equation-d_t). On a right-hand side it
is refused (`d_t(…) appears on the right-hand side; write the rates themselves.`). Time
itself is the name `t`.

### `d_<axis>`: transport and diffusion

`d_` followed by a continuous axis, `d_a(…)`, is a derivative along that axis. It appears
only as a whole term of a field's equation (or on the left of a boundary condition), never
inside a formula. Each term of `d_t(F) = …` is recognised as follows:

| Term | Read as | Python |
|---|---|---|
| `-d_a(v*F)` | **transport** of `F` along `a` at velocity `v` (conservation form; `v` may change sign, depend on time, on fields, on `F` itself) | `m.advect(F, axis=a, velocity=v)` |
| `-d_a(F)` | transport at velocity 1 | `m.advect(F, axis=a, velocity=1)` |
| `d_x(D*d_x(F))` | **diffusion** with diffusivity `D` | `m.diffuse(F, axis=x, D=D)` |
| `-r*F`, `r` free of `F` | a **loss** at per-capita rate `r`, its own line in the ledger | `m.hazard(F, r)` |
| `trans_s(M, X)` | **transitions** of `X` along `s` | `m.transition(F, axis=s, matrix=M, of=X)` |
| `±J`, `J` a [flux](#flux) of this place | the flux | `m.transfer(…)` |
| anything else | a **reaction**, added cell by cell | `m.reaction(F, …)` |

Transport plus diffusion along the same axis is a Fokker–Planck equation; see
[Transport plus diffusion](../guide/terms.md#transport-plus-diffusion-a-fokkerplanck-equation).

Errors:

```text
the transport term must be in conservation form: write -d_a(v*I), not -v*d_a(I) (TenQSP keeps every cell's books balanced only in this form).
transport is written with a minus, -d_a(v*I); to move toward lower a, give a negative velocity: -d_a(-v*I).
diffusion is written +d_z(D*d_z(F)); a negative diffusion has no solution.
in the equation of I, transport moves I: write -d_a(v*I) (this one moves J).
write diffusion as d_z(D*d_z(I)), with the diffusivity D.
's' is categorical: there is no d_s; moves between its categories are written + trans_s(M, B) with 'transitions M on s: …'.
```

### `int_<axis>` and `sum_<axis>`

`int_z(f)` integrates the formula `f` over the continuous axis `z`; `sum_s(f)` adds it up
over the categorical axis `s`. Nest them for several axes: `int_z(sum_s(B))`. The result
varies over the remaining axes. Each becomes a named total (the same integral written twice
is stored once); for a field with weights the total is an amount. They may stand in any
formula. See [Integrals and sums](../guide/integrals.md#integrals-and-sums).

Python: `m.integral(B, over=[z, s])`.

```text
's' is categorical: add over it with sum_s(…).
'z' is continuous: integrate over it with int_z(…).
int_y: 'y' is not an axis.
int_z(…) takes one formula.
```

### `trans_<axis>` and `trans_{a,b}`

`trans_s(M, X)` is a term of a field's equation: it moves `X` between the categories of `s`
with the rates of [`transitions M on s`](#transitions). With `X` the field itself, cells
**switch** category; with `X` a rate of production (`div`), a fraction of that production
ends up in another category, at the **parent's** rate (mutation at division). Joint
transitions on two axes are `trans_{r1,r2}(X, div)`; `trans_s` is short for `trans_{s}`.
Written out, in category `d`:

$$
\sum_{o \ne d} M_{d,o} X_o \;-\; \Big(\sum_{d' \ne d} M_{d',d}\Big) X_d
$$

Python: `m.transition(B, axis=s, matrix=M, of=B)` (switching) or `of=div` (production). See
[Categories and transitions](../guide/transitions.md).

```text
a transition term is added, + trans_s(M, X): what leaves one category is derived from the moves.
'M' is not declared with 'transitions M on …'.
'M' moves along r, not s: write trans_r(M, …).
write trans_s(M, X) as a term of its own: + trans_s(M, X).
```

### `grad_<axis>` and `grad(F, x)`

`grad_x(S)` (or `grad(S, x)`) is the slope of the field `S` along `x`, taken at the cell
faces. It is allowed only inside a transport velocity along `x`, e.g. chemotaxis
`-d_x(chi*grad_x(S)*B)`. Python: `m.advect(B, axis=x, velocity=chi*tq.grad(S, x))`. See
[Slopes](../guide/integrals.md#slopes-grad_xs-only-inside-a-velocity).

```text
grad(S, z) (grad_z(S)) can only be used in a velocity along 'z' (gradients are taken at the cell faces, where velocities live); write it inside the transport term, e.g. -d_z(chi*grad_z(S)*B) (in Python, m.advect(B, axis=z, velocity=chi*tq.grad(S, z))).
grad_z(…) takes one field: grad_z(S).
```

### A field at an end

`F(x=0)`, `F(x=lower)`, `F(x=upper)`, or the coordinate of either end: the field `F` read at
one end of its continuous axis `x`, the value the boundary condition there gives (the level
an exchange sees, the value held, the edge value an outflow carries out). Usable in any
formula, so what crosses an end and what another equation reads are the same number.
Python: `F(x=0)`, `I(a="upper")`. See [A field read at an end](../guide/integrals.md#a-field-read-at-an-end).

```tq
d_t(L) = v*I(a=upper) - L
```

```text
'D(z=0.5)': 0.5 is not an end of axis 'z' at column 1 of 'D(z=0.5)'; write D(z=lower) or D(z=upper).
```

### Slices `F[s=R]`

`C[organ=liver, sub=tissue]` is the field `C` with those categorical axes fixed. Square
brackets with `=` are a slice. A slice may be read in any formula, be the target of an
equation, a value at time 0, a dose, an end of a flux, an observable, or a member of an
extinction group. Labels in quotes keep their quotes: `B[s="wild type"]`. Python:
`C.at(organ="liver", sub="tissue")`. See
[One category in a formula](../guide/integrals.md#one-category-in-a-formula-slices).

```text
'C[o=a, s]': a slice fixes axes by label, e.g. C[organ=liver]; every entry needs axis=label.
'A' is a compartment; only a field can be sliced.
'B' has no axis 'y' (it has ['z']).
```

### Per-category parameters `k[s]`

Square brackets **without** `=` index a parameter declared over categorical axes:
`k[s]` is the value for each category of `s`, `V[organ, sub]` one value per combination,
indices in the declared order.

```text
'k' has one value per level of axis 's'; write 'k[s]', not 'k[r]'.
Only vector parameters can be indexed, 'B' is not one at column 1 of 'B[s]'.
```

### `t`: time

`t` is time, in any formula: `kin if t > t_on else 0`. A condition on time and parameters
only is found before solving and the solve restarts exactly there.

### Arithmetic and comparison

In order of increasing binding strength:

| Operators | Meaning |
|---|---|
| `a if c else b` | conditional (lowest) |
| `or` | either condition |
| `and` | both conditions |
| `not` | negation |
| `< <= > >= == !=` | comparisons; chains `0 < x <= 1` mean `0 < x and x <= 1` |
| `+ -` | sum, difference |
| `* /` | product, quotient |
| unary `- +` | sign |
| `^` or `**` | power; `-x^2` is `-(x^2)`, `2^-1` is 0.5 |

`^` and `**` both mean power, as in NONMEM and Monolix. A condition is not a number:
`k*(x > 1)` is refused; write `k if x > 1 else 0`.

### Conditional expressions

`a if c else b`, chained `k1 if x > t1 else k2 if x > t2 else k3`: the first true condition
wins, and the final value after the last `else` is required. Conditions are comparisons
joined by `and`, `or`, `not` and brackets; `&`, `|`, `~` are refused. A value that is not
chosen cannot spoil derivatives: `log(P) if P > 0 else 0` is safe at `P = 0`. Python:
`tq.cases((x > t1, k1), (x > t2, k2), otherwise=k3)`; in Python, `&`, `|`, `~` join
conditions and `tq.eq(a, b)`, `tq.ne(a, b)` stand for `==` and `!=`. See
[Conditionals](../guide/terms.md#conditionals).

```text
'x > 1' is a condition (true or false), not a number at column 5 of '-k*(x > 1)*A'. Use it to choose between values: 'v1 if x > 1 else v2'.
After 'if' comes a condition such as 'x > 1' (comparisons joined by and, or, not), not 'x' at column 8 of '-(x if x else 0)*A'.
'&' is not allowed; join conditions with 'and' (x > a and x < b) at column 12 of '-(1 if x > 1 & y < 2 else 0)*A'.
'|' is not allowed; join conditions with 'or' (x < a or x > b) at column 12 of '-(1 if x > 1 | y < 2 else 0)*A'.
'~' is not allowed; write 'not' at column 8 of '-(1 if ~(x > 1) else 0)*A'.
'x in 2': only < <= > >= == != compare numbers at column 8 of '-(1 if x in 2 else 0)*A'.
```

## Functions

Every function acts cell by cell on numbers. Functions take no keyword arguments. Every
function has a Python helper of the same name (`tq.exp`, `tq.normcdf`, `tq.sigmoid`).

Each function has one name. The column *Elsewhere* lists names other programs use for it;
TenQSP refuses them and names the function to write instead, e.g.
`'pnorm' is not a TenQSP function …: write normcdf(...) (the same function as R's pnorm).`

| Function | Meaning | Elsewhere | NONMEM | Monolix |
|---|---|---|---|---|
| `exp(x)` | exponential | | `EXP` | `exp` |
| `log(x)` | natural logarithm | | `LOG` | `log` |
| `log10(x)` | base-10 logarithm | | `LOG10` | `log10` |
| `sqrt(x)` | square root | | `SQRT` | `sqrt` |
| `abs(x)` | absolute value | | `ABS` | `abs` |
| `max(a, b, …)`, `min(a, b, …)` | largest, smallest; at least 2 arguments | | `MAX MIN` (7.3) | `max min` |
| `sin cos tan` | trigonometric (radians) | | `SIN COS TAN` | same |
| `asin acos atan` | inverse trigonometric | | `ASIN ACOS ATAN` | same |
| `atan2(y, x)` | angle of the point (x, y) | | | `atan2` |
| `sinh cosh tanh` | hyperbolic | | | same |
| `logit(p)` | `log(p/(1-p))` | | | `logit` |
| `sigmoid(x)` | `1/(1+exp(-x))`, the inverse of `logit` | `invlogit`, `expit` | | `invlogit` |
| `normcdf(x)` | standard normal cumulative probability | R `pnorm` | `PHI` | `normcdf` |
| `probit(p)` | its inverse (accurate in the tails) | R `qnorm`, MATLAB `norminv` | | `probit` |
| `erf(x)` | error function | | | |
| `gammaln(x)` | log gamma | R `lgamma` | `GAMLN` (7.3) | `gammaln` |
| `factorial(x)` | `gamma(x+1)` | | | `factorial` |
| `factln(x)` | `gammaln(x+1)` | | | `factln` |
| `floor(x)`, `ceil(x)` | round down, up (a jump: the validator notes it) | | `INT` truncates | `floor ceil` |
| `rem(a, b)` | remainder, with the sign of `a` (a jump: noted) | `mod` | `MOD` (7.3) | `rem` |
| `grad(F, x)` | slope of field `F` along `x`; only in a velocity along `x` ([above](#grad_axis-and-gradf-x)) | `grad_x(F)` | | |

`atan2` and `rem` take exactly 2 arguments, `max` and `min` at least 2, the others
exactly 1.

```text
Function 'besselj' is not allowed at column 2 of '-besselj(x)*A'. Allowed: abs, acos, asin, atan, atan2, beta, ceil, cos, cosh, erf, exp, exponential, factln, factorial, floor, gamma, gammaln, grad, log, log10, logit, lognormal, max, min, normal, normcdf, point, probit, rem, sigmoid, sin, sinh, sqrt, tan, tanh, uniform.
'pnorm' is not a TenQSP function at column 2 of '-pnorm(x)*A': write normcdf(...) (the same function as R's pnorm).
max() takes at least 2 arguments, not 1 at column 2 of '-max(x)*A'.
atan2() takes exactly 2 arguments, not 1 at column 2 of '-atan2(x)*A'.
exp() takes no keyword arguments at column 2 of '-exp(x=1)*A'.
'exp' is a function; call it as exp(...) at column 2 of '-exp*A'.
'np.exp' cannot be called at column 2 of '-np.exp(1)*A'.
```

### Shape functions

Shapes describe how an amount is spread along an axis, in a
[value at time 0](#value-at-time-0) or a [dose](#dose) into a field. The first argument is
the axis; the rest are keywords, in one of the accepted combinations. A shape is
integrated exactly over each cell and cut to the axis.

| Shape | Keyword combinations |
|---|---|
| `normal(z, …)` | `mean` with `sd`, `var` or `cv` |
| `lognormal(z, …)` | `meanlog` with `sdlog` or `cv`; `median` with `sdlog` or `cv`; `mean` with `sdlog` or `cv` |
| `gamma(z, …)` | `shape` with `rate` or `scale`; `mean` with `sd`, `var` or `cv` |
| `exponential(z, …)` | `rate`, or `mean` |
| `beta(z, …)` (on [0, 1]) | `alpha` and `beta`; `mean` with `concentration`, `var` or `sd` |
| `uniform(z, …)` | nothing (the whole axis), `lower`, `upper`, or both |
| `point(z, at=…)` | `at`: everything in the cell containing it (on a face, the cell above; at the upper end, the last cell); a point outside the axis is refused |
| `shape(f)` | any formula `f` of the axes, used as a shape |

How a value with a shape is read (axis `z` in [0, 1], 50 cells; the totals were computed by
TenQSP):

| Written | Means | Total on the axis |
|---|---|---|
| `1e7*exp(-z)` | a density, as written | 6.32e6 |
| `1e7*shape(exp(-z))` | 1e7 spread like exp(−z) | exactly 1e7 |
| `1e7*normal(z, mean=0.9, sd=0.2)` | 1e7 spread like the normal, cut to the axis | exactly 1e7 |
| `1e7*density(normal(z, mean=0.9, sd=0.2))` | the normal curve itself, times 1e7 | 6.91e6 (the part of the curve inside [0, 1]) |
| `1e7*point(z, at=0)` | all of 1e7 in the first cell | exactly 1e7 |

`density(…)` is needed only around a named shape: a formula with no shape is already a
density. `beta` and `gamma` remain usable as parameter names (`param beta=50`); called with
an axis, they are the shapes. In Python: `tq.normal(z, mean=0.1, sd=0.08)` and the others,
passed as `shape=`.

```text
normal: the arguments ['mean'] are not a valid combination at column 1 of 'normal(z, mean=0.5)'. Give exactly one location and one spread argument. Accepted: normal(x, mean=..., sd=...); normal(x, mean=..., var=...); normal(x, mean=..., cv=...).
```

## Reserved words and names

| Name | Why it cannot be used | Message |
|---|---|---|
| `t` | it means time | `parameter name 't' is reserved (it means time).` |
| Python keywords: `if`, `else`, `and`, `or`, `not`, `in`, `is`, `lambda`, `None`, `True`, `False`, `for`, `class`, … | reserved words of the formula language | `parameter name 'lambda' is a reserved word of the formula language; choose another (e.g. 'lambda_').` |
| a name already used, of any kind | names are unique | `The name 'z' is already used by an axis; every name must be unique.` |
| non-ASCII or starting with a digit (`α`, `1abc`) | not a name | `'α=1': write name=value (or name[axis] = {…}), or the name alone.` |
| a statement keyword as a **named formula**: `model`, `axis`, `param`, `field`, `compartment`, `flux`, `transitions`, `observable`, `dose`, `extinction`, `inflow`, `outflow`, `dirichlet`, `robin`, `zero_flux` | a line starting with one of these words is a statement | `'dose' starts a statement, so it cannot name a formula; choose another name.` |

These names are **not** reserved:

- **Function names.** A field named `erf` or a formula named `exp` is fine; `erf(x)` with
  arguments is still the function. `param beta=50` and `param gamma=0.17` are common and fine.
- The words `lower`, `upper`, `at`, `lag`, `bioavailability`, `otherwise`, `every`, `using`,
  `sharp`: `lag = 2*x` is a named formula (`lag depot = …`, with a place, is the statement).
- Names that start like an operator, `d_x` or `int_y` as a parameter: an operator is
  recognised only when it is called, `d_x(…)`. They are best avoided all the same, since a
  reader of the model will take them for operators.

## What is refused, and why

The common mistakes, with the message (shortened) and the correct form. Every message comes
with its line, and with a caret under the column where TenQSP can point at it:

```text
line 2: d_t(A) = -k*(x > 1)*A
                     ^
'x > 1' is a condition (true or false), not a number at column 5 of '-k*(x > 1)*A'. …
```

| Written | Message (shortened) | Instead |
|---|---|---|
| `d_t(I) = -v*d_a(I)` | the transport term must be in conservation form | `-d_a(v*I)` |
| `d_t(I) = d_a(v*I)` | transport is written with a minus; to move toward lower `a`, give a negative velocity | `-d_a(-v*I)` |
| `-d_x(D*d_x(F))` | diffusion is written `+d_x(D*d_x(F))`; a negative diffusion has no solution | `+d_x(D*d_x(F))` |
| `-d_a(v*J)` in the equation of `I` | in the equation of I, transport moves I | `-d_a(v*I)` |
| `d_a(d_a(I))` | write diffusion as `d_a(D*d_a(I))`, with the diffusivity D | `d_a(D*d_a(I))` |
| `d_a(I)` in a compartment's equation | a compartment has no axis 'a'; integrate it | `int_a(…)` |
| `d_t(I)` given twice | write one equation per field or slice | one equation with all terms |
| `h(2)` | h is a named formula, not a function | `h(a)` or `h` |
| `compartment P = 1e7` | a declaration has no value | `P(0) = 1e7` |
| `init A = 1` | a value at time 0 is written X(0) = value | `A(0) = 1` |
| `d/dt(A) = …` | an equation is written d_t(X) = … | `d_t(A) = …` |
| `k <- 2` | assignments use '=', not '<-' | `k = 2` |
| `transfer A -> B` | material moving between places is a flux | `flux J: A -> B = k*A` |
| `trans_z(M, B)` with `z` continuous | 'z' is a continuous axis; a categorical axis is needed | `-d_z(…)` |
| `trans_s(M, B)` with `M` on `r` | 'M' moves along r, not s | `trans_r(M, B)` |
| `-trans_s(M, B)` | a transition term is added | `+ trans_s(M, B)` |
| `flux J: A -> B` without `+J` in `B` | flux J arrives in B: d_t(B) needs exactly one term +J | add `+ J` |
| a flux whose volumes do not match | divide it by B's volume, V2 (not V1) | divide by the place's own volume |
| `flux J: P -> I` used as `-int_z(J)` | an arrow has exactly two ends that match element by element | the term in each equation, no `flux` |
| `at z=0: -2*Dz*d_z(D) = …` | the left side is the flux of D along z, from its equation | `-Dz*d_z(D) = …` |
| `at z=0: …` with no equation above | a boundary condition belongs to an equation | write it under `d_t(D) = …` |
| `at z=0: -Dz*d_z(D) = h*D^2` | not linear in D | terms proportional to `D` or free of it |
| a `one_way` axis whose upper end is not stated | the flow carries I to the upper end of 'a'; say what happens there | `at a=2: outflow` |
| `k*(x > 1)` | a condition, not a number | `k if x > 1 else 0` |
| `x if x else 0` | after 'if' comes a condition | `x if x > 0 else 0` |
| `x > 1 & y < 2` | join conditions with 'and' | `x > 1 and y < 2` |
| `besselj(x)` | function not allowed (list given) | — |
| `k[t]`, `k[r]` | 'k' has one value per level of axis 's' | `k[s]` |
| `grad(S, x)` in a reaction | only in an advection velocity along x | `-d_x(chi*grad(S, x)*B)` |
| `'text'`, `[1, 2]`, `lambda: 1` | not allowed in a formula | a parameter or a named formula |
| `g_mx` (typo) | Unknown name 'g_mx'. Did you mean 'g_max'? | `g_max` |
| `param k=2*j` | must be a number; name a formula instead | `k = 2*j` |
| `dose P at t=0 to 140 every 24: …` | not a whole number of steps of 24 | `0 to 144 every 24` |
| `dose B at t=0: 1e6`, `B` over `z` | needs a shape saying where the amount goes | `1e6*point(z, at=0)` or `density(1e6)` |
| `hello world` | cannot read this statement (the list of statement forms) | — |

## Complete models

### A tour of the statements

One model that uses most statements at once: two continuous axes and a categorical one,
parameters with units, without a value and per category, fields and a compartment, named
formulas and totals, a cases brace, transitions, transport, diffusion and a renewal
condition, a value at time 0 with a shape, a regimen with a lag and a bioavailability, an
observable on a slice, and an extinction group with a noise floor. Phage are dosed into a
`depot` and reach the bacteria from there.

```tq
model notation_tour

axis a in [0, 1.5] n=15 units=h kind=one_way      # infection age
axis z in [0, 1] n=10                              # dormancy level
axis s categorical [S, R]                          # susceptibility

param g=1, K=1e9, beta=50, delta=0.1, h_max=5, tau=0.8, s_h=0.05,
      Dz=0.01, mu=1e-6, ka=1.2
param k[s] = {S: 1e-9, R: 1e-12}
param tlag = 0.5 units=h
param F                                            # no value: given when solving

field B(z, s) units=cells                          # bacteria
field I(a)                                         # infected cells
compartment P units=PFU                            # free phage

N      = int_z(sum_s(B))
div    = g*(1 - z)*B*(1 - N/K)
h(a)   = h_max*sigmoid((a - tau)/s_h)
infect = k[s]*B*P
crowd  = 1   if N < K/10
         0.5 if N < K/2
         0   otherwise

transitions M on s: S -> R = mu, R -> S = mu

d_t(B) = crowd*div - infect + trans_s(M, div) + d_z(Dz*d_z(B))
d_t(I) = -d_a(I) - h*I                             using sharp
    at a=0: 1*I = int_z(sum_s(infect))
    at a=1.5: outflow
d_t(P) = beta*int_a(h*I) - int_z(sum_s(infect)) - delta*P + ka*depot
d_t(depot) = -ka*depot

B[s=S](0) = 1e6*normal(z, mean=0.1, sd=0.08)
dose depot at t=0 to 48 every 24: 1e9
lag depot = tlag
bioavailability depot = F

observable bacteria = N
observable resistant = int_z(B[s=R])
extinction phage: depot, P, I below 1 with noise_floor
```

Read and solved from Python (`F` was declared without a value, so it is given here):

```python
import numpy as np
import tenqsp as tq

tour = """
model notation_tour
axis a in [0, 1.5] n=15 units=h kind=one_way
axis z in [0, 1] n=10
axis s categorical [S, R]
param g=1, K=1e9, beta=50, delta=0.1, h_max=5, tau=0.8, s_h=0.05,
      Dz=0.01, mu=1e-6, ka=1.2
param k[s] = {S: 1e-9, R: 1e-12}
param tlag = 0.5 units=h
param F
field B(z, s) units=cells
field I(a)
compartment P units=PFU
N      = int_z(sum_s(B))
div    = g*(1 - z)*B*(1 - N/K)
h(a)   = h_max*sigmoid((a - tau)/s_h)
infect = k[s]*B*P
crowd  = 1   if N < K/10
         0.5 if N < K/2
         0   otherwise
transitions M on s: S -> R = mu, R -> S = mu
d_t(B) = crowd*div - infect + trans_s(M, div) + d_z(Dz*d_z(B))
d_t(I) = -d_a(I) - h*I                             using sharp
    at a=0: 1*I = int_z(sum_s(infect))
    at a=1.5: outflow
d_t(P) = beta*int_a(h*I) - int_z(sum_s(infect)) - delta*P + ka*depot
d_t(depot) = -ka*depot
B[s=S](0) = 1e6*normal(z, mean=0.1, sd=0.08)
dose depot at t=0 to 48 every 24: 1e9
lag depot = tlag
bioavailability depot = F
observable bacteria = N
observable resistant = int_z(B[s=R])
extinction phage: depot, P, I below 1 with noise_floor
"""
m = tq.Model.from_text(tour)
sol = m.solve((0, 24), times=np.linspace(0, 24, 5), params={"F": 0.8})
print(sol["bacteria"].shape, sol["B"].shape)     # (5,) and (5, 10, 2)
```

`m.to_text()` writes it back in the notation, and `m.explain()` describes it term by term
([Reading a model back](../guide/explain.md)).

### A small PBPK model, both ways

Concentrations in three organs and two tissue spaces, a parameter table over two axes,
weights and a volume, two fluxes, transitions on each axis, equations on slices, a dose and
two observables. The Python tab builds the identical model.

=== "Notation"

    ```tq
    model mini_pbpk
    axis organ categorical [gut, liver, muscle]
    axis sub categorical [vascular, tissue]

    param V[organ, sub] = {
        gut:    {vascular: 0.06, tissue: 1.0},
        liver:  {vascular: 0.09, tissue: 1.5},
        muscle: {vascular: 1.4,  tissue: 25},
    }
    param Q_art[organ] = {gut: 60, liver: 20, muscle: 60}
    param Q_ven[organ] = {gut: 0, liver: 80, muscle: 60}
    param PS=10, Kp=2, CLint=30, V_art=1.7, Q_gut=60

    field C(organ, sub) weights=V[organ, sub] units=mg/L
    compartment Cart volume=V_art units=mg/L

    transitions X on sub: vascular -> tissue = PS, tissue -> vascular = PS/Kp
    transitions portal on organ: gut -> liver = Q_gut

    flux J_art: Cart -> C[sub=vascular] = Q_art[organ]*Cart
    flux J_ven: C[sub=vascular] -> Cart = Q_ven[organ]*C

    d_t(C)                          = trans_sub(X, C)
    d_t(C[sub=vascular])            = trans_organ(portal, C) + (J_art - J_ven)/V[organ, sub]
    d_t(C[organ=liver, sub=tissue]) = -CLint*C/V[organ, sub]
    d_t(Cart) = (sum_organ(J_ven) - sum_organ(J_art))/V_art

    A_body = sum_organ(sum_sub(C))
    dose Cart at t=0: 100
    observable C_liver = C[organ=liver, sub=tissue]
    observable amount = A_body + Cart*V_art
    ```

=== "Python"

    ```python
    m = tq.Model("mini_pbpk")
    organ = m.axis("organ", labels=["gut", "liver", "muscle"])
    sub = m.axis("sub", labels=["vascular", "tissue"])
    V = m.param("V", {"gut": {"vascular": 0.06, "tissue": 1.0},
                      "liver": {"vascular": 0.09, "tissue": 1.5},
                      "muscle": {"vascular": 1.4, "tissue": 25}}, over=[organ, sub])
    Q_art = m.param("Q_art", {"gut": 60, "liver": 20, "muscle": 60}, over=organ)
    Q_ven = m.param("Q_ven", {"gut": 0, "liver": 80, "muscle": 60}, over=organ)
    PS, Kp, CLint, V_art, Q_gut = m.params(PS=10, Kp=2, CLint=30, V_art=1.7, Q_gut=60)

    C = m.field("C", axes=[organ, sub], weights=V, units="mg/L")
    Cart = m.compartment("Cart", volume=V_art, units="mg/L")

    X = m.transitions("X", axis=sub, rates={"vascular->tissue": PS, "tissue->vascular": PS / Kp})
    portal = m.transitions("portal", axis=organ, rates={"gut->liver": Q_gut})

    m.transfer(Cart, to=C.at(sub="vascular"), rate=Q_art[organ])      # flux J_art
    m.transfer(C.at(sub="vascular"), to=Cart, rate=Q_ven[organ])      # flux J_ven

    m.transition(C, axis=sub, matrix=X, of=C)
    vascular = C.at(sub="vascular")
    m.transition(vascular, axis=organ, matrix=portal, of=vascular)
    m.hazard(C.at(organ="liver", sub="tissue"), CLint / V[organ, sub])

    A_body = m.integral(C, over=[organ, sub], name="A_body")
    m.dose(Cart, times=[0], amounts=100)
    m.observable("C_liver", C.at(organ="liver", sub="tissue"))
    m.observable("amount", A_body + Cart * V_art)
    ```

The amount in the body falls only by hepatic clearance:

```python
sol = m.solve((0, 24), times=[0, 6, 12, 24])
print(np.round(sol["amount"], 1))      # 100 at t=0, then falling
```

## See also

- [User guide](../guide/index.md): the same language, by task.
- [A first model](../start/first-model.md): a complete model, solved and plotted.
- [Troubleshooting](../guide/troubleshooting.md): what to do when a text is refused or a
  run fails.
