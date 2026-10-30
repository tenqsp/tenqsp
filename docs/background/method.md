# The numerical method

This page is for the modeller who wants to know what TenQSP does with a model once it is
written: how a partial differential equation becomes a system of ordinary differential
equations, how that system is solved, and how derivatives with respect to the parameters are
obtained. It uses a little mathematics. Nothing here is needed to build or run a model; the
practical side (which scheme, which tolerance, which solver) is in
[Numerical choices](../guide/numerics.md).

## The method of lines

A model in TenQSP has two kinds of variable. **Compartments** depend on time only, as in
NONMEM or Monolix. **Fields** also depend on one or more axes: infection age, depth in a
biofilm, progress through the gut, a resistance genotype. For a field \(u(t, a)\) on one
continuous axis, an equation such as

$$
\frac{\partial u}{\partial t} = -\frac{\partial}{\partial a}\big(v\,u\big)
  + \frac{\partial}{\partial a}\Big(D\,\frac{\partial u}{\partial a}\Big) + r(u, \dots)
$$

has a transport term, a diffusion term and a local term (reactions, losses, births).

TenQSP solves it by the **method of lines**: each continuous axis is cut into cells, the
derivatives along the axes are replaced by differences between cells, and what is left is a
system of ordinary differential equations in time, one per cell, which a standard stiff ODE
solver integrates. Time is never cut into a fixed grid: the solver chooses its own steps, as
it does for a compartmental model.

## Cell-centred finite volumes

The axis \([a_0, a_n]\) is cut into \(n\) cells \([a_{i-1/2}, a_{i+1/2}]\) of width
\(\Delta a_i\). The boundaries between cells are the **faces**. The unknown in each cell is
the **cell average**

$$
u_i(t) = \frac{1}{\Delta a_i}\int_{a_{i-1/2}}^{a_{i+1/2}} u(t, a)\,da ,
$$

so \(u_i\,\Delta a_i\) is the amount in the cell. Integrating the equation over one cell
turns the slope of the flux into the difference of the fluxes at its two faces:

$$
\frac{du_i}{dt} = -\frac{F_{i+1/2} - F_{i-1/2}}{\Delta a_i} + r(u_i, \dots),
$$

where \(F\) is the total flux (transport plus diffusion) through a face, positive toward
larger \(a\). This is the **conservation form**. TenQSP computes one flux per face and uses
it twice: taken from the cell on one side, given to the cell on the other. Summed over all
cells, the interior fluxes cancel exactly, and

$$
\frac{d}{dt}\sum_i u_i\,\Delta a_i = F_{\text{lower end}} - F_{\text{upper end}}
  + \sum_i r_i\,\Delta a_i .
$$

The change in the total amount is what came in and out through the two ends plus the local
terms, to rounding, whatever the grid and whatever the velocity. This is why transport and
diffusion are written as `-d_a(v*I)` and `d_x(D*d_x(F))`: the notation is the conservation
form, and the discretisation keeps it. It is also what lets the
[ledger](../guide/explain.md#the-ledger-where-every-amount-came-from) account for every amount
(below).

Starting values are cell averages too. A shape such as `point`, `normal`, `gamma` or `uniform` gives
each cell its exact share of the mass (differences of the distribution function); any other
formula of the coordinate is averaged over each cell with three-point Gauss–Legendre
quadrature.

## Transport: upwind fluxes and transit chains

At a face with velocity \(v\), the transport flux takes its value from the side the flow comes
from (**upwinding**):

$$
F_{i+1/2} = \max(v, 0)\,u^{L}_{i+1/2} + \min(v, 0)\,u^{R}_{i+1/2},
$$

where \(u^L\) and \(u^R\) are the values just left and right of the face. The velocity may
change sign along the axis or in time; the formula needs no branch. A velocity that depends on
the coordinate is evaluated at the face; one that depends on cell values (a field, a total) is
interpolated linearly between the two neighbouring cell centres, so stretched grids stay
consistent.

### `erlang`: first-order upwind, exactly a transit chain

The default scheme takes the face value from the neighbouring cell average:
\(u^L_{i+1/2} = u_i\), \(u^R_{i+1/2} = u_{i+1}\). With a constant velocity \(v > 0\) on equal
cells of width \(\Delta a\), the equation of cell \(i\) in amounts \(A_i = u_i\,\Delta a\) is

$$
\frac{dA_i}{dt} = k\,A_{i-1} - k\,A_i, \qquad k = \frac{v}{\Delta a},
$$

which is exactly a chain of transit compartments with transfer rate \(k\). Material entering
the first of \(N\) cells leaves the last after a time that follows an Erlang (gamma)
distribution with \(N\) stages: mean \(N/k\), the transit time of the axis, and coefficient of
variation \(1/\sqrt{N}\). This is the transit chain of the Savic absorption model and of the
Friberg myelosuppression model, and `m.explain()` describes an `erlang` axis in those words.

So with `erlang` the number of cells is part of the model: it sets the spread of the delay.
Refining the grid does not converge to anything, it makes the delay sharper. The scheme is
**monotone**: it creates no new peaks and never makes a value negative.

### `sharp`: MUSCL with the van Leer limiter

For delays that should be (nearly) exact, `using sharp` reconstructs a straight line in each
cell instead of a constant, and reads the face values from it. The slope in cell \(i\) comes
from the two differences across its faces,
\(g_- = (u_i - u_{i-1})/(a_i - a_{i-1})\) and \(g_+ = (u_{i+1} - u_i)/(a_{i+1} - a_i)\), through
the **van Leer limiter**

$$
s_i = \begin{cases}
\dfrac{2\,g_-\,g_+}{g_- + g_+} & \text{if } g_- \text{ and } g_+ \text{ have the same sign,}\\[1ex]
0 & \text{otherwise (a peak or a trough).}
\end{cases}
$$

The limited slope keeps every face value between the neighbouring cell averages, so no new
peaks or negative values appear. Where the solution is smooth the scheme is second order: the
error falls about four times when the cells are halved, and refining the grid converges to
transport without spread. At peaks the slope is set to zero and the scheme drops to first
order there, which is the price of staying positive. At an end where the inflowing value is
known (an inflow, a held value) the end cell uses it for its slope; elsewhere the end cells
are first order.

What it buys: fronts keep their shape, a latent period stays a latent period. What it costs:
the solver now resolves the front as it moves from cell to cell, so its steps are at most
about one cell-crossing time, and the Jacobian (below) changes along the way and must be
recomputed more often. On the models measured in [Performance](performance.md), `sharp` costs
about 3 to 11 times as much as `erlang`.

## Diffusion

Diffusion uses the two-point flux across each face,

$$
F_{i+1/2} = -D_{i+1/2}\,\frac{u_{i+1} - u_i}{a_{i+1} - a_i},
$$

which is second order. When \(D\) varies, the two half-cells on either side of the face act
as resistances in series, so the face value is their harmonic mean:

$$
\frac{a_{i+1} - a_i}{D_{i+1/2}} = \frac{\Delta a_i/2}{D_i} + \frac{\Delta a_{i+1}/2}{D_{i+1}} .
$$

A cell with \(D = 0\) therefore blocks the face completely.

### Ends as face fluxes

Every boundary condition enters in one place only: the flux through the end face. No ghost
cells are added outside the axis.

| Condition | Flux through the end face |
|---|---|
| no flux (the default) | 0 |
| a flux condition, `at a=0: v*I = k*B*P` | the formula, as written |
| outflow | the upwind flux of the edge cell, \(v\,u_{\text{edge}}\) |
| a held value \(c\) with diffusion | \(-D\,(u_{\text{edge}} - c)/(\Delta a/2)\): the gradient over the half-cell between the end and the edge cell's centre |
| exchange with an outside level \(c\) at coefficient \(h\) (Robin) | \(h\,g\,(c - u_{\text{edge}})/(h + g)\), with \(g = D/(\Delta a/2)\): the exchange and the half-cell in series |

The series form of the exchange keeps second order at the end, where using the edge cell's
value alone would be first order. With flow across the same end, the incoming flow carries
the outside level and the outgoing flow the value at the face, and the half-cell balance gives
that value in closed form. The [Boundary conditions](../guide/boundaries.md) chapter shows
each one in a model.

## Categorical axes and transitions

A categorical axis (genotype, phenotype, organ, tissue space) is not discretised: each
category is a value. Transitions between categories are a **generator matrix**
\(M[\text{dest}, \text{origin}]\). The model gives only the off-diagonal rates
(`S -> R = mu`); the diagonal is minus the sum of the rest of its column, so every column sums
to zero:

$$
M = \begin{pmatrix} -\mu & \nu \\ \mu & -\nu \end{pmatrix}
\quad\text{for}\quad S \to R \text{ at } \mu,\; R \to S \text{ at } \nu .
$$

A column summing to zero means that what leaves one category arrives in another, so a
transition can neither create nor lose cells. TenQSP applies the moves one by one (each move
takes \(\text{rate} \times X[\text{origin}]\) from the origin and gives it to the destination),
never forming the matrix, so a rate may differ from cell to cell along the other axes and may
depend on the state. A mutation term, `trans_s(M, div)`, moves a share of the **parent's**
production: daughters of a susceptible cell that carry a mutation arrive at the susceptible
cells' division rate.

### Several axes

The state of a field over several axes is the tensor product of its axes. For a field
\(B(z, s)\) with \(n_z\) cells and \(n_s\) categories, and constant coefficients, the
transport operator along \(z\) and the transitions along \(s\) act on separate factors:

$$
\frac{d\mathbf{b}}{dt} = \big(A_z \otimes I_s + I_z \otimes M\big)\,\mathbf{b},
$$

where \(A_z\) is the one-axis transport matrix and \(I\) an identity. Each operator only ever
works along its own axis, cell by cell over the others. That is what keeps a structured model
cheap: the number of non-zero couplings per cell does not depend on the grid. A joint
transition on two categorical axes (one mutation giving resistance to two phages) is a
generator over the pairs of categories of both axes.

## Totals and integrals

A total over an axis, `int_a(h(a)*I)`, is a sum over cells of the integrand at the cell
times the cell width:

$$
\int h(a)\,I(t, a)\,da \;\approx\; \sum_i h(a_i)\,I_i\,\Delta a_i .
$$

For a field alone (\(h = 1\)) this is exact: the cell averages times the widths are the
amounts. With a weighting function it is the midpoint rule, second order in the cell width.
Categorical axes are summed plainly (`sum_s`). The same integral written twice in a model is
computed once.

## The state vector

The solver sees one vector \(y\). The fields come first, in the order they were declared, each
flattened over its own axes with the last axis varying fastest; then the compartments. A field
\(P(a, s)\) on 5 age cells and 2 genotypes is 10 entries, \(P(a_1, S), P(a_1, R), P(a_2, S),
\dots\), and `m.summary()` reports the count per field and the product of axis sizes that
makes it. The ledger channels, when the ledger is on, follow as extra entries (below).

The parameters, the axis geometry, the time and the state are all inputs of the compiled
functions, not constants built into them. That is why changing a parameter value, a dose
amount or the output times never recompiles a model.

## The sparse Jacobian

A stiff solver needs the Jacobian \(J = \partial f/\partial y\) of the right-hand side. In a
structured model almost all of it is zero: a cell is coupled to its own state, to its
neighbours along each axis (one on each side for `erlang` and diffusion, two for `sharp`), to
the other categories of its own cell, and to the compartments and totals it reads. TenQSP
reads this **pattern** off the model's equations; it never probes the function to find it.

**Colouring.** Two columns of \(J\) that never have a non-zero in the same row can be
computed together: perturb both states at once, and each row's change belongs to only one of
them. The columns are coloured so that columns sharing a row get different colours, and one
forward-mode derivative per colour (a Jacobian–vector product with the sum of that colour's
unit vectors) gives every entry of those columns. The number of passes depends on how many
neighbours a cell has, not on how many cells there are: the benchmark phage model, a field on
two axes plus compartments, needs 10 passes at 1,261 states and 10 at 100,489.

**Totals lifted out.** A total couples every cell to every other: logistic crowding on the
whole population, the total burst rate in the phage equation. Treated directly it would make
whole rows and columns dense, and the number of colours would grow with the grid. TenQSP
instead writes the right-hand side as \(f(y) = g(y, q)\), with \(q\) the totals the model
declares (and any value of a field read at an end), so that

$$
J = \frac{\partial g}{\partial y} + \frac{\partial g}{\partial q}\,\frac{\partial q}{\partial y} .
$$

\(\partial g/\partial y\) is local and coloured as above. \(\partial q/\partial y\) has one row
per total, computed by reverse-mode derivatives; totals over disjoint cells share a pass.
Compartment equations, which may read whole fields, are rows computed the same way, one
reverse pass each. Every entry is an exact derivative (by JAX's automatic differentiation),
not a finite difference.

The result is a sparse matrix. At 10^5 states a dense Jacobian would need 80 GB; the sparse
one has about 10^6 non-zeros. `m.compile().summary()` reports the size of the pattern, the
passes, and the memory of the matrix and its LU factors.

## Stiff integration

Structured models are **stiff**: they mix fast processes (adsorption, small cells crossed
quickly) with slow ones (growth, elimination). An explicit solver would need steps as short as
the fastest process. TenQSP uses **BDF** (backward differentiation formulas), an implicit
multistep method, in the variable-order, quasi-constant step form of scipy (the numerical
differentiation formulas of MATLAB's `ode15s`): orders 1 to 5, chosen as the run
goes, with error control on every step. Each step solves a nonlinear system by a simplified
Newton method with the iteration matrix \(I - c\,J\) (\(c\) proportional to the step), factorised once and reused over many
steps; the Jacobian is recomputed only when Newton fails to converge.

**scipy BDF** (the default backend) uses scipy's integrator with TenQSP's sparse Jacobian and
a sparse LU (SuperLU). If BDF fails on a segment, that segment is retried once with **Radau**,
an implicit Runge–Kutta method, and the retry is reported.

**The JAX BDF** (`backend="jax"`) is a port of scipy's BDF to JAX: the same formulas, the same
constants, the same step-size and order rules, written as one compiled loop. On the same model
the two take the same steps and agree to about 1e-13. It solves its linear systems in one of
two **tiers**, chosen by size: a dense LU inside JAX up to 250 states (500 on a GPU), and above
that SuperLU on the host, factorising the same sparse matrix as scipy. It has no Radau
fallback. Its purpose is what scipy cannot do: derivatives, and many parameter sets in one
call (`jax.vmap`).

### Doses, events and segments

A bolus is a jump in the state, which no ODE solver should step across. Doses therefore split
a run into **segments**. Each segment starts afresh at the state just after the dose, at
order 1, with a first step chosen by the usual rule but never below \(10^{-8}\) of the
segment's length (a step near \(10^{-12}\) h after a very large bolus would leave Newton's
corrections at the level of rounding). An infusion is a constant input between its start and
stop, which are segment boundaries too. A condition on time (`k = kmax if t < tstop else 0`)
adds its switch times as boundaries; a condition on the state is an event, located to rounding
on the step's interpolant, where the run stops and restarts.

### Extinction groups and noise floors

An **extinction group** is an event on the total of its members, with the members counted as
amounts. Once the total has reached its threshold the group is armed; when it then falls
below, every member is set to zero at that crossing and the run restarts there. With a
**noise floor**, while the group's total is below the threshold every equation (the
right-hand side, the totals, the Jacobian) sees the group's states as zero, so a ghost of
\(10^{-12}\) cannot feed itself; outputs and the ledger still report the state as it is.
[Numerical choices](../guide/numerics.md#extinction-groups-and-noise-floors) shows both at
work.

### The ledger

Each ledger channel (a reaction term, a boundary flux, a dose, an extinction) is integrated
alongside the state as an extra entry of \(y\) whose rate is that channel's rate. Because the
interior fluxes cancel, the amount of each field is exactly its starting amount plus the sum of
its channels; a linear multistep method with an exact Jacobian keeps that linear relation to
the accuracy of its Newton iterations, so the books close to rounding (around 1e-15 relative on
the benchmark models). The ledger entries do not steer the steps, and the tolerances are
rescaled so a run with the ledger is as accurate as one without.

## Derivatives: record and replay

TenQSP **discretises, then differentiates**: the derivatives it returns are the exact
derivatives of the solution it computed, step for step, not an approximation to the
derivatives of the exact solution. That is what an optimizer needs: the objective and its
gradient are consistent.

The JAX BDF decides its steps inside a loop that reverse-mode differentiation cannot pass
through. So a run with derivatives happens twice:

1. **Record.** The run is solved as usual, and every accepted step is written down: its time,
   its order, the step-size changes, the segment starts, the doses, the extinctions.
2. **Replay.** The same steps are taken again as a fixed sequence, with no error control,
   rejection or Newton iteration. Each step's value is taken from the record, so the replay's
   outputs equal the run's to rounding. Each step's derivative is that of its implicit
   equation (the implicit function theorem, solved with the step's own factorisation), so
   forward mode (`jax.jacfwd`) and reverse mode (`jax.grad`) of the replay are the derivatives
   of the computed solution. They agree with each other to rounding.

The solver also chooses its steps for the derivative: the derivative along the parameters is
carried in the run and enters the error test, so a solution at rest whose derivative is not
cannot take steps too coarse for the derivative. Dose times, lag times and the times at which
a condition on time switches move with the parameters, so their derivatives are exact too. How to use all this, and what it costs, is in
[Gradients and sensitivities](../guide/gradients.md).

## See also

- [Numerical choices](../guide/numerics.md): grids, schemes, tolerances and solvers in
  practice
- [Reactions, transport and diffusion](../guide/terms.md): how each term is written
- [How TenQSP is checked](verification.md): the evidence that all of the above is right
- [Performance](performance.md): what it costs
