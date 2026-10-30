# ADR 0005 — Phase 2 numerical choices

- **Status:** Accepted by the project lead, 2026-10-22 (Phase 2 gate; decision 23 with the warning
  for a floored group fed from outside, the project lead's choice over documenting it only).
- **Supersedes:** ADR 0004 items 1 (edge value at an inflow end, where the field also
  diffuses), 3 ("Robin together with advection is refused until Phase 2") and 5 (how a
  velocity made of cell values reaches the faces).
- **Code:** `src/tenqsp/operators/boundaries.py`, `src/tenqsp/operators/advection.py`,
  `src/tenqsp/compile.py` (`_transport`, ledger labels), `src/tenqsp/backends/scipy_backend.py`.
  `src/tenqsp/lower.py` (`Evaluator.at_faces`, `grad`), `src/tenqsp/expr.py` (`Grad`).
  Tests: `tests/test_v_boundaries_phase2.py`, `tests/test_v13_v14.py`,
  `tests/test_v_chemotaxis_fronts.py`,
  `tests/test_gate_model_level.py` (V15 Robin with advection), `tests/test_sparse_jacobian.py`.

## Context

The Phase 2 gate requires an ADR for every numerical choice made in the phase. Step 2.1's
choices (sparsity pattern, colouring, integral lifting) are those of ADR 0003. This ADR
records the later steps.

## Decisions (step 2.2)

1. **Robin end with advection: one half-cell balance.** With flow and diffusion at a Robin
   end, the total flux is computed from the boundary value `c_f` that balances the half cell
   between the edge cell centre and the end:

       c_f = ((v_in + h) c_out + g u_edge) / (v_in + h + g),   g = D / (Δ/2)
       flux into the domain = v_in c_out − v_out c_f + h (c_out − c_f)

   Flow entering carries the outside level, flow leaving carries `c_f`, and the exchange acts
   on `c_f`. With `v = 0` this is ADR 0004's series resistance. *Rejected:* adding the upwind
   advective flux to the series-resistance exchange flux. It ignores the flow inside the half
   cell and was first order (MMS: order 1.2 where 2 is designed). Without diffusion there is no
   half-cell balance: upwind flow (outside level in, edge value out) plus `h (c_out − u_edge)`.
   *Evidence:* exact steady solutions at either end converge at order 1.9–2.1 (sharp) and 0.9
   (erlang); the model-level MMS matrix (V15) passes on uniform and stretched grids.
2. **End-cell slope of the sharp scheme where the edge value is unknown.** ADR 0004 took
   `J/v` as the edge value at every inflow end. With diffusion along the axis (Danckwerts
   inlet) the edge value is not `J/v` (the profile jumps below the feed level), and using it
   made the dispersion model first order (observed 1.06). Such ends, and Robin ends, now limit
   the end cell's slope from the two nearest interior gradients (van Leer of the two). Inflow
   without diffusion still uses `J/v`; Dirichlet still uses the held value; other ends keep a
   zero slope. *Evidence:* dispersion availability now converges at order 2.5 → 2 (V26).
3. **Danckwerts ends need no new kind.** The inlet `v C − D C' = v C_in` is a prescribed total
   flux, i.e. an inflow `J = v C_in`; the outlet `C' = 0` is an outflow (advective flux only).
   The closed-form availability (Wehner–Wilhelm form in Roberts & Rowland 1986) was checked
   against an independent `solve_bvp` (agreement 4e-15) before being used as the oracle.
4. **Exchange routed to a compartment** (`boundary(..., to=)`, approved by the project lead API). Whatever
   crosses a Dirichlet or Robin end is taken from (or given to) the compartment, through the
   same routing path as outflows, so the two ledgers mirror each other exactly (1e-14). The
   field's amounts and the compartment's are assumed commensurable (as for routed outflow).
5. **First step of every segment is floored at `1e-8` of the segment length** (BDF and Radau),
   otherwise scipy's own rule (Hairer–Nørsett–Wanner, reimplemented rather than importing a
   private scipy function). After a 1e10 bolus scipy's rule proposed ~8e-13 h; at that size
   Newton's corrections are round-off, scipy's convergence-rate test reads them as divergence,
   and BDF fails ("required step size is less than spacing between numbers"). This is the
   likely mechanism of pbisim's BDF failure at a large site bolus. *Measured:* every first step
   from 1e-10 to 1e-1 h integrated the post-bolus segment at the same cost (≈5,000 RHS
   evaluations) and accuracy; error control shrinks a start that is too large. V14 now passes
   on BDF alone (no Radau retry), with and without the ledger.
6. **Ledger variables never steer the integration.** Their absolute tolerance is infinite, so
   their scaled errors count as zero in step-size and Newton control; `rtol` and the states'
   `atol` are multiplied by `sqrt(n / (n + n_q))`, which cancels the dilution of scipy's RMS
   norm by those zeros. A run with the ledger is therefore as accurate as one without
   (trajectories agree to solver accuracy; not bit-identical, because scipy's Newton tolerance
   depends on `rtol`). The books still close to ~1e-15, because Newton with the exact Jacobian
   preserves the linear identity `amount = initial + channels`. *Note:* this alone did not fix
   V14 (decision 5 did); it removes a second, independent way bookkeeping could affect results.
7. **Ledger header shows the start of the solve** (`Ledger.t0`), not the first output time:
   a Phase 1 labelling bug found while hand-testing.

## Decisions (step 2.3)

8. **Velocities are evaluated on the faces.** ADR 0004 evaluated a velocity made of cell
   values at the cells and then interpolated the *result* to the faces, and refused formulas
   mixing the coordinate with cell values (shapes `n + 1` and `n` do not combine). Now every
   cell value that varies along the axis (fields, integrals, shape masses) is interpolated to
   the faces first (linear by distance between the two centres; the end cells' values on the
   boundary faces, as before), the coordinate is taken at the faces, and the formula is
   applied there. For a velocity linear in the cell values the two are identical (all Phase 1
   tests unchanged); for nonlinear ones both are second order. Mixed formulas such as
   `w*(x - 0.5)*S` now work.
9. **`grad(F, x)`** is the difference of `F` across each interior
   face divided by the distance between the two centres: second order at the face on smooth
   grids, and exactly what the flux needs (a Boltzmann steady state `n ∝ exp(chi S / D)` is
   reached at second order with `sharp`, first with `erlang`; V15 chemotaxis on uniform and
   stretched grids). Each boundary face takes the nearest interior face's gradient (first
   order there, like the end-cell values of fields; it matters only where flow crosses the
   end). Only an advection velocity along the same axis, written in the term itself, may use
   `grad`: it is a face quantity. *Considered:* gradients in reactions (cell-centred), and
   inside definitions; neither is needed by the gallery, so both are refused with a message
   until a model needs them.
10. **Linearised dynamics are exact in the discrete sense.** For Keller–Segel the upwinded
   cell density multiplies a velocity that is already first order in the perturbation, so the
   linear growth rate of the discrete system is the continuum eigenvalue with `k^2` replaced
   by the discrete Laplacian's eigenvalue: V43 agrees with the continuum rate to 5e-5
   (unstable) and 2e-4 (stable) at 100 cells.

## Decisions (formula functions, between steps 2.3 and 2.4; project lead request)

11. **Function set = Monolix's Mlxtran list**, which covers NONMEM's abbreviated-code
    built-ins (LOG10, SIN, COS, TAN, ASIN, ACOS, ATAN, PHI, GAMLN, MOD) and the smooth part of
    SBML's MathML: `log10`, `sin cos tan asin acos atan atan2`, `sinh cosh tanh`, `logit`,
    `invlogit`, `normcdf`, `probit`, `erf`, `gammaln`, `factorial`, `factln`, `floor ceil rem`,
    with aliases `expit`/`sigmoid`, `pnorm`/`phi`, `norminv`/`qnorm`, `lgamma`, `mod`.
    *Superseded 2026-10-30:* one name per function, so the notation and the Python helpers
    match. The aliases, and `invlogit` (in favour of `sigmoid`, the name the examples use), are
    refused with the TenQSP name to write.
    - `normcdf`, `probit` and `rem` are kept unevaluated in SymPy and computed with
      `jax.scipy.special.ndtr`/`ndtri` and `fmod`: SymPy would rewrite `erfc(-x)` as
      `2 - erfc(x)` and lose the tails (now `normcdf(-30)` and `probit(1e-300)` are exact to
      1e-12).
    - `rem`/`mod` keep the sign of the dividend (Fortran MOD, C fmod, Monolix rem), not
      Python's `%`.
    - `floor`, `ceil` and `rem` jump: the validator adds a note wherever they are used (a
      stiff solver must cut its steps at each jump; the derivative there is zero or undefined,
      so fitting does not see it).
    - Not included: Mlxtran `if/else` and NONMEM `IF/THEN` conditionals (a grammar change with
      comparisons; open project lead decision), `delay` (TenQSP expresses delays as transport along
      an age axis), and less common functions (Bessel, etc.).

## Decisions (step 2.4, extinction groups)

12. **Events, not checks.** Each group is one scipy terminal event, `weights · y − threshold`
    (weights: cell widths for field cells, 1 for compartments, i.e. the group's amount),
    watched upward until the group is armed and downward afterwards. At a downward crossing
    the integration stops at the root, every member is set to zero, the amounts removed are
    booked as an `extinction` ledger line (an instantaneous jump, like doses), the group is
    disarmed, and integration restarts. Arming also happens at the start and after doses. A
    restart costs a fresh first step (decision 5); groups cross rarely. The crossing time is as
    accurate as the trajectory (1e-8 relative at rtol 1e-10). *Rejected:* checking only at dose
    boundaries or output times (pbisim's dip-and-regrow), and RHS clamps (hidden).
13. **Groups are disjoint** (validator): zeroing one group must not change another's total.
    Members are summed as amounts, so they must be counted in the same units (documented).
14. **Noise floor** (`noise_floor=True`): while a group's total is below its threshold, every
    equation sees the group's states as zero (`where(member & below, 0, y)` at the start of the
    right-hand side, the integrals and the Jacobian alike). Sources from outside the group still
    add to it; its own growth cannot amplify a ghost. Observables and the ledger's amounts use
    the state as it is. The floor's switch is a discontinuity in the right-hand side exactly at
    the threshold, which is also where the group's event stops and restarts the integration, so
    the solver never steps across it. The pattern of the Jacobian is unchanged (the switch has
    no derivative); its columns for the group vanish below the threshold. *(Mechanism revised
    after the Phase 2 review: decision 23.)*
15. **IR 0.2:** `ExtinctionSpec.noise_floor` (default `False`); version 0.1 files still load.

## Decisions (step 2.5, transitions)

16. **Moves, cell by cell.** A transition term is applied move by move: each `origin -> dest`
    move transfers `rate * F[origin]` (or `rate * flux[origin]` for `mutate`) from the origin
    category to the destination, in every cell of the other axes. The rate is evaluated as an
    array over the field's axes, so it may vary over the other axes (dormancy depth, `PS[organ]`)
    and depend on fields, compartments and totals. The derived
    diagonal is implicit (the amount added is the amount removed), so every cell conserves
    exactly. This replaces the dense `Q @ F` of Phase 1 (same results; existing tests
    unchanged). Cost: one scatter per move and cell; generators are sparse (few moves).
    *Rejected:* a per-cell dense generator (`einsum("...od,...o->...d")`): `n^2` entries per
    cell, mostly zero.
17. **A rate may not vary along the axes it moves between.** It is the rate of leaving the
    origin for the destination; a value per category of that same axis would be ambiguous
    (read at the origin or the destination?). Dependence on the whole population goes through a
    total (an integral), as in Nielsen 2007. The validator reports it once per transition set,
    and per term it checks that the rate's other axes are axes of the target field.
18. **Joint transitions** (a shared receptor: one mutation, resistant to two phages):
    `axis=[r1, r2]` in `transitions`, `switch` and `mutate`; each category is written
    `"S,S"` (labels in axis order). A joint move may also change only one of its axes,
    which expresses a per-axis move conditional on the other axis. **IR 0.4:**
    `TransitionSpec.axis` (and a switch/mutate `TermSpec.axis`) may be a tuple of axes, the
    categories tuples of labels; 0.1–0.3 files load unchanged. The Jacobian pattern couples
    every category combination of the joint axes within a cell (a superset of the moves).
19. The validator reports an identical issue once (a formula checked from two places).

### Step 2.7: gallery regression tests, xarray output (2026-10-19)

20. **What the gallery tolerances mean for each scheme** (measured, `tests/test_gallery_phase2.py`):
    - V20 (characteristics, smooth data): L1 order 0.8 for `erlang` (pre-asymptotic, tending
      to 1) and 1.9 for `sharp`.
    - V25 (lifespan step, kinks): the maximum error falls at order about 0.5 for `erlang`
      (diffusive smearing, width ~ sqrt(da T_R)) and 0.7 for `sharp`. This is the documented
      limitation of first- and limited second-order schemes at a kink, not a defect.
    - V40 (capillary–tissue exchange): with `erlang` cells each cell is exactly a well-mixed
      capillary–tissue unit, so the mean transit time (L/v)(1 + V_t/V) is exact on every
      grid and the variance is the continuous value plus mean^2/N (tanks in series, derived
      in the test). The gallery's "2nd order" is met by the `sharp` scheme's variance. The
      `sharp` mean is not exact (the limiter is nonlinear; error order about 3).
    - V39 (CAT): N `erlang` cells absorb exactly 1 - (1 + K_a T_si/N)^-N, which tends to plug
      flow; `sharp` is closer to plug flow on the same grid.
21. **Gallery equations against the papers** (checked 2026-10-19; details in each model's
    docstring). Read in full: Nelson 2004 only; the others from abstracts, the authors'
    posters or implementations derived from the papers. Two findings: Nelson's printed
    closed form for the burst size (Eqs 21, 26) holds only for a1 = 0 (the tests integrate
    Eq 3 directly); Rong 2007 may apply the RT inhibitor within the infection age rather than
    as (1 - eps_RT) on new infections (unconfirmed, full text not accessible), so the HIV
    tests claim Nelson's model with standard drug factors, not Rong's form. Sangren–Sheppard
    1953 not accessible; the equations are taken as written in the gallery.
22. **xarray output.** `sol.to_xarray()`: fields over `("t", *axes)`, compartments over
    `("t",)`, observables over the axes they vary over; continuous axes carry cell centres
    with `kind`, `units` and `faces` attributes and an `<axis>_width` coordinate, so
    `(ds["I"] * ds["a_width"]).sum("a")` is the amount; categorical axes carry labels. Values
    are what `sol[name]` returns (densities, or concentrations for weighted fields). xarray
    stays optional (imported only when called).

## Decisions from the Phase 2 review (2026-10-21)

Three independent reviews (numerics, front end, test audit) of `c5362df..4891439`. Each
defect below was reproduced before it was fixed and has a regression test.

23. **The noise floor switches with the group's armed state, not with its total.** A floored
    group is seen as zero while it is disarmed (it has not reached its threshold since the
    start or its last extinction) and counts while it is armed. The two rules agree
    everywhere except on the threshold itself, where the total-based rule chattered: a group
    fed from outside (`dP/dt = 0.5 - P`, threshold 1) filled to 1, counted, was pushed back
    below 1, was floored, filled again, all within one step; BDF and then Radau failed. The
    flag is held in one theta slot per floored group (NaN outside a solve, where the total
    decides as before), like the time slot of ADR 0006. *Consequence, documented:* below its
    threshold a floored group does not see its own losses either, so a feed from outside
    fills it at the feed rate until it reaches the threshold. With a feed too weak to hold it
    there (true level below the threshold) the result is a sawtooth of fill-and-extinction
    cycles; to model a source of real material, put the source in the group (as a PK
    reservoir in a phage lineage) or do not use the floor. *Warning (project lead's choice at the
    gate, 2026-10-22):* validation warns, and `compile()` raises a `UserWarning`, when a
    floored group is fed from outside it: a reaction or rate that is not zero when the group
    is, an inflow, Dirichlet or Robin end with a value that is not, a transfer or a routed
    outflow from outside, or a switch or mutation into a slice. A warning, not an error: a
    feed strong enough to lift the group over its threshold is legitimate.
24. **All events at one instant are settled together.** scipy reports only the first
    terminal event; a second group crossing at the same time was already below its threshold
    at the restart and was never zeroed (two strains with equal kill rates: one left at
    8e-5). After any event every armed group at or below its threshold is zeroed, every
    disarmed group at or above it is armed, and every switch is re-aimed from where it stands.
25. **A switch standing exactly on its boundary is watched until it leaves it** (`|v|`
    leaving 0, either way), and every switch event is offset by the smallest normal number
    (2.2e-308) in the direction watched. A compartment at 0 under `C > 0` (a drug before its
    first dose) used to fire at every step: 10,000 events at t = 0.
26. **Robin exchange in conductance form**, `h g / (h + g)` (and `h` alone without
    diffusion), not `1 / (1/h + 1/g)`: the same value, but finite derivatives when `h`
    reaches 0 (`h = k C` with `C = 0` gave 0 * inf = NaN in the Jacobian and a singular LU).
27. A singular sparse LU (`RuntimeError` from `splu`) counts as a failed segment, so the
    Radau retry and the plain-language `SolverError` apply.
28. **Formulas.** A formula containing a division by zero, an undefined or a complex value
    (`1/0`, `log(-1)`, `sqrt(-1)`) is a validation error wherever it appears (it used to
    crash the validator for transition and transfer rates and weights, or pass and fail at
    compile time). Exact numeric powers with exponents beyond 1024 are computed in floating
    point and powers or factorials beyond double precision are refused (`9**9**9` hung the
    parser). `^` is always power, as the grammar says: `x > 1 ^ y < 2` reads `x > 1^y < 2`
    (the hint about `^` between conditions could never fire and was removed).
29. **Convergence helper, deviation from PRD §12.** `assert_converges` does not tighten the
    solver tolerance itself; each caller solves at rtol 1e-10 to 1e-12, well below its
    finest-grid error, and the helper refuses a finest error below 1e-12. V26 accepts order
    2 ± 0.6 on 20/40/80 cells (observed 2.6, falling toward 2 on finer grids, decision 2).
30. **Test oracles strengthened** (test audit): V31 now checks the model's own dynamics (the
    early growth rate of log V against the root of the renewal equation with Nelson's
    production ramp; `sharp` to 1e-3 at 160 cells, `erlang` at first order; a doubled death
    rate fails it); velocities made of cell values are checked by hand on an uneven grid and
    by a drift steady state on a rough grid; V14 checks that the whole bolus arrives; V17
    checks a slice's crossing time with uneven cell widths. A swapped cell-to-face
    interpolation weight is still second order on grids that refine smoothly, so only the
    by-hand flux test sees it.

## Consequences

- Danckwerts, biofilm–bulk and Robin-with-advection models are expressible and second order
  with the sharp scheme.
- Solutions of models whose end cells used a zero slope at an inflow end with diffusion change
  slightly (more accurate). Phase 1 tests are unaffected.
- `atol` may now be given per state (`n` values).
- `tq.grad(S, x)` is new public API (it was already in PRD §8.1).
- The formula functions above are new public API (requested by the project lead, 2026-10-13).
- `sol.extinctions` lists `(time, group)`; `m.extinction(..., noise_floor=)` is the PRD §5.10 opt-in.
- Slices (project lead's choice over a `per=` option, 2026-10-13): a member may be `B.at(s="R")`; groups may share a field when their slices are disjoint (IR 0.3, `ExtinctionSpec.where`). Slices for operators follow in step 2.6.
- Step 2.5: transition rates may vary over the other axes of a field; `axis=[r1, r2]` (joint
  transitions) is new public API, additive (shown to the project lead with the step 2.5 hand-test).
- Step 2.7: `Solution.to_xarray()`, `Solution.coords`/`units`/`model` and
  `tenqsp.examples.toy_three_axis_model` are new, additive.
- Review fixes: `cm.n_theta` counts one slot per noise-floor group (`solve(theta=)` still
  accepts the parameter-only vector); a text condition in `tq.cases` must be one condition
  (kept in parentheses); automatic extinction-group names replace characters that are not
  letters, digits or `_`; `noise_floor=` must be `True` or `False`; `to_xarray` leaves out a
  `<axis>_width` coordinate whose name the model uses (the widths are also an axis attribute).
