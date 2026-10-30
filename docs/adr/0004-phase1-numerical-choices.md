# ADR 0004 — Phase 1 numerical choices

- **Status:** Accepted by the project lead, 2026-10-12 (Phase 1 gate; `ledger=False` option kept).
- **Code:** `src/tenqsp/operators/`, `src/tenqsp/compile.py`, `src/tenqsp/ledger.py`,
  `src/tenqsp/backends/scipy_backend.py`. Tests: `tests/test_v_operators_1d.py`,
  `tests/test_compile.py`, `tests/test_ledger.py`, `tests/test_v16_compile_once.py`.

## Context

The Phase 1 gate requires an ADR for every numerical choice made in the phase.
ADRs 0001–0003 cover the solver, the formula grammar and lowering, and scale. This one
records the choices made while building steps 1.1–1.5, each with the test that backs it.

## Decisions

1. **Advection by flux splitting on the total face velocity, two schemes**.
   - Several `advect` terms on one axis act as one: their face velocities are added, then
     upwinded once, so writing a velocity as drift + chemotaxis adds no numerical diffusion.
     They must share a scheme; mixing schemes on one axis is refused. (Upwinding each term
     on its own was a defect found in the Phase 1 review.)
   - `erlang`: first-order upwind. With N cells and speed v over length A it is exactly the
     N-compartment transit chain with rate N·v/A (V2: right-hand sides agree to 1e-12, and a
     full model solve matches the hand-written chain to 1e-10 for N = 1, 2, 5, 20).
   - `sharp`: MUSCL reconstruction with the van Leer limiter. It is second order in smooth
     regions and creates no new extrema (V1 tests).
   - At an inflow boundary the `sharp` scheme takes the edge value from the boundary condition
     (J/v for inflow, the held value for Dirichlet). Using the end cell instead made the end
     cells first order and spoiled the whole age profile (step 1.1).
   - Outflow is always first-order upwind using the end cell. So the design order of `sharp`
     depends on the norm: 2 in L1 (and for the outflow flux), 1 in the max norm, where the
     largest error sits in the outflow cell. V15 tests both. The McKendrick test criterion is
     the largest error relative to J/v, below 2e-3 at 80 cells.
2. **Diffusion with harmonic face averages**. A zero diffusivity in a cell blocks
   the flux through its faces. Fourier-mode decay is second order (V3). Several diffusion
   terms are summed first. A negative diffusivity is used as given, never clipped (no hidden
   clamps, PRD §5.10); warning about it belongs in `diagnostics()`.
3. **Boundary conditions as fluxes at the two end faces.**
   - Dirichlet: an upwinded advective flux with the held value, plus a half-cell diffusive
     flux.
   - Robin: a series resistance (exchange coefficient h in series with the half-cell
     diffusion), which is second order. Using the end cell value directly is only first order
     (MMS, V15).
   - Robin together with advection is refused until Phase 2.
   - Inflow: the prescribed flux J enters, and anything the flow carries outward through the
     same end leaves by upwinding. The ledger books both on that end's line.
4. **Default ends follow the validator's notes.** A one_way axis with advection gets inflow 0
   at the lower end and absorbed outflow at the upper end. Every other end is zero-flux.
5. **Velocities are taken at the faces.**
   - A formula of the coordinate is evaluated at the faces.
   - A formula of cell values is interpolated linearly onto the interior faces, by distance
     from the two centres, so nonuniform grids stay consistent.
   - A constant is broadcast.
   - The curved-velocity test tells face values apart from averaged centre values.
6. **Categorical transitions as generators** `Q[dest, origin]`. The diagonal is derived as
   minus the column sums and applied with one `tensordot` along the axis. `mutate`
   redistributes the production flux at the parent's rate (V13 in miniature).
7. **Initial values are cell averages.**
   - Shape functions use exact CDF differences, truncated to the axis and normalised.
   - Other formulas use 3-point Gauss–Legendre per continuous axis whose coordinate they use.
   - `total` + `shape` is normalised to the exact total.
8. **Doses split the run into segments.**
   - Each segment starts afresh at the post-dose state.
   - At a dose time the state is reported just after the dose (right-continuous).
   - A failed BDF segment is retried once with Radau, and the retry is reported. That
     includes an error raised inside the integrator, such as a non-finite matrix. LSODA is
     refused.
   - A dose before the start time is refused.
9. **Jacobian: dense `jacfwd`, handed to scipy as a sparse matrix.** scipy then uses a sparse
   LU, which made the toy model 13× faster. A 5,000-state limit holds until the coloured
   Jacobian of Phase 2.
10. **Compile once.** Parameters, geometry, state and time are runtime inputs of every jitted
    function. Observables and ledger rates are evaluated one output time at a time, so a new
    number of output times does not recompile either. Dose amounts given as
    parameters change without recompiling. Literal dose amounts are part of the model text, so
    changing one makes a new model. V16: 100 parameter sets, including dose amounts, give one
    compilation of every function. Trace time is the same from 121 to 160,801 states.
11. **Conservation ledger as quadrature states**.
    - The amount of a field is its density times the cell widths, summed; categories are
      summed plainly.
    - Interior fluxes cancel in that sum, so the change in amount is exactly the sum of:
      - reaction, hazard and compartment-rate formulas, booked piece by piece (split at
        top-level `+`/`−`, named by the text as written);
      - transitions, which sum to zero by construction;
      - each boundary end that is not zero-flux;
      - routed outflows;
      - doses.
    - Every channel is integrated by the solver as an extra state with derivative equal to its
      rate.
    - Options considered:
      - (a) Trapezoid over the output times: inaccurate unless outputs are dense.
      - (b) The solver's dense output: tied to scipy internals.
      - (c) Quadrature states.
    - Chose (c). A linear multistep method preserves linear invariants, so "start + channels =
      amount" holds to round-off (≈1e-15 relative in every test model), not just to the
      truncation error. The cost was within run-to-run noise (toy model 17.3 s with the ledger,
      18.5 s without). `solve(..., ledger=False)` turns it off.

## Verification (Phase 1 gate)

V15 runs through the compiled model with a manufactured solution, for every Phase 1
combination: {erlang, sharp} × {inflow flux, held value} at the upstream end × {constant,
curved} velocity of either sign × {uniform, stretched} grid. It also covers diffusion with
constant and variable D × {Dirichlet, Robin, zero-flux} at both ends, and advection with
diffusion. V4 includes the renewal condition (births = b × population), checked against the
Euler–Lotka stable age distribution at and above replacement fertility. The full matrices run
in the slow tier and a representative subset in the fast tier.

## Consequences

- The ledger checks the bookkeeping, not the biology. A model that leaves out a loss term
  still balances, but the piece-by-piece statements make such omissions visible: phage lost to
  adsorption appears next to infected cells gained.
- `sharp` costs about 25× more solver work than `erlang` on the toy model. The front crosses
  an age cell every 0.04 h and BDF follows it (benchmarks). This belongs in `explain()`
  (Phase 3).
- `beta(...)` shapes have no gradient with respect to their own parameters (JAX `betainc`).
  This is a Phase 4 note.
