# Changelog

All notable changes to this project are documented here. The project follows semantic
versioning; `0.x` releases may change the API.

## [0.6.0] — first public release

TenQSP's first public release. What it offers:

### Writing a model
- **Fields over axes** next to ordinary compartments: continuous axes (age, maturation,
  dormancy, depth, time since a dose) with uniform or geometric spacing, and categorical axes
  (phenotype, susceptibility, organ), in any combination.
- **The PDE notation**: a model written as equations, as in a paper (`d_t(B) = -d_a(B) - k*B`),
  with boundary conditions under their equation, read with `tq.load` or
  `tq.Model.from_text`. The same model can be built step by step in Python; the two give the
  identical model, and `m.to_text()` writes either one back as notation.
- **Terms**: transport with an `erlang` (transit-compartment) or `sharp` scheme, diffusion,
  chemotaxis (`grad`), reactions and hazards, transitions between categories (one-way or both
  ways, on a state or on a flux), integrals and sums over axes, named formulas, conditionals.
- **Boundaries**: inflow (renewal), outflow, zero flux, Dirichlet and Robin conditions, and
  fields read at an end.
- **Doses**: boluses and infusions into compartments or into fields, spread by a shape
  (`normal`, `gamma`, `point`, …), with lag times, bioavailability and regimens; dose amounts
  and times are inputs, not structure.
- **Extinction groups** with an explicit threshold, as exact events.
- **One name per function**, the same in the notation and in Python; names used by NONMEM, R
  or other programs are refused with the TenQSP name to write.

### Checking and explaining
- `validate()` with errors that name the line and say what to write instead; `explain()`
  says in words what every term means and how it is computed, and lists every numerical
  safeguard; `to_latex()` prints the equations for an article.
- A **ledger** that accounts for everything dosed, moved, gained and lost, and closes to
  rounding.

### Solving
- scipy's BDF (Radau as fallback) with a structurally sparse, coloured Jacobian; TenQSP's own
  BDF solver in JAX, with dense and sparse linear-algebra tiers chosen by size, on CPU or GPU.
- **Exact derivatives** of the computed solution with respect to parameters, dose amounts and
  dose times (`Function`, `Sensitivity`).
- **Many runs from one build**: `Simulator` and `Scenario` for batches of parameter sets and
  regimens, grids, and pre-runs to a steady state or a fixed time.
- Results as arrays or labelled `xarray` data, and plots of trajectories, fans, sensitivities
  and snapshots.

### Documentation
- A user guide in twelve chapters, a complete notation reference and Python API reference,
  background on the method, its verification and its performance, and twelve tutorials, each
  checked against a published model or a known result.
