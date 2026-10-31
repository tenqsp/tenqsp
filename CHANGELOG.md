# Changelog

All notable changes to this project are documented here. The project follows semantic
versioning; `0.x` releases may change the API.

## [Unreleased]

### Added
- `sim.function(..., derivatives="forward")`: forward sensitivities. The run carries the
  derivative of every state with respect to every name in `wrt`, with no record and no
  replay. Cost grows with the number of names (2–3 solves plus a fraction of a solve per
  name); memory stays small (a batch of 4 on a 61-state model with 12 names: 3.5 MB
  against 840 MB). `derivatives="reverse"` is the record and replay as before.
  `f.derivatives` says which a function uses.
- `tq.formula(text)`: one formula in TenQSP's formula language, read and checked with the
  same grammar, functions and messages as in a model. `f.names` lists the names it uses and
  `f.jax(names)` is a JAX function of values for them (differentiable, `jit`, `vmap`). For
  programs that write formulas outside a model, such as a fitting part.
- A model text may continue with a fitting part: a line `fit` or `fit <name>` ends the model,
  and `tq.load` and `tq.Model.from_text` read the model above it. One `.tq` file can then hold
  a model and its fit.

### Changed
- `sim.function`'s default is now `derivatives="auto"`: forward up to 40 names in `wrt`,
  reverse above (`True`, the old default, means `"auto"`). Measured on the CPU, forward was
  faster for every whole table of derivatives (`jax.jacfwd`) and, for a gradient
  (`jax.grad`), up to 40–50 names: 12 theophylline subjects in one `jax.vmap` call, 111 ms
  to 32 ms per gradient. Results agree with reverse to the solver's tolerance. A function
  made by `"auto"` keeps no record, so the status "too many steps for the record" no longer
  occurs with it.
- `sim.function` derivatives: the replay skips the unused room in its record in blocks
  instead of step by step, forward and in reverse. Same results; a gradient of a small model
  costs about half as much (one theophylline subject: 54 ms to 24 ms).

## [0.6.1] — 2026-10-31

### Added
- `tq.regimens_from_table`: one regimen per subject (`{ID: Regimen}`) from a NONMEM dose
  table with several subjects.

### Fixed
- A model with no states (no `d_t(...)` equation) is refused by `validate()` with a plain
  message; it used to fail at the solve with a NumPy error.
- `sim.function`: with a dose after the start time, the value could be NaN (with status
  "ok") when the right-hand side is not defined below zero, such as a Hill term `C^h` with
  a non-integer `h`.
- `sim.function`: in a model with an extinction group, the gradient (`jax.grad`) could be
  NaN while the value and forward-mode derivatives were right, under the same condition.

### Documentation
- A page listing what TenQSP does not do yet (Start here → What TenQSP does not do yet).
- Performance: a note on many subjects, and on padding scenarios to one shape (zero-amount
  doses and extra output times change results within the solver's tolerance, tested).
- Tutorial pages moved from `tutorials/_build/…` to `tutorials/…`.

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
