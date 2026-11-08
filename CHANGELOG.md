# Changelog

All notable changes to this project are documented here. The project follows semantic
versioning; `0.x` releases may change the API.

## [Unreleased]

### Added
- **One entry of a parameter table in a formula:** `Vvas[organ=spleen]` is the spleen's value,
  usable anywhere a number is; over several axes each axis is given once, by name (it still
  varies) or fixed by label: `iKp[phage, organ=liver]`. Python: `Vvas.at(organ="spleen")`.
  The entry is the parameter itself, so values given when solving reach it; a model no longer
  needs a second copy of a table's value. The model file format (IR) is version 0.11; files of
  earlier versions still load.

### Fixed
- A dose time written as a name (`dose A at t=Tdose: D1`) was refused with the advice to name a
  formula, which cannot help: dose times are fixed when the model is read. The message now
  says so, that a dose amount may be a formula of parameters, and that a delay to be estimated
  is a lag time (`lag A = tlag`).

## [0.7.0] — 2026-11-07

### Added
- The constant `pi` in formulas (π), in models and in `tq.formula` (where it is not one of
  `f.names`). A model that declares a name `pi` of its own keeps its meaning, as a model name
  may match a function name, and in `tq.formula` a value given for `pi` (`f.jax(["pi", ...])`)
  replaces the constant. A name formulas cannot read, such as an observable, cannot be called
  `pi` (formulas would silently read the constant). There is no constant `e`: Euler's number
  is `exp(1)`.
- `sim.function(..., derivatives="forward")`: forward sensitivities. The run carries the
  derivative of every state with respect to every name in `wrt`, with no record and no
  replay. Cost grows with the number of names (2–3 solves plus a fraction of a solve per
  name); memory stays small (a batch of 4 on a 61-state model with 12 names: 3.5 MB
  against 840 MB). `derivatives="reverse"` is the record and replay as before.
  `f.derivatives` says which a function uses.
- `sim.function([sc1, sc2, ...], wrt=...)`: several scenarios as one JAX function, for a
  population. Each scenario keeps its own regimen, output times, start and starting state;
  `fs(X)` takes one row of values per scenario and gives `{output: (scenarios, times, ...)}`,
  NaN past each scenario's own times (`fs.t`, `fs.mask`, `fs.n_times`); `fs.status(X)` gives
  one status per scenario. All run in one `jax.vmap` call of one build: the scenarios are
  padded to the largest numbers of doses, infusions and output times with slots the solver
  never reaches, so the padding is exact.
- `f(x, rtol=..., atol=...)` for a function from `sim.function` (and for `f.status` and a
  function of several scenarios): one call at other tolerances than the simulator's, with no
  new build or compile (the tolerances were already inputs of the compiled program). For
  checking that a tolerance is tight enough. With `derivatives="reverse"`, a much tighter
  tolerance can need more steps than the record holds: NaN, with that status.
- `tq.formula(text)`: one formula in TenQSP's formula language, read and checked with the
  same grammar, functions and messages as in a model. `f.names` lists the names it uses and
  `f.jax(names)` is a JAX function of values for them (differentiable, `jit`, `vmap`). For
  programs that write formulas outside a model, such as a fitting part.
- `f.steps(x)` for a function from `sim.function`: the most solver steps that the run of
  `f(x)` takes between two doses or events, the number the simulator's `max_steps` limits
  (`fs.steps(X)`: one per scenario). So that a program calling `f` at an optimizer's
  values can set `max_steps` from the work at reasonable values, and a run at an extreme
  value fails quickly instead of taking minutes.
- A model text may continue with a fitting part: a line `fit` or `fit <name>` ends the model,
  and `tq.load` and `tq.Model.from_text` read the model above it. One `.tq` file can then hold
  a model and its fit.

### Fixed
- A slice read in a formula that leaves some axes free, `C[sub=cellular]` with `organ` free,
  is the field's value there, one per organ, as documented. It was refused as an integral
  that is not an amount when the field's weights varied over a free axis (volumes per organ).
- A flux may join ends whose free axes differ when one end's are all free at the other: a
  blood pool per drug, `field Cart(drug)`, feeding `C[sub=vascular]` over drug and organ.
  The flux goes element by element over the shared axes and the smaller end gives or takes
  the sum over the others (`sum_organ(J_art)`), as a compartment already did. A whole-body
  PBPK model with two drugs can then keep its fluxes and their balance check. Ends that share
  no such order (`B(drug) -> C(organ)`) are still refused.
- `m.transfer` with an end that fixes an axis by its slice, `A.at(organ="liver") -> B` over
  drug and organ (or the reverse): the fixed end now gives or takes the sum over that axis,
  so the transfer conserves the amount, as `explain()` says. It gave or took one
  destination element's share only. The case with every axis fixed,
  `A.at(drug="d1", organ="liver") -> B`, was already wrong in 0.6.1.
- Second derivatives of a function from `sim.function` (`jax.hessian`, or `jax.jacfwd` or
  `jax.jacrev` of `jax.grad`). With `derivatives="reverse"` they were wrong, with no
  warning; with `"forward"`, `jax.hessian` failed with an error from inside the solver.
  Both are now right (checked against central differences of the gradient, with dose
  amounts and times and an infusion's rate, start and duration among the inputs). Where a
  second derivative is not available, a plain error says so and what to do instead:
  models with extinction groups or switches (an event's time has an exact first
  derivative only), `jax.jacrev(jax.grad(f))` with forward derivatives (use `jax.hessian`),
  and large models on the sparse tier with forward derivatives (use `"reverse"`).
- The JAX solver could run for hours, or never end, when the first step from t = 0 kept
  failing (values far outside a model's range, such as an optimizer's trial point with a
  rate constant of 1e133). The smallest step there is a number so small that the CPU treats
  it as zero, so the step was halved to 0 and the "step too small" test never fired. The
  run now ends at once with that status (NaN from `sim.function`, an error from
  `simulate`), as the scipy backend does.

### Changed
- `sim.function`'s default is now `derivatives="auto"`: forward up to 40 names in `wrt`,
  reverse above (`True`, the old default, means `"auto"`). Measured on the CPU, forward was
  faster for every whole table of derivatives (`jax.jacfwd`) and, for a gradient
  (`jax.grad`), up to 40–50 names: 12 theophylline subjects in one `jax.vmap` call, 111 ms
  to 32 ms per gradient. Results agree with reverse to the solver's tolerance. A function
  made by `"auto"` with up to 40 names keeps no record, so the status "too many steps for
  the record" no longer occurs with it.
- The JAX solver's step limit (`max_steps`) is an input of its compiled programs, like the
  tolerances: simulators that differ only in `max_steps` share every build, with no new
  compile.
- `sim.function` derivatives: the replay skips the unused room in its record in blocks
  instead of step by step, forward and in reverse. Same results; a gradient of a small model
  costs about half as much (one theophylline subject: 54 ms to 24 ms).
- `m.diagnostics(sol)`: the note on a total that fell below `atol` and stayed there is only
  for a field or compartment that can grow from a remnant of itself (on a loop of feeds:
  bacteria, phage through infected cells), no longer for a depot that empties at the end of
  a dose.

### Documentation
- Install page: the combined `pip install "tenqsp[plot,xarray]"`, the command for an NVIDIA
  GPU (`pip install -U "jax[cuda12]"`), and when a GPU pays (batches of scenarios as one
  JAX function).
- About page: why the name.

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
