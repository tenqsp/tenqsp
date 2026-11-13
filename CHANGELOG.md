# Changelog

All notable changes to this project are documented here. The project follows semantic
versioning; `0.x` releases may change the API.

## [Unreleased]

### Added
- **A warning when values go negative:** a run (`m.solve`, `sim(...)`, either backend) whose
  state went below −10 × `atol` warns with `tq.NegativeValuesWarning`, naming the state, how
  far and when, and whether the negative part grew; it used to be reported only by
  `m.diagnostics(sol)`. It counts a dip that is small next to the state's values, or any dip
  of a state its equations keep from going below zero (an undershoot that growth or
  selection then amplifies). A run that fails after such a dip says so in its error message.
- **Known issues and updates:** `tq.known_issues()` lists the known problems of the installed
  version, from a list shipped with it (`known_issues.json`: the versions affected, the fix,
  whether results could change without an error, the areas touched); no network.
  `tq.check_updates()` asks PyPI for the newest version, only when called, and prints the
  fixes since the installed version, those that can change results first, then the problems
  still open and the new features. `show=False` returns either as data. Nothing checks for
  updates on its own.
- **A bioavailability that differs by level:** into a field over a categorical axis, the
  fraction of each dose that arrives may be one value per level, `bioavailability C =
  F[phage]` (two phages dosed into their own slices of `C`, each scaled by its own F). It
  can be differentiated by each level, as before. A lag is still one number per place.
- **Models given by formulas** (NONMEM's `$PRED`): a model with no differential equations,
  its outputs written as formulas of `t`, `observable E = E0 + prog*t`. It is evaluated at the
  output times with no solver: `m.solve`, `sim.function` (exact derivatives, `jit`, `vmap`),
  batches and sensitivities work as for other models. A dose enters the formula as a parameter;
  doses, starting values, lags and extinction groups are refused with a message saying so.
  `explain()` and `to_latex()` show the formulas. Such models used to be refused; a model
  with neither an equation nor an observable still is. See the guide's "Models given by
  formulas".
- **Spheres and cylinders:** `axis r in [0, R] n=… geometry=spherical` (or `cylindrical`)
  makes the axis a radius. Its cells are shells, a field over it is a density per unit volume,
  `d_r(…)` is the divergence through the shells, (1/r²)∂r(r²…), and `int_r` counts their
  volumes, so totals and the ledger are amounts. The divergence may also be written out as in
  a paper, `1/r^2*d_r(r^2*D*d_r(C))`; on a flat axis that form is refused with a hint to
  declare the geometry. The centre takes no condition. `to_latex()` writes the full form and
  `explain()` says what the geometry changes. Model files are IR version 0.12 (older files
  still load). See the guide's "Spheres and cylinders".

- **`record_room=` for reverse derivatives:** `sim.function(..., derivatives="reverse",
  record_room=4)` sizes the record that reverse derivatives keep: this many times the steps of
  the run at `f.x0` (default 2). A run at an `x` that needs more gives NaN as before, and
  `f.status(x)` now says how much room it had and which `record_room` to ask for. Passing it
  to a function without a record (forward derivatives, `derivatives=False`) raises.

### Changed
- **Large models run faster on the JAX backend:** a 1,062-state tumour model took 27 s
  instead of 62 s (the scipy backend: 18 s), with the same steps and results. The sparse
  Jacobian is assembled once per Jacobian rather than at every factorisation (67 times
  instead of 2,900 in that run), and, on Linux, a single run keeps its factorisations and
  solves on one core while it lasts, instead of moving between idle cores that run at a low
  clock. `tenqsp._threads.PIN_HOST_THREAD = False` turns the second off; batches are not
  pinned.
- **Reverse derivatives need much less memory:** the record holds 2 times the steps of the
  run at `f.x0` on every model (was 4 times, 2 above 2,000 states), rounded up by at most a
  quarter instead of to the next power of two; and the replay keeps only the start of each
  block of 32 steps for the backward pass, recomputing the block from it. The derivatives of 4
  outputs with respect to 96 dose times and amounts, for a 1,062-state model whose run takes
  18,000 steps, needed more than 70 GB and now take 20 GB (and 16 minutes, against 85 for
  forward derivatives); values and derivatives are unchanged. A run far from `x0`
  meets "too many steps for the record" sooner than with the old 4 times: `record_room=4`
  restores it.

### Fixed
- A per-state `atol` given as a list for a model with one state (`atol=[1e-8]`) failed with
  a `TypeError` at the end of the run.
- **Derivatives with respect to a later dose, with a small `atol`:** `sim.function`,
  `sim.sensitivity` and the steering of reverse derivatives failed ("required step size is
  less than spacing between numbers") just after a dose that is not at the start, when
  `atol` was small (about 1e-18 or below; seen in a tumour model that needs 1e-25). Inside a
  field that the dose reaches through its neighbours, the derivative starts from 0 while the
  state does not, and its first steps have an error estimate as large as itself; its error
  scale was `atol + rtol |derivative|`, which then accepts no step. The derivative is now asked
  to be as accurate as the state itself as well, `atol + rtol (|derivative| + |state|)`
  (each derivative is taken for a change of its number in proportion to its value, so both
  are in the state's units). Derivatives agree with central differences as before; a
  derivative much smaller than its state is resolved to `rtol` of the state.
- **The same after a large dose moved late in a run:** a derivative with respect to the time
  of a dose given late (its direction moves the dose in proportion to its time, so the
  derivative is large) is huge next to where the dose enters and tiny a few cells away, below
  the rounding of the linear solve; with a small `atol` the steps were again rejected down to
  round-off. Entries below 1e-12 of the derivative's largest entry are now held only to that
  floor.

## [0.8.0] — 2026-11-09

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
