# What TenQSP does not do yet

TenQSP refuses what it does not support, with a message, rather than doing something
different from what you meant. This page lists those limits in one place, so you don't have to
find them one refusal at a time.

## Models

- **Every model is solved numerically.** There are no closed-form (analytical) solutions,
  not even for linear one-, two- or three-compartment PK; such a model is solved like any
  other. It is exact to the solver's tolerance, but slower than a formula would be.
- **A model needs at least one state:** an equation `d_t(...)` for a compartment or a field.
  A model given only by formulas of `t` (a closed-form concentration, say) is refused.
- **Parameters are constant during a run.** A value that changes over time (weight,
  creatinine clearance, a concomitant drug) can be written as a formula of `t` with a
  conditional (`x1 if t < 24 else x2`), but there are no per-subject tables of values over
  time.

## Doses

- **No steady-state doses.** Give the doses that lead to the steady state (`ADDL`, `II`), or
  start from a pre-run state ([Simulation and exploration](../guide/simulation.md)).
- From NONMEM dose tables, **resets** (`EVID` 3 or 4) and **rates or durations modelled in
  the model** (`RATE` = -1 or -2) are refused ([Doses and regimens](../guide/doses.md)).

## Estimation and exchange

- **TenQSP simulates and differentiates; it does not estimate.** It gives exact derivatives
  for an optimizer or a sampler of your choice ([Gradients and
  sensitivities](../guide/gradients.md)), but has no fitting routines of its own.
- **Many subjects:** a function built for derivatives is compiled once per scenario *shape*
  (how many doses go where, how many output times). Subjects with different shapes can be
  padded to one ([Performance](../background/performance.md#many-subjects)).
- **No SBML import, and no export** to NONMEM, Monolix or nlmixr2 code.
