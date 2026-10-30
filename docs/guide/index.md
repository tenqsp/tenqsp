# User guide

One chapter per task. Each shows the **notation** and the **Python API** side by side, in
tabs: choose one and every page follows. The two build the same model, and a model built
either way can be written out as the other (`m.to_text()`, and `tq.Model.from_text`).

| Chapter | What it covers |
|---|---|
| [Axes, fields and compartments](axes-fields.md) | the parts of a model: continuous and categorical axes, fields, compartments, parameters, named formulas |
| [Reactions, transport and diffusion](terms.md) | the terms of an equation and how each is read |
| [Categories and transitions](transitions.md) | switching and mutation between categories, joint transitions, per-category parameters |
| [Boundary conditions](boundaries.md) | what happens at each end of an axis |
| [Totals, integrals and ends](integrals.md) | sums and integrals over axes, a field read at an end, slices |
| [Starting values](initial.md) | values at time 0: densities, amounts and shapes |
| [Doses and regimens](doses.md) | doses in the model and regimens at simulation, lag and bioavailability |
| [Simulation and exploration](simulation.md) | solving, scenarios, grids and batches, pre-runs and steady state, plots |
| [Gradients and sensitivities](gradients.md) | derivatives with respect to parameters and dosing |
| [Reading a model back](explain.md) | `explain()`, `diagnostics()`, the ledger, `to_text()`, `to_latex()` |
| [Numerical choices](numerics.md) | grids, schemes, tolerances, the two solvers and their tiers |
| [Troubleshooting](troubleshooting.md) | error messages and what to do about them |
