# Python API reference

Every public class and function of `tenqsp`, generated from the docstrings. Import the
package as

```py
import tenqsp as tq
```

| Page | What is there |
|---|---|
| [Building a model](model.md) | `tq.Model` and its methods; slices, conditions |
| [Reading and writing text](text.md) | `tq.load`, `tq.table`, `tq.Model.from_text`, `tq.formula` |
| [Formula functions](functions.md) | `tq.exp`, `tq.sigmoid`, …; shapes `tq.normal`, `tq.point`, … |
| [Regimens](regimens.md) | `tq.bolus`, `tq.infusion`, `tq.regimen_from_table`, `tq.regimens_from_table`, `tq.Regimen` |
| [Simulation](simulation.md) | `tq.Scenario`, the simulator, results, grids, batches, pre-runs |
| [Derivatives](derivatives.md) | `sim.function`, `sim.sensitivity` and what they return |
| [Solutions and the ledger](solution.md) | what `m.solve` returns; the conservation ledger |
| [Plots](plot.md) | `tq.plot.trajectories`, `fan`, `sensitivity`, `snapshot` |
| [Checks and errors](errors.md) | `tq.validate`, reports, the IR, every exception |

The tasks these serve are explained in the [User guide](../../guide/index.md); the model
language in the [Notation reference](../notation.md).
