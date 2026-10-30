# ADR 0007 — Hierarchical compartments (PBPK as a tensor)

- **Status:** Accepted by the project lead, 2026-10-18 (design shown before building, as PRD §15.9
  asks; the project lead chose "weighted fields hold concentrations; moves take clearances").
- **Code:** `src/tenqsp/model.py` (`param(over=[...])`, `field(weights=)`,
  `compartment(volume=)`, slices in `reaction`/`hazard`/`switch`/`mutate`/`integral`/
  `observable`/`dose`, `transfer`), `src/tenqsp/compile.py` (`measure`, `_size`,
  `_apply_moves`, `_transfer`, `_deliver`, `_mask`, `_pick`), `src/tenqsp/lower.py` (tables
  over several axes, weighted and sliced integrals), `src/tenqsp/sparsity.py` (transfers),
  `src/tenqsp/validate.py`, `src/tenqsp/backends/scipy_backend.py` (doses and extinction
  totals use `measure`). Tests: `tests/test_hierarchical.py`, `tests/test_v32_v35_pbpk.py`
  (model in `tests/pbpk_models.py`).

## Context

Whole-body PBPK is a tensor `A[organ, sub]` with blood pools as hubs (gallery Model 9,
PRD §5.11). Compartments differ in size (volumes), flows connect only the vascular spaces,
and exchange coefficients are indexed by organ. The mathematics of transfers between
elements of different size must conserve amount, and the user should write PBPK quantities
(flows, permeability-surface products, clearances) without dividing by volumes.

## Decision

1. **Weights play the role of cell widths.** `m.field("C", axes=[organ, sub], weights=V)`:
   the field holds a concentration; amount = value x widths (continuous axes) x weight.
   Weights are formulas of parameters (usually a table `V[organ, sub]`), > 0, varying only
   over the field's categorical axes. `m.compartment("Cart", volume="V_art")` likewise holds
   a concentration. Without weights or a volume nothing changes.
2. **Moves take clearances.** A `switch` rate on a weighted field, and every `transfer` rate,
   moves rate x (value of the origin) of amount per time: a flow or clearance in volume per
   time. The destination's value rises by that amount over its own size. Without weights
   this is the usual per-capita rate, so one rule covers both. *Rejected:* storing amounts
   and per-capita rates (the project lead's alternative): PBPK rates would then be `PS/V` of the
   origin, a rate varying along the axis it moves between (ADR 0005, decision 17).
3. **`mutate` moves a fraction of production:** daughters are `mu` x the amount the origin
   produces, so the destination rises by `mu * flux * W_origin / W_destination` (the same as
   before when sizes are equal). Reactions and hazards stay per unit of the field's value.
4. **Slices in operators:** `C.at(sub="vascular")` in `reaction`, `hazard`, `switch`,
   `mutate`, `transfer`, `integral`, `observable` and `dose` (one element, every axis
   fixed). A switch on a slice may not fix the axis it moves along. Implemented with masks
   (`reaction`, `hazard`; the rates of `switch`/`mutate`) and one-category picks (sources,
   integrals, observables), so the state layout and the Jacobian pattern are unchanged.
5. **`m.transfer(source, to=, rate=)`:** compartment -> slice (broadcast; the compartment
   loses the sum), slice -> compartment (sum), slice -> slice (element by element over the
   same free axes), `to=None` (elimination). Fields with continuous axes are excluded (their
   coupling is a boundary condition or a reaction). The ledger records a line on both sides
   (`transfer to X`, `transfer from Y`).
6. **Integrals of weighted fields are amounts:** the integrand is multiplied by the weights;
   every axis the weights vary over must be integrated or fixed by the slice, and fields with
   different weights are not mixed in one integrand (validator).
7. **Parameters over several axes:** `m.param("V", nested_dict, over=[organ, sub])`, stored
   flattened with the first axis slowest, written `V[organ, sub]`; `params={"V": {"liver":
   {"cellular": 2.0}}}` changes one entry without recompiling.
8. **`measure(theta)`** (per state: widths x weights, or volume) replaces the fixed width
   vector: weights are parameters, so ledger amounts, dose increments (amount / size) and
   extinction totals are computed from the parameters of each solve.
9. **IR 0.5:** `ParamSpec.over` may be a tuple; `FieldSpec.weights`;
   `CompartmentSpec.volume`; `where` on `TermSpec`, `IntegralSpec`, `ObservableSpec`,
   `DoseSpec`; `TransferSpec`. Files of versions 0.1–0.4 load unchanged.

Deferred: masks for ragged hierarchies (composition with separate
fields and slice-to-slice transfers covers them now) and dotted names (`organ.liver.cellular`;
`C.at(organ="liver", sub="cellular")` says the same).

## Consequences

- V32–V35 pass: infusion steady state = R/CL_H (1e-9), AUC = dose/CL_H (1e-7), partition
  equilibrium (1e-8), equal to 30 hand-written ODEs (1e-7), and the perfusion-limited model
  approached at the rate 1/PS. The ledger closes to round-off.
- New public API (approved): `param(over=[...])`, `field(weights=)`, `compartment(volume=)`,
  `transfer`, slices in operators and doses.
