# ADR 0008 — Phase 3: the notation, the round trip, and what the library reports

- **Status:** Accepted by the project lead at the Phase 3 gate, 2026-10-23; project lead decisions recorded below.
- **Code:** `src/tenqsp/notation.py`, `src/tenqsp/_notation_ast.py` (reading text),
  `src/tenqsp/writer.py` (`m.to_text()`), `src/tenqsp/canonical.py` (comparison of two models),
  `src/tenqsp/explain.py` (`m.explain()`, `m.diagnostics()`, the slope for `m.fokker_planck`),
  `src/tenqsp/latex.py` (`m.to_latex()`), `src/tenqsp/model.py`, `src/tenqsp/compile.py`
  (`transport_coefficients`). Spec: `docs/language/notation.md` (approved by the project lead 2026-10-22).
  Tests: `tests/test_notation*.py`, `tests/test_v12_round_trip.py` (`pytest --roundtrip`),
  `tests/test_explain.py`, `tests/notation_models/*.tq` (41 models written by an independent
  agent from the spec alone).

## Context

Phase 3 adds a second front end, text in PDE notation, which must build exactly the models
the Python API builds (V12), and the reports the PRD asks for (`explain()`, diagnostics,
§5.10's warnings). The notation's syntax is the project lead's decision (notation.md §8). This ADR
records the choices underneath it that decide numbers or what the user is told.

## Decisions

1. **Text lowers to the same IR as Python.** The reader calls the public `Model` methods; it
   has no lowering of its own. A model read from text therefore compiles and solves through
   the same code as the API model (Phase 3 benchmark: identical trajectories and RHS counts).
2. **Numbers are written exactly.** `to_text()` writes the shortest text that reads back to
   the same double (`repr`, scientific notation outside [1e-4, 1e5)). Face lists of
   nonuniform axes are written in full. *Why:* V12 compares models, and a rounded number is
   a different model.
3. **Two models are compared in meaning, not in text.** `canonical.differences` inlines named
   formulas, then compares each pair of expressions: symbolically when SymPy can decide;
   conditionals branch by branch (`piecewise_fold`), since a difference confined to one
   branch is missed by random points about 60% of the time; otherwise at 3 random points
   (seeded; every symbol uniform in [0.3, 1.7]), equal within 1e-10 relative, evaluated with
   30 digits. Numbers in the IR compare to 1e-14 relative. *Rejected:* textual or tree
   equality (`a*b` vs `b*a` fails; two writings of one model are common).
4. **Starting values are cell averages.** A formula of the axes is averaged over each cell by
   3-point Gauss–Legendre (exact to degree 5). A value with no shape is a density, evaluated
   as written. `total*named_shape(...)` puts each cell's exact share (from the shape's CDF)
   times `total` in it. `total*shape(formula)` averages the formula by Gauss and scales the
   cells to hold exactly `total`. `density(named shape)` uses the curve itself as a density,
   unscaled. (Project lead 2026-10-23; notation §2.1 has the side-by-side table.)
5. **`fokker_planck` writes the Itô form.** For `dz = w dt + sqrt(2D) dW` it writes velocity
   `w − dD/dz` (worked out by SymPy) and diffusion `D`. It refuses a `D` that varies along
   the axis through a field or an integral, uses `grad`, or jumps or has a kink (a
   conditional, abs, max, min, floor, ceiling, rem), since then the slope is not a formula.
   `scheme="sharp"` by default: with `erlang` the scheme's own spread (about v·width/2) adds
   to `D` (16% off at 160 cells on the test case). *Evidence:* the stationary law
   `(1+z)^-11` of `w = −1, D = 0.1(1+z)` is reproduced at second order; without the correction
   one gets `(1+z)^-10`. (Project lead 2026-10-23.)
6. **`explain()` states only what holds.** A uniform `erlang` transport is described as a
   chain of N compartments at rate |v|/width, with mean transit time L/|v| and CV 1/sqrt(N),
   only when the velocity is the same at all times and states; otherwise only the rate. A
   negative velocity runs the chain towards the lower end. Numbers are printed to 6
   significant digits next to the modeller's own names. The list of numerical safeguards is
   the backend's fixed behaviours (no clipping; BDF with one Radau retry; the first-step
   floor, read from the backend's constant) plus the model's declared noise floors and
   cell-by-cell rules. A new safeguard in the backend must be added to this list.
7. **Diagnostics thresholds.** Cell Péclet number |v|·width/D, flagged above 2 (where
   `erlang`'s spread exceeds D). Transport coefficients are sampled at up to 25 output
   times. With a solution, every output time is checked: a total below `atol` that later grows above 10×`atol` is a *warning*,
   unless a dose to that place falls between the dip and the regrowth; a total below `atol`
   that never rises above 10×`atol` again is a *note* (project lead 2026-10-23: kept, although it
   also fires on true clearance; it says how to check: a smaller `atol`, or an extinction
   threshold); values below −10×`atol` are a warning. Diagnostics read `sol.options` (the
   solver settings as the user gave them).
8. **LaTeX is generated from `to_text()`,** not from the IR directly, so the two can't
   disagree on what the model is. Transitions are written out per category as arrivals minus
   departures; formulas are printed with precedence-aware brackets. All 47 test models
   compile with a real TeX engine (checked at the gate in a scratch environment; TeX is not a
   dependency).

## Decisions from the gate review (2026-10-23)

Two independent reviewers (one for reading/writing/comparison, one for explain, LaTeX,
diagnostics and `fokker_planck`) found 12 and 12 problems; all are fixed with a test each
(`tests/test_phase3_review.py`). Those that set a rule:

9. **A flux that is not a rate times its source** (zero order `= k0`, `= k*A^2` is still a
   rate times `A`, saturable `= Vmax*A/(Km + A)` is) becomes two terms, `-J` and `+J`, as
   written in the two equations; the balance check still applies. Out of a concentration
   (a place with a volume or weights) a flux that does not read its source is refused: it is
   almost always a flow without the concentration (`= Q` for `= Q*C`).
10. **A transition term moves the place whose equation it is in.** `d_t(B) = trans_s(M,
    B[r=Y])` is refused with the equation to write instead (it was read as a transition of
    all of `B`).
11. **Named formulas do not shadow functions.** `phi = 0.5` leaves `phi(x)` the normal CDF
    (formulas.md §2); it was read as the formula.
    (2026-10-30: `phi` is no longer a function; the rule holds for every function name.)
12. **Repeated transport or diffusion terms on one axis add up**, in the comparison and in
    `explain()` (one transit chain at the total velocity). Two transports on one axis with
    different schemes cannot be written as text (`WriteError`).
13. **Comparison at both signs.** When the difference contains abs, max, min, sign, floor,
    ceiling or a step, it is also evaluated at an all-negative point and two mixed-sign points
    (positive points alone cannot tell `abs(v)` from `v`).
14. **`explain()` claims a transit time only for a chain with an exit**, and only when the
    velocity has one sign (a per-category velocity is checked over its table). Where D jumps
    along the axis it gives no SDE reading.
15. **`fokker_planck`** reads D strictly (an unreadable D is an error, not a slope of 0),
    refuses only jumps *along the axis* (a D that switches in time is fine), and writes the
    slope with full-precision numbers (`pi` as a number), checked by the comparison of
    decision 3.
16. **LaTeX** divides a transition of a weighted field by the category's weight (the rates are
    clearances), brackets a sum over categories when it is a factor and a number written with
    `×10^` when raised to a power, and marks a slice's equation as added to its field's.
17. **Diagnostics** check every dip, not only the first; a dose anywhere in the model between
    the dip and the regrowth excuses it (it may refill a place downstream); a compartment that
    books what crosses a Dirichlet or Robin end may go negative (it is credited both ways).

## Known limitations (accepted)

- In LaTeX, `a` and `a_`, or `k1` and `k_1`, print alike.
- Two slices sharing one transition matrix are written out once.
- The "write this as a flux" hint can show internal names of lifted slices
  (`C__lung_vascular`); a hand-written exchange `Q*(A − B)` is not hinted.
- An error from a transition along a continuous axis, or from an integral used as an equation,
  can lack a precise line number.
- Diagnostics see only the output times: a dip below `atol` and back between two output
  times is not seen.
- The comparison's random points (decision 3) can in principle miss a difference that holds
  only on a set of measure zero outside conditionals; none is known in practice.
