# ADR 0006 — Conditionals in formulas

- **Status:** Accepted by the project lead, 2026-10-13 (design proposed in the session; the project lead asked
  for conditionals at least as expressive as NONMEM's and Monolix's, in TenQSP's own style).
- **Code:** `src/tenqsp/expr.py` (grammar), `src/tenqsp/model.py` (`Condition`, `cases`,
  `eq`, `ne`), `src/tenqsp/lower.py` (`Evaluator.cases`, guards, time conditions),
  `src/tenqsp/compile.py` (`_parse_switches`, `switch_times`, `switch_values`),
  `src/tenqsp/backends/scipy_backend.py` (segment boundaries, events), `src/tenqsp/validate.py`
  (`check_switches`). Tests: `tests/test_conditionals.py`.

## Context

NONMEM (`IF/ELSE IF/ELSE`), Monolix (`if/elseif/else`) and rxode2 all have conditionals;
TenQSP's language must be at least as expressive. Conditionals are also jumps in the
equations, and jumps are what make stiff solvers stall (pbisim's dose-junction failures).

## Decision

**One meaning:** a list of cases tried in order; the first true condition wins; a final
`otherwise` value is required (no silent default: NONMEM keeps an old value when no branch
matches, a classic bug).

**Three spellings of it:**

1. Formula text (what the IR stores), Python's conditional expression, which chains without
   nesting: `k1 if x > thr else k2 if x > thr2 else k3`. Comparisons `< <= > >= == !=`
   (chains `a < x <= b`), joined by `and`, `or`, `not`. Refused with a plain message: a
   condition used as a number (`k*(x > 1)`), `&`/`|`/`~` (in Python syntax `x > 1 & y < 2`
   means `x > (1 & y) < 2`), a non-condition after `if`, `True`/`False`.
2. Python API: `<`, `<=`, `>`, `>=` on formulas give a `Condition`; `&`, `|`, `~` join them;
   `tq.cases((cond, value), ..., otherwise=value)`; `tq.eq`, `tq.ne` (Python's `==` keeps its
   meaning). Using a condition as a number or as a Python truth value raises with a hint.
3. PDE notation (Phase 3): the paper's cases brace (`k = k1 if x > thr / k2 if x > thr2 /
   k3 otherwise`), and an `if / elif / else` block for code ported from NONMEM or Monolix;
   both lower to the same IR text. Each conditional defines one quantity.

**Evaluation:** `jnp.where` from the last case back. Each value is evaluated under a guard
(where it is chosen); elsewhere the arguments of functions singular somewhere (`log`, `1/x`,
`sqrt`, `x^p`, `asin`, `acos`, `tan`, `atan2`, `gammaln`, `factorial`, `probit`, `rem`) are
replaced by a harmless number. The result is unchanged, and reverse-mode derivatives cannot
pick up `0 * inf = NaN` from a value that is not chosen (the "double where" pattern, applied
automatically). Guarded definitions are not shared with unguarded uses.

**Switches handled exactly by the solver**, by what a condition depends on:

| Depends on | Example | Handling |
|---|---|---|
| time and scalar parameters, closed-form roots | `t > t_on`, `30 < t <= 30.001` | roots solved once (SymPy `solveset`, only a proven finite set), evaluated per solve: segment boundaries, like doses |
| time and scalar parameters, no closed form | `sin(t) > 0` | sampled at 20,001 points over the solve span and polished with brentq: segment boundaries. A pair of switches closer than span/20,000 can be missed (documented). |
| state, as one number (compartments, totals) | `C > MIC`, `B_tot > K` | solver events: stop at the crossing, restart, then watch that condition only for the way back |
| state that varies cell by cell | `B > 3` (per cell) | evaluated where it stands; the validator adds a note |
| coordinates and parameters only | `x < 0.5` | never switches in time; nothing to do |

- Conditions on time are constant between their switch times, so within a solve they are
  evaluated once per segment, at its midpoint, held in one extra slot of `theta` (NaN outside
  a solve: the current time is then used; `cm.n_theta`). *Why:* evaluated at the time
  itself, the first point of a segment still sees the old branch and BDF fights a jump at its
  starting point (a 1e-3 h pulse failed at rtol 1e-10 and was 6e-7 off at the default).
- Events are not used for conditions on time: with a right-hand side constant on each side of
  the switch the solver takes steps longer than the gap between two switches, and an event
  test (a sign change between step ends) misses both (seen with `sin(t) > 0`).
- A state that slides along a boundary (pushed back across it from both sides) has no
  solution as a jump; the solver error then names the condition that had just switched and
  suggests a smooth form. Events per solve are capped (10,000).

## Consequences

- Piecewise ODEs match their closed forms to the solver tolerance (tests: time switch, pulse,
  periodic switch, compartment switch, quorum on a total), and a new switch time is a new
  parameter value (no recompilation).
- `theta` gains one slot only for models with conditions on time; `cm.theta()`/`theta0` and
  `solve(theta=...)` accept the parameter-only length too.
- Comparisons on formulas, previously refused, are now conditions (API change, approved).
