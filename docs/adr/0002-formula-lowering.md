# ADR 0002 — From formula text to whole-array JAX code

- **Status:** Accepted by the project lead, 2026-10-10, with the additions in "Project lead decisions" below
  (Phase 0.5 spike (b), 2026-10-10).
- **Spike code:** `spikes/symbolic/` (`lower.py`, `run_b.py`, `results.json`, `handtest_b.py`).

## Context

Modellers write coefficients as formulas: `k[s]*(1 - 0.5*z)*B*P` or
`h_max*sigmoid((a - tau)/s_h)`. The library must turn each formula into JAX code
that:
- is evaluated for every cell of a multi-axis field at once;
- takes parameters as array inputs, so it compiles once per model structure;
- has a trace cost that does not grow with grid size;
- is differentiable;
- never executes user text.

Categorical-axis parameters such as `k[s]` must broadcast along their
own axis.

## What the spike did

1. **Parse.** The formula is read with Python's `ast` module. The SymPy expression is built by
   walking an allow-list: numbers, names, `+ - * / ** ^`, unary minus, calls to `exp log sqrt
   abs tanh max min sigmoid`, and `vector[axis]` subscripts. Nothing reaches `eval`.
   `sympy.parse_expr` was *not* used, because it calls `eval` internally.
2. **Lower.** `k[s]` is replaced by a plain symbol. `sympy.lambdify(..., modules="jax")`
   produces the function. Each symbol is then fed an array shaped to sit on its own
   dimension of the target field:
   - axis coordinates as `(n_z, 1)`;
   - category vectors as `(1, n_s)`;
   - scalar parameters as `theta[i]`;
   - fields and compartments passed in.

   Broadcasting then evaluates every cell at once.
3. **Integrals** are a sum over an axis times the cell width, applied to a lowered integrand.

## Results (`run_b.py`, all PASS)

| Question | Result |
|---|---|
| Q1 Lowered formulas equal the hand-written toy-model terms (adsorption, hazard, growth with `cost[s]` and `B_tot`, lysis integral) | max rel. diff ≤ 1.8e-15 |
| Q2 100 different parameter vectors → compilations | 1 |
| Q3 Trace time from 200 to 200k cells | 14 ms → 10 ms (flat) |
| Q3 Compile time from 200 to 200k cells | 31 ms → 94 ms (mild growth, see below) |
| Q4 Autodiff gradient vs central finite difference | rel. err 2e-10 |
| Q5 Unsafe or wrong text (import, attribute access, `open`, lambda, comprehension, unknown name, unindexed vector, wrong index axis) | 8/8 rejected with a plain message |
| Q6 `max`/`min` | lowered to `jnp.maximum`/`jnp.minimum`, exact |

Example messages: `'k' has one value per level of axis 's'; write 'k[s]'`;
`Unknown name 'Bmax' at column 1 of 'Bmax * B'. Did you mean 'max'?`

## Decision

1. **The parser is the stdlib `ast` module plus an allow-list walker that builds SymPy.** No
   new dependency, no `eval`. The PRD §8.2 PDE-notation parser (Phase 3) is built on the same
   walker: `d_t(...)`, `d_a(...)` and `int_a(...)` are recognised as special calls whose
   arguments are walked in the same way.
2. **Lowering uses `sympy.lambdify(modules="jax")` on plain symbols, with broadcast-shaped
   arguments.** Categorical-indexed parameters are rewritten to symbols and fed as
   theta-slices reshaped onto their axis.
3. **Axis coordinates and cell widths become runtime inputs**, not constants closed over at
   trace time. The spike closed over them, and the mild compile-time growth (31 → 94 ms over
   1000× cells) is XLA embedding those arrays as constants.
4. **Error messages name the column and suggest a fix** (unknown-name suggestions via
   `difflib`). This is the PRD's "language aimed at non-specialists".

## Consequences

- No parser dependency. The allow-list is the security boundary; tests must cover it (fuzzing
  with `hypothesis` in Phase 3).
- `^` is accepted as power for familiarity. Bit-xor has no meaning in a model, so this is safe.
- `difflib` suggestions can be odd (`Bmax` → `max`). Acceptable; it can be improved later by
  preferring declared model names over function names.
- Piecewise expressions (`Piecewise`/`where`) were not exercised. Add them to the Phase 1
  allow-list with a test.

## Project lead decisions (2026-10-10)

The project lead reviewed the toy phage model written in the proposed PDE-notation text and in the
Python API (see the PRD §8 examples). The text form reads well. Additions and confirmations:

1. **Notation confirmed:**
   - `k[s]` for per-class parameters;
   - `d_t(F)` (alias `d/dt(F)`);
   - conservative `d_z(v*F)`;
   - `int_z(...)` and `sum_s(...)`;
   - `transitions M on s: S->R = mu` with the diagonal derived;
   - `mutate(s, M, flux)` acting on a production flux;
   - named intermediate expressions;
   - `^` accepted as power.
2. **Initial conditions separate amount from shape:** `init F[...] total=<amount> shape=<expr>`.
   The library:
   - truncates the shape to the grid;
   - normalises it by quadrature so the discrete total equals `total` exactly;
   - assigns each cell its exact mass (CDF differences where available, sub-cell quadrature
     otherwise), because finite-volume cells hold averages.

   `split s: S=0.99, R=0.01` gives fractions over a categorical axis.
3. **Built-in shape functions**, usable anywhere a formula is allowed. Their parameters are
   **keyword arguments with standard statistical names**; the axis is the only positional
   argument. Exactly one argument per group:

   | Shape | Location / scale | Spread |
   |---|---|---|
   | `normal(x, …)` | `mean` | `sd` \| `var` \| `cv` |
   | `lognormal(x, …)` | `meanlog` \| `median` \| `mean` | `sdlog` \| `cv` |
   | `gamma(x, …)` | `shape`+`rate` \| `shape`+`scale`, or `mean` | with `mean`: `sd` \| `var` \| `cv` |
   | `exponential(x, …)` | `rate` \| `mean` | — |
   | `beta(x, …)` | `alpha`+`beta`, or `mean` | with `mean`: `sd` \| `var` \| `concentration` |
   | `uniform(x, …)` | optional `lower`, `upper` (default: whole axis) | — |
   | `point(x, at=…)` | `at` | — |

   Naming follows R/scipy conventions (`dlnorm` meanlog/sdlog, `dgamma` shape/rate). `median`
   is offered only for the lognormal, where it is exact (`exp(meanlog)`) and matches the
   pharmacometric "typical value". Missing, conflicting or impossible arguments (e.g. `sd <= 0`,
   beta `var >= mean*(1-mean)`) are rejected before solving, with the accepted combinations
   listed. Argument values may be formulas of parameters (e.g. `mean=tau`). Keyword names are
   scoped to the call, so they never clash with model parameters (e.g. a burst size `beta`).
4. **Multiple axes:**
   - independent axes use products of marginals (normalised jointly);
   - correlated shapes are written as an explicit formula of the coordinates;
   - no built-in `mvnormal` until a real model needs it.
5. **Later, not now:** steady-state / pre-run initialisation stays a v1.0 goal.
   Initialisation from a data table comes after v1.0.

The spike parser rejects all keyword arguments; Phase 1 (API) and Phase 3 (text) allow them
for the shape functions only.


## Implementation note (Phase 1 step 1.4, 2026-10-11)

The production compiler (`tenqsp.lower`) keeps decisions 1, 3 and 4 but replaces
`sympy.lambdify` (decision 2) with a small walker over the SymPy tree, run while JAX traces the
right-hand side. Reason: shape functions (`normal(z, ...)` etc.) mean a cell-average density,
which needs the cell faces, not the value at a point, and lambdify has no hook for that. The
principle is unchanged: every symbol is fed an array shaped onto its own axis of one shared
frame, parameters come from `theta`, nothing is executed from user text, and tracing cost does
not depend on the grid. `1/(1 + exp(-x))` is evaluated as `jax.nn.sigmoid(x)` so derivatives do
not overflow. Piecewise expressions are still not in the grammar.
