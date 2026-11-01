# The TenQSP formula language — reference

*Status: complete for Phase 2 (2026-10-13). The model language that uses it (PDE notation,
PRD §8.2) is Phase 3 and will get its own reference.*

A **formula** is one expression: a rate, a velocity, a boundary value, a definition, an
initial value, an observable. Every front end produces the same formula text, and the text is
what a saved model (the IR) stores:

| Front end | Example |
|---|---|
| Formula text | `m.reaction(B, "g*B*(1 - B_tot/K) - k*B*P")` |
| Python operators | `m.reaction(B, g*B*(1 - B_tot/K) - k*B*P)` |
| PDE notation (Phase 3) | `d_t(B) = g*B*(1 - B_tot/K) - k*B*P` |

A formula is evaluated later, for every cell of the grid at once, so it never contains
statements, loops or assignments: quantities are named with `m.define`, totals with
`m.integral`, and the equations themselves with `m.reaction`, `m.advect`, `m.ode`, … .

---

## 1. Grammar

Formulas use Python's expression syntax, read through an allow-list (nothing is ever
evaluated as Python; ADR 0002). In order of increasing binding strength:

```
formula     ::= choice
choice      ::= arith [ "if" condition "else" choice ]        # chains to the right
condition   ::= conj { "or" conj }
conj        ::= neg { "and" neg }
neg         ::= "not" neg | comparison
comparison  ::= arith relop arith { relop arith }             # a < x <= b means a < x and x <= b
relop       ::= "<" | "<=" | ">" | ">=" | "==" | "!="
arith       ::= term { ("+" | "-") term }
term        ::= unary { ("*" | "/") unary }
unary       ::= ("-" | "+") unary | power
power       ::= atom [ ("**" | "^") unary ]                   # -x^2 = -(x^2); 2^-1 = 0.5
atom        ::= number | name | name "[" axis { "," axis } "]" | call | "(" formula ")"
call        ::= function "(" formula { "," formula } ")"
              | shape "(" axis { "," keyword "=" formula } ")"
              | "grad" "(" field "," axis ")"
```

- `^` and `**` both mean power (`^` as in NONMEM and Monolix).
- Numbers: `3`, `0.5`, `1e-9`, `2.5E+3`. Not allowed: `inf`, `nan`, `True`, `False`.
- A **condition** is never a number: `k*(x > 1)` is refused; write `k if x > 1 else 0`.

## 2. Names

| Name | Meaning in a formula |
|---|---|
| parameter (`m.params(k=1e-9)`) | its value; changing it never recompiles |
| vector parameter (`m.param("k", {...}, over=s)`) | `k[s]`: the value for each category of axis `s` |
| table parameter (`m.param("V", {...}, over=[organ, sub])`) | `V[organ, sub]`: one value per combination, indices in the declared order |
| continuous axis (`age`, `x`) | the coordinate: cell centres (faces inside a velocity) |
| field (`B`) | the density in each cell (interpolated to the faces inside a velocity) |
| compartment (`P`) | its value |
| definition (`m.define("h", ...)`) | the named formula, as if written out |
| integral (`m.integral(B, over=z)`) | the total over the named axes; varies over the others |
| `t` | time |

Names are identifiers, unique across the model, and not Python keywords (`if`, `and`, `not`,
`lambda`, … are refused with a suggestion such as `lambda_`). Function names (`sin`, `max`)
are not reserved: a field named `erf` is fine; `erf(x)` is still the function.

A formula's value varies over the axes of everything it uses; the validator checks that it
fits where it is used (a compartment's rate must be one number, a field's reaction may vary
only over that field's axes).

## 3. Functions

All take numbers and act cell by cell. Every one has a Python helper of the same name
(`tq.sin`, `tq.normcdf`, …).

Each function has one name (2026-10-30: the aliases were removed). Names other programs use
are refused with the TenQSP name to write (`expr.ELSEWHERE`); `invlogit` is now `sigmoid`.

| Function | Meaning | Elsewhere (refused) | NONMEM | Monolix |
|---|---|---|---|---|
| `exp(x)`, `log(x)`, `log10(x)`, `sqrt(x)`, `abs(x)` | as usual (`log` is natural) | | `EXP LOG LOG10 SQRT ABS` | same names |
| `max(a, b, ...)`, `min(a, b, ...)` | largest, smallest | | `MAX MIN` (7.3) | `max min` |
| `sin cos tan asin acos atan` | trigonometric (radians) | | `SIN COS TAN ASIN ACOS ATAN` | same |
| `atan2(y, x)` | angle of the point (x, y) | | | `atan2` |
| `sinh cosh tanh` | hyperbolic | | | same |
| `logit(p)`, `sigmoid(x)` | `log(p/(1-p))`, `1/(1+exp(-x))` | `invlogit`, `expit` | | `logit invlogit` |
| `normcdf(x)`, `probit(p)` | standard normal CDF and its inverse (accurate in the tails) | `pnorm`, `phi`, `PHI`; `norminv`, `qnorm` | `PHI` | `normcdf probit` |
| `erf(x)` | error function | | | |
| `gammaln(x)`, `factorial(x)`, `factln(x)` | log gamma, `gamma(x+1)`, `gammaln(x+1)` | `lgamma`, `GAMLN` | `GAMLN` (7.3) | `gammaln factorial factln` |
| `floor(x)`, `ceil(x)` | round down, up (jump: validator note) | | `INT` truncates | `floor ceil` |
| `rem(a, b)` | remainder, sign of `a` (jump: note) | `mod`, `MOD` | `MOD` (7.3) | `rem` |
| `grad(F, x)` | slope of field `F` along `x`, at the faces; **only** inside an advection velocity along `x` | | | |

### Shape functions (initial conditions)

`m.init(B, total=1e7, shape=normal(z, mean=0.2, sd=0.05))` places an amount on an axis; the
shape is integrated exactly over each cell and cut to the axis. The first argument is the
axis; the rest are keywords, one of each pair:

| Shape | Keyword combinations |
|---|---|
| `normal` | `mean` with `sd`, `var` or `cv` |
| `lognormal` | `meanlog`, `median` or `mean`, with `sdlog` or `cv` |
| `gamma` | `shape` with `rate` or `scale`; or `mean` with `sd`, `var` or `cv` |
| `exponential` | `rate` or `mean` |
| `beta` (on [0, 1]) | `alpha` and `beta`; or `mean` with `concentration`, `var` or `sd` |
| `uniform` | optional `lower`, `upper` (default: the whole axis) |
| `point` | `at`: everything in the cell containing it |

## 4. Conditionals

```
k1 if x > thr else k2 if x > thr2 else k3
```

- The cases are tried in order; the **first true condition wins**; the final value after the
  last `else` is required (every cell always has a value).
- Conditions: comparisons `< <= > >= == !=`, chains such as `thr2 < x <= thr`, joined with
  `and`, `or`, `not` and parentheses. `&`, `|`, `~` are refused (in Python syntax
  `x > 1 & y < 2` would silently mean `x > (1 & y) < 2`).
- A value that is not chosen cannot spoil derivatives: `log(P) if P > 0 else 0` is safe at
  `P = 0`.

How the solver treats a switch (ADR 0006):

| The condition depends on | Example | What happens |
|---|---|---|
| time and parameters | `kin if t > t_on else 0`, `1 if sin(2*3.14159*t/24) > 0 else 0` | switch times found before solving; the solve restarts exactly there, like at a dose |
| compartments or totals (one number) | `0.5 if C > MIC else 0.05` | the solver stops exactly at the crossing and restarts |
| fields, cell by cell | `B if B > 1 else 0` | evaluated where it stands; the validator notes it |
| coordinates and parameters only | `D1 if x < 0.5 else D2` | does not change in time: nothing special |

A state pushed back across its own switch from both sides (it would slide along the
boundary) has no solution as a jump; the solver stops with a message naming the condition
and suggesting a smooth form (`sigmoid`).

**Python API:**

```python
k = m.define("k", tq.cases((x > thr, k1),
                           (x > thr2, k2),
                           otherwise=k3))
both = (x > a) & (x <= b)        # and;  | is or;  ~ is not
tq.eq(n_doses, 3), tq.ne(a, b)   # == and != (Python's own == is kept)
```

## 5. What is refused, and why

| Written | Message (shortened) | Instead |
|---|---|---|
| `k*(x > 1)` | a condition is not a number | `k if x > 1 else 0` |
| `x if x else 0` | after `if` comes a condition | `x if x > 0 else 0` |
| `x > 1 & y < 2` | join conditions with `and` | `x > 1 and y < 2` |
| `besselj(x)` | function not allowed (list given) | — |
| `k[t]` | `k` has one value per level of axis `s` | `k[s]` |
| `grad(S, x)` in a reaction | only in an advection velocity along `x` | `m.advect(B, axis=x, velocity=chi*grad(S, x))` |
| `x = 3`, `[1, 2]`, `"text"` | cannot read the formula / not allowed in a formula | a definition, a parameter |
| a name `lambda`, `if`, `and` | a reserved word; choose another (e.g. `lambda_`) | `lambda_` |
| `g_mx` (typo) | unknown name; did you mean `g_max`? | — |

## 6. Not part of the formula language, by design

- **Derivatives** are operators (`m.advect`, `m.diffuse`; `d_a(v*F)` in PDE notation), never
  formulas, so conservation can be guaranteed.
- **Integrals and sums** are declared (`m.integral`) and used by name; PDE notation's inline
  `int_z(...)` becomes such a declaration.
- **Delays** are transport along an age or maturation axis (exact, and the familiar transit
  compartments are its `erlang` form).
- **Statements** (`A = ...`) are definitions, one formula per name.
- **Random effects and residual error** belong to the estimation engine.

## 7. From NONMEM and Monolix

| NONMEM | Monolix | TenQSP formula |
|---|---|---|
| `CL = THETA(1)*EXP(ETA(1))` | `CL = CL_pop*exp(eta)` | a parameter `CL` (individual values come from the estimation engine) |
| `X**2` | `x^2` | `x^2` or `x**2` |
| `.GT. .GE. .EQ. .NE.` | `> >= == ~=` | `> >= == !=` |
| `.AND. .OR. .NOT.` | `& | ~` | `and or not` |
| `IF (C.GT.MIC) THEN; K=K1; ELSE; K=K2; ENDIF` | `if C > MIC  k = k1  else  k = k2  end` | `m.define("k", "k1 if C > MIC else k2")` |
| `PHI(X)`, `GAMLN(X)`, `MOD(A,B)` | `normcdf(x)`, `gammaln(x)`, `rem(a,b)` | `normcdf(x)`, `gammaln(x)`, `rem(a,b)` |
| `DADT(1) = -KA*A(1)` | `ddt_A = -ka*A` | `m.ode(A, -ka*A)` (`d/dt(A) = -ka*A` in PDE notation) |
| transit compartments `DADT(2)=KTR*A(1)-KTR*A(2) ...` | same | an age axis with `scheme="erlang"` |

## 8. Outside a model

`tq.formula("CL_pop * (WT/70)^0.75")` reads one formula with this grammar and these
functions without a model (TenQSP 0.7.0). Every name is then one value given by the caller:
`f.names` lists them and `f.jax(["CL_pop", "WT"])` evaluates the formula as a JAX function.
Shape functions, `grad`, `k[s]` and a field at one end need a model's axes and are refused.
