# Formula functions

Functions used in formulas built in Python, e.g. `tq.sigmoid((a - tau)/s)`. Each returns a
formula, written into the model as the function of the same name in the notation (see the
[Notation reference](../notation.md#functions) for the names NONMEM, Monolix and R use).

## Mathematical functions

Each takes formulas (or numbers) and returns a formula, evaluated cell by cell.

| Python | Meaning |
|---|---|
| `tq.exp(x)` | Exponential, `e**x`. |
| `tq.log(x)` | Natural logarithm. |
| `tq.log10(x)` | Base-10 logarithm. |
| `tq.sqrt(x)` | Square root. |
| `tq.abs(x)` | Absolute value, cell by cell. |
| `tq.max(*args)` | Larger of two or more values, cell by cell. |
| `tq.min(*args)` | Smaller of two or more values, cell by cell. |
| `tq.sin(x)` | Sine (radians). |
| `tq.cos(x)` | Cosine (radians). |
| `tq.tan(x)` | Tangent (radians). |
| `tq.asin(x)` | Inverse sine. |
| `tq.acos(x)` | Inverse cosine. |
| `tq.atan(x)` | Inverse tangent. |
| `tq.atan2(y, x)` | Angle of the point (x, y), in (-pi, pi]. |
| `tq.sinh(x)` | Hyperbolic sine. |
| `tq.cosh(x)` | Hyperbolic cosine. |
| `tq.tanh(x)` | Hyperbolic tangent. |
| `tq.logit(x)` | `log(p / (1 - p))`. |
| `tq.sigmoid(x)` | `1 / (1 + exp(-x))`, the inverse of logit (Monolix invlogit). |
| `tq.normcdf(x)` | Standard normal cumulative distribution (NONMEM PHI, R pnorm). |
| `tq.probit(x)` | Inverse of normcdf (R qnorm). |
| `tq.erf(x)` | Error function. |
| `tq.gammaln(x)` | Log of the gamma function (NONMEM GAMLN, R lgamma). |
| `tq.factorial(x)` | `gamma(x + 1)`. |
| `tq.factln(x)` | Log of the factorial, `gammaln(x + 1)`. |
| `tq.floor(x)` | Round down (a jump at every whole number: see the validator's note). |
| `tq.ceil(x)` | Round up (a jump at every whole number). |
| `tq.rem(a, b)` | Remainder of a / b with the sign of a (NONMEM MOD, Monolix rem); jumps at multiples of b. |
| `tq.grad(field, axis)` | Gradient of a field along a continuous axis, for a velocity that follows it (chemotaxis): `m.advect(B, axis=x, velocity=chi * tq.grad(S, x))`. It is taken at the cell faces, where velocities live, so it can only be used in an advection velocity along the same axis. |

## Shapes

Shapes spread an amount over a continuous axis (starting values, doses into fields); see
[Starting values](../../guide/initial.md).

::: tenqsp.functions
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - normal
        - lognormal
        - gamma
        - exponential
        - beta
        - uniform
        - point
