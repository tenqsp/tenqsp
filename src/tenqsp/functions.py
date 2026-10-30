# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Functions for formulas: ``tq.exp``, ``tq.sigmoid``, ``tq.max``, ... and the shape functions
``tq.normal``, ``tq.gamma``, .... Each returns a ``Formula``.

They live in their own module because ``abs``, ``max`` and ``min`` deliberately share their names
with Python built-ins (``tq.max`` is the formula version).
"""

from __future__ import annotations

from collections.abc import Mapping

import sympy as sp

from .expr import FUNCTIONS, SHAPES, FormulaError, make_shape
from .model import Axis, Formula, Name, as_formula

__all__ = [
    "exp",
    "log",
    "sqrt",
    "abs",
    "tanh",
    "max",
    "min",
    "sigmoid",
    "normal",
    "lognormal",
    "gamma",
    "exponential",
    "beta",
    "uniform",
    "point",
    "grad",
    "log10",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "logit",
    "normcdf",
    "probit",
    "erf",
    "gammaln",
    "factorial",
    "factln",
    "floor",
    "ceil",
    "rem",
]


def _text(x) -> str:
    return as_formula(x).text


def _call(fname: str, *args) -> Formula:
    return Formula(f"{fname}({', '.join(_text(a) for a in args)})")


def exp(x) -> Formula:
    """Exponential, ``e**x``."""
    return _call("exp", x)


def log(x) -> Formula:
    """Natural logarithm."""
    return _call("log", x)


def sqrt(x) -> Formula:
    """Square root."""
    return _call("sqrt", x)


def abs(x) -> Formula:  # noqa: A001  (tq.abs, as in the formula grammar)
    """Absolute value, cell by cell."""
    return _call("abs", x)


def tanh(x) -> Formula:
    """Hyperbolic tangent."""
    return _call("tanh", x)


def sigmoid(x) -> Formula:
    """``1 / (1 + exp(-x))``, the inverse of logit (Monolix invlogit)."""
    return _call("sigmoid", x)


def max(*args) -> Formula:  # noqa: A001
    """Larger of two or more values, cell by cell."""
    if len(args) < 2:
        raise TypeError("tq.max needs at least two values.")
    return _call("max", *args)


def min(*args) -> Formula:  # noqa: A001
    """Smaller of two or more values, cell by cell."""
    if len(args) < 2:
        raise TypeError("tq.min needs at least two values.")
    return _call("min", *args)


def _one(fname: str, doc: str):
    def f(x) -> Formula:
        return _call(fname, x)

    f.__name__ = f.__qualname__ = fname
    f.__doc__ = doc
    return f


log10 = _one("log10", "Base-10 logarithm.")
sin = _one("sin", "Sine (radians).")
cos = _one("cos", "Cosine (radians).")
tan = _one("tan", "Tangent (radians).")
asin = _one("asin", "Inverse sine.")
acos = _one("acos", "Inverse cosine.")
atan = _one("atan", "Inverse tangent.")
sinh = _one("sinh", "Hyperbolic sine.")
cosh = _one("cosh", "Hyperbolic cosine.")
logit = _one("logit", "``log(p / (1 - p))``.")
normcdf = _one("normcdf", "Standard normal cumulative distribution (NONMEM PHI, R pnorm).")
probit = _one("probit", "Inverse of normcdf (R qnorm).")
erf = _one("erf", "Error function.")
gammaln = _one("gammaln", "Log of the gamma function (NONMEM GAMLN, R lgamma).")
factorial = _one("factorial", "``gamma(x + 1)``.")
factln = _one("factln", "Log of the factorial, ``gammaln(x + 1)``.")
floor = _one("floor", "Round down (a jump at every whole number: see the validator's note).")
ceil = _one("ceil", "Round up (a jump at every whole number).")


def atan2(y, x) -> Formula:
    """Angle of the point (x, y), in (-pi, pi]."""
    return _call("atan2", y, x)


def rem(a, b) -> Formula:
    """Remainder of a / b with the sign of a (NONMEM MOD, Monolix rem); jumps at multiples
    of b."""
    return _call("rem", a, b)


# every grammar function has a tq.* helper of the same name
assert set(FUNCTIONS) <= set(globals()), set(FUNCTIONS) - set(globals())


def _shape(fname: str, x, kwargs: Mapping) -> Formula:
    if isinstance(x, Axis) and x.type != "continuous":
        raise FormulaError(
            f"{fname}: '{x.name}' is a categorical axis; shapes are for continuous axes. "
            "Use split= in m.init for fractions over categories."
        )
    # Check the keyword combination (and plain-number values) now, with a clear message.
    probe = {}
    for key, val in kwargs.items():
        f = as_formula(val)
        try:
            probe[key] = sp.Float(float(f.text))
        except ValueError:
            probe[key] = sp.Symbol("v", positive=True)
    make_shape(fname, sp.Symbol("x"), probe, f"in {fname}(...)")
    parts = [_text(x)] + [f"{k}={_text(v)}" for k, v in kwargs.items()]
    return Formula(f"{fname}({', '.join(parts)})")


def normal(x, **kwargs) -> Formula:
    """Normal shape over a continuous axis: ``mean`` with ``sd``, ``var`` or ``cv``."""
    return _shape("normal", x, kwargs)


def lognormal(x, **kwargs) -> Formula:
    """Lognormal shape: ``meanlog``, ``median`` or ``mean``, with ``sdlog`` or ``cv``."""
    return _shape("lognormal", x, kwargs)


def gamma(x, **kwargs) -> Formula:
    """Gamma shape: ``shape`` with ``rate`` or ``scale``, or ``mean`` with ``sd``/``var``/``cv``."""
    return _shape("gamma", x, kwargs)


def exponential(x, **kwargs) -> Formula:
    """Exponential shape: ``rate`` or ``mean``."""
    return _shape("exponential", x, kwargs)


def beta(x, **kwargs) -> Formula:
    """Beta shape on [0, 1]: ``alpha`` and ``beta``, or ``mean`` with ``sd``/``var``/
    ``concentration``."""
    return _shape("beta", x, kwargs)


def uniform(x, **kwargs) -> Formula:
    """Flat shape, optionally between ``lower`` and ``upper`` (default: the whole axis)."""
    return _shape("uniform", x, kwargs)


def point(x, **kwargs) -> Formula:
    """Everything in the cell containing ``at``."""
    return _shape("point", x, kwargs)


assert set(SHAPES) == {"normal", "lognormal", "gamma", "exponential", "beta", "uniform", "point"}


def grad(field, axis) -> Formula:
    """Gradient of a field along a continuous axis, for a velocity that follows it
    (chemotaxis): ``m.advect(B, axis=x, velocity=chi * tq.grad(S, x))``. It is taken at the cell
    faces, where velocities live, so it can only be used in an advection velocity along the
    same axis."""
    if not (isinstance(field, Name) and field.what == "field") and not isinstance(field, str):
        raise TypeError(f"grad: the first argument must be a field, not {field!r}.")
    if isinstance(axis, Axis):
        if axis.type != "continuous":
            raise ValueError(
                f"grad: '{axis.name}' is a categorical axis; a gradient needs a continuous axis."
            )
        axis = axis.name
    if not isinstance(axis, str):
        raise TypeError(f"grad: the second argument must be an axis, not {axis!r}.")
    name = field.name if isinstance(field, Name) else field
    return Formula(f"grad({name}, {axis})")
