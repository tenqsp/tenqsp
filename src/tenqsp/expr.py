# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The formula grammar (ADR 0002): text -> SymPy, safely, with plain-language errors.

A formula is read with Python's ``ast`` module and turned into SymPy by walking an allow-list
of syntax. Nothing is passed to ``eval``/``exec``. Allowed:

- numbers, names, ``+ - * /``, power as ``**`` or ``^``, unary minus;
- functions (the set of Monolix's Mlxtran, which covers NONMEM's built-ins; ADR 0005):
  ``exp log log10 sqrt abs max min``; trigonometric ``sin cos tan asin acos atan atan2`` and
  hyperbolic ``sinh cosh tanh``; ``logit`` and its inverse ``sigmoid``;
  the standard normal ``normcdf`` (NONMEM's PHI) and its inverse ``probit``; ``erf``;
  ``gammaln``, ``factorial``, ``factln``; and the non-smooth ``floor ceil rem`` (``rem``:
  remainder with the sign of the dividend, as NONMEM's MOD and Monolix's rem). Each function
  has one name; a name another program uses (``pnorm``, ``PHI``, ``mod``, ...) is refused
  with the TenQSP name to use (``ELSEWHERE``);
- shape functions ``normal lognormal gamma exponential beta uniform point``. These take the
  axis coordinate as their only positional argument and their parameters as keywords with
  standard statistical names; see ``SHAPES``;
- ``vector[axis]``: a parameter with one value per level of a categorical axis;
- conditionals (ADR 0006): ``v1 if c1 else v2 if c2 else v3`` (first true condition wins; the
  final ``else`` is required by the syntax). Conditions are comparisons ``< <= > >= == !=``
  (chains such as ``a < x <= b`` allowed) joined by ``and``, ``or``, ``not``. A condition is
  never a number (``k*(x > 1)`` is refused), and ``& | ~`` are refused with a hint (in Python
  syntax ``x > 1 & y < 2`` would mean ``x > (1 & y) < 2``).

Integrals and sums over axes are *not* part of a formula. They are declared separately (IR
``IntegralSpec``) and referred to by name; the PDE-notation front end lifts inline
``int_z(...)`` / ``sum_s(...)`` into such declarations.
"""

from __future__ import annotations

import ast
import difflib
import math
from collections.abc import Callable, Mapping

import sympy as sp


class FormulaError(ValueError):
    """A formula cannot be accepted. The message says what is wrong, where, and how to fix it."""


def sigmoid(x):
    return 1 / (1 + sp.exp(-x))


class _Special(sp.Function):
    """A special function kept unevaluated (SymPy would otherwise rewrite it, e.g.
    ``erfc(-x) -> 2 - erfc(x)``, losing precision in the tails). ``_scipy`` evaluates it on
    numbers (validator checks, ``evalf``)."""

    _scipy: Callable = staticmethod(lambda *a: math.nan)

    @classmethod
    def eval(cls, *args):  # keep unevaluated
        return None

    def _eval_evalf(self, prec):
        if all(a.is_number for a in self.args):
            return sp.Float(float(self._scipy(*(float(a) for a in self.args))), prec)
        return None


def _scipy_special(name: str) -> Callable:
    def f(*args):
        import scipy.special as sc

        return getattr(sc, name)(*args)

    return f


class NormCdf(_Special):
    """Standard normal cumulative distribution (NONMEM PHI, Monolix normcdf)."""

    nargs = 1
    _scipy = staticmethod(_scipy_special("ndtr"))
    _imp_ = staticmethod(_scipy_special("ndtr"))


class Probit(_Special):
    """Inverse of ``NormCdf`` (Monolix probit)."""

    nargs = 1
    _scipy = staticmethod(_scipy_special("ndtri"))
    _imp_ = staticmethod(_scipy_special("ndtri"))


class Rem(_Special):
    """Remainder of a / b with the sign of a (C fmod; NONMEM MOD, Monolix rem)."""

    nargs = 2
    _scipy = staticmethod(math.fmod)
    _imp_ = staticmethod(lambda a, b: __import__("numpy").fmod(a, b))


def _log10(x):
    return sp.log(x) / sp.log(10)


def _logit(x):
    return sp.log(x / (1 - x))


FUNCTIONS: dict[str, Callable] = {
    "exp": sp.exp,
    "log": sp.log,
    "log10": _log10,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "max": sp.Max,
    "min": sp.Min,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "atan2": sp.atan2,  # atan2(y, x): angle of the point (x, y)
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "logit": _logit,
    "sigmoid": sigmoid,
    "normcdf": NormCdf,
    "probit": Probit,
    "erf": sp.erf,
    "gammaln": sp.loggamma,
    "factorial": lambda x: sp.gamma(_bounded(x, "factorial", 170) + 1),
    "factln": lambda x: sp.loggamma(x + 1),
    "floor": sp.floor,
    "ceil": sp.ceiling,
    "rem": Rem,
}
# names other programs use for these functions: refused, with the TenQSP name to use
ELSEWHERE: dict[str, tuple[str, str]] = {
    "pnorm": ("normcdf", "R's pnorm"),
    "phi": ("normcdf", "NONMEM's PHI"),
    "PHI": ("normcdf", "NONMEM's PHI"),
    "qnorm": ("probit", "R's qnorm"),
    "norminv": ("probit", "MATLAB's norminv"),
    "invlogit": ("sigmoid", "Monolix's invlogit"),
    "expit": ("sigmoid", "SciPy's expit"),
    "lgamma": ("gammaln", "R's lgamma"),
    "GAMLN": ("gammaln", "NONMEM's GAMLN"),
    "mod": ("rem", "NONMEM's MOD"),
    "MOD": ("rem", "NONMEM's MOD"),
    "EXP": ("exp", "NONMEM's EXP"),
    "LOG": ("log", "NONMEM's LOG"),
    "SQRT": ("sqrt", "NONMEM's SQRT"),
}
# number of arguments: exactly one unless listed; negative = at least that many
ARITY: dict[str, int] = {"max": -2, "min": -2, "atan2": 2, "rem": 2}
# non-smooth: piecewise constant or with jumps; zero or undefined derivative at the jumps,
# which a stiff solver and gradient-based fitting both feel (the validator notes their use)
NON_SMOOTH = frozenset({"floor", "ceil", "rem"})
NON_SMOOTH_TYPES: dict[type, str] = {sp.floor: "floor", sp.ceiling: "ceil", Rem: "rem"}


# ----------------------------------------------------------------------------- gradients


class Grad(sp.Function):
    """``grad(F, x)``: the gradient of field ``F`` along continuous axis ``x``, evaluated at
    the cell faces. Kept unevaluated; only an advection velocity along ``x`` may
    use it (the validator checks this), e.g. chemotaxis ``chi * grad(S, x)``."""

    nargs = 2

    @classmethod
    def eval(cls, *args):  # keep unevaluated
        return None

    @property
    def field(self) -> str:
        return self.args[0].name

    @property
    def axis(self) -> str:
        return self.args[1].name


def grads_used(expr: sp.Expr) -> list[Grad]:
    return [f for f in expr.atoms(sp.Function) if isinstance(f, Grad)]


# ----------------------------------------------------------------------------- shape functions
# Each shape is stored canonically as an unevaluated SymPy function of
# (coordinate, canonical parameters...). The compiler turns it into exact cell masses.


class Shape(sp.Function):
    """Base for canonical shape functions; never evaluated symbolically."""

    nargs = None

    @classmethod
    def eval(cls, *args):  # keep unevaluated
        return None


def _shape_class(name: str, params: tuple[str, ...]) -> type[Shape]:
    cls = type(f"{name}_shape", (Shape,), {"params": params, "shape_name": name})
    return cls


# canonical parameterisations
SHAPE_CLASSES: dict[str, type[Shape]] = {
    "normal": _shape_class("normal", ("mean", "sd")),
    "lognormal": _shape_class("lognormal", ("meanlog", "sdlog")),
    "gamma": _shape_class("gamma", ("shape", "rate")),
    "exponential": _shape_class("exponential", ("rate",)),
    "beta": _shape_class("beta", ("alpha", "beta")),
    "uniform": _shape_class("uniform", ("lower", "upper")),
    "point": _shape_class("point", ("at",)),
}

# Accepted keyword combinations for each shape and how they map to the canonical parameters.
# Each entry: (frozenset of keywords, converter returning the canonical parameter tuple).
SHAPES: dict[str, list[tuple[frozenset[str], Callable]]] = {
    "normal": [
        (frozenset({"mean", "sd"}), lambda k: (k["mean"], k["sd"])),
        (frozenset({"mean", "var"}), lambda k: (k["mean"], sp.sqrt(k["var"]))),
        (frozenset({"mean", "cv"}), lambda k: (k["mean"], k["cv"] * k["mean"])),
    ],
    "lognormal": [
        (frozenset({"meanlog", "sdlog"}), lambda k: (k["meanlog"], k["sdlog"])),
        (frozenset({"median", "sdlog"}), lambda k: (sp.log(k["median"]), k["sdlog"])),
        (
            frozenset({"median", "cv"}),
            lambda k: (sp.log(k["median"]), sp.sqrt(sp.log(1 + k["cv"] ** 2))),
        ),
        (frozenset({"meanlog", "cv"}), lambda k: (k["meanlog"], sp.sqrt(sp.log(1 + k["cv"] ** 2)))),
        (
            frozenset({"mean", "cv"}),
            lambda k: (
                sp.log(k["mean"]) - sp.log(1 + k["cv"] ** 2) / 2,
                sp.sqrt(sp.log(1 + k["cv"] ** 2)),
            ),
        ),
        (
            frozenset({"mean", "sdlog"}),
            lambda k: (sp.log(k["mean"]) - k["sdlog"] ** 2 / 2, k["sdlog"]),
        ),
    ],
    "gamma": [
        (frozenset({"shape", "rate"}), lambda k: (k["shape"], k["rate"])),
        (frozenset({"shape", "scale"}), lambda k: (k["shape"], 1 / k["scale"])),
        (
            frozenset({"mean", "sd"}),
            lambda k: ((k["mean"] / k["sd"]) ** 2, k["mean"] / k["sd"] ** 2),
        ),
        (frozenset({"mean", "var"}), lambda k: (k["mean"] ** 2 / k["var"], k["mean"] / k["var"])),
        (frozenset({"mean", "cv"}), lambda k: (1 / k["cv"] ** 2, 1 / (k["cv"] ** 2 * k["mean"]))),
    ],
    "exponential": [
        (frozenset({"rate"}), lambda k: (k["rate"],)),
        (frozenset({"mean"}), lambda k: (1 / k["mean"],)),
    ],
    "beta": [
        (frozenset({"alpha", "beta"}), lambda k: (k["alpha"], k["beta"])),
        (
            frozenset({"mean", "concentration"}),
            lambda k: (k["mean"] * k["concentration"], (1 - k["mean"]) * k["concentration"]),
        ),
        (frozenset({"mean", "var"}), lambda k: _beta_from_moments(k["mean"], k["var"])),
        (frozenset({"mean", "sd"}), lambda k: _beta_from_moments(k["mean"], k["sd"] ** 2)),
    ],
    "uniform": [
        (frozenset(), lambda k: (-sp.oo, sp.oo)),
        (frozenset({"lower"}), lambda k: (k["lower"], sp.oo)),
        (frozenset({"upper"}), lambda k: (-sp.oo, k["upper"])),
        (frozenset({"lower", "upper"}), lambda k: (k["lower"], k["upper"])),
    ],
    "point": [
        (frozenset({"at"}), lambda k: (k["at"],)),
    ],
}


def _beta_from_moments(mean, var):
    nu = mean * (1 - mean) / var - 1
    return (mean * nu, (1 - mean) * nu)


# Numeric sanity checks applied when an argument is a plain number (symbolic arguments are
# checked by the compiler with actual parameter values).
_POSITIVE = {
    "sd",
    "var",
    "cv",
    "sdlog",
    "median",
    "shape",
    "rate",
    "scale",
    "concentration",
    "alpha",
    "beta",
}


def _num(v) -> str:
    """A number for a message: -1 and 0.05, not -1.00000000000000."""
    return f"{float(v):g}"


def _check_numeric(fname: str, kw: Mapping[str, sp.Expr], where: str) -> None:
    for key, val in kw.items():
        if val.is_number and key in _POSITIVE and not bool(val > 0):
            raise FormulaError(f"{fname}: '{key}' must be > 0 (got {_num(val)}) {where}.")
    if fname == "lognormal" and "mean" in kw and kw["mean"].is_number and not bool(kw["mean"] > 0):
        raise FormulaError(f"lognormal: 'mean' must be > 0 {where}.")
    if (
        fname == "exponential"
        and "mean" in kw
        and kw["mean"].is_number
        and not bool(kw["mean"] > 0)
    ):
        raise FormulaError(f"exponential: 'mean' must be > 0 {where}.")
    if fname == "beta" and "mean" in kw and kw["mean"].is_number:
        m = kw["mean"]
        if not (0 < m < 1):
            raise FormulaError(f"beta: 'mean' must be between 0 and 1 (got {_num(m)}) {where}.")
        var = kw.get("var", kw["sd"] ** 2 if "sd" in kw else None)
        if var is not None and var.is_number and not bool(var < m * (1 - m)):
            raise FormulaError(
                f"beta: variance {_num(var)} is impossible for mean {_num(m)}; it must be below "
                f"mean*(1-mean) = {_num(m * (1 - m))} {where}."
            )


# Display order of keywords: location / scale first, then spread.
_KEY_ORDER = (
    "mean",
    "median",
    "meanlog",
    "shape",
    "alpha",
    "beta",
    "lower",
    "upper",
    "at",
    "rate",
    "scale",
    "sd",
    "var",
    "cv",
    "sdlog",
    "concentration",
)


def _accepted(fname: str) -> str:
    """Accepted keyword combinations, shown as usable examples: gamma(x, mean=..., cv=...)."""
    forms = []
    for combo, _ in SHAPES[fname]:
        keys = sorted(combo, key=_KEY_ORDER.index)
        forms.append(f"{fname}(x" + "".join(f", {k}=..." for k in keys) + ")")
    return "; ".join(forms)


def make_shape(fname: str, coord: sp.Expr, kw: Mapping[str, sp.Expr], where: str = "") -> sp.Expr:
    """Build the canonical shape expression from keyword arguments, validating the combination."""
    given = frozenset(kw)
    for combo, conv in SHAPES[fname]:
        if combo == given:
            _check_numeric(fname, kw, where)
            return SHAPE_CLASSES[fname](coord, *conv(kw))
    known = set().union(*(c for c, _ in SHAPES[fname]))
    unknown = sorted(given - known)
    if unknown:
        hint = difflib.get_close_matches(unknown[0], sorted(known), n=1)
        extra = f" Did you mean '{hint[0]}'?" if hint else ""
        raise FormulaError(
            f"{fname}: unknown argument '{unknown[0]}' {where}.{extra} "
            f"Accepted: {_accepted(fname)}."
        )
    raise FormulaError(
        f"{fname}: the arguments {sorted(given) or '(none)'} are not a valid combination {where}. "
        f"Give exactly one location and one spread argument. Accepted: {_accepted(fname)}."
    )


# ----------------------------------------------------------------------------- parser

_COMPARE = {
    ast.Lt: sp.Lt,
    ast.LtE: sp.Le,
    ast.Gt: sp.Gt,
    ast.GtE: sp.Ge,
    ast.Eq: sp.Eq,
    ast.NotEq: sp.Ne,
}
_LOGIC_HINT = {
    ast.BitAnd: "'&' is not allowed; join conditions with 'and' (x > a and x < b)",
    ast.BitOr: "'|' is not allowed; join conditions with 'or' (x < a or x > b)",
}


def is_condition(e) -> bool:
    """A true/false value (a comparison, and/or/not of comparisons, or true/false)."""
    # (SymPy symbols are also Booleans; a condition is a Boolean that is not an expression)
    return isinstance(e, sp.logic.boolalg.Boolean) and not isinstance(e, sp.Expr)


_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: _power(a, b),
}

_MAX_EXACT_POWER = 1024


def _bounded(x, fname: str, limit: float):
    """A numeric argument beyond ``limit`` would be evaluated exactly (and slowly) by SymPy,
    and overflows double precision anyway: refused."""
    if x.is_number and x.is_extended_real and float(x) > limit:
        raise FormulaError(f"{fname}({x}) is too large for double precision (limit {limit:g}).")
    return x


def _power(a, b):
    """``a**b``. Numbers are powered exactly only for modest exponents: ``9**9**9`` would
    otherwise build an integer with hundreds of millions of digits while parsing."""
    if not (a.is_number and b.is_number):
        return a**b
    try:
        approx = math.pow(float(a), float(b))
    except OverflowError:
        raise FormulaError(f"{a}**{b} is too large for double precision.") from None
    except (TypeError, ValueError, ZeroDivisionError):
        return a**b  # complex or 0**-1: kept, and reported by the validator
    if isinstance(b, sp.Rational) and max(abs(b.p), b.q) > _MAX_EXACT_POWER:
        return sp.Float(approx)
    return a**b


def end_key(field: str, axis: str, side: str) -> str:
    """Name of the value of ``field`` at one end of ``axis`` (``side``: lower, upper, or
    ``#<coordinate>`` for an alias by position). The symbol prints as the notation reads it,
    ``D(z=lower)``."""
    return f"{field}({axis}={side})"


def _near_end(names, field: str, ax: str, num: float) -> str | None:
    """The key of the end of ``ax`` whose coordinate is ``num`` up to rounding (1e-9 of the
    axis length), or None."""
    prefix = f"{field}({ax}=#"
    coords = {}
    for k in names:
        if k.startswith(prefix) and k.endswith(")"):
            try:
                coords[k] = float(k[len(prefix) : -1])
            except ValueError:
                continue
    if len(coords) != 2:
        return None
    lo, hi = sorted(coords.values())
    tol = 1e-9 * (hi - lo)
    return next((k for k, c in coords.items() if abs(c - num) <= tol), None)


def parse(text: str, names: Mapping[str, sp.Basic]) -> sp.Expr:
    """Parse one formula. ``names`` maps every name the formula may use to a SymPy object
    (``sp.Symbol`` for scalars, coordinates and states; ``sp.IndexedBase`` for vector
    parameters). Raises ``FormulaError`` with a plain-language message.
    """
    if not isinstance(text, str):
        text = str(text)
    src = text.strip()
    if not src:
        raise FormulaError("Empty formula.")
    # ^ means power, as in many modelling languages. Python reads ^ as XOR, which binds more
    # loosely than + - * /, so it is rewritten before parsing (the grammar has no strings, so
    # every ^ is an operator).
    src = src.replace("^", "**")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"Cannot read formula {src!r}: {e.msg} at column {e.offset}.") from None

    def where(node) -> str:
        return f"at column {getattr(node, 'col_offset', 0) + 1} of {src!r}"

    def value(node):
        """A number-valued part: a condition is refused here."""
        v = walk(node)
        if is_condition(v):
            raise FormulaError(
                f"'{ast.unparse(node)}' is a condition (true or false), not a number "
                f"{where(node)}. Use it to choose between values: "
                f"'v1 if {ast.unparse(node)} else v2'."
            )
        return v

    def condition(node):
        c = walk(node)
        if not is_condition(c):
            raise FormulaError(
                f"After 'if' comes a condition such as 'x > 1' (comparisons joined by and, "
                f"or, not), not '{ast.unparse(node)}' {where(node)}."
            )
        return c

    def end_value(node):
        """``F(z=lower)``, ``F(z=upper)`` or ``F(z=<number at an end>)``: field F read at one
        end of its continuous axis z (IR 0.10)."""
        fname = node.func.id
        usage = (
            f"'{ast.unparse(node)}': a field is read at one end of an axis as {fname}(z=lower) "
            f"or {fname}(z=upper) (or the coordinate of that end, {fname}(z=0)) {where(node)}"
        )
        if node.args or len(node.keywords) != 1 or node.keywords[0].arg is None:
            raise FormulaError(usage + ".")
        ax, at = node.keywords[0].arg, node.keywords[0].value
        if isinstance(at, ast.Name) and at.id in ("lower", "upper"):
            key = end_key(fname, ax, at.id)
        else:
            try:
                num = float(ast.literal_eval(at))
            except (ValueError, TypeError, SyntaxError):
                raise FormulaError(usage + ".") from None
            key = end_key(fname, ax, f"#{num!r}")
            if key not in names:  # the end's coordinate up to rounding (relative to the axis)
                key = _near_end(names, fname, ax, num) or key
        if key in names:
            return names[key]
        if end_key(fname, ax, "lower") in names:
            raise FormulaError(
                f"'{ast.unparse(node)}': {ast.unparse(at)} is not an end of axis '{ax}' "
                f"{where(node)}; write {fname}({ax}=lower) or {fname}({ax}=upper)."
            )
        raise FormulaError(usage + f"; '{fname}' is not a field with a continuous axis '{ax}'.")

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            if isinstance(node.value, float) and not math.isfinite(node.value):
                raise FormulaError(f"The number {ast.unparse(node)} is not finite {where(node)}.")
            return sp.Float(node.value) if isinstance(node.value, float) else sp.Integer(node.value)
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            if node.id in FUNCTIONS or node.id in SHAPES or node.id == "grad":
                raise FormulaError(
                    f"'{node.id}' is a function; call it as {node.id}(...) {where(node)}."
                )
            hint = difflib.get_close_matches(node.id, list(names), n=1)
            extra = f" Did you mean '{hint[0]}'?" if hint else ""
            raise FormulaError(f"Unknown name '{node.id}' {where(node)}.{extra}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](value(node.left), value(node.right))
        if isinstance(node, ast.BinOp) and type(node.op) in _LOGIC_HINT:
            raise FormulaError(f"{_LOGIC_HINT[type(node.op)]} {where(node)}.")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = value(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            raise FormulaError(f"'~' is not allowed; write 'not' {where(node)}.")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return sp.Not(condition(node.operand))
        if isinstance(node, ast.Compare):
            terms = [value(node.left), *(value(c) for c in node.comparators)]
            parts = []
            for op, left, right in zip(node.ops, terms[:-1], terms[1:], strict=True):
                if type(op) not in _COMPARE:
                    raise FormulaError(
                        f"'{ast.unparse(node)}': only < <= > >= == != compare numbers "
                        f"{where(node)}."
                    )
                parts.append(_COMPARE[type(op)](left, right))
            return parts[0] if len(parts) == 1 else sp.And(*parts)
        if isinstance(node, ast.BoolOp):
            parts = [condition(v) for v in node.values]
            return sp.And(*parts) if isinstance(node.op, ast.And) else sp.Or(*parts)
        if isinstance(node, ast.IfExp):
            cases = []
            while isinstance(node, ast.IfExp):  # a chain: v1 if c1 else v2 if c2 else v3
                cases.append((value(node.body), condition(node.test)))
                node = node.orelse
            cases.append((value(node), sp.true))
            return sp.Piecewise(*cases)
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            raise FormulaError(
                f"'{node.value}' is not a number {where(node)}; for 'otherwise' write the "
                "final value after the last 'else'."
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FormulaError(f"'{ast.unparse(node.func)}' cannot be called {where(node)}.")
            fname = node.func.id
            if (
                fname in names
                and not (node.args and (fname in FUNCTIONS or fname in SHAPES))
                and any(k.startswith(fname + "(") for k in names)
            ):
                return end_value(node)  # a field read at one end of an axis
            if fname in SHAPES:
                if len(node.args) != 1:
                    raise FormulaError(
                        f"{fname}: give the axis as the only positional argument and the rest as "
                        f"keywords {where(node)}. Accepted: {_accepted(fname)}."
                    )
                if not isinstance(node.args[0], ast.Name):
                    raise FormulaError(
                        f"{fname}: the first argument must be an axis name, like {fname}(z, ...), "
                        f"not '{ast.unparse(node.args[0])}' {where(node)}."
                    )
                kw = {}
                for k in node.keywords:
                    if k.arg is None:
                        raise FormulaError(f"'**' is not allowed {where(node)}.")
                    kw[k.arg] = value(k.value)
                return make_shape(fname, walk(node.args[0]), kw, where(node))
            if fname == "grad":
                if (
                    len(node.args) != 2
                    or node.keywords
                    or not all(isinstance(x, ast.Name) for x in node.args)
                ):
                    raise FormulaError(
                        "grad takes a field and an axis, like grad(S, x) (the gradient of S "
                        f"along x) {where(node)}."
                    )
                return Grad(*(walk(x) for x in node.args))
            if fname in FUNCTIONS:
                if node.keywords:
                    raise FormulaError(f"{fname}() takes no keyword arguments {where(node)}.")
                want, got = ARITY.get(fname, 1), len(node.args)
                if (want > 0 and got != want) or (want < 0 and got < -want):
                    need = (
                        f"at least {-want} arguments"
                        if want < 0
                        else f"exactly {want} argument{'s' if want > 1 else ''}"
                    )
                    raise FormulaError(f"{fname}() takes {need}, not {got} {where(node)}.")
                return FUNCTIONS[fname](*[value(a) for a in node.args])
            if fname in ELSEWHERE:
                ours, theirs = ELSEWHERE[fname]
                raise FormulaError(
                    f"'{fname}' is not a TenQSP function {where(node)}: write {ours}(...) "
                    f"(the same function as {theirs})."
                )
            allowed = ", ".join(sorted([*FUNCTIONS, *SHAPES, "grad"]))
            raise FormulaError(
                f"Function '{fname}' is not allowed {where(node)}. Allowed: {allowed}."
            )
        if isinstance(node, ast.Subscript):
            base = walk(node.value)
            if not isinstance(base, sp.IndexedBase):
                raise FormulaError(
                    f"Only vector parameters can be indexed, '{ast.unparse(node.value)}' is not "
                    f"one {where(node)}."
                )
            idx = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            return base[tuple(value(i) for i in idx)]
        raise FormulaError(
            f"'{ast.unparse(node)}' is not allowed in a formula {where(node)}. Use numbers, names, "
            "+ - * / ** (or ^), the allowed functions, vector[axis], and conditionals "
            "'v1 if condition else v2'."
        )

    return value(tree.body)


def names_used(expr: sp.Expr) -> set[str]:
    """Every model name a parsed expression refers to (symbols and vector-parameter bases)."""
    out = {s.name for s in expr.free_symbols if isinstance(s, sp.Symbol)}
    for ind in expr.atoms(sp.Indexed):
        out.add(str(ind.base))
        out.discard(str(ind.base.label))
    return out


def shapes_used(expr: sp.Expr) -> list[Shape]:
    return [f for f in expr.atoms(sp.Function) if isinstance(f, Shape)]
