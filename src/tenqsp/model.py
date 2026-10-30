# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The Python API: build a model step by step; it produces the IR (``tenqsp.ir``).

Example:

    import tenqsp as tq

    m = tq.Model("phage_latency")
    age = m.axis("age", lower=0.0, upper=2.0, n=40, units="h", kind="one_way")
    B = m.compartment("B", init=1e6)
    P = m.compartment("P", init=1e7)
    I = m.field("I", axes=[age], init=0.0)
    k, beta, delta = m.params(k=1e-9, beta=50.0, delta=0.1)
    m.advect(I, axis=age, velocity=1.0, scheme="sharp")
    m.inflow(I, axis=age, at="lower", flux=k * B * P)
    ...

Formulas are written with ordinary Python operators on the objects the model returns
(parameters, fields, compartments, axes, definitions, integrals) and the functions in this module
in ``functions``
(``tq.exp``, ``tq.sigmoid``, ``tq.gamma(...)``, ...). A formula may also be given as text in the
ADR 0002 grammar, e.g. ``"k*B*P"``. Either way it is stored in the IR as formula text, so the
model can be saved, compared and read back.

Mistakes that can be seen immediately (a duplicate name, a field where a compartment is
expected, an unknown shape argument) raise at once. Everything else is checked by
``Model.validate``, which lists every problem in the model together.
"""

from __future__ import annotations

import ast
import keyword
import re
from collections.abc import Iterable, Mapping, Sequence
from itertools import product
from numbers import Real

import numpy as np

from .expr import FormulaError
from .grids import CategoricalAxis, ContinuousAxis
from .ir import (
    RESERVED_NAMES,
    AxisSpec,
    BioavailabilitySpec,
    BoundarySpec,
    CompartmentSpec,
    DefinitionSpec,
    DoseSpec,
    ExtinctionSpec,
    FieldSpec,
    InitSpec,
    IntegralSpec,
    LagSpec,
    ModelIR,
    ObservableSpec,
    ParamSpec,
    TermSpec,
    TransferSpec,
    TransitionSpec,
)
from .validate import Report, validate

__all__ = [
    "Model",
    "Formula",
    "Condition",
    "Slice",
    "cases",
    "Axis",
    "Name",
    "VectorParam",
    "Transitions",
    "as_formula",
]

# ----------------------------------------------------------------------------- formulas
# Precedence levels, used only to place parentheses when formula text is joined.
_RAW, _ADD, _MUL, _NEG, _POW, _ATOM = 0, 1, 2, 3, 4, 5
_SIMPLE = re.compile(r"[A-Za-z_]\w*(\[[A-Za-z_]\w*\])?|\d+(\.\d*)?([eE][-+]?\d+)?")


class Formula:
    """A formula, built with Python operators and stored as text (ADR 0002 grammar).

    ``str(f)`` shows the text. ``<``, ``<=``, ``>``, ``>=`` give a ``Condition``, used to
    choose between values with ``cases`` (``==``/``!=`` are ``tq.eq``/``tq.ne``, since
    Python's own equality is kept). A formula has no true/false value of its own: the model is
    evaluated later, for every cell at once.
    """

    # No __slots__: IPython autoreload (on in Positron) copies a reloaded class's attributes
    # onto the old class, and slot descriptors then refuse the old instances.
    __array_ufunc__ = None  # numpy scalars defer to our reflected operators

    def __init__(self, text: str, prec: int = _ATOM):
        self.text = text
        self.prec = prec

    # ---------------------------------------------------------------- display
    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Formula({self.text!r})"

    # ---------------------------------------------------------------- arithmetic
    def __add__(self, other):
        return _binary(self, other, "+")

    def __radd__(self, other):
        return _binary(other, self, "+")

    def __sub__(self, other):
        return _binary(self, other, "-")

    def __rsub__(self, other):
        return _binary(other, self, "-")

    def __mul__(self, other):
        return _binary(self, other, "*")

    def __rmul__(self, other):
        return _binary(other, self, "*")

    def __truediv__(self, other):
        return _binary(self, other, "/")

    def __rtruediv__(self, other):
        return _binary(other, self, "/")

    def __pow__(self, other):
        return _binary(self, other, "**")

    def __rpow__(self, other):
        return _binary(other, self, "**")

    def __neg__(self):
        a = as_formula(self)
        return Formula("-" + _paren(a, a.prec < _NEG), _NEG)

    def __pos__(self):
        return as_formula(self)

    # ---------------------------------------------------------------- conditions
    def __lt__(self, other):
        return _compare(self, other, "<")

    def __le__(self, other):
        return _compare(self, other, "<=")

    def __gt__(self, other):
        return _compare(self, other, ">")

    def __ge__(self, other):
        return _compare(self, other, ">=")

    # ---------------------------------------------------------------- misuse
    def __bool__(self):
        raise TypeError(
            f"The formula '{self}' has no true/false value: it is evaluated later, for every "
            "cell at once. To switch between values use tq.cases((condition, value), ..., "
            "otherwise=value)."
        )

    def _no_equality(self, other):
        raise TypeError(
            f"Formulas are not compared with == or != ('{self}'); for a condition use "
            "tq.eq(a, b) or tq.ne(a, b)."
        )

    __eq__ = __ne__ = _no_equality
    __hash__ = object.__hash__


class Condition:
    """A true/false formula, e.g. ``x > thr`` or ``(x > a) & (x <= b)``; used only to choose
    between values in ``cases``. Combine with ``&`` (and), ``|`` (or), ``~`` (not); stored
    as text with ``and``, ``or``, ``not``."""

    def __init__(self, text: str, kind: str):
        self.text = text
        self.kind = kind  # "compare", "not", "and" or "or": for parentheses only

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Condition({self.text!r})"

    def _join(self, other, op: str) -> Condition:
        if not isinstance(other, Condition):
            raise TypeError(f"'{op}' joins two conditions, not '{self}' and {other!r}.")
        parts = [c.text if c.kind != "or" or op == "or" else f"({c.text})" for c in (self, other)]
        return Condition(f"{parts[0]} {op} {parts[1]}", op)

    def __and__(self, other):
        return self._join(other, "and")

    def __or__(self, other):
        return self._join(other, "or")

    def __invert__(self):
        inner = self.text if self.kind in ("compare", "not") else f"({self.text})"
        return Condition(f"not {inner}", "not")

    def _not_a_number(self, *other):
        raise TypeError(
            f"'{self}' is a condition (true or false), not a number; use it to choose between "
            "values with tq.cases((condition, value), ..., otherwise=value)."
        )

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _not_a_number
    __truediv__ = __rtruediv__ = __pow__ = __rpow__ = __neg__ = _not_a_number
    __lt__ = __le__ = __gt__ = __ge__ = _not_a_number

    def __bool__(self):
        raise TypeError(
            f"The condition '{self}' has no true/false value yet: it is evaluated later, for "
            "every cell at once. Use it in tq.cases((condition, value), ..., otherwise=value). "
            "Join conditions with & and |, e.g. (a < x) & (x <= b), not 'a < x <= b' or 'and'."
        )

    __hash__ = object.__hash__


def _value_text(x) -> str:
    f = as_formula(x)
    return f"({f.text})" if f.prec <= _RAW else f.text


def _compare(a, b, op: str) -> Condition:
    return Condition(f"{_value_text(a)} {op} {_value_text(b)}", "compare")


def _condition_text(c) -> str:
    if isinstance(c, Condition):
        return c.text
    if isinstance(c, str) and c.strip():
        # one whole condition, kept whole by parentheses (the formula itself is checked when
        # the model is validated); text such as "a > 1 else 99 if b > 1" would otherwise
        # splice into the surrounding formula and change its structure
        try:
            body = ast.parse(c.strip().replace("^", "**"), mode="eval").body
        except SyntaxError:
            body = None
        is_cond = isinstance(body, (ast.Compare, ast.BoolOp)) or (
            isinstance(body, ast.UnaryOp) and isinstance(body.op, ast.Not)
        )
        if not is_cond:
            raise TypeError(
                f"tq.cases: {c!r} is not one condition. Write a comparison such as 'x > thr', "
                "or comparisons joined by and, or, not."
            )
        return f"({c.strip()})"
    raise TypeError(f"A condition is needed (e.g. x > thr), not {c!r}.")


def cases(*pairs, otherwise) -> Formula:
    """Choose a value by conditions; the first true condition wins:

        k = m.define("k", tq.cases((x > thr, k1),
                                   (x > thr2, k2),
                                   otherwise=k3))

    The same as the formula text ``"k1 if x > thr else k2 if x > thr2 else k3"``.
    ``otherwise`` is required: every cell always has a value.
    """
    if not pairs:
        raise TypeError("tq.cases needs at least one (condition, value) pair, then otherwise=.")
    parts = []
    for pair in pairs:
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise TypeError(f"tq.cases: give (condition, value) pairs, not {pair!r}.")
        cond, val = pair
        parts.append(f"{_value_text(val)} if {_condition_text(cond)} else")
    tail = as_formula(otherwise)
    tail_text = f"({tail.text})" if tail.prec < _RAW else tail.text
    return Formula(" ".join(parts) + " " + tail_text, _RAW)


def eq(a, b) -> Condition:
    """``a == b`` as a condition."""
    return _compare(a, b, "==")


def ne(a, b) -> Condition:
    """``a != b`` as a condition."""
    return _compare(a, b, "!=")


def _paren(f: Formula, needed: bool) -> str:
    return f"({f.text})" if needed else f.text


def _binary(a, b, op: str) -> Formula:
    a, b = as_formula(a), as_formula(b)
    if op in ("+", "-"):
        right = b.prec <= _ADD if op == "-" else b.prec < _ADD
        return Formula(f"{_paren(a, a.prec < _ADD)} {op} {_paren(b, right)}", _ADD)
    if op in ("*", "/"):
        right = b.prec <= _MUL if op == "/" else b.prec < _MUL
        return Formula(f"{_paren(a, a.prec < _MUL)}{op}{_paren(b, right)}", _MUL)
    return Formula(f"{_paren(a, a.prec <= _POW)}**{_paren(b, b.prec < _POW)}", _POW)


def as_formula(x) -> Formula:
    """Turn a number, text, or model object into a ``Formula``."""
    if isinstance(x, Condition):
        raise TypeError(
            f"'{x}' is a condition (true or false), not a number; use it to choose between "
            "values with tq.cases((condition, value), ..., otherwise=value)."
        )
    if isinstance(x, Axis):
        return x.coord
    if isinstance(x, Formula):
        return x
    if isinstance(x, VectorParam):
        x._misuse()
    if isinstance(x, bool) or isinstance(x, np.bool_):
        raise TypeError(f"{x!r} is not a number for a formula.")
    if isinstance(x, (int, np.integer)):
        v = int(x)
        return Formula(str(v), _ATOM if v >= 0 else _NEG)
    if isinstance(x, (Real, np.floating)):
        v = float(x)
        if not np.isfinite(v):
            raise ValueError(f"{v} is not a finite number.")
        return Formula(_float_text(v), _ATOM if v >= 0 else _NEG)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            raise FormulaError("Empty formula.")
        # the text must be one complete expression on its own, so splicing it into a larger
        # formula cannot change its meaning (e.g. "b) + (c")
        try:
            body = ast.parse(s.replace("^", "**"), mode="eval").body
        except SyntaxError as e:
            raise FormulaError(f"Cannot read formula {s!r}: {e.msg}.") from None
        if isinstance(body, ast.Tuple):
            raise FormulaError(f"{s!r} is several formulas separated by commas; give one.")
        return Formula(s, _ATOM if _SIMPLE.fullmatch(s) else _RAW)
    raise TypeError(
        f"Cannot use {type(x).__name__} {x!r} in a formula. Use numbers, model objects "
        "(parameters, fields, axes, ...), tq functions, or formula text."
    )


def _float_text(v: float) -> str:
    """Shortest exact text for a number, in scientific notation when large or small
    (``1e7``, ``1.5e-9``) and plain otherwise (``0.1``, ``50.0``)."""
    if v == 0 or 1e-4 <= abs(v) < 1e5:
        return repr(v)
    for digits in range(17):
        t = f"{v:.{digits}e}"
        if float(t) == v:
            break
    mantissa, exponent = t.split("e")
    return f"{mantissa}e{int(exponent)}"


def _text(x) -> str:
    return as_formula(x).text


def _part_label(name: str, where) -> str:
    if not where:
        return name
    return f"{name}.at({', '.join(f'{a}={lab!r}' for a, lab in where)})"


def _size_text(x) -> str:
    """A weight or volume: a formula, or a per-category parameter used with all its axes."""
    if isinstance(x, VectorParam):
        return f"{x.name}[{x._index}]"
    return _text(x)


# ----------------------------------------------------------------------------- model objects
class Name(Formula):
    """A named model object usable in formulas: a parameter, field, compartment, definition or
    integral. ``.name`` is its name and ``.what`` says which kind it is."""

    def __init__(self, name: str, what: str):
        super().__init__(name, _ATOM)
        self.name = name
        self.what = what

    def __repr__(self) -> str:
        return f"<{self.what} {self.name}>"

    def __call__(self, **end) -> Formula:
        """A field's value at one end of a continuous axis, as a paper writes D(t, 0):
        ``D(z="lower")``, ``D(z="upper")``, or the coordinate of that end, ``D(z=0)``. Its
        value is the one the boundary condition at that end gives (e.g. the level that the
        exchange there sees, or the edge value carried out by an outflow)."""
        if self.what != "field":
            raise TypeError(
                f"'{self.name}' is a {self.what}; only a field can be read at the end of an "
                "axis, as F(z='lower')."
            )
        if len(end) != 1:
            raise TypeError(
                f"{self.name}(...): name one axis and one end, e.g. {self.name}(z='lower')."
            )
        ((ax, at),) = end.items()
        if isinstance(at, str):
            if at not in ("lower", "upper"):
                raise ValueError(
                    f"{self.name}({ax}={at!r}): the end is 'lower', 'upper' or its coordinate."
                )
            side = at
        elif isinstance(at, (int, float)) and not isinstance(at, bool):
            side = repr(float(at))
        else:
            raise TypeError(
                f"{self.name}({ax}=...): the end is 'lower', 'upper' or its coordinate, not {at!r}."
            )
        return Formula(f"{self.name}({ax}={side})", _ATOM)

    def at(self, **where) -> Slice:
        """The part of a field with some categorical axes fixed, e.g. ``B.at(s="R")`` (the
        resistant class of B, over all its other axes). Accepted by extinction groups,
        ``reaction``, ``hazard``, ``transition``, ``transfer``, ``integral``,
        ``observable`` and ``dose``."""
        if self.what != "field":
            raise TypeError(f"Only a field can be sliced; '{self.name}' is a {self.what}.")
        if not where:
            raise TypeError(f"{self.name}.at(...): fix at least one axis, e.g. at(s='R').")
        return Slice(self.name, tuple((str(a), str(lab)) for a, lab in where.items()))


class Slice:
    """Part of a field: ``field`` with the categories in ``where`` fixed (``B.at(s="R")``)."""

    def __init__(self, field: str, where: tuple[tuple[str, str], ...]):
        self.field = field
        self.where = where

    def __repr__(self) -> str:
        inner = ", ".join(f"{a}={lab!r}" for a, lab in self.where)
        return f"{self.field}.at({inner})"


class Axis:
    """An axis. A continuous axis can be used in formulas as its coordinate (``age`` or
    ``age.coord``); a categorical axis indexes vector parameters (``k[s]``)."""

    def __init__(self, name: str, grid: ContinuousAxis | CategoricalAxis):
        self.name = name
        self.grid = grid
        self.type = "continuous" if isinstance(grid, ContinuousAxis) else "categorical"

    @property
    def coord(self) -> Formula:
        """The cell-centre coordinate, for use in formulas."""
        if self.type != "continuous":
            raise FormulaError(
                f"'{self.name}' is a categorical axis; it has no coordinate. Index a vector "
                f"parameter with it instead, e.g. k[{self.name}]."
            )
        return Formula(self.name)

    @property
    def n(self) -> int:
        return self.grid.n

    @property
    def labels(self) -> tuple[str, ...]:
        if self.type != "categorical":
            raise AttributeError(f"'{self.name}' is a continuous axis; it has no labels.")
        return tuple(self.grid.labels)

    @property
    def faces(self) -> np.ndarray:
        return self._continuous().faces

    @property
    def centers(self) -> np.ndarray:
        return self._continuous().centers

    @property
    def widths(self) -> np.ndarray:
        return self._continuous().widths

    def _continuous(self) -> ContinuousAxis:
        if self.type != "continuous":
            raise AttributeError(f"'{self.name}' is a categorical axis.")
        return self.grid

    def __repr__(self) -> str:
        return f"<axis {self.grid!r}>"

    # arithmetic on a continuous axis means its coordinate
    def __add__(self, o):
        return self.coord + o

    def __radd__(self, o):
        return o + self.coord

    def __sub__(self, o):
        return self.coord - o

    def __rsub__(self, o):
        return o - self.coord

    def __mul__(self, o):
        return self.coord * o

    def __rmul__(self, o):
        return o * self.coord

    def __truediv__(self, o):
        return self.coord / o

    def __rtruediv__(self, o):
        return o / self.coord

    def __pow__(self, o):
        return self.coord**o

    def __rpow__(self, o):
        return o**self.coord

    def __neg__(self):
        return -self.coord

    __array_ufunc__ = None


class VectorParam:
    """A parameter with one value per category of an axis, or per combination of categories of
    several axes. Use it indexed: ``k[s]``, ``V[organ, sub]``."""

    def __init__(self, name: str, over: str | tuple[str, ...]):
        self.name = name
        self.over = over

    @property
    def _index(self) -> str:
        return ", ".join(self.over) if isinstance(self.over, tuple) else self.over

    def __getitem__(self, axis) -> Formula:
        axes = axis if isinstance(axis, tuple) else (axis,)
        names = tuple(a.name if isinstance(a, Axis) else a for a in axes)
        want = self.over if isinstance(self.over, tuple) else (self.over,)
        if names != want:
            raise FormulaError(
                f"'{self.name}' has one value per category of {_which(self.over)}; "
                f"write {self.name}[{self._index}], not {self.name}[{', '.join(names)}]."
            )
        return Formula(f"{self.name}[{self._index}]")

    def _misuse(self, *_):
        raise FormulaError(
            f"'{self.name}' has one value per category of {_which(self.over)}; "
            f"write {self.name}[{self._index}]."
        )

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _misuse
    __truediv__ = __rtruediv__ = __pow__ = __rpow__ = __neg__ = _misuse
    __array_ufunc__ = None

    def __repr__(self) -> str:
        return f"<vector parameter {self.name}[{self._index}]>"


def _which(over) -> str:
    if isinstance(over, tuple):
        return "axes " + ", ".join(f"'{a}'" for a in over)
    return f"axis '{over}'"


class Transitions:
    """A set of rates between categories of one axis, or of several axes jointly
    (``M[dest, origin]``, diagonal derived)."""

    def __init__(self, name: str, axis: str | tuple[str, ...]):
        self.name = name
        self.axis = axis

    def __repr__(self) -> str:
        return f"<transitions {self.name} on {self.axis}>"


def _table(value, labels: list, axes: tuple[str, ...], role: str) -> list:
    """Values of a per-category parameter, flattened in label order (first axis slowest)."""
    if isinstance(value, Mapping):
        labs = labels[0]
        missing = [lab for lab in labs if lab not in value]
        extra = [k for k in value if k not in labs]
        if missing or extra:
            raise ValueError(
                f"{role} over '{axes[0]}' needs one value for each of {list(labs)}"
                + (f"; missing {missing}" if missing else "")
                + (f"; unknown {extra}" if extra else "")
                + "."
            )
        if len(axes) == 1:
            return [value[lab] for lab in labs]
        out = []
        for lab in labs:
            out += _table(value[lab], labels[1:], axes[1:], f"{role}, '{lab}'")
        return out
    vals = list(np.ravel(value))
    n = int(np.prod([len(lab) for lab in labels]))
    if len(vals) != n:
        per = " x ".join(f"{len(lab)} ({a})" for a, lab in zip(axes, labels, strict=True))
        raise ValueError(f"{role} needs {n} values ({per}), got {len(vals)}.")
    return vals


# ----------------------------------------------------------------------------- the model
_SIDES = ("lower", "upper")


def _is_zero_text(x) -> bool:
    """A formula written as plain zero ("0", "0.0"): nothing crosses."""
    if isinstance(x, Real):
        return False
    try:
        return float(_text(x)) == 0.0
    except (TypeError, ValueError):
        return False


class Model:
    """A model under construction. Each method adds one piece; ``to_ir`` returns the
    complete description and ``validate`` checks it."""

    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError("A model needs a name, e.g. tq.Model('phage_latency').")
        self.name = name
        self._kind: dict[str, str] = {}  # name -> what it is
        self._handles: dict[str, object] = {}
        self._axes: list[AxisSpec] = []
        self._params: list[ParamSpec] = []
        self._fields: list[FieldSpec] = []
        self._compartments: list[CompartmentSpec] = []
        self._definitions: list[DefinitionSpec] = []
        self._integrals: list[IntegralSpec] = []
        self._transitions: list[TransitionSpec] = []
        self._terms: list[TermSpec] = []
        self._boundaries: list[BoundarySpec] = []
        self._inits: list[InitSpec] = []
        self._doses: list[DoseSpec] = []
        self._lags: list[LagSpec] = []
        self._bioavailabilities: list[BioavailabilitySpec] = []
        self._observables: list[ObservableSpec] = []
        self._extinctions: list[ExtinctionSpec] = []
        self._transfers: list[TransferSpec] = []

    def __repr__(self) -> str:
        return f"<tq.Model {self.name!r}: {len(self._fields)} fields, " + (
            f"{len(self._compartments)} compartments, {len(self._terms)} terms>"
        )

    def __getitem__(self, name: str):
        """The object for a name, e.g. ``m["I"]`` (useful for models loaded from the IR)."""
        try:
            return self._handles[name]
        except KeyError:
            raise KeyError(f"Model '{self.name}' has nothing named '{name}'.") from None

    # ---------------------------------------------------------------- names
    def _claim(self, name: str, what: str) -> None:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(
                f"{what} name {name!r}: use letters, digits and _, not starting with a digit."
            )
        if name in RESERVED_NAMES:
            raise ValueError(f"{what} name '{name}' is reserved (it means time).")
        if keyword.iskeyword(name):
            raise ValueError(
                f"{what} name '{name}' is a reserved word of the formula language; choose "
                f"another (e.g. '{name}_')."
            )
        if name in self._kind:
            kind = self._kind[name]
            article = "an" if kind[0] in "aeiou" else "a"
            raise ValueError(
                f"The name '{name}' is already used by {article} {kind}; every name must be unique."
            )
        self._kind[name] = what

    def _name_of(self, x, *allowed: str, role: str) -> str:
        """Name of a model object passed by handle or by name, checking what kind it is."""
        if isinstance(x, (Name, Axis, VectorParam, Transitions)):
            name = x.name
        elif isinstance(x, str):
            name = x
        else:
            raise TypeError(f"{role}: expected a model object or its name, got {x!r}.")
        what = self._kind.get(name)
        if what is None:
            raise ValueError(f"{role}: model '{self.name}' has nothing named '{name}'.")
        if allowed and what not in allowed:
            need = " or ".join(allowed)
            raise ValueError(f"{role}: '{name}' is a {what}, but a {need} is needed.")
        return name

    def _axis_name(self, axis, role: str, type_: str | None = None) -> str:
        name = self._name_of(axis, "axis", role=role)
        if type_ is not None:
            grid = self._handles[name].grid
            actual = "continuous" if isinstance(grid, ContinuousAxis) else "categorical"
            if actual != type_:
                raise ValueError(f"{role}: '{name}' is a {actual} axis; a {type_} axis is needed.")
        return name

    # ---------------------------------------------------------------- structure
    def axis(
        self,
        name: str,
        lower: float | None = None,
        upper: float | None = None,
        n: int | None = None,
        *,
        kind: str | None = None,
        units: str = "",
        faces: Sequence[float] | None = None,
        spacing: str = "uniform",
        ratio: float | None = None,
        labels: Sequence[str] | None = None,
    ) -> Axis:
        """Add an axis.

        Continuous: ``m.axis("age", lower=0, upper=2, n=40, kind="one_way", units="h")``, or
        explicit cell boundaries with ``faces=[...]``, or ``spacing="geometric", ratio=...``.
        ``kind`` is ``"one_way"`` (moves only upward, e.g. age), ``"bidirectional"`` (default)
        or ``"spatial"``.

        Categorical: ``m.axis("s", labels=["S", "R"])``.
        """
        if labels is not None:
            if any(v is not None for v in (lower, upper, n, faces, ratio)):
                raise ValueError(
                    f"Axis '{name}': give labels (categorical) or a range (continuous), not both."
                )
            if kind not in (None, "categorical"):
                raise ValueError(f"Axis '{name}': a categorical axis has no kind {kind!r}.")
            if spacing != "uniform":
                raise ValueError(f"Axis '{name}': a categorical axis has no spacing.")
            grid = CategoricalAxis(name, tuple(labels))
            self._claim(name, "axis")
            self._axes.append(AxisSpec(name, "categorical", labels=tuple(grid.labels), units=units))
        else:
            kind = kind or "bidirectional"
            if faces is not None:
                if any(v is not None for v in (lower, upper, n, ratio)):
                    raise ValueError(f"Axis '{name}': give faces, or lower/upper/n, not both.")
                grid = ContinuousAxis(name, np.asarray(faces, float), kind=kind, units=units)
            else:
                if lower is None or upper is None or n is None:
                    raise ValueError(
                        f"Axis '{name}': give lower, upper and n (number of cells), faces=[...] "
                        "for a continuous axis, or labels=[...] for a categorical axis."
                    )
                if spacing == "uniform":
                    if ratio is not None:
                        raise ValueError(f"Axis '{name}': ratio is for spacing='geometric'.")
                    grid = ContinuousAxis.uniform(name, lower, upper, n, kind=kind, units=units)
                elif spacing == "geometric":
                    if ratio is None:
                        raise ValueError(f"Axis '{name}': spacing='geometric' needs ratio=.")
                    grid = ContinuousAxis.geometric(
                        name, lower, upper, n, ratio, kind=kind, units=units
                    )
                else:
                    raise ValueError(
                        f"Axis '{name}': spacing must be 'uniform' or 'geometric', not {spacing!r}."
                    )
            self._claim(name, "axis")
            self._axes.append(
                AxisSpec(
                    name,
                    "continuous",
                    faces=tuple(float(f) for f in grid.faces),
                    kind=kind,
                    units=units,
                )
            )
        h = Axis(name, grid)
        self._handles[name] = h
        return h

    def param(
        self,
        name: str,
        value: float | Sequence[float] | Mapping[str, float] | None = None,
        *,
        over=None,
        units: str = "",
    ) -> Name | VectorParam:
        """Add one parameter. Without a value (``m.param("k")``, ``m.param("k", over=s)``) it is
        declared only: its value must be given when the model is solved or simulated
        (``params={"k": 2}``); a missing value is an error, never a guess.

        Scalar: ``m.param("k", 1e-9)``. One value per category: ``m.param("k", {"S": 1e-9,
        "R": 1e-12}, over=s)`` (or a list in label order); use it as ``k[s]``. Over several
        axes, a nested dict (or a table with one row per category of the first axis):
        ``m.param("V", {"liver": {"vascular": 0.1, "cellular": 1.5}, ...}, over=[organ, sub])``,
        used as ``V[organ, sub]``.
        """
        if over is None:
            if isinstance(value, (Mapping, Sequence, np.ndarray)) and not isinstance(value, str):
                raise ValueError(
                    f"Parameter '{name}' has several values; say which categorical axis they "
                    "belong to with over=."
                )
            v = None if value is None else _number(value, f"parameter '{name}'")
            self._claim(name, "parameter")
            self._params.append(ParamSpec(name, v, units=units))
            h: Name | VectorParam = Name(name, "parameter")
        else:
            role = f"parameter '{name}'"
            if isinstance(over, (list, tuple)):
                axes = tuple(self._axis_name(a, role, "categorical") for a in over)
                if not axes or len(set(axes)) != len(axes):
                    raise ValueError(f"{role}: over= needs different categorical axes.")
            else:
                axes = (self._axis_name(over, role, "categorical"),)
            if value is None:
                vec = None
            else:
                vals = _table(value, [self._handles[a].labels for a in axes], axes, role)
                vec = tuple(_number(x, role) for x in vals)
            key = axes if len(axes) > 1 else axes[0]
            self._claim(name, "parameter")
            self._params.append(ParamSpec(name, vec, over=key, units=units))
            h = VectorParam(name, key)
        self._handles[name] = h
        return h

    def params(self, *names: str, **values: float) -> Name | tuple[Name, ...]:
        """Add scalar parameters: ``k, beta = m.params(k=1e-9, beta=50.0)``. Names alone are
        declared without a value, given when the model is solved or simulated:
        ``ka, CL = m.params("ka", "CL")``. Returns them in the order given (one object if only
        one is given)."""
        if not names and not values:
            raise ValueError("m.params(...) needs at least one name or name=value.")
        out = tuple(self.param(n) for n in names)
        out += tuple(self.param(n, v) for n, v in values.items())
        return out[0] if len(out) == 1 else out

    def field(self, name: str, axes: Sequence, *, init=None, units: str = "", weights=None) -> Name:
        """Add a population density over one or more axes, e.g. ``m.field("I", axes=[age])``.
        ``init`` sets a starting density (a number or formula); see ``init`` for more.

        ``weights`` gives each category a size, as cell widths do on a continuous axis, e.g.
        organ volumes ``V`` (a parameter over ``[organ, sub]``). The field then holds a
        concentration; its amount is concentration x volume, and ``transition`` and
        ``transfer`` take clearances or flows (volume per time), so what leaves one
        category is exactly what arrives in another.
        """
        if isinstance(axes, (str, Axis)):
            axes = [axes]
        names = tuple(self._axis_name(a, f"field '{name}'") for a in axes)
        if not names:
            raise ValueError(
                f"Field '{name}' needs at least one axis; otherwise use a compartment."
            )
        if len(set(names)) != len(names):
            raise ValueError(f"Field '{name}': an axis is listed twice ({list(names)}).")
        w = None if weights is None else _size_text(weights)
        self._claim(name, "field")
        self._fields.append(FieldSpec(name, names, units=units, weights=w))
        h = Name(name, "field")
        self._handles[name] = h
        if init is not None:
            self.init(h, init)
        return h

    def compartment(self, name: str, *, init=None, units: str = "", volume=None) -> Name:
        """Add a well-mixed amount (free phage, drug, nutrient), e.g.
        ``m.compartment("P", init=1e7)``. With ``volume`` (a parameter or formula of
        parameters) it holds a concentration: doses and transfers are divided by the volume,
        and its amount is value x volume."""
        vol = None if volume is None else _text(volume)
        self._claim(name, "compartment")
        self._compartments.append(CompartmentSpec(name, units=units, volume=vol))
        h = Name(name, "compartment")
        self._handles[name] = h
        if init is not None:
            self.init(h, init)
        return h

    def define(self, name: str, expr) -> Name:
        """Name an intermediate formula, e.g. ``h = m.define("h", h_max*tq.sigmoid(...))``.
        Using the name keeps the stored equations short and readable."""
        text = _text(expr)
        self._claim(name, "definition")
        self._definitions.append(DefinitionSpec(name, text))
        h = Name(name, "definition")
        self._handles[name] = h
        return h

    def integral(self, expr, *, over, name: str | None = None) -> Name:
        """Integral of ``expr`` over one or more axes (a sum on categorical axes). The result
        varies over the axes not integrated. ``expr`` may be a slice, ``C.at(sub="vascular")``.
        A field with ``weights`` is weighted by them (its integral is an amount). Without
        ``name`` one is made up (``int_age``); the same integral asked for twice is stored
        once."""
        if isinstance(over, (str, Axis)):
            over = [over]
        axes = tuple(self._axis_name(a, "integral") for a in over)
        if not axes:
            raise ValueError("integral: give the axes to integrate over, e.g. over=age.")
        where: tuple = ()
        if isinstance(expr, Slice):
            text, where = self._field(expr.field, "integral"), expr.where
        else:
            text = _text(expr)
        if name is None:
            for s in self._integrals:
                if s.integrand == text and s.over == axes and s.where == where:
                    return self._handles[s.name]
            base = "int_" + "_".join(axes)
            name, i = base, 2
            while name in self._kind:
                name, i = f"{base}_{i}", i + 1
        self._claim(name, "integral")
        self._integrals.append(IntegralSpec(name, text, axes, where=where))
        h = Name(name, "integral")
        self._handles[name] = h
        return h

    def _categorical_axes(self, axis, role: str):
        """One categorical axis (its name) or, for joint transitions, a list of two or more
        (a tuple of names)."""
        if isinstance(axis, (list, tuple)):
            names = tuple(self._axis_name(a, role, "categorical") for a in axis)
            if len(names) < 2 or len(set(names)) != len(names):
                raise ValueError(
                    f"{role}: give one categorical axis, or two or more different ones for a "
                    f"joint transition, not {list(names)}."
                )
            return names
        return self._axis_name(axis, role, "categorical")

    def transitions(self, name: str, *, axis, rates) -> Transitions:
        """Rates between the categories of an axis, e.g.
        ``M = m.transitions("M", axis=s, rates={"S->R": mu, "R->S": mu})``.

        Only the moves are given; the "stay" (diagonal) rates are derived so that nothing is
        created or lost. Use with ``transition``.

        A rate may vary over the field's other axes (e.g. with dormancy depth, or ``k[organ]``)
        and depend on fields, compartments and totals; it may not vary along ``axis`` itself.

        **Joint transitions** change several categorical axes at once (a shared receptor:
        one mutation gives resistance to two phages). Give the axes as a list and each
        category as its labels in that order, separated by commas:
        ``m.transitions("X", axis=[r1, r2], rates={"S,S->R,R": mu_x})``.
        """
        role = f"transitions '{name}'"
        ax = self._categorical_axes(axis, role)
        joint = isinstance(ax, tuple)
        items: Iterable
        if isinstance(rates, Mapping):
            items = rates.items()
        else:
            try:
                items = [((r[0], r[1]), r[2]) for r in rates]
            except (TypeError, IndexError):
                raise ValueError(
                    f"{role}: give rates as a dict, e.g. "
                    "{'S->R': mu, 'R->S': nu}, or as (from, to, rate) triples."
                ) from None

        def category(c):
            if not joint:
                return str(c)
            parts = [p.strip() for p in c.split(",")] if isinstance(c, str) else list(c)
            if len(parts) != len(ax):
                raise ValueError(
                    f"{role}: {c!r} must name one category of each of {list(ax)}, in that "
                    f"order, e.g. '{','.join('S' for _ in ax)}'."
                )
            return tuple(str(p) for p in parts)

        triples = []
        for key, rate in items:
            if isinstance(key, str):
                parts = [p.strip() for p in key.split("->")]
                if len(parts) != 2 or not all(parts):
                    example = "'S,S->R,R'" if joint else "'FROM->TO'"
                    raise ValueError(f"{role}: write each move as {example}, not {key!r}.")
                frm, to = parts
            else:
                frm, to = key
            triples.append((category(frm), category(to), _text(rate)))
        if not triples:
            raise ValueError(f"{role}: give at least one move.")
        self._claim(name, "transitions")
        self._transitions.append(TransitionSpec(name, ax, tuple(triples)))
        h = Transitions(name, ax)
        self._handles[name] = h
        return h

    # ---------------------------------------------------------------- equations
    def _field(self, x, role: str) -> str:
        return self._name_of(x, "field", role=role)

    def _part(self, x, role: str) -> tuple[str, tuple]:
        """A field, or a slice of one: (field name, fixed categories)."""
        if isinstance(x, Slice):
            return self._field(x.field, role), x.where
        return self._field(x, role), ()

    def _end_of(self, x, role: str) -> tuple[str, tuple]:
        """A compartment, a field or a slice of a field: (name, fixed categories)."""
        if isinstance(x, Slice):
            return self._field(x.field, role), x.where
        return self._name_of(x, "compartment", "field", role=role), ()

    def ode(self, compartment, rhs) -> None:
        """The equation of a compartment: ``m.ode(P, beta*lysis - delta*P)`` means
        ``dP/dt = beta*lysis - delta*P``."""
        c = self._name_of(compartment, "compartment", role="ode")
        if any(t.kind == "rate" and t.target == c for t in self._terms):
            raise ValueError(
                f"Compartment '{c}' already has an equation; write the whole right-hand side "
                "in one m.ode call."
            )
        self._terms.append(TermSpec("rate", c, _text(rhs)))

    def reaction(self, field, expr) -> None:
        """Add ``expr`` to the field's rate of change, cell by cell (births, sources, losses).
        ``field`` may be a slice, ``C.at(organ="liver")``: only there."""
        f, where = self._part(field, "reaction")
        self._terms.append(TermSpec("reaction", f, _text(expr), where=where))

    def hazard(self, field, rate) -> None:
        """Remove the field at per-capita ``rate``: adds ``-rate*field`` (on a slice, only
        there)."""
        f, where = self._part(field, "hazard")
        self._terms.append(TermSpec("hazard", f, _text(rate), where=where))

    def advect(self, field, *, axis, velocity, scheme: str = "erlang") -> None:
        """Move the field along a continuous axis at ``velocity`` (ageing, maturation, drift).

        ``scheme="erlang"`` (default) behaves exactly like a transit-compartment chain;
        ``"sharp"`` keeps fronts and peaks sharper on the same grid.
        """
        f = self._field(field, "advect")
        ax = self._axis_name(axis, "advect", "continuous")
        if scheme not in ("erlang", "sharp"):
            raise ValueError(f"advect: scheme must be 'erlang' or 'sharp', not {scheme!r}.")
        self._terms.append(TermSpec("advection", f, _text(velocity), axis=ax, scheme=scheme))

    def diffuse(self, field, *, axis, D) -> None:
        """Spread the field along a continuous axis with diffusivity ``D``."""
        f = self._field(field, "diffuse")
        ax = self._axis_name(axis, "diffuse", "continuous")
        self._terms.append(TermSpec("diffusion", f, _text(D), axis=ax))

    def fokker_planck(self, field, *, axis, drift, D, scheme: str = "sharp") -> None:
        """Single cells moving along ``axis`` as ``dz = drift dt + sqrt(2 D) dW`` (Itô): their
        density follows the Fokker–Planck equation ``d_t(F) = -d_z((drift - dD/dz)*F) +
        d_z(D*d_z(F))``. Adds that transport (velocity ``drift - dD/dz``, the slope of ``D``
        worked out here) and that diffusion. ``D`` may vary along the axis through its
        coordinate, but not through a field (write the two terms yourself then, with
        ``grad``).

        The transport scheme is ``sharp`` by default: here ``D`` is the spread, and
        ``erlang`` would add the grid's own spread (about ``|drift| x width / 2``) to it."""
        from .explain import diffusion_slope

        f = self._field(field, "fokker_planck")
        ax = self._axis_name(axis, "fokker_planck", "continuous")
        slope = diffusion_slope(self.to_ir(), _text(D), ax)
        simple = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*|[0-9.]+(e[+-]?[0-9]+)?", slope)
        minus = slope if simple else f"({slope})"
        d = _text(drift)
        d = d if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*|[0-9.]+(e[+-]?[0-9]+)?", d) else f"({d})"
        velocity = _text(drift) if slope == "0" else f"{d} - {minus}"
        self.advect(f, axis=ax, velocity=velocity, scheme=scheme)
        self.diffuse(f, axis=ax, D=D)

    def transition(self, field, *, axis, matrix, of) -> None:
        """Move ``of`` between the categories of ``axis`` with the rates of ``matrix`` (from
        ``transitions``). ``axis`` is the matrix's axis, or its list of axes for a joint
        transition. ``field`` may be a slice: blood flow between organs only in
        ``C.at(sub="vascular")``. ``of`` has no default, so it always says which case is meant:

        - ``of=`` the field itself (``m.transition(B, axis=s, matrix=Q, of=B)``): cells switch
          category at the rates of ``matrix`` per cell (phenotypic switching, exchange between
          tissue spaces). On a field with ``weights`` a rate is then a clearance or flow
          (volume per time): it moves rate x concentration of the origin, an amount per time.
        - ``of=`` a rate of production (``of=div``, the division flux ``g*B``): a fraction of it
          ends up in another category, at the parent's rate (daughters carrying a mutation).
          The production itself is added with ``reaction`` as usual; this only moves the
          daughters.

        Only the moves are given; what leaves the origin is derived, so nothing is created or
        lost."""
        f, where = self._part(field, "transition")
        ax = self._categorical_axes(axis, "transition")
        tr = self._name_of(matrix, "transitions", role="transition")
        if isinstance(of, Slice):
            if of.field != f or of.where != where:
                raise ValueError(
                    f"transition of '{f}': of= is a slice of '{of.field}'; give the field "
                    f"itself ({_part_label(f, where)}) or a rate of production (a formula)."
                )
            itself = True
        else:
            itself = isinstance(of, Name) and of.what == "field" and of.name == f
            itself = itself or (isinstance(of, str) and of.strip() == f)
        if itself:
            self._terms.append(TermSpec("switch", f, "0", axis=ax, transitions=tr, where=where))
        else:
            self._terms.append(
                TermSpec("mutate", f, _text(of), axis=ax, transitions=tr, where=where)
            )

    # ---------------------------------------------------------------- boundaries
    def _end(self, field, axis, at, role: str) -> tuple[str, str, str]:
        f = self._field(field, role)
        ax = self._axis_name(axis, role, "continuous")
        if at not in _SIDES:
            raise ValueError(f"{role}: at must be 'lower' or 'upper', not {at!r}.")
        return f, ax, at

    def inflow(self, field, *, axis, at: str, flux) -> None:
        """Material entering at one end, e.g. newly infected cells entering age 0:
        ``m.inflow(I, axis=age, at="lower", flux=k*B*P)``."""
        f, ax, side = self._end(field, axis, at, "inflow")
        self._boundaries.append(BoundarySpec(f, ax, side, "inflow", value=_text(flux)))

    def outflow(self, field, *, axis, at: str, to=None) -> None:
        """Material carried out of one end by advection. It is absorbed (``to=None``) or added
        to compartment ``to``; either way it is recorded in the ledger."""
        f, ax, side = self._end(field, axis, at, "outflow")
        dest = None if to is None else self._name_of(to, "compartment", role="outflow to")
        self._boundaries.append(BoundarySpec(f, ax, side, "outflow", to=dest))

    def at(self, field, *, flux=None, value=None, outflow: bool = False, **end) -> None:
        """A boundary condition, written as a paper writes it under the equation ("at z = 0:
        ..."). Name the end as ``z="lower"``, ``z="upper"`` or by its coordinate (``z=0``),
        and give exactly one of:

        - ``flux=``: the flux of the field along the axis at that end, positive toward
          larger coordinates, equals this formula. Inside it the field's own name means its
          value at that end. ``m.at(I, l=0, flux=k*P*B)`` (cells entering infection age 0,
          ``v*I(t, 0) = k*P*B``); ``m.at(D, z=0, flux=alpha*(1 - frac)*B - beta*frac*D)``
          (entry minus exit, ``-Dz*dD/dz = ...``); ``flux=0``: nothing crosses.
        - ``value=``: the field is held at this value there (``C(t, 1) = 1``).
        - ``outflow=True``: what the flow carries to this end leaves through it.

        What crosses is recorded in the ledger. To use it in another equation, read the
        field there: ``D(z=0)``, ``I(l="upper")``."""
        f = self._field(field, "at")
        if len(end) != 1:
            raise TypeError(
                "at: name one axis and one end, e.g. m.at(D, z='lower', flux=...); "
                f"got {sorted(end) or 'none'}."
            )
        ((axis, where),) = end.items()
        ax = self._axis_name(axis, "at", "continuous")
        if ax not in next(x.axes for x in self._fields if x.name == f):
            raise ValueError(f"at: '{ax}' is not an axis of '{f}'.")
        if isinstance(where, str):
            if where not in _SIDES:
                raise ValueError(
                    f"at: the end is 'lower', 'upper' or its coordinate, not {where!r}."
                )
            side = where
        else:
            faces = next(a.faces for a in self._axes if a.name == ax)
            try:
                c = float(where)
            except (TypeError, ValueError):
                raise TypeError(
                    f"at: the end is 'lower', 'upper' or its coordinate, not {where!r}."
                ) from None
            tol = 1e-9 * abs(faces[-1] - faces[0])  # rounding, relative to the axis length
            if abs(c - faces[0]) <= tol:
                side = "lower"
            elif abs(c - faces[-1]) <= tol:
                side = "upper"
            else:
                raise ValueError(
                    f"at: {ax}={c:g} is not an end of '{ax}' (its ends are {faces[0]:g} and "
                    f"{faces[-1]:g})."
                )
        given = [n for n, x in (("flux", flux), ("value", value)) if x is not None]
        if outflow:
            given.append("outflow")
        if len(given) != 1:
            raise TypeError(
                "at: give exactly one of flux= (the flux along the axis there), value= (the "
                f"field held at a value) or outflow=True; got {given or 'none'}."
            )
        if outflow:
            spec = BoundarySpec(f, ax, side, "outflow")
        elif value is not None:
            spec = BoundarySpec(f, ax, side, "dirichlet", value=_text(value))
        elif (isinstance(flux, Real) and float(flux) == 0.0) or _is_zero_text(flux):
            spec = BoundarySpec(f, ax, side, "zero_flux")
        else:
            spec = BoundarySpec(f, ax, side, "flux", value=_text(flux))
        self._boundaries.append(spec)

    def boundary(self, field, *, axis, at: str, kind: str, value=None, h=None, to=None) -> None:
        """Any boundary condition: ``kind`` is ``"zero_flux"`` (nothing crosses),
        ``"dirichlet"`` (held at ``value``), or ``"robin"`` (exchange ``h*(value - field)``,
        e.g. a biofilm surface with the bulk liquid). Inflow and outflow have their own
        methods.

        With ``to=`` a compartment (dirichlet or robin only), whatever crosses this end is
        taken from that compartment (or given to it, if it leaves the field), so the two
        exchange mass exactly: ``m.boundary(C, axis=z, at="upper", kind="robin",
        value=Cb/V, h=h, to=Cb)``. The ledger records it on both sides."""
        f, ax, side = self._end(field, axis, at, "boundary")
        dest = None
        if to is not None:
            if kind not in ("dirichlet", "robin"):
                raise ValueError(
                    "boundary: to= (exchange with a compartment) is for dirichlet and robin "
                    "conditions; an outflow is routed with m.outflow(..., to=)."
                )
            dest = self._name_of(to, "compartment", role="boundary to")
        if kind == "zero_flux":
            if value is not None or h is not None:
                raise ValueError("boundary: a zero_flux condition takes no value or h.")
            spec = BoundarySpec(f, ax, side, "zero_flux")
        elif kind == "dirichlet":
            if value is None or h is not None:
                raise ValueError("boundary: a dirichlet condition needs value= and no h.")
            spec = BoundarySpec(f, ax, side, "dirichlet", value=_text(value), to=dest)
        elif kind == "robin":
            if value is None or h is None:
                raise ValueError("boundary: a robin condition needs value= (outside level) and h=.")
            spec = BoundarySpec(f, ax, side, "robin", value=_text(value), h=_text(h), to=dest)
        elif kind in ("inflow", "outflow"):
            raise ValueError(f"boundary: use m.{kind}(...) for an {kind} condition.")
        else:
            raise ValueError(
                f"boundary: kind must be 'zero_flux', 'dirichlet' or 'robin', not {kind!r}."
            )
        self._boundaries.append(spec)

    # ---------------------------------------------------------------- initial state, doses
    def init(self, target, value=None, *, total=None, shape=None, where=None, split=None) -> None:
        """Starting state of a field or compartment.

        - ``m.init(P, 1e7)``: a value (for a field, a density; may be a formula of the axes).
        - ``m.init(B, total=1e7, shape=tq.normal(z, mean=0.1, sd=0.08))``: an amount and its
          shape; the shape is cut to the grid and scaled so the cells hold exactly ``total``.
        - ``where={s: "S"}``: only these categories.
        - ``split={s: {"S": 0.99, "R": 0.01}}``: share ``total`` (or ``value``) between
          categories; the fractions must add up to 1.
        """
        tgt = self._name_of(target, "field", "compartment", role="init")
        if value is not None and (total is not None or shape is not None):
            raise ValueError(f"init of '{tgt}': give a value, or total and shape, not both.")
        if value is None and (total is None or shape is None):
            raise ValueError(
                f"init of '{tgt}': give a value, or both total= and shape= "
                "(e.g. total=1e7, shape=tq.normal(z, mean=0.1, sd=0.08))."
            )
        fixed = tuple(
            (self._axis_name(a, f"init of '{tgt}'", "categorical"), str(lab))
            for a, lab in (where or {}).items()
        )
        parts: list[tuple[tuple[tuple[str, str], ...], float | None]] = [(fixed, None)]
        if split:
            choices = []
            for a, fractions in split.items():
                ax = self._axis_name(a, f"init of '{tgt}' (split)", "categorical")
                if any(ax == f for f, _ in fixed):
                    raise ValueError(f"init of '{tgt}': '{ax}' is in both where and split.")
                fr = {str(k): _number(v, f"split fraction on '{ax}'") for k, v in fractions.items()}
                if any(v < 0 for v in fr.values()) or abs(sum(fr.values()) - 1.0) > 1e-12:
                    raise ValueError(
                        f"init of '{tgt}': the fractions on '{ax}' must be >= 0 and add up to 1 "
                        f"(they add up to {sum(fr.values())})."
                    )
                choices.append([((ax, lab), v) for lab, v in fr.items()])
            parts = []
            for combo in product(*choices):
                frac = float(np.prod([v for _, v in combo]))
                parts.append((fixed + tuple(c for c, _ in combo), frac))
        for where_, frac in parts:
            scale = (lambda x: x) if frac is None else (lambda x, f=frac: as_formula(x) * f)
            if value is not None:
                spec = InitSpec(tgt, where=where_, value=_text(scale(value)))
            else:
                spec = InitSpec(tgt, where=where_, total=_text(scale(total)), shape=_text(shape))
            self._inits.append(spec)

    def dose(
        self, target, *, times: Sequence[float], amounts=None, shape=None, density=None
    ) -> None:
        """Bolus doses at ``times``, with one amount for every time or one per time.

        - Into a compartment: ``m.dose(P, times=[0, 24], amounts=1e8)``; with a volume, the
          value rises by amount / volume.
        - Into one element of a field (a slice fixing every axis):
          ``m.dose(C.at(organ="gut", sub="vascular"), times=[0], amounts=100)``.
        - Into a field over continuous axes, as an initial value: an amount and its shape,
          ``m.dose(G, times=[0, 12], amounts=100, shape=tq.point(g, at=0))`` (each dose is
          spread by the shape and the cells receive exactly the amount), or a density added
          to the field, ``m.dose(B, times=[0], density=1e6*tq.exp(-z))``. A slice leaving some
          axes free doses only there.
        """
        tgt, where = self._end_of(target, "dose")
        times_t = tuple(_number(t, "dose time") for t in np.ravel(times))
        axes = next((f.axes for f in self._fields if f.name == tgt), ())
        free = [a for a in axes if a not in dict(where)]
        categorical = {a.name for a in self._axes if a.type == "categorical"}
        if free and not where and any(a in categorical for a in free):
            raise ValueError(
                f"dose: '{tgt}' is a field over categories; dose into one element of it, e.g. "
                f"{tgt}.at(...) with every categorical axis fixed."
            )
        if any(a in categorical for a in free):
            free = []  # the validator says which category axes are left free
        if not free:
            if amounts is None or shape is not None or density is not None:
                raise ValueError(
                    f"dose into '{tgt}': {'a compartment' if not axes else 'one element'} takes "
                    "amounts= only (shape= and density= are for a field with free axes)."
                )
        elif (density is None) == (amounts is None) or (density is not None and shape):
            raise ValueError(
                f"dose into the field '{tgt}' (free axes {free}): give amounts= and shape= "
                f"(e.g. amounts=100, shape=tq.point({free[0]}, at=0)), or density= (a formula "
                "of the axes added to the field), as for an initial value."
            )
        elif amounts is not None and shape is None:
            raise ValueError(
                f"dose into the field '{tgt}': an amount needs a shape saying where it goes "
                f"along {free}, e.g. shape=tq.point({free[0]}, at=0); or give density= instead."
            )
        given = amounts if density is None else density
        if isinstance(given, (Sequence, np.ndarray)) and not isinstance(given, str):
            amts = tuple(_text(a) for a in given)
        else:
            amts = (_text(given),) * len(times_t)
        self._doses.append(
            DoseSpec(tgt, times_t, amts, where=where, shape=None if shape is None else _text(shape))
        )

    def lag(self, target, value) -> None:
        """Every dose into ``target`` (a compartment, or every element of a field) arrives
        ``value`` later, as NONMEM's ALAG: ``m.lag(depot, "tlag")``. ``value`` is a formula of
        parameters, >= 0 (checked when the model is run). It applies to the model's own doses
        and to a regimen's alike; an infusion is shifted whole (start and stop)."""
        tgt = self._name_of(target, "field", "compartment", role="lag")
        if any(x.target == tgt for x in self._lags):
            raise ValueError(f"lag: '{tgt}' already has a lag; give one per place.")
        self._lags.append(LagSpec(tgt, _text(value)))

    def bioavailability(self, target, value) -> None:
        """Only the fraction ``value`` of every dose into ``target`` arrives, as NONMEM's F:
        ``m.bioavailability(depot, "F")``. ``value`` is a formula of parameters, >= 0 (checked
        when the model is run; above 1 is allowed). Bolus amounts and infusion rates are scaled
        by it; the ledger books the rest as "not absorbed"."""
        tgt = self._name_of(target, "field", "compartment", role="bioavailability")
        if any(x.target == tgt for x in self._bioavailabilities):
            raise ValueError(
                f"bioavailability: '{tgt}' already has a bioavailability; give one per place."
            )
        self._bioavailabilities.append(BioavailabilitySpec(tgt, _text(value)))

    # ---------------------------------------------------------------- outputs, extinction
    def observable(self, name: str, expr) -> None:
        """A named output, e.g. ``m.observable("bacteria", m.integral(B, over=[z, s]))``, or a
        slice of a field, ``m.observable("C_liver", C.at(organ="liver", sub="vascular"))``."""
        if isinstance(expr, Slice):
            text, where = self._field(expr.field, "observable"), expr.where
        else:
            text, where = _text(expr), ()
        self._claim(name, "observable")
        self._observables.append(ObservableSpec(name, text, where=where))

    def transfer(self, source, *, to, rate) -> None:
        """Move material from ``source`` to ``to`` at ``rate`` x the source's value (an amount
        per time). Each end is a compartment or a slice of a field; ``to=None`` removes it
        (elimination, recorded in the ledger).

        - compartment -> slice: every element of the slice receives ``rate`` x the
          compartment (``rate`` may vary over the slice, e.g. ``Q[organ]``); the compartment
          loses the sum.
        - slice -> compartment: the compartment receives the sum over the slice.
        - slice -> slice: element by element (the two slices leave the same axes free).

        With volumes or weights, the value is a concentration and ``rate`` a flow or
        clearance (volume per time): ``m.transfer(C.at(sub="vascular"), to=Cven,
        rate=Q[organ])`` is venous return.
        """
        src, swhere = self._end_of(source, "transfer")
        if to is None:
            dst, dwhere = None, ()
        else:
            dst, dwhere = self._end_of(to, "transfer to")
        self._transfers.append(TransferSpec(src, _text(rate), swhere, dst, dwhere))

    def extinction(
        self, group, *, threshold: float, name: str | None = None, noise_floor: bool = False
    ) -> None:
        """Declare a lineage group (fields, slices of fields such as ``B.at(s="R")``, and
        compartments) that is set to zero when its total falls below ``threshold`` after
        having been above it. One group per strain:

            for strain in ["S", "R"]:
                m.extinction(B.at(s=strain), threshold=1.0)

        The total is the sum of the members' amounts (field densities times cell widths, plus
        compartment values), so the members must be counted in the same units. A group is
        *armed* once its total reaches the threshold; an armed group crossing it downward is
        set to zero at the exact crossing (``sol.extinctions``, and an ``extinction`` line in
        the ledger) and disarmed. A group that has never reached the threshold (e.g. mutants
        seeded from zero) is never touched.

        ``noise_floor=True``: while the group's total is below the threshold, every equation
        sees the whole group as zero. Its sources still add to it, but round-off "ghost"
        amounts cannot grow by feeding themselves. Nor does it see its own losses, so a feed
        from outside the group fills it up; a feed too weak to hold it above the threshold
        gives repeated fill-and-extinction cycles. Validation warns about such feeds: put
        the source in the group (e.g. the blood compartment of a phage lineage).
        """
        if isinstance(group, (str, Name, Slice)):
            group = [group]
        members, wheres = [], []
        for g in group:
            if isinstance(g, Slice):
                members.append(self._name_of(g.field, "field", role="extinction"))
                for ax, lab in g.where:
                    labels = self._handles[self._axis_name(ax, "slice", "categorical")].labels
                    if lab not in labels:
                        raise ValueError(
                            f"{g!r}: '{lab}' is not a category of '{ax}' ({list(labels)})."
                        )
                wheres.append(g.where)
            else:
                members.append(self._name_of(g, "field", "compartment", role="extinction"))
                wheres.append(())
        if name is None:
            parts = [
                m + "".join(f"_{a}_{lab}" for a, lab in w)
                for m, w in zip(members, wheres, strict=True)
            ]
            # labels may hold spaces or dashes ("wild type", "R-1"); names may not
            name = re.sub(r"\W", "_", "extinct_" + "_".join(parts))
        self._claim(name, "extinction group")
        self._extinctions.append(
            ExtinctionSpec(
                name,
                tuple(members),
                _number(threshold, "extinction threshold"),
                _flag(noise_floor, "noise_floor"),
                tuple(wheres) if any(wheres) else (),
            )
        )

    # ---------------------------------------------------------------- IR, checks, summary
    def to_ir(self) -> ModelIR:
        """The complete model description (plain data; see ``tenqsp.ir``)."""
        return ModelIR(
            name=self.name,
            axes=tuple(self._axes),
            params=tuple(self._params),
            fields=tuple(self._fields),
            compartments=tuple(self._compartments),
            definitions=tuple(self._definitions),
            integrals=tuple(self._integrals),
            transitions=tuple(self._transitions),
            terms=tuple(self._terms),
            boundaries=tuple(self._boundaries),
            inits=tuple(self._inits),
            doses=tuple(self._doses),
            lags=tuple(self._lags),
            bioavailabilities=tuple(self._bioavailabilities),
            observables=tuple(self._observables),
            extinctions=tuple(self._extinctions),
            transfers=tuple(self._transfers),
        )

    def to_text(self) -> str:
        """The model as text in the TenQSP notation (``docs/language/notation.md``); it reads
        back with ``from_text`` to the same model."""
        from .writer import to_text

        return to_text(self.to_ir())

    def to_latex(self) -> str:
        """The equations as a paper shows them (an amsmath ``align*`` block): every transition
        written out category by category, then named formulas, boundaries and starting
        values; see ``latex``."""
        from .latex import to_latex

        return to_latex(self.to_ir())

    def explain(self) -> str:
        """The model in plain language: what each term means and which numerical scheme
        carries it (a transit chain, its rate, mean transit time and CV, for ``erlang``
        transport), boundaries, starting values, doses, extinction groups and the complete
        list of numerical safeguards; see ``explain``. ``print(m.explain())``."""
        from .explain import explain

        return explain(self.to_ir())

    def diagnostics(self, sol=None) -> Report:
        """Accuracy guidance: per transported axis the cell Péclet number, the time
        to cross a cell and the Erlang CV. With a solution (``sol = m.solve(...)``) also:
        undershoot below ``atol`` followed by regrowth, negative values, solver
        retries and how well the ledger closed. ``print(m.diagnostics(sol))``."""
        from .explain import diagnostics

        return diagnostics(self, sol)

    @classmethod
    def from_text(cls, text: str, *, name: str | None = None, cells=None, params=None) -> Model:
        """Read a model written in the TenQSP notation (``docs/language/notation.md``); see
        ``from_text``. ``tq.load("model.tq")`` reads a file."""
        from .notation import from_text

        return from_text(text, name=name, cells=cells, params=params)

    @classmethod
    def from_ir(cls, ir: ModelIR) -> Model:
        """Rebuild a model from its description (e.g. one loaded with ``ModelIR.from_json``).
        Objects are available by name: ``m["I"]``."""
        m = cls(ir.name)
        for a in ir.axes:
            if a.type == "categorical":
                m.axis(a.name, labels=a.labels, units=a.units)
            else:
                m.axis(a.name, faces=a.faces, kind=a.kind, units=a.units)
        for p in ir.params:
            if p.over is None:
                m.param(p.name, p.value, units=p.units)
            else:
                m.param(p.name, p.value, over=p.over, units=p.units)
        for f in ir.fields:
            m.field(f.name, list(f.axes), units=f.units, weights=f.weights)
        for c in ir.compartments:
            m.compartment(c.name, units=c.units, volume=c.volume)
        for d in ir.definitions:
            m.define(d.name, d.expr)
        for i in ir.integrals:
            m._claim(i.name, "integral")
            m._integrals.append(i)
            m._handles[i.name] = Name(i.name, "integral")
        for t in ir.transitions:
            m._claim(t.name, "transitions")
            m._transitions.append(t)
            m._handles[t.name] = Transitions(t.name, t.axis)
        for o in ir.observables:
            m._claim(o.name, "observable")
        for e in ir.extinctions:
            m._claim(e.name, "extinction group")
        m._terms = list(ir.terms)
        m._boundaries = list(ir.boundaries)
        m._inits = list(ir.inits)
        m._doses = list(ir.doses)
        m._lags = list(ir.lags)
        m._bioavailabilities = list(ir.bioavailabilities)
        m._observables = list(ir.observables)
        m._extinctions = list(ir.extinctions)
        m._transfers = list(ir.transfers)
        return m

    def validate(self, *, raise_on_error: bool = True) -> Report:
        """Check the whole model; list every problem, warning and note (see ``tenqsp.validate``).
        Raises ``ModelError`` if there are errors, unless ``raise_on_error=False``."""
        return validate(self.to_ir(), raise_on_error=raise_on_error)

    def compile(self):
        """Compile the model to JAX functions (validates first). Builds are shared by models with
        the same structure, so calling it again, solving again, or changing only parameter
        values or dose times does not recompile."""
        from tenqsp.compile import build

        ir = self.to_ir()
        key = ir.content_hash()
        cached = self.__dict__.get("_compiled")
        if cached is None or cached[0] != key:
            cached = self.__dict__["_compiled"] = (key, build(ir))
        return cached[1]

    def solve(self, t_span, *, backend: str = "scipy", **options):
        """Solve the model over ``t_span = (start, end)``.

        ``backend="scipy"`` (default) or ``"jax"`` (the JAX BDF, for gradients and batches;
        same results; see ``jax_backend``). Options: ``times``, ``params`` (e.g.
        ``{"k": 2e-9}``), ``theta``, ``method`` (``"BDF"`` or ``"Radau"``), ``rtol``, ``atol``,
        ``max_step``, ``regimen`` (a ``Regimen`` replacing the model's doses);
        see ``solve``.
        """
        if backend == "scipy":
            return self.compile().solve(t_span, **options)
        if backend == "jax":
            from tenqsp.backends.jax_backend import solve

            return solve(self.compile(), t_span, **options)
        raise ValueError(f"backend must be 'scipy' or 'jax', not {backend!r}.")

    def simulator(self, outputs=None, **options):
        """A simulator: the model built once, run with values, regimens and times given per
        run. ``sim = m.simulator(outputs=["conc"])``, then ``sim(scenario)`` or
        ``sim(params={"CL": 5}, regimen=tq.bolus("depot", 100, every=12, n=7),
        times=[...])``. ``outputs``: observables, fields or compartments (default: every
        observable). Solver options (``backend`` "scipy" or "jax", ``method``, ``rtol``,
        ``atol``, ``max_step``, ``ledger``) are fixed here. See ``simulate``."""
        from .simulate import Simulator

        return Simulator(self, outputs, **options)

    def simulate(
        self,
        scenario=None,
        *,
        outputs=None,
        params=None,
        regimen=None,
        times=None,
        start=None,
        **options,
    ):
        """One run: ``m.simulate(scenario)`` or ``m.simulate(params=..., regimen=...,
        times=...)``; shorthand for ``m.simulator(outputs, **options)(...)``."""
        from .simulate import MODEL_DOSES, Simulator

        sim = Simulator(self, outputs, **options)
        return sim(
            scenario,
            params=params,
            regimen=MODEL_DOSES if regimen is None else regimen,
            times=times,
            start=start,
        )

    def summary(self) -> str:
        """Model size: state variables per field and compartment, and which axes drive it."""
        ir = self.to_ir()
        counts = ir.state_count()
        sizes = {
            a.name: (len(a.faces) - 1 if a.type == "continuous" else len(a.labels)) for a in ir.axes
        }
        total = sum(counts.values())
        lines = [f"Model '{self.name}': {total:,} state variables"]
        for f in ir.fields:
            prod_txt = " x ".join(f"{sizes[a]} ({a})" for a in f.axes)
            lines.append(f"  field {f.name}: {counts[f.name]:,} = {prod_txt}")
        if ir.compartments:
            names = ", ".join(c.name for c in ir.compartments)
            lines.append(f"  compartments: {len(ir.compartments)} ({names})")
        if total > 100_000:
            lines.append(
                "  WARNING: above the 100,000-state target (ADR 0003): consider fewer cells, a "
                "separate field for structures that exist only at some sites, or a "
                "reachable-genotype axis."
            )
        return "\n".join(lines)


def _number(x, role: str) -> float:
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (Real, np.number)):
        raise TypeError(f"{role}: expected a number, got {x!r}.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{role}: {v} is not a finite number.")
    return v


def _flag(x, role: str) -> bool:
    if not isinstance(x, (bool, np.bool_)):
        raise TypeError(f"{role}: expected True or False, got {x!r}.")
    return bool(x)
