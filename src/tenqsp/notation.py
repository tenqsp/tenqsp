# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The model language: PDE notation as text -> a ``Model``.

The notation is specified in ``docs/language/notation.md`` (approved
2026-10-22). The reader builds the model through the Python API, so a model written as text
and the same model written in Python give the same IR.

Step 3.2 reads everything a model of compartments needs: ``model``, ``axis``, ``param``
(numbers and tables in nested braces), ``compartment``, named formulas (with the three
spellings of a conditional, ADR 0006), ``d_t(X) = …``, ``X(0) = …``, ``dose`` and
``observable``. Fields, fluxes, transitions, boundaries and extinction groups come in
step 3.3; their statements are recognised and refused with a message saying so.

Formulas inside statements are the formula language (``docs/language/formulas.md``), read by
the same ``ast`` allow-list walker (ADR 0002). Nothing here calls ``eval``.
"""

from __future__ import annotations

import ast
import math
import re
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sympy as sp

from . import _notation_ast as A
from .expr import FUNCTIONS, SHAPES, FormulaError
from .expr import parse as parse_formula
from .ir import IntegralSpec
from .model import Model, Name
from .validate import Issue, ModelError
from .writer import _flux_text

__all__ = ["NotationError", "from_text", "load", "table"]

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_OPENERS, _CLOSERS = "([{", ")]}"
# a boundary condition under an equation, "at z=0: -D*d_z(C) = h*(C - Cb)" (IR 0.10)
_AT_START = re.compile(rf"at\s+{_NAME}\s*=")
_AT = re.compile(rf"at\s+({_NAME})\s*=\s*([^:]+?)\s*:\s*(.*)$", re.S)


class NotationError(ModelError):
    """A model text cannot be read. The message lists every problem with its line."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        ValueError.__init__(
            self,
            f"The model text has {len(problems)} problem(s):\n\n" + "\n\n".join(problems),
        )
        self.issues = [Issue("error", "model text", p) for p in problems]


# ----------------------------------------------------------------------------- statements
# words that start a statement (and so cannot name a formula)
_KEYWORDS = frozenset(
    "model axis param compartment observable dose flux field transitions extinction "
    "inflow outflow dirichlet robin zero_flux".split()
)


@dataclass
class _Statement:
    """One statement: its lines (number, text without comment), joined for reading."""

    lines: list[tuple[int, str]]

    @property
    def lineno(self) -> int:
        return self.lines[0][0]

    @property
    def text(self) -> str:
        """The statement on one line (continuation lines joined with a space)."""
        return " ".join(t.strip() for _, t in self.lines)

    @property
    def first(self) -> str:
        return self.lines[0][1].strip()


def _strip_comment(line: str) -> str:
    quote = None
    for i, c in enumerate(line):
        if quote:
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "#":
            return line[:i]
    return line


def _depth_change(text: str) -> int:
    d, quote = 0, None
    for c in text:
        if quote:
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c in _OPENERS:
            d += 1
        elif c in _CLOSERS:
            d -= 1
    return d


def _statements(text: str) -> list[_Statement]:
    """Split the text into statements. A line continues the statement above when a bracket
    is still open, when it is indented, when it starts with an operator (``+ - * /``), or
    when it is the ``elif``/``else`` of an ``if`` block."""
    out: list[_Statement] = []
    depth = 0
    for no, raw in enumerate(textwrap.dedent(text.expandtabs(4)).splitlines(), 1):
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip()
        at_line = depth == 0 and _AT_START.match(stripped) is not None
        cont = (
            bool(out)
            and not at_line
            and (
                depth > 0
                or line[0] in " \t"
                or stripped[0] in "+-*/"
                or out[-1].lines[-1][1].rstrip().endswith(",")
                or (
                    re.match(r"(elif\b|else\s*:)", stripped) is not None
                    and out[-1].first.startswith("if ")
                )
            )
        )
        if cont:
            out[-1].lines.append((no, line))
        else:
            out.append(_Statement([(no, line)]))
            depth = 0
        depth = max(depth + _depth_change(line), 0)
    return out


def _split_top(text: str, sep: str = ",") -> list[str]:
    """Split at ``sep`` outside brackets and quotes."""
    parts, d, quote, start = [], 0, None, 0
    for i, c in enumerate(text):
        if quote:
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c in _OPENERS:
            d += 1
        elif c in _CLOSERS:
            d -= 1
        elif c == sep and d == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts]


def _top_level_word(text: str, word: str) -> int:
    """Index of ``word`` (a whole word) outside brackets and quotes, or -1."""
    d, quote = 0, None
    for i, c in enumerate(text):
        if quote:
            if c == quote:
                quote = None
            continue
        if c in "\"'":
            quote = c
        elif c in _OPENERS:
            d += 1
        elif c in _CLOSERS:
            d -= 1
        elif d == 0 and text.startswith(word, i):
            before = text[i - 1] if i else " "
            after = text[i + len(word)] if i + len(word) < len(text) else " "
            if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                return i
    return -1


def _options(text: str, allowed: Sequence[str]) -> tuple[str, dict[str, str]]:
    """Split ``head key=value key=value`` into the head and the options. A value runs up to
    the next ``key=`` of an allowed key outside brackets (so it may contain spaces)."""
    found = []
    d, quote = 0, None
    for i, c in enumerate(text):
        if quote:
            if c == quote:
                quote = None
            continue
        if c in "\"'":
            quote = c
        elif c in _OPENERS:
            d += 1
        elif c in _CLOSERS:
            d -= 1
        elif d == 0 and (i == 0 or text[i - 1] in " \t"):
            m = re.match(rf"({_NAME})\s*=(?!=)", text[i:])
            if m and m.group(1) in allowed:
                found.append((i, m.group(1), i + m.end()))
            elif m and found and found[-1][1] != "units":
                # after the options start, an unknown key is a misspelt option (after units=,
                # the units check below explains what went wrong)
                import difflib

                hint = difflib.get_close_matches(m.group(1), list(allowed), n=1)
                raise _Problem(
                    f"unknown option '{m.group(1)}='"
                    + (f"; did you mean '{hint[0]}='?" if hint else "")
                    + f" The options here are {', '.join(k + '=' for k in allowed)}."
                )
    head = text[: found[0][0]] if found else text
    opts = {}
    for k, (_, key, start) in enumerate(found):
        end = found[k + 1][0] if k + 1 < len(found) else len(text)
        if key in opts:
            raise _Problem(f"'{key}=' is given twice.")
        opts[key] = _unquote(text[start:end].strip())
        if key == "units" and re.search(r"[,=]", opts[key]):
            raise _Problem(
                f"units={opts[key]}: units are one word such as 1/h or mL/h; put other "
                "options before units=, or give them on their own line."
            )
    return head.strip(), opts


def _top_level_char(text: str, char: str) -> int:
    """Index of the first ``char`` outside brackets and quotes (``=`` not part of ``==``,
    ``<=``, ``>=``, ``!=``), or -1."""
    d, quote = 0, None
    for i, c in enumerate(text):
        if quote:
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c in _OPENERS:
            d += 1
        elif c in _CLOSERS:
            d -= 1
        elif d == 0 and c == char:
            nxt = text[i + 1] if i + 1 < len(text) else ""
            prev = text[i - 1] if i else ""
            if char == "=" and (nxt == "=" or prev in "=<>!"):
                continue
            return i
    return -1


def _unquote(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _line_key(problem: str) -> int:
    m = re.match(r"line (\d+)", problem)
    return int(m.group(1)) if m else 0


def _norm(text: str | None) -> str | None:
    return None if text is None else re.sub(r"[\s()]", "", text)


class _Problem(Exception):
    """A problem in one statement (turned into a message with the line by the reader)."""

    def __init__(self, message: str, column: int | None = None):
        super().__init__(message)
        self.column = column


class _ProblemAt(_Problem):
    """A problem found later, reported at the statement where it is."""

    def __init__(self, statement: _Statement, message: str):
        super().__init__(message)
        self.statement = statement


# ----------------------------------------------------------------------------- values
def _number(text: str, what: str) -> float:
    """A number, possibly written as arithmetic of numbers (``4/100``, ``-1e-9``)."""
    try:
        v = parse_formula(text, {})
    except FormulaError as e:
        if "Unknown name" in str(e):
            raise _Problem(
                f"{what} must be a number, not '{text}'. To compute a value from other "
                f"parameters, name a formula instead (e.g. 'ktr = 4/mtt')."
            ) from None
        raise _Problem(f"{what}: {e}") from None
    try:
        out = float(v)
    except TypeError:
        raise _Problem(f"{what} must be a number, not '{text}'.") from None
    if not math.isfinite(out):
        raise _Problem(f"{what} is not finite ('{text}').")
    return out


def _label(node: ast.expr, what: str) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise _Problem(f"{what}: '{ast.unparse(node)}' is not a label (a name, or text in quotes).")


def _braces(text: str, what: str):
    """A table in nested braces, ``{S: 1e-9, R: {a: 1, b: 2}}``, as nested dicts of floats."""
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError as e:
        raise _Problem(f"{what}: cannot read the table ({e.msg}).") from None

    def walk(node):
        if isinstance(node, ast.Dict):
            out = {}
            for k, v in zip(node.keys, node.values, strict=True):
                if k is None:
                    raise _Problem(f"{what}: '**' is not allowed in a table.")
                lab = _label(k, what)
                if lab in out:
                    raise _Problem(f"{what}: label '{lab}' is given twice.")
                out[lab] = walk(v)
            return out
        return _number(ast.unparse(node), what)

    if not isinstance(tree, ast.Dict):
        raise _Problem(f"{what}: a table is written in braces, {{S: 1e-9, R: 1e-12}}.")
    return walk(tree)


def _labels_list(text: str, what: str) -> list[str]:
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError:
        tree = None
    if not isinstance(tree, ast.List):
        raise _Problem(f"{what}: write the labels in square brackets, e.g. [S, R].")
    return [_label(e, what) for e in tree.elts]


def _numbers_list(text: str, what: str) -> list[float]:
    t = text.strip()
    if not (t.startswith("[") and t.endswith("]")):
        raise _Problem(f"{what}: write the numbers in square brackets, e.g. [0, 0.5, 1].")
    return [_number(p, what) for p in _split_top(t[1:-1]) if p]


# ----------------------------------------------------------------------------- the reader
_SCHEME = re.compile(r"\s+using\s+sharp(?:\s+on\s+([A-Za-z_][\w\s,]*?))?\s*$")
_BOUNDARY = re.compile(
    rf"(inflow|outflow|dirichlet|robin|zero_flux)\s+({_NAME})\s+at\s+({_NAME})\s*=\s*"
    r"(lower|upper)\b\s*(.*)$",
    re.S,
)


@dataclass
class _Flux:
    name: str
    frm: tuple[str, tuple]  # (compartment or field, slice)
    to: tuple[str, tuple]
    expr: ast.expr
    st: _Statement


@dataclass
class _FluxRef:
    flux: str
    key: tuple  # ("eq", name, where) or ("inflow", field)
    sign: int
    sums: tuple[str, ...]
    divisor: str | None
    st: _Statement


class _Lift(ast.NodeTransformer):
    """Turn the notation's operators inside a formula into model objects: ``int_a``/``sum_s``
    into integrals, slices into slice values, ``grad_x(S)`` into ``grad(S, x)``, and calls
    of named formulas ``h(a)`` into ``h``."""

    def __init__(self, reader: _Reader, integral_name: str | None = None):
        self.r = reader
        self.integral_name = integral_name
        self.changed = False

    def visit_Call(self, node: ast.Call):
        name = A.call_name(node)
        if name is None:
            return self.generic_visit(node)
        if name == A.SLICE:
            self.changed = True
            field, where = A.slice_of(node)
            return ast.Name(self.r.slice_value(field, where))
        for prefix in ("int", "sum"):
            if A.operator_axes(name, prefix):
                self.changed = True
                return ast.Name(self.integral(node))
        axes = A.operator_axes(name, "grad")
        if axes:
            self.changed = True
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
                raise _Problem(f"{name}(…) takes one field: {name}(S).")
            return ast.Call(ast.Name("grad"), [node.args[0], ast.Name(axes[0])], [])
        if A.operator_axes(name, "d") or A.operator_axes(name, "trans"):
            raise _Problem(
                f"'{name}(…)' stands for a whole term of a field's equation, as in "
                f"d_t(F) = … - d_a(v*F) or + trans_s(M, F); it cannot be part of a formula."
            )
        if name in self.r.def_axes and not (
            self.r.def_axes[name] is None and name in FUNCTIONS
        ):  # a formula named like a function leaves the function callable (formulas.md §2)
            self.changed = True
            declared = self.r.def_axes[name]
            got = [a.id if isinstance(a, ast.Name) else None for a in node.args]
            if node.keywords or None in got or (declared is not None and got != declared):
                want = f"{name}({', '.join(declared)})" if declared else name
                raise _Problem(
                    f"'{A.unparse(node)}': {name} is a named formula, not a function; write "
                    f"{want} (or just {name})."
                )
            return ast.Name(name)
        return self.generic_visit(node)

    def integral(self, node: ast.Call) -> str:
        top, axes = node, []
        while True:
            name = A.call_name(node)
            kind = next((p for p in ("int", "sum") if name and A.operator_axes(name, p)), None)
            if kind is None:
                break
            if len(node.args) != 1 or node.keywords:
                raise _Problem(f"{name}(…) takes one formula.")
            (ax,) = A.operator_axes(name, kind)
            what = self.r.m._kind.get(ax)
            if what != "axis":
                raise _Problem(f"{name}: '{ax}' is not an axis.")
            continuous = self.r.m[ax].type == "continuous"
            if kind == "int" and not continuous:
                raise _Problem(f"'{ax}' is categorical: add over it with sum_{ax}(…).")
            if kind == "sum" and continuous:
                raise _Problem(f"'{ax}' is continuous: integrate over it with int_{ax}(…).")
            axes.append(ax)
            node = node.args[0]
        name = self.integral_name if top is self.r.top_integral else None
        sl = A.slice_of(node)
        if sl:
            target = self.r.m[sl[0]].at(**dict(sl[1]))
            return self.r.m.integral(target, over=axes, name=name).name
        inner = _Lift(self.r)
        text = A.unparse(inner.visit(node))
        if name is None:  # the same integral in another axis order is stored once
            for i in self.r.m._integrals:
                if i.integrand == text and set(i.over) == set(axes) and not i.where:
                    return i.name
        return self.r.m.integral(text, over=axes, name=name).name


@dataclass
class _Reader:
    text: str
    name: str | None = None
    cells: Mapping[str, int] | None = None
    params: Mapping | None = None
    problems: list[str] = field(default_factory=list)
    where_lines: dict[tuple[str, str], _Statement] = field(default_factory=dict)
    m: Model | None = None

    def problem(self, st: _Statement, message: str, column: int | None = None) -> None:
        no, line = st.lines[0]
        out = f"line {no}: {line.strip()}"
        if column is not None and len(st.lines) == 1:
            indent = len(line) - len(line.lstrip())
            out += "\n" + " " * (len(f"line {no}: ") + column - indent) + "^"
        self.problems.append(f"{out}\n{message}")

    def run(self) -> Model:
        statements = _statements(self.text)
        kinds: dict[str, list[_Statement]] = {}
        # a condition "at z=0: …" belongs to the equation right above it (other conditions
        # of the same equation may come between)
        self.at_owner: dict[int, _Statement | None] = {}
        owner: _Statement | None = None
        for st in statements:
            k = self.kind(st)
            kinds.setdefault(k, []).append(st)
            if k == "equation":
                owner = st
            elif k == "at":
                self.at_owner[id(st)] = owner
            else:
                owner = None
        names = [self.model_name(st) for st in kinds.get("model", [])]
        if len(names) > 1:
            self.problem(kinds["model"][1], "the model is named twice.")
        self.m = Model(names[0] if names and names[0] else (self.name or "model"))
        self.used_cells: set[str] = set()
        self.implied: set[str] = set()  # compartments declared only by their equation
        self.used_params: set[str] = set()
        self.def_axes = self.collect_definitions(kinds)
        self.top_integral = None
        self.fluxes: dict[str, _Flux] = {}
        self.flux_refs: list[_FluxRef] = []
        self.equations_seen: dict[tuple, _Statement] = {}
        self.init_parts: dict[str, list[tuple[tuple, _Statement]]] = {}
        self.pending_odes: dict[str, tuple[str | None, _Statement]] = {}
        self.failed: list[_Statement] = []
        steps = [
            ("axis", self.axis),
            ("param", self.param),
            ("compartment", self.compartment),
            ("field", self.field_),
            ("implied", None),
            ("flux", self.flux),
            ("definition", self.definition),
            ("if", self.if_block),
            ("transitions", self.transitions),
            ("equation", self.equation),
            ("at", self.at_condition),
            ("boundary", self.boundary),
            ("init", self.init),
            ("dose", self.dose),
            ("arrival", self.arrival),
            ("observable", self.observable),
            ("extinction", self.extinction),
        ]
        for kind, handler in steps:
            if kind == "implied":
                self.implied_compartments(kinds.get("equation", []))
                continue
            for st in kinds.get(kind, []):
                self.guard(st, handler)
        for st in kinds.get("unknown", []):
            self.unknown(st)
            self.failed.append(st)
        self.emit_fluxes()
        for target, (text, st) in self.pending_odes.items():
            if text is not None:
                self.guard(st, lambda st, t=target, x=text: self.m.ode(self.m[t], self.lift(x)))
        self.leftover_keys()
        if not self.problems:
            self.check_model()
        if self.problems:
            raise NotationError(sorted(self.problems, key=_line_key))
        return self.m

    def guard(self, st: _Statement, handler) -> None:
        before = len(self.problems)
        try:
            self._guard(st, handler)
        finally:
            if len(self.problems) > before:
                self.failed.append(st)

    def _guard(self, st: _Statement, handler) -> None:
        try:
            handler(st)
        except _ProblemAt as e:
            self.problem(e.statement, str(e))
        except _Problem as e:
            self.problem(st, str(e), e.column)
        except A.NotationSyntaxError as e:
            self.problem(st, str(e))
        except (ValueError, TypeError, KeyError) as e:  # from the Python API
            self.problem(st, str(e).strip('"'))

    # ------------------------------------------------------------------ dispatch
    def kind(self, st: _Statement) -> str:
        t = st.first
        word = t.split()[0] if t.split() else ""
        named = re.match(rf"({_NAME})\s*(\([^)]*\))?\s*=(?!=)", t)
        if named and named.group(1) in _KEYWORDS:  # `dose = 2*x`: a keyword as a formula name
            return "unknown"
        if word in ("model", "axis", "param", "compartment", "observable", "dose", "flux"):
            return word
        if word in ("field", "transitions", "extinction"):
            return word
        if re.match(rf"(lag|bioavailability)\s+{_NAME}", t):  # not `lag = …`, a formula
            return "arrival"
        if word in ("inflow", "outflow", "dirichlet", "robin", "zero_flux"):
            return "boundary"
        if _AT_START.match(t):
            return "at"
        if re.match(r"if\b", t):
            return "if"
        if re.match(r"d_t\s*\(", t):
            return "equation"
        if re.match(rf"{_NAME}(\s*\[[^\]]*\])?\s*\(\s*0\s*\)\s*=", t):
            return "init"
        if re.match(rf"{_NAME}\s*(\([^)]*\))?\s*=(?!=)", t):
            return "definition"
        return "unknown"

    def unknown(self, st: _Statement) -> None:
        t = st.first
        named = re.match(rf"({_NAME})\s*(\([^)]*\))?\s*=(?!=)", t)
        if named and named.group(1) in _KEYWORDS:
            self.problem(
                st,
                f"'{named.group(1)}' starts a statement, so it cannot name a formula; choose "
                "another name.",
            )
        elif t.startswith("d/dt"):
            self.problem(st, "an equation is written d_t(X) = …, not d/dt(X).")
        elif "<-" in t:
            self.problem(st, "assignments use '=', not '<-'.")
        elif t.startswith(("elif", "else")):
            self.problem(st, f"'{t.split()[0]}' without an 'if' block above it.")
        elif re.match(r"init\b", t):
            self.problem(st, "a value at time 0 is written X(0) = value.")
        elif re.match(r"transfer\b", t):
            self.problem(
                st,
                "material moving between places is a flux: flux J: A -> B = rate*A, then "
                "- J in the equation of A and + J in the equation of B.",
            )
        elif re.match(rf"{_NAME}(\s*\[[^\]]*\])?\s*\(\s*0\s*\)\s*$", t):
            self.problem(st, "a value at time 0 needs a value: X(0) = value.")
        else:
            self.problem(
                st,
                "cannot read this statement. A statement starts with a keyword (model, axis, "
                "param, field, compartment, flux, transitions, inflow, outflow, dose, lag, "
                "bioavailability, observable, extinction, …), or is an equation d_t(X) = …, a "
                "value at time 0 X(0) = …, or a named formula name = ….",
            )

    def model_name(self, st: _Statement) -> str | None:
        m = re.fullmatch(rf"model\s+({_NAME})", st.text)
        if not m:
            self.problem(st, "write the model name as one word: model my_model.")
            return None
        return m.group(1)

    def collect_definitions(self, kinds) -> dict[str, list[str] | None]:
        """Names of formulas (and the axes declared in their brackets), before reading any
        formula, so that ``h(a)`` is understood wherever it is used."""
        out: dict[str, list[str] | None] = {}
        for st in kinds.get("definition", []):
            m = re.match(rf"({_NAME})\s*(?:\(([^)]*)\))?\s*=", st.first)
            out[m.group(1)] = (
                None if m.group(2) is None else [a.strip() for a in m.group(2).split(",")]
            )
        for st in kinds.get("if", []):
            for _, line in st.lines[1:]:
                a = re.fullmatch(rf"\s+({_NAME})\s*=(?!=).*", line)
                if a:
                    out.setdefault(a.group(1), None)
        return out

    # ------------------------------------------------------------------ formulas
    def lift(self, text: str, integral_name: str | None = None) -> str:
        """A formula of the notation as formula-language text (unchanged when it uses none
        of the notation's operators)."""
        if "<-" in text:
            raise _Problem("assignments use '=', not '<-'.")
        node = A.parse(text)
        return self.lift_node(node, integral_name, original=text)

    def lift_node(self, node, integral_name=None, original: str | None = None) -> str:
        self.top_integral = node if integral_name else None
        t = _Lift(self, integral_name)
        new = t.visit(node)
        if not t.changed and original is not None:
            return original.strip()
        return A.unparse(new)

    def slice_value(self, field_name: str, where, name: str | None = None) -> str:
        """The value of a field on a slice, read inside a formula: a sum over the fixed axes
        restricted to the slice (divided by the weights, which the sum multiplies by)."""
        spec = self.field_spec(field_name)
        self.check_where(field_name, where)
        integrand = field_name if spec.weights is None else f"{field_name}/({spec.weights})"
        over: tuple[str, ...] = ()  # the slice fixes those axes; nothing left to add up
        for i in self.m._integrals:
            if (i.integrand, i.over, tuple(i.where)) == (integrand, over, tuple(where)):
                if name is None or i.name == name:
                    return i.name
        if name is None:
            labels = "_".join(re.sub(r"\W", "_", lab) for _, lab in where)
            name = f"{field_name}__{labels}"
        self.m._claim(name, "integral")
        self.m._integrals.append(IntegralSpec(name, integrand, over, where=tuple(where)))
        self.m._handles[name] = Name(name, "integral")
        return name

    def field_spec(self, name: str):
        for f in self.m._fields:
            if f.name == name:
                return f
        what = self.m._kind.get(name)
        raise _Problem(
            f"'{name}' is {'not declared' if what is None else 'a ' + what}; only a field can "
            "be sliced."
        )

    def check_where(self, field_name: str, where) -> None:
        spec = self.field_spec(field_name)
        for ax, lab in where:
            if ax not in spec.axes:
                raise _Problem(f"'{field_name}' has no axis '{ax}' (it has {list(spec.axes)}).")
            if self.m[ax].type != "categorical":
                raise _Problem(f"a slice fixes categorical axes; '{ax}' is continuous.")
            labels = self.m[ax].labels
            if lab not in labels:
                raise _Problem(f"axis '{ax}' has no category '{lab}' (it has {list(labels)}).")

    def end(self, text: str):
        """A compartment, field or slice written in a statement: (name, where)."""
        node = A.parse(text)
        sl = A.slice_of(node)
        if sl:
            self.check_where(*sl)
            return sl
        if not isinstance(node, ast.Name):
            raise _Problem(f"'{text.strip()}' is not a compartment, field or slice.")
        what = self.m._kind.get(node.id)
        if what not in ("compartment", "field"):
            raise _Problem(
                f"'{node.id}' is {'not declared' if what is None else 'a ' + what}; give a "
                "compartment, a field or a slice of a field."
            )
        return node.id, ()

    def handle(self, name: str, where=()):
        h = self.m[name]
        return h.at(**dict(where)) if where else h

    # ------------------------------------------------------------------ declarations
    def axis(self, st: _Statement) -> None:
        m = re.match(rf"axis\s+({_NAME})\s*(.*)$", st.text)
        if not m:
            raise _Problem("an axis needs a name: axis a in [0, 2] n=40.")
        name, rest = m.group(1), m.group(2)
        if rest.startswith("categorical"):
            head, opts = _options(rest[len("categorical") :], ("units",))
            labels = _labels_list(head, f"axis '{name}'")
            self.m.axis(name, labels=labels, **opts)
            if self.cells and name in self.cells:
                raise _Problem(f"cells= changes continuous axes; '{name}' is categorical.")
            self.where_lines[("axis", name)] = st
            return
        head, opts = _options(rest, ("n", "units", "kind", "spacing", "ratio", "faces"))
        kw: dict = {k: opts[k] for k in ("units", "kind", "spacing") if k in opts}
        if "ratio" in opts:
            kw["ratio"] = _number(opts["ratio"], f"axis '{name}', ratio")
        n = None
        if "n" in opts:
            n_val = _number(opts["n"], f"axis '{name}', n")
            if n_val != int(n_val) or n_val < 1:
                raise _Problem(f"axis '{name}': n is a number of cells (a whole number >= 1).")
            n = int(n_val)
        if self.cells and name in self.cells:
            self.used_cells.add(name)
            if "faces" in opts:
                raise _Problem(
                    f"cells= cannot change axis '{name}': its cells are given by faces=."
                )
            n = int(self.cells[name])
        if "faces" in opts:
            if head:
                raise _Problem(f"axis '{name}': give faces=[...] or 'in [lower, upper] n=…'.")
            faces = _numbers_list(opts["faces"], f"axis '{name}', faces")
            if "n" in opts and n != len(faces) - 1:
                raise _Problem(
                    f"axis '{name}': faces=[...] gives {len(faces) - 1} cells but n={opts['n']}; "
                    "leave out n= (the faces decide it)."
                )
            self.m.axis(name, faces=faces, **kw)
        else:
            r = re.fullmatch(r"in\s*(\[.*\])", head)
            if not r:
                raise _Problem(
                    f"axis '{name}': write 'axis {name} in [lower, upper] n=cells', "
                    f"'axis {name} faces=[...]' or 'axis {name} categorical [A, B]'."
                )
            bounds = _numbers_list(r.group(1), f"axis '{name}'")
            if len(bounds) != 2:
                raise _Problem(f"axis '{name}': the range is [lower, upper].")
            if n is None:
                raise _Problem(f"axis '{name}': give the number of cells, n=….")
            self.m.axis(name, bounds[0], bounds[1], n, **kw)
        self.where_lines[("axis", name)] = st

    def param(self, st: _Statement) -> None:
        body, opts = _options(st.text[len("param") :], ("units",))
        parts = [p for p in _split_top(body) if p]
        if not parts:
            raise _Problem(
                "give parameters as name=value, e.g. param k=1e-9, beta=50, or names alone "
                "(values given at simulation): param ka, CL."
            )
        if "units" in opts and len(parts) > 1:
            raise _Problem("units= is for one parameter per line: param k = 1e-9 units=mL/h.")
        for part in parts:
            m = re.fullmatch(rf"({_NAME})\s*(?:\[([^\]]*)\])?\s*(?:=\s*(.+))?", part, re.S)
            if not m or (m.group(3) is None and "=" in part):
                raise _Problem(
                    f"'{part}': write name=value (or name[axis] = {{…}}), or the name alone."
                )
            name, over, value = m.group(1), m.group(2), (m.group(3) or "").strip()
            declared = m.group(3) is None  # no value: given at simulation
            override = self.params is not None and name in self.params
            if override:
                self.used_params.add(name)
            if over is None:
                if value.startswith("{"):
                    raise _Problem(
                        f"'{name}' has a table of values; say which axes it is over: "
                        f"{name}[s] = {{…}}."
                    )
                if override:
                    v = self.params[name]
                else:
                    v = None if declared else _number(value, f"parameter '{name}'")
                self.m.param(name, v, units=opts.get("units", ""))
            else:
                axes = [a.strip() for a in over.split(",")]
                if not all(re.fullmatch(_NAME, a) for a in axes):
                    raise _Problem(f"'{name}[{over}]': the brackets list axis names.")
                if override:
                    v = self.params[name]
                elif declared:
                    v = None
                elif value.startswith("{"):
                    v = _braces(value, f"parameter '{name}'")
                else:
                    raise _Problem(
                        f"'{name}[{over}]' needs one value per category, in braces: "
                        f"{name}[{over}] = {{…}}."
                    )
                over_arg = axes if len(axes) > 1 else axes[0]
                self.m.param(name, v, over=over_arg, units=opts.get("units", ""))
            self.where_lines[("parameter", name)] = st

    def compartment(self, st: _Statement) -> None:
        body, opts = _options(st.text[len("compartment") :], ("units", "volume"))
        if "=" in body:
            name = body.split("=")[0].strip()
            value = body.split("=", 1)[1].strip()
            raise _Problem(
                f"a declaration has no value; the value at time 0 is written {name}(0) = {value}."
            )
        names = [n for n in _split_top(body) if n]
        if not names or not all(re.fullmatch(_NAME, n) for n in names):
            raise _Problem("write compartment names: compartment P (or compartment A, B).")
        if opts and len(names) > 1:
            raise _Problem("units= and volume= are for one compartment per line.")
        for n in names:
            self.m.compartment(n, **opts)
            self.where_lines[("compartment", n)] = st

    def field_(self, st: _Statement) -> None:
        body, opts = _options(st.text[len("field") :], ("units", "weights"))
        m = re.fullmatch(rf"({_NAME})\s*\(([^)]*)\)\s*(=.*)?", body, re.S)
        if not m:
            raise _Problem("write a field with its axes: field I(a, z).")
        name, axes, value = m.group(1), m.group(2), m.group(3)
        if value:
            raise _Problem(
                f"a declaration has no value; the value at time 0 is written "
                f"{name}(0) = {value[1:].strip()}."
            )
        axes = [a.strip() for a in axes.split(",") if a.strip()]
        self.m.field(name, axes, **opts)
        self.where_lines[("field", name)] = st

    def undeclared_hint(self, name: str) -> str:
        """For a name made a compartment only by its equation: perhaps a field undeclared."""
        if name not in self.implied:
            return ""
        return (
            f" '{name}' is not declared, so its equation made it a compartment; if it is a "
            f"field, declare it with its axes: field {name}(<axes>)."
        )

    def implied_compartments(self, equations: list[_Statement]) -> None:
        """An equation d_t(X) of an undeclared name declares the compartment X."""
        for st in equations:
            m = re.match(rf"d_t\s*\(\s*({_NAME})\s*\)\s*=", st.text)
            if m and m.group(1) not in self.m._kind:
                self.m.compartment(m.group(1))
                self.where_lines[("compartment", m.group(1))] = st
                self.implied.add(m.group(1))

    def flux(self, st: _Statement) -> None:
        m = re.fullmatch(rf"flux\s+({_NAME})\s*:\s*(.+?)\s*->\s*(.+)", st.text, re.S)
        eq = _top_level_char(m.group(3), "=") if m else -1
        if not m or eq < 0:
            raise _Problem(
                "write a flux as 'flux J: from -> to = amount per time', e.g. "
                "flux infect: B -> I = k*B*P."
            )
        name, frm = m.group(1), m.group(2)
        to, expr = m.group(3)[:eq], m.group(3)[eq + 1 :]
        if name in self.fluxes:
            raise _Problem(f"flux '{name}' is declared twice.")
        if name in self.m._kind or name in self.def_axes:
            raise _Problem(f"the name '{name}' is already used.")
        f = _Flux(name, self.end(frm), self.end(to), A.parse(expr), st)
        if f.frm == f.to:
            raise _Problem(f"flux '{name}' leaves and arrives in the same place.")
        self.fluxes[name] = f
        self.where_lines[("flux", name)] = st

    def definition(self, st: _Statement) -> None:
        m = re.match(rf"({_NAME})\s*(?:\(([^)]*)\))?\s*=", st.first)
        name, axes = m.group(1), m.group(2)
        if axes is not None:
            for a in (x.strip() for x in axes.split(",")):
                if self.m._kind.get(a) != "axis":
                    raise _Problem(
                        f"'{name}({axes})': the brackets list the axes the formula varies "
                        f"over, and '{a}' is not an axis."
                    )
        first_rest = st.first[m.end() :].strip()
        rest_lines = [first_rest] + [t.strip() for _, t in st.lines[1:]]
        if _top_level_word(first_rest, "if") >= 0 and _top_level_word(first_rest, "else") < 0:
            text = self.cases(rest_lines)
        else:
            text = " ".join(rest_lines)
        node = A.parse(text)
        sl = A.slice_of(node)
        if sl:  # a named value on a slice: C_liver = C[organ=liver, sub=cellular]
            self.slice_value(*sl, name=name)
            self.where_lines[("definition", name)] = st
            self.where_lines[("integral", name)] = st
            return
        # a formula that is one integral names it: B_tot = int_z(sum_s(B))
        if A.call_name(node) and (
            A.operator_axes(A.call_name(node), "int") or A.operator_axes(A.call_name(node), "sum")
        ):
            self.lift_node(node, integral_name=name)
        else:
            self.m.define(name, self.lift_node(node, original=text))
        self.where_lines[("definition", name)] = st
        self.where_lines[("integral", name)] = st

    def transitions(self, st: _Statement) -> None:
        m = re.fullmatch(
            rf"transitions\s+({_NAME})\s+on\s+(\([^)]*\)|{_NAME})\s*:\s*(.+)", st.text, re.S
        )
        if not m:
            raise _Problem(
                "write transitions as 'transitions M on s: S -> R = mu, R -> S = nu' (moves on "
                "one line separated by commas, or one per indented line)."
            )
        name, axes_text, moves_text = m.groups()
        joint = axes_text.startswith("(")
        axes = [a.strip() for a in axes_text.strip("()").split(",")] if joint else axes_text
        moves = []
        for _, line in [(0, ln) for ln in [moves_text]]:
            moves += [p for p in _split_top(line) if p]
        # moves may also be one per line: split the original lines too
        if len(st.lines) > 1:
            moves = []
            first = st.lines[0][1].split(":", 1)[1] if ":" in st.lines[0][1] else ""
            joined = ""  # one move per line, or a rate continued while a bracket is open
            for line in [first] + [t for _, t in st.lines[1:]]:
                joined += line.strip()
                depth = sum(joined.count(c) for c in _OPENERS) - sum(
                    joined.count(c) for c in _CLOSERS
                )
                joined += " " if depth > 0 else ", "
            moves = [p for p in _split_top(joined) if p]
        rates = {}
        for mv in moves:
            mm = re.fullmatch(r"(.+?)\s*->\s*(.+?)\s*=(?!=)\s*(.+)", mv, re.S)
            if not mm:
                raise _Problem(f"'{mv}': write each move as FROM -> TO = rate.")
            frm, to, rate = (x.strip() for x in mm.groups())

            def category(c: str) -> str:
                if joint:
                    if not (c.startswith("(") and c.endswith(")")):
                        raise _Problem(
                            f"'{c}': on {axes_text} a category is a label per axis, e.g. "
                            f"({', '.join('S' for _ in axes)})."
                        )
                    return ",".join(_unquote(x) for x in _split_top(c[1:-1]))
                return _unquote(c)

            key = f"{category(frm)}->{category(to)}"
            if key in rates:
                raise _Problem(f"the move {frm} -> {to} is given twice.")
            rates[key] = self.lift(rate)
        self.m.transitions(name, axis=axes, rates=rates)
        self.where_lines[("transitions", name)] = st

    # ------------------------------------------------------------------ equations
    def equation(self, st: _Statement) -> None:
        m = re.match(r"d_t\s*\(\s*([^()]*?)\s*\)\s*=", st.text)
        if not m:
            raise _Problem("write an equation as d_t(X) = … (X a compartment, field or slice).")
        target, where = self.end(m.group(1))
        key = (target, where)
        if key in self.equations_seen:
            other = self.equations_seen[key].lineno
            raise _Problem(
                f"d_t({m.group(1)}) is given twice (also line {other}); write one equation per "
                "field or slice, with all its terms."
            )
        self.equations_seen[key] = st
        rhs = st.text[m.end() :].strip()
        scheme = _SCHEME.search(rhs)
        sharp_axes: set[str] | None = None
        if scheme:
            rhs = rhs[: scheme.start()]
            on = scheme.group(1)
            sharp_axes = {a.strip() for a in on.split(",")} if on else set()
        if not rhs:
            raise _Problem(f"d_t({m.group(1)}) = … needs a right-hand side.")
        if "d_t(" in rhs.replace(" ", ""):
            raise _Problem("d_t(…) appears on the right-hand side; write the rates themselves.")
        node = A.parse(rhs)
        ends = {f.name for f in self.fluxes.values() if target in (f.frm[0], f.to[0])}

        def has_flux(n) -> bool:
            return bool(A.names_in(n) & ends)

        items = A.terms(node, 1, has_flux)
        what = self.m._kind[target]
        self.where_lines[("equation", target)] = st
        rest = []
        for sign, term in items:
            if has_flux(term):
                self.flux_ref(st, ("eq", target, where), sign, term, ends)
            else:
                rest.append((sign, term))
        if what == "compartment":
            if sharp_axes is not None:
                raise _Problem(
                    "a compartment has no axis to move along: 'using sharp' is for fields."
                    + self.undeclared_hint(target)
                )
            for _, term in rest:
                for n in ast.walk(term):
                    ax = A.call_name(n) and A.operator_axes(A.call_name(n), "d")
                    if ax:
                        raise _Problem(
                            f"a compartment has no axis '{ax[0]}'; to use a field's total, "
                            f"integrate it: int_{ax[0]}(…)."
                        )
            # written after the fluxes are known (a flux into an inflow stays in this equation)
            text = None if not rest else A.join(rest) if len(rest) < len(items) else rhs
            self.pending_odes[target] = (text, st)
            return
        self.field_terms(st, target, where, rest, sharp_axes)

    def field_terms(self, st, F: str, where, items, sharp_axes) -> None:
        h = self.handle(F, where)
        moved: set[str] = set()
        for sign, term in items:
            name = A.call_name(term)
            d_axes = name and A.operator_axes(name, "d")
            t_axes = name and A.operator_axes(name, "trans")
            if d_axes:
                moved.add(self.transport(st, F, where, sign, term, d_axes[0], sharp_axes))
                continue
            if t_axes:
                self.transition_term(F, where, h, sign, term, t_axes)
                continue
            for n in ast.walk(term):
                inner = A.call_name(n)
                if inner and A.operator_axes(inner, "d"):
                    raise _Problem(
                        f"the transport term must be in conservation form: write "
                        f"-{inner}(v*{F}), not -v*{inner}({F}) (TenQSP keeps every cell's "
                        "books balanced only in this form)."
                    )
                if inner and A.operator_axes(inner, "trans"):
                    raise _Problem(f"write {inner}(M, X) as a term of its own: + {inner}(M, X).")
            rate = A.remove_factor(term, A.is_name(F)) if sign < 0 else None
            if rate is not None and F not in A.names_in(rate):
                self.m.hazard(h, self.lift_node(rate))
            else:
                self.m.reaction(h, self.lift_node(A.signed(sign, term)))
        if sharp_axes:
            missing = sorted(sharp_axes - moved)
            if missing:
                raise _Problem(
                    f"'using sharp on {', '.join(missing)}': this equation moves {F} along "
                    f"{sorted(moved) or 'no axis'}."
                )
        elif sharp_axes is not None and not moved:
            raise _Problem("'using sharp' is for an equation with a transport term -d_a(v*F).")

    def transport(self, st, F, where, sign, term, ax, sharp_axes) -> str:
        if where:
            raise _Problem(
                "transport and diffusion act on a whole field; write them in d_t(" + F + ")."
            )
        if len(term.args) != 1 or term.keywords:
            raise _Problem(f"d_{ax}(…) takes one formula.")
        if self.m._kind.get(ax) != "axis":
            raise _Problem(f"d_{ax}: '{ax}' is not an axis.")
        if self.m[ax].type != "continuous":
            raise _Problem(
                f"'{ax}' is categorical: there is no d_{ax}; moves between its categories are "
                f"written + trans_{ax}(M, {F}) with 'transitions M on {ax}: …'."
            )
        arg = term.args[0]

        def inner_d(n) -> bool:
            return A.call_name(n) == f"d_{ax}"

        if any(inner_d(n) for n in ast.walk(arg)):  # diffusion d_x(D*d_x(F))
            D = A.remove_factor(arg, inner_d)
            if D is None or isinstance(D, ast.Constant) and D.value == 1 and inner_d(arg):
                raise _Problem(f"write diffusion as d_{ax}(D*d_{ax}({F})), with the diffusivity D.")
            inner = next(n for n in ast.walk(arg) if inner_d(n))
            if not (len(inner.args) == 1 and A.is_name(F)(inner.args[0])):
                raise _Problem(f"diffusion is written d_{ax}(D*d_{ax}({F})).")
            if sign < 0:
                raise _Problem(
                    f"diffusion is written +d_{ax}(D*d_{ax}({F})); a negative diffusion has "
                    "no solution."
                )
            self.m.diffuse(self.m[F], axis=ax, D=self.lift_node(D))
            return ax
        v = A.remove_factor(arg, A.is_name(F))
        if v is None:
            fields = sorted(n for n in A.names_in(arg) if self.m._kind.get(n) == "field")
            if F in A.names_in(arg):
                raise _Problem(f"transport is written -d_{ax}(v*{F}), the velocity times {F}.")
            raise _Problem(
                f"in the equation of {F}, transport moves {F}: write -d_{ax}(v*{F})"
                + (f" (this one moves {', '.join(fields)})." if fields else ".")
            )
        # the velocity is what multiplies F; it may depend on fields, F itself included
        # (Burgers: -d_x(0.3*u*u)), or be driven by another field (-d_x(2*S*B))
        if sign > 0:
            raise _Problem(
                f"transport is written with a minus, -d_{ax}(v*{F}); to move toward lower "
                f"{ax}, give a negative velocity: -d_{ax}(-v*{F})."
            )
        sharp = sharp_axes is not None and (not sharp_axes or ax in sharp_axes)
        self.m.advect(
            self.m[F], axis=ax, velocity=self.lift_node(v), scheme="sharp" if sharp else "erlang"
        )
        return ax

    @staticmethod
    def inside_grad(node, name: str) -> bool:
        for n in ast.walk(node):
            c = A.call_name(n)
            if c and (c == "grad" or A.operator_axes(c, "grad")) and name in A.names_in(n):
                return True
        return False

    def transition_term(self, F, where, h, sign, term, axes) -> None:
        if sign < 0:
            raise _Problem(
                "a transition term is added, + trans_s(M, X): what leaves one category is "
                "derived from the moves."
            )
        if len(term.args) != 2 or term.keywords or not isinstance(term.args[0], ast.Name):
            raise _Problem("trans_s takes the transitions and what moves: trans_s(M, X).")
        M = term.args[0].id
        if self.m._kind.get(M) != "transitions":
            raise _Problem(f"'{M}' is not declared with 'transitions {M} on …'.")
        declared = self.m[M].axis if hasattr(self.m[M], "axis") else None
        declared = tuple(declared) if isinstance(declared, (tuple, list)) else (declared,)
        if declared != tuple(axes):
            want = axes[0] if len(axes) == 1 else "{" + ",".join(axes) + "}"
            have = declared[0] if len(declared) == 1 else "{" + ",".join(declared) + "}"
            raise _Problem(f"'{M}' moves along {have}, not {want}: write trans_{have}({M}, …).")
        X = term.args[1]
        sl = A.slice_of(X)
        eq = A.slice_text(F, where) if where else F
        if sl is not None and sl[0] == F and tuple(sorted(sl[1])) != tuple(sorted(where)):
            x = A.slice_text(*sl)
            raise _Problem(
                f"trans_{'_'.join(axes)}({M}, {x}) moves part of {F}, but this is the "
                f"equation of {eq}: write the transitions in the equation of what moves, "
                f"d_t({x}) = … + trans_{'_'.join(axes)}({M}, {x})."
            )
        itself = A.is_name(F)(X) or (sl is not None and sl[0] == F)
        axis = list(axes) if len(axes) > 1 else axes[0]
        of = h if itself else self.lift_node(X)
        self.m.transition(h, axis=axis, matrix=self.m[M], of=of)

    def flux_ref(self, st, key, sign, term, ends) -> None:
        """A term of an equation (or an inflow) using a flux whose end it is: ±J, summed
        over axes and divided by the weights or volume."""
        divisor = None
        node = term
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            divisor, node = A.unparse(node.right), node.left
        sums = []
        while A.call_name(node) and A.operator_axes(A.call_name(node), "sum"):
            if len(node.args) != 1:
                break
            sums += A.operator_axes(A.call_name(node), "sum")
            node = node.args[0]
        if not (isinstance(node, ast.Name) and node.id in ends) or (
            divisor is not None and A.names_in(ast.parse(divisor, mode="eval")) & ends
        ):
            used = sorted(A.names_in(term) & ends)
            J = used[0]
            integrated = [
                c for n in ast.walk(term) if (c := A.call_name(n)) and A.operator_axes(c, "int")
            ]
            if integrated:
                ax = A.operator_axes(integrated[0], "int")[0]
                raise _Problem(
                    f"the flux {J} joins two places cell by cell, so it cannot be integrated "
                    f"over '{ax}' here: an arrow has exactly two ends that match element by "
                    "element (summed only over categories, sum_s). A process with more "
                    "participants, like B + P -> I, is written without an arrow: give the rate "
                    f"a plain name ({J} = …, no 'flux') and write its term in every equation it "
                    f"changes, e.g. - int_{ax}(…) here."
                )
            raise _Problem(
                f"the flux {J} moves material in and out of this place, so it appears "
                f"as -{J} where it leaves and +{J} where it arrives (divided by the "
                f"volume, or summed over axes: sum_s({J})), not as "
                f"'{A.unparse(A.signed(sign, term))}'."
            )
        self.flux_refs.append(_FluxRef(node.id, key, sign, tuple(sums), divisor, st))

    def at_condition(self, st: _Statement) -> None:
        """``at z=0: <flux of F along z> = <formula>``, ``at z=1: F = <value>`` or
        ``at l=1: outflow``, under the equation of F, as a paper writes it."""
        eq = self.at_owner.get(id(st))
        if eq is None:
            raise _Problem(
                "a boundary condition 'at z=…: …' belongs to an equation: write it right "
                "below d_t(F) = … (other conditions of the same equation may come between)."
            )
        target = re.match(r"d_t\s*\(\s*([^()]*?)\s*\)", eq.text)
        if not target:  # the equation itself is reported
            return
        F, where = self.end(target.group(1))
        if where or self.m._kind.get(F) != "field":
            raise _Problem(
                "a boundary condition is for a field's equation (the ends of its axes); "
                f"'{target.group(1)}' is not a whole field." + self.undeclared_hint(F)
            )
        m = _AT.fullmatch(st.text)
        if not m:
            raise _Problem(
                "write a boundary condition as 'at z=0: -D*d_z(C) = …', 'at z=1: C = …' or "
                "'at a=1: outflow'."
            )
        ax, end_text, body = m.group(1), m.group(2).strip(), m.group(3).strip()
        if self.m._kind.get(ax) != "axis" or ax not in self.field_spec(F).axes:
            raise _Problem(f"'{ax}' is not an axis of {F}.")
        end = end_text if end_text in ("lower", "upper") else _number(end_text, "the end")
        h = self.m[F]
        self.where_lines[("boundary", f"{F} at {ax}={self.side_of(ax, end)}")] = st
        if body == "outflow":
            self.m.at(h, outflow=True, **{ax: end})
            return
        i = _top_level_char(body, "=")
        if i < 0:
            raise _Problem(
                f"a condition is an equation: 'at {ax}={end_text}: <flux of {F}> = …', "
                f"'{F} = <value>', or 'outflow'."
            )
        lhs, rhs = body[:i].strip(), body[i + 1 :].strip()
        if not rhs:
            raise _Problem("the condition needs a right-hand side.")
        node = A.parse(lhs)
        if A.is_name(F)(node):  # the field held at a value: C(t, 1) = 1
            self.m.at(h, value=self.lift(rhs), **{ax: end})
            return
        terms = [t for t in self.m._terms if t.target == F and t.axis == ax]
        adv = [t for t in terms if t.kind == "advection"]
        dif = [t for t in terms if t.kind == "diffusion"]
        grad = f"d_{ax}"
        if A.call_name(node) == grad and len(node.args) == 1 and A.is_name(F)(node.args[0]):
            if adv or not dif:
                raise _Problem(
                    f"{grad}({F}) = … fixes the gradient, which is a flux only for a field that "
                    f"diffuses without flowing along {ax}; write the flux: "
                    f"{self.flux_text(F, ax, terms)} = …"
                )
            try:
                zero = float(rhs) == 0.0
            except ValueError:
                zero = False
            if not zero:
                raise _Problem(
                    f"write a gradient condition as the flux it gives: "
                    f"{self.flux_text(F, ax, terms)} = …"
                )
            self.m.at(h, flux=0, **{ax: end})
            return
        if not terms:
            raise _Problem(
                f"{F} does not move along {ax} (no -d_{ax}(v*{F}) or "
                f"d_{ax}(D*d_{ax}({F})) in its equation), so it has no flux there to set."
            )
        if not self.same_flux(lhs, F, ax, terms):
            raise _Problem(
                f"the left side is the flux of {F} along {ax}, from its equation: "
                f"{self.flux_text(F, ax, terms)} (not {lhs})."
            )
        self.m.at(h, flux=self.lift(rhs), **{ax: end})

    def side_of(self, ax: str, end) -> str:
        if isinstance(end, str):
            return end
        return "upper" if np.isclose(end, self.m[ax].faces[-1]) else "lower"

    @staticmethod
    def flux_text(F: str, ax: str, terms) -> str:
        """The flux of F along ax as its equation's transport terms give it."""
        return _flux_text(F, ax, terms)

    def same_flux(self, lhs: str, F: str, ax: str, terms) -> bool:
        """Is ``lhs`` the flux of F along ax (the same formula, however arranged)?"""
        placeholder = "__grad__"

        class _Grad(ast.NodeTransformer):
            def visit_Call(self, n):
                if A.call_name(n) == f"d_{ax}" and len(n.args) == 1 and A.is_name(F)(n.args[0]):
                    return ast.Name(placeholder)
                return self.generic_visit(n)

        try:
            got = self.lift_node(_Grad().visit(A.parse(lhs)))
        except _Problem:
            return False
        want = self.flux_text(F, ax, terms).replace(f"d_{ax}({F})", placeholder)
        vectors = {p.name for p in self.m._params if p.over is not None}
        names: dict[str, object] = {}
        for text in (got, want):
            for n in re.findall(_NAME, text):
                names[n] = sp.IndexedBase(n) if n in vectors else sp.Symbol(n, real=True)
        try:
            a, b = parse_formula(got, names), parse_formula(want, names)
        except FormulaError:
            return False
        return sp.simplify(a - b) == 0

    def boundary(self, st: _Statement) -> None:
        m = _BOUNDARY.fullmatch(st.text)
        if not m:
            raise _Problem(
                "write a boundary as e.g. 'inflow I at a=lower: k*B*P', 'outflow I at a=upper "
                "[to L]', 'dirichlet C at z=upper: 1', 'robin C at z=upper: h*(M/V - C) "
                "[with M]' or 'zero_flux C at z=lower'."
            )
        kind, F, ax, side, rest = m.groups()
        if self.m._kind.get(F) != "field":
            raise _Problem(f"'{F}' is not a field; boundaries are ends of a field's axis.")
        rest = rest.strip()
        h = self.m[F]
        if kind in ("zero_flux", "outflow"):
            to = None
            if kind == "outflow":
                t = re.fullmatch(rf"(?:to\s+({_NAME}))?", rest)
                if not t:
                    raise _Problem(f"write 'outflow {F} at {ax}={side}' or '… to L'.")
                to = t.group(1)
            elif rest:
                raise _Problem("a zero_flux end takes nothing more.")
            if kind == "zero_flux":
                self.m.boundary(h, axis=ax, at=side, kind="zero_flux")
            else:
                self.m.outflow(h, axis=ax, at=side, to=None if to is None else self.m[to])
            self.where_lines[("boundary", f"{F} at {ax}={side}")] = st
            return
        if not rest.startswith(":"):
            raise _Problem(f"'{kind} {F} at {ax}={side}: …' needs a value after the colon.")
        rest = rest[1:].strip()
        w = re.search(rf"\s+with\s+({_NAME})\s*$", rest)
        partner = None
        if w:
            if kind == "inflow":
                raise _Problem(
                    "an inflow is not taken from a compartment with 'with'; to take it from "
                    "another place, write it as a flux."
                )
            partner, rest = self.m[w.group(1)], rest[: w.start()]
        if kind == "inflow":
            node = A.parse(rest)
            ends = {f.name for f in self.fluxes.values() if f.to[0] == F}
            if A.names_in(node) & ends:
                self.flux_ref(st, ("inflow", F), 1, node, ends)
            self.m.inflow(h, axis=ax, at=side, flux=self.lift(rest))
        elif kind == "dirichlet":
            self.m.boundary(
                h, axis=ax, at=side, kind="dirichlet", value=self.lift(rest), to=partner
            )
        else:  # robin: h*(outside - F)
            node = A.parse(rest)
            found = []

            def gap(n) -> bool:
                ok = (
                    isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub) and A.is_name(F)(n.right)
                )
                if ok:
                    found.append(n)
                return ok

            coef = A.remove_factor(node, gap)
            if coef is None or F in A.names_in(coef) or F in A.names_in(found[0].left):
                raise _Problem(
                    f"write the exchange as h*(outside - {F}): a coefficient times the outside "
                    f"level minus {F}."
                )
            self.m.boundary(
                h,
                axis=ax,
                at=side,
                kind="robin",
                value=self.lift_node(found[0].left),
                h=self.lift_node(coef),
                to=partner,
            )
        self.where_lines[("boundary", f"{F} at {ax}={side}")] = st

    # ------------------------------------------------------------------ initial values, doses
    def amount_and_shape(self, node):
        """``amount*shape(…)`` (or several shapes, one per axis) -> (amount, shape text);
        None when the formula has no shape (a density)."""
        inside = {  # shapes within shape(…) belong to that formula
            id(n)
            for c in ast.walk(node)
            if A.call_name(c) == "shape"
            for n in ast.walk(c)
            if n is not c
        }
        shapes = sorted(  # in the order written
            (
                n
                for n in ast.walk(node)
                if A.call_name(n) in (*SHAPES, "shape") and id(n) not in inside
            ),
            key=lambda n: (n.lineno, n.col_offset),
        )
        if not shapes:
            return None
        total = node
        for _ in shapes:
            total = (
                A.remove_factor(total, lambda n: A.call_name(n) in (*SHAPES, "shape"))
                if total is not None
                else None
            )
        if (
            total is None
            or any(A.call_name(n) in (*SHAPES, "shape") for n in ast.walk(total))
            or set(A.names_in(total))
            & {n for n in A.names_in(total) if self.m._kind.get(n) == "axis"}
        ):
            raise _Problem(
                "write an amount times one shape, e.g. 1e7*normal(z, mean=0.1, sd=0.08), "
                "1e7*point(z, at=0) or 1e7*shape(exp(-z))."
            )
        parts = []
        for shp in shapes:  # one per axis; their product is the shape
            if A.call_name(shp) == "shape":  # any formula of the axes
                if len(shp.args) != 1 or shp.keywords:
                    raise _Problem("shape(…) takes one formula of the axes, e.g. shape(exp(-z)).")
                parts.append(self.lift_node(shp.args[0]))
            else:
                parts.append(A.unparse(shp))
        text = parts[0] if len(parts) == 1 else "*".join(f"({p})" for p in parts)
        return self.lift_node(total), text

    def density_value(self, node, example: str):
        """``density(f)`` or ``a*density(f)`` -> the density formula ``a*f`` (shapes in it are
        the curve itself); None when the value has no ``density(…)`` factor."""
        found = []

        def is_density(n):
            if A.call_name(n) == "density":
                found.append(n)
                return True
            return False

        rest = A.remove_factor(node, is_density)
        if rest is None:
            if any(A.call_name(n) == "density" for n in ast.walk(node)):
                raise _Problem(f"density(…) multiplies the rest of the value, e.g. {example}.")
            return None
        call = found[0]
        if len(call.args) != 1 or call.keywords:
            raise _Problem(f"density(…) takes one formula, e.g. {example}.")
        if isinstance(rest, ast.Constant) and rest.value == 1:
            return self.lift_node(call.args[0])
        return self.lift_node(ast.BinOp(rest, ast.Mult(), call.args[0]))

    def init(self, st: _Statement) -> None:
        m = re.match(rf"({_NAME}(?:\s*\[[^\]]*\])?)\s*\(\s*0\s*\)\s*=", st.text)
        target, where = self.end(m.group(1))
        what = self.m._kind[target]
        value = st.text[m.end() :].strip()
        if not value:
            raise _Problem("a value at time 0 needs a value: X(0) = value.")
        prior = self.init_parts.setdefault(target, [])
        for w, other in prior:
            if all(dict(w).get(a, lab) == lab for a, lab in where) and all(
                dict(where).get(a, lab) == lab for a, lab in w
            ):
                raise _Problem(
                    f"the value at time 0 of this part of {target} is already given (line "
                    f"{other.lineno})."
                )
        prior.append((where, st))
        kw = {"where": dict(where)} if where else {}
        node = A.parse(value)
        dens = self.density_value(node, "1e7*density(normal(z, mean=0.1, sd=0.08))")
        if dens is not None:  # a density formula as it is (shapes: the curve itself)
            self.m.init(self.m[target], dens, **kw)
            self.where_lines[("init of", target)] = st
            return
        split = self.amount_and_shape(node)
        if split is None:
            self.m.init(self.m[target], self.lift(value), **kw)
        else:
            if what != "field":
                raise _Problem("a shape spreads an amount over an axis; a compartment has none.")
            total, text = split
            self.m.init(self.m[target], total=total, shape=text, **kw)
        self.where_lines[("init of", target)] = st

    def dose(self, st: _Statement) -> None:
        m = re.fullmatch(
            rf"dose\s+({_NAME}(?:\s*\[[^\]]*\])?)\s+at\s+t\s*=\s*(.+?)\s*:\s*(.+)", st.text
        )
        if not m:
            raise _Problem(
                "write a dose as 'dose P at t=0, 24: 1e8' or 'dose P at t=0 to 144 every 24: 1e8'."
            )
        target_text, times_text, amounts_text = m.groups()
        target, where = self.end(target_text)
        r = re.fullmatch(r"(.+?)\s+to\s+(.+?)\s+every\s+(.+)", times_text)
        if r:
            t0, t1, step = (_number(x, "dose time") for x in r.groups())
            if step <= 0:
                raise _Problem("the interval after 'every' must be > 0.")
            k = (t1 - t0) / step
            if k < 0 or abs(k - round(k)) > 1e-9 * max(1.0, abs(k)):
                raise _Problem(
                    f"from {r.group(1)} to {r.group(2)} is not a whole number of steps of "
                    f"{r.group(3)}; the last dose must fall on the end time."
                )
            times = [t0 + i * step for i in range(round(k) + 1)]
        else:
            times = [_number(x, "dose time") for x in _split_top(times_text)]
        texts = _split_top(amounts_text)
        if len(texts) not in (1, len(times)):
            raise _Problem(
                f"{len(times)} dose time(s) but {len(texts)} amounts: give one amount for "
                "all, or one per time."
            )
        axes = next((f.axes for f in self.m._fields if f.name == target), ())
        free = [a for a in axes if a not in dict(where)]
        cats = [
            a for a in free if any(x.name == a and x.type == "categorical" for x in self.m._axes)
        ]
        if cats:  # a dose goes into one category of each categorical axis
            example = ", ".join(f"{a}={self.m[a].labels[0]}" for a in cats)
            raise _Problem(
                f"a dose into {target} goes into one category of {', '.join(cats)}: name it, "
                f"e.g. dose {target}[{example}] at …"
            )
        if not free:  # a compartment or one element: amounts as they are
            amounts = [self.lift(a) for a in texts]
            self.m.dose(
                self.handle(target, where),
                times=times,
                amounts=amounts if len(amounts) > 1 else amounts[0],
            )
        else:  # into a field, as an initial value: amount*shape, or a density
            amounts, shapes, densities = [], set(), []
            for a in texts:
                node = A.parse(a)
                dens = self.density_value(node, "density(1e6*exp(-z))")
                if dens is not None:
                    densities.append(dens)
                    continue
                split = self.amount_and_shape(node)
                if split is None:
                    if not set(A.names_in(node)) & set(free):
                        raise _Problem(
                            f"a dose into the field {target} needs a shape saying where the "
                            f"amount goes along {free}, e.g. {a}*point({free[0]}, at=0); or "
                            f"density({a}) to add {a} per unit of {free[0]} everywhere."
                        )
                    densities.append(self.lift(a))
                else:
                    amounts.append(split[0])
                    shapes.add(split[1])
            if amounts and densities or len(shapes) > 1:
                raise _Problem(
                    "the doses of one line share one shape (or are all densities); write a "
                    "line per shape."
                )
            if amounts:
                self.m.dose(
                    self.handle(target, where),
                    times=times,
                    amounts=amounts if len(amounts) > 1 else amounts[0],
                    shape=shapes.pop(),
                )
            else:
                self.m.dose(
                    self.handle(target, where),
                    times=times,
                    density=densities if len(densities) > 1 else densities[0],
                )
        self.where_lines.setdefault(("dose into", target), st)

    def arrival(self, st: _Statement) -> None:
        m = re.fullmatch(
            rf"(lag|bioavailability)\s+({_NAME}(?:\s*\[[^\]]*\])?)\s*=\s*(.+)", st.text, re.S
        )
        if not m:
            word = st.text.split()[0]
            raise _Problem(
                f"write it as '{word} depot = formula' (a place, then '=' and a formula of "
                "parameters), e.g. 'lag depot = tlag' or 'bioavailability depot = F'."
            )
        word, target, value = m.groups()
        if "[" in target:
            raise _Problem(f"a {word} applies to a whole place; write its name only.")
        getattr(self.m, word)(self.m[target], self.lift(value.strip()))
        self.where_lines[(word, target)] = st

    def observable(self, st: _Statement) -> None:
        m = re.fullmatch(rf"observable\s+({_NAME})\s*=\s*(.+)", st.text, re.S)
        if not m:
            raise _Problem("write an output as 'observable name = formula'.")
        expr = m.group(2).strip()
        sl = A.slice_of(A.parse(expr))
        if sl:
            self.check_where(*sl)
            self.m.observable(m.group(1), self.handle(*sl))
        else:
            self.m.observable(m.group(1), self.lift(expr))
        self.where_lines[("observable", m.group(1))] = st

    def extinction(self, st: _Statement) -> None:
        m = re.fullmatch(
            rf"extinction\s+(?:({_NAME})\s*:\s*)?(.+?)\s+below\s+(\S+)(\s+with\s+noise_floor)?",
            st.text,
        )
        if not m:
            raise _Problem(
                "write an extinction group as 'extinction [name:] P, I below 1 [with noise_floor]'."
            )
        name, members, thr, floor = m.groups()
        group = [self.handle(*self.end(x)) for x in _split_top(members)]
        self.m.extinction(
            group, threshold=_number(thr, "threshold"), name=name, noise_floor=bool(floor)
        )
        self.where_lines[("extinction group", self.m._extinctions[-1].name)] = st

    # ------------------------------------------------------------------ fluxes
    def weight_of(self, end) -> str | None:
        name, _ = end
        for c in self.m._compartments:
            if c.name == name:
                return c.volume
        return self.field_spec(name).weights

    def free_axes(self, end) -> list[str]:
        name, where = end
        if self.m._kind[name] == "compartment":
            return []
        fixed = {a for a, _ in where}
        return [a for a in self.field_spec(name).axes if a not in fixed]

    def emit_fluxes(self) -> None:
        for f in self.fluxes.values():
            # a line that could not be read may hold one of its terms: say nothing about it
            ends = {f.name, f.frm[0], f.to[0]}
            if any(ends & set(re.findall(_NAME, st.text)) for st in self.failed):
                continue
            try:
                self.emit_flux(f)
            except _Problem as e:
                self.problem(e.statement if hasattr(e, "statement") else f.st, str(e))

    def emit_flux(self, f: _Flux) -> None:
        refs = [r for r in self.flux_refs if r.flux == f.name]
        frm_key = ("eq", f.frm[0], f.frm[1])
        to_eq, to_in = ("eq", f.to[0], f.to[1]), ("inflow", f.to[0])
        for r in refs:
            if r.key not in (frm_key, to_eq, to_in):
                place = A.slice_text(r.key[1], r.key[2]) if r.key[2] else r.key[1]
                ends = " -> ".join(A.slice_text(*e) if e[1] else e[0] for e in (f.frm, f.to))
                raise _ProblemAt(
                    r.st,
                    f"flux {f.name} is declared {ends} (line {f.st.lineno}), but this equation "
                    f"is of {place}.",
                )
        leave = [r for r in refs if r.key == frm_key]
        arrive = [r for r in refs if r.key in (to_eq, to_in)]
        frm_text = A.slice_text(*f.frm) if f.frm[1] else f.frm[0]
        to_text = A.slice_text(*f.to) if f.to[1] else f.to[0]
        if len(leave) != 1 or leave[0].sign > 0:
            raise _Problem(
                f"flux {f.name} leaves {frm_text}: the equation d_t({frm_text}) needs exactly "
                f"one term -{f.name}"
                + (" (divided by its volume)" if self.weight_of(f.frm) else "")
                + "."
            )
        if len(arrive) != 1 or arrive[0].sign < 0:
            raise _Problem(
                f"flux {f.name} arrives in {to_text}: d_t({to_text})"
                + (f" (or an inflow of {f.to[0]})" if self.m._kind[f.to[0]] == "field" else "")
                + f" needs exactly one term +{f.name}"
                + (" (divided by its volume)" if self.weight_of(f.to) else "")
                + "."
            )
        inflow = arrive[0].key == to_in
        to_side = self.free_axes(f.to)
        if inflow:
            ends_at = {
                b.axis for b in self.m._boundaries if b.target == f.to[0] and b.kind == "inflow"
            }
            to_side = [a for a in to_side if a not in ends_at]
        axes = list(dict.fromkeys(self.free_axes(f.frm) + to_side))
        for r, end in ((leave[0], f.frm), (arrive[0], f.to)):
            place = A.slice_text(*end) if end[1] else end[0]
            side = to_side if r is arrive[0] else self.free_axes(end)
            need = sorted(set(axes) - set(side))
            if sorted(r.sums) != need:
                write = f.name
                for a in need:
                    write = f"sum_{a}({write})"
                raise _ProblemAt(
                    r.st,
                    f"flux {f.name} varies over {axes or 'nothing'}; {place} varies over "
                    f"{side or 'nothing'}, so it takes {write} here.",
                )
            want = None if (inflow and r is arrive[0]) else self.weight_of(end)
            got = r.divisor
            if _norm(got) != _norm(want):
                if want is None:
                    msg = (
                        f"flux {f.name}: {place} has no volume or weights; use {f.name} "
                        "without dividing."
                    )
                else:
                    msg = (
                        f"flux {f.name} is an amount per time; in the equation of {place} "
                        f"divide it by {place}'s "
                        f"{'volume' if self.m._kind[end[0]] == 'compartment' else 'weights'}, "
                        f"{want}"
                        + (f" (not {got})" if got else "")
                        + ", so that what leaves one place arrives in the other."
                    )
                raise _ProblemAt(r.st, msg)
        pending = (text for text, _ in self.pending_odes.values() if text)
        used_elsewhere = any(
            re.search(rf"\b{re.escape(f.name)}\b", t) for t in (*self.formula_texts(), *pending)
        )
        if inflow or used_elsewhere:
            if f.frm[1] or f.to[1]:
                raise _Problem(
                    f"flux {f.name} goes from or to a slice; it can be used only in its two "
                    "equations."
                )
            self.m.define(f.name, self.lift_node(f.expr))
        if inflow:  # the leaving term stays in its equation, as written (sums, divisor)
            self.keep_term(f, leave[0], f.frm, -1)
            return
        source = f.frm[0]
        if f.frm[1]:
            frm = f.frm
            test = lambda n: A.slice_of(n) == frm or A.is_name(source)(n)  # noqa: E731
        else:
            test = A.is_name(source)
        rate = A.remove_factor(f.expr, test)
        if rate is None:  # not a rate times its source (zero order, saturable, ...): two terms
            if self.weight_of(f.frm) and source not in A.names_in(f.expr):
                raise _Problem(
                    f"a flux out of {source} (a concentration) is a rate times what it takes "
                    f"from: write {f.name} = rate*{source}. A flux that does not read {source} "
                    "(zero order) can leave only a place without a volume or weights."
                )
            if f.frm[1] or f.to[1]:
                raise _Problem(
                    f"a flux from or to a slice is a rate times what it takes from: write "
                    f"{f.name} = rate*{A.slice_text(*f.frm) if f.frm[1] else source}."
                )
            if not (inflow or used_elsewhere):
                self.m.define(f.name, self.lift_node(f.expr))
            self.keep_term(f, leave[0], f.frm, -1)
            self.keep_term(f, arrive[0], f.to, +1)
            return
        self.m.transfer(self.handle(*f.frm), to=self.handle(*f.to), rate=self.lift_node(rate))

    def keep_term(self, f: _Flux, ref: _FluxRef, end, sign: int) -> None:
        """Put ±flux back into the equation of ``end`` as it was written there."""
        term = f.name
        for a in reversed(ref.sums):
            term = f"sum_{a}({term})"
        if ref.divisor:
            d = ref.divisor
            term = f"{term}/{d}" if re.fullmatch(r"[\w.]+(\[[\w, ]*\])?", d) else f"{term}/({d})"
        op = "-" if sign < 0 else "+"
        if self.m._kind[end[0]] == "compartment":
            text, st = self.pending_odes[end[0]]
            self.pending_odes[end[0]] = (f"{text} {op} {term}" if text else f"{op}{term}", st)
        else:
            self.m.reaction(self.handle(*end), self.lift(f"{op}{term}"))

    def formula_texts(self):
        for t in self.m._terms:
            yield t.expr
        for d in self.m._definitions:
            yield d.expr
        for i in self.m._integrals:
            yield i.integrand
        for b in self.m._boundaries:
            yield b.value or ""
        for o in self.m._observables:
            yield o.expr
        for tr in self.m._transitions:
            for *_, rate in tr.rates:
                yield rate

    # ------------------------------------------------------------------ formulas (cases)
    def cases(self, lines: list[str]) -> str:
        """The cases brace: lines 'value if condition', the last 'value otherwise'."""
        values, conds = [], []
        for k, line in enumerate(lines):
            last = k == len(lines) - 1
            i = _top_level_word(line, "if")
            if i >= 0:
                if last:
                    raise _Problem(
                        "the last case must be 'value otherwise' (a value when no condition holds)."
                    )
                values.append(line[:i].strip())
                conds.append(line[i + 2 :].strip())
            else:
                mm = re.fullmatch(r"(.+?)\s+otherwise", line)
                if not mm or not last:
                    raise _Problem(
                        f"'{line}': each case is 'value if condition', and the last one "
                        "'value otherwise'."
                    )
                values.append(mm.group(1).strip())
        text = f"({values[-1]})"
        for v, c in zip(reversed(values[:-1]), reversed(conds), strict=True):
            text = f"({v}) if ({c}) else {text}"
        return text

    def if_block(self, st: _Statement) -> None:
        """if cond: / elif cond: / else: with indented assignments; one formula per name."""
        branches: list[tuple[str | None, list[tuple[str, str]]]] = []
        for no, line in st.lines:
            s = line.strip()
            indented = line[0] in " \t"
            head = re.fullmatch(r"(if|elif)\s+(.+?)\s*:\s*(.*)", s) or re.fullmatch(
                r"(else)\s*:\s*(.*)", s
            )
            if not indented and head:
                kw = head.group(1)
                if kw == "else":
                    cond, after = None, head.group(2)
                else:
                    cond, after = head.group(2), head.group(3)
                if (kw == "if") != (not branches):
                    raise _Problem(f"line {no}: '{kw}' is out of place in the if block.")
                if branches and branches[-1][0] is None:
                    raise _Problem(f"line {no}: nothing may follow 'else' in an if block.")
                branches.append((cond, []))
                if after:
                    raise _Problem(f"line {no}: put each assignment on its own indented line.")
                continue
            if not indented:
                raise _Problem(f"line {no}: inside an if block, indent the assignments.")
            if re.match(r"(if|elif|else)\b", s):
                raise _Problem(
                    f"line {no}: an if block inside another is not read; join the conditions "
                    "with 'and' instead."
                )
            a = re.fullmatch(rf"({_NAME})\s*=(?!=)\s*(.+)", s)
            if not a:
                raise _Problem(
                    f"line {no}: an if block only names formulas (name = value); equations "
                    "d_t(X) = … are never inside one."
                )
            if not branches:
                raise _Problem(f"line {no}: an assignment before 'if'.")
            if any(n == a.group(1) for n, _ in branches[-1][1]):
                raise _Problem(f"line {no}: '{a.group(1)}' is set twice in one branch.")
            branches[-1][1].append((a.group(1), a.group(2).strip()))
        if branches[-1][0] is not None:
            raise _Problem(
                "an if block ends with 'else:' (the values when no condition holds), so that "
                "every name always has a value."
            )
        names = [n for n, _ in branches[0][1]]
        for cond, body in branches:
            got = [n for n, _ in body]
            if sorted(got) != sorted(names):
                which = "else" if cond is None else f"'{cond}'"
                raise _Problem(
                    f"every branch must set the same names: the first sets {names}, the "
                    f"branch {which} sets {got}."
                )
        for n in names:
            vals = [dict(body)[n] for _, body in branches]
            text = f"({vals[-1]})"
            for v, (cond, _) in zip(reversed(vals[:-1]), reversed(branches[:-1]), strict=True):
                text = f"({v}) if ({cond}) else {text}"
            self.m.define(n, self.lift(text))
            self.where_lines[("definition", n)] = st

    # ------------------------------------------------------------------ whole model
    def leftover_keys(self) -> None:
        for what, given, used in (
            ("cells", self.cells, self.used_cells),
            ("params", self.params, self.used_params),
        ):
            extra = sorted(set(given or {}) - used)
            if extra:
                kind = "continuous axes" if what == "cells" else "parameters"
                self.problems.append(
                    f"{what}= names {extra}, which are not {kind} of the model text."
                )

    def check_model(self) -> None:
        report = self.m.validate(raise_on_error=False)
        for issue in report.errors:
            st = self.statement_of(issue.where)
            if st is None:
                self.problems.append(f"model ({issue.where}): {issue.message}")
                continue
            column = None
            c = re.search(r"at column (\d+) of '(.*?)'(?=[.,;:]?(?:\s|$))", issue.message)
            if c and len(st.lines) == 1:
                at = st.lines[0][1].find(c.group(2))
                if at >= 0:
                    column = at + int(c.group(1)) - 1
            self.problem(st, issue.message, column)

    def statement_of(self, where: str) -> _Statement | None:
        for pattern, kind in (  # the validator's names for parts read from one statement
            (r"init \d+ of '([^']+)'", "init of"),
            (r"dose \d+ into '([^']+)'", "dose into"),
            (r"lag of '([^']+)'", "lag"),
            (r"bioavailability of '([^']+)'", "bioavailability"),
            (r"extinction group '([^']+)'", "extinction group"),
        ):
            k = re.search(pattern, where)
            if k and (kind, k.group(1)) in self.where_lines:
                return self.where_lines[(kind, k.group(1))]
        b = re.search(r"boundary \w+ of '([^']+)' at (\S+)", where)
        if b and ("boundary", f"{b.group(1)} at {b.group(2)}") in self.where_lines:
            return self.where_lines[("boundary", f"{b.group(1)} at {b.group(2)}")]
        m = re.search(
            r"\b(rate|reaction|hazard|advection|diffusion|switch|mutate) (?:term )?(?:on|of|for) "
            r"'([^']+)'",
            where,
        )
        if m:
            return self.where_lines.get(("equation", m.group(2))) or self.where_lines.get(
                ("compartment", m.group(2))
            )
        for (kind, name), st in self.where_lines.items():
            if re.match(rf"{re.escape(kind)} '{re.escape(name)}'", where):
                return st
        for (_, name), st in self.where_lines.items():
            if name and re.search(rf"'{re.escape(name)}'", where):
                return st
        return None


# ----------------------------------------------------------------------------- entry points
def from_text(
    text: str,
    *,
    name: str | None = None,
    cells: Mapping[str, int] | None = None,
    params: Mapping | None = None,
) -> Model:
    """Read a model written in the TenQSP notation (``docs/language/notation.md``).

    ``name`` is used when the text has no ``model`` line. ``cells={"a": 80}`` changes the
    number of cells of continuous axes; ``params={"V": {...}}`` replaces values written in
    the text. Every problem is reported at once, with its line, as a ``NotationError``;
    the model is validated before it is returned."""
    if not isinstance(text, str):
        raise TypeError("from_text reads a string; to read a file use tq.load(path).")
    return _Reader(text, name=name, cells=cells, params=params).run()


def load(
    path: str | Path,
    *,
    cells: Mapping[str, int] | None = None,
    params: Mapping | None = None,
) -> Model:
    """Read a model file (``.tq``, plain text in the TenQSP notation). The model is named by
    its ``model`` line, or else by the file name. See ``from_text`` for ``cells`` and
    ``params``."""
    p = Path(path)
    if p.suffix != ".tq":
        raise ValueError(f"tq.load reads .tq model files, not '{p.name}'.")
    return from_text(p.read_text(encoding="utf-8"), name=p.stem, cells=cells, params=params)


def table(data, *, axes: Sequence[str], value: str) -> dict:
    """A long table (one row per value: a column per axis and a value column) as the nested
    dict a parameter over several axes takes, e.g.
    ``tq.table(pd.read_csv("volumes.csv"), axes=["organ", "sub"], value="V")``.

    ``data`` is anything indexable by column name (a pandas DataFrame, a dict of lists). A
    combination given twice is an error; a missing one is reported when the parameter is
    made."""
    if isinstance(axes, str):
        axes = [axes]
    if not axes:
        raise ValueError("tq.table: give the axis columns, e.g. axes=['organ', 'sub'].")
    try:
        cols = [list(data[a]) for a in axes]
        vals = list(data[value])
    except (KeyError, IndexError) as e:
        raise ValueError(f"tq.table: no column {e} in the data.") from None
    out: dict = {}
    for row, v in zip(zip(*cols, strict=True), vals, strict=True):
        node = out
        for lab in row[:-1]:
            node = node.setdefault(str(lab), {})
            if not isinstance(node, dict):
                raise ValueError(f"tq.table: inconsistent rows at {row}.")
        key = str(row[-1])
        if key in node:
            raise ValueError(f"tq.table: the row {tuple(str(x) for x in row)} is given twice.")
        v = float(np.asarray(v))
        node[key] = v
    return out
