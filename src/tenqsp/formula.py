# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Formulas outside a model: ``tq.formula(text)``.

Another program that writes formulas in TenQSP's language (tenqsp-fit's fit part, for one)
reads them here with the same grammar, functions and messages as a model, lists the names a
formula uses, and evaluates it as a JAX function of values for those names. Inside a model,
names mean parameters, compartments and fields; here they are plain values given by the
caller. What only makes sense over a model's axes (shape functions, ``grad``, a parameter
over an axis ``k[s]``, a field at one end ``F(z=lower)``) is refused with a message.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence

import sympy as sp

from .expr import SHAPES, FormulaError, is_condition, parse
from .model import Formula, as_formula

__all__ = ["formula"]


def _tree(text: str) -> ast.Expression:
    try:
        return ast.parse(text.strip().replace("^", "**"), mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"Cannot read formula {text.strip()!r}: {e.msg}.") from None


def names_in(text: str) -> tuple[str, ...]:
    """The names a formula uses (not the functions it calls), sorted, after checking that
    the formula is one usable outside a model."""
    tree = _tree(text)
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            raise FormulaError(
                f"'{ast.unparse(node)}' in {text.strip()!r}: a parameter over an axis belongs "
                "in a model; outside a model every name is one value."
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(id(node.func))
            if node.func.id in SHAPES or node.func.id == "grad":
                raise FormulaError(
                    f"'{node.func.id}(...)' in {text.strip()!r} works over a model's axis; it "
                    "cannot be used outside a model."
                )
    names = sorted(
        {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in called}
    )
    expr = parse(text, {n: sp.Symbol(n) for n in names})  # grammar, functions, messages
    if is_condition(expr):
        raise FormulaError(
            f"{text.strip()!r} is a condition (true or false), not a number; use it to choose "
            f"between values: 'v1 if {text.strip()} else v2'."
        )
    return tuple(names)


def formula(text: str | Formula) -> Formula:
    """A formula in TenQSP's formula language, read and checked now (the same grammar,
    functions and messages as in a model, ``docs/language/formulas.md``).

    ``f.names`` lists the names it uses; ``f.jax(["a", "b"])`` is a JAX function of values
    for those names, in that order::

        f = tq.formula("CL_pop * (WT/70)^0.75")
        f.names                           # ('CL_pop', 'WT')
        g = f.jax(["CL_pop", "WT"])
        g(0.04, jnp.array([60.0, 80.0]))  # broadcasts like any JAX function

    It is an ordinary formula: ``tq.exp(f)``, ``2*f`` and model statements accept it."""
    if isinstance(text, Formula):
        text = text.text
    if not isinstance(text, str):
        raise TypeError(f"tq.formula reads formula text, not {type(text).__name__}.")
    names_in(text)
    return as_formula(text.strip())


def compile_free(text: str, names: Sequence[str]) -> Callable:
    """``Formula.jax``: a JAX function of values for ``names`` (positional, in that order)."""
    if isinstance(names, str):
        raise TypeError(f"jax(names): give a list of names, e.g. ['{names}'], not a string.")
    names = list(names)
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"jax(names): {', '.join(dup)} given more than once.")
    used = names_in(text)
    missing = [n for n in used if n not in names]
    if missing:
        raise ValueError(
            f"The formula {text.strip()!r} uses {', '.join(map(repr, missing))}, not in "
            f"names={names}; give every name it uses ({', '.join(used)})."
        )
    expr = parse(text, {n: sp.Symbol(n) for n in used})
    from .lower import Evaluator  # JAX is imported only when a formula is evaluated

    class _Values(Evaluator):
        """The model evaluator's tree walk, with names looked up in given values."""

        def __init__(self, values):  # noqa: D107 (only what the tree walk reads)
            self.plan = _NO_PLAN
            self.values = values
            self.guard = None
            self.face_axis = None
            self.defs = {}

        def name(self, n):
            return self.values[n]

    def f(*values):
        if len(values) != len(names):
            raise TypeError(
                f"This function takes {len(names)} value(s) ({', '.join(names)}), "
                f"got {len(values)}."
            )
        import jax.numpy as jnp

        out = _Values(dict(zip(names, values, strict=True))).expr(expr)
        return jnp.asarray(out)

    f.__doc__ = f"{text.strip()} as a function of ({', '.join(names)})."
    return f


class _NoPlan:
    time_atoms = ()


_NO_PLAN = _NoPlan()
