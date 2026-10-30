# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Syntax-tree helpers for the notation reader (``notation.py``).

Formulas in the notation are formula-language text (ADR 0002) plus the operators of the
notation: ``d_a(…)``, ``int_a(…)``, ``sum_s(…)``, ``trans_s(M, X)``, ``grad_x(S)`` and
slices ``C[organ=liver, sub=vascular]``. Python's ``ast`` reads them after two textual
rewrites that keep the syntax valid Python (nothing is evaluated):

- a slice ``C[organ=liver]`` becomes the call ``__slice__(C, organ="liver")``;
- ``trans_{r1,r2}(`` becomes ``trans__r1__r2(`` (joint transitions);
- ``^`` becomes ``**``.
"""

from __future__ import annotations

import ast
import re

SLICE = "__slice__"
_NAME = r"[A-Za-z_][A-Za-z0-9_]*"


class NotationSyntaxError(ValueError):
    """A formula of the notation cannot be read (message for the user)."""


def _quote_label(lab: str) -> str:
    lab = lab.strip()
    if len(lab) >= 2 and lab[0] == lab[-1] and lab[0] in "\"'":
        return lab
    return repr(lab)


def prepare(text: str) -> str:
    """The textual rewrites above."""
    text = text.replace("^", "**")
    text = re.sub(
        r"\btrans_\{([^}]*)\}\s*\(",
        lambda m: "trans__" + "__".join(a.strip() for a in m.group(1).split(",")) + "(",
        text,
    )

    def slice_(m: re.Match) -> str:
        items = []
        for part in m.group(2).split(","):
            if "=" not in part:
                raise NotationSyntaxError(
                    f"'{m.group(0)}': a slice fixes axes by label, e.g. C[organ=liver]; every "
                    "entry needs axis=label."
                )
            ax, lab = part.split("=", 1)
            if not re.fullmatch(_NAME, ax.strip()):
                raise NotationSyntaxError(f"'{m.group(0)}': '{ax.strip()}' is not an axis name.")
            if any(i.split("=", 1)[0] == ax.strip() for i in items):
                raise NotationSyntaxError(
                    f"'{m.group(0)}': the axis '{ax.strip()}' is fixed twice; a slice takes one "
                    "label per axis."
                )
            items.append(f"{ax.strip()}={_quote_label(lab)}")
        return f"{SLICE}({m.group(1)}, {', '.join(items)})"

    return re.sub(rf"\b({_NAME})\s*\[([^\[\]]*=[^\[\]]*)\]", slice_, text)


def parse(text: str) -> ast.expr:
    src = prepare(text)
    try:
        return ast.parse(src.strip(), mode="eval").body
    except SyntaxError as e:
        raise NotationSyntaxError(f"cannot read '{text.strip()}': {e.msg}.") from None


def unparse(node: ast.AST) -> str:
    return ast.unparse(node)


def names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    return None


def slice_of(node: ast.AST) -> tuple[str, tuple[tuple[str, str], ...]] | None:
    """``__slice__(C, organ="liver")`` -> ("C", (("organ", "liver"),))."""
    if call_name(node) != SLICE:
        return None
    field = node.args[0].id
    return field, tuple((k.arg, k.value.value) for k in node.keywords)


def slice_text(field: str, where) -> str:
    def lab(v: str) -> str:
        return v if re.fullmatch(_NAME, v) else repr(v)

    return f"{field}[{', '.join(f'{a}={lab(v)}' for a, v in where)}]"


def operator_axes(name: str, prefix: str) -> tuple[str, ...] | None:
    """``d_a`` -> ("a",); ``trans__r1__r2`` -> ("r1", "r2"); otherwise None."""
    if prefix == "trans" and name.startswith("trans__"):
        return tuple(name[len("trans__") :].split("__"))
    if name.startswith(prefix + "_") and len(name) > len(prefix) + 1:
        return (name[len(prefix) + 1 :],)
    return None


def terms(node: ast.expr, sign: int = 1, split=lambda n: False) -> list[tuple[int, ast.expr]]:
    """The additive terms of an expression with their signs. A product or quotient of a
    bracketed sum is distributed when ``split(sum_node)`` is true (used for fluxes, so that
    ``(J_in - J_out)/V`` becomes two terms)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        right = -sign if isinstance(node.op, ast.Sub) else sign
        return terms(node.left, sign, split) + terms(node.right, right, split)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return terms(node.operand, -sign if isinstance(node.op, ast.USub) else sign, split)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        inner = _without_leading_minus(node)
        if inner is not None:  # -a*b is read by Python as (-a)*b
            return terms(inner, -sign, split)
        if _is_sum(node.left) and split(node.left):
            return [
                (s, ast.BinOp(t, node.op, node.right)) for s, t in terms(node.left, sign, split)
            ]
        if isinstance(node.op, ast.Mult) and _is_sum(node.right) and split(node.right):
            return [
                (s, ast.BinOp(node.left, ast.Mult(), t)) for s, t in terms(node.right, sign, split)
            ]
    return [(sign, node)]


def _without_leading_minus(node: ast.expr) -> ast.expr | None:
    """``(-a)*b/c`` -> ``a*b/c``; None if the product does not start with a minus."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return node.operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        left = _without_leading_minus(node.left)
        return None if left is None else ast.BinOp(left, node.op, node.right)
    return None


def _is_sum(node: ast.expr) -> bool:
    return (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub))) or (
        isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
    )


def signed(sign: int, node: ast.expr) -> ast.expr:
    return node if sign > 0 else ast.UnaryOp(ast.USub(), node)


def join(items: list[tuple[int, ast.expr]]) -> str:
    """Signed terms back to one formula."""
    node = None
    for s, n in items:
        if node is None:
            node = signed(s, n)
        else:
            node = ast.BinOp(node, ast.Add() if s > 0 else ast.Sub(), n)
    return unparse(node)


def is_factor(node: ast.expr, test) -> bool:
    return remove_factor(node, test) is not None


def remove_factor(node: ast.expr, test) -> ast.expr | None:
    """``node`` with one factor for which ``test(factor)`` holds taken out of the product, or
    None if there is no such factor at the top of the product (``k*B*P`` without ``B`` is
    ``k*P``; ``B/(K + B)`` without ``B`` is ``1/(K + B)``; ``B`` alone gives 1)."""
    if test(node):
        return ast.Constant(1)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        if test(node.left):
            return node.right
        if test(node.right):
            return node.left
        left = remove_factor(node.left, test)
        if left is not None:
            return _times(left, node.right)
        right = remove_factor(node.right, test)
        if right is not None:
            return _times(node.left, right)
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = remove_factor(node.left, test)
        if left is not None:
            return ast.BinOp(left, ast.Div(), node.right)
        return None
    if (  # B^2 without B is B^1
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Pow)
        and test(node.left)
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, (int, float))
        and node.right.value > 1
    ):
        n = node.right.value - 1
        return node.left if n == 1 else ast.BinOp(node.left, ast.Pow(), ast.Constant(n))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = remove_factor(node.operand, test)
        return None if inner is None else ast.UnaryOp(ast.USub(), inner)
    return None


def _times(a: ast.expr, b: ast.expr) -> ast.expr:
    if isinstance(a, ast.Constant) and a.value == 1:
        return b
    if isinstance(b, ast.Constant) and b.value == 1:
        return a
    return ast.BinOp(a, ast.Mult(), b)


def is_name(name: str):
    return lambda n: isinstance(n, ast.Name) and n.id == name
