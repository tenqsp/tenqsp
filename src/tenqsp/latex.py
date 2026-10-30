# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The model's equations as a paper shows them: ``m.to_latex()``.

The equations are those of ``m.to_text()`` (so a model built in Python and one read from
text print alike), typeset as an ``align*`` block (amsmath):

- ``d_t(B)`` as a partial derivative for a field with continuous axes, ``dP/dt`` otherwise;
  ``d_a(v*I)``, ``d_x(D*d_x(u))``, ``int_a``, ``sum_s`` and ``grad_x`` as in a paper;
- every transition term ``trans_s(M, X)`` is written out category by category: what arrives
  from the other categories minus what leaves (notation §3), so the equations of an article
  come from the model itself;
- named formulas, totals and fluxes follow under "where" (totals the library named itself,
  ``int_z_s``, are written in place), then boundaries and starting values. Doses and outputs
  are a protocol, not equations, and are left out.
"""

from __future__ import annotations

import ast
import re

from ._notation_ast import SLICE, parse
from .ir import ModelIR
from .writer import to_text

__all__ = ["to_latex"]

_GREEK = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Upsilon",
    "Phi",
    "Psi",
    "Omega",
}
_FUNCS = {
    "exp": r"\exp",
    "log": r"\log",
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "sinh": r"\sinh",
    "cosh": r"\cosh",
    "tanh": r"\tanh",
    "max": r"\max",
    "min": r"\min",
}
_CMP = {ast.Lt: "<", ast.LtE: r"\le", ast.Gt: ">", ast.GtE: r"\ge", ast.Eq: "=", ast.NotEq: r"\ne"}
_ADD, _MUL, _NEG, _POW, _ATOM = 1, 2, 3, 4, 5
_SUM = 1.5  # a sum over s: bracketed as a factor, not inside an integral (closed by dx)


def _word(s: str) -> str:
    if s in _GREEK:
        return "\\" + s
    if len(s) == 1:
        return s
    m = re.fullmatch(r"([A-Za-z]|[a-z]+)([0-9]+)", s)  # k1 -> k_1, beta2 -> beta_2
    if m and (len(m.group(1)) == 1 or m.group(1) in _GREEK):
        return _word(m.group(1)) + "_{" + m.group(2) + "}"
    return r"\mathrm{" + s.replace("_", r"\_") + "}"


def name(s: str) -> str:
    """A name in LaTeX: ``h_max`` -> ``h_{\\mathrm{max}}``, ``beta`` -> ``\\beta``."""
    head, _, rest = s.partition("_")
    if not head:
        return _word(s)
    return _word(head) + (
        ("_{" + ",".join(_word(p) for p in rest.split("_") if p) + "}") if rest else ""
    )


def _sub(base: str, index: str) -> str:
    """``base`` with subscript ``index``; a base that has a subscript already (``Q_{art}``)
    takes the index in brackets, ``Q_{art}[organ]`` (TeX refuses a double subscript)."""
    depth = 0
    for ch in base:
        depth += ch == "{"
        depth -= ch == "}"
        if ch == "_" and depth == 0:
            return rf"{base}\left[{index}\right]"
    return f"{base}_{{{index}}}"


_TEX_SPECIAL = {c: "\\" + c for c in "_%&#${}"} | {
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
    " ": r"\ ",
}


def _label(s: str) -> str:
    """A category label in roman type, TeX's special characters escaped."""
    if len(s) == 1 and s.isalnum():
        return s
    text = "".join(_TEX_SPECIAL.get(c, c) for c in s)
    return r"\text{" + text + "}" if re.search(r"[%&#$~^\\]", s) else r"\mathrm{" + text + "}"


def number(v: float | int) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int) or (float(v).is_integer() and abs(v) < 1e4):
        return str(int(v))
    v = float(v)
    if v != 0 and (abs(v) >= 1e4 or abs(v) < 1e-3):
        mant, exp = f"{v:.11e}".split("e")
        mant = mant.rstrip("0").rstrip(".")
        e = int(exp)
        return f"10^{{{e}}}" if mant == "1" else f"{mant} \\times 10^{{{e}}}"
    return f"{v:.12g}"


class _Printer:
    def __init__(self, ir: ModelIR, inline: dict[str, ast.expr]):
        self.axes = {a.name: a for a in ir.axes}
        self.inline = inline
        self.fields = {f.name for f in ir.fields}

    def __call__(self, node) -> str:
        return self.p(node)[0]

    def wrap(self, node, at_least: int) -> str:
        s, prec = self.p(node)
        return s if prec >= at_least else r"\left(" + s + r"\right)"

    def p(self, n) -> tuple[str, int]:  # noqa: PLR0911, PLR0912
        if isinstance(n, ast.Constant):
            t = number(n.value)
            return t, (_MUL if "^" in t else _ATOM)  # 10^{8} squared needs brackets
        if isinstance(n, ast.Name):
            if n.id in self.inline:
                return self.p(self.inline[n.id])
            return name(n.id), _ATOM
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return "-" + self.wrap(n.operand, _MUL), _NEG
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd):
            return self.p(n.operand)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return r"\text{not } " + self.wrap(n.operand, _ATOM), _NEG
        if isinstance(n, ast.BinOp):
            inner = _lead_neg(n)
            if inner is not None:  # (-a)*b/c is written -a b / c
                return "-" + self.wrap(inner, _MUL), _NEG
            if isinstance(n.op, (ast.Add, ast.Sub)):
                right = self.wrap(n.right, _MUL)
                op = "+" if isinstance(n.op, ast.Add) else "-"
                if right.startswith("-"):  # a + -b is a - b; a - -b is a + b
                    op, right = ("-" if op == "+" else "+"), right[1:]
                return f"{self.wrap(n.left, _ADD)} {op} {right}", _ADD
            if isinstance(n.op, ast.Mult):
                a, b = self.wrap(n.left, _MUL), self.wrap(n.right, _NEG + 1)
                sep = r" \cdot " if (b[:1].isdigit() or b.startswith("-")) else r" \, "
                return a + sep + b, _MUL
            if isinstance(n.op, ast.Div):
                return r"\frac{" + self(n.left) + "}{" + self(n.right) + "}", _POW
            if isinstance(n.op, ast.Pow):
                return self.wrap(n.left, _ATOM) + "^{" + self(n.right) + "}", _POW
        if isinstance(n, ast.Subscript):  # k[s], V[organ, sub]
            idx = n.slice.elts if isinstance(n.slice, ast.Tuple) else [n.slice]
            return _sub(self.wrap(n.value, _ATOM), ",".join(self(i) for i in idx)), _ATOM
        if isinstance(n, ast.IfExp):
            rows, node = [], n
            while isinstance(node, ast.IfExp):
                rows.append(f"{self(node.body)} & \\text{{if }} {self(node.test)}")
                node = node.orelse
            rows.append(f"{self(node)} & \\text{{otherwise}}")
            return r"\begin{cases} " + r" \\ ".join(rows) + r" \end{cases}", _ATOM
        if isinstance(n, ast.Compare):
            parts = [self(n.left)]
            for op, c in zip(n.ops, n.comparators, strict=True):
                parts += [_CMP[type(op)], self(c)]
            return " ".join(parts), _ADD
        if isinstance(n, ast.BoolOp):
            word = r" \text{ and } " if isinstance(n.op, ast.And) else r" \text{ or } "
            return word.join(self.wrap(v, _ATOM) for v in n.values), _ADD
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            return self.call(n)
        return ast.unparse(n), _ATOM

    def call(self, n: ast.Call) -> tuple[str, int]:  # noqa: PLR0911
        f, args = n.func.id, n.args
        if f in self.fields and not args and len(n.keywords) == 1:  # D(z=0): D at an end
            ax, at = n.keywords[0].arg, n.keywords[0].value
            a = self.axes.get(ax)
            if isinstance(at, ast.Name) and a is not None and at.id in ("lower", "upper"):
                pos = number(a.faces[0] if at.id == "lower" else a.faces[-1])
            else:
                pos = self(at)
            return rf"{name(f)}\big|_{{{name(ax)}={pos}}}", _ATOM
        if f == SLICE:
            labels = [_label(k.value.value) for k in n.keywords]
            return _sub(self.wrap(args[0], _ATOM), r",\,".join(labels)), _ATOM
        m = re.fullmatch(r"(d|int|sum|grad|trans)_(\w+)", f)
        if f.startswith("trans__"):
            axes = f[len("trans__") :].split("__")
            return self.trans(axes, args), _ATOM
        if m:
            op, ax = m.groups()
            x = name(ax)
            if op == "d":
                if isinstance(args[0], ast.Name):
                    return rf"\frac{{\partial {self(args[0])}}}{{\partial {x}}}", _ATOM
                return rf"\frac{{\partial}}{{\partial {x}}}\left({self(args[0])}\right)", _ATOM
            if op == "grad":
                return rf"\frac{{\partial {self(args[0])}}}{{\partial {x}}}", _ATOM
            if op == "int":
                a = self.axes.get(ax)
                lo, hi = (number(a.faces[0]), number(a.faces[-1])) if a is not None else ("", "")
                return (
                    rf"\int_{{{lo}}}^{{{hi}}} {self.wrap(args[0], _SUM)} \, \mathrm{{d}}{x}",
                    _MUL,
                )
            if op == "sum":
                return rf"\sum_{{{x}}} {self.wrap(args[0], _MUL)}", _SUM
            if op == "trans":
                return self.trans([ax], args), _ATOM
        inner = ", ".join(self(a) for a in args)
        if n.keywords:
            kws = r",\ ".join(f"\\mathrm{{{k.arg}}}={self(k.value)}" for k in n.keywords)
            inner = f"{inner};\\ {kws}" if inner else kws
        if f == "sqrt":
            return r"\sqrt{" + inner + "}", _ATOM
        if f == "abs":
            return r"\left|" + inner + r"\right|", _ATOM
        head = _FUNCS.get(f, r"\operatorname{" + f.replace("_", r"\_") + "}")
        return head + r"\left(" + inner + r"\right)", _ATOM

    def trans(self, axes: list[str], args) -> str:
        M = self(args[0])
        on = ",".join(name(a) for a in axes)
        return rf"\mathcal{{T}}^{{{M}}}_{{{on}}}\!\left[{self(args[1])}\right]"


def _lead_neg(n: ast.expr) -> ast.expr | None:
    """``(-a)*b/c`` without its leading minus (``a*b/c``), or None if it has none."""
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return n.operand
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult | ast.Div):
        left = _lead_neg(n.left)
        if left is not None:
            return ast.BinOp(left, n.op, n.right)
    return None


def _auto_integrals(ir: ModelIR) -> set[str]:
    out = set()
    for i in ir.integrals:
        base = "int_" + "_".join(i.over)
        if re.fullmatch(rf"{re.escape(base)}(_\d+)?", i.name):
            out.add(i.name)
    return out


def _split(text: str) -> tuple[str, str]:
    lhs, _, rhs = text.partition(" = ")
    return lhs.strip(), rhs.strip()


def to_latex(ir: ModelIR) -> str:
    """The model's equations, definitions, transitions written out, boundaries and starting
    values, as an amsmath ``align*`` block."""
    from .notation import _statements

    stmts = [s.text for s in _statements(to_text(ir))]
    auto = _auto_integrals(ir)
    inline: dict[str, ast.expr] = {}
    for s in stmts:
        lhs, rhs = _split(s)
        if lhs in auto:
            inline[lhs] = parse(rhs)
    P = _Printer(ir, inline)
    fields = {f.name: f for f in ir.fields}
    continuous = {a.name for a in ir.axes if a.type == "continuous"}

    eqs, defs, bounds, starts = [], [], [], []
    whole = {m.group(1) for s in stmts if (m := re.match(r"d_t\((\w+)\)", s))}
    last = None  # the field whose equation conditions "at z=…" stand under
    for s in stmts:
        word = s.split(None, 1)[0]
        at = re.fullmatch(r"at\s+(\w+)\s*=\s*([^:]+?)\s*:\s*(.*)", s)
        if at and last is not None:
            ax, pos, body = at.groups()
            where = rf"{name(last)}\big|_{{{name(ax)}={P(parse(pos))}}}"
            if body.strip() == "outflow":
                bounds.append(rf"{where} &: \text{{outflow}}")
            else:
                lhs, rhs = _split(body)
                bounds.append(
                    rf"{P(parse(lhs))} &= {P(parse(rhs))} \quad \text{{at }} "
                    rf"{name(ax)}={P(parse(pos))}"
                )
            continue
        if s.startswith("d_t("):
            last = re.match(r"d_t\((\w+)", s).group(1)
        if re.match(r"(lag|bioavailability)\s+\w+\s*=", s):
            continue  # about the doses, which latex leaves out
        if word in (
            "model",
            "axis",
            "param",
            "field",
            "compartment",
            "transitions",
            "dose",
            "observable",
            "extinction",
        ):
            continue
        if s.startswith("d_t("):
            lhs, rhs = _split(s)
            rhs = re.sub(r"\s+using sharp.*$", "", rhs)
            target = ast.parse(_prep_lhs(lhs[4:-1]), mode="eval").body
            base = target.args[0].id if isinstance(target, ast.Call) else target.id
            partial = base in fields and any(a in continuous for a in fields[base].axes)
            d = r"\partial" if partial else r"\mathrm{d}"
            added = ""
            if isinstance(target, ast.Call) and base in whole:  # terms only on this slice
                added = rf" \quad \text{{(added to the equation of }} {name(base)}\text{{)}}"
            eqs.append(rf"\frac{{{d} {P(target)}}}{{{d} t}} &= {P(parse(rhs))}{added}")
        elif word == "flux":
            head, rhs = _split(s)
            fname, _, ends = head[len("flux ") :].partition(":")
            a, _, b = ends.partition("->")
            route = rf"{P(_lhs(a))} \to {P(_lhs(b))}"
            defs.append(rf"{name(fname.strip())} &= {P(parse(rhs))} \qquad \left({route}\right)")
        elif word in ("inflow", "outflow", "zero_flux", "dirichlet", "robin"):
            bounds.append(_boundary(P, ir, s))
        elif re.match(r"^[A-Za-z_]\w*(\[[^\]]*\])?\(0\)\s*=", s):
            lhs, rhs = _split(s)
            if re.fullmatch(r"0(\.0*)?", rhs):
                continue
            target = ast.parse(_prep_lhs(lhs[:-3]), mode="eval").body
            starts.append(rf"{P(target)}(0) &= {P(parse(rhs))}")
        else:
            lhs, rhs = _split(s)
            if lhs in auto:
                continue
            defs.append(rf"{name(lhs)} &= {P(parse(rhs))}")

    trans = _transitions(P, ir)
    rows = list(eqs)
    for title, block in (
        ("where", defs),
        ("transitions, written out", trans),
        ("boundaries (other ends: nothing crosses)", bounds),
        ("starting values (anything not given starts at 0)", starts),
    ):
        if block:
            rows.append(rf"\intertext{{{title}}}" + block[0])
            rows += block[1:]
    return "\\begin{align*}\n" + " \\\\\n".join(rows) + "\n\\end{align*}\n"


def _prep_lhs(text: str) -> str:
    from ._notation_ast import prepare

    return prepare(text)


def _lhs(text: str) -> ast.expr:
    """A place (``C[organ=lung]``) as a syntax tree."""
    return ast.parse(_prep_lhs(text.strip()), mode="eval").body


def _boundary(P: _Printer, ir: ModelIR, s: str) -> str:
    head, _, value = s.partition(":")
    words = head.split()
    kind, target = words[0], words[1]
    ax, _, side = words[3].partition("=")
    a = next(x for x in ir.axes if x.name == ax)
    at = number(a.faces[0] if side == "lower" else a.faces[-1])
    where = rf"{name(target)}\big|_{{{name(ax)}={at}}}"
    extra = " ".join(words[4:])
    if kind == "zero_flux":
        return rf"{where} &: \text{{nothing crosses}}"
    if kind == "outflow":
        to = f" into {extra.split()[-1]}" if extra.startswith("to ") else ""
        return rf"{where} &: \text{{outflow{to}}}"
    value = value.strip()
    partner = ""
    if " with " in value:
        value, _, who = value.rpartition(" with ")
        partner = rf" \quad \text{{(booked in {who.strip()})}}"
    if kind == "inflow":
        return rf"{where} &: \text{{inflow }} {P(parse(value))}"
    if kind == "dirichlet":
        return rf"{where} &= {P(parse(value))}{partner}"
    return rf"{where} &: \text{{exchange }} {P(parse(value))}{partner}"


def _at(node: ast.expr, labels: dict[str, str]) -> ast.expr:
    """A formula with the index ``s`` of tables replaced by a category's label."""

    class _Fix(ast.NodeTransformer):
        def visit_Subscript(self, n):
            idx = n.slice.elts if isinstance(n.slice, ast.Tuple) else [n.slice]
            idx = [
                ast.Name(labels[i.id]) if isinstance(i, ast.Name) and i.id in labels else i
                for i in idx
            ]
            n.slice = ast.Tuple(idx) if len(idx) > 1 else idx[0]
            return n

    return _Fix().visit(node)


def _transitions(P: _Printer, ir: ModelIR) -> list[str]:
    """Each transition term written out per category: arrivals minus departures."""
    rows, seen = [], set()
    specs = {t.name: t for t in ir.transitions}
    weights = {f.name: f.weights for f in ir.fields if f.weights}
    for t in ir.terms:
        if t.kind not in ("switch", "mutate"):
            continue
        X = t.target if t.kind == "switch" else t.expr
        key = (t.transitions, X)
        if key in seen:
            continue
        seen.add(key)
        tr = specs[t.transitions]
        joint = isinstance(tr.axis, tuple)
        axes = tr.axis if joint else (tr.axis,)
        on = ",".join(name(a) for a in axes)
        Xn = parse(X)
        Xt = P(Xn)
        simple = isinstance(Xn, ast.Name)

        def at(c, Xt=Xt, simple=simple) -> str:
            lab = r",\,".join(_label(x) for x in (c if isinstance(c, tuple) else (c,)))
            return _sub(Xt, lab) if simple else rf"\left[{Xt}\right]_{{{lab}}}"

        cats = []
        for frm, to_, _ in tr.rates:
            for c in (frm, to_):
                if c not in cats:
                    cats.append(c)
        for d in cats:
            gains = [rf"{P.wrap(parse(r), _MUL)} \, {at(o)}" for o, dd, r in tr.rates if dd == d]
            losses = [P(parse(r)) for o, dd, r in tr.rates if o == d]
            if len(losses) == 1:
                losses = [P.wrap(parse(r), _MUL) for o, dd, r in tr.rates if o == d]
            parts = " + ".join(gains)
            if losses:
                out = losses[0] if len(losses) == 1 else r"\left(" + " + ".join(losses) + r"\right)"
                parts += rf" - {out} \, {at(d)}"
            lab = r",\,".join(_label(x) for x in (d if isinstance(d, tuple) else (d,)))
            M = name(t.transitions)
            head = rf"\mathcal{{T}}^{{{M}}}_{{{on}}}\!\left[{Xt}\right]_{{{lab}}}"
            rhs = parts.lstrip(" +")
            rhs = "-" + rhs[2:] if rhs.startswith("- ") else rhs
            w = weights.get(t.target) if t.kind == "switch" else None
            if w:  # rates are clearances: amount per time, spread over the category's size
                at_d = dict(zip(axes, d if isinstance(d, tuple) else (d,), strict=True))
                rhs = rf"\frac{{1}}{{{P(_at(parse(w), at_d))}}}\left({rhs}\right)"
            rows.append(f"{head} &= {rhs}")
    return rows
