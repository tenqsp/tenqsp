# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The formula functions (ADR 0005 item 11): the Mlxtran set, which covers NONMEM's built-ins.

Each function is compiled into a reaction term and compared with numpy/scipy at many points;
aliases must give the same values; derivatives (used by the Jacobian) are compared with
finite differences; tails of the normal functions must stay accurate.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.special as sc

import tenqsp as tq

U = np.linspace(0.05, 0.95, 19)  # inside every function's domain (probit, logit, acos...)

CASES = {
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "logit": lambda u: np.log(u / (1 - u)),
    "sigmoid": lambda u: 1 / (1 + np.exp(-u)),
    "normcdf": sc.ndtr,
    "probit": sc.ndtri,
    "erf": sc.erf,
    "gammaln": sc.gammaln,
    "factorial": lambda u: sc.gamma(u + 1),
    "factln": lambda u: sc.gammaln(u + 1),
    "floor": lambda u: np.floor(4 * u),
    "ceil": lambda u: np.ceil(4 * u),
}
# names other programs use: refused, with the TenQSP name in the message
ELSEWHERE = {
    "expit": "sigmoid",
    "invlogit": "sigmoid",
    "pnorm": "normcdf",
    "phi": "normcdf",
    "PHI": "normcdf",
    "norminv": "probit",
    "qnorm": "probit",
    "lgamma": "gammaln",
    "mod": "rem",
}


def rates(formula: str, u=U):
    m = tq.Model("fn")
    x = m.axis("x", 0, 1, len(u))
    F = m.field("F", axes=[x])
    m.reaction(F, formula)
    m.init(F, 1.0)
    cm = m.compile()
    return np.asarray(cm.rhs(0.0, np.asarray(u, float)))


@pytest.mark.parametrize("name", sorted(CASES))
def test_function_values(name):
    arg = "4*F" if name in ("floor", "ceil") else "F"
    np.testing.assert_allclose(rates(f"{name}({arg})"), CASES[name](U), rtol=1e-13, atol=1e-15)


def test_two_argument_functions():
    np.testing.assert_allclose(rates("atan2(F - 0.5, 0.3)"), np.arctan2(U - 0.5, 0.3), rtol=1e-14)
    # remainder with the sign of the dividend (Fortran MOD / C fmod), not Python's %
    u = np.linspace(-3, 3, 13)
    np.testing.assert_allclose(rates("rem(F, 1.25)", u), np.fmod(u, 1.25), atol=1e-15)
    assert rates("rem(F, 2)", np.array([-3.0]))[0] == -1.0


@pytest.mark.parametrize("other, name", sorted(ELSEWHERE.items()))
def test_names_from_other_programs_point_to_ours(other, name):
    args = "F, 0.4" if name == "rem" else "F"
    with pytest.raises(tq.ModelError, match=rf"'{other}' is not a TenQSP function.*{name}\("):
        rates(f"{other}({args})")


def test_notation_and_python_share_every_function_name():
    from tenqsp.expr import FUNCTIONS

    assert sorted(n for n in FUNCTIONS if not hasattr(tq, n)) == []


def test_normal_tails_stay_accurate():
    """normcdf(-30) ~ 5e-198 and probit(1e-300) ~ -37: computed without cancellation."""
    u = np.array([-30.0, -8.0, 0.0, 8.0])
    np.testing.assert_allclose(rates("normcdf(F)", u), sc.ndtr(u), rtol=1e-12)
    p = np.array([1e-300, 1e-12, 0.5, 1 - 1e-12])
    np.testing.assert_allclose(rates("probit(F)", p), sc.ndtri(p), rtol=1e-12)


@pytest.mark.parametrize(
    "name", ["sin", "acos", "logit", "normcdf", "probit", "erf", "gammaln", "factorial", "atan2"]
)
def test_derivatives_match_finite_differences(name):
    formula = "atan2(F - 0.5, 0.3)" if name == "atan2" else f"{name}(F)"
    m = tq.Model("d")
    x = m.axis("x", 0, 1, len(U))
    F = m.field("F", axes=[x])
    m.reaction(F, formula)
    m.init(F, 0.5)
    cm = m.compile()
    J = np.diag(np.asarray(cm.jacobian(0.0, U)))
    h = 1e-6
    fd = (rates(formula, U + h) - rates(formula, U - h)) / (2 * h)
    np.testing.assert_allclose(J, fd, rtol=1e-7)


def test_daily_rhythm_with_sin():
    """dA/dt = k_in (1 + a sin(2 pi t / 24)) - k A: the closed-form solution."""
    m = tq.Model("rhythm")
    A = m.compartment("A")
    m.ode(A, "1 + 0.5*sin(2*3.141592653589793*t/24) - 0.2*A")
    m.init(A, 5.0)
    t = np.linspace(0, 72, 13)
    sol = m.solve((0, 72), times=t, rtol=1e-10, atol=1e-12)
    w, k = 2 * np.pi / 24, 0.2
    # particular solution of A' + kA = 0.5 sin(wt), plus the homogeneous part
    amp = 0.5 / (k**2 + w**2)
    part = amp * (k * np.sin(w * t) - w * np.cos(w * t))
    c = 5.0 - 1 / k - part[0]
    np.testing.assert_allclose(sol["A"], 1 / k + part + c * np.exp(-k * t), rtol=1e-8)


def test_non_smooth_functions_are_noted():
    m = tq.Model("steps")
    x = m.axis("x", 0, 1, 3)
    F = m.field("F", axes=[x])
    m.reaction(F, tq.floor(F) - F)
    m.init(F, 1.0)
    notes = [i for i in m.validate(raise_on_error=False).issues if i.level == "note"]
    assert any("floor jumps" in i.message for i in notes), notes


def test_velocity_check_evaluates_the_new_functions():
    """The validator samples coordinate-only velocities on one_way axes: sin(...) < 0 there
    is caught before solving."""
    m = tq.Model("age")
    a = m.axis("a", 0, 10, 5, kind="one_way")
    I = m.field("I", axes=[a])
    m.advect(I, axis=a, velocity=tq.sin(a))
    m.init(I, 1.0)
    with pytest.raises(tq.ModelError, match="negative somewhere"):
        m.compile()


@pytest.mark.parametrize(
    "formula, message",
    [
        ("atan2(F)", "exactly 2 arguments"),
        ("rem(F, 1, 2)", "exactly 2"),
        ("sin(F, 2)", "exactly 1"),
    ],
)
def test_wrong_number_of_arguments(formula, message):
    m = tq.Model("bad")
    x = m.axis("x", 0, 1, 2)
    F = m.field("F", axes=[x])
    m.reaction(F, formula)
    with pytest.raises(tq.ModelError, match=message):
        m.compile()


def test_every_grammar_function_has_a_python_helper():
    for name in [
        "log10", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "logit",
        "sigmoid", "normcdf", "probit", "erf", "gammaln", "factorial", "factln", "floor", "ceil",
        "rem",
    ]:  # fmt: skip
        assert callable(getattr(tq, name)), name
    assert tq.atan2("y", 2).text == "atan2(y, 2)"


@pytest.mark.parametrize(
    "formula, message",
    [
        ("9**9**9", "too large"),
        ("10^400", "too large"),
        ("factorial(10**6)", "too large"),
    ],
)
def test_huge_numbers_are_refused_at_once(formula, message):
    """Exact integer arithmetic would build numbers with millions of digits (and hang)."""
    from tenqsp.expr import FormulaError, parse

    with pytest.raises(FormulaError, match=message):
        parse(formula, {})


@pytest.mark.parametrize(
    "formula, found",
    [("1/0", "division by zero"), ("A/0", "division by zero"), ("log(-1)*A", "imaginary"),
     ("sqrt(-1)", "imaginary")],
)  # fmt: skip
def test_formulas_that_are_not_finite_real_numbers_are_errors(formula, found):
    m = tq.Model("bad")
    A = m.compartment("A", init=1.0)
    m.ode(A, formula)
    errors = [str(i) for i in m.validate(raise_on_error=False).errors]
    assert any(found in e for e in errors), errors


def test_non_finite_constants_are_reported_everywhere_not_raised():
    """Transition and transfer rates, weights: reported with the other problems."""
    m = tq.Model("bad_everywhere")
    s = m.axis("s", labels=["a", "b"])
    m.param("V", {"a": 1.0, "b": 2.0}, over=s)
    B = m.field("B", axes=[s], weights="V[s]*sqrt(-1)", init=1.0)
    m.transition(B, axis=s, matrix=m.transitions("Q", axis=s, rates={"a->b": "1/0"}), of=B)
    A = m.compartment("A", init=1.0)
    m.transfer(A, to=None, rate="1/0")
    places = {i.where.split(",")[0] for i in m.validate(raise_on_error=False).errors}
    assert {"field 'B'", "transitions 'Q'", "transfer 1 (A -> elimination)"} <= places
