# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Python API: it must produce the intended IR, faithful formula text, and plain
messages for mistakes."""

from __future__ import annotations

import dataclasses
import importlib.util
import math

import numpy as np
import pytest
import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st
from models import toy_phage_ir

import tenqsp as tq
from tenqsp.examples import phage_latency_model, toy_phage_model
from tenqsp.expr import FormulaError, parse
from tenqsp.ir import InitSpec, ModelIR

# ---------------------------------------------------------------------------- whole models


def test_api_toy_model_gives_the_hand_written_ir():
    api = toy_phage_model().to_ir()
    ref = toy_phage_ir()  # written directly as IR, independently of the API
    assert api == ref
    assert api.content_hash() == ref.content_hash()


def test_prd_example_builds_and_validates():
    m = phage_latency_model()
    report = m.validate()
    assert report.ok and not report.warnings, str(report)
    ir = m.to_ir()
    # the integral used in the P equation and in the observable is stored once
    assert [i.name for i in ir.integrals] == ["int_age"]
    assert ir.observables[0].expr == "int_age"
    assert {t.kind for t in ir.terms} == {"rate", "advection", "hazard"}


def test_from_ir_and_json_round_trip():
    ir = toy_phage_model().to_ir()
    again = tq.Model.from_ir(ModelIR.from_json(ir.to_json()))
    assert again.to_ir() == ir
    assert again["I"].what == "field" and again["s"].labels == ("S", "R")


def test_summary_reports_state_count_and_warns_when_large():
    assert "1,261 state variables" in toy_phage_model().summary()
    assert "field I: 1,200 = 40 (a) x 30 (z)" in toy_phage_model().summary()
    assert "WARNING" in toy_phage_model(n_z=400, n_a=400).summary()


# ---------------------------------------------------------------------------- formula text


def small_model():
    m = tq.Model("small")
    x = m.axis("x", 0, 1, 4)
    s = m.axis("s", labels=["S", "R"])
    a, b = m.params(a=2.0, b=3.0)
    k = m.param("k", [1.0, 2.0], over=s)
    u = m.field("u", axes=[x, s])
    c = m.compartment("c")
    return m, x, s, a, b, k, u, c


@pytest.mark.parametrize(
    ("build", "text"),
    [
        (lambda a, b: a + b * 2, "a + b*2"),
        (lambda a, b: (a + b) * 2, "(a + b)*2"),
        (lambda a, b: a - (b - 1), "a - (b - 1)"),
        (lambda a, b: a / (b * 2), "a/(b*2)"),
        (lambda a, b: -(a + b), "-(a + b)"),
        (lambda a, b: (-a) ** 2, "(-a)**2"),
        (lambda a, b: a**b**2, "a**b**2"),
        (lambda a, b: (a**b) ** 2, "(a**b)**2"),
        (lambda a, b: 1e7 * a + 1.5e-9 * b, "1e7*a + 1.5e-9*b"),
        (lambda a, b: tq.max(a, 0.5) * tq.exp(-b), "max(a, 0.5)*exp(-b)"),
        (lambda a, b: np.float64(2.0) * a, "2.0*a"),
        (lambda a, b: a * "b + 1", "a*(b + 1)"),
    ],
)
def test_formula_text(build, text):
    _, _, _, a, b, *_ = small_model()
    assert str(build(a, b)) == text


_leaf = st.sampled_from(["a", "b", 0.5, 2, 1.5e-3, 3e5])


def _trees(depth: int):
    if depth == 0:
        return _leaf
    sub = _trees(depth - 1)
    return st.one_of(
        _leaf,
        st.tuples(st.sampled_from(["+", "-", "*", "/", "**"]), sub, sub),
        st.tuples(st.just("neg"), sub),
    )


def _build(tree, env):
    if isinstance(tree, tuple):
        if tree[0] == "neg":
            return -_build(tree[1], env)
        op, left, right = tree
        x, y = _build(left, env), _build(right, env)
        return {"+": x + y, "-": x - y, "*": x * y, "/": x / y, "**": x**y}[op]
    return env.get(tree, tree) if isinstance(tree, str) else tree


@settings(max_examples=300, deadline=None, derandomize=True)  # seeded
@given(_trees(4))
def test_formula_text_means_what_python_computed(tree):
    """Whatever expression is built with Python operators, its stored text, read back by the
    formula parser, gives the same number as Python's own arithmetic."""
    _, _, _, a, b, *_ = small_model()
    values = {"a": 1.7, "b": 0.6}
    try:
        expected = _build(tree, values)
    except (ZeroDivisionError, OverflowError):
        return
    try:
        formula = _build(tree, {"a": a, "b": b})
    except TypeError as e:  # a numbers-only piece such as (-0.5)**0.5 is complex in Python
        assert "complex" in str(e)
        return
    if not isinstance(formula, tq.model.Formula):
        return  # the tree had no names in it
    symbols = {n: sp.Symbol(n) for n in values}
    got = complex(parse(str(formula), symbols).subs({symbols[n]: v for n, v in values.items()}))
    if not (np.isfinite(expected) if isinstance(expected, float) else True):
        return
    assert np.isclose(got, complex(expected), rtol=1e-9, atol=1e-12), str(formula)


def test_numbers_are_stored_exactly():
    m = tq.Model("n")
    (a,) = (m.params(a=1.0),)
    for v in [1e7, 1e-9, 0.1, 1 / 3, 123456.0, 2.5e-300]:
        assert float(str(a * v).split("*")[1]) == v


# ---------------------------------------------------------------------------- vectors, axes


def test_vector_parameter_indexing():
    _, x, s, a, _, k, u, _ = small_model()
    assert str(k[s] * u) == "k[s]*u"
    assert str(k["s"]) == "k[s]"
    with pytest.raises(FormulaError, match=r"write k\[s\]"):
        k * u
    with pytest.raises(FormulaError, match=r"write k\[s\], not k\[x\]"):
        k[x]


def test_axes_in_formulas():
    _, x, s, a, *_ = small_model()
    assert str(a * (1 - x)) == str(a * (1 - x.coord)) == "a*(1 - x)"
    with pytest.raises(FormulaError, match="categorical axis"):
        s * a
    assert x.n == 4 and np.allclose(x.centers, [0.125, 0.375, 0.625, 0.875])


def test_formulas_have_no_truth_value_but_compare_into_conditions():
    _, _, _, a, b, *_ = small_model()
    with pytest.raises(TypeError, match="tq.cases"):
        bool(a)
    assert isinstance(a > b, tq.Condition) and (a > b).text == f"{a} > {b}"
    with pytest.raises(TypeError, match="no true/false value yet"):
        if a > b:
            pass


def test_param_value_shapes():
    m = tq.Model("p")
    s = m.axis("s", labels=["S", "R", "X"])
    with pytest.raises(ValueError, match="needs 3 values"):
        m.param("k", [1, 2], over=s)
    with pytest.raises(ValueError, match=r"missing \['X'\]"):
        m.param("k", {"S": 1, "R": 2}, over=s)
    with pytest.raises(ValueError, match="over="):
        m.param("k", [1, 2, 3])
    assert m.param("k", {"X": 3, "S": 1, "R": 2}, over=s)  # order follows the labels
    assert m.to_ir().params[0].value == (1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------- shapes, inits


def test_shape_functions_check_arguments_immediately():
    m = tq.Model("sh")
    z = m.axis("z", 0, 1, 10)
    s = m.axis("s", labels=["S", "R"])
    tau = m.param("tau", 0.5)
    assert str(tq.gamma(z, mean=tau, cv=0.3)) == "gamma(z, mean=tau, cv=0.3)"
    with pytest.raises(FormulaError, match="must be > 0"):
        tq.gamma(z, mean=1, cv=-1)
    with pytest.raises(FormulaError, match="Did you mean 'sd'"):
        tq.normal(z, mean=1, sdd=1)
    with pytest.raises(FormulaError, match="not a valid combination"):
        tq.normal(z, mean=1)
    with pytest.raises(FormulaError, match="categorical"):
        tq.normal(s, mean=1, sd=1)


def test_init_split_shares_the_total():
    m = tq.Model("split")
    z = m.axis("z", 0, 1, 10)
    s = m.axis("s", labels=["S", "R"])
    B = m.field("B", axes=[z, s])
    m.reaction(B, 0)
    m.init(B, total=1e7, shape=tq.uniform(z), split={s: {"S": 0.99, "R": 0.01}})
    assert m.to_ir().inits == (
        InitSpec("B", where=(("s", "S"),), total="1e7*0.99", shape="uniform(z)"),
        InitSpec("B", where=(("s", "R"),), total="1e7*0.01", shape="uniform(z)"),
    )
    m.validate()
    with pytest.raises(ValueError, match="add up to 1"):
        m.init(B, total=1, shape=tq.uniform(z), split={s: {"S": 0.9, "R": 0.2}})
    with pytest.raises(ValueError, match="not both"):
        m.init(B, 1.0, total=1, shape=tq.uniform(z))


# ---------------------------------------------------------------------------- mistakes


def test_mistakes_raise_at_once_with_plain_messages():
    m, x, s, a, b, k, u, c = small_model()
    with pytest.raises(ValueError, match="already used by a parameter"):
        m.compartment("a")
    with pytest.raises(ValueError, match="reserved"):
        m.param("t", 1.0)
    with pytest.raises(ValueError, match="is a field, but a compartment is needed"):
        m.ode(u, a)
    with pytest.raises(ValueError, match="is a compartment, but a field is needed"):
        m.hazard(c, a)
    with pytest.raises(ValueError, match="categorical axis; a continuous axis is needed"):
        m.advect(u, axis=s, velocity=1)
    with pytest.raises(ValueError, match="nothing named 'v'"):
        m.reaction("v", 1)
    with pytest.raises(ValueError, match="'erlang' or 'sharp'"):
        m.advect(u, axis=x, velocity=1, scheme="central")
    with pytest.raises(TypeError):
        m.advect(u, axis=x, velocty=1)  # unknown keyword: never silently ignored
    with pytest.raises(ValueError, match="'lower' or 'upper'"):
        m.inflow(u, axis=x, at="left", flux=1)
    with pytest.raises(ValueError, match="FROM->TO"):
        m.transitions("M", axis=s, rates={"S to R": 1})
    m.ode(c, -a * c)
    with pytest.raises(ValueError, match="already has an equation"):
        m.ode(c, b)


def test_model_level_mistakes_are_found_by_validate():
    m, x, s, a, b, k, u, c = small_model()
    m.reaction(u, -a * u)
    m.ode(c, b * u)  # forgot to integrate u
    with pytest.raises(tq.ModelError, match="varies over"):
        m.validate()
    report = m.validate(raise_on_error=False)
    assert not report.ok


def test_dose_amounts():
    m = tq.Model("d")
    P = m.compartment("P")
    m.ode(P, -P)
    m.dose(P, times=[0, 12, 24], amounts=1e8)
    assert m.to_ir().doses[0].amounts == ("1e8",) * 3
    m.validate()
    assert math.isclose(m.to_ir().doses[0].times[1], 12.0)


# ---------------------------------------------------------------------------- Phase 1 review
# Regression tests for front-end defects found by the independent Phase 1 review (2026-10-11).


def _errors(ir) -> list[str]:
    return [str(i) for i in tq.validate(ir, raise_on_error=False).errors]


def test_python_keywords_cannot_be_names():
    m = tq.Model("kw")
    with pytest.raises(ValueError, match="reserved word"):
        m.param("lambda", 0.1)
    ir = toy_phage_ir()
    ir = dataclasses.replace(ir, params=(*ir.params, tq.ir.ParamSpec("lambda", 0.1)))
    assert any("reserved word" in e for e in _errors(ir))


@pytest.mark.parametrize("order", [0, 1])
def test_overlapping_initial_conditions_are_an_error_in_either_order(order):
    m = tq.Model("overlap")
    s = m.axis("s", labels=["S", "R"])
    x = m.axis("x", 0, 1, 2)
    B = m.field("B", axes=[s, x])
    m.reaction(B, 0)
    calls = [lambda: m.init(B, 1.0), lambda: m.init(B, 5.0, where={s: "S"})]
    for c in calls[:: 1 if order == 0 else -1]:
        c()
    with pytest.raises(tq.ModelError, match="overlaps init"):
        m.validate()


def test_different_categories_do_not_overlap():
    m = tq.Model("parts")
    s = m.axis("s", labels=["S", "R"])
    x = m.axis("x", 0, 1, 2)
    B = m.field("B", axes=[s, x])
    m.reaction(B, 0)
    m.init(B, 1.0, where={s: "S"})
    m.init(B, 2.0, where={s: "R"})
    m.validate()


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda m, z, P, B: m.reaction(B, "normal(P, mean=0.5, sd=0.1)"), "continuous axis"),
        (lambda m, z, P, B: m.init(B, "normal(2*z, mean=0.5, sd=0.1)"), "axis name"),
        (lambda m, z, P, B: m.init(B, P), "parameters and coordinates only"),
        (lambda m, z, P, B: m.dose(P, times=[1.0], amounts=[2 * P]), "parameters and coord"),
    ],
)
def test_rules_enforced_by_the_validator_not_only_the_compiler(build, message):
    m = tq.Model("rules")
    z = m.axis("z", 0, 1, 4)
    P = m.compartment("P", init=1.0)
    B = m.field("B", axes=[z])
    m.ode(P, -P)
    try:
        build(m, z, P, B)
    except (FormulaError, ValueError) as e:  # caught already while building
        assert message in str(e)
        return
    with pytest.raises(tq.ModelError, match=message):
        m.validate()


@pytest.mark.parametrize(
    "text", ["exp(x, x)", "sqrt()", "max()", "min()", "max(x)", "log(x, 2)", "1e999 * x"]
)
def test_function_argument_counts_and_numbers_are_checked(text):
    with pytest.raises(FormulaError):
        parse(text, {"x": sp.Symbol("x")})


def test_validator_reports_bad_argument_count_instead_of_crashing():
    m = tq.Model("arity")
    P = m.compartment("P", init=1.0)
    m.ode(P, "exp(P, P)")
    report = m.validate(raise_on_error=False)
    assert any("exactly 1 argument" in str(i) for i in report.errors)


@pytest.mark.parametrize("bad", ["b) + (c", "a, b"])
def test_text_must_be_one_complete_formula(bad):
    m = tq.Model("splice")
    a = m.param("a", 1.0)
    with pytest.raises(FormulaError):
        a * bad
    with pytest.raises(FormulaError):
        tq.exp(bad)


def test_equal_models_have_equal_hashes():
    ir = toy_phage_ir()
    as_int = dataclasses.replace(
        ir,
        params=tuple(
            dataclasses.replace(p, value=50) if p.name == "beta" else p for p in ir.params
        ),
    )
    assert as_int == ir
    assert as_int.content_hash() == ir.content_hash()
    assert tq.Model.from_ir(as_int).to_ir().content_hash() == as_int.content_hash()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda ir: {"params": (*ir.params, tq.ir.ParamSpec("q", float("nan")))}, "finite"),
        (lambda ir: {"params": (*ir.params, tq.ir.ParamSpec("q", True))}, "finite"),
        (
            lambda ir: {
                "axes": (*ir.axes, tq.ir.AxisSpec("w", "continuous", faces=(0.0, float("inf"))))
            },
            "finite",
        ),
    ],
)
def test_non_finite_or_boolean_numbers_are_errors(change, message):
    ir = toy_phage_ir()
    assert any(message in e for e in _errors(dataclasses.replace(ir, **change(ir))))


def test_two_equations_for_one_compartment_in_an_ir_are_an_error():
    ir = toy_phage_ir()
    ir = dataclasses.replace(ir, terms=(*ir.terms, tq.ir.TermSpec("rate", "P", "-P")))
    assert any("more than one equation" in e for e in _errors(ir))


def test_smaller_api_checks():
    m = tq.Model("small")
    with pytest.raises(ValueError, match="no spacing"):
        m.axis("s", labels=["S", "R"], spacing="geometric")
    s = m.axis("s", labels=["S", "R"], units="strain")
    assert tq.Model.from_ir(m.to_ir()).to_ir().axes[0].units == "strain"
    mu = m.param("mu", 1e-6)
    with pytest.raises(ValueError, match="as a dict"):
        m.transitions("M", axis=s, rates=[("S->R", mu)])
    x = m.axis("x", 0, 1, 2)
    B = m.field("B", axes=[s, x])
    with pytest.raises(ValueError, match="add up to 1"):
        m.init(
            B, total=1.0, shape=tq.uniform(x, lower=0, upper=1), split={s: {"S": 0.5, "R": 0.49999}}
        )
    with pytest.raises(TypeError, match="tq.eq"):
        B == mu  # noqa: B015


def test_formulas_survive_an_autoreload():
    """IPython autoreload (on in Positron's console) copies a reloaded class's attributes onto
    the old class. With __slots__ the old objects then failed: "descriptor 'text' for 'Formula'
    objects doesn't apply to a 'Formula' object"."""
    from tenqsp import model as model_module

    m = tq.Model("reload")
    k = m.param("k", 0.5)
    P = m.compartment("P")
    old_formula = -k * P
    # a second copy of model.py, as after an edit, without touching the loaded module
    spec = importlib.util.spec_from_file_location("tenqsp._model_copy", model_module.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    skip = ("__dict__", "__weakref__", "__doc__", "__module__")
    saved = []
    try:
        for name in ("Formula", "Name"):
            old, new = getattr(model_module, name), getattr(fresh, name)
            saved.append((old, dict(old.__dict__)))
            for attr, value in new.__dict__.items():
                if attr not in skip:
                    setattr(old, attr, value)  # what autoreload's update_class does
        assert str(old_formula) == "-k*P"
        assert k.name == "k" and P.what == "compartment"
    finally:
        for old, attrs in saved:
            for attr, value in attrs.items():
                if attr not in skip:
                    setattr(old, attr, value)
    m.ode(P, old_formula)
    assert m.solve((0, 1), times=[1])["P"][-1] == 0.0


def test_a_text_condition_in_cases_must_be_one_condition():
    """It is kept whole by parentheses; text that would splice into the formula is refused."""
    with pytest.raises(TypeError, match="not one condition"):
        tq.cases(("a > 1 else 99 if b > 1", 1.0), otherwise=2.0)
    with pytest.raises(TypeError, match="not one condition"):
        tq.cases(("k1 + 2", 1.0), otherwise=2.0)
    f = tq.cases(("a > 1 or b < 2", 1.0), ("not a > 3", 3.0), otherwise=2.0)
    assert str(f) == "1.0 if (a > 1 or b < 2) else 3.0 if (not a > 3) else 2.0"


def test_extinction_names_from_labels_and_the_noise_floor_flag():
    m = tq.Model("labels")
    s = m.axis("s", labels=["wild type", "R-1"])
    B = m.field("B", axes=[s], init=1.0)
    m.hazard(B, 0.1)
    m.extinction(B.at(s="R-1"), threshold=1.0)
    m.extinction(B.at(s="wild type"), threshold=1.0)
    assert [e.name for e in m.to_ir().extinctions] == ["extinct_B_s_R_1", "extinct_B_s_wild_type"]
    with pytest.raises(TypeError, match="True or False"):
        m.extinction(B, threshold=1.0, name="whole", noise_floor="no")


def test_grad_in_a_transfer_rate_gets_the_gradient_message():
    m = tq.Model("grad_transfer")
    x = m.axis("x", 0, 1, 4)
    S = m.field("S", axes=[x], init=1.0)
    A = m.compartment("A", init=1.0)
    m.transfer(A, to=None, rate=tq.grad(S, x))
    errors = " ".join(str(i) for i in m.validate(raise_on_error=False).errors)
    assert "can only be used in a velocity along" in errors
