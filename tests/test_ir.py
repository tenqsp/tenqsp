# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""IR: JSON round trip, content hash, state counts, and validation messages."""

from __future__ import annotations

import dataclasses

import pytest
import sympy as sp
from models import toy_phage_ir

from tenqsp.expr import FormulaError, parse, shapes_used
from tenqsp.ir import (
    BoundarySpec,
    DefinitionSpec,
    DoseSpec,
    InitSpec,
    IntegralSpec,
    ModelIR,
    ParamSpec,
    TermSpec,
    TransitionSpec,
)
from tenqsp.validate import ModelError, validate


def replace(ir: ModelIR, **changes) -> ModelIR:
    return dataclasses.replace(ir, **changes)


# ---------------------------------------------------------------------------- the toy model


def test_toy_model_validates_with_only_expected_notes():
    report = validate(toy_phage_ir())
    assert report.ok, str(report)
    assert not report.warnings, str(report)
    text = str(report)
    assert "zero-flux" in text  # defaults on the dormancy axis are reported
    assert "upper end: outflow" not in text  # given explicitly in the toy model
    assert "not used" not in text  # every parameter is used (g_max, K only inside 'div')


def test_indexed_parameter_does_not_hide_the_rest_of_a_product():
    # '(1 - cost[s])' must not be simplified away: the field B inside the same product still
    # gives the definition its axes, so using it in a compartment equation is an error.
    ir = toy_phage_ir()
    bad = replace(ir, terms=(*ir.terms[:-1], TermSpec("rate", "P", "div - delta*P")))
    with pytest.raises(ModelError, match="varies over"):
        validate(bad)


def test_json_round_trip_and_hash():
    ir = toy_phage_ir()
    again = ModelIR.from_json(ir.to_json())
    assert again == ir
    assert again.content_hash() == ir.content_hash()
    changed = replace(ir, params=(ParamSpec("g_max", 2.0), *ir.params[1:]))
    assert changed.content_hash() != ir.content_hash()


def test_hash_ignores_meta_and_object_identity():
    a, b = toy_phage_ir(), toy_phage_ir()
    assert a is not b and a.content_hash() == b.content_hash()
    assert replace(a, meta={"author": "x"}).content_hash() == a.content_hash()


def test_state_count():
    sizes = toy_phage_ir(n_z=30, n_a=40).state_count()
    assert sizes == {"B": 60, "I": 1200, "P": 1}


def test_version_and_unknown_keys_rejected():
    d = toy_phage_ir().to_dict()
    with pytest.raises(ValueError, match="IR version"):
        ModelIR.from_dict({**d, "ir_version": "9.9"})
    with pytest.raises(ValueError, match="Unknown keys in model file"):
        ModelIR.from_dict({**d, "colour": "blue"})
    d2 = toy_phage_ir().to_dict()
    d2["params"][0]["typo"] = 1
    with pytest.raises(ValueError, match="Unknown keys for ParamSpec"):
        ModelIR.from_dict(d2)


def test_large_model_warns_with_advice():
    report = validate(toy_phage_ir(n_z=400, n_a=400))
    assert any("state variables" in i.message and "ADR 0003" in i.message for i in report.warnings)


# ---------------------------------------------------------------------------- one mistake each


def _with_term(ir, term):
    return replace(ir, terms=(*ir.terms, term))


BASE = toy_phage_ir()

MISTAKES = [
    ("duplicate name", replace(BASE, params=(*BASE.params, ParamSpec("B", 1.0))), "already used"),
    ("reserved name", replace(BASE, params=(*BASE.params, ParamSpec("t", 1.0))), "reserved"),
    (
        "vector length",
        replace(BASE, params=(*BASE.params[:-1], ParamSpec("cost", (0.0,), over="s"))),
        "needs 2 values",
    ),
    ("vector without index", _with_term(BASE, TermSpec("reaction", "B", "k*B")), "write 'k[s]'"),
    ("vector wrong index", _with_term(BASE, TermSpec("reaction", "B", "k[z]*B")), "write 'k[s]'"),
    (
        "categorical axis as value",
        _with_term(BASE, TermSpec("reaction", "B", "s*B")),
        "only be used as an index",
    ),
    (
        "compartment varies over z",
        _with_term(BASE, TermSpec("rate", "P", "B")),
        "is a single number",
    ),
    (
        "field term varies over a",
        _with_term(BASE, TermSpec("reaction", "B", "I")),
        "varies over ['a']",
    ),
    (
        "reaction on compartment",
        _with_term(BASE, TermSpec("reaction", "P", "1")),
        "use a 'rate' term",
    ),
    (
        "advection along categorical",
        _with_term(BASE, TermSpec("advection", "B", "1", axis="s")),
        "continuous axis",
    ),
    (
        "one_way negative velocity",
        _with_term(BASE, TermSpec("advection", "I", "tau - a", axis="a")),
        "negative somewhere",
    ),
    ("noise term", _with_term(BASE, TermSpec("noise", "B")), "planned for TenQSP v2"),
    (
        "unknown scheme",
        _with_term(BASE, TermSpec("advection", "B", "1", axis="z", scheme="weno")),
        "scheme must be",
    ),
    (
        "transition bad label",
        replace(BASE, transitions=(TransitionSpec("M", "s", (("S", "X", "mu"),)),)),
        "no category 'X'",
    ),
    (
        "self transition",
        replace(BASE, transitions=(TransitionSpec("M", "s", (("S", "S", "mu"),)),)),
        "derived automatically",
    ),
    (
        "negative rate",
        replace(BASE, transitions=(TransitionSpec("M", "s", (("S", "R", "-1e-6"),)),)),
        "must be >= 0",
    ),
    (
        "integral over absent axis",
        replace(BASE, integrals=(*BASE.integrals, IntegralSpec("bad", "B", ("a",)))),
        "does not vary over",
    ),
    (
        "circular definition",
        replace(
            BASE,
            definitions=(
                *BASE.definitions,
                DefinitionSpec("x1", "x2"),
                DefinitionSpec("x2", "x1*2"),
            ),
            terms=(*BASE.terms, TermSpec("reaction", "B", "x1")),
        ),
        "circular definition",
    ),
    (
        "dose shape into a compartment",
        replace(BASE, doses=(DoseSpec("P", (0.0,), ("1",), shape="uniform(a)"),)),
        "a compartment or one element of a field has none",
    ),
    (
        "dose lengths",
        replace(BASE, doses=(DoseSpec("P", (0.0, 1.0), ("1",)),)),
        "2 times but 1 amounts",
    ),
    (
        "init both forms",
        replace(BASE, inits=(InitSpec("I", value="0", total="1", shape="uniform(a)"),)),
        "not both",
    ),
    (
        "init bad category",
        replace(BASE, inits=(InitSpec("B", where=(("s", "Q"),), value="0"),)),
        "no category 'Q'",
    ),
    (
        "outflow without advection",
        replace(BASE, boundaries=(*BASE.boundaries, BoundarySpec("I", "z", "upper", "outflow"))),
        "no advection term",
    ),
    (
        "two conditions on one end",
        replace(BASE, boundaries=(*BASE.boundaries, BoundarySpec("I", "a", "upper", "outflow"))),
        "already has a boundary condition",
    ),
    (
        "inflow varies over its own axis",
        replace(
            BASE,
            boundaries=(BoundarySpec("I", "a", "lower", "inflow", value="I"), BASE.boundaries[1]),
        ),
        "varies over ['a']",
    ),
    (
        "positional shape args",
        replace(BASE, inits=(InitSpec("I", total="1", shape="normal(a, 0.5, 0.1)"),)),
        "only positional",
    ),
    (
        "conflicting shape args",
        replace(
            BASE, inits=(InitSpec("I", total="1", shape="normal(a, mean=0.5, sd=0.1, var=0.01)"),)
        ),
        "not a valid combination",
    ),
    (
        "impossible beta",
        replace(BASE, inits=(InitSpec("I", total="1", shape="beta(z, mean=0.5, var=0.3)"),)),
        "impossible for mean",
    ),
    (
        "unknown function",
        _with_term(BASE, TermSpec("reaction", "B", "besselj(z)*B")),
        "not allowed",
    ),
    ("typo in name", _with_term(BASE, TermSpec("reaction", "B", "g_mx*B")), "Did you mean 'g_max'"),
]


@pytest.mark.parametrize("label, ir, expected", MISTAKES, ids=[m[0] for m in MISTAKES])
def test_mistake_is_reported_in_plain_language(label, ir, expected):
    with pytest.raises(ModelError) as exc:
        validate(ir)
    assert expected in str(exc.value), f"{label}: got\n{exc.value}"


def test_all_errors_reported_at_once():
    ir = _with_term(
        _with_term(BASE, TermSpec("reaction", "B", "k*B")), TermSpec("hazard", "B", "no_such_rate")
    )
    with pytest.raises(ModelError) as exc:
        validate(ir)
    assert len(exc.value.issues and [i for i in exc.value.issues if i.level == "error"]) == 2


def test_missing_inflow_on_one_way_axis_warns():
    ir = replace(BASE, boundaries=(BASE.boundaries[1],))
    field = BASE.boundaries[1].target
    empty = replace(
        ir,
        inits=tuple(i for i in ir.inits if i.target != field),
        doses=tuple(d for d in ir.doses if d.target != field),
    )
    assert any("nothing enters" in w.message for w in validate(empty).warnings)
    # filled by a starting value or doses: a note, not a warning
    if len(empty.inits) < len(ir.inits) or len(empty.doses) < len(ir.doses):
        report = validate(ir)
        assert not any("nothing enters" in w.message for w in report.warnings)
        assert any("nothing enters" in n.message for n in report.notes)


# ---------------------------------------------------------------------------- shape functions


def _shape_args(text, **names):
    syms = {n: sp.Symbol(n, real=True) for n in ["x", *names]}
    (shape,) = shapes_used(parse(text, syms))
    return shape.shape_name, [float(a) if a.is_number else a for a in shape.args[1:]]


@pytest.mark.parametrize(
    "text, name, canonical",
    [
        ("normal(x, mean=2, sd=0.5)", "normal", [2.0, 0.5]),
        ("normal(x, mean=2, var=0.25)", "normal", [2.0, 0.5]),
        ("normal(x, mean=2, cv=0.25)", "normal", [2.0, 0.5]),
        ("lognormal(x, median=1, sdlog=0.3)", "lognormal", [0.0, 0.3]),
        ("gamma(x, mean=2, cv=0.5)", "gamma", [4.0, 2.0]),  # shape 1/cv^2, rate shape/mean
        ("gamma(x, mean=2, sd=1)", "gamma", [4.0, 2.0]),
        ("gamma(x, shape=4, scale=0.5)", "gamma", [4.0, 2.0]),
        ("exponential(x, mean=4)", "exponential", [0.25]),
        ("beta(x, mean=0.2, concentration=10)", "beta", [2.0, 8.0]),
        ("beta(x, mean=0.5, var=0.05)", "beta", [2.0, 2.0]),
        ("point(x, at=0)", "point", [0.0]),
    ],
)
def test_shape_keywords_convert_to_canonical_parameters(text, name, canonical):
    got_name, got = _shape_args(text)
    assert got_name == name
    assert got == pytest.approx(canonical)


def test_lognormal_mean_cv_matches_moments():
    _, (meanlog, sdlog) = _shape_args("lognormal(x, mean=3, cv=0.4)")
    import math

    assert math.exp(meanlog + sdlog**2 / 2) == pytest.approx(3.0)
    assert math.sqrt(math.exp(sdlog**2) - 1) == pytest.approx(0.4)


def test_shape_unknown_keyword_suggests_fix():
    with pytest.raises(FormulaError, match="Did you mean 'sd'"):
        parse("normal(x, mean=1, sdd=2)", {"x": sp.Symbol("x", real=True)})
