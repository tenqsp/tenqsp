# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``tq.formula``: TenQSP's formula language outside a model (names, a JAX function), and a
model text that ends at a ``fit`` line (TenQSP 0.7.0, for tenqsp-fit). Oracles: the same
formulas written with NumPy, and the same model read without its fitting part."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq
from tenqsp.expr import FormulaError


def test_names_and_value():
    f = tq.formula("CL_pop * (WT/70)^0.75")
    assert f.names == ("CL_pop", "WT")
    g = f.jax(["CL_pop", "WT"])
    wt = np.array([50.0, 70.0, 90.0])
    np.testing.assert_allclose(g(0.04, jnp.asarray(wt)), 0.04 * (wt / 70) ** 0.75, rtol=1e-14)


def test_order_of_names_is_the_callers_and_extra_names_are_allowed():
    g = tq.formula("a - b").jax(["b", "unused", "a"])
    assert float(g(1.0, 99.0, 5.0)) == 4.0


def test_functions_and_conditionals():
    f = tq.formula("exp(-k*t) if t > 0 else log10(c) + sigmoid(0)")
    assert f.names == ("c", "k", "t")
    t = jnp.array([-1.0, 0.5, 2.0])
    got = f.jax(["k", "t", "c"])(0.3, t, 100.0)
    want = np.where(np.asarray(t) > 0, np.exp(-0.3 * np.asarray(t)), 2.0 + 0.5)
    np.testing.assert_allclose(got, want, rtol=1e-14)


def test_derivatives_jit_and_vmap():
    g = tq.formula("a * exp(-k*t)").jax(["a", "k", "t"])
    t = jnp.linspace(0.0, 5.0, 6)
    d = jax.grad(lambda k: jnp.sum(g(2.0, k, t)))(0.4)
    np.testing.assert_allclose(d, np.sum(-2.0 * t * np.exp(-0.4 * t)), rtol=1e-12)
    ks = jnp.array([0.1, 0.2])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(lambda k: g(2.0, k, 1.0)))(ks), 2.0 * np.exp(-ks), rtol=1e-14
    )


def test_a_conditional_value_not_chosen_does_not_spoil_the_derivative():
    # log(x) is evaluated only where it is chosen ("double where", as in a model)
    g = tq.formula("log(x) if x > 0 else 0").jax(["x"])
    assert float(jax.grad(lambda x: g(x))(-1.0)) == 0.0


def test_composes_with_python_formulas():
    f = 2 * tq.exp(tq.formula("a + b"))
    assert f.names == ("a", "b")
    np.testing.assert_allclose(f.jax(["a", "b"])(0.5, 0.25), 2 * np.exp(0.75), rtol=1e-14)
    assert str(tq.formula(tq.formula("x^2"))) == "x^2"


@pytest.mark.parametrize(
    "text, words",
    [
        ("normal(z, mean=1, sd=1)", "model's axis"),
        ("grad(F, z)", "model's axis"),
        ("k[s] * 2", "parameter over an axis"),
        ("x > 1", "condition"),
        ("foo(x)", "not allowed"),
        ("CL * (", "never closed"),
        ("a, b", "not allowed in a formula"),
    ],
)
def test_refused_with_a_message(text, words):
    with pytest.raises(FormulaError, match=words):
        tq.formula(text)


def test_jax_names_checked():
    f = tq.formula("a * b")
    with pytest.raises(ValueError, match="uses 'b'"):
        f.jax(["a"])
    with pytest.raises(ValueError, match="more than once"):
        f.jax(["a", "b", "a"])
    with pytest.raises(TypeError, match="list of names"):
        f.jax("a")
    with pytest.raises(TypeError, match="takes 2 value"):
        f.jax(["a", "b"])(1.0)
    with pytest.raises(TypeError, match="formula text"):
        tq.formula(3.0)


MODEL = """model theoph
param ka=1.6, CL=0.04, ke=0.086
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - ke*centr
observable conc = centr*ke/CL
"""

FIT = """
fit theoph_pop
data "theoph.csv"
estimate ka ~ lognormal(median=1.5, sdlog=0.5)
method laplace
"""


@pytest.mark.parametrize("line", ["fit theoph_pop", "fit"])
def test_model_text_ends_at_a_fit_line(line):
    whole = tq.Model.from_text(MODEL + FIT.replace("fit theoph_pop", line))
    alone = tq.Model.from_text(MODEL)
    assert whole.to_text() == alone.to_text()


def test_load_reads_the_model_part_of_a_combined_file(tmp_path):
    p = tmp_path / "theoph_pop.tq"
    p.write_text(MODEL + FIT, encoding="utf-8")
    assert tq.load(p).to_text() == tq.Model.from_text(MODEL).to_text()


def test_fit_as_a_formula_name_is_not_a_fit_line():
    m = tq.Model.from_text(MODEL + "fit = 2*ka\nobservable twice = fit\n")
    assert "fit = 2*ka" in m.to_text()
