# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""An entry of a parameter table in a formula: ``V[organ=spleen]`` (one value) and
``K[phage, organ=spleen]`` (the spleen's value for each phage), in the notation and as
``V.at(organ="spleen")`` in Python.

Oracles: closed-form exponential decay; the same model built with the Python API (identical
IR); exact derivatives against the closed form's."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq

jax.config.update("jax_enable_x64", True)

TEXT = """model entries
axis organ categorical [liver, spleen]
axis phage categorical [p1, p2]
param V[organ] = {liver: 2, spleen: 0.5}
param K[phage, organ] = {p1: {liver: 1, spleen: 2}, p2: {liver: 3, spleen: 4}}
field C(phage)
compartment A
d_t(C) = -K[phage, organ=spleen]*C
d_t(A) = -K[phage=p1, organ=liver]*A
C(0) = 1
A(0) = 1
observable a = V[organ=spleen]*C[phage=p1]
observable b = C[phage=p2]
observable c = A
"""


def model():
    return tq.Model.from_text(TEXT)


def test_entries_read_the_table_against_the_closed_form():
    sol = model().solve((0, 1), times=[0.5, 1.0], rtol=1e-10, atol=1e-12)
    t = np.array([0.5, 1.0])
    np.testing.assert_allclose(np.asarray(sol["a"]), 0.5 * np.exp(-2 * t), rtol=1e-8)
    np.testing.assert_allclose(np.asarray(sol["b"]), np.exp(-4 * t), rtol=1e-8)
    np.testing.assert_allclose(np.asarray(sol["c"]), np.exp(-1 * t), rtol=1e-8)


def test_python_at_gives_the_same_model():
    m = tq.Model("entries")
    organ = m.axis("organ", labels=["liver", "spleen"])
    phage = m.axis("phage", labels=["p1", "p2"])
    V = m.param("V", {"liver": 2, "spleen": 0.5}, over=organ)
    K = m.param(
        "K", {"p1": {"liver": 1, "spleen": 2}, "p2": {"liver": 3, "spleen": 4}}, over=[phage, organ]
    )
    C = m.field("C", [phage], init=1)
    A = m.compartment("A", init=1)
    m.reaction(C, -K.at(organ="spleen") * C)
    m.ode(A, -K.at(phage="p1", organ="liver") * A)
    sol = m.solve((0, 1), times=[1.0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(float(np.asarray(sol["C"]).ravel()[0]), np.exp(-2), rtol=1e-8)
    m.observable("c", V.at(organ="spleen") * A)
    text = m.to_text()
    assert "d_t(C) = -K[phage, organ=spleen]*C" in text
    assert "V[organ=spleen]*A" in text
    with pytest.raises(tq.FormulaError, match="has no axis 'tissue'"):
        V.at(tissue="spleen")


def test_written_and_explained_as_in_the_notation_and_read_back():
    m = model()
    text = m.to_text()
    assert "K[phage, organ=spleen]" in text and "K[phage=p1, organ=liver]" in text
    assert tq.Model.from_text(text).to_ir() == m.to_ir()
    assert "K[phage, organ=spleen]" in m.explain() and "'spleen'" not in m.explain()
    assert m.to_latex()


def test_the_stored_form_and_its_version():
    ir = model().to_ir()
    d = json.loads(ir.to_json())
    assert d["ir_version"] == "0.12"
    assert "K[phage, 'spleen']" in json.dumps(d)
    assert type(ir).from_json(ir.to_json()) == ir
    # a model saved by the previous version still loads
    old = tq.Model.from_text("model old\nparam k = 0.5\nd_t(A) = -k*A\nA(0) = 1\n").to_ir()
    d_old = json.loads(old.to_json())
    d_old["ir_version"] = "0.10"
    assert type(ir).from_json(json.dumps(d_old)).params == old.params


@pytest.mark.parametrize(
    "written, message",
    [
        ("V[organ=spleem]", r"axis 'organ' has no category 'spleem' .*Did you mean 'spleen'"),
        ("K[organ=spleen]", r"'phage' missing. 'K' has one value per category of 'phage'"),
        ("V[phage=p1]", r"'V' has no axis 'phage'"),
        ("C[phage, organ=spleen]", r"every entry needs axis=label"),
    ],
)
def test_messages(written, message):
    text = TEXT.replace("V[organ=spleen]*C[phage=p1]", f"{written}*A")
    with pytest.raises(tq.ModelError, match=message):
        tq.Model.from_text(text)


def test_a_scalar_parameter_cannot_be_sliced():
    with pytest.raises(tq.ModelError, match="only a field or a parameter table"):
        tq.Model.from_text("model s\naxis o categorical [x, y]\nparam k = 1\nfield F(o)\n"
                           "d_t(F) = -k[o=x]*F\nF(0) = 1\n")  # fmt: skip


def test_derivatives_reach_the_entries():
    """d/dV and d/dK through the slices: only the entries read have a derivative, and it is
    the closed form's."""
    f = (
        model()
        .simulator(outputs=["a", "b", "c"], rtol=1e-10, atol=1e-12)
        .function(tq.Scenario(times=np.array([1.0])), wrt=["V", "K"])
    )
    x = jnp.asarray(
        f.x0
    )  # V[liver], V[spleen], K[p1,liver], K[p1,spleen], K[p2,liver], K[p2,spleen]
    J = np.asarray(jax.jacfwd(lambda v: jnp.concatenate([jnp.ravel(f(v)[o]) for o in "abc"]))(x))
    e = np.exp
    want = np.array(
        [
            [0, e(-2), 0, -0.5 * e(-2), 0, 0],
            [0, 0, 0, 0, 0, -e(-4)],
            [0, 0, -e(-1), 0, 0, 0],
        ]
    )
    np.testing.assert_allclose(J, want, atol=1e-8)
