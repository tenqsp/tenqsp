# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Step 4.1a: parameters declared without a value, and builds shared by models with the same
structure (design decisions 2026-10-24)."""

from __future__ import annotations

import json

import numpy as np
import pytest

import tenqsp as tq
from tenqsp.ir import IR_VERSION, ModelIR
from tenqsp.validate import ModelError

DECLARED = """param ka, CL
param V = 30
compartment depot, centr
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
observable conc = centr/V
dose depot at t=0: 100
"""


def bateman(t, dose, ka, ke, V, t_dose=0.0):
    """One oral dose into a one-compartment model (closed form)."""
    s = np.clip(np.asarray(t) - t_dose, 0, None)
    return dose * ka / (V * (ka - ke)) * (np.exp(-ke * s) - np.exp(-ka * s))


def test_text_and_python_declare_the_same_model():
    text = tq.Model.from_text(DECLARED, name="pk").to_ir()
    m = tq.Model("pk")
    m.params("ka", "CL", V=30)
    depot, centr = m.compartment("depot"), m.compartment("centr")
    m.ode(depot, "-ka*depot")
    m.ode(centr, "ka*depot - CL/V*centr")
    m.dose(depot, times=[0], amounts="100")
    m.observable("conc", "centr/V")
    assert text.content_hash() == m.to_ir().content_hash()
    assert [p.value for p in text.params] == [None, None, 30]


def test_a_declared_parameter_survives_to_text_and_back():
    m = tq.Model.from_text(DECLARED, name="pk")
    text = m.to_text()
    assert "param ka\n" in text and "param CL\n" in text
    assert tq.Model.from_text(text, name="pk").to_ir().content_hash() == m.to_ir().content_hash()


def test_values_given_at_solve_time_match_the_closed_form():
    m = tq.Model.from_text(DECLARED, name="pk")
    t = np.linspace(0, 24, 49)
    sol = m.solve((0, 24), times=t, params={"ka": 1.2, "CL": 4}, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sol["conc"], bateman(t, 100, 1.2, 4 / 30, 30), rtol=1e-7, atol=1e-10)


def test_a_missing_value_is_an_error_that_names_it():
    m = tq.Model.from_text(DECLARED, name="pk")
    with pytest.raises(tq.MissingValues, match=r"No value for ka, CL") as e:
        m.solve((0, 1))
    assert e.value.names == ["ka", "CL"]
    with pytest.raises(tq.MissingValues, match=r"No value for CL:"):
        m.solve((0, 1), params={"ka": 1})
    with pytest.raises(ValueError, match="finite"):
        m.solve((0, 1), params={"ka": float("nan"), "CL": 4})


def test_per_category_values_missing_in_part_or_whole():
    text = "axis s categorical [S, R]\nparam k[s]\nfield B(s)\nd_t(B) = -k[s]*B\nB(0) = 1\n"
    m = tq.Model.from_text(text, name="c")
    assert "param k[s]\n" in m.to_text()
    with pytest.raises(tq.MissingValues, match=r"No value for k\[s\]"):
        m.solve((0, 1))
    with pytest.raises(tq.MissingValues, match=r"No value for k\[R\]"):
        m.solve((0, 1), params={"k": {"S": 1}})
    sol = m.solve((0, 1), times=[1.0], params={"k": {"S": 1, "R": 2}}, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sol["B"][0], np.exp([-1.0, -2.0]), rtol=1e-8)


def test_values_given_at_solve_time_are_checked():
    """Checks that need numbers (volumes > 0, one_way velocities >= 0) run on the values a
    solve uses, since the model had none to check."""
    vol = tq.Model.from_text("param V, k=1\ncompartment A volume=V\nd_t(A) = -k*A\nA(0)=1\n")
    with pytest.raises(ModelError, match="must be > 0"):
        vol.solve((0, 1), params={"V": -2})
    vol.solve((0, 1), params={"V": 2})
    age = tq.Model.from_text(
        "axis a in [0, 1] n=10 kind=one_way\nparam v\nfield N(a)\nd_t(N) = -d_a(v*N)\n"
        "at a=1: outflow\nN(0) = 1\n"
    )
    with pytest.raises(ModelError, match="one_way"):
        age.solve((0, 1), params={"v": -1})
    age.solve((0, 1), params={"v": 1})


@pytest.mark.scipy_backend
def test_changing_values_never_rebuilds_even_across_model_objects():
    a = tq.Model.from_text(DECLARED, name="pk")
    for ka in np.linspace(0.5, 2, 20):
        a.solve((0, 24), times=[24], params={"ka": ka, "CL": 4})
    assert a.compile().compilations()["solver rhs (with ledger)"] == 1
    # another model object, values written in the file: same structure, same build
    b = tq.Model.from_text(DECLARED.replace("param ka, CL", "param ka=1, CL=3"), name="pk")
    assert b.compile() is not a.compile() and b.compile()._aug is a.compile()._aug
    sol = b.solve((0, 24), times=[24.0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sol["conc"], bateman([24.0], 100, 1, 3 / 30, 30), rtol=1e-7)


def test_other_dose_times_share_the_build_and_are_used():
    a = tq.Model.from_text(DECLARED, name="pk")
    b = tq.Model.from_text(DECLARED.replace("at t=0:", "at t=5:"), name="pk")
    assert b.compile()._aug is a.compile()._aug
    t = np.linspace(0, 24, 25)
    sol = b.solve((0, 24), times=t, params={"ka": 1.2, "CL": 4}, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        sol["conc"], bateman(t, 100, 1.2, 4 / 30, 30, t_dose=5), rtol=1e-7, atol=1e-10
    )


def test_a_model_sharing_a_build_is_still_validated():
    ok = "param V = 2, k=1\ncompartment A volume=V\nd_t(A) = -k*A\nA(0)=1\n"
    tq.Model.from_text(ok, name="v").compile()
    with pytest.raises(ModelError, match="must be > 0"):
        tq.Model.from_text(ok.replace("V = 2", "V = -2"), name="v").compile()


def test_ir_0_7_values_read_and_hash_structure_apart_from_values():
    ir = tq.Model.from_text(DECLARED, name="pk").to_ir()
    assert ir.ir_version == IR_VERSION
    back = ModelIR.from_json(ir.to_json())
    assert back == ir and back.params[0].value is None
    # a 0.6 file (every parameter has a value) still reads
    d = json.loads(ir.to_json())
    d["ir_version"] = "0.6"
    d["params"][0]["value"], d["params"][1]["value"] = 1.2, 4.0
    old = ModelIR.from_dict(d)
    assert old.params[0].value == 1.2
    # same structure, other values and dose times: one structure hash, two content hashes
    assert old.structure_hash() == ir.structure_hash() and old.content_hash() != ir.content_hash()
    moved = tq.Model.from_text(DECLARED.replace("at t=0:", "at t=5:"), name="pk").to_ir()
    assert moved.structure_hash() == ir.structure_hash()
    two = tq.Model.from_text(DECLARED.replace("at t=0:", "at t=0, 5:"), name="pk").to_ir()
    assert two.structure_hash() != ir.structure_hash()  # the number of doses is structural


def test_explain_lists_parameters_without_a_value():
    text = tq.Model.from_text(DECLARED, name="pk").explain()
    assert "Parameters without a value in the model" in text and "ka, CL" in text


def test_summary_and_diagnostics_with_parameters_without_a_value():
    m = tq.Model.from_text(DECLARED, name="pk")
    text = m.compile().summary()
    assert "timed with 1 in their place" in text and "LU factors" in text
    with pytest.raises(tq.MissingValues, match=r"pass a solution"):
        m.diagnostics()
    sol = m.solve((0, 24), params={"ka": 1.2, "CL": 4})
    m.diagnostics(sol)  # values from the solution
