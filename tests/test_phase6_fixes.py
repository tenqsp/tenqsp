# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Problems found while writing the user guide (Phase 6), each fixed in the library."""

import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq

IV = "model iv_bolus\nparam CL=4, V=30\nd_t(A) = -CL/V*A\nobservable conc = A/V\n"


@pytest.mark.timeout(60)
def test_jax_stops_when_rates_are_not_finite():
    """V = 0 divides by zero: the JAX solver used to loop for ever on a NaN step size."""
    m = tq.Model.from_text(IV)
    with pytest.raises(tq.SolverError, match="not finite numbers"):
        m.solve((0, 24), times=[0, 24], params={"V": 0}, backend="jax",
                regimen=tq.bolus("A", 100, times=[0]))  # fmt: skip


@pytest.mark.timeout(120)
@pytest.mark.parametrize("derivatives", [True, False])
def test_function_gives_nan_and_says_why(derivatives):
    m = tq.Model.from_text(IV)
    sim = m.simulator(outputs=["conc"])
    sc = tq.Scenario(regimen=tq.bolus("A", 100, times=[0]), times=[0.5, 1, 24])
    f = sim.function(sc, wrt=["CL", "V"], derivatives=derivatives)
    for x in ([float("nan"), 30.0], [4.0, 0.0]):
        assert np.all(np.isnan(np.asarray(f(jnp.array(x))["conc"])))
        assert "not finite" in f.status(jnp.array(x))
    assert f.status(f.x0) == "ok"


def test_misspelt_axis_option_is_named():
    with pytest.raises(tq.NotationError, match=r"unknown option 'knd='; did you mean 'kind='"):
        tq.Model.from_text("model a\naxis z in [0, 1] n=10 knd=one_way\n")


def test_parameter_misspelt_by_case_gets_a_suggestion():
    m = tq.Model.from_text(IV)
    with pytest.raises(KeyError, match="Did you mean 'CL'"):
        m.solve((0, 1), times=[1], params={"Cl": 4})


def test_fixed_time_pre_run_is_not_called_steady():
    m = tq.Model.from_text(IV)
    sim = m.simulator(outputs=["conc"])
    sc = tq.Scenario(regimen=tq.bolus("A", 100, times=[0]), times=[1.0])
    assert "fixed time" in repr(sim.pre_run(sc, until=5))
    assert "steady" in repr(sim.pre_run(sc, until="steady"))


def test_an_undeclared_field_is_pointed_out():
    text = (
        "model u\naxis a in [0, 1] n=5 kind=one_way\nparam h=1\n"
        "d_t(I) = -d_a(I) - h*I  using sharp\n    at a=1: outflow\n"
    )
    with pytest.raises(tq.NotationError, match=r"declare it with its axes: field I\(<axes>\)"):
        tq.Model.from_text(text)


@pytest.mark.parametrize(
    "text, message",
    [
        ("axis z in [0,1] n=4\nfield B(z)\nd_t(B) = -B\nB(0) = 1e7*normal(z, mean=0.5)",
         r"line 4: B\(0\)"),
        ("d_t(P) = -P\ndose P at t=-1: 1", r"line 2: dose P"),
        ("d_t(P) = -P\nlag P = P", r"line 2: lag P"),
        ("d_t(P) = -P\nextinction P below -1", r"line 2: extinction P"),
        ("axis s categorical [S, R]\nfield B(s)\nd_t(B) = -B\ndose B at t=0: 1e6",
         r"dose B\[s=S\] at"),
        ("param x=1\ncompartment = 2*x", r"'compartment' starts a statement"),
        ("axis z in [0,1] n=4\nfield B(z)\nfield S(z)\nd_t(S) = 0*S\nd_t(B) = -grad(S, z)*B",
         r"-d_z\(chi\*grad_z\(S\)\*B\)"),
        ("axis z in [0,1] n=4\nfield B(z)\nd_t(B) = -B\nB(0) = 1e7*point(z, at=2)",
         r"outside the axis z, which runs from 0 to 1"),
    ],
)  # fmt: skip
def test_messages_name_the_line_and_the_fix(text, message):
    with pytest.raises(tq.NotationError, match=message):
        tq.Model.from_text(text)
