# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The native backend (ADR 0011): ``sim.function([...], backend="native")`` solves with
TenQSP's BDF compiled by numba. Oracle: the JAX backend, whose algorithm it ports: the same
steps up to round-off, so values and forward derivatives agree closely, and the step counts
match. Each model compiles its solver once (tens of seconds; cached on disk), so the tests
that build more models are marked slow."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq

pytest.importorskip("numba")

TIGHT = {"rtol": 1e-8, "atol": 1e-10}
PK = """model pk_native
param ka=1.2, CL=4, V=30
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
observable conc = centr/V
"""


@pytest.fixture(autouse=True)
def _cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("TENQSP_NATIVE_CACHE", str(tmp_path_factory.getbasetemp() / "native"))


def boluses():
    out = []
    for i, doses in enumerate([[(0.0, 100.0)], [(0.0, 50.0), (12.0, 80.0)],
                               [(2.0, 30.0), (6.0, 30.0)]]):  # fmt: skip
        reg = tq.Regimen()
        for td, amt in doses:
            reg = reg + tq.bolus("depot", amt, times=[td])
        out.append(tq.Scenario(params={"ka": 1.0 + 0.3 * i}, regimen=reg,
                               times=tuple(np.linspace(0.5, 24, 9))))  # fmt: skip
    return out


def compare(sim, scenarios, wrt, shift=None, rel=1e-8):
    fj = sim.function(scenarios, wrt=wrt)
    fn = sim.function(scenarios, wrt=wrt, backend="native")
    assert fn.backend == "native" and fj.backend == "jax"
    X = jnp.asarray(fj.x0) if shift is None else jnp.asarray(fj.x0) + shift
    out = list(fj.outputs)[0]
    yj, yn = (np.asarray(f(X)[out]) for f in (fj, fn))
    np.testing.assert_allclose(yn, yj, rtol=rel, atol=1e-12)
    assert fn.steps(X) == fj.steps(X)  # the same steps
    Jj = jax.jacfwd(lambda X_: jnp.where(fj.mask, fj(X_)[out], 0.0))(X)
    Jn = jax.jacfwd(lambda X_: jnp.where(fn.mask, fn(X_)[out], 0.0))(X)
    np.testing.assert_allclose(np.asarray(Jn), np.asarray(Jj), rtol=1e-6,
                               atol=1e-9 * float(np.max(np.abs(Jj))))  # fmt: skip
    g = jax.grad(lambda X_: jnp.sum(jnp.where(fn.mask, fn(X_)[out], 0.0)))(X)
    assert np.all(np.isfinite(np.asarray(g)))  # reverse mode transposes the sensitivities
    return fj, fn


def test_boluses_with_dose_times_and_amounts():
    """Derivatives along parameters, a dose time (the jump of the derivative where the
    boundary moves) and a dose amount."""
    sim = tq.Model.from_text(PK).simulator(outputs=["conc"], **TIGHT)
    shift = jnp.zeros((3, 4)).at[:, 2].add(0.5)
    compare(sim, boluses(), ["ka", "CL", "dose 1 time", "dose 1 amount"], shift)


def test_values_without_derivatives():
    sim = tq.Model.from_text(PK).simulator(outputs=["conc"], **TIGHT)
    fj = sim.function(boluses(), wrt=["ka"], derivatives=False)
    fn = sim.function(boluses(), wrt=["ka"], derivatives=False, backend="native")
    X = jnp.asarray(fj.x0)
    np.testing.assert_allclose(np.asarray(fn(X)["conc"]), np.asarray(fj(X)["conc"]),
                               rtol=1e-8)  # fmt: skip


def test_a_cache_directory_with_any_characters(tmp_path, monkeypatch):
    """The generated module is UTF-8, as Python reads source, whatever the system's
    encoding: on Windows a user name outside the code page used to break the write."""
    d = tmp_path / "사용자 é"
    monkeypatch.setenv("TENQSP_NATIVE_CACHE", str(d))
    # a model of its own: one built earlier in this process would not be written again
    text = PK.replace("model pk_native", "model pk_cache_path").replace(
        "d_t(depot) = -ka*depot", "d_t(depot) = -ka*depot - 0.01*depot"
    )
    sim = tq.Model.from_text(text).simulator(outputs=["conc"], **TIGHT)
    fj = sim.function(boluses(), wrt=["ka"], derivatives=False)
    fn = sim.function(boluses(), wrt=["ka"], derivatives=False, backend="native")
    X = jnp.asarray(fj.x0)
    np.testing.assert_allclose(np.asarray(fn(X)["conc"]), np.asarray(fj(X)["conc"]),
                               rtol=1e-8)  # fmt: skip
    (module,) = d.glob("*.py")
    assert "사용자 é" in module.read_text("utf-8")  # the path, as a Python literal


def test_where_native_does_not_apply():
    sim = tq.Model.from_text(PK).simulator(outputs=["conc"])
    with pytest.raises(ValueError, match="reverse derivatives"):
        sim.function(boluses(), wrt=["ka"], derivatives="reverse", backend="native")
    with pytest.raises(ValueError, match="for a list of scenarios"):
        sim.function(boluses()[0], wrt=["ka"], backend="native")
    with pytest.raises(ValueError, match="backend= is 'jax'"):
        sim.function(boluses(), wrt=["ka"], backend="numba")
    ext = tq.Model.from_text("model ext\nparam r=1, k=2\nd_t(B) = r*B - k*B\nB(0) = 10\n"
                             "extinction B below 1\nobservable y = B\n")  # fmt: skip
    scs = [tq.Scenario(params={"k": k}, times=(1, 2, 4)) for k in (1.5, 2.0)]
    with pytest.raises(ValueError, match="events"):
        ext.simulator(outputs=["y"]).function(scs, wrt=["r"], backend="native")


@pytest.mark.slow
def test_infusions_with_rates_and_durations():
    sim = tq.Model.from_text(PK.replace("pk_native", "pk_native_inf")).simulator(
        outputs=["conc"], **TIGHT
    )
    scs = [
        tq.Scenario(regimen=tq.infusion("centr", rate=r, start=s, duration=d),
                    times=tuple(np.linspace(0.5, 24, 12)))
        for r, s, d in ((10.0, 0.0, 2.0), (5.0, 1.0, 6.0), (20.0, 3.0, 0.5))
    ]  # fmt: skip
    compare(sim, scs, ["CL", "infusion 1 rate", "infusion 1 duration"])


@pytest.mark.slow
def test_model_doses_with_lag_and_bioavailability():
    text = """model lagged
param ka=1.2, CL=4, V=30, D=100, tl=0.7, F=0.8
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
dose depot at t=0, 12: D
lag depot = tl
bioavailability depot = F
observable conc = centr/V
"""
    sim = tq.Model.from_text(text).simulator(outputs=["conc"], **TIGHT)
    scs = [tq.Scenario(params={"D": d}, times=tuple(np.linspace(0.5, 24, 10)))
           for d in (100.0, 150.0)]  # fmt: skip
    compare(sim, scs, ["ka", "D", "tl", "F"])


@pytest.mark.slow
def test_a_model_with_an_axis():
    """Myelosuppression with maturation as an axis (scalar-mode kernels), and the same with
    the kernels as loops over flat buffers (the mode of large models)."""
    from tenqsp.backends.native import translate

    text = open(__file__.replace("test_native_backend.py",
                                 "notation_models/friberg.tq")).read()  # fmt: skip
    m = tq.Model.from_text(text)
    sim = m.simulator(outputs=["Circ"], **TIGHT)
    scs = [tq.Scenario(params={"slope": s_}, times=tuple(np.linspace(10, 300, 8)))
           for s_ in (0.3, 0.5)]  # fmt: skip
    names = ["mtt", "slope"]
    compare(sim, scs, names, rel=1e-7)
    old = translate.SCALAR_MAX
    translate.SCALAR_MAX = 0
    try:
        sim2 = tq.Model.from_text(text.replace("model ", "model loops_", 1)).simulator(
            outputs=["Circ"], **TIGHT
        )
        compare(sim2, scs, names, rel=1e-7)
    finally:
        translate.SCALAR_MAX = old


def test_nested_batches():
    """A batch of scenario tables (``jax.vmap`` over ``fs``, as a population fit's inner
    steps take): every input reaches the native solve with two leading axes."""
    sim = tq.Model.from_text(PK).simulator(outputs=["conc"], **TIGHT)
    fj = sim.function(boluses(), wrt=["ka", "CL"])
    fn = sim.function(boluses(), wrt=["ka", "CL"], backend="native")
    Xs = jnp.stack([jnp.asarray(fj.x0), jnp.asarray(fj.x0) * 1.1])
    yj = jax.vmap(lambda X: fj(X)["conc"])(Xs)
    yn = jax.vmap(lambda X: fn(X)["conc"])(Xs)
    np.testing.assert_allclose(np.asarray(yn), np.asarray(yj), rtol=1e-8)


def test_invalid_values_give_nan_as_in_jax():
    """A division by zero, the log or square root of a negative number and 0 to a negative
    power give inf or NaN, as in JAX, not an error that ends the batch (an optimizer's trial
    point can be anywhere): that scenario is NaN, the others are untouched."""
    text = """model invalid
param ka=1.2, CL=4, V=30, a=1, b=1, c=1
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr - 1e-6*(log(a) + sqrt(b) + c**(-1))*centr
observable conc = centr/V
"""
    sim = tq.Model.from_text(text).simulator(outputs=["conc"])
    reg = tq.bolus("depot", 100.0, times=[0.0])
    scs = [tq.Scenario(regimen=reg, times=(1.0, 2.0))] * 5
    fj = sim.function(scs, wrt=["V", "a", "b", "c"])
    fn = sim.function(scs, wrt=["V", "a", "b", "c"], backend="native")
    X = jnp.asarray(fj.x0)
    X = X.at[1, 0].set(0.0).at[2, 1].set(-1.0).at[3, 2].set(-1.0).at[4, 3].set(0.0)
    yj, yn = np.asarray(fj(X)["conc"]), np.asarray(fn(X)["conc"])
    assert np.isnan(yj[1:]).all() and np.isnan(yn[1:]).all()
    np.testing.assert_allclose(yn[0], yj[0], rtol=1e-8)
    Jn = np.asarray(jax.jacfwd(lambda X_: fn(X_)["conc"])(X))
    assert np.isfinite(Jn[0]).all()
