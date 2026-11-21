# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``f.steps(x)``: the most solver steps a run takes between two doses or events. Oracle: the
simulator's ``max_steps`` is a limit on that number, so the run at ``x`` passes with
``max_steps = f.steps(x)`` and fails with one step fewer."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq
from tenqsp.backends import jax_backend as jb

PK = """model pk
param ka=1.2, CL=4, V=30
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
observable conc = centr/V
"""
TOL = {"rtol": 1e-8, "atol": 1e-10}


def scenario(doses, ka=1.2):
    reg = tq.Regimen()
    for td, amt in doses:
        reg = reg + tq.bolus("depot", amt, times=[td])
    return tq.Scenario(params={"ka": ka}, regimen=reg, times=np.linspace(0.5, 24, 12))


ONE = [(0.0, 100.0)]
TWO = [(0.0, 100.0), (12.0, 100.0)]


def simulator(**options):
    return tq.Model.from_text(PK).simulator(outputs=["conc"], **TOL, **options)


@pytest.mark.parametrize("derivatives", ["forward", "reverse", False])
@pytest.mark.parametrize("doses", [ONE, TWO], ids=["one dose", "two doses"])
def test_max_steps_is_a_limit_on_the_count(doses, derivatives):
    """For each kind of function: its run's step sizes also follow the derivatives."""
    f = simulator().function(scenario(doses), wrt=["ka", "CL"], derivatives=derivatives)
    x = jnp.asarray(f.x0)
    n = f.steps(x)
    assert isinstance(n, int) and n > 5
    at = simulator(max_steps=n).function(scenario(doses), wrt=["ka", "CL"], derivatives=derivatives)
    assert at.status(x) == "ok"
    assert at.steps(x) == n
    below = simulator(max_steps=n - 1)
    try:  # the build's own run at x0 may already fail
        g = below.function(scenario(doses), wrt=["ka", "CL"], derivatives=derivatives)
    except jb.SolverError as e:
        assert "too many steps" in str(e)
    else:
        assert g.status(x) == "too many steps"


def test_the_count_is_per_segment():
    """Two doses 12 h apart: the most steps in one segment, not their sum."""
    f = simulator().function(scenario(TWO), wrt=["ka"])
    g = simulator().function(scenario(ONE), wrt=["ka"])
    one = g.steps(g.x0)
    assert f.steps(f.x0) < 2 * one


def test_harder_runs_take_more_steps():
    f = simulator().function(scenario(ONE), wrt=["ka", "CL"])
    x = jnp.asarray(f.x0)
    assert f.steps(x, rtol=1e-11, atol=1e-13) > f.steps(x)
    assert f.steps(x.at[0].set(1e4)) > f.steps(x)  # a fast absorption: a stiff start


def test_a_failing_run_still_reports_its_steps():
    n = simulator().function(scenario(ONE), wrt=["ka", "CL"]).steps(jnp.array([1.2, 4.0]))
    f = simulator(max_steps=n).function(scenario(ONE), wrt=["ka", "CL"])
    x = jnp.asarray(f.x0).at[0].set(1e6)  # a stiff start: more steps than at x0
    assert f.status(x) == "too many steps"
    assert f.steps(x) == n


def test_one_count_per_scenario_in_a_batch():
    scs = [scenario(ONE, ka=0.5), scenario(TWO, ka=1.2), scenario(ONE, ka=50.0)]
    fs = simulator().function(scs, wrt=["ka", "CL"], batch="vmap")
    counts = fs.steps(fs.x0)
    assert isinstance(counts, tuple) and len(counts) == 3
    singles = [simulator().function(sc, wrt=["ka", "CL"]) for sc in scs]
    assert counts == tuple(f.steps(f.x0) for f in singles)
    # stacked: one run, so one count for all
    st = simulator().function(scs, wrt=["ka", "CL"], batch="stacked").steps(fs.x0)
    assert len(st) == 3 and len(set(st)) == 1 and st[0] > 0


def test_a_new_max_steps_compiles_nothing_new():
    """The step limit is an input of the compiled programs, like the tolerances: functions
    from simulators that differ only in max_steps share every build, and the count comes
    from the function's own program."""
    m = tq.Model.from_text(PK)
    f = m.simulator(outputs=["conc"], **TOL).function(scenario(ONE), wrt=["ka", "CL"])
    x = jnp.asarray(f.x0)
    f(x), f.steps(x)
    cm = m.compile()
    builds = {k for k in cm._lazy if isinstance(k, tuple)}
    sizes = {k: v._cache_size() for k, v in cm._lazy.items() if hasattr(v, "_cache_size")}
    g = m.simulator(outputs=["conc"], max_steps=f.steps(x), **TOL).function(
        scenario(ONE), wrt=["ka", "CL"]
    )
    g(x), g.steps(x), g.status(x)
    assert {k for k in cm._lazy if isinstance(k, tuple)} == builds
    assert {k: v._cache_size() for k, v in cm._lazy.items() if hasattr(v, "_cache_size")} == sizes


@pytest.mark.parametrize("batch", ["vmap", "stacked"])
def test_max_steps_and_steps_at_call_time(batch):
    """``fs(X, max_steps=...)`` limits that call, without a new build or compile, as the
    simulator's ``max_steps`` does; ``with_steps=True`` gives ``fs.steps(X)`` from the same run,
    also inside ``jax.jit``; ``status`` takes the same limit."""
    import jax

    scs = [scenario(ONE), scenario(TWO, ka=0.8)]
    fs = simulator().function(scs, wrt=["ka", "CL"], batch=batch)
    X = jnp.asarray(fs.x0)
    n = fs.steps(X)
    out, steps = fs(X, with_steps=True)
    assert tuple(int(v) for v in steps) == n
    np.testing.assert_array_equal(out["conc"], fs(X)["conc"])
    f = jax.jit(lambda X_, m: fs(X_, max_steps=m, with_steps=True))
    _, at = f(X, max(n))
    assert tuple(int(v) for v in at) == n
    y, _ = f(X, max(n) - 1)  # one step short: a NaN run, from the same compiled function
    assert np.isnan(np.asarray(y["conc"])).any()
    assert "too many steps" in fs.status(X, max_steps=max(n) - 1)
    assert f._cache_size() == 1
    with pytest.raises(ValueError, match="max_steps must be a whole number"):
        fs(X, max_steps=0)


def test_a_list_is_checked_when_built_unless_check_false():
    """A scenario that does not run at its values raises when the list is built, naming it;
    ``check=False`` builds without running (the first call says, by NaN and ``fs.status``)."""
    sim = simulator(max_steps=3)
    scs = [scenario(ONE), scenario(TWO)]
    with pytest.raises(jb.SolverError, match="scenario 1"):
        sim.function(scs, wrt=["ka", "CL"], batch="vmap")
    fs = sim.function(scs, wrt=["ka", "CL"], batch="vmap", check=False)
    assert set(fs.status(jnp.asarray(fs.x0))) == {"too many steps"}
    with pytest.raises(ValueError, match="check= is for a list"):
        sim.function(scenario(ONE), wrt=["ka"], check=False)
