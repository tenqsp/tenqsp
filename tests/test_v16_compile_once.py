# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""V16: compile once per model structure.

100 different parameter vectors, including the parameter that sets the dose amounts, must
trigger exactly one compilation of every jitted function; tracing must not grow with the grid.
"""

from __future__ import annotations

import time

import jax
import jax.monitoring
import numpy as np
import pytest

import tenqsp as tq
import tenqsp.compile
from tenqsp.examples import toy_phage_model


def dosed_transit():
    m = tq.Model("transit")
    a = m.axis("a", 0, 2.0, 5, kind="one_way")
    A, C = m.compartment("A"), m.compartment("C")
    X = m.field("X", axes=[a])
    ka, ke, D = m.params(ka=1.5, ke=0.2, D=100.0)
    m.advect(X, axis=a, velocity=1.0, scheme="erlang")
    m.inflow(X, axis=a, at="lower", flux=ka * A)
    m.outflow(X, axis=a, at="upper", to=C)
    m.ode(A, -ka * A)
    m.ode(C, -ke * C)
    m.dose(A, times=[0.0, 6.0], amounts=[D, 2 * D])
    m.observable("concentration", C / 10.0)
    return m


@pytest.mark.scipy_backend
def test_v16_100_parameter_sets_and_dose_amounts_compile_once():
    """100 solves with new parameters, dose amounts and numbers of output times. Counted two
    ways: the library's own counters (each function compiled once), and, independently, JAX's
    backend-compile events (none after the first solve)."""
    events: list[str] = []

    def listen(name, duration, **kw):
        if "backend_compile" in name:
            events.append(name)

    jax.monitoring.register_event_duration_secs_listener(listen)
    try:
        m = dosed_transit()
        rng = np.random.default_rng(16)
        finals, dosed = [], []
        first = None
        for i in range(100):
            p = {"ka": rng.uniform(0.5, 3), "ke": rng.uniform(0.05, 0.5), "D": rng.uniform(10, 1e3)}
            times = [0, 6, 12] if i == 0 else [*np.sort(rng.uniform(0, 12, rng.integers(1, 8))), 12]
            sol = m.solve((0, 12), times=times, params=p)
            if first is None:
                first = len(events)
            finals.append(sol["concentration"][-1])
            dosed.append((sol.ledger.moved["A"]["doses"][-1], 3 * p["D"]))
    finally:
        jax.monitoring.unregister_event_duration_listener(listen)
    np.testing.assert_allclose(*np.array(dosed).T, rtol=1e-15)  # the new amounts were used
    assert len(set(finals)) == 100
    assert first > 0, "the listener saw no compilation at all"
    assert len(events) == first, events[first:]
    counts = m.compile().compilations()
    used = [
        "solver rhs (with ledger)",
        "jacobian: field rows",
        "jacobian: compartment and ledger rows",
        "initial state",
        "dose amounts",
        "observables",
        "ledger",
    ]
    assert all(counts[k] == 1 for k in used), counts
    assert m.compile() is m.compile()


@pytest.mark.scipy_backend
def test_v16_without_ledger_uses_the_plain_functions_once():
    tenqsp.compile._BUILDS.clear()  # a fresh build: other tests may have solved this structure
    m = dosed_transit()
    cm = m.compile()  # with --jacobian, the V7 check compiles a Jacobian here (or not, if this
    base = cm.compilations()  # worker saw the model before): count from here
    doses = np.linspace(10, 1000, 10)
    m.solve((0, 12), times=[12], params={"D": doses[0]}, ledger=False)
    first = cm.compilations()
    for D in doses[1:]:
        m.solve((0, 12), times=[12], params={"D": D}, ledger=False)
    counts = cm.compilations()
    assert counts == first, (first, counts)  # nothing compiled after the first solve
    new = {k: counts[k] - base.get(k, 0) for k in counts}
    for k in ("rhs", "jacobian: compartment rows", "jacobian: field rows"):
        assert counts[k] >= 1 and new[k] <= 1, (k, base, counts)
    for k in ("solver rhs (with ledger)", "jacobian: compartment and ledger rows"):
        assert new[k] == 0, (k, base, counts)  # the solves never needed the ledger versions


def _trace_seconds(cm) -> float:
    y = jax.ShapeDtypeStruct((cm.n_states,), np.float64)
    best = np.inf
    for _ in range(7):  # the best of several: a shared CI runner is noisy
        f = jax.jit(cm._aug_impl)
        z = jax.ShapeDtypeStruct((cm.n_states + len(cm.ledger_entries),), np.float64)
        t0 = time.perf_counter()
        f.trace(0.0, z, cm.theta0, cm.geom)
        jax.jit(cm._rhs_impl).trace(0.0, y, cm.theta0, cm.geom)
        best = min(best, time.perf_counter() - t0)
    return best


def test_v16_trace_time_is_flat_in_grid_size():
    """121 states vs 160,801 states: the same Python work, whole-array operations only."""
    small = _trace_seconds(toy_phage_model(n_z=10, n_a=10).compile())
    large = _trace_seconds(toy_phage_model(n_z=400, n_a=400).compile())
    # 0.04 s at both sizes here; one Python operation per cell would take seconds at
    # 160,801 states, so the margin only absorbs a busy CI runner (0.04 vs 0.16 s seen)
    assert large < 2.0 * small + 0.25, (small, large)
