# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The JAX sparse tier's host work: the Jacobian is assembled once per Jacobian, not once
per factorisation, and the run's host calls are kept on one core while it runs
(``tenqsp._threads.PIN_HOST_THREAD``), which changes where the work runs and nothing else.
Oracles: the uncached matrix ``I - c J``, the scipy backend's step counts, the same run
with pinning off, and the threads' cores read back from the operating system."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sps

import tenqsp as tq
from tenqsp import _threads
from tenqsp.backends import jax_backend as jb
from tenqsp.backends import jax_sparse
from tenqsp.compile import CompiledModel
from tenqsp.examples import toy_phage_model

T = np.linspace(0, 24, 25)
LINUX = hasattr(os, "sched_setaffinity") and len(os.sched_getaffinity(0)) > 1


@pytest.fixture
def sparse(monkeypatch):
    """Every model on the sparse tier, whatever its size."""
    monkeypatch.setattr(jb, "SPARSE_FROM_STATES", 0)


@pytest.fixture
def calls(monkeypatch):
    """The calls of ``CompiledModel.assemble_sparse``, counted in ``calls["n"]``."""
    calls = {"n": 0}
    assemble = CompiledModel.assemble_sparse

    def count(self, *args, **kwargs):
        calls["n"] += 1
        return assemble(self, *args, **kwargs)

    monkeypatch.setattr(CompiledModel, "assemble_sparse", count)
    return calls


def one_core_threads() -> set[int]:
    return {int(t) for t in os.listdir("/proc/self/task") if len(os.sched_getaffinity(int(t))) == 1}


def test_a_jacobian_is_assembled_once_however_often_it_is_factorised(sparse, calls):
    cm = toy_phage_model(4, 6).compile()
    j = jb.solve(cm, (0, 24), times=T)
    assert j.stats["nlu"] > 2 * j.stats["njev"]  # many factorisations per Jacobian
    assert calls["n"] <= j.stats["njev"]
    s = cm.solve((0, 24), times=T)
    for k in ("B", "P"):
        assert np.allclose(j[k], s[k], rtol=1e-5, atol=1e-6 * np.max(np.abs(s[k])))


def test_the_kept_jacobian_gives_the_same_matrix_and_a_new_one_is_assembled(monkeypatch, calls):
    cm = toy_phage_model(4, 6).compile()
    th = np.asarray(cm.theta0)
    y = np.asarray(cm.initial_state(th))
    pieces = [np.asarray(p) for p in cm.jacobian_pieces(0.0, y, th)]
    J = cm.assemble_sparse(pieces)
    calls["n"] = 0

    def uncached(J, c):
        return (sps.eye_array(J.shape[0]) - c * J).toarray()

    a = jax_sparse._Assembled(cm, False)
    for c in (0.1, 0.2, 0.3):
        assert np.array_equal(a.matrix(None, pieces, c).toarray(), uncached(J, c))
    assert calls["n"] == 1
    changed = [p.copy() for p in pieces]
    changed[0][0] *= 1.5
    J2 = cm.assemble_sparse(changed)
    calls["n"] = 0
    assert np.array_equal(a.matrix(None, changed, 0.1).toarray(), uncached(J2, 0.1))
    assert calls["n"] == 1
    a.matrix(1, pieces, 0.1)  # one per member of a batch: both kept
    a.matrix(None, changed, 0.2), a.matrix(1, pieces, 0.2)
    assert calls["n"] == 2
    # past the budget nothing is kept, and every matrix is still right
    monkeypatch.setattr(jax_sparse._Assembled, "NNZ_BUDGET", 0)
    b = jax_sparse._Assembled(cm, False)
    calls["n"] = 0
    for c in (0.1, 0.2):
        assert np.array_equal(b.matrix(None, pieces, c).toarray(), uncached(J, c))
    assert calls["n"] == 2


def test_pinning_changes_nothing_but_where_the_work_runs(sparse, monkeypatch):
    cm = toy_phage_model(4, 6).compile()
    runs = []
    for pin in (True, False):
        monkeypatch.setattr(_threads, "PIN_HOST_THREAD", pin)
        runs.append(jb.solve(cm, (0, 24), times=T))
    for k in ("B", "P"):
        assert np.array_equal(runs[0][k], runs[1][k])
    assert runs[0].stats["nlu"] == runs[1].stats["nlu"]


@pytest.mark.skipif(not LINUX, reason="thread placement (Linux, more than one core)")
def test_after_a_run_only_tenqsps_own_thread_stays_on_one_core(sparse, monkeypatch):
    monkeypatch.setattr(_threads, "PIN_HOST_THREAD", True)
    cm = toy_phage_model(4, 6).compile()
    before = one_core_threads()
    jb.solve(cm, (0, 24), times=T)
    owner = jax_sparse._OWNER
    assert owner._core is not None and not owner._callers
    pinned = one_core_threads() - before
    assert len(pinned) <= 1  # the owner thread (none if it was pinned before this test)
    for tid in pinned:
        assert os.sched_getaffinity(tid) == {owner._core}
    # switched off: the owner thread gets its cores back at its next call
    monkeypatch.setattr(_threads, "PIN_HOST_THREAD", False)
    jb.solve(cm, (0, 24), times=T)
    assert owner._core is None and not (one_core_threads() - before)


@pytest.mark.skipif(not LINUX, reason="thread placement (Linux, more than one core)")
def test_a_batch_leaves_no_thread_on_one_core(sparse, monkeypatch):
    monkeypatch.setattr(_threads, "PIN_HOST_THREAD", True)
    m = tq.Model.from_text(
        """param g = 0.5, k = 0.2
compartment C, B
d_t(C) = -k*C
d_t(B) = g*B - C*B
C(0) = 1
B(0) = 1
""",
        name="growth_pin",
    )
    f = m.simulator(outputs=["B"], rtol=1e-9, atol=1e-12).function(
        tq.Scenario(times=T), wrt=["g", "k"], derivatives=False
    )
    jax_sparse._on_owner(lambda: 0)  # the owner thread exists and is placed
    before = one_core_threads()
    xs = jnp.asarray(np.array([[0.5, 0.2]]) * np.linspace(0.9, 1.1, 6)[:, None])
    out = jax.vmap(f)(xs)
    assert np.all(np.isfinite(np.asarray(out["B"])))
    assert not jax_sparse._OWNER._callers and not (one_core_threads() - before)
    assert len(jax_sparse.REGISTRY) == 0
