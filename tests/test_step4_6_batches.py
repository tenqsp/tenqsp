# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Step 4.6: ``jax.vmap`` over parameter sets (design note docs/design/step4_6_batches.md).
Oracles: closed forms, one call per member (a batch must give what its members give alone),
and counts of the work done (a batch must not do work no member needs)."""

from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq
from tenqsp.backends import jax_backend as jb
from tenqsp.backends import jax_sparse
from tenqsp.backends.jax_bdf import when, while_running
from tenqsp.examples import toy_phage_model

T = np.linspace(0, 30, 31)
TIGHT = dict(rtol=1e-10, atol=1e-10)

GROWTH = """param g = 0.2, k = 0.3
compartment C, B
d_t(C) = -k*C
d_t(B) = (g - k)*B
B(0) = 100
dose C at t=0: 10
dose C at t=12: 10
extinction B below 1
"""


def closed_form(g, k):
    """C: two doses decaying at k; B: 100 exp((g - k) t), zero once it falls below 1."""
    C = 10 * np.exp(-k * T) + np.where(T >= 12, 10 * np.exp(-k * (T - 12)), 0.0)
    B = 100 * np.exp((g - k) * T)
    t_ext = np.log(100) / (k - g) if k > g else np.inf
    return C, np.where(T < t_ext, B, 0.0), t_ext


@pytest.fixture
def growth():
    m = tq.Model.from_text(GROWTH, name="growth")
    sim = m.simulator(outputs=["C", "B"], **TIGHT)
    return sim.function(tq.Scenario(times=T), wrt=["g", "k"])


# k = 0.6: B dies at t = 11.5 (before the second dose); 0.3: never; 0.25: never, slowly
MEMBERS = np.array([[0.2, 0.3], [0.2, 0.6], [0.2, 0.25], [0.1, 0.6]])


def test_when_works_only_if_some_member_needs_it():
    calls = []

    def count(x):
        calls.append(np.shape(x))
        return np.asarray(x)

    def work(x):
        x = jax.pure_callback(
            count, jax.ShapeDtypeStruct(x.shape, x.dtype), x, vmap_method="broadcast_all"
        )
        return jnp.sin(x)

    def h(x):
        return when(x > 0, work, lambda x: 2 * x, x)

    xs = jnp.array([-1.0, -2.0, -3.0])
    assert np.allclose(jax.jit(jax.vmap(h))(xs), 2 * xs) and calls == []  # nobody: not run
    ys = jnp.array([-1.0, 2.0, -3.0])
    assert np.allclose(jax.vmap(h)(ys), [-2, np.sin(2), -6]) and len(calls) == 1  # once
    assert np.allclose(jax.vmap(jax.vmap(h))(jnp.stack([xs, ys])), [2 * xs, [-2, np.sin(2), -6]])

    # derivatives through it, member by member (without the counting callback)
    def d(x):
        return when(x > 0, jnp.sin, lambda x: 2 * x, x)

    assert np.allclose(jax.vmap(jax.grad(d))(ys), [2, np.cos(2), 2])
    assert np.allclose(jnp.diag(jax.jacrev(jax.vmap(d))(ys)), [2, np.cos(2), 2])


def test_while_running_keeps_done_members_and_their_record():
    def run(n):  # count to n, writing each count to a log at its own row
        def body(c):
            i = c["i"]
            return {
                "i": i + 1,
                "log": c["log"].at[jnp.where(i < n, i, 99)].set(i + 1.0, mode="drop"),
            }

        return while_running(
            lambda c: c["i"] < n, body, {"i": 0, "log": jnp.zeros(8)}, unkept="log"
        )

    ns = jnp.array([2, 5, 0, 8])
    out = jax.vmap(run)(ns)
    assert np.array_equal(out["i"], ns)  # done members stopped counting
    for k, n in enumerate(np.asarray(ns)):
        assert np.array_equal(out["log"][k], np.where(np.arange(8) < n, np.arange(1, 9), 0))


def test_a_batch_equals_its_members_with_doses_and_extinction(growth):
    xs = jnp.asarray(MEMBERS)
    out = jax.jit(jax.vmap(growth))(xs)
    for k, (g, k_) in enumerate(MEMBERS):
        C, B, _ = closed_form(g, k_)
        assert np.allclose(out["C"][k], C, rtol=1e-8)
        assert np.allclose(out["B"][k], B, rtol=1e-8, atol=1e-8)
        one = growth(xs[k])
        # batched arithmetic rounds differently, so steps may differ: equal to the tolerance
        for name in ("B", "C"):
            assert np.allclose(out[name][k], one[name], rtol=1e-8, atol=1e-8)


def test_derivatives_of_a_batch(growth):
    xs = jnp.asarray(MEMBERS)
    J = jax.vmap(jax.jacfwd(lambda x: growth(x)["B"]))(xs)
    R = jax.vmap(jax.jacrev(lambda x: growth(x)["B"]))(xs)
    for k, (g, k_) in enumerate(MEMBERS):
        _, B, t_ext = closed_form(g, k_)
        dB = np.stack([T * B, -T * B], axis=1)  # d/dg, d/dk; 0 once extinct
        near = np.abs(T - t_ext) < 1e-6
        assert np.allclose(J[k][~near], dB[~near], rtol=1e-6, atol=1e-6)
        assert np.allclose(R[k], J[k], rtol=1e-9, atol=1e-9)


def test_sparse_batches_keep_and_then_free_every_factorisation(monkeypatch, growth):
    """Every member keeps its own factorisation while it runs (before step 4.6 the oldest
    were freed while still in use, beyond the registry's room), and each is freed when it
    is replaced or its integration ends (step 4.7: no room, nothing evicted)."""
    monkeypatch.setattr(jb, "SPARSE_FROM_STATES", 0)
    m = tq.Model.from_text(GROWTH, name="growth_sparse")
    f = m.simulator(outputs=["C", "B"], **TIGHT).function(tq.Scenario(times=T), wrt=["g", "k"])
    assert jb.tier(f._cm) == "sparse"
    xs = jnp.asarray(np.tile(MEMBERS, (2, 1)) * np.linspace(0.95, 1.05, 8)[:, None])
    out = jax.vmap(f)(xs)
    nested = jax.vmap(jax.vmap(f))(xs.reshape(2, 4, 2))
    for k, (g, k_) in enumerate(np.asarray(xs)):
        C, B, _ = closed_form(g, k_)
        assert np.allclose(out["C"][k], C, rtol=1e-8)
        assert np.allclose(out["B"][k], B, rtol=1e-8, atol=1e-8)
        assert np.allclose(nested["B"][k // 4, k % 4], B, rtol=1e-8, atol=1e-8)
    assert len(jax_sparse.REGISTRY) == 0


def test_sparse_runs_in_two_threads_keep_their_factorisations(monkeypatch):
    """Step 4.7 review: a batch in one Python thread and single runs in another. Before,
    the registry's room was set by whichever ran last, and the batch's factorisations were
    evicted while in use (NaN solves, failed members)."""
    import threading

    monkeypatch.setattr(jb, "SPARSE_FROM_STATES", 0)
    m = tq.Model.from_text(GROWTH, name="growth_threads")
    f = m.simulator(outputs=["B"], **TIGHT).function(
        tq.Scenario(times=T), wrt=["g", "k"], derivatives=False
    )
    xs = jnp.asarray(np.tile(MEMBERS, (24, 1)) * np.linspace(0.95, 1.05, 96)[:, None])
    batch, one = jax.jit(jax.vmap(f)), jax.jit(f)
    batch(xs), one(xs[0])  # compiled before the race
    stop, singles = threading.Event(), []

    def single_runs():
        while not stop.is_set():
            singles.append(np.asarray(one(xs[1])["B"]))

    t = threading.Thread(target=single_runs)
    t.start()
    try:
        out = np.asarray(batch(xs)["B"])
    finally:
        stop.set()
        t.join()
    assert singles
    for k, (g, k_) in enumerate(np.asarray(xs)):
        assert np.allclose(out[k], closed_form(g, k_)[1], rtol=1e-8, atol=1e-8)
    assert len(jax_sparse.REGISTRY) == 0


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="os.fork")
def test_the_owner_thread_works_in_a_forked_child():
    """Step 4.7 review: the owner thread does not exist in a child after os.fork; a child
    used to wait for it forever."""
    import multiprocessing as mp

    jax_sparse._on_owner(lambda: 0)  # the parent's thread exists
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=lambda: q.put(jax_sparse._on_owner(lambda: 42)))
    p.start()
    p.join(30)
    alive = p.is_alive()
    if alive:
        p.kill()
    assert not alive and q.get(timeout=5) == 42


@pytest.mark.slow
def test_a_batch_costs_about_what_its_members_cost():
    """Before step 4.6, vmap copied every member's step record at every step and computed
    both sides of every branch: 7-12x the time of one call per member. Now the members run
    in lockstep (measured ~2x on CPU at this size); a generous bound for a loaded machine."""
    m = toy_phage_model(6, 8)
    names = list(m.compile().params.index)
    f = m.simulator(outputs=["P"]).function(tq.Scenario(times=np.linspace(2, 24, 12)), wrt=names)
    xs = jnp.asarray(f.x0 * np.exp(0.2 * np.random.default_rng(0).standard_normal((8, f.x0.size))))
    one, batch = jax.jit(lambda x: f(x)["P"]), jax.jit(jax.vmap(lambda x: f(x)["P"]))
    one(xs[0]).block_until_ready()
    batch(xs).block_until_ready()
    t0 = time.perf_counter()
    for x in xs:
        one(x).block_until_ready()
    loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    batch(xs).block_until_ready()
    assert time.perf_counter() - t0 < 5 * loop


# ------------------------------------------------- derivatives=False (decision 1, 2026-10-26)


@pytest.mark.parametrize("sparse", [False, True])
def test_a_forward_only_function_gives_the_same_outputs(monkeypatch, sparse):
    monkeypatch.setattr(jb, "SPARSE_FROM_STATES", 0 if sparse else 10**9)  # whatever --tier
    name = "growth_fwd_sparse" if sparse else "growth_fwd"
    sim = tq.Model.from_text(GROWTH, name=name).simulator(outputs=["C", "B"], **TIGHT)
    g = sim.function(tq.Scenario(times=T), wrt=["g", "k"], derivatives=False)
    assert jb.tier(g._cm) == ("sparse" if sparse else "dense")
    assert g.derivatives is False and g.capacity == 0 and g.names == ("g", "k")
    xs = jnp.asarray(MEMBERS)
    out = jax.jit(jax.vmap(g))(xs)
    for k, (gr, k_) in enumerate(MEMBERS):
        C, B, _ = closed_form(gr, k_)
        assert np.allclose(out["C"][k], C, rtol=1e-8)
        assert np.allclose(out["B"][k], B, rtol=1e-8, atol=1e-8)
        one = g(xs[k])
        assert np.allclose(one["B"], out["B"][k], rtol=1e-8, atol=1e-8)
        assert g.status(xs[k]) == "ok"


def test_a_forward_only_function_refuses_derivatives(growth):
    sim = tq.Model.from_text(GROWTH, name="growth").simulator(outputs=["C", "B"], **TIGHT)
    g = sim.function(tq.Scenario(times=T), wrt=["g", "k"], derivatives=False)
    x = jnp.asarray(MEMBERS[0])
    for d in (jax.grad(lambda x: g(x)["B"].sum()), jax.jacfwd(lambda x: g(x)["B"])):
        with pytest.raises(ValueError, match="derivatives=False"):
            d(x)
    with pytest.raises(ValueError, match="derivatives=False"):
        jax.vmap(jax.jacrev(lambda x: g(x)["B"]))(jnp.asarray(MEMBERS))
    with pytest.raises(TypeError, match="derivatives="):
        sim.function(tq.Scenario(times=T), wrt=["g"], derivatives="no")
    # a scenario that does not run at all is an error at once, as with derivatives
    short = tq.Model.from_text(GROWTH, name="growth").simulator(outputs=["B"], max_steps=3)
    with pytest.raises(jb.SolverError):
        short.function(tq.Scenario(times=T), wrt=["g"], derivatives=False)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="XLA's compile-time memory analysis crashed the worker on the Windows CI runner",
)
def test_a_forward_only_function_needs_far_less_memory():
    """The point of derivatives=False: no record of the run (rows x orders x states)."""
    m = toy_phage_model(6, 8)
    names = list(m.compile().params.index)
    sim = m.simulator(outputs=["P"])
    sc = tq.Scenario(times=np.linspace(2, 24, 12))
    f = sim.function(sc, wrt=names)
    g = sim.function(sc, wrt=names, derivatives=False)
    X = jnp.asarray(np.tile(f.x0, (4, 1)))

    def temp(h):
        lowered = jax.jit(jax.vmap(lambda x: h(x)["P"])).lower(X)
        return lowered.compile().memory_analysis().temp_size_in_bytes

    assert temp(g) * 5 < temp(f)
    assert np.allclose(g(X[0])["P"], f(X[0])["P"], rtol=1e-4)


def test_the_tier_threshold_is_higher_on_a_gpu(monkeypatch):
    """Design decision 2 (2026-10-26): dense up to 250 states on a CPU, 500 on a GPU."""
    from types import SimpleNamespace

    monkeypatch.setattr(jb, "SPARSE_FROM_STATES", 250)  # the defaults, whatever --tier
    monkeypatch.setattr(jb, "SPARSE_FROM_STATES_GPU", 500)
    sizes = [SimpleNamespace(n_states=n) for n in (250, 251, 500, 501)]
    monkeypatch.setattr(jb, "_platform", lambda: "cpu")
    assert [jb.tier(c) for c in sizes] == ["dense", "sparse", "sparse", "sparse"]
    monkeypatch.setattr(jb, "_platform", lambda: "gpu")
    assert [jb.tier(c) for c in sizes] == ["dense", "dense", "dense", "sparse"]


def _in_use_mb():
    import ctypes

    class _MallInfo(ctypes.Structure):
        _fields_ = [(f, ctypes.c_uint) for f in "a b c d hblkhd f g uordblks i j".split()]

    libc = ctypes.CDLL("libc.so.6")
    libc.mallinfo.restype = _MallInfo
    m = libc.mallinfo()
    return (m.uordblks + m.hblkhd) / 1e6


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc's mallinfo")
def test_sparse_runs_do_not_leak_factorisations(monkeypatch):
    """Step 4.7: with scipy 1.18 a SuperLU made in one JAX host call and freed in another
    never returns its memory (JAX's threads get a fresh Python state at each call): 57 GB for
    1,000 members at 505 states, before every kept factorisation was made and freed on one
    Python thread (jax_sparse._on_owner)."""
    monkeypatch.setattr(jb, "SPARSE_FROM_STATES", 0)
    m = toy_phage_model(10, 12)
    names = list(m.compile().params.index)
    g = m.simulator(outputs=["P"]).function(
        tq.Scenario(times=np.linspace(2, 24, 12)), wrt=names, derivatives=False
    )
    X = jnp.asarray(np.tile(g.x0, (4, 1)))
    batch = jax.jit(jax.vmap(lambda x: g(x)["P"]))
    batch(X).block_until_ready()
    before = _in_use_mb()
    batch(X).block_until_ready()
    # what stays: the factorisation each member holds at its end (~1 MB each); ~600 MB leaked
    assert _in_use_mb() - before < 50
