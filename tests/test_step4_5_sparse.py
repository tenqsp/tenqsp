# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Step 4.5: the JAX backend's sparse tier (SuperLU on the host; design note
docs/design/step4_5_sparse_tier.md). Oracles: the scipy backend (the same matrix and
factorisation, so the same steps: V9), the dense tier (the same derivatives to round-off),
finite differences, and forward = reverse."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq
from tenqsp.backends import jax_backend as jb
from tenqsp.backends import jax_replay, jax_sparse
from tenqsp.examples import toy_phage_model

T = np.linspace(0, 24, 25)


def erlang(m: tq.Model) -> tq.Model:
    ir = m.to_ir()
    terms = tuple(
        dataclasses.replace(t, scheme="erlang") if t.kind == "advection" else t for t in ir.terms
    )
    return tq.Model.from_ir(dataclasses.replace(ir, terms=terms))


@pytest.fixture
def sparse(monkeypatch):
    """Every model on the sparse tier, whatever its size."""
    monkeypatch.setattr(jb, "SPARSE_FROM_STATES", 0)


def test_the_tier_follows_the_size(monkeypatch):
    monkeypatch.setattr(jb, "SPARSE_FROM_STATES", 100)  # the rule, whatever the threshold
    small, big = toy_phage_model(4, 6).compile(), toy_phage_model(24, 38).compile()
    assert small.n_states <= jb.SPARSE_FROM_STATES < big.n_states
    assert jb.tier(small) == "dense" and jb.tier(big) == "sparse"


def test_above_a_thousand_states_on_jax_same_steps_as_scipy():
    cm = erlang(toy_phage_model(40, 60)).compile()
    assert cm.n_states > 2000 and jb.tier(cm) == "sparse"
    s = cm.solve((0, 24), times=T)
    j = jb.solve(cm, (0, 24), times=T)
    # the same matrices factorised the same way: the same Jacobians and LUs as scipy
    assert (j.stats["nlu"], j.stats["njev"]) == (s.stats["nlu"], s.stats["njev"])
    for k in ("B", "I", "P"):
        assert np.allclose(j[k], s[k], rtol=1e-10, atol=1e-10 * np.max(np.abs(s[k])))
    assert max(j.ledger.check().values()) < 1e-9


def test_doses_events_extinction_and_the_ledger_like_scipy(sparse):
    m = tq.Model.from_text(
        """param k = 0.5, kin = 1
compartment C, B
d_t(C) = -k*C
d_t(B) = (-0.8 if C > 2 else 0.3)*B
B(0) = 100
dose C at t=0: 10
dose C at t=12: 10
extinction B below 1
""",
        name="switches",
    )
    cm = m.compile()
    assert jb.tier(cm) == "sparse"
    kw = dict(times=np.linspace(0, 30, 31), rtol=1e-9, atol=1e-9)
    j, s = jb.solve(cm, (0, 30), **kw), cm.solve((0, 30), **kw)
    assert [e[1] for e in j.extinctions] == [e[1] for e in s.extinctions]
    assert np.allclose([e[0] for e in j.extinctions], [e[0] for e in s.extinctions], rtol=1e-8)
    for k in ("C", "B"):
        assert np.allclose(j[k], s[k], rtol=1e-7, atol=1e-9)
    assert max(j.ledger.check().values()) < 1e-9


def _derivatives(cm, rec, direction, w):
    th = jnp.asarray(rec.run.th)
    _, G = jax.jvp(lambda x: jax_replay.replay(cm, rec, x), (th,), (direction,))
    g = jax.grad(lambda x: jnp.sum(jax_replay.replay(cm, rec, x) * w))(th)
    return np.asarray(G), np.asarray(g)


def test_derivatives_equal_the_dense_tiers(monkeypatch):
    m = erlang(toy_phage_model(6, 8))
    times = np.array([6.0, 24.0])
    out = {}
    for threshold in (10**6, 0):
        monkeypatch.setattr(jb, "SPARSE_FROM_STATES", threshold)
        cm = m.compile()
        rec = jax_replay.record(cm, (0, 24), times=times, rtol=1e-9, atol=1e-9)
        p = cm.params.size
        rng = np.random.default_rng(5)
        d = np.zeros_like(rec.run.th)
        d[:p] = rec.run.th[:p] * rng.uniform(-1, 1, p)
        w = rng.standard_normal(rec.ys.shape)
        out[jb.tier(cm)] = (_derivatives(cm, rec, jnp.asarray(d), w), d, w, rec)
    (G_d, g_d), d, w, _ = out["dense"]
    (G_s, g_s), _, _, rec = out["sparse"]
    assert np.allclose(G_s, G_d, rtol=1e-10, atol=1e-12 * np.max(np.abs(G_d)))
    assert np.allclose(g_s, g_d, rtol=1e-10, atol=1e-12 * np.max(np.abs(g_d)))
    # forward = reverse on the sparse tier (the transposed host solve)
    assert np.isclose(g_s @ d, np.sum(G_s * w), rtol=1e-10)
    # and finite differences of whole solves
    cm = m.compile()
    h = 1e-5
    ys = [
        jax_replay.record(
            cm, (0, 24), times=[6.0, 24.0], theta=rec.run.th + k * h * d, rtol=1e-9, atol=1e-9
        ).ys
        for k in (1, -1)
    ]
    fd = (ys[0] - ys[1]) / (2 * h)
    assert np.max(np.abs(G_s - fd)) < 1e-5 * np.max(np.abs(fd))


def test_function_jacfwd_and_vmap_on_the_sparse_tier(sparse):
    m = tq.Model.from_text(
        "param ka = 1.2, CL = 4, V = 30\ncompartment depot, centr\n"
        "d_t(depot) = -ka*depot\nd_t(centr) = ka*depot - CL/V*centr\n"
        "observable conc = centr/V\n"
    )
    t = np.array([1, 4, 12.0])
    sim = m.simulator(outputs=["conc"], rtol=1e-10, atol=1e-12)
    sc = tq.Scenario(regimen=tq.bolus("depot", 100, times=[0]), times=t)
    f = sim.function(sc, wrt=["ka", "CL", "V"])
    assert jb.tier(f._cm) == "sparse"

    def conc(x):
        ka, CL, V = x
        k = CL / V
        return 100 * ka / (V * (ka - k)) * (jnp.exp(-k * t) - jnp.exp(-ka * t))

    J = np.asarray(jax.jacfwd(lambda x: f(x)["conc"])(f.x0))
    assert np.allclose(J, jax.jacfwd(conc)(jnp.asarray(f.x0)), rtol=1e-6, atol=1e-9)
    xs = jnp.asarray(f.x0) * jnp.array([[1.0, 1.0, 1.0], [1.2, 0.8, 1.1]])
    V = np.asarray(jax.vmap(lambda x: f(x)["conc"])(xs))
    assert np.allclose(V, np.stack([conc(x) for x in xs]), rtol=1e-7)
    # host factorisations are freed as they go out of use, the last when each run ends
    assert len(jax_sparse.REGISTRY) == 0


def test_less_record_room_on_large_models():
    # design decision 2026-10-25 (design note §5.2): memory grows with states x room
    from tenqsp.gradients import record_room

    assert record_room(141) == 4 and record_room(2_000) == 4 and record_room(2_001) == 2
