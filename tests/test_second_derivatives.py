# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Second derivatives through ``sim.function`` (``jax.hessian``, ``jax.jacfwd(jax.grad)``,
``jax.jacrev(jax.grad)``), with forward and reverse derivatives. Oracle: central differences
of the exact gradient, which is tested on its own elsewhere. Cases: doses at segment
junctions with dose numbers and times among the inputs, an infusion (its rate, start and
duration), and a model on the sparse tier. Where a second derivative is not available, a
clear error: in forward mode, the reverse-mode derivative of the gradient and the sparse
tier; in both modes, a model with events (the event time is exact to first order only)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq
from tenqsp.backends import jax_backend as jb

TIGHT = {"rtol": 1e-10, "atol": 1e-12}

PK = """model pk
param ka=1.2, CL=4, V=30
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
observable conc = centr/V
"""
PK_REGIMEN = tq.bolus("depot", 100, times=[0, 12]) + tq.infusion(
    "centr", rate=20, duration=2, start=6
)
# outputs at a dose time are kept off the cases where that time moves: an output there jumps
# from one side of the dose to the other, and central differences across it mean nothing
PK_CASES = {
    "amounts": (
        tq.Scenario(regimen=PK_REGIMEN, times=[1, 4, 7, 9, 12, 13, 18, 24]),
        ["ka", "CL", "V", "dose 2 amount", "infusion 1 rate"],
    ),
    "times": (
        tq.Scenario(regimen=PK_REGIMEN, times=[1, 4, 7, 9, 13, 18, 24]),
        ["ka", "CL", "dose 2 time", "infusion 1 start", "infusion 1 duration"],
    ),
}

DIFFUSION = """model spread
axis x in [0, 1] n=40
param D=0.01, k=0.3
field u(x)
d_t(u) = d_x(D*d_x(u)) - k*u
u(0) = normal(x, mean=0.5, sd=0.1)
observable total = int_x(u)
observable left = int_x(u*(1 - x))
"""

# B dies at rate kill - g; C counts the cells killed until B goes extinct at 1 cell, so C at
# the end depends on the time of that event
EXTINCT = """model kill
param g=0.5, kill=2, B0=1e3
d_t(B) = g*B - kill*B
d_t(C) = kill*B
B(0) = B0
extinction B below 1
"""


def central(grad, x, rel=1e-5):
    """Central differences of ``grad`` at ``x``, column by column."""
    x = np.asarray(x, float)
    H = np.zeros((x.size, x.size))
    for j in range(x.size):
        e = np.zeros(x.size)
        e[j] = rel * (abs(x[j]) if x[j] != 0 else 1.0)
        H[:, j] = np.asarray(grad(jnp.asarray(x + e))) - np.asarray(grad(jnp.asarray(x - e)))
        H[:, j] /= 2 * e[j]
    return (H + H.T) / 2


FORWARD = lambda h: jax.jacfwd(jax.grad(h))  # noqa: E731
REVERSE = lambda h: jax.jacrev(jax.grad(h))  # noqa: E731


def loss_of(f, output):
    return lambda v: jnp.sum(f(v)[output] ** 2)


def check(f, x, output, rtol, seconds=(jax.hessian, FORWARD, REVERSE)):
    loss = loss_of(f, output)
    ref = central(jax.grad(loss), x)
    scale = np.max(np.abs(ref))
    for second in seconds:
        H = np.asarray(second(loss)(jnp.asarray(x)))
        np.testing.assert_allclose(H, H.T, rtol=0, atol=rtol * scale)
        np.testing.assert_allclose(H, ref, rtol=0, atol=rtol * scale)


@pytest.mark.timeout(600)  # 15-35 s here; a slow CI runner takes about 4 times as long
@pytest.mark.parametrize("case", list(PK_CASES))
def test_doses_and_an_infusion_reverse(case):
    sim = tq.Model.from_text(PK).simulator(outputs=["conc"], **TIGHT)
    f = sim.function(PK_CASES[case][0], wrt=PK_CASES[case][1], derivatives="reverse")
    check(f, f.x0, "conc", rtol=1e-5)


@pytest.mark.timeout(600)  # 15-35 s here; a slow CI runner takes about 4 times as long
@pytest.mark.parametrize("case", list(PK_CASES))
def test_doses_and_an_infusion_forward(case):
    sim = tq.Model.from_text(PK).simulator(outputs=["conc"], **TIGHT)
    f = sim.function(PK_CASES[case][0], wrt=PK_CASES[case][1], derivatives="forward")
    check(f, f.x0, "conc", rtol=1e-5, seconds=(jax.hessian, FORWARD))
    # the gradient's derivative in reverse cannot pass the sensitivity run: said plainly
    with pytest.raises(ValueError, match=r"not in reverse.*derivatives='reverse'"):
        REVERSE(loss_of(f, "conc"))(jnp.asarray(f.x0))


def sparse_function(derivatives, monkeypatch):
    # a small model put on the sparse tier (the threshold lowered): at its usual size
    # (300 cells) the reverse check took ~40 s on a workstation and over 120 s on a CI runner
    monkeypatch.setattr(jb, "SPARSE_FROM_STATES", 0)
    monkeypatch.setattr(jb, "SPARSE_FROM_STATES_GPU", 0)
    m = tq.Model.from_text(DIFFUSION)
    assert jb.tier(m.compile()) == "sparse"
    return m.simulator(outputs=["left"], **TIGHT).function(
        tq.Scenario(times=[0.5, 2, 5]), wrt=["D", "k"], derivatives=derivatives
    )


@pytest.mark.timeout(600)  # 15-35 s here; a slow CI runner takes about 4 times as long
def test_a_model_on_the_sparse_tier_reverse(monkeypatch):
    f = sparse_function("reverse", monkeypatch)
    # jax.hessian only: each form is a compile of its own, and the forms are checked above
    check(f, f.x0, "left", rtol=1e-5, seconds=(jax.hessian,))


@pytest.mark.parametrize("second", [jax.hessian, FORWARD, REVERSE], ids=["hessian", "fwd", "rev"])
def test_a_model_on_the_sparse_tier_forward_refuses(second, monkeypatch):
    f = sparse_function("forward", monkeypatch)
    jax.grad(loss_of(f, "left"))(jnp.asarray(f.x0))  # the gradient itself is fine
    with pytest.raises(ValueError, match="sparse linear algebra.*derivatives='reverse'"):
        second(loss_of(f, "left"))(jnp.asarray(f.x0))


@pytest.mark.parametrize("second", [jax.hessian, FORWARD, REVERSE], ids=["hessian", "fwd", "rev"])
@pytest.mark.parametrize("derivatives", ["forward", "reverse"])
def test_through_an_extinction_event_refuses(derivatives, second):
    sim = tq.Model.from_text(EXTINCT).simulator(outputs=["C", "B"], **TIGHT)
    f = sim.function(tq.Scenario(times=[1, 3, 10]), wrt=["g", "kill", "B0"],
                     derivatives=derivatives)  # fmt: skip
    x = jnp.asarray(f.x0)
    assert float(f(x)["B"][-1]) == 0.0  # extinct before t = 10
    loss = loss_of(f, "C")
    # the gradient (which moves the event time) is right
    np.testing.assert_allclose(jax.grad(loss)(x), central_grad(loss, f.x0), rtol=1e-6)
    with pytest.raises(ValueError, match="extinction groups or switches"):
        second(loss)(x)


def central_grad(loss, x, rel=1e-6):
    x = np.asarray(x, float)
    g = np.zeros(x.size)
    for j in range(x.size):
        e = np.zeros(x.size)
        e[j] = rel * abs(x[j])
        g[j] = (float(loss(jnp.asarray(x + e))) - float(loss(jnp.asarray(x - e)))) / (2 * e[j])
    return g
