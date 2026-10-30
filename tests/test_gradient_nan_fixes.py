# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``sim.function`` with a drug in a Hill term, C^h with h = 1.5 (NaN for C < 0):

- a dose after the start gave a NaN value (and status "ok"): to carry the derivative with
  respect to dose times at the start, the replay took every dose back from the starting
  state, a later one too, leaving C < 0;
- with an extinction group, the gradient was NaN (the value right): on every step, the
  replay read the step's polynomial at its segment's end, far outside the step, where it
  can give C < 0; discarded, but 0 x NaN in reverse mode.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq

HILL = """model hill_kill
param g=0.8, Emax=2.5, EC50=1, h=1.5, kel=0.3, k=1e-7, beta=50
d_t(B) = g*B - Emax*A^h/(EC50^h + A^h)*B - k*B*P
d_t(P) = beta*k*B*P - k*B*P - 0.1*P
d_t(A) = -kel*A
B(0) = 1e6
observable logB = log10(B)
"""
FLOOR = "extinction phage: P below 1e-6 with noise_floor\n"


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    ("text", "doses"),
    [
        (HILL, [("A", 0.0), ("P", 2.0)]),
        (HILL, [("A", 2.0)]),
        (HILL + FLOOR, [("A", 0.0), ("P", 2.0)]),
        (HILL + FLOOR, [("A", 0.0)]),
    ],
    ids=["dose at start", "dose later", "group, phage dosed", "group, no phage"],
)
def test_value_and_gradient_are_finite_and_right(text, doses):
    m = tq.Model.from_text(text)
    sim = m.simulator(outputs=["logB"], rtol=1e-10, atol=1e-10)
    regimen = None
    for target, t in doses:
        dose = tq.bolus(target, 4.0 if target == "A" else 1e6, times=[t])
        regimen = dose if regimen is None else regimen + dose
    sc = tq.Scenario(regimen=regimen, times=[6.0])
    f = sim.function(sc, wrt=["g", "EC50", "h"])
    x0 = jnp.asarray(np.asarray(f.x0))
    value, grad = jax.value_and_grad(lambda x: f(x)["logB"][0])(x0)
    assert f.status(x0) == "ok"
    assert float(value) == pytest.approx(float(np.asarray(sim(sc)["logB"])[0]), rel=1e-8)
    grad = np.asarray(grad)
    assert np.all(np.isfinite(grad))
    for i, name in enumerate(f.names):
        step = 1e-5 * float(x0[i])
        out = [
            float(np.asarray(sim(sc.replace(params={name: float(x0[i]) + s}))["logB"])[0])
            for s in (step, -step)
        ]
        assert grad[i] == pytest.approx((out[0] - out[1]) / (2 * step), rel=1e-4)
