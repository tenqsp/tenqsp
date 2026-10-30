# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Step 4.4b: derivatives of a recorded run (V8; design note docs/design/step4_4_gradients.md
§3). Each replayed step gets the derivative of its implicit equation, so forward and reverse
mode of the replay are derivatives of the computed solution. Oracles: closed forms
differentiated by JAX, forward = reverse, and central finite differences of full solves.
Dose and event *times* moving with the parameters is step 4.4c."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq
from tenqsp.backends import jax_replay
from tenqsp.examples import toy_phage_model

TIGHT = {"rtol": 1e-10, "atol": 1e-12}
T = np.array([0.5, 1, 2, 4, 8, 12, 24.0])


def pk_model(extra=""):
    return tq.Model.from_text(
        "param ka = 1.2, CL = 4, V = 30, Dose = 100, C0 = 0\n"
        "compartment depot, centr\n"
        "d_t(depot) = -ka*depot\n"
        "d_t(centr) = ka*depot - CL/V*centr\n"
        "centr(0) = C0\n" + extra,
        name="pk",
    )


def oral(p, doses):
    """Amount in the central compartment, superposed oral doses plus C0 eliminated."""
    ka, CL, V, dose, c0 = p
    ke = CL / V
    out = c0 * jnp.exp(-ke * T)
    for td in doses:
        s = jnp.clip(T - td, 0, None)
        out = out + dose * ka / (ka - ke) * (jnp.exp(-ke * s) - jnp.exp(-ka * s))
    return out


NAMES = ("ka", "CL", "V", "Dose", "C0")


def jacobians(m, t_span, **kw):
    """d centr(T) / d (ka, CL, V, Dose, C0), forward and reverse mode."""
    cm = m.compile()
    rec = jax_replay.record(cm, t_span, times=T, **kw)
    idx = [cm.params.index[k].start for k in NAMES]
    out = cm.layout.slices["centr"].start

    def f(th):
        return jax_replay.replay(cm, rec, th)[:, out]

    th = jnp.asarray(rec.run.th)
    fwd = np.asarray(jax.jacfwd(f)(th))[:, idx]
    rev = np.asarray(jax.jacrev(f)(th))[:, idx]
    return fwd, rev


def test_oral_pk_against_the_closed_form():
    """One dose of the model at t = 0 (its amount a parameter) and a starting amount C0."""
    fwd, rev = jacobians(
        pk_model("dose depot at t=0: Dose\n"), (0, 24), params={"C0": 5.0}, **TIGHT
    )
    exact = np.asarray(jax.jacfwd(lambda p: oral(p, [0.0]))(jnp.array([1.2, 4, 30, 100, 5.0])))
    scale = np.max(np.abs(exact), axis=0)
    np.testing.assert_allclose(fwd / scale, exact / scale, atol=1e-7)
    np.testing.assert_allclose(rev, fwd, rtol=0, atol=1e-12 * np.max(scale))


def test_across_doses_at_fixed_times():
    """Doses of the model at 0, 6 and 12 h split the run into segments; the derivatives
    carry across each boundary and each dose."""
    m = pk_model("dose depot at t=0, 6, 12: Dose\n")
    fwd, rev = jacobians(m, (0, 24), **TIGHT)
    p = jnp.array([1.2, 4, 30, 100, 0.0])
    exact = np.asarray(jax.jacfwd(lambda p: oral(p, [0.0, 6.0, 12.0]))(p))
    scale = np.max(np.abs(exact), axis=0) + 1e-300
    np.testing.assert_allclose(fwd / scale, exact / scale, atol=1e-7)
    np.testing.assert_allclose(rev, fwd, rtol=0, atol=1e-12 * np.max(scale))


def _toy(beta, delta):
    return toy_phage_model(8, 6).solve(
        (0, 6), times=[3.0, 6.0], params={"beta": beta, "delta": delta}, ledger=False,
        rtol=1e-11, atol=1e-8,
    )  # fmt: skip


@pytest.mark.slow  # 22 s: in the slow tier (fast tier < 2 min)
def test_a_field_model_against_finite_differences():
    """The toy phage model (age x space x strain): d(phage at 3 and 6 h)/d(burst size, decay)
    against central differences of full solves (each re-chooses its steps, hence the
    tolerance 1e-4 of V8)."""
    m = toy_phage_model(8, 6)
    cm = m.compile()
    beta, delta = 50.0, 0.1
    rec = jax_replay.record(
        cm, (0, 6), times=[3.0, 6.0], params={"beta": beta, "delta": delta}, rtol=1e-11,
        atol=1e-8,
    )  # fmt: skip
    P = cm.layout.slices["P"].start
    idx = [cm.params.index[k].start for k in ("beta", "delta")]
    g = np.asarray(jax.jacfwd(lambda th: jax_replay.replay(cm, rec, th)[:, P])(rec.run.th))
    g = g[:, idx]
    for j, (name, v) in enumerate((("beta", beta), ("delta", delta))):
        h = 1e-4 * v
        up = _toy(beta + h * (name == "beta"), delta + h * (name == "delta"))["P"]
        down = _toy(beta - h * (name == "beta"), delta - h * (name == "delta"))["P"]
        fd = (up - down) / (2 * h)
        np.testing.assert_allclose(g[:, j], fd, rtol=1e-4)


@pytest.mark.parametrize("mode", ["jvp", "vjp"])
def test_without_differentiation_the_replay_is_unchanged(mode):
    """The derivative rule changes nothing in the values: the primal is the recorded run."""
    cm = pk_model("dose depot at t=0: Dose\n").compile()
    rec = jax_replay.record(cm, (0, 24), times=T, **TIGHT)
    th = jnp.asarray(rec.run.th)
    f = lambda th: jax_replay.replay(cm, rec, th)  # noqa: E731
    if mode == "jvp":
        y, _ = jax.jvp(f, (th,), (jnp.ones_like(th),))
    else:
        y, _ = jax.vjp(f, th)
    np.testing.assert_array_equal(np.asarray(y), rec.ys)


# ----------------------------------------------------------------------------------- 4.4c
# Moving boundaries: dose, infusion and switch times and
# event times move with the parameters. Each case's closed form; with times held fixed the
# derivative would be zero (t_on, w, dose and stop times) or wrong (C0, k).


def d_dparams(m, t_span, out, names, times, **kw):
    """d out(times) / d names, forward and reverse mode of the replay."""
    cm = m.compile()
    rec = jax_replay.record(cm, t_span, times=times, **kw)
    idx = [cm.params.index[k].start for k in names]
    sl = cm.layout.slices[out].start

    def f(th):
        return jax_replay.replay(cm, rec, th)[:, sl]

    th = jnp.asarray(rec.run.th)
    fwd = np.asarray(jax.jacfwd(f)(th))[:, idx]
    rev = np.asarray(jax.jacrev(f)(th))[:, idx]
    np.testing.assert_allclose(rev, fwd, rtol=0, atol=1e-11 * max(1.0, np.max(np.abs(fwd))))
    return fwd


def test_a_switch_time_on_time_moves_with_t_on():
    m = tq.Model("on")
    A = m.compartment("A", init=0.0)
    m.params(t_on=5.0, kin=1.0)
    m.ode(A, "(kin if t > t_on else 0) - 0.1*A")
    t = np.array([2.0, 6.0, 10.0, 20.0])
    g = d_dparams(m, (0, 20), "A", ["t_on", "kin"], t, **TIGHT)
    after = t > 5
    e = np.exp(-0.1 * (t - 5))
    np.testing.assert_allclose(g[:, 0], np.where(after, -e, 0.0), atol=1e-7)  # -kin e^...
    np.testing.assert_allclose(g[:, 1], np.where(after, 10 * (1 - e), 0.0), atol=1e-7)


def test_a_sampled_switch_on_time_moves_with_its_parameter():
    """A' = 1 while sin(w t) > 0: A(T) = pi / w for pi/w < T < 2 pi/w, so dA/dw = -pi/w^2."""
    m = tq.Model("days")
    A = m.compartment("A", init=0.0)
    m.params(w=1.0)
    m.ode(A, "1 if sin(w*t) > 0 else 0")
    g = d_dparams(m, (0, 5), "A", ["w"], np.array([2.0, 5.0]), **TIGHT)
    np.testing.assert_allclose(g[:, 0], [0.0, -np.pi], atol=1e-7)  # A(2) = 2: no w in it


def test_a_state_switch_moves_with_the_starting_amount():
    """C' = -0.5 C while C > 2, then -0.1 C: after ts = 2 ln(C0/2), C = 2 e^(-0.1 (t - ts)),
    so dC/dC0 = 0.2 C / C0 (held fixed it would be C / C0, five times too big)."""
    m = tq.Model("sat")
    C = m.compartment("C")
    m.params(C0=10.0)
    m.init(C, "C0")
    m.ode(C, -tq.cases((C > 2, 0.5), otherwise=0.1) * C)
    t = np.array([1.0, 5.0, 10.0])
    g = d_dparams(m, (0, 10), "C", ["C0"], t, **TIGHT)
    ts = 2 * np.log(5.0)
    c = np.where(t < ts, 10 * np.exp(-0.5 * t), 2 * np.exp(-0.1 * (t - ts)))
    np.testing.assert_allclose(g[:, 0], np.where(t < ts, c / 10, 0.2 * c / 10), rtol=1e-7)


def test_an_extinction_time_moves_with_the_death_rate():
    """B' = -k B from 100, extinct at 1 (te = ln(100)/k); X' = B collects it: after te,
    X = 99 / k, so dX/dk = -99/k^2 (with te held fixed: te/k more)."""
    m = tq.Model("decay")
    B, X = m.compartment("B", init=100.0), m.compartment("X", init=0.0)
    m.params(k=0.5)
    m.ode(B, "-k*B")
    m.ode(X, "B")
    m.extinction(B, threshold=1.0)
    t = np.array([5.0, 12.0, 20.0])
    g = d_dparams(m, (0, 20), "X", ["k"], t, **TIGHT)
    before = 100 * (5 * np.exp(-2.5) / 0.5 - (1 - np.exp(-2.5)) / 0.25)
    np.testing.assert_allclose(g[:, 0], [before, -99 / 0.25, -99 / 0.25], rtol=1e-7)


def test_dose_and_infusion_times_of_a_regimen():
    """Oral doses at 1 (a lag) and 13 h, and a 2 h infusion into centr from 30 h: the
    derivatives with respect to the dose times and the infusion's stop, against the closed
    forms (the infusion: d/dT of R/ke (1 - e^(-ke T)) e^(-ke (t - T)), T its duration)."""
    m = pk_model()
    cm = m.compile()
    t = np.array([0.5, 2.0, 12.0, 14.0, 24.0, 31.0, 33.0, 40.0])
    reg = tq.bolus("depot", 100, times=[1.0, 13.0]) + tq.infusion(
        "centr", rate=50, duration=2, start=30
    )
    rec = jax_replay.record(cm, (0, 40), times=t, regimen=reg, **TIGHT)
    sl = cm.layout.slices["centr"].start

    def f(times, stop):
        ys = jax_replay.replay(cm, rec, dose_times=times, infusion_stops=stop)
        return ys[:, sl]

    times, stop = jnp.array([1.0, 13.0]), jnp.array([32.0])
    fwd = jax.jacfwd(f, argnums=(0, 1))(times, stop)
    rev = jax.jacrev(f, argnums=(0, 1))(times, stop)
    for a, b in zip(fwd, rev, strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), atol=1e-11)
    ka, ke = 1.2, 4 / 30

    def central(times, stop):
        out = jnp.zeros_like(jnp.asarray(t))
        for td in times:
            s = jnp.clip(t - td, 0, None)
            out = out + 100 * ka / (ka - ke) * (jnp.exp(-ke * s) - jnp.exp(-ka * s))
        T = stop[0] - 30.0
        on = jnp.clip(t - 30.0, 0, None)
        inf = jnp.where(
            t < stop[0],
            50 / ke * (1 - jnp.exp(-ke * on)),
            50 / ke * (1 - jnp.exp(-ke * T)) * jnp.exp(-ke * (t - stop[0])),
        )
        return out + inf

    exact = jax.jacfwd(central, argnums=(0, 1))(times, stop)
    for a, b in zip(fwd, exact, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6)


def test_an_extinction_mid_run_against_finite_differences():
    """A drug kills bacteria below one cell (extinct there), then washes out; X counts the
    bacteria-hours. d(X, C)/d(kill, washout) against central differences of full solves."""

    def model():
        m = tq.Model("dip")
        S, C, X = (m.compartment(n) for n in ("S", "C", "X"))
        m.params(kmax=3.0, kel=0.3)
        m.init(S, 1e3)
        m.init(C, 1.0)
        m.ode(S, "S - kmax*C/(0.05 + C)*S")
        m.ode(C, "-kel*C")
        m.ode(X, "S")
        m.extinction(S, threshold=1.0)
        return m

    m = model()
    t = np.array([5.0, 20.0, 30.0])
    kw = {"rtol": 1e-11, "atol": 1e-10}
    sol = m.solve((0, 30), times=t, ledger=False, backend="jax", **kw)
    assert sol.extinctions  # it does die out
    cm = m.compile()
    rec = jax_replay.record(cm, (0, 30), times=t, **kw)
    idx = [cm.params.index[k].start for k in ("kmax", "kel")]
    g = np.asarray(jax.jacfwd(lambda th: jax_replay.replay(cm, rec, th))(rec.run.th))[..., idx]
    for j, (name, v) in enumerate((("kmax", 3.0), ("kel", 0.3))):
        h = 1e-5 * v
        up = m.solve((0, 30), times=t, params={name: v + h}, ledger=False, **kw).y
        down = m.solve((0, 30), times=t, params={name: v - h}, ledger=False, **kw).y
        fd = (up - down) / (2 * h)
        # dC/dkmax is 0: both sides are solver noise (~1e-9), so an absolute floor
        np.testing.assert_allclose(g[:, :, j], fd, rtol=1e-4, atol=1e-6 * np.max(np.abs(fd)))


def test_an_event_after_a_moving_event():
    """Two state switches in a row (C > MIC, then C > MIC/2): the first moves with k1 and
    MIC, and the second with it. Found by pytest --gradients (step 4.4d): the second event
    time took the interpolant's slope across the kink (0.77 of the true derivative)."""
    m = tq.Model.from_text(
        "param k1=0.5, k2=0.2, k3=0.05, MIC=4\nd_t(C) = -k*C\nC(0) = 10\n"
        "k = k1 if C > MIC else k2 if C > MIC/2 else k3\n",
        name="cond",
    )
    t1 = np.log(2.5) / 0.5
    t = np.array([0.5 * t1, t1 + 1.5, t1 + np.log(2) / 0.2 + 10])  # off the switch times

    def exact(p):
        k1, k2, k3, mic = p
        s1 = jnp.log(10 / mic) / k1
        s2 = s1 + jnp.log(2.0) / k2
        late = mic / 2 * jnp.exp(-k3 * (t - s2))
        mid = mic * jnp.exp(-k2 * (t - s1))
        return jnp.where(t < s1, 10 * jnp.exp(-k1 * t), jnp.where(t < s2, mid, late))

    cm = m.compile()
    rec = jax_replay.record(cm, (0, t[-1]), times=t, **TIGHT)
    th = jnp.asarray(rec.run.th)
    fwd = np.asarray(jax.jacfwd(lambda x: jax_replay.replay(cm, rec, x)[:, 0])(th))[:, :4]
    rev = np.asarray(jax.jacrev(lambda x: jax_replay.replay(cm, rec, x)[:, 0])(th))[:, :4]
    want = np.asarray(jax.jacfwd(exact)(jnp.array([0.5, 0.2, 0.05, 4.0])))
    np.testing.assert_allclose(fwd, want, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(fwd, rev, rtol=1e-9, atol=1e-12)
