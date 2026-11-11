# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Axes that are a radius (geometry=cylindrical, spherical): cells are shells, d_r(F) is the
divergence through them, (1/r^k) d_r(r^k F), and totals are amounts. Oracles: closed-form
steady profiles (sinh(phi r/R)/r in a sphere, I0 in a cylinder), Crank's series for uptake
into a sphere, exact volumes and moments, and the zero-flux equilibrium exp(v r/D), which
holds in every geometry."""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest
from scipy.special import i0

import tenqsp as tq
from tenqsp.grids import ContinuousAxis, measures

TIGHT = {"rtol": 1e-10, "atol": 1e-12}
R, DE, KR = 200.0, 6000.0, 0.6
PHI = R * np.sqrt(KR / DE)


def slab(geometry, n, rhs="d_r(De*d_r(C)) - kr*C"):
    return tq.Model.from_text(f"""
model s
axis r in [0, {R}] n={n} geometry={geometry}
param De={DE}, kr={KR}, Cb=1
field C(r)
d_t(C) = {rhs}
    at r={R}: C = Cb
""")


EXACT = {
    "spherical": lambda r: R * np.sinh(PHI * r / R) / (r * np.sinh(PHI)),
    "cylindrical": lambda r: i0(PHI * r / R) / i0(PHI),
    "planar": lambda r: np.cosh(PHI * r / R) / np.cosh(PHI),
}


@pytest.mark.parametrize("geometry", ["spherical", "cylindrical", "planar"])
def test_the_steady_profile_converges_at_second_order(geometry):
    errors = []
    for n in (20, 40, 80):
        y = np.asarray(slab(geometry, n).solve((0, 200), times=[200], **TIGHT)["C"])[-1]
        rc = (np.arange(n) + 0.5) * R / n
        errors.append(np.max(np.abs(y - EXACT[geometry](rc))))
    order = np.log2(np.array(errors[:-1]) / errors[1:])
    assert errors[-1] < 1e-4
    assert np.all(order > 1.9)


def test_uptake_into_a_sphere_follows_cranks_series():
    t = np.array([0.01, 0.03, 0.1, 0.3])
    k = np.arange(1, 400)
    crank = 1 - 6 / np.pi**2 * np.sum(
        np.exp(-(k[None] ** 2) * np.pi**2 * t[:, None]) / k[None] ** 2, axis=1
    )
    errors = []
    for n in (40, 80):
        m = tq.Model.from_text(f"""
model up
axis r in [0, 1] n={n} geometry=spherical
param D=1
field C(r)
d_t(C) = d_r(D*d_r(C))
    at r=1: C = 1
observable frac = int_r(C)/(4/3*pi)
""")
        f = np.asarray(m.solve((0, 0.3), times=t, rtol=1e-11, atol=1e-13)["frac"])
        errors.append(np.max(np.abs(f - crank)))
    assert errors[-1] < 3e-4 and errors[0] / errors[1] > 3.5


def test_the_books_close_with_a_compartment_outside_the_sphere():
    m = tq.Model.from_text("""
model ex
axis r in [0, 300] n=30 geometry=spherical units=um
param D=6000, h=50, V=1e9, k=0.1
field C(r)
compartment M
d_t(C) = d_r(D*d_r(C)) - k*C
    at r=300: -D*d_r(C) = h*(C - M/V)
d_t(M) = 4*pi*300^2*h*(C(r=300) - M/V)
M(0) = 1e9
observable inside = int_r(C)
""")
    sol = m.solve((0, 60), times=np.linspace(0, 60, 7), **TIGHT)
    moved = sol.ledger.moved
    entered = moved["C"]["boundary r=upper: flux"]
    left = moved["M"]["rate: 4*pi*300**2*h*(C(r=300) - M/V)"]
    np.testing.assert_allclose(entered, -left, rtol=1e-12, atol=1e-6)
    assert "unexplained" in sol.ledger.table()
    for name in ("C", "M"):
        assert sol.ledger.check()[name] < 1e-12


def test_volumes_totals_and_cell_averages_are_exact():
    r_max = 3.0
    for geometry, volume, moment in [
        ("spherical", 4 / 3 * np.pi * r_max**3, 4 * np.pi * r_max**5 / 5),
        ("cylindrical", np.pi * r_max**2, 2 * np.pi * r_max**4 / 4),
    ]:
        m = tq.Model.from_text(f"""
model v
axis r in [0, {r_max}] n=7 geometry={geometry}
field C(r)
field E(r)
d_t(C) = 0*C
d_t(E) = 0*E
C(0) = 1
E(0) = r^2
observable vol = int_r(C)
observable moment = int_r(E)
""")
        sol = m.solve((0, 1), times=[0])
        np.testing.assert_allclose(sol["vol"], [volume], rtol=1e-13)
        np.testing.assert_allclose(sol["moment"], [moment], rtol=1e-13)  # Gauss on shells
    a = ContinuousAxis.uniform("r", 0, 2, 4, geometry="spherical")
    np.testing.assert_allclose(a.volumes.sum(), 4 / 3 * np.pi * 8, rtol=1e-14)
    np.testing.assert_allclose(a.areas, 4 * np.pi * a.faces**2)
    areas, volumes = measures([0, 1, 2], "planar")
    np.testing.assert_array_equal(areas, [1, 1, 1])
    np.testing.assert_array_equal(volumes, [1, 1])


def test_an_amount_spread_by_a_shape_is_kept():
    m = tq.Model.from_text("""
model sh
axis r in [0, 100] n=10 geometry=spherical
field C(r)
d_t(C) = 0*C
C(0) = 5*uniform(r)
observable total = int_r(C)
""")
    sol = m.solve((0, 1), times=[0])
    np.testing.assert_allclose(sol["total"], [5.0], rtol=1e-13)
    # uniform: the same concentration in every shell
    np.testing.assert_allclose(sol["C"][0], 5 / (4 / 3 * np.pi * 100**3), rtol=1e-12)
    ring = tq.Model.from_text("""
model ring
axis r in [0, 100] n=20 geometry=spherical
field C(r)
d_t(C) = 0*C
C(0) = 7*point(r, at=52)
observable total = int_r(C)
""")
    sol = ring.solve((0, 1), times=[0])
    np.testing.assert_allclose(sol["total"], [7.0], rtol=1e-13)
    shell = 4 / 3 * np.pi * (55**3 - 50**3)
    np.testing.assert_allclose(sol["C"][0][10], 7 / shell, rtol=1e-12)


def test_the_zero_flux_equilibrium_holds_in_a_sphere():
    """Drift and diffusion with sealed ends: the flux v*C - D*dC/dr vanishes, so the
    equilibrium is exp(v r / D) in any geometry; the amount is kept. Second order with
    sharp, first with erlang (its upwind spread)."""
    errors = {}
    for scheme, n in [(" using sharp", 40), (" using sharp", 80), ("", 40), ("", 80)]:
        m = tq.Model.from_text(f"""
model fp
axis r in [0, 1] n={n} geometry=spherical
param D=0.1, v=-0.2
field C(r)
d_t(C) = d_r(D*d_r(C)) - d_r(v*C){scheme}
C(0) = 1
observable total = int_r(C)
""")
        sol = m.solve((0, 40), times=[0, 40], **TIGHT)
        y = np.asarray(sol["C"])[-1]
        shape = np.exp(-2 * (np.arange(n) + 0.5) / n)
        errors[scheme, n] = np.max(np.abs(y / y[0] - shape / shape[0]))
        np.testing.assert_allclose(sol["total"][1], sol["total"][0], rtol=1e-10)
    assert errors[" using sharp", 80] < 1e-4
    assert errors[" using sharp", 40] / errors[" using sharp", 80] > 3.8
    assert 1.9 < errors["", 40] / errors["", 80] < 2.1


def test_the_written_out_divergence_is_the_same_term():
    short = slab("spherical", 40, "d_r(De*d_r(C)) - kr*C")
    for rhs in ("1/r^2*d_r(r^2*De*d_r(C)) - kr*C", "d_r(r^2*De*d_r(C))/r^2 - kr*C"):
        long = slab("spherical", 40, rhs)
        assert long.to_ir() == short.to_ir()
    assert (
        slab("cylindrical", 10, "(1/r)*d_r(r*De*d_r(C)) - kr*C").to_ir()
        == slab("cylindrical", 10).to_ir()
    )
    # a diffusivity growing as r^2 is a model of its own, on any axis
    for geometry in ("planar", "spherical"):
        grown = slab(geometry, 10, "d_r(r^2*De*d_r(C)) - kr*C")
        assert grown.to_ir() != slab(geometry, 10).to_ir()


@pytest.mark.parametrize(
    "geometry, rhs, message",
    [
        ("planar", "1/r^2*d_r(r^2*De*d_r(C))", "declare geometry=spherical on axis 'r'"),
        ("planar", "1/r*d_r(r*De*d_r(C))", "declare geometry=cylindrical on axis 'r'"),
        ("cylindrical", "1/r^2*d_r(r^2*De*d_r(C))", "axis 'r' is cylindrical"),
        ("planar", "d_r(r*De*d_r(C))/r", "is the cylindrical form"),
    ],
)
def test_the_written_out_form_on_another_geometry_is_refused(geometry, rhs, message):
    with pytest.raises(tq.NotationError, match=message):
        slab(geometry, 10, rhs)


@pytest.mark.parametrize(
    "text, message",
    [
        ("axis r in [0, 1] n=4 geometry=spherical\nparam D=1\nfield C(r)\n"
         "d_t(C) = d_r(D*d_r(C))\n    at r=0: C = 1\n", "the centre of the sphere"),
        ("axis r in [-1, 1] n=4 geometry=spherical\n", "starts at 0 or above"),
        ("axis r in [0, 1] n=4 geometry=sphere\n", "geometry must be one of"),
        ("axis r categorical [a, b] geometry=spherical\n", "categorical axis has no geometry"),
    ],
)  # fmt: skip
def test_refusals(text, message):
    with pytest.raises(tq.NotationError, match=message):
        tq.Model.from_text("model s\n" + text)


def test_an_axis_that_does_not_start_at_the_centre_takes_conditions_at_both_ends():
    """A hollow sphere (a shell from r = 50 to 200): the inner surface is a surface."""
    m = tq.Model.from_text("""
model hollow
axis r in [50, 200] n=60 geometry=spherical
param D=1
field C(r)
d_t(C) = d_r(D*d_r(C))
    at r=50: C = 1
    at r=200: C = 0
""")
    y = np.asarray(m.solve((0, 2e5), times=[2e5], **TIGHT)["C"])[-1]
    rc = 50 + (np.arange(60) + 0.5) * 2.5
    exact = (1 / rc - 1 / 200) / (1 / 50 - 1 / 200)  # steady: C linear in 1/r
    np.testing.assert_allclose(y, exact, atol=1e-3)


def test_the_text_python_api_json_latex_and_explain():
    m = slab("spherical", 10)
    assert "axis r in [0, 200] n=10 geometry=spherical" in m.to_text()
    assert tq.Model.from_text(m.to_text()).to_ir() == m.to_ir()
    ir = m.to_ir()
    d = json.loads(ir.to_json())
    assert d["ir_version"] == "0.12" and d["axes"][0]["geometry"] == "spherical"
    assert type(ir).from_json(ir.to_json()) == ir
    py = tq.Model("s")
    r = py.axis("r", 0, 200, 10, geometry="spherical")
    np.testing.assert_allclose(r.volumes.sum(), 4 / 3 * np.pi * 200**3, rtol=1e-14)
    tex = m.to_latex()
    assert r"\frac{1}{r^{2}}\frac{\partial}{\partial r}\left(r^{2} \mathrm{De}" in tex
    text = m.explain()
    assert "the radius of a sphere (spherical)" in text and "(1/r^2)*d_r(r^2*F)" in text
    assert "the centre, which has no surface" in text
    with pytest.raises(ValueError, match="no geometry"):
        tq.Model("c").axis("s", labels=["a", "b"], geometry="spherical")


def test_both_backends_and_gradients_agree():
    m = slab("spherical", 20)
    a = np.asarray(m.solve((0, 0.5), times=[0.5], **TIGHT)["C"])
    b = np.asarray(m.solve((0, 0.5), times=[0.5], backend="jax", **TIGHT)["C"])
    np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-10)
    sim = tq.Simulator(m, outputs=["C"], **TIGHT)
    f = sim.function(tq.Scenario(times=[0.5]), wrt=["De"])
    x = np.array([DE])
    g = np.asarray(jax.jacfwd(lambda x: f(x)["C"].sum())(x))
    h = 1e-3 * DE
    fd = (float(f(x + h)["C"].sum()) - float(f(x - h)["C"].sum())) / (2 * h)
    np.testing.assert_allclose(g[0], fd, rtol=1e-5)
