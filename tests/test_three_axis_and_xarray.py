# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 exit criteria: a model with a three-axis field runs and conserves
cells; ``Solution.to_xarray()`` carries every output with its axis coordinates."""

from __future__ import annotations

import numpy as np
import pytest

import tenqsp as tq
from tenqsp.examples import toy_three_axis_model


@pytest.fixture(scope="module")
def toy():
    m = toy_three_axis_model()
    t = np.linspace(0, 12, 7)
    return m, m.solve((0, 12), times=t, rtol=1e-9, atol=1e-6)


def test_three_axis_field_has_its_shape_and_the_books_close(toy):
    _, sol = toy
    assert sol["I"].shape == (7, 24, 16, 2)  # (t, a, z, s)
    assert sol["infected"].max() > 1e6 and sol["L"][-1] > 1e7  # infection and lysis happen
    assert max(sol.ledger.check().values()) < 1e-9


def test_without_growth_cells_are_conserved_exactly():
    """g_max = 0: bacteria + infected + lysed stays at the inoculum (to round-off)."""
    m = toy_three_axis_model()
    sol = m.solve((0, 12), times=np.linspace(0, 12, 7), params={"g_max": 0.0}, rtol=1e-10)
    total = sol["bacteria"] + sol["infected"] + sol["L"]
    np.testing.assert_allclose(total, total[0], rtol=1e-12)


def test_genotypes_are_kept_after_infection(toy):
    """Resistant cells adsorb 100x less, so their share of the infected cells is far below
    their share of the bacteria; infected cells never change genotype."""
    _, sol = toy
    w_a, w_z = 1.5 / 24, 1 / 16
    infected_by_s = sol["I"][3].sum(axis=(0, 1)) * w_a * w_z
    bacteria_by_s = sol["B"][3].sum(axis=0) * w_z
    assert infected_by_s[1] / infected_by_s.sum() < 0.05 * bacteria_by_s[1] / bacteria_by_s.sum()


def test_three_axis_sparse_jacobian_matches_the_dense_one():
    cm = toy_three_axis_model(n_z=6, n_a=8).compile()
    rng = np.random.default_rng(3)
    y = rng.uniform(0, 1e6, cm.n_states)
    np.testing.assert_allclose(
        cm.sparse_jacobian(0.0, y).toarray(), np.asarray(cm.jacobian(0.0, y)), rtol=1e-12, atol=0
    )


def test_to_xarray_carries_axes_coordinates_and_units(toy):
    xr = pytest.importorskip("xarray")
    m, sol = toy
    ds = sol.to_xarray()
    assert isinstance(ds, xr.Dataset)
    assert ds["I"].dims == ("t", "a", "z", "s") and ds["B"].dims == ("t", "z", "s")
    assert ds["P"].dims == ("t",) and ds["bacteria"].dims == ("t",)
    np.testing.assert_array_equal(ds["t"], sol.t)
    np.testing.assert_allclose(ds["a"], m["a"].centers)
    np.testing.assert_allclose(ds["a"].attrs["faces"], m["a"].faces)
    np.testing.assert_allclose(ds["a_width"], m["a"].widths)
    assert list(ds["s"].values) == ["S", "R"]
    assert ds["a"].attrs["units"] == "h" and ds["a"].attrs["kind"] == "one_way"
    assert ds["s"].attrs["kind"] == "categorical"
    assert ds["I"].attrs["units"] == "cells" and ds["P"].attrs["units"] == "PFU"
    assert ds.attrs["model"] == "three_axis_toy"
    np.testing.assert_array_equal(ds["I"].values, sol["I"])
    # selecting by label and integrating by the widths gives the observable back
    infected = (ds["I"] * ds["a_width"] * ds["z_width"]).sum(("a", "z", "s"))
    np.testing.assert_allclose(infected, sol["infected"], rtol=1e-12)
    assert float(ds["B"].sel(s="R").isel(t=0).sum()) > 0


def test_to_xarray_observable_over_an_axis():
    pytest.importorskip("xarray")
    m = tq.Model("obs")
    s = m.axis("s", labels=["S", "R"])
    x = m.axis("x", 0, 1, 4)
    B = m.field("B", axes=[x, s], init=1.0)
    m.hazard(B, 0.1)
    m.observable("per_class", m.integral(B, over=x, name="by_s"))
    ds = m.solve((0, 1), times=[0, 1], rtol=1e-10, atol=1e-12).to_xarray()
    assert ds["per_class"].dims == ("t", "s")
    np.testing.assert_allclose(ds["per_class"].sel(s="R"), np.exp(-0.1 * np.array([0, 1])), 1e-6)


def test_to_xarray_leaves_out_a_width_coordinate_whose_name_is_taken():
    pytest.importorskip("xarray")
    m = tq.Model("taken")
    z = m.axis("z", 0, 1, 4)
    N = m.field("N", axes=[z])
    c = m.compartment("z_width", init=1.0)
    m.hazard(N, 0.1)
    m.ode(c, "-z_width")
    m.init(N, 1.0)
    ds = m.solve((0, 1), times=[0, 1]).to_xarray()
    assert "z_width" in ds.data_vars and "z_width" not in ds.coords
    np.testing.assert_allclose(ds["z"].attrs["widths"], 0.25)
