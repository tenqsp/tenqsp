# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""V12 (Phase 3, step 3.4): models of the test suite written by hand in the PDE notation.

Each file in ``tests/notation_models/`` was written by an independent agent from the Python
model code and ``docs/language/notation.md`` only (not from ``to_text()``). The text, read
with ``tq.load``, must be the same model as the Python version: the canonical comparison
(``tenqsp.canonical.differences``) lists nothing. No solving; each case takes well under a
second.

A few models exist only inside test functions; they are copied below as small builders
(``_local_*``), unchanged apart from being returned.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pbpk_models import pbpk
from test_conditionals import switched_on
from test_gallery_phase2 import capillary_tissue, cat, friberg, hiv_age_model, mckendrick, savic
from test_hierarchical import two_spaces
from test_ledger import decay, downward, switch_dirichlet_robin, transit, two_way
from test_sparse_jacobian import chemotaxis_by_class, coupled
from test_transitions_phase2 import joint_model, two_phage
from test_v13_v14 import mutants
from test_v17_extinction import lineage, pk_lineage, resistant_slice
from test_v_boundaries_phase2 import biofilm, dispersion, dormancy, ou, robin_advection
from test_v_chemotaxis_fronts import boltzmann, fisher, keller_segel

import tenqsp as tq
from tenqsp.canonical import differences
from tenqsp.examples import phage_latency_model, toy_phage_model, toy_three_axis_model

HERE = Path(__file__).parent / "notation_models"


# ------------------------------------------------- models that live inside test functions
def _local_conditional():
    """Three-case conditionals on a compartment (test_conditionals style)."""
    m = tq.Model("cond")
    C = m.compartment("C", init=10.0)
    A = m.compartment("A")
    MIC, k1, k2, k3, Emax = m.params(MIC=2.0, k1=0.5, k2=0.2, k3=0.05, Emax=1.0)
    k = m.define("k", tq.cases((C > MIC, k1), (C > MIC / 2, k2), otherwise=k3))
    E = m.define("E", tq.cases((C > MIC, Emax), (C > MIC / 2, Emax / 2), otherwise=0))
    m.ode(C, -k * C)
    m.ode(A, E)
    return m


def _local_weighted_dose():
    """test_hierarchical.test_weighted_initial_total_and_dose_into_one_element."""
    m = tq.Model("dose")
    s = m.axis("s", labels=["a", "b"])
    V = m.param("V", {"a": 2.0, "b": 8.0}, over=s)
    C = m.field("C", axes=[s], weights=V)
    P = m.compartment("P", volume="Vp")
    m.params(Vp=5.0)
    m.hazard(C, 0.0)
    m.ode(P, 0 * P)
    m.init(C, total=10.0, shape=1.0)
    m.init(P, 0.0)
    m.dose(C.at(s="b"), times=[1], amounts=[16.0])
    m.dose(P, times=[1], amounts=[10.0])
    return m


def _local_oral_pk():
    """notation.md §6.1 with a second regimen (test_notation_compartments)."""
    m = tq.Model("oral_pk")
    depot, centr = m.compartment("depot"), m.compartment("centr")
    ka, CL, V = m.params(ka=1.2, CL=4.0, V=30.0)
    m.ode(depot, -ka * depot)
    m.ode(centr, ka * depot - CL / V * centr)
    m.dose(depot, times=[0, 12, 24, 36, 48, 60, 72], amounts=100.0)
    m.dose(depot, times=[0, 48], amounts=50.0)
    m.observable("conc", centr / V)
    return m


def _local_slab():
    """test_hierarchical: reaction, hazard, transition, total and output on slices."""
    m = tq.Model("slab")
    o = m.axis("organ", labels=["gut", "liver"])
    s = m.axis("sub", labels=["v", "c"])
    V = m.param("V", {"gut": {"v": 1, "c": 3}, "liver": {"v": 2, "c": 5}}, over=[o, s])
    C = m.field("C", axes=[o, s], weights=V)
    m.init(C, 1.0)
    m.reaction(C.at(sub="v"), "2.0")
    m.hazard(C.at(organ="liver", sub="c"), 0.5)
    G = m.transitions("G", axis=o, rates={"gut->liver": 4.0})
    m.transition(C.at(sub="v"), axis=o, matrix=G, of=C.at(sub="v"))
    m.observable("vasc", m.integral(C.at(sub="v"), over=o, name="vascular_amount"))
    m.observable("liver_c", C.at(organ="liver", sub="c"))
    m.observable("amount", m.integral(C, over=[o, s], name="all"))
    return m


def _local_sliced_decay():
    """test_v17_extinction.test_v17_slice_crossing_time_counts_cell_widths."""
    m = tq.Model("sliced_decay")
    c = m.axis("c", labels=["a", "b"])
    x = m.axis("x", faces=[0, 0.1, 0.5, 0.6, 1.4, 2.0])
    B = m.field("B", axes=[x, c])
    m.hazard(B, 0.5)
    m.init(B, 10.0)
    m.extinction(B.at(c="a"), threshold=1.0)
    return m


# ------------------------------------------------------------- file name -> Python model
BUILDERS = {
    "decay": decay,
    "transit": transit,
    "phage_latency": lambda: phage_latency_model(n=40),
    "toy_phage": lambda: toy_phage_model(n_z=30, n_a=40),
    "toy_phage_int_order": lambda: toy_phage_model(n_z=30, n_a=40),
    "three_axis": lambda: toy_three_axis_model(n_z=16, n_a=24),
    "mckendrick": lambda: mckendrick(50, beta="b", b=0.8, mu=0.5)[0],
    "savic": lambda: savic(4)[0],
    "friberg": lambda: friberg(dose=10.0),
    "hiv_age": lambda: hiv_age_model()[0],
    "cat": lambda: cat(7),
    "capillary_tissue": lambda: capillary_tissue(20, "sharp"),
    "ou": lambda: ou(20, "sharp"),
    "dormancy": lambda: dormancy(10, g=1.0, d=0.1, E=0.5),
    "dispersion": lambda: dispersion(10, 0.17, 2.0, bolus=True),
    "biofilm_bulk": lambda: biofilm(10, kind="bulk"),
    "robin_inlet": lambda: robin_advection(8, "lower", "sharp"),
    "boltzmann": lambda: boltzmann(10, "sharp"),
    "keller_segel": lambda: keller_segel(10, 5.0),
    "fisher": lambda: fisher(20),
    "two_spaces": two_spaces,
    "two_phage": lambda: two_phage()[0],
    "joint_switch": joint_model,
    "pbpk": pbpk,
    "lineage": lineage,
    "resistant_slice": resistant_slice,
    "pk_lineage": lambda: pk_lineage(True),
    "switch_bc": switch_dirichlet_robin,
    "two_way": two_way,
    "downward": downward,
    "coupled": coupled,
    "chemotaxis_by_class": chemotaxis_by_class,
    "mutants_nutrient": lambda: mutants(with_nutrient=True),
    "cond_inline": _local_conditional,
    "cond_cases": _local_conditional,
    "cond_block": _local_conditional,
    "switched_on": switched_on,
    "weighted_dose": _local_weighted_dose,
    "oral_pk": _local_oral_pk,
    "slab": _local_slab,
    "sliced_decay": _local_sliced_decay,
}

# Hand-written texts that fail for a reason outside the text (reader, comparison or notation).
XFAIL: dict[str, str] = {}  # both found here were fixed in step 3.4 (floats, int_ order)


def _cases():
    for name in sorted(BUILDERS):
        marks = [pytest.mark.xfail(reason=XFAIL[name], strict=True)] if name in XFAIL else []
        yield pytest.param(name, marks=marks, id=name)


def test_every_file_has_a_builder():
    assert sorted(p.stem for p in HERE.glob("*.tq")) == sorted(BUILDERS)


@pytest.mark.parametrize("name", list(_cases()))
def test_hand_written_text_is_the_python_model(name):
    python = BUILDERS[name]()
    text = tq.load(HERE / f"{name}.tq")
    assert differences(python.to_ir(), text.to_ir()) == []
