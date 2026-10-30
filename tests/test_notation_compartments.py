# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Phase 3 step 3.2: the notation reader for compartments and formulas
(docs/language/notation.md).

Oracles: closed-form solutions (Bateman superposition for repeated oral doses; piecewise
exponential decay for a conditional rate), and the same model built with the Python API
(the reader must give the identical IR)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

import tenqsp as tq

PK = """
# one-compartment oral PK
param ka=1.2, CL=4, V=30

d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
C2 = centr/V

dose depot at t=0 to 72 every 12: 100
observable conc = C2
"""


def bateman(t, times, amount=100.0, ka=1.2, k=4 / 30, V=30.0):
    out = np.zeros_like(t)
    for td in times:
        s = np.clip(t - td, 0, None)
        out += np.where(
            t >= td, amount * ka / (V * (ka - k)) * (np.exp(-k * s) - np.exp(-ka * s)), 0
        )
    return out


def test_oral_pk_with_a_regimen_matches_bateman_superposition():
    m = tq.Model.from_text(PK, name="pk")
    t = np.linspace(0, 96, 193)
    sol = m.solve((0, 96), times=t, rtol=1e-10, atol=1e-12)
    expected = bateman(t, range(0, 73, 12))
    np.testing.assert_allclose(sol["conc"], expected, rtol=1e-7, atol=1e-9 * expected.max())


def test_text_and_python_give_the_same_ir():
    text = tq.Model.from_text(PK, name="pk").to_ir()
    m = tq.Model("pk")
    m.params(ka=1.2, CL=4, V=30)
    depot, centr = m.compartment("depot"), m.compartment("centr")
    m.ode(depot, "-ka*depot")
    m.ode(centr, "ka*depot - CL/V*centr")
    m.define("C2", "centr/V")
    m.dose(depot, times=[0, 12, 24, 36, 48, 60, 72], amounts="100")
    m.observable("conc", "C2")
    assert text.content_hash() == m.to_ir().content_hash()


def test_statement_order_does_not_matter():
    lines = [ln for ln in PK.splitlines() if ln.strip() and not ln.startswith("#")]
    t = np.linspace(0, 96, 49)
    ref = tq.Model.from_text(PK, name="pk").solve((0, 96), times=t)["conc"]
    rng = random.Random(4)
    for _ in range(3):
        rng.shuffle(lines)
        m = tq.Model.from_text("\n".join(lines), name="pk")
        # A new order is a new state layout, so the solver adds in a different order:
        # equal to rounding, not bit for bit (CI CPUs differ by ~1e-14).
        np.testing.assert_allclose(m.solve((0, 96), times=t)["conc"], ref, rtol=1e-12, atol=0)


# ----------------------------------------------------------------------------- conditionals
COND_HEAD = """
param k1=0.5, k2=0.2, k3=0.05, MIC=4
d_t(C) = -k*C
C(0) = 10
"""
SPELLINGS = {
    "inline": "k = k1 if C > MIC else k2 if C > MIC/2 else k3",
    "cases": """
k = k1   if C > MIC
    k2   if C > MIC/2
    k3   otherwise
""",
    "block": """
if C > MIC:
    k = k1
    unused = 1
elif C > MIC/2:
    k = k2
    unused = 2
else:
    k = k3
    unused = 3
""",
}


@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_the_three_conditional_spellings_give_the_closed_form_solution(spelling):
    """C falls at k1 until it reaches MIC, then at k2 until MIC/2, then at k3."""
    m = tq.Model.from_text(COND_HEAD + SPELLINGS[spelling], name="cond")
    t1 = math.log(10 / 4) / 0.5
    t2 = t1 + math.log(2) / 0.2
    t = np.array([0.0, 0.5 * t1, t1, 0.5 * (t1 + t2), t2, t2 + 10])
    exact = np.array(
        [
            10,
            10 * math.exp(-0.5 * 0.5 * t1),
            4,
            4 * math.exp(-0.2 * 0.5 * (t2 - t1)),
            2,
            2 * math.exp(-0.5),
        ]
    )
    sol = m.solve((0, t[-1]), times=t, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sol["C"], exact, rtol=1e-7)


def test_block_sets_one_conditional_formula_per_name():
    m = tq.Model.from_text(COND_HEAD + SPELLINGS["block"], name="cond")
    names = {d.name for d in m.to_ir().definitions}
    assert names == {"k", "unused"}


# ----------------------------------------------------------------------------- declarations
def test_axes_parameters_and_tables_match_the_python_api():
    text = """
axis a in [0, 2] n=40 units=h kind=one_way
axis x faces=[0, 0.1, 0.5, 1]
axis g in [0, 1] n=5 spacing=geometric ratio=1.5
axis s categorical [S, R]
axis organ categorical [lung, "wild liver"]
axis sub categorical [v, c]
param k[s] = {S: 1e-9, R: 1e-12}
param kel = 4/100 units=1/h
param V[organ, sub] = {
    lung:         {v: 0.06, c: 0.96},
    "wild liver": {c: 1.44, v: 0.09},   # labels in any order
}
param F[s, organ, sub] = {
    S: {lung: {v: 1, c: 2}, "wild liver": {v: 3, c: 4}},
    R: {lung: {v: 5, c: 6}, "wild liver": {v: 7, c: 8}},
}
d_t(P) = -kel*P
"""
    got = tq.Model.from_text(text, name="decl").to_ir()
    m = tq.Model("decl")
    m.axis("a", 0, 2, 40, units="h", kind="one_way")
    m.axis("x", faces=[0, 0.1, 0.5, 1])
    m.axis("g", 0, 1, 5, spacing="geometric", ratio=1.5)
    s = m.axis("s", labels=["S", "R"])
    organ = m.axis("organ", labels=["lung", "wild liver"])
    sub = m.axis("sub", labels=["v", "c"])
    m.param("k", {"S": 1e-9, "R": 1e-12}, over=s)
    m.param("kel", 0.04, units="1/h")
    m.param(
        "V",
        {"lung": {"v": 0.06, "c": 0.96}, "wild liver": {"v": 0.09, "c": 1.44}},
        over=[organ, sub],
    )
    m.param("F", [1, 2, 3, 4, 5, 6, 7, 8], over=[s, organ, sub])
    P = m.compartment("P")
    m.ode(P, "-kel*P")
    assert got.axes == m.to_ir().axes
    assert got.params == m.to_ir().params
    assert got.content_hash() == m.to_ir().content_hash()


def test_compartment_options_and_lists():
    m = tq.Model.from_text(
        """
param V_art=1.7
compartment A, B
compartment Cart units=mg/L volume=V_art
d_t(A) = -A
d_t(B) = A
d_t(Cart) = 0
""",
        name="c",
    )
    specs = {c.name: c for c in m.to_ir().compartments}
    assert set(specs) == {"A", "B", "Cart"}
    assert specs["Cart"].volume == "V_art" and specs["Cart"].units == "mg/L"


def test_continuation_lines_and_indented_text():
    text = """
        param a=1, b=2,
              c=3
        d_t(X) = a*X
                 - b*X
            + c
        X(0) = 1
    """
    m = tq.Model.from_text(text, name="cont")
    (term,) = m.to_ir().terms
    assert term.expr.replace(" ", "") == "a*X-b*X+c"
    assert {p.name for p in m.to_ir().params} == {"a", "b", "c"}


def test_two_regimens_on_one_compartment_add_up():
    """10 every 8 h and 20 every 6 h: at 0, 24 and 48 both fire (30)."""
    m = tq.Model.from_text(
        """
d_t(P) = 0
dose P at t=0 to 48 every 8: 10
dose P at t=0 to 48 every 6: 20
""",
        name="two",
    )
    sol = m.solve((0, 50), times=[1, 23, 25, 49])
    # amounts given at 0, 6, 8, 12, 16, 18, 24, 30, 32, 36, 40, 42, 48
    n8 = np.array([1, 3, 4, 7])  # doses of 10 given by t = 1, 23, 25, 49
    n6 = np.array([1, 4, 5, 9])  # doses of 20
    np.testing.assert_allclose(sol["P"], 10 * n8 + 20 * n6, rtol=1e-12)


def test_dose_amounts_per_time_and_lists_of_times():
    m = tq.Model.from_text("d_t(P) = 0\ndose P at t=0, 24: 1e8, 5e7", name="d")
    (d,) = m.to_ir().doses
    assert d.times == (0.0, 24.0) and d.amounts == ("1e8", "5e7")


# ----------------------------------------------------------------------------- python entry
def test_load_cells_and_params(tmp_path):
    f = tmp_path / "decay.tq"
    f.write_text(
        "axis a in [0, 1] n=10\nparam k=1, w[s] = {S: 1, R: 2}\n"
        "axis s categorical [S, R]\nd_t(P) = -k*P\nP(0) = 1\n",
        encoding="utf-8",
    )
    m = tq.load(f, cells={"a": 25}, params={"k": 2.0, "w": {"S": 3, "R": 4}})
    ir = m.to_ir()
    assert ir.name == "decay"
    assert len(ir.axis("a").faces) == 26
    assert {p.name: p.value for p in ir.params} == {"k": 2.0, "w": (3.0, 4.0)}
    np.testing.assert_allclose(
        m.solve((0, 1), times=[1], rtol=1e-11, atol=1e-13)["P"], math.exp(-2), rtol=1e-8
    )
    with pytest.raises(tq.NotationError, match=r"params= names \['kk'\]"):
        tq.load(f, params={"kk": 1})
    with pytest.raises(tq.NotationError, match=r"cells= names \['b'\]"):
        tq.load(f, cells={"b": 3})
    with pytest.raises(ValueError, match=".tq model files"):
        tq.load(tmp_path / "decay.txt")


def test_model_line_names_the_model():
    assert tq.Model.from_text("model friberg\nd_t(P) = 0").name == "friberg"
    assert tq.Model.from_text("d_t(P) = 0").name == "model"


def test_table_turns_a_long_table_into_nested_braces():
    data = {
        "organ": ["lung", "lung", "liver", "liver"],
        "sub": ["v", "c", "v", "c"],
        "V": [0.06, 0.96, 0.09, 1.44],
    }
    assert tq.table(data, axes=["organ", "sub"], value="V") == {
        "lung": {"v": 0.06, "c": 0.96},
        "liver": {"v": 0.09, "c": 1.44},
    }
    data["organ"][1], data["sub"][1] = "lung", "v"
    with pytest.raises(ValueError, match="given twice"):
        tq.table(data, axes=["organ", "sub"], value="V")
    with pytest.raises(ValueError, match="no column"):
        tq.table(data, axes=["organ"], value="X")
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"s": ["S", "R"], "k": [1e-9, 1e-12]})
    m = tq.Model.from_text(
        "axis s categorical [S, R]\nparam k[s] = {S: 0, R: 0}\nd_t(P) = 0",
        params={"k": tq.table(df, axes=["s"], value="k")},
    )
    assert m.to_ir().params[0].value == (1e-9, 1e-12)


# ----------------------------------------------------------------------------- refusals
@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "compartment P = 1e7",
            r"a declaration has no value; the value at time 0 is written P\(0\) = 1e7",
        ),
        ("d/dt(P) = -P", r"written d_t\(X\) = …, not d/dt"),
        ("k <- 2", "use '=', not '<-'"),
        ("d_t(P) = -k*P\nk = 1 if P > 1", "last case must be 'value otherwise'"),
        ("d_t(P) = 0\nP(0) = 1\nP(0) = 2", r"value at time 0 of this part of P is already given"),
        ("Q(0) = 1", "'Q' is not declared"),
        ("d_t(P) = 0\ndose P at t=0 to 100 every 24: 1", "not a whole number of steps"),
        ("d_t(P) = 0\ndose P at t=0, 12, 24: 1, 2", "3 dose time"),
        ("field B(z)", "nothing named 'z'"),
        ("d_t(P) = -int_a(P)", "'a' is not an axis"),
        ("param k = kk/2", "name a formula instead"),
        ("param k[s] = 1", "in braces"),
        ("init P = 1", r"written X\(0\) = value"),
        (
            "d_t(P) = -k*P\nif P > 1:\n    k = 1\nelse:\n    j = 2",
            "every branch must set the same names",
        ),
        ("d_t(P) = -k*P\nif P > 1:\n    k = 1\nelif P > 0:\n    k = 2", "ends with 'else:'"),
        ("d_t(P) = -k*P\nif P > 1:\n    d_t(P) = 1\nelse:\n    k = 2", "only names formulas"),
        ("axis a in [0, 1]", "n="),
        ("xyz", "cannot read this statement"),
    ],
)
def test_refusals_give_the_line_and_a_fix(text, message):
    with pytest.raises(tq.NotationError, match=message) as e:
        tq.Model.from_text(text)
    assert "line " in str(e.value)


def test_every_problem_is_reported_at_once_with_its_line():
    text = """param k=1
d_t(P) = -kk*P
compartment Q = 3
d/dt(R) = 1
"""
    with pytest.raises(tq.NotationError) as e:
        tq.Model.from_text(text)
    msg = str(e.value)
    assert "line 3:" in msg and "line 4:" in msg
    assert len(e.value.problems) == 2  # read errors first; the unknown name after they are fixed
    fixed = text.replace("compartment Q = 3", "Q(0) = 3\nd_t(Q) = 0").replace("d/dt", "d_t")
    caret = r"(?s)line 2: d_t\(P\) = -kk\*P\n +\^\nUnknown name 'kk'.*Did you mean 'k'"
    with pytest.raises(tq.NotationError, match=caret):
        tq.Model.from_text(fixed)


def test_a_formula_error_points_at_the_column():
    with pytest.raises(tq.NotationError) as e:
        tq.Model.from_text("d_t(P) = -P & 2")
    lines = str(e.value).splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("line 1: d_t(P) = -P & 2"))
    assert lines[i + 1].index("^") == lines[i].index("-P & 2")  # column 1 of the formula


def test_a_line_ending_with_a_comma_continues():
    """Found by the independent V12 check: a trailing comma used to end the statement, and
    the next line became a named formula without a word."""
    m = tq.Model.from_text("param a=1,\nb=2\nd_t(P) = -a*b*P")
    assert {p.name for p in m.to_ir().params} == {"a", "b"}
    assert not m.to_ir().definitions


def test_the_same_total_in_another_axis_order_is_stored_once():
    m = tq.Model.from_text(
        "axis a in [0, 1] n=3\naxis z in [0, 1] n=2\nfield I(a, z)\nd_t(I) = 0\n"
        "d_t(P) = int_z(int_a(I)) - int_a(int_z(I))"
    )
    assert len(m.to_ir().integrals) == 1
