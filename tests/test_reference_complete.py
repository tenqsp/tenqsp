# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The references cover everything (Phase 6): every statement, operator and function of the
notation is in docs/reference/notation.md, and every public name has a docstring and an
entry in the API reference (docs/reference/api/)."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import tenqsp as tq
from tenqsp import notation
from tenqsp.expr import FUNCTIONS, SHAPES

DOCS = Path(__file__).resolve().parents[1] / "docs" / "reference"

# statements: the keywords the reader dispatches on (checked against its source below), and
# the other forms a statement can take
STATEMENT_WORDS = [
    "model", "axis", "param", "compartment", "observable", "dose", "flux", "field",
    "transitions", "extinction", "lag", "bioavailability", "inflow", "outflow", "dirichlet",
    "robin", "zero_flux",
]  # fmt: skip
OTHER_FORMS = [
    "d_t", "at", "if", "elif", "else", "otherwise", "using sharp", "every", "noise_floor",
    "categorical", "kind", "spacing", "ratio", "faces", "units", "weights", "volume",
    "one_way", "bidirectional", "spatial", "lower", "upper", "outflow",
]  # fmt: skip
OPERATORS = ["d_", "int_", "sum_", "trans_", "grad_", "shape", "density", "t"]
FUNCTION_NAMES = sorted(set(FUNCTIONS) | set(SHAPES) | {"grad"})


def mentioned(text: str, word: str) -> bool:
    """``word`` inside inline code or a code block."""
    pat = re.compile(rf"(?<![\w]){re.escape(word)}(?![\w])")
    code = re.findall(r"`[^`\n]+`", text) + re.findall(r"```.*?```", text, re.S)
    return any(pat.search(c) for c in code)


def test_statement_list_matches_the_reader():
    src = inspect.getsource(notation._Reader.kind)
    words = set(re.findall(r'"(\w+)"', src))
    missing = words - set(STATEMENT_WORDS) - {"arrival", "boundary", "at", "if", "equation",
                                              "init", "definition", "unknown"}  # fmt: skip
    assert not missing, f"new statement keywords: add them to STATEMENT_WORDS: {missing}"


@pytest.mark.parametrize("word", STATEMENT_WORDS + OTHER_FORMS + OPERATORS + FUNCTION_NAMES)
def test_notation_reference_covers(word):
    page = DOCS / "notation.md"
    assert page.exists(), "docs/reference/notation.md is missing"
    assert mentioned(page.read_text(), word), f"'{word}' is not in the notation reference"


PUBLIC = [n for n in tq.__all__ if n != "__version__"]


@pytest.mark.parametrize("name", PUBLIC)
def test_public_name_has_a_docstring(name):
    assert inspect.getdoc(getattr(tq, name)), f"tq.{name} has no docstring"


def test_model_methods_have_docstrings():
    bare = [
        n
        for n, f in inspect.getmembers(tq.Model)
        if not n.startswith("_") and callable(f) and not inspect.getdoc(f)
    ]
    assert not bare, f"Model methods without a docstring: {bare}"


@pytest.mark.parametrize("name", PUBLIC)
def test_public_name_is_in_the_api_reference(name):
    text = "\n".join(p.read_text() for p in sorted((DOCS / "api").glob("*.md")))
    listed = set(re.findall(r"^::: tenqsp\.(\w+)", text, re.M))
    listed |= set(re.findall(r"^\s+- (\w+)$", text, re.M))  # members of a ::: tenqsp block
    listed |= set(re.findall(r"^\| `tq\.(\w+)\(", text, re.M))  # a row of a table
    assert name in listed, f"tq.{name} is not in docs/reference/api/"


INTERNAL = re.compile(r"PRD|ADR \d|\bADR\b|design note|\bowner\b|Phase \d|devlog|IR 0\.\d")


def _public_docs():
    import importlib

    from tenqsp import gradients, ledger, results

    report = importlib.import_module("tenqsp.validate").Report
    objs = {n: getattr(tq, n) for n in PUBLIC}
    classes = (tq.Model, gradients.Function, gradients.Sensitivity, results.Solution,
               ledger.Ledger, report, tq.Simulator, tq.Scenario, tq.SimResult, tq.BatchResult,
               tq.Regimen, tq.Bolus, tq.Infusion, tq.Grid, tq.PreRun, tq.Slice)  # fmt: skip
    for c in classes:
        objs[c.__qualname__] = c
        for n, f in inspect.getmembers(c):
            if not n.startswith("_"):
                objs[f"{c.__qualname__}.{n}"] = f
    for n in ("trajectories", "fan", "sensitivity", "snapshot"):
        objs[f"plot.{n}"] = getattr(tq.plot, n)
    return {k: inspect.getdoc(v) or "" for k, v in objs.items() if callable(v) or k == "plot"}


def test_public_docstrings_name_no_internal_documents():
    bad = [f"{k}: {line.strip()}" for k, d in _public_docs().items() for line in d.splitlines()
           if INTERNAL.search(line)]  # fmt: skip
    assert not bad, "docstrings shown in the API reference cite internal documents:\n" + "\n".join(
        bad
    )


def test_every_source_file_carries_the_licence_header():
    root = Path(__file__).resolve().parents[1]
    bare = [
        str(p.relative_to(root))
        for folder in ("src", "tests")
        for p in sorted((root / folder).rglob("*.py"))
        if "SPDX-License-Identifier: Apache-2.0" not in p.read_text()[:300]
    ]
    assert not bare, f"files without the SPDX licence header: {bare}"
