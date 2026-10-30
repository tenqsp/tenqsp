# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Every example on the documentation site runs (Phase 6).

Rules for the pages (docs/index.md, docs/start/, docs/guide/, docs/reference/):

- every ```python block runs, in page order, in one namespace per page (so a page reads as one
  script); a block written ```py is shown only (for code that needs a GPU, a file, ...);
- every ```tq block that declares a model (a line `model <name>`) is read with
  ``tq.Model.from_text``; shorter ```tq blocks are excerpts and are not read;
- a pair of tabs ``=== "Notation"`` then ``=== "Python"`` shows one model both ways: the
  Python block must leave the model in ``m``, and the two must be the same model
  (``tenqsp.canonical.differences`` empty).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import tenqsp as tq
from tenqsp.canonical import differences

DOCS = Path(__file__).resolve().parents[1] / "docs"
FOLDERS = ("start", "guide", "reference", "background", "project")
PAGES = sorted([DOCS / "index.md"] + [p for d in FOLDERS for p in (DOCS / d).glob("*.md")])

FENCE = re.compile(r"^(?P<indent>[ \t]*)```(?P<lang>[\w-]*)[^\n]*\n(?P<body>.*?)^(?P=indent)```",
                   re.M | re.S)  # fmt: skip
TAB = re.compile(r'^[ \t]*=== "(?P<title>[^"]+)"', re.M)


def blocks(text: str) -> list[tuple[str, str, str | None]]:
    """(language, code, title of the tab the block sits in or None), in page order."""
    out = []
    for f in FENCE.finditer(text):
        indent = f["indent"]
        code = "\n".join(line[len(indent) :] for line in f["body"].splitlines())
        tab = None
        if indent:
            before = list(TAB.finditer(text, 0, f.start()))
            if before:
                tab = before[-1]["title"]
        out.append((f["lang"], code, tab))
    return out


def is_model(code: str) -> bool:
    return re.search(r"^model\s+\w+", code, re.M) is not None


# pages that show what the scipy backend does (its Radau retry), skipped by --backend=jax
SCIPY_PAGES = {"guide/troubleshooting.md"}


@pytest.mark.parametrize(
    "page",
    [
        pytest.param(p, marks=pytest.mark.scipy_backend)
        if str(p.relative_to(DOCS)) in SCIPY_PAGES
        else p
        for p in PAGES
    ],
    ids=lambda p: str(p.relative_to(DOCS)),
)
def test_page_examples_run(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # examples that save files write here
    ns: dict = {"__name__": "__docs__"}
    notation = None  # the model of the last "Notation" tab, waiting for its "Python" twin
    for lang, code, tab in blocks(page.read_text(encoding="utf-8")):
        if lang == "tq" and is_model(code):
            read = tq.Model.from_text(code)
            notation = read if tab == "Notation" else None
        elif lang == "python":
            try:
                exec(compile(code, f"{page.name}", "exec"), ns)
            except Exception as e:
                raise AssertionError(f"{page.name}: this block failed:\n{code}") from e
            if tab == "Python" and notation is not None:
                assert "m" in ns, f"{page.name}: a Python tab must leave its model in m"
                diff = differences(notation.to_ir(), ns["m"].to_ir())
                assert diff == [], f"{page.name}: notation and Python differ: {diff}\n{code}"
            notation = None
    plt.close("all")


def test_pages_found():
    assert DOCS / "index.md" in PAGES and len(PAGES) > 3
