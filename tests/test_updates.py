# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``tq.known_issues()`` and ``tq.check_updates()``: the shipped list is well formed, the
installed version's problems are found, and ``check_updates`` reports fixes, open problems and
features from the newest version's list and changelog. No test touches the network: the
fetch is replaced by canned answers."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

import tenqsp as tq
from tenqsp import updates

ROOT = Path(__file__).resolve().parent.parent
AREAS = {
    "doses",
    "solver",
    "derivatives",
    "model",
    "transfers",
    "extinction",
    "geometry",
    "notation",
    "results",
    "batches",
}
FIELDS = {
    "id",
    "title",
    "area",
    "introduced",
    "fixed_in",
    "changes_results",
    "details",
    "workaround",
}


def issue(id_, introduced, fixed_in, changes_results, area=("derivatives",)):
    return {
        "id": id_,
        "title": f"problem {id_}",
        "area": list(area),
        "introduced": introduced,
        "fixed_in": fixed_in,
        "changes_results": changes_results,
        "details": "d",
        "workaround": "w",
    }


CHANGELOG = """# Changelog

## [Unreleased]

### Added
- **Not released yet:** something.

## [0.9.0] — 2026-12-01

### Added
- **Bigger models:** more states.
- `tq.thing(x)`: a new function. It does a lot.

### Fixed
- A fix.

## [0.8.0] — 2026-11-09

### Added
- **Old feature:** already installed.
"""


def fake_fetch(latest, issues, changelog=CHANGELOG, calls=None):
    def fetch(url):
        if calls is not None:
            calls.append(url)
        if url == updates.PYPI:
            return json.dumps({"info": {"version": latest}})
        if url.endswith("known_issues.json"):
            assert f"/v{latest}/" in url
            return json.dumps({"schema": 1, "issues": issues})
        if url.endswith("CHANGELOG.md"):
            return changelog
        raise AssertionError(url)

    return fetch


def test_the_shipped_list_is_well_formed():
    data = json.loads((ROOT / "src/tenqsp/known_issues.json").read_text())
    assert data["schema"] == 1 and data["issues"]
    ids = [i["id"] for i in data["issues"]]
    assert len(ids) == len(set(ids))
    next_release = updates._base(tq.__version__)
    for i in data["issues"]:
        assert set(i) == FIELDS, i["id"]
        assert set(i["area"]) <= AREAS and i["area"], i["id"]
        assert isinstance(i["changes_results"], bool)
        assert updates._parse(i["introduced"])
        if i["fixed_in"] is not None:
            assert updates._parse(i["introduced"]) < updates._parse(i["fixed_in"]), i["id"]
            # a fix names a released version or the one this code will be released as
            assert updates._parse(i["fixed_in"]) <= next_release, i["id"]
    assert updates._shipped() == data  # what the installed package reads


def test_versions_sort_as_releases_do():
    p = updates._parse
    assert p("0.8.1.dev0") < p("0.8.1") < p("0.9.0") < p("0.10.0") < p("1.0.0")
    assert updates._base("0.8.1.dev3") == p("0.8.1")
    with pytest.raises(ValueError, match="not a TenQSP version"):
        p("0.9")


def test_known_issues_lists_the_problems_of_the_installed_version(monkeypatch, capsys):
    monkeypatch.setattr(updates, "_installed", lambda: "0.6.0")
    found = tq.known_issues(show=False)
    every = updates._shipped()["issues"]
    assert {i["id"] for i in found} == {i["id"] for i in every}  # all from 0.6.0 on
    assert [i["changes_results"] for i in found] == sorted(
        (i["changes_results"] for i in found), reverse=True
    )  # those that can change results first
    assert tq.known_issues() is None
    out = capsys.readouterr().out
    assert out.startswith(f"TenQSP 0.6.0: {len(found)} known issues in this version.")
    assert out.index("Can change results:") < out.index("Fail with an error")
    assert "tq.check_updates()" in out


def test_a_fixed_problem_is_not_listed_from_its_fix_on(monkeypatch):
    shipped = {
        "schema": 1,
        "issues": [
            issue("a", "0.6.0", "0.7.0", True),
            issue("b", "0.6.0", None, False),
            issue("c", "0.7.0", "0.8.1", False),
        ],
    }
    monkeypatch.setattr(updates, "_shipped", lambda: shipped)
    for version, ids in (
        ("0.6.0", {"a", "b"}),
        ("0.6.1", {"a", "b"}),
        ("0.7.0", {"b", "c"}),
        ("0.8.0", {"b", "c"}),
        ("0.8.1.dev0", {"b"}),
        ("0.8.1", {"b"}),
    ):
        monkeypatch.setattr(updates, "_installed", lambda v=version: v)
        assert {i["id"] for i in tq.known_issues(show=False)} == ids, version


def test_no_known_issues(monkeypatch, capsys):
    monkeypatch.setattr(updates, "_shipped", lambda: {"schema": 1, "issues": []})
    tq.known_issues()
    assert "no known issues in this version" in capsys.readouterr().out


def test_check_updates_reports_fixes_open_problems_and_features(monkeypatch, capsys):
    newest = [
        issue("old", "0.6.0", "0.7.0", True),  # fixed before the installed version
        issue("res", "0.6.0", "0.9.0", True, ("doses",)),
        issue("err", "0.8.0", "0.9.0", False),
        issue("open", "0.6.0", None, False, ("solver",)),
    ]
    monkeypatch.setattr(updates, "_installed", lambda: "0.8.0")
    monkeypatch.setattr(updates, "_fetch", fake_fetch("0.9.0", newest))
    info = tq.check_updates(show=False)
    assert info["installed"] == "0.8.0" and info["latest"] == "0.9.0" and info["error"] is None
    assert [i["id"] for i in info["fixes_that_change_results"]] == ["res"]
    assert [i["id"] for i in info["other_fixes"]] == ["err"]
    assert [i["id"] for i in info["open_issues"]] == ["open"]
    assert info["features"] == ["Bigger models (0.9.0)", "`tq.thing(x)` (0.9.0)"]

    assert tq.check_updates() is None
    out = capsys.readouterr().out
    assert out.startswith("TenQSP 0.8.0 installed, 0.9.0 available.")
    order = [
        "Fixes that may change your results:",
        "problem res [doses] (fixed in 0.9.0)",
        "Other fixes:",
        "Known issues not yet fixed:",
        "problem open [solver]",
        "What to do: w",
        "New features:",
        "Bigger models",
        "pip install -U tenqsp",
    ]
    assert [out.index(s) for s in order] == sorted(out.index(s) for s in order)
    assert "problem old" not in out


def test_up_to_date_reads_nothing_more(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(updates, "_installed", lambda: "0.9.0")
    monkeypatch.setattr(updates, "_fetch", fake_fetch("0.9.0", [], calls=calls))
    tq.check_updates()
    out = capsys.readouterr().out
    assert out.startswith("TenQSP 0.9.0 installed: the newest release is 0.9.0.")
    assert "Upgrade" not in out and calls == [updates.PYPI]


def test_no_network_says_so_and_changes_nothing(monkeypatch, capsys):
    def offline(url):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(updates, "_fetch", offline)
    info = tq.check_updates(show=False)
    assert info["latest"] is None and "could not reach PyPI" in info["error"]
    tq.check_updates()
    out = capsys.readouterr().out
    assert "could not reach PyPI" in out and "Nothing was changed" in out
    assert "tq.known_issues()" in out


def test_the_newest_list_unreadable_falls_back_to_the_installed_one(monkeypatch):
    def fetch(url):
        if url == updates.PYPI:
            return json.dumps({"info": {"version": "0.9.0"}})
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(updates, "_installed", lambda: "0.6.0")
    monkeypatch.setattr(updates, "_fetch", fetch)
    info = tq.check_updates(show=False)
    assert "known issues" in info["error"] and "changelog" in info["error"]
    assert info["fixes_that_change_results"]  # from the installed list
    assert info["features"] == []


def test_the_changelog_of_this_repository_parses():
    text = (ROOT / "CHANGELOG.md").read_text()
    feats = updates._features(text, "0.6.0", "0.8.0")
    assert feats and all(len(f) < 110 for f in feats)
    assert any(f.endswith("(0.8.0)") for f in feats) and any(f.endswith("(0.6.1)") for f in feats)
    assert not any("(0.6.0)" in f for f in feats)  # the installed version's own


def test_options_are_keywords():
    with pytest.raises(TypeError):
        tq.known_issues(False)
    with pytest.raises(TypeError):
        tq.check_updates(False)
