# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Known problems and newer versions: ``tq.known_issues()`` and ``tq.check_updates()``.

``known_issues.json``, shipped inside the package, lists problems found in released versions:
the first version affected, the version that fixed it (none while open), whether it could
change results without an error, and the areas it touches. ``known_issues()`` reads the copy
installed with TenQSP and needs no network. ``check_updates()`` asks PyPI for the newest
version, only when called, and reads that version's list and changelog, so it knows of
fixes made after this copy was installed. Nothing here runs on import.
"""

from __future__ import annotations

import json
import re
import urllib.request
from importlib import resources

PYPI = "https://pypi.org/pypi/tenqsp/json"
RAW = "https://raw.githubusercontent.com/tenqsp/tenqsp/v{version}/{path}"
TIMEOUT = 10  # seconds per request

_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:\.dev(\d+))?$")


def _parse(version: str) -> tuple[int, int, int, int]:
    """A version as a tuple that sorts as versions do; ``.devN`` comes before the release."""
    m = _VERSION.match(version.strip())
    if not m:
        raise ValueError(f"not a TenQSP version: {version!r}")
    major, minor, patch, dev = m.groups()
    return int(major), int(minor), int(patch), (int(dev) - 10**9) if dev is not None else 0


def _base(version: str) -> tuple[int, int, int, int]:
    """The release a version leads to: a development build of 0.8.1 holds the fixes of 0.8.1
    made so far, and lists them as fixed."""
    return _parse(version)[:3] + (0,)


def _affects(issue: dict, version: str) -> bool:
    if _parse(version) < _parse(issue["introduced"]):
        return False
    return issue["fixed_in"] is None or _base(version) < _parse(issue["fixed_in"])


def _shipped() -> dict:
    return json.loads(resources.files("tenqsp").joinpath("known_issues.json").read_text("utf-8"))


def _installed() -> str:
    from . import __version__

    return __version__


def _entry_text(issue: dict, *, fixed: bool = False) -> list[str]:
    head = f"  - {issue['title']} [{', '.join(issue['area'])}]"
    if fixed:
        head += f" (fixed in {issue['fixed_in']})"
    lines = [head]
    if not fixed and issue.get("workaround"):
        lines.append(f"    What to do: {issue['workaround']}")
    return lines


def known_issues(*, show: bool = True):
    """The known problems of the installed version, from the list shipped with it (no network).

    ``show=True`` prints them, the ones that can change results first; ``show=False`` returns
    them as a list of dicts (fields ``id``, ``title``, ``area``, ``introduced``, ``fixed_in``,
    ``changes_results``, ``details``, ``workaround``). Problems found after this version was
    released are not in its list: ``tq.check_updates()`` reads the newest one.
    """
    version = _installed()
    found = [i for i in _shipped()["issues"] if _affects(i, version)]
    found.sort(key=lambda i: not i["changes_results"])
    if not show:
        return found
    if not found:
        print(f"TenQSP {version}: no known issues in this version.")
    else:
        n = len(found)
        print(f"TenQSP {version}: {n} known issue{'s' if n > 1 else ''} in this version.")
        for results, label in (
            (True, "Can change results"),
            (False, "Fail with an error, or other problems"),
        ):
            group = [i for i in found if i["changes_results"] is results]
            if group:
                print(f"{label}:")
                for i in group:
                    print("\n".join(_entry_text(i)))
    print("Problems found later, and their fixes: tq.check_updates() (asks PyPI).")
    return None


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:  # noqa: S310 (fixed https URLs)
        return r.read().decode("utf-8")


_SECTION = re.compile(r"^## \[(?P<version>[^\]]+)\]", re.M)


def _features(changelog: str, installed: str, latest: str) -> list[str]:
    """The headings of the 'Added' entries of the versions after ``installed`` up to
    ``latest``, from a changelog in this repository's format."""
    out = []
    starts = list(_SECTION.finditer(changelog))
    for k, m in enumerate(starts):
        try:
            v = _parse(m["version"])
        except ValueError:  # "Unreleased"
            continue
        if not _parse(installed) < v <= _parse(latest):
            continue
        end = starts[k + 1].start() if k + 1 < len(starts) else len(changelog)
        body = changelog[m.end() : end]
        added = re.search(r"^### Added\n(.*?)(?=^### |\Z)", body, re.M | re.S)
        if not added:
            continue
        for item in re.findall(r"^- (.+(?:\n  .+)*)", added[1], re.M):
            text = " ".join(item.split())
            bold = re.match(r"\*\*(.+?):?\*\*", text)
            if bold:
                text = bold[1].rstrip(":")
            else:
                text = re.split(r":\s| \(|(?<=\.)\s", text, maxsplit=1)[0].rstrip(".")
                if len(text) > 90:
                    text = text[:87].rsplit(" ", 1)[0] + "..."
            out.append(f"{text} ({m['version']})")
    return out


def check_updates(*, show: bool = True):
    """Ask PyPI whether a newer TenQSP exists, and what it fixes. Needs the network; runs only
    when called.

    With a newer version, lists the fixes since the installed one that can change results,
    then the other fixes, the problems still open, and the new features. ``show=False``
    returns the same as a dict: ``installed``, ``latest``, ``fixes_that_change_results``,
    ``other_fixes``, ``open_issues``, ``features`` (lists) and ``error`` (``None``, or why
    PyPI or the newest version's list could not be read). Nothing is installed or changed.
    """
    installed = _installed()
    info = {
        "installed": installed,
        "latest": None,
        "fixes_that_change_results": [],
        "other_fixes": [],
        "open_issues": [],
        "features": [],
        "error": None,
    }
    try:
        info["latest"] = json.loads(_fetch(PYPI))["info"]["version"]
    except Exception as e:  # no network, proxy, PyPI down: say so and stop
        info["error"] = f"could not reach PyPI ({type(e).__name__}: {e})"
        info["open_issues"] = [i for i in _shipped()["issues"] if _affects(i, installed)]
        return _report(info, show)
    latest = info["latest"]
    issues = _shipped()["issues"]
    if _parse(latest) > _parse(installed):
        problems = []
        try:
            path = "src/tenqsp/known_issues.json"
            issues = json.loads(_fetch(RAW.format(version=latest, path=path)))["issues"]
        except Exception as e:
            problems.append(f"its list of known issues ({type(e).__name__}: {e})")
        try:
            changelog = _fetch(RAW.format(version=latest, path="CHANGELOG.md"))
            info["features"] = _features(changelog, installed, latest)
        except Exception as e:
            problems.append(f"its changelog ({type(e).__name__}: {e})")
        if problems:
            info["error"] = (
                f"could not read {' or '.join(problems)} for TenQSP {latest}; "
                "the fixes shown are those the installed list knows"
            )
    for i in issues:
        if not _affects(i, installed):
            continue
        if i["fixed_in"] is not None and _parse(i["fixed_in"]) <= _parse(latest):
            key = "fixes_that_change_results" if i["changes_results"] else "other_fixes"
            info[key].append(i)
        else:
            info["open_issues"].append(i)
    return _report(info, show)


def _report(info: dict, show: bool):
    if not show:
        return info
    installed, latest = info["installed"], info["latest"]
    if latest is None:
        print(f"TenQSP {installed} installed; {info['error']}. Nothing was changed.")
        print("Known issues of this version: tq.known_issues() (no network needed).")
        return None
    if _parse(latest) > _parse(installed):
        print(f"TenQSP {installed} installed, {latest} available.")
    else:
        print(f"TenQSP {installed} installed: the newest release is {latest}.")
    if info["error"]:
        print(f"Note: {info['error']}.")
    blocks = (
        ("Fixes that may change your results", info["fixes_that_change_results"], True),
        ("Other fixes", info["other_fixes"], True),
        ("Known issues not yet fixed", info["open_issues"], False),
    )
    for label, items, fixed in blocks:
        if items:
            print(f"{label}:")
            for i in items:
                print("\n".join(_entry_text(i, fixed=fixed)))
    if info["features"]:
        print("New features:")
        for f in info["features"]:
            print(f"  - {f}")
    if _parse(latest) > _parse(installed):
        print("Upgrade: pip install -U tenqsp")
    return None
