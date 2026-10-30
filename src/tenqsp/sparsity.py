# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Structural sparsity of the right-hand side, from the IR.

The Jacobian is never formed densely. It is split into three parts:

- **local part**: rows of the fields, with the integrals held fixed. Every term only couples a
  cell to nearby cells along its own axis (transport), to all categories along a categorical
  axis (transitions), or to the same cell (reactions). So the pattern is known from the
  model text, and a few columns can share one forward-mode derivative ("colours").
- **integral part**: each integral ``q`` the model declares is lifted out: ``J = dg/dy +
  dg/dq · dq/dy``. ``dg/dq`` is local; ``dq/dy`` has few rows, computed by reverse mode, and
  rows with disjoint support share one reverse pass.
- **dense rows**: compartments (and ledger channels) depend on whole fields, so their rows are
  computed exactly by reverse mode, one pass each.

This module derives the patterns and the colourings; ``compile`` evaluates them.

A pattern only has to *contain* every non-zero; extra entries cost a little speed, never
accuracy. Missing ones would be wrong, which the tests check against dense ``jax.jacfwd``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import numpy as np
import scipy.sparse as sps
import sympy as sp

FULL = "full"  # depends on every cell along the axis


@dataclasses.dataclass(frozen=True)
class Reach:
    """How far, along one axis, a cell of the target looks at cells of a source.

    - ``band`` (int): cells within that distance (0 = the same cell);
    - ``FULL``: every cell along the axis;
    - ``edge``: (side, depth) for a source that lacks the axis but is only used at one end
      of it (a boundary condition): target cells within ``depth`` of that end.
    """

    band: int | str = 0
    edge: tuple[str, int] | None = None


def merge(a: Reach | None, b: Reach) -> Reach:
    """The union of two reaches (anything mixed widens to the whole axis)."""
    if a is None:
        return b
    if a.edge is not None and b.edge is not None:
        if a.edge[0] == b.edge[0]:
            return Reach(edge=(a.edge[0], max(a.edge[1], b.edge[1])))
        return Reach(band=FULL)
    if a.edge is not None or b.edge is not None:
        return Reach(band=FULL)
    if FULL in (a.band, b.band):
        return Reach(band=FULL)
    return Reach(band=max(a.band, b.band))


# ----------------------------------------------------------------------------- references


def references(plan, e: sp.Expr | None, _seen: frozenset[str] = frozenset()) -> set[str]:
    """Fields, compartments, integrals and field-at-end values a formula uses, through
    definitions."""
    if e is None:
        return set()
    out: set[str] = set()
    kind = plan.tables.kind
    for s in e.free_symbols:
        k = kind.get(s.name)
        if k in ("field", "compartment", "integral", "end"):
            out.add(s.name)
        elif k == "definition" and s.name not in _seen:
            out |= references(plan, plan.definitions[s.name], _seen | {s.name})
    return out


def integral_sources(plan, name: str, _seen: frozenset[str] = frozenset()) -> dict[str, set[str]]:
    """``{field or compartment: axes summed over}`` for one integral, nested integrals
    included (they are computed inside it, not lifted separately)."""
    integrand, over = plan.integrals[name]
    out: dict[str, set[str]] = {}
    for r in references(plan, integrand):
        if plan.tables.kind[r] == "integral":
            if r in _seen:  # pragma: no cover - the validator refuses cycles
                continue
            for src, axes in integral_sources(plan, r, _seen | {name}).items():
                out.setdefault(src, set()).update(axes, over)
        else:
            out.setdefault(r, set()).update(over)
    return out


def quantity_sources(cm, name: str, _seen: frozenset[str] = frozenset()) -> dict[str, set[str]]:
    """``{field or compartment: axes it reaches over}`` for one lifted quantity: an integral,
    or a field's value at one end of an axis (``D(z=lower)``), which depends on the field
    near that end and on whatever its boundary condition and transport there read."""
    plan = cm.plan
    if name in plan.integrals:
        integrand, over = plan.integrals[name]
        refs, own = references(plan, integrand), set(over)
    else:
        field, ax, side = plan.tables.ends[name]
        refs, own = {field}, set(cm.layout.axes[field])
        spec = cm.transport[field].get(ax)
        if spec is not None:
            for _, e in spec["terms"]:
                refs |= references(plan, e)
            b = spec["ends"][side]
            refs |= references(plan, b.value) | references(plan, b.h)
    out: dict[str, set[str]] = {}
    for r in refs:
        if plan.tables.kind[r] in ("integral", "end"):
            if r in _seen or r == name:  # pragma: no cover - the validator refuses cycles
                continue
            for src, axes in quantity_sources(cm, r, _seen | {name}).items():
                out.setdefault(src, set()).update(axes, own)
        elif plan.tables.kind[r] == "field" and name not in plan.integrals:
            out.setdefault(r, set()).update(cm.layout.axes[r])  # conservative: all of it
        else:
            out.setdefault(r, set()).update(own)
    return out


# ----------------------------------------------------------------------------- dependencies


def field_dependencies(cm) -> dict[str, dict[str, dict[str, Reach]]]:
    """For each field: ``{source: {axis: Reach}}``, sources being fields, compartments and
    integrals its equation uses (integrals held fixed: they are lifted)."""
    plan = cm.plan
    deps: dict[str, dict[str, dict[str, Reach]]] = {}
    for f in cm.ir.fields:
        d: dict[str, dict[str, Reach]] = {}

        def add(sources: Iterable[str], reach: dict[str, Reach], d=d) -> None:
            # every axis is recorded: an axis not named means "the same cell" (or, for a
            # source without that axis, every cell along it)
            for s in sources:
                per = d.setdefault(s, {})
                for ax in plan.frame.axes:
                    per[ax] = merge(per.get(ax), reach.get(ax, Reach()))

        name = f.name
        for (t, e), pieces in zip(cm.field_terms[name], cm.term_parts[name], strict=True):
            if t.kind == "reaction":
                for _, p in pieces:
                    add(references(plan, p), {})
            elif t.kind == "hazard":
                add({name}, {})
                for _, p in pieces:
                    add(references(plan, p), {})
            elif t.kind in ("switch", "mutate"):
                along, rates = cm.generators[t.transitions]
                full = {a: Reach(band=FULL) for a in along}
                for _, _, r in rates:
                    add(references(plan, r), full)
                add({name} if t.kind == "switch" else references(plan, e), full)
        for ax, spec in cm.transport[name].items():
            sharp = any(t.kind == "advection" and t.scheme == "sharp" for t, _ in spec["terms"])
            w = 2 if sharp else 1
            along = {ax: Reach(band=w)}
            add({name}, along)
            for _, e in spec["terms"]:
                add(references(plan, e), along)
            for side, b in spec["ends"].items():
                at_end = {ax: Reach(edge=(side, w))}
                add(references(plan, b.value) | references(plan, b.h), at_end)
        for tr, rate in cm.transfers:  # element by element; slices may pick other categories
            if name not in (tr.source, tr.to):
                continue
            fixed = {a for a, _ in (*tr.source_where, *tr.to_where)}
            across = {a: Reach(band=FULL) for a in fixed}
            add({tr.source}, across)
            add(references(plan, rate), across)
        deps[name] = d
    return deps


# ----------------------------------------------------------------------------- blocks


def _relation(n_row: int | None, n_col: int | None, reach: Reach | None) -> sps.csr_matrix:
    """1-D relation along one axis: which source cells a target cell looks at."""
    if n_row is None and n_col is None:
        return sps.csr_matrix(np.ones((1, 1), dtype=bool))
    if n_row is None:  # source has the axis, target not: only possible through a sum
        return sps.csr_matrix(np.ones((1, n_col), dtype=bool))
    if n_col is None:  # target has the axis, source is constant along it
        rows = np.ones((n_row, 1), dtype=bool)
        if reach is not None and reach.edge is not None and reach.band == 0:
            side, depth = reach.edge
            i = np.arange(n_row)
            rows[:, 0] = i <= depth if side == "lower" else i >= n_row - 1 - depth
        return sps.csr_matrix(rows)
    band = 0 if reach is None or reach.edge is not None else reach.band
    if band == FULL:
        return sps.csr_matrix(np.ones((n_row, n_col), dtype=bool))
    return sps.diags(
        [np.ones(n_row - abs(k), dtype=bool) for k in range(-band, band + 1)],
        list(range(-band, band + 1)),
        shape=(n_row, n_col),
        format="csr",
        dtype=bool,
    )


def frame_to_layout(shape_by_frame: list[int | None], layout_axes, frame_axes) -> np.ndarray:
    """Map a flat index in frame order (axes sorted by the frame) to the flat index in the
    quantity's own axis order."""
    present = [a for a in frame_axes if a in layout_axes]
    sizes = dict(zip(frame_axes, shape_by_frame, strict=True))
    layout_shape = [sizes[a] for a in layout_axes]
    idx = np.arange(int(np.prod(layout_shape, dtype=np.int64))).reshape(layout_shape)
    perm = [list(layout_axes).index(a) for a in present]
    return np.transpose(idx, perm).ravel() if perm else idx.ravel()


@dataclasses.dataclass(frozen=True)
class Node:
    """A block of variables: a field, a compartment, or a lifted integral."""

    name: str
    axes: tuple[str, ...]  # in its own storage order
    offset: int
    size: int


def block(row: Node, col: Node, reach: dict[str, Reach], frame) -> tuple[np.ndarray, np.ndarray]:
    """Global (row, col) indices of a dependency block."""
    mat = sps.csr_matrix(np.ones((1, 1), dtype=bool))
    for ax in frame.axes:
        n = frame.sizes[ax]
        r = _relation(n if ax in row.axes else None, n if ax in col.axes else None, reach.get(ax))
        mat = sps.kron(mat, r, format="csr")
    coo = mat.tocoo()
    rmap = frame_to_layout([frame.sizes[a] for a in frame.axes], row.axes, frame.axes)
    cmap = frame_to_layout([frame.sizes[a] for a in frame.axes], col.axes, frame.axes)
    return row.offset + rmap[coo.row], col.offset + cmap[coo.col]


# ----------------------------------------------------------------------------- colouring


def colour_columns(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Greedy colouring: columns that share a row get different colours, so one forward
    derivative per colour recovers every entry."""
    n_row, n_col = shape
    colours = np.full(n_col, -1, dtype=np.int64)
    if n_col == 0:
        return colours
    P = sps.csr_matrix((np.ones(rows.size, dtype=bool), (rows, cols)), shape=shape)
    P.sum_duplicates()
    Pc = P.tocsc()
    rptr, rind = P.indptr, P.indices
    cptr, cind = Pc.indptr, Pc.indices
    # columns with the most entries first (dense columns get their own colour early)
    order = np.argsort(-np.diff(cptr), kind="stable")
    for j in order:
        rs = cind[cptr[j] : cptr[j + 1]]
        if rs.size == 0:
            colours[j] = 0
            continue
        if rs.size == 1:
            nb = rind[rptr[rs[0]] : rptr[rs[0] + 1]]
        else:
            nb = np.concatenate([rind[rptr[r] : rptr[r + 1]] for r in rs])
        used = colours[nb]
        used = np.unique(used[used >= 0])
        c = 0
        for u in used:  # smallest colour not used by a neighbour
            if u != c:
                break
            c += 1
        colours[j] = c
    return colours


def check_colouring(rows, cols, colours) -> bool:
    """No row has two entries of the same colour (each compressed entry is one derivative)."""
    pairs = np.unique(np.stack([rows, cols]), axis=1)
    key = pairs[0].astype(np.int64) * (int(colours.max(initial=0)) + 1) + colours[pairs[1]]
    return np.unique(key).size == key.size


__all__ = [
    "FULL",
    "Node",
    "Reach",
    "block",
    "check_colouring",
    "colour_columns",
    "field_dependencies",
    "integral_sources",
    "quantity_sources",
    "references",
]
