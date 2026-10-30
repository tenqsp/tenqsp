# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Plots (needs the optional matplotlib: ``pip install tenqsp[plot]``).

Each function draws on ``ax`` (a new figure if not given) and returns the matplotlib
``Axes``; it sets labels from the model's names and units, and nothing else (no style).

- ``trajectories``: an output over time, one line per scenario (a result or a batch).
- ``fan``: the median and a band of a batch's scenarios over time, per group.
- ``sensitivity``: a heat map of a sensitivity table, parameter x time.
- ``snapshot``: a field along one of its axes at chosen times.
"""

from __future__ import annotations

import numpy as np


def _axes(ax):
    if ax is not None:
        return ax
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - the optional extra
        raise ImportError(
            "tq.plot needs matplotlib: pip install 'tenqsp[plot]' (or matplotlib)."
        ) from None
    _, ax = plt.subplots()
    return ax


def _scalar_series(values, name):
    """(scenario, time) from an output; a field output must be reduced first."""
    v = np.asarray(values)
    if v.ndim > 2:
        raise ValueError(
            f"'{name}' has values along the model's axes ({v.shape[2:]}); plot an observable "
            "(e.g. a total), or a field with tq.plot.snapshot."
        )
    return v


def trajectories(result, output: str, *, ax=None, log: bool = False, label=None):
    """``output`` over time: one line for a ``SimResult``, one per
    scenario for a ``BatchResult`` (labelled by scenario name when
    there are at most 10). ``log``: a log y axis."""
    from tenqsp.explore import BatchResult

    ax = _axes(ax)
    if isinstance(result, BatchResult):
        v = _scalar_series(result[output], output)
        names = [s.name for s in result.scenarios]
        for k in range(v.shape[0]):
            lab = names[k] if v.shape[0] <= 10 else None
            ax.plot(result.t, v[k], label=lab, lw=1)
        if v.shape[0] <= 10 and any(names):
            ax.legend(fontsize="small")
    else:
        v = np.asarray(result[output])
        if v.ndim > 1:
            raise ValueError(
                f"'{output}' has values along the model's axes ({v.shape[1:]}); plot an "
                "observable, or use tq.plot.snapshot."
            )
        ax.plot(result.t, v, label=label)
        if label:
            ax.legend(fontsize="small")
    ax.set_xlabel("time")
    ax.set_ylabel(output)
    if log:
        ax.set_yscale("log")
    return ax


def fan(batch, output: str, *, by: str | None = None, q=(5, 50, 95), ax=None, log=False):
    """The middle quantile of ``q`` as a line and the outer two as a band, over the batch's
    scenarios at each time; one per group of ``by`` (a varied column, e.g. ``"regimen"``),
    else one for the whole batch."""
    ax = _axes(ax)
    if len(q) != 3 or not q[0] < q[1] < q[2]:
        raise ValueError(f"q takes three increasing percentiles, e.g. (5, 50, 95); not {q!r}.")
    v = _scalar_series(batch[output], output)
    if by is None:
        groups = {"": np.arange(v.shape[0])}
    else:
        if by not in batch.columns:
            raise KeyError(
                f"by='{by}' is not a varied column of the batch; they are {batch.columns}."
            )
        keys = [str(lab.get(by)) for lab in batch.labels]
        groups = {
            k: np.array([i for i, x in enumerate(keys) if x == k]) for k in dict.fromkeys(keys)
        }
    for name, idx in groups.items():
        lo, mid, hi = np.nanpercentile(v[idx], q, axis=0)
        (line,) = ax.plot(batch.t, mid, label=name or None, lw=1.5)
        ax.fill_between(batch.t, lo, hi, color=line.get_color(), alpha=0.25, lw=0)
    if by is not None:
        ax.legend(title=by, fontsize="small")
    ax.set_xlabel("time")
    ax.set_ylabel(f"{output} (median, {q[0]:g}-{q[2]:g}%)")
    if log:
        ax.set_yscale("log")
    return ax


def sensitivity(s, output: str, *, ax=None, cmap="RdBu_r"):
    """A heat map of ``s[output]`` (a ``Sensitivity``): one row per
    name in ``wrt``, one column per output time; colours centred on 0."""
    ax = _axes(ax)
    rows = []
    for name in s.names:
        d = np.asarray(s[output][name])
        if d.ndim > 1:
            raise ValueError(f"'{output}' has values along the model's axes; plot an observable.")
        rows.append(d)
    m = np.vstack(rows)
    lim = np.nanmax(np.abs(m)) if np.isfinite(m).any() else 1.0
    im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_yticks(range(len(s.names)), list(s.names))
    step = max(1, len(s.t) // 8)
    ticks = list(range(0, len(s.t), step))
    ax.set_xticks(ticks, [f"{s.t[i]:g}" for i in ticks])
    ax.set_xlabel("time")
    what = "d log y / d log θ" if s.relative else "d y / d θ"
    ax.figure.colorbar(im, ax=ax, label=f"{output}: {what}")
    return ax


def snapshot(result, field: str, *, axis: str, times, ax=None, log=False):
    """A field along ``axis`` at the output times nearest to ``times`` (one line each),
    summed over its other axes. ``result``: a SimResult or Solution whose field is in it."""
    ax = _axes(ax)
    sol = getattr(result, "solution", result)
    if field not in sol.axes:
        raise KeyError(f"'{field}' is not a field of this result; they are {list(sol.axes)}.")
    axes = sol.axes[field]
    if axis not in axes:
        raise KeyError(f"'{field}' has axes {axes}, not '{axis}'.")
    values = np.asarray(sol[field])  # (time, *axes)
    k = axes.index(axis)
    other = tuple(1 + i for i in range(len(axes)) if i != k)
    along = values.sum(axis=other) if other else values
    coords = sol.coords[axis]
    x = np.asarray(coords["values"])
    categorical = coords.get("kind") == "categorical"
    for t in np.atleast_1d(times):
        i = int(np.argmin(np.abs(sol.t - t)))
        xs = np.arange(len(x)) if categorical else x
        ax.plot(xs, along[i], marker="o" if categorical else None, label=f"t = {sol.t[i]:g}")
    if categorical:
        ax.set_xticks(range(len(x)), [str(v) for v in x])
    units = coords.get("units") or ""
    ax.set_xlabel(f"{axis} ({units})" if units else axis)
    ax.set_ylabel(field)
    ax.legend(fontsize="small")
    if log:
        ax.set_yscale("log")
    return ax
