# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Solution of a model run."""

from __future__ import annotations

import dataclasses

import numpy as np

from tenqsp.ledger import Ledger


@dataclasses.dataclass
class Solution:
    """Trajectories of a solved model.

    - ``t``: output times, shape ``(n_t,)``;
    - ``sol["B"]``: a field, shape ``(n_t, *its axis sizes)`` in the field's axis order, or a
      compartment, shape ``(n_t,)``;
    - ``sol["burst_rate"]``: an observable, shape ``(n_t,)`` or ``(n_t, *sizes)`` if it varies
      over axes (``sol.observable_axes[name]``);
    - ``y``: the flat state, shape ``(n_t, n_states)``;
    - ``stats``: solver counters and wall time; ``messages``: notes from the solver (e.g. a
      segment retried with Radau); ``options``: the solver settings used (``method``,
      ``rtol``, ``atol``, ``max_step``), read by ``m.diagnostics(sol)``;
    - ``ledger``: the conservation ledger (``Ledger``), unless the solve
      was run with ``ledger=False``;
    - ``extinctions``: ``(time, group)`` for each extinction group set to zero;
    - ``doses``: ``(time, target, amount, rise)`` for each bolus given in the run (the model's
      doses or the regimen that replaced them); ``rise`` is how much the dose raised the
      target state's value (the amount divided by the cell width, weight or volume);
    - ``infusions``: ``(start, stop, target, rate)`` for each infusion in the run, as it
      arrived (lag and bioavailability applied; the stop clipped to the end of the run);
    - ``coords``: per axis, its cell centres (continuous) or labels (categorical), with kind,
      units and, for continuous axes, cell faces; ``units``: per field and compartment;
    - ``to_xarray()``: everything above as an ``xarray.Dataset`` (needs the optional xarray).

    At a dose time the state is reported just *after* the dose.
    """

    t: np.ndarray
    y: np.ndarray
    states: dict[str, np.ndarray]
    observables: dict[str, np.ndarray]
    observable_axes: dict[str, tuple[str, ...]]
    axes: dict[str, tuple[str, ...]]
    theta: np.ndarray
    method: str
    stats: dict[str, float]
    messages: list[str] = dataclasses.field(default_factory=list)
    ledger: Ledger | None = None
    extinctions: list[tuple[float, str]] = dataclasses.field(default_factory=list)
    doses: list[tuple[float, str, float, float]] = dataclasses.field(default_factory=list)
    infusions: list[tuple[float, float, str, float]] = dataclasses.field(default_factory=list)
    coords: dict[str, dict] = dataclasses.field(default_factory=dict)
    units: dict[str, str] = dataclasses.field(default_factory=dict)
    model: str = ""
    options: dict = dataclasses.field(default_factory=dict)

    def __getitem__(self, name: str) -> np.ndarray:
        if name in self.states:
            return self.states[name]
        if name in self.observables:
            return self.observables[name]
        known = sorted([*self.states, *self.observables])
        raise KeyError(f"No field, compartment or observable named '{name}'; there are {known}.")

    def __contains__(self, name: str) -> bool:
        return name in self.states or name in self.observables

    def at(self, name: str, t: float) -> np.ndarray:
        """Value at an output time (must be one of ``sol.t``)."""
        i = np.flatnonzero(np.isclose(self.t, t, rtol=0, atol=1e-12 * max(1.0, abs(t))))
        if i.size == 0:
            raise KeyError(f"t = {t} is not an output time; request it with times=[...].")
        return self[name][i[-1]]

    def to_xarray(self):
        """The solution as an ``xarray.Dataset``.

        Each field is a variable over ``("t", *its axes)``, each compartment over ``("t",)``,
        each observable over ``("t", *the axes it varies over)``. Continuous axes carry their
        cell centres as coordinates (attributes: ``kind``, ``units``, ``faces``) and a
        ``<axis>_width`` coordinate (the widths are also in the axis attribute ``widths``, and
        the coordinate is left out if the model uses that name); categorical axes carry their
        labels. Field values are densities (or concentrations for weighted fields), exactly as
        ``sol[name]``.
        """
        try:
            import xarray as xr
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError(
                "to_xarray needs the optional dependency xarray: "
                'python -m pip install -e ".[xarray]"'
            ) from e
        coords: dict[str, tuple] = {"t": ("t", np.asarray(self.t), {"long_name": "time"})}
        for ax, info in self.coords.items():
            attrs = {"kind": info["kind"], "units": info["units"]}
            if info["kind"] == "categorical":
                coords[ax] = (ax, np.asarray(info["values"], dtype=object), attrs)
            else:
                faces = np.asarray(info["faces"])
                widths = np.diff(faces)
                coords[ax] = (
                    ax,
                    np.asarray(info["values"]),
                    {**attrs, "faces": faces, "widths": widths},
                )
                if f"{ax}_width" not in self:
                    coords[f"{ax}_width"] = (ax, widths, {"units": info["units"]})
        data = {}
        for name, v in self.states.items():
            dims = ("t", *self.axes.get(name, ()))
            data[name] = (dims, np.asarray(v), {"units": self.units.get(name, "")})
        for name, v in self.observables.items():
            data[name] = (("t", *self.observable_axes.get(name, ())), np.asarray(v), {})
        attrs = {"model": self.model, "method": self.method}
        return xr.Dataset(data, coords=coords, attrs=attrs)

    def __repr__(self) -> str:
        names = ", ".join([*self.states, *self.observables])
        return (
            f"<Solution: {self.t.size} times in [{self.t[0]:g}, {self.t[-1]:g}], {self.method}; "
            f"{names}>"
        )
