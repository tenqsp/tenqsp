# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Axes: the discretised dimensions a field lives on.

A *continuous* axis is a line cut into cells. Each cell is bounded by two faces and holds the
cell average of the field (finite volumes). A *categorical* axis is a list of
discrete states (phenotype, genotype, organ) with no geometry.

Faces, centres and widths are stored as arrays, so nonuniform grids work everywhere. Uniform
grids are just the common special case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

AxisKind = Literal["one_way", "bidirectional", "spatial"]
AXIS_KINDS: tuple[str, ...] = ("one_way", "bidirectional", "spatial")
Geometry = Literal["planar", "cylindrical", "spherical"]
GEOMETRIES: tuple[str, ...] = ("planar", "cylindrical", "spherical")
# the power of r in the area of the surface at radius r: the cell volumes and face areas
# follow (planar: per unit area; cylindrical: per unit length)
GEOMETRY_POWER = {"planar": 0, "cylindrical": 1, "spherical": 2}
_GEOMETRY_CONSTANT = {"planar": 1.0, "cylindrical": 2.0 * np.pi, "spherical": 4.0 * np.pi}


def measures(faces, geometry: str = "planar") -> tuple[np.ndarray, np.ndarray]:
    """Face areas (``n + 1``) and cell volumes (``n``) of an axis in a geometry.

    Planar: areas 1 and volumes the widths. Cylindrical (``r`` a radius, per unit length):
    areas ``2*pi*r`` and volumes ``pi*(r1^2 - r0^2)``. Spherical: areas ``4*pi*r^2`` and
    volumes ``4/3*pi*(r1^3 - r0^3)``. The volumes are exact, so totals are amounts.
    """
    f = np.asarray(faces, dtype=np.float64)
    k = GEOMETRY_POWER[geometry]
    c = _GEOMETRY_CONSTANT[geometry]
    if k == 0:
        return np.ones_like(f), np.diff(f)
    return c * f**k, c / (k + 1) * np.diff(f ** (k + 1))


@dataclass(frozen=True, eq=False)
class ContinuousAxis:
    """A continuous axis cut into cells.

    Parameters
    ----------
    name
        Axis name used in formulas (e.g. ``"a"`` for infection age).
    faces
        Increasing face positions, length ``n + 1``. ``faces[0]`` and ``faces[-1]`` are the
        lower and upper ends of the axis.
    kind
        Semantic tag that sets defaults and validation: ``"one_way"``,
        ``"bidirectional"`` or ``"spatial"``.
    geometry
        ``"planar"`` (default), or ``"cylindrical"``/``"spherical"``: the axis is a radius,
        a field is a density per unit volume, and transport and diffusion along it cross
        surfaces whose area grows with it (see :func:`measures`).
    units
        Free-text units, for documentation and reports.
    """

    name: str
    faces: np.ndarray
    kind: AxisKind = "bidirectional"
    units: str = ""
    geometry: Geometry = "planar"
    centers: np.ndarray = field(init=False, repr=False)
    widths: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        faces = np.asarray(self.faces, dtype=np.float64)
        if faces.ndim != 1 or faces.size < 2:
            raise ValueError(
                f"Axis '{self.name}': faces must be a 1-D array with at least 2 entries."
            )
        if not np.all(np.diff(faces) > 0):
            raise ValueError(f"Axis '{self.name}': faces must be strictly increasing.")
        if self.kind not in AXIS_KINDS:
            raise ValueError(
                f"Axis '{self.name}': kind must be one of {', '.join(AXIS_KINDS)}, "
                f"not '{self.kind}'."
            )
        if self.geometry not in GEOMETRIES:
            raise ValueError(
                f"Axis '{self.name}': geometry must be one of {', '.join(GEOMETRIES)}, "
                f"not '{self.geometry}'."
            )
        if self.geometry != "planar" and faces[0] < 0:
            raise ValueError(
                f"Axis '{self.name}': a {self.geometry} axis is a radius, so it starts at 0 or "
                f"above, not at {faces[0]:g}."
            )
        faces.setflags(write=False)
        centers = 0.5 * (faces[1:] + faces[:-1])
        widths = np.diff(faces)
        centers.setflags(write=False)
        widths.setflags(write=False)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "widths", widths)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def uniform(
        cls,
        name: str,
        lower: float,
        upper: float,
        n: int,
        *,
        kind: AxisKind = "bidirectional",
        units: str = "",
        geometry: Geometry = "planar",
    ) -> ContinuousAxis:
        """``n`` equal cells between ``lower`` and ``upper``."""
        if n < 1:
            raise ValueError(f"Axis '{name}': n must be at least 1, not {n}.")
        return cls(
            name, np.linspace(lower, upper, n + 1), kind=kind, units=units, geometry=geometry
        )

    @classmethod
    def geometric(
        cls,
        name: str,
        lower: float,
        upper: float,
        n: int,
        ratio: float,
        *,
        kind: AxisKind = "bidirectional",
        units: str = "",
        geometry: Geometry = "planar",
    ) -> ContinuousAxis:
        """``n`` cells whose widths grow by ``ratio`` from one cell to the next.

        ``ratio < 1`` refines towards ``upper``; ``ratio > 1`` refines towards ``lower``.
        """
        if n < 1 or ratio <= 0:
            raise ValueError(f"Axis '{name}': need n >= 1 and ratio > 0.")
        w = ratio ** np.arange(n)
        w = w / w.sum() * (upper - lower)
        faces = lower + np.concatenate(([0.0], np.cumsum(w)))
        faces[-1] = upper  # exactly the end declared (the sum rounds in the last digit)
        return cls(name, faces, kind=kind, units=units, geometry=geometry)

    # ------------------------------------------------------------------ properties
    @property
    def n(self) -> int:
        """Number of cells."""
        return self.widths.size

    @property
    def lower(self) -> float:
        return float(self.faces[0])

    @property
    def upper(self) -> float:
        return float(self.faces[-1])

    @property
    def center_distances(self) -> np.ndarray:
        """Distance between neighbouring cell centres, one per interior face (length ``n-1``)."""
        return np.diff(self.centers)

    @property
    def volumes(self) -> np.ndarray:
        """Cell volumes in the axis's geometry (the widths on a planar axis)."""
        return measures(self.faces, self.geometry)[1]

    @property
    def areas(self) -> np.ndarray:
        """Face areas in the axis's geometry (ones on a planar axis), length ``n + 1``."""
        return measures(self.faces, self.geometry)[0]

    @property
    def is_uniform(self) -> bool:
        return bool(np.allclose(self.widths, self.widths[0], rtol=1e-12, atol=0.0))

    def integral(self, values, axis: int = 0):
        """Quadrature over this axis: ``sum(value * volume)`` along array dimension ``axis``
        (the volumes are the widths on a planar axis).

        Exact for cell averages: it returns the total amount in the domain.
        """
        import jax.numpy as jnp

        shape = [1] * jnp.ndim(values)
        shape[axis] = self.n
        return jnp.sum(values * jnp.asarray(self.volumes).reshape(shape), axis=axis)

    def __repr__(self) -> str:
        grid = "uniform" if self.is_uniform else "nonuniform"
        return (
            f"ContinuousAxis('{self.name}', [{self.lower:g}, {self.upper:g}], n={self.n}, "
            f"{grid}, kind='{self.kind}'"
            f"{'' if self.geometry == 'planar' else ', geometry=' + repr(self.geometry)}"
            f"{', units=' + repr(self.units) if self.units else ''})"
        )


@dataclass(frozen=True, eq=False)
class CategoricalAxis:
    """A list of discrete states, e.g. ``["S", "R"]`` or organ names."""

    name: str
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        labels = tuple(str(x) for x in self.labels)
        if not labels:
            raise ValueError(f"Axis '{self.name}': a categorical axis needs at least one label.")
        if len(set(labels)) != len(labels):
            raise ValueError(f"Axis '{self.name}': labels must be unique, got {labels}.")
        object.__setattr__(self, "labels", labels)

    @property
    def n(self) -> int:
        return len(self.labels)

    def index(self, label: str) -> int:
        """Position of ``label`` on the axis, with a helpful error if it does not exist."""
        try:
            return self.labels.index(label)
        except ValueError:
            raise KeyError(
                f"Axis '{self.name}' has no category '{label}'; it has {list(self.labels)}."
            ) from None

    def __repr__(self) -> str:
        return f"CategoricalAxis('{self.name}', {list(self.labels)})"
