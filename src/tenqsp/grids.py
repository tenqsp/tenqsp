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
    units
        Free-text units, for documentation and reports.
    """

    name: str
    faces: np.ndarray
    kind: AxisKind = "bidirectional"
    units: str = ""
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
    ) -> ContinuousAxis:
        """``n`` equal cells between ``lower`` and ``upper``."""
        if n < 1:
            raise ValueError(f"Axis '{name}': n must be at least 1, not {n}.")
        return cls(name, np.linspace(lower, upper, n + 1), kind=kind, units=units)

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
        return cls(name, faces, kind=kind, units=units)

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
    def is_uniform(self) -> bool:
        return bool(np.allclose(self.widths, self.widths[0], rtol=1e-12, atol=0.0))

    def integral(self, values, axis: int = 0):
        """Quadrature over this axis: ``sum(value * width)`` along array dimension ``axis``.

        Exact for cell averages: it returns the total amount in the domain.
        """
        import jax.numpy as jnp

        shape = [1] * jnp.ndim(values)
        shape[axis] = self.n
        return jnp.sum(values * jnp.asarray(self.widths).reshape(shape), axis=axis)

    def __repr__(self) -> str:
        grid = "uniform" if self.is_uniform else "nonuniform"
        return (
            f"ContinuousAxis('{self.name}', [{self.lower:g}, {self.upper:g}], n={self.n}, "
            f"{grid}, kind='{self.kind}'{', units=' + repr(self.units) if self.units else ''})"
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
