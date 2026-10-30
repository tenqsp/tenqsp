# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers: moving the operated axis to the front, assembling faces, divergence."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def to_front(x, dim: int):
    """Move array dimension ``dim`` to position 0 (no-op for 0-d or 1-d inputs on dim 0)."""
    return jnp.moveaxis(x, dim, 0) if jnp.ndim(x) > 1 else x


def from_front(x, dim: int):
    return jnp.moveaxis(x, 0, dim) if jnp.ndim(x) > 1 else x


def along(values, dim: int, ndim: int):
    """Reshape a 1-D array so it lies along ``dim`` of an ``ndim``-dimensional field."""
    shape = [1] * ndim
    shape[dim] = -1
    return jnp.reshape(jnp.asarray(values), shape)


def assemble_faces(lower, interior, upper, dim: int):
    """Stack ``lower`` face, ``n-1`` interior faces and ``upper`` face into ``n+1`` faces.

    ``lower`` and ``upper`` have the field's shape without ``dim`` (or broadcast to it);
    ``interior`` has the field's shape with ``n-1`` along ``dim``.
    """
    interior_f = to_front(interior, dim)
    rest = interior_f.shape[1:]
    lo = jnp.broadcast_to(lower, rest)[None]
    up = jnp.broadcast_to(upper, rest)[None]
    return from_front(jnp.concatenate([lo, interior_f, up], axis=0), dim)


def divergence(face_flux, widths: np.ndarray, dim: int):
    """Rate of change of cell averages from face fluxes: ``-(F[i+1] - F[i]) / width[i]``."""
    F = to_front(face_flux, dim)
    w = jnp.asarray(widths).reshape((-1,) + (1,) * (F.ndim - 1))
    return from_front(-(F[1:] - F[:-1]) / w, dim)
