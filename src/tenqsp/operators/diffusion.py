# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Diffusion: spreading down a gradient.

Interior face flux, positive in the direction of increasing coordinate:

    F = -D_face * (u_right - u_left) / (distance between the two cell centres)

When ``D`` varies, the two half-cells either side of a face act like resistances in series,
so ``D_face`` is their harmonic (series) average:

    distance / D_face = (w_left / 2) / D_left + (w_right / 2) / D_right

A cell with ``D = 0`` blocks the face completely.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tenqsp.operators.core import from_front, to_front


def diffusive_flux_interior(u, D, *, dim: int, widths: np.ndarray):
    """Diffusive flux at the ``n-1`` interior faces of the axis on dimension ``dim``.

    Parameters
    ----------
    u
        Field of cell averages; the axis occupies dimension ``dim`` with ``n`` cells.
    D
        Diffusivity: a scalar, or cell values (length ``n`` along ``dim``, broadcastable
        elsewhere). A cell with ``D == 0`` blocks its faces. Values are never altered: a
        negative ``D`` is used as given (anti-diffusion, ill-posed), not clipped.
    widths
        Cell widths of the axis (length ``n``).
    """
    uf = to_front(u, dim)
    w = jnp.asarray(widths).reshape((-1,) + (1,) * (uf.ndim - 1))
    if jnp.ndim(D) == 0:
        dc = 0.5 * (w[:-1] + w[1:])
        conduct = D / dc
    else:
        Df = (
            to_front(jnp.asarray(D), dim)
            if jnp.ndim(D) > 1
            else jnp.reshape(jnp.asarray(D), (-1,) + (1,) * (uf.ndim - 1))
        )
        dl, dr = Df[:-1], Df[1:]
        both = (dl != 0) & (dr != 0)
        resist = 0.5 * w[:-1] / jnp.where(both, dl, 1.0) + 0.5 * w[1:] / jnp.where(both, dr, 1.0)
        conduct = jnp.where(both, 1.0 / resist, 0.0)
    F = -conduct * (uf[1:] - uf[:-1])
    return from_front(F, dim)
