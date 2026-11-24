# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Advection: transport at a velocity along an axis.

Flux at each interior face, with face velocity ``v``:

    F = max(v, 0) * u_left_state + min(v, 0) * u_right_state

i.e. take the value from the side the flow comes from ("upwind"). Velocity may change sign
in space or time; no branching is needed.

Schemes
-------
``"erlang"``
    First-order upwind: the face state is the neighbouring cell average. With constant
    velocity on ``N`` equal cells this is exactly an ``N``-compartment transit chain, so
    the numerical spread *is* the model's variability (residence time ~ Erlang/gamma with
    CV = 1/sqrt(N)). Monotone and positivity-preserving.
``"sharp"``
    MUSCL reconstruction with the van Leer limiter: piecewise-linear cell profiles whose
    slopes are limited so face states stay between neighbouring averages (no new extrema,
    so positivity is kept). Second order where the solution is smooth; refining the grid
    converges to undispersed transport.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tenqsp.operators.core import from_front, to_front

SCHEMES = ("erlang", "sharp")


def _van_leer(gl, gr):
    same_sign = gl * gr > 0
    denom = jnp.where(same_sign, gl + gr, 1.0)  # safe: avoids 0/0 in the unused branch
    return jnp.where(same_sign, 2.0 * gl * gr / denom, 0.0)


def _van_leer_slopes(u, centers, faces, lower_value, upper_value, one_sided=(False, False)):
    """Limited slope in every cell; ``u`` has the axis first.

    An end cell gets a limited slope when the boundary value on its outer face is known (an
    inflow without diffusion, a Dirichlet end). Where the end value is not known but flow may
    enter there (an inflow with diffusion, a Robin end: ``one_sided``), the slope is limited
    from the two nearest interior gradients, which keeps second order there. Otherwise the
    slope is zero (first order at that boundary).
    """
    shape = (-1,) + (1,) * (u.ndim - 1)
    centers, faces = jnp.asarray(centers), jnp.asarray(faces)  # traced inside a compiled model
    x = centers.reshape(shape)
    g = (u[1:] - u[:-1]) / (x[1:] - x[:-1])  # gradient across each interior face
    # A cell's face values must stay between its neighbours' values, or an empty cell next to
    # a full one gets a negative face value and a negative flux. With centres midway between
    # faces that holds for the van Leer slope when the cell is no wider than its neighbours
    # (always on a uniform grid); on a nonuniform grid the slope is capped so that it holds:
    # |s| (centre to face) <= |gradient to the neighbour| (centre to centre).
    to_lo = (centers - faces[:-1]).reshape(shape)  # centre to lower face
    to_hi = (faces[1:] - centers).reshape(shape)  # centre to upper face
    span = (centers[1:] - centers[:-1]).reshape(shape)  # centre to centre
    cap_lo = jnp.abs(g) * (span / to_lo[1:])  # cells 1..n-1, from the lower neighbour
    cap_hi = jnp.abs(g) * (span / to_hi[:-1])  # cells 0..n-2, from the upper neighbour

    def capped(s, cap):
        return jnp.sign(s) * jnp.minimum(jnp.abs(s), cap)

    s_mid = capped(capped(_van_leer(g[:-1], g[1:]), cap_lo[:-1]), cap_hi[1:])
    n = u.shape[0]
    if lower_value is not None:
        g_lo = (u[0] - lower_value) / (centers[0] - faces[0])
        s_lo = capped(_van_leer(g_lo, g[0]), cap_hi[0])[None]
    elif one_sided[0] and n >= 3:
        # no lower neighbour: the slope must not take the lower face value below zero, so an
        # empty end cell sends nothing on (its outflow is zero, as for an interior cell)
        s = capped(_van_leer(g[0], g[1]), cap_hi[0])
        s_lo = jnp.where(s > 0, jnp.minimum(s, jnp.maximum(u[0], 0.0) / to_lo[0]), s)[None]
    else:
        s_lo = jnp.zeros_like(u[:1])
    if upper_value is not None:
        g_hi = (upper_value - u[-1]) / (faces[-1] - centers[-1])
        s_hi = capped(_van_leer(g[-1], g_hi), cap_lo[-1])[None]
    elif one_sided[1] and n >= 3:
        s = capped(_van_leer(g[-2], g[-1]), cap_lo[-1])
        s_hi = jnp.where(s < 0, jnp.maximum(s, -jnp.maximum(u[-1], 0.0) / to_hi[-1]), s)[None]
    else:
        s_hi = jnp.zeros_like(u[:1])
    return jnp.concatenate([s_lo, s_mid, s_hi], axis=0)


def advective_flux_interior(
    u,
    v_interior,
    *,
    dim: int,
    centers: np.ndarray,
    faces: np.ndarray,
    scheme: str = "erlang",
    lower_value=None,
    upper_value=None,
    one_sided=(False, False),
):
    """Advective flux at the ``n-1`` interior faces of the axis on dimension ``dim``.

    Parameters
    ----------
    u
        Field of cell averages; the axis occupies dimension ``dim`` with ``n`` cells.
    v_interior
        Velocity at the interior faces: length ``n-1`` along ``dim``, broadcastable elsewhere.
    centers, faces
        Axis geometry (``n`` centres, ``n+1`` faces).
    scheme
        ``"erlang"`` or ``"sharp"`` (see module docstring).
    lower_value, upper_value
        ``"sharp"`` only: the field's value on the lower / upper boundary face, when an inflow
        condition defines it (inflow flux J at face velocity v gives J / v). It lets the end
        cell use a limited slope, keeping second order up to an inflow boundary. Leave as
        ``None`` for zero-flux or outflow boundaries, and wherever diffusion or an exchange
        resistance makes the edge value differ from that.
    one_sided
        ``"sharp"`` only: (lower, upper); where the edge value is unknown, limit the end cell's
        slope from the two nearest interior gradients instead of setting it to zero.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown advection scheme '{scheme}'; use one of {', '.join(SCHEMES)}.")
    uf = to_front(u, dim)
    v = (
        to_front(jnp.asarray(v_interior), dim)
        if jnp.ndim(v_interior) > 1
        else jnp.reshape(jnp.asarray(v_interior), (-1,) + (1,) * (uf.ndim - 1))
    )
    if scheme == "erlang":
        u_l, u_r = uf[:-1], uf[1:]
    else:
        s = _van_leer_slopes(uf, centers, faces, lower_value, upper_value, one_sided)
        x = jnp.asarray(centers).reshape((-1,) + (1,) * (uf.ndim - 1))
        xf = jnp.asarray(faces[1:-1]).reshape((-1,) + (1,) * (uf.ndim - 1))
        u_l = uf[:-1] + s[:-1] * (xf - x[:-1])
        u_r = uf[1:] - s[1:] * (x[1:] - xf)
    F = jnp.maximum(v, 0.0) * u_l + jnp.minimum(v, 0.0) * u_r
    return from_front(F, dim)
