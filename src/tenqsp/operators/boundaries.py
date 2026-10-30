# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Boundary-face fluxes: the only place boundary conditions enter.

Each function returns the flux through the lower or upper face, positive in the direction of
increasing coordinate (so a positive flux at the *lower* face enters the domain, and a
positive flux at the *upper* face leaves it). The result has the field's shape without the
axis dimension.

Zero-flux needs no function (flux 0). An inflow / renewal condition is a flux supplied
directly by the model (e.g. ``k*B*P`` entering infection age 0) and also needs no function:
at the lower face it is used as given, at the upper face with a minus sign.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

from tenqsp.operators.core import to_front

Side = Literal["lower", "upper"]


def _edge(u, dim: int, side: Side):
    uf = to_front(u, dim)
    return uf[0] if side == "lower" else uf[-1]


def advective_outflow_flux(u, v_face, *, dim: int, side: Side):
    """Material carried out across a boundary face by the flow (upwind value of the edge cell).

    Only flow *leaving* the domain is counted: at the upper face ``max(v, 0) * u_last``, at the
    lower face ``min(v, 0) * u_first``. Flow pointing into the domain contributes nothing here
    (supply it as an inflow if material should enter).
    """
    ue = _edge(u, dim, side)
    if side == "upper":
        return jnp.maximum(v_face, 0.0) * ue
    return jnp.minimum(v_face, 0.0) * ue


def dirichlet_diffusive_flux(u, value, D_edge, *, dim: int, side: Side, width_edge: float):
    """Diffusive flux when the field is held at ``value`` on the boundary itself.

    The gradient is taken over the half-cell between the boundary and the edge cell centre.
    """
    ue = _edge(u, dim, side)
    half = 0.5 * width_edge
    if side == "lower":
        return -D_edge * (ue - value) / half
    return -D_edge * (value - ue) / half


def robin_flux(
    u,
    h,
    outside,
    *,
    dim: int,
    side: Side,
    D_edge=None,
    width_edge: float | None = None,
    v_face=None,
):
    """Exchange with an outside reservoir at concentration ``outside`` (e.g. biofilm surface
    exchanging with the bulk liquid), with exchange coefficient ``h``.

    If the field also diffuses along the axis (``D_edge`` and ``width_edge`` given), the
    exchange and the half-cell between the edge cell centre and the boundary act as
    resistances in series:

        flux into domain = (outside - u_edge) / (1/h + (width_edge/2) / D_edge)
                         = (outside - u_edge) * h g / (h + g),   g = D_edge / (width_edge/2)

    This is second-order accurate; using ``u_edge`` with ``1/h`` alone would be first order.
    Without diffusion (``D_edge`` omitted) the flux into the domain is ``h * (outside - u_edge)``.

    With flow across the end as well (``v_face``, the face velocity, together with diffusion),
    the result is the *total* flux: flow entering carries the outside level, flow leaving
    carries the boundary value ``c_f``, and the exchange acts on ``c_f``. Balancing the
    half-cell (``g = D_edge / (width_edge/2)``) gives

        c_f = ((v_in + h) * outside + g * u_edge) / (v_in + h + g)
        flux into domain = v_in * outside - v_out * c_f + h * (outside - c_f)

    with ``v_in``, ``v_out`` the inward and outward parts of the velocity. With ``v = 0`` this
    is the series resistance above. (Without diffusion there is no half-cell balance; the
    caller adds the upwind advective flux, see ``upwind_end_flux``.)
    """
    ue = _edge(u, dim, side)
    if v_face is not None:
        if D_edge is None or width_edge is None:
            raise ValueError("robin_flux: v_face needs D_edge and width_edge.")
        v_into = v_face if side == "lower" else -v_face
        v_in, v_out = jnp.maximum(v_into, 0.0), jnp.maximum(-v_into, 0.0)
        g = D_edge / (0.5 * width_edge)
        den = v_in + h + g
        ok = den > 0
        safe = jnp.where(ok, den, 1.0)  # all three zero: a closed end, c_f = u_edge
        cf = jnp.where(ok, ((v_in + h) * outside + g * ue) / safe, ue)
        into_domain = v_in * outside - v_out * cf + h * (outside - cf)
        return into_domain if side == "lower" else -into_domain
    if D_edge is None:
        into_domain = h * (outside - ue)
    else:
        if width_edge is None:
            raise ValueError("robin_flux: give width_edge together with D_edge.")
        # conductances in series, h g / (h + g): written without 1/h, so h may reach 0 (e.g.
        # a coefficient proportional to a state) with a finite derivative
        g = D_edge / (0.5 * width_edge)
        den = h + g
        ok = den > 0
        into_domain = jnp.where(ok, h * g / jnp.where(ok, den, 1.0), 0.0) * (outside - ue)
    return into_domain if side == "lower" else -into_domain


def upwind_end_flux(u, outside, velocities, *, dim: int, side: Side, idx: int):
    """Advective flux through an end held at (Dirichlet) or exchanging with (Robin) an outside
    level ``outside``: flow entering the domain carries the outside level, flow leaving carries
    the edge cell's value (upwind). ``velocities`` is a list of ``(scheme, v_faces)``; the
    boundary face is index ``idx`` (0 or n) along ``dim``. Positive along the axis."""
    ue = _edge(u, dim, side)
    F = jnp.zeros(jnp.shape(ue))
    for _, v in velocities:
        ve = jnp.take(v, idx, axis=dim)
        if side == "lower":
            F = F + jnp.maximum(ve, 0.0) * outside + jnp.minimum(ve, 0.0) * ue
        else:
            F = F + jnp.maximum(ve, 0.0) * ue + jnp.minimum(ve, 0.0) * outside
    return F
