# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Finite-volume operators on one axis of a field.

The pattern for every transport term is:

    face fluxes (interior faces from the operator, boundary faces from the boundary condition)
    -> divergence  du/dt = -(F_right - F_left) / width

Fluxes are positive in the direction of increasing coordinate. Boundary conditions enter only
through the two boundary faces, which makes the conservation ledger exact: what crosses a
boundary face is exactly what entered or left the domain.

All functions are pure ``jax.numpy`` whole-array operations. ``dim`` is the array dimension
of the field that the axis occupies; every other dimension is carried along by broadcasting.
"""

from tenqsp.operators.advection import SCHEMES, advective_flux_interior
from tenqsp.operators.boundaries import (
    advective_outflow_flux,
    dirichlet_diffusive_flux,
    robin_flux,
    upwind_end_flux,
)
from tenqsp.operators.core import assemble_faces, divergence
from tenqsp.operators.diffusion import diffusive_flux_interior

__all__ = [
    "SCHEMES",
    "advective_flux_interior",
    "advective_outflow_flux",
    "assemble_faces",
    "diffusive_flux_interior",
    "dirichlet_diffusive_flux",
    "divergence",
    "robin_flux",
    "upwind_end_flux",
]
