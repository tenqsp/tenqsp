# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Guards on the order of derivatives: where a second derivative would be wrong, or would
fail with an error that does not say why, these raise one that does.

- ``refuse(x, message)``: ``x``; differentiating through it raises ``message``. Placed
  inside a first-order derivative rule, it lets that rule run and stops a derivative of it.
- ``first_order_only(x, message)``: ``x``, with its first derivative; a second derivative
  through it raises ``message``.
- ``forward_only(x, message)``: ``x``; derivatives through it are taken forward, and a
  reverse-mode derivative of one of them raises ``message``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

THROUGH_EVENTS = (
    "Second derivatives (jax.hessian, or a derivative of jax.grad) are not available for "
    "models with extinction groups or switches: the derivative of an event's time is exact "
    "to first order only. Take central differences of the gradient instead."
)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def refuse(x, message: str):
    return x


@refuse.defjvp
def _refuse_jvp(message, primals, tangents):
    raise ValueError(message)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def first_order_only(x, message: str):
    return x


@first_order_only.defjvp
def _first_order_only_jvp(message, primals, tangents):
    (x,), (dx,) = primals, tangents
    # the factor is 1; it is made from x so that a derivative of this rule meets ``refuse``
    return x, dx * refuse(jnp.ones_like(x) + 0 * x, message)


# An identity whose derivative is itself (it is linear) and whose transpose raises.
_forward_only = Primitive("tenqsp_forward_only")
_forward_only.def_impl(lambda x, *, message: x)
_forward_only.def_abstract_eval(lambda x, *, message: x)
mlir.register_lowering(_forward_only, mlir.lower_fun(lambda x, *, message: x, False))
batching.primitive_batchers[_forward_only] = lambda args, dims, *, message: (
    _forward_only.bind(args[0], message=message),
    dims[0],
)


def _forward_only_jvp(primals, tangents, *, message):
    (x,), (dx,) = primals, tangents
    dx = ad.instantiate_zeros(dx)
    return _forward_only.bind(x, message=message), _forward_only.bind(dx, message=message)


def _forward_only_transpose(cotangent, x, *, message):
    raise ValueError(message)


ad.primitive_jvps[_forward_only] = _forward_only_jvp
ad.primitive_transposes[_forward_only] = _forward_only_transpose


def forward_only(x, message: str):
    return _forward_only.bind(x, message=message)
