# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Phase 0 smoke tests: the package imports and the numeric stack is configured."""

import jax.numpy as jnp

import tenqsp


def test_version():
    assert tenqsp.__version__ == "0.6.0"


def test_float64_enabled():
    assert jnp.zeros(1).dtype == jnp.float64


def test_stack_importable():
    import diffrax  # noqa: F401
    import equinox  # noqa: F401
    import lineax  # noqa: F401
    import scipy.integrate  # noqa: F401
    import sympy  # noqa: F401
