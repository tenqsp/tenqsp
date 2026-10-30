# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Integrator backends. Phase 1: ``scipy`` (BDF, Radau fallback). The differentiable JAX
backend follows in Phase 4 (ADR 0001)."""

BACKENDS = ("scipy",)
