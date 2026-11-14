# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""How many BLAS threads TenQSP's sparse LU uses.

SuperLU (``scipy.sparse.linalg.splu``, the linear solver of the scipy backend and of the
JAX sparse tier) calls BLAS for its dense blocks. With OpenBLAS's default of one thread per
core, its idle threads spin: measured on a 64-core machine, a 9,600-state solve took 54 s
of wall time and 3,400 s of CPU time, against 34 s and 38 s with one thread, and it slowed
everything else on the machine. So TenQSP sets BLAS to ``BLAS_THREADS`` threads while it
factors and solves, and restores the previous setting after (via ``threadpoolctl``).

``tenqsp._threads.BLAS_THREADS = None`` leaves BLAS as it is.

The JAX backend's sparse tier does its host work (factorisations and solves) on one thread
of its own (``jax_sparse``). ``PIN_HOST_THREAD = True`` (the default, Linux only) keeps that
thread on one core, the one it starts on. On a machine whose idle cores run at a low clock
(the ``ondemand`` governor), a thread woken thousands of times a second otherwise lands on
cold cores: measured on a 1,062-state model, each factorisation took twice as long, and the
run 62 s instead of 29 s. Only TenQSP's own thread is pinned; ``False`` releases it.
"""

from __future__ import annotations

import contextlib
import functools

BLAS_THREADS: int | None = 1
PIN_HOST_THREAD: bool = True
_CONTROLLER = None  # made on first use, after scipy has loaded its BLAS


@contextlib.contextmanager
def blas_threads():
    """BLAS limited to ``BLAS_THREADS`` threads inside the block (no effect if None)."""
    global _CONTROLLER
    if BLAS_THREADS is None:
        yield
        return
    if _CONTROLLER is None:
        from threadpoolctl import ThreadpoolController

        _CONTROLLER = ThreadpoolController()
    with _CONTROLLER.limit(limits=BLAS_THREADS, user_api="blas"):
        yield


def limited(fn):
    """``fn`` run inside ``blas_threads``."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with blas_threads():
            return fn(*args, **kwargs)

    return wrapper


def current_cpu() -> int | None:
    """The core the calling thread runs on (Linux), or None where that cannot be read."""
    try:
        import ctypes

        cpu = ctypes.CDLL(None).sched_getcpu()
    except (OSError, AttributeError):
        return None
    return cpu if cpu >= 0 else None
