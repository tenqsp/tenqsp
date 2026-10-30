# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The JAX backend's sparse tier (step 4.5; ADR 0001 decision 3): the iteration matrix
``I - c J`` factorised by SuperLU on the host, called from inside the jitted program.

The Jacobian's parts are computed in JAX (``CompiledModel.jacobian_pieces``: the same
coloured passes as the scipy backend), then assembled and factorised on the host
(``CompiledModel.assemble_sparse``, ``scipy.sparse.linalg.splu``), exactly as scipy's BDF
does with the scipy backend's Jacobian: the same matrix, the same factorisation, so the two
backends take the same steps (V9).

- In the run (``linear_solver``, for ``bdf``), a
  factorisation lives on the host and the program holds a **handle** to it (an integer):
  ``factor`` makes one, ``solve`` uses it. The program reuses a factorisation over many
  steps as scipy does. A factorisation is freed when its run (or member of a batch)
  replaces it; one per member is live.
- In the replay (``solve_step``), each step's derivative solves with its own matrix:
  factorised and solved in one host call, as a ``jax.lax.custom_linear_solve`` whose
  transpose (reverse mode) solves with the transposed factors. Several right-hand sides
  (``jacfwd``) share one factorisation.

Host calls (``jax.pure_callback``) run on the CPU. Under ``jax.vmap`` (step 4.6) one call
takes the whole batch and works through its members one after another.

Every factorisation kept between host calls is made and freed on one Python thread
(``_on_owner``). With scipy 1.18 a ``SuperLU`` made in one host call and freed in
another never returns its memory (~2 MB at 500 states): JAX runs host calls on its own
threads, whose Python state lasts one call, even when the same thread makes and frees it
(found at step 4.7: 57 GB for 1,000 members at 505 states). Solves, which neither make
nor free one, run where JAX calls them. The cost, on a machine whose idle cores run at a
low clock (the ``ondemand`` governor): one run 1.1-1.6x slower (none on one core, none
for batches).
"""

from __future__ import annotations

import itertools
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu

from tenqsp._threads import limited
from tenqsp.backends.jax_bdf import LinearSolver, when


class _Owner:
    """The one thread that makes and frees the kept factorisations (see the module
    docstring). Made on first use, and again in a child process after ``os.fork`` (the
    parent's thread does not exist there)."""

    def __init__(self):
        self._pool: ThreadPoolExecutor | None = None
        self._ident: int | None = None
        self._lock = threading.Lock()

    def reset(self):
        self._pool, self._ident, self._lock = None, None, threading.Lock()

    def run(self, fn, *args):
        if threading.get_ident() == self._ident:
            return fn(*args)
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tenqsp-superlu")
                self._ident = self._pool.submit(threading.get_ident).result()
            pool = self._pool
        try:
            future = pool.submit(fn, *args)
        except RuntimeError:  # interpreter shutting down: no new work for threads
            return fn(*args)
        return future.result()


_OWNER = _Owner()


def _on_owner(fn, *args):
    """``fn(*args)`` on the owner thread."""
    return _OWNER.run(fn, *args)


class _Registry:
    """Host factorisations by handle, one per member of a running integration: made when a
    member needs one, freed when it replaces it and when its integration ends
    (``LinearSolver.release``), always on the owner thread. Nothing else frees one, so runs
    in several Python threads at once never lose each other's."""

    def __init__(self):
        self._lu: dict[int, object] = {}
        self._next = itertools.count(1)
        self._lock = threading.Lock()

    def add(self, lu) -> int:
        with self._lock:
            h = next(self._next)
            self._lu[h] = lu
            return h

    def get(self, h: int):
        with self._lock:
            return self._lu.get(h)

    def drop(self, h: int):
        with self._lock:
            self._lu.pop(h, None)

    def clear(self):
        def clear():
            with self._lock:
                self._lu.clear()

        _on_owner(clear)

    def __len__(self):
        return len(self._lu)


REGISTRY = _Registry()


def _after_fork_in_child():  # the parent's thread and handles mean nothing here
    _OWNER.reset()
    REGISTRY._lu = {}
    REGISTRY._lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def _matrix(cm, ledger, pieces, c):
    J = cm.assemble_sparse(pieces, ledger=ledger)
    return (sps.eye_array(J.shape[0], format="csc") - float(c) * J).tocsc()


class _Failed:
    """A matrix that could not be factorised (not finite, or singular): every solve gives
    NaN, as a dense LU of such a matrix does, so the solver treats it as a failed Newton
    iteration (a fresh Jacobian, then a smaller step) or reports the failure."""

    def solve(self, b, trans="N"):
        return np.full(np.shape(b), np.nan)


def _splu(M):
    if not np.all(np.isfinite(M.data)):
        return _Failed()
    try:
        return splu(M)
    except RuntimeError:  # "Factor is exactly singular"
        return _Failed()


def linear_solver(cm, *, ledger: bool) -> LinearSolver:
    """The sparse tier's ``LinearSolver`` for ``jax_bdf.bdf``: ``jac`` returns
    ``CompiledModel.jacobian_pieces``; ``F`` is a handle to a host factorisation.

    Under ``jax.vmap`` (step 4.6) each host call takes the whole batch: one call per Newton
    iteration rather than one per member, and only the members that need a new
    factorisation get one (``refactor``). (Threads do not help: measured, step 4.6.)"""
    n_all = cm.n_states + (len(cm.ledger_entries) if ledger else 0)

    @limited
    def factor_host(c, *pieces):
        return np.int64(REGISTRY.add(_splu(_matrix(cm, ledger, pieces, c))))

    @limited
    def refactor_host(need, F, c, *pieces):
        # ``broadcast_all``: every argument carries the batch's leading dimensions
        need, F, c = np.asarray(need), np.asarray(F), np.asarray(c)
        lead = need.shape
        size = int(np.prod(lead, dtype=np.int64))
        flat = [np.asarray(p).reshape(size, *np.shape(p)[len(lead) :]) for p in pieces]
        need, F, c = need.reshape(size), F.reshape(size).copy(), c.reshape(size)
        for i in np.flatnonzero(need):
            lu = _splu(_matrix(cm, ledger, [p[i] for p in flat], c[i]))
            REGISTRY.drop(int(F[i]))  # replaced: nothing holds it any more
            F[i] = REGISTRY.add(lu)
        return F.reshape(lead)

    @limited
    def solve_host(F, b):
        F = np.asarray(F)
        lead = F.shape
        flat_F, flat_b = F.reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1, n_all)
        out = np.empty_like(flat_b)
        for i, h in enumerate(flat_F):
            lu = REGISTRY.get(int(h))
            # freed: only a done member of a batch, whose result is not used
            out[i] = np.nan if lu is None else lu.solve(flat_b[i])
        return out.reshape(*lead, n_all)

    def factor(J, c):
        return jax.pure_callback(
            lambda *args: _on_owner(factor_host, *args),
            jax.ShapeDtypeStruct((), jnp.int64),
            c,
            *J,
            vmap_method="sequential",
        )

    def refactor(F, J, c, need):
        def call():
            return jax.pure_callback(
                lambda *args: _on_owner(refactor_host, *args),
                jax.ShapeDtypeStruct((), jnp.int64),
                need,
                F,
                c,
                *J,
                vmap_method="broadcast_all",
            )

        return when(need, call, lambda: F)

    def solve(F, b):
        return jax.pure_callback(
            solve_host,
            jax.ShapeDtypeStruct((n_all,), jnp.float64),
            F,
            b,
            vmap_method="broadcast_all",
        )

    def release_host(F):
        F = np.asarray(F)
        for h in F.reshape(-1):
            REGISTRY.drop(int(h))  # the integration has ended: nothing will use it
        return np.zeros(F.shape, dtype=np.int64)

    def release(F):
        return jax.pure_callback(
            lambda F: _on_owner(release_host, F),
            jax.ShapeDtypeStruct((), jnp.int64),
            F,
            vmap_method="broadcast_all",
        )

    return LinearSolver(
        factor=factor,
        solve=solve,
        empty=lambda J: jnp.int64(0),
        refactor=refactor,
        release=release,
    )


def solve_step(cm, pieces, c, b, matvec):
    """``(I - c J) x = b`` for one replayed step, differentiable in ``b`` (forward and
    reverse mode): factorised and solved on the host in one call; the transpose solves with
    the transposed factors. ``matvec(x) = (I - c J) x`` in JAX (for JAX's bookkeeping).
    ``b`` may carry leading batch dimensions (several right-hand sides, one factorisation)."""
    n = cm.n_states

    @limited
    def host(trans, c, b, *pieces):
        # under vmap (``expand_dims``) every argument has the same number of leading batch
        # dimensions, of size 1 where it is not batched
        b, c = np.asarray(b, dtype=np.float64), np.asarray(c, dtype=np.float64)
        pieces = [np.asarray(p, dtype=np.float64) for p in pieces]
        leads = [p.shape[: p.ndim - k] for p, k in zip(pieces, base_ndim, strict=True)]
        leads.append(c.shape)
        lead = np.broadcast_shapes(b.shape[:-1], *leads)
        b = np.broadcast_to(b, (*lead, n))
        if all(int(np.prod(sh)) == 1 for sh in leads):  # one matrix for every right-hand side
            one = [p.reshape(p.shape[p.ndim - k :]) for p, k in zip(pieces, base_ndim, strict=True)]
            lu = _splu(_matrix(cm, False, one, c.reshape(())))
            return lu.solve(b.reshape(-1, n).T, trans=trans).T.reshape(b.shape)
        out = np.empty(b.shape)
        for idx in np.ndindex(*lead):  # a batch of matrices (vmap over parameters)

            def at(a, k, idx=idx):
                a = np.broadcast_to(a, (*lead, *a.shape[a.ndim - k :]))
                return a[idx]

            p_i = [at(p, k) for p, k in zip(pieces, base_ndim, strict=True)]
            out[idx] = _splu(_matrix(cm, False, p_i, at(c, 0))).solve(b[idx], trans=trans)
        return out

    base_ndim = [np.ndim(p) for p in pieces]

    def call(trans):
        def run(_matvec, rhs):
            return jax.pure_callback(
                lambda c, r, *p: host(trans, c, r, *p),
                jax.ShapeDtypeStruct(rhs.shape, jnp.float64),
                c,
                rhs,
                *pieces,
                vmap_method="expand_dims",
            )

        return run

    return jax.lax.custom_linear_solve(
        matvec, b, solve=call("N"), transpose_solve=call("T"), symmetric=False
    )
