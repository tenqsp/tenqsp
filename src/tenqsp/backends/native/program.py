# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The native backend's program for ``sim.function`` (ADR 0011).

What stays in JAX: laying a scenario's inputs out from ``x`` (``gradients._inputs``), the
model's doses, lag and bioavailability, the initial state, and their derivatives along each
entry of ``x``, and the outputs read from the states (``gradients._outputs_of``): small
programs, quick to trace. What runs natively: the solve with its forward sensitivities
(``bdf_template``), reached by ``jax.pure_callback`` inside a ``jax.custom_jvp`` whose
tangent is the sensitivities times the tangent of ``x`` (a linear map, so reverse mode
transposes it). Under ``jax.vmap`` (a batch of scenarios) the callback gets the whole batch
(``vmap_method="broadcast_all"``) and numba runs the scenarios in parallel.

Each model structure gets one module, written to the cache directory (``cache_dir``): its
kernels (``rhs``, ``jac``, ``sens``, translated from its JAX right-hand side) in front of the
solver's source, so that numba caches the compiled solver on disk; a second fit of the same
model loads it instead of compiling (the first compile takes seconds to tens of seconds).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import threading
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tenqsp.backends.native.translate import NotTranslatable, source

__all__ = ["available", "cache_dir", "native_sens_program", "unsupported"]

_TEMPLATE = Path(__file__).with_name("bdf_template.py")
_LOCK = threading.Lock()
_MODULES: dict[str, object] = {}
VERSION = 2  # of the generated modules (2: numpy's error model, inf and NaN as in JAX)


def cache_dir() -> Path:
    """Where the generated modules (and numba's compiled code beside them) are kept:
    ``TENQSP_NATIVE_CACHE``, or ``~/.cache/tenqsp/native``; owner-only permissions (numba
    runs what it finds there)."""
    d = Path(os.environ.get("TENQSP_NATIVE_CACHE") or Path.home() / ".cache" / "tenqsp" / "native")
    d.mkdir(parents=True, exist_ok=True)
    try:
        d.chmod(0o700)
    except OSError:  # pragma: no cover - a directory the user set up otherwise
        pass
    return d


def available() -> bool:
    """Whether numba can be imported."""
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def unsupported(cm, derivatives) -> str | None:
    """Why the native backend cannot run this model (None: it can)."""
    from tenqsp.backends import jax_backend as jb

    if not available():
        return 'numba is not installed (pip install "tenqsp[native]")'
    if derivatives == "reverse":
        return "reverse derivatives (the native backend carries forward sensitivities)"
    if cm.extinction_groups or cm.switch_events:
        return "the model has events (an extinction threshold or a condition on its state)"
    if cm.plan.switch_slot is not None:
        return "the model has conditions on time"
    if cm.floor_slots:
        return "the model has noise floors"
    if jb.tier(cm) == "sparse":
        return f"the model has {cm.n_states} states (the native backend is dense)"
    if cm.n_states == 0:
        return "the model has no states"
    return None


def _key(cm, p: int) -> str:
    """The module's name: the model's structure (``ModelIR.structure_hash``: parameter
    values are inputs of the kernels, not part of them), the number of directions, and the
    sources of the solver and of the translator (a change of either is a new module)."""
    from tenqsp.backends.native import translate

    blob = hashlib.sha256()
    blob.update(f"tenqsp native {VERSION} {p} {cm.n_states} {cm.n_theta}\n".encode())
    blob.update(cm.ir.structure_hash().encode())
    blob.update(_TEMPLATE.read_bytes())
    blob.update(Path(translate.__file__).read_bytes())
    blob.update(Path(__file__).read_bytes())  # writes the module (its decorators)
    blob.update(jax.__version__.encode())
    return blob.hexdigest()[:24]


def _module(cm, p: int):
    """The compiled module of this model's kernels and the solver, for ``p`` directions:
    loaded from the cache directory when written before (no translation), else written."""
    key = _key(cm, p)
    with _LOCK:
        if key in _MODULES:
            return _MODULES[key]
        d = cache_dir()
        path = d / f"tq_{key}.py"
        data = d / f"tq_{key}.npz"
        if not (path.exists() and data.exists()):
            _write(cm, p, path, data)
        name = f"tenqsp_native_{key}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        # registered first: numba's disk cache imports a cached function's module by name
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _MODULES[key] = mod
        return mod


def _write(cm, p: int, path: Path, data: Path) -> None:
    """Translate the model's kernels and write them with the solver to ``path`` (their
    constants to ``data``)."""
    n, nth = cm.n_states, cm.n_theta
    geom = cm.geom
    # example inputs: only their shapes matter (what depends on them is not folded)
    z = jnp.ones(n, dtype=jnp.float64)
    th = jnp.ones(nth, dtype=jnp.float64)
    S = jnp.zeros((p, n))
    TD = jnp.zeros((p, nth))

    def rhs(t, y, theta, g):
        return cm._rhs_impl(t, y, theta, g)

    def jac(t, y, theta, g):
        return jax.jacfwd(cm._rhs_impl, argnums=1)(t, y, theta, g)

    def sens(t, y, theta, Sd, Td, g):
        def along(s_, td_):
            return jax.jvp(lambda yy, tt: cm._rhs_impl(t, yy, tt, g), (y, theta), (s_, td_))[1]

        return jax.vmap(along)(Sd, Td)

    parts, consts = [], {}
    for name, fn, ex, static in (
        ("rhs", rhs, (0.0, z, th, geom), (3,)),
        ("jac", jac, (0.0, z, th, geom), (3,)),
        ("sens", sens, (0.0, z, th, S, TD, geom), (5,)),
    ):
        src, cs = source(fn, *ex, static=static, name=name, prefix=f"{name.upper()}_C")
        parts.append('@njit(cache=True, error_model="numpy")\n' + src)
        consts.update(cs)
    np.savez(data, **consts)
    head = (
        "# Generated by TenQSP's native backend (ADR 0011): a model's kernels and the\n"
        "# solver. Do not edit: the file name is a hash of the model's structure.\n"
        "import numpy as np\nfrom numba import njit\n"
        f"_DATA = np.load({str(data)!r})\n"
        + "".join(f"{k} = _DATA[{k!r}]\n" for k in sorted(consts))
    )
    tmp = path.with_suffix(".tmp")
    # UTF-8 as Python reads source: the cache path in the file may hold any character
    text = head + "\n\n" + "\n\n\n".join(parts) + "\n\n\n" + _TEMPLATE.read_text("utf-8")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def native_sens_program(cm, model_doses: bool, outputs, p: int):
    """``impl(x, data) -> ({output: (T, ...)}, status, most steps in one segment)`` for one
    scenario, as ``gradients._sens_program`` (or ``_forward_program`` with ``p = 0``), solved
    natively; ``jax.vmap`` of it runs a batch in one native call."""
    from tenqsp import gradients as gr
    from tenqsp.backends import jax_backend as jb

    key = ("native_sens_function", model_doses, tuple(outputs), p)
    if key in cm._lazy:
        return cm._lazy[key]
    try:
        mod = _module(cm, max(p, 1))
    except NotTranslatable as e:
        raise NotImplementedError(f"the native backend cannot translate this model: {e}") from e
    outputs_of = gr._outputs_of(cm, outputs)
    n = cm.n_states
    geom = cm.geom
    P = max(p, 1)  # the kernels carry at least one direction (zero when p = 0)

    def prepared(x, data):
        """The solver's inputs at ``x`` and their derivatives along each entry of ``x``
        (scaled by its size, as the JAX program; divided back out after the solve)."""
        args, _ = gr._inputs(x, data, model_doses)

        def moved(x_):
            a, _ = gr._inputs(x_, data, model_doses)
            return {k: a[k] for k in gr._REGIMEN if a[k] is not None}

        def inputs(theta, b_time, b_amount, i_start, i_stop, i_rate):
            measure = cm._measure_impl(theta, geom)
            if model_doses:
                k = b_time.shape[0]
                amounts = cm._dose_impl(theta, geom)
                b_amount = jnp.zeros(k).at[: amounts.shape[0]].set(amounts)
            b_time, i_start, i_stop, rise, i_rise, bad = jb.arrive(
                cm, theta, geom, measure, b_time, args["b_index"], b_amount, i_start, i_stop,
                args["i_index"], i_rate,
            )  # fmt: skip
            y0 = cm._y0_impl(theta, geom)
            if args.get("y_start") is not None:
                y0 = jnp.where(jnp.isnan(args["y_start"]), y0, args["y_start"])
            return (theta, b_time, i_start, i_stop, rise, i_rise, y0), bad

        b_amount = args["b_amount"] if args["b_amount"] is not None else jnp.zeros_like(
            args["b_time"])  # fmt: skip
        primals = (args["theta"], args["b_time"], b_amount, args["i_start"], args["i_stop"],
                   args["i_rate"])  # fmt: skip
        size = jnp.where(x != 0, jnp.abs(x), 1.0)
        if p:
            dot = jax.vmap(lambda e: jax.jvp(moved, (x,), (e,))[1])(jnp.diag(size))
            dot = {k: jnp.where(jnp.isfinite(d), d, 0.0) for k, d in dot.items()}
            names_in = ("theta", "b_time", "b_amount", "i_start", "i_stop", "i_rate")
            tang = tuple(
                jnp.zeros((p,) + v.shape) if dot.get(k) is None else dot[k]
                for k, v in zip(names_in, primals, strict=True)
            )
            got, dgot, bad = jax.vmap(
                lambda tg: jax.jvp(inputs, primals, tg, has_aux=True), out_axes=(None, 0, None)
            )(tang)
            dgot = tuple(jnp.where(jnp.isfinite(d), d, 0.0) for d in dgot)
        else:
            got, bad = inputs(*primals)
            dgot = tuple(jnp.zeros((1,) + v.shape) for v in got)
        return args, got, dgot, bad, size

    def solve(x, data):
        args, got, dgot, bad, size = prepared(x, data)
        theta, b_time, i_start, i_stop, rise, i_rise, y0 = got
        dth, db, dis, dist, drise, dirise, dy0 = dgot
        T = args["t_out"].shape[0]
        ins = (
            theta, dth, y0, dy0, b_time, db, args["b_index"], rise, drise, i_start, dis,
            i_stop, dist, args["i_index"], i_rise, dirise, args["t0"], args["t1"],
            args["t_out"], args["rtol"], args["atol"][:n], args["max_step"], args["max_steps"],
        )  # fmt: skip
        ins = tuple(jnp.asarray(a) for a in ins)
        shapes = (
            jax.ShapeDtypeStruct((T, n), jnp.float64),
            jax.ShapeDtypeStruct((T, P, n), jnp.float64),
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int64),
        )
        ys, S, status, most = jax.pure_callback(
            _host(mod), shapes, *ins, vmap_method="broadcast_all"
        )
        status = jnp.where(bad, jb.BAD_ARRIVAL, status)
        real = jnp.isfinite(data["t_out"])
        S = S[:, :p, :] / size[None, :, None] if p else S[:, :0, :]
        S = jnp.where(real[:, None, None], S, 0.0)
        ys = jnp.where(real[:, None], ys, ys[0])
        return ys, S, status, most

    @jax.custom_jvp
    def states(x, data):
        ys, _, status, most = solve(x, data)
        return ys, status, most

    @states.defjvp
    def states_jvp(primals, tangents):
        x, data = primals
        ys, S, status, most = solve(x, data)
        dys = jnp.einsum("tpn,p->tn", S, tangents[0])
        none = lambda a: np.zeros(a.shape, dtype=jax.dtypes.float0)  # noqa: E731 - counts
        return (ys, status, most), (dys, none(status), none(most))

    def impl(x, data):
        ys, status, most = states(x, data)
        args, _ = gr._inputs(x, data, model_doses)  # theta with x's derivatives, for outputs
        ok = status == jb.OK
        ys = ys * jnp.where(ok, 1.0, jnp.nan) + jnp.where(ok, 0.0, jnp.nan) * jnp.sum(x)
        return outputs_of(args["theta"], ys, data["t_out"]), status, most

    cm._lazy[key] = impl
    return impl


_POOL = None


def _pool():
    """The threads that run a batch's scenarios (``run_one`` releases the GIL), one per core;
    a numba ``prange`` loop did the same but added seconds to every model's first compile."""
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _POOL = ThreadPoolExecutor(max_workers=max(1, os.cpu_count() or 1),
                                   thread_name_prefix="tenqsp-native")  # fmt: skip
    return _POOL


# the rank of each input of ``run_one`` (theta, its derivatives, y0, ..., max_steps): a batch
# (``jax.vmap``, nested or not) adds leading axes to every one
_RANKS = (1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 0, 0, 1, 0, 1, 0, 0)
_INTS = (6, 13, 22)  # b_index, i_index, max_steps


def _host(mod):
    """The callback: numpy arrays in (under ``jax.vmap``, with the same leading batch axes on
    every input), each scenario solved natively on the pool's threads, numpy arrays out
    with those leading axes."""

    def call(*ins):
        ins = [np.asarray(a) for a in ins]
        batch = ins[2].shape[: ins[2].ndim - _RANKS[2]]  # y0's leading axes
        U = int(np.prod(batch)) if batch else 1
        flat = []
        for k, (a, r) in enumerate(zip(ins, _RANKS, strict=True)):
            a = np.broadcast_to(a, batch + a.shape[a.ndim - r :]) if a.ndim - r != len(batch) else a
            a = a.reshape((U,) + a.shape[len(batch) :])
            flat.append(np.ascontiguousarray(a, dtype=np.int64 if k in _INTS else np.float64))

        def one(j):
            args = [f[j] for f in flat]
            for k in (16, 17, 19, 21):  # t0, t1, rtol, max_step: numbers
                args[k] = float(args[k])
            args[22] = int(args[22])
            return mod.run_one(*args)

        outs = list(_pool().map(one, range(U))) if U > 1 else [one(0)]
        YS = np.stack([o[0] for o in outs]).reshape(batch + outs[0][0].shape)
        SS = np.stack([o[1] for o in outs]).reshape(batch + outs[0][1].shape)
        st = np.array([o[2] for o in outs], dtype=np.int64).reshape(batch)
        mo = np.array([o[3] for o in outs], dtype=np.int64).reshape(batch)
        return YS, SS, st, mo

    return call
