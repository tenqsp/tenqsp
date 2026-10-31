# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Test-session setup.

Under pytest-xdist each worker is a separate process. Left alone, every worker's XLA and BLAS
would use all cores, oversubscribing the machine (8 workers x 64 threads); one compute thread
per worker is faster overall. This must run before JAX is imported, hence here.
"""

import functools
import os

import pytest

if os.environ.get("PYTEST_XDIST_WORKER"):
    os.environ.setdefault(
        "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")


# ----------------------------------------------------------------- compiled code per worker
# Every XLA:CPU executable holds memory mappings, and Linux allows 65,530 per process
# (vm.max_map_count). A worker running a few hundred tests on the JAX backend kept every
# program it compiled and reached that limit (measured 65,472), after which XLA fails at
# random ("Failed to materialize symbols", "releaseMappedMemory failed: Cannot allocate
# memory") or the worker crashes. So after each test, once a worker holds more than
# MAX_MAPPINGS, its compiled programs are released (they are compiled again when needed).

MAX_MAPPINGS = 40_000


@pytest.fixture(autouse=True)
def _release_compiled_programs():
    yield
    try:
        with open("/proc/self/maps", encoding="utf-8") as f:
            n = sum(1 for _ in f)
    except OSError:  # not Linux: no such limit to watch
        return
    if n > MAX_MAPPINGS:
        import gc

        import jax

        jax.clear_caches()
        gc.collect()


# ----------------------------------------------------------------- checks on every compiled model
# `pytest --roundtrip` (V12): every model a test compiles is also written as text
# (m.to_text()), read back (tq.Model.from_text) and compared in meaning
# (tenqsp.canonical.differences).
# `pytest --jacobian` (V7): for every model a test compiles with at most V7_MAX_STATES states,
# the sparse Jacobian the solver uses equals the dense jax.jacfwd of the same right-hand side,
# with and without the ledger, at the starting state and at a random positive state.
# Failures are collected and reported at the end; both options can be combined.

V7_MAX_STATES = 2_000  # the dense oracle's size
V7_TOL = 1e-10  # max |sparse - dense| <= V7_TOL * max(1, max |dense|)
_CHECKS = {"roundtrip": "V12 round trip", "jacobian": "V7 sparse Jacobian"}


def pytest_addoption(parser):
    parser.addoption(
        "--roundtrip",
        action="store_true",
        help="V12: round-trip every compiled model through the PDE notation",
    )
    parser.addoption(
        "--backend",
        default="scipy",
        choices=("scipy", "jax"),
        help="V9: run every solve on this backend (default scipy; 'jax' runs the whole suite "
        "on the JAX BDF, except runs that ask for method='Radau')",
    )
    parser.addoption(
        "--replay",
        action="store_true",
        help="step 4.4a: with --backend=jax, record and replay every JAX solve and compare "
        "the replay with the recorded run",
    )
    parser.addoption(
        "--gradients",
        action="store_true",
        help="step 4.4d (V8): with --backend=jax, check the derivative of every distinct "
        "model's JAX solve (<= GRADIENT_MAX_STATES states) against finite differences",
    )
    parser.addoption(
        "--forward",
        action="store_true",
        help="with --gradients, check the forward sensitivities (sim.function(..., "
        "derivatives='forward')) instead of the replay",
    )
    parser.addoption(
        "--tier",
        default="auto",
        choices=("auto", "sparse"),
        help="step 4.5: with --backend=jax, 'sparse' puts every JAX solve on the sparse tier "
        "(SuperLU on the host), whatever its size (default: by size)",
    )
    parser.addoption(
        "--jacobian",
        action="store_true",
        help="V7: compare the sparse Jacobian with dense jacfwd for every compiled model",
    )


def _round_trip(model, compiled, ir):
    import tenqsp as tq
    from tenqsp.canonical import differences

    text = model.to_text()
    back = tq.Model.from_text(text)
    diff = differences(ir, back.to_ir())
    again = back.to_text()
    if again != tq.Model.from_text(again).to_text():
        diff.append("the text is not a fixed point of to_text(from_text(...))")
    return diff


def _v7(model, cm, ir):
    """Differences between the sparse and the dense Jacobian, or the reason it cannot be
    checked (a string): too big for the dense oracle, or no state where the right-hand side
    and its derivative are finite (such models test functions at their domain edges)."""
    import numpy as np

    n = cm.n_states
    if n > V7_MAX_STATES:
        return f"above {V7_MAX_STATES:,} states"
    rng = np.random.default_rng(7)
    th = np.array(cm.theta0)
    p = cm.params.size
    gone = np.isnan(th[:p])  # declared without a value: any positive value will do
    th[:p][gone] = rng.uniform(0.5, 2.0, int(gone.sum()))
    y0 = np.asarray(cm.initial_state(th), dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(y0)))) if n else 1.0
    y1 = np.abs(y0) * rng.uniform(0.5, 1.5, n) + 0.1 * scale * rng.uniform(0.0, 1.0, n)
    n_q = len(cm.ledger_entries)
    out, compared = [], 0
    t = 0.37
    for label, y in (("the starting state", y0), ("a random positive state", y1)):
        # at a domain edge (rhs or a derivative infinite), forward-mode jacfwd turns 0 x inf
        # into NaN even where the derivative is structurally zero: nothing to compare there
        if not np.all(np.isfinite(np.asarray(cm.rhs(t, y, th)))):
            continue
        if not np.all(np.isfinite(np.asarray(cm.jacobian(t, y, th)))):
            continue
        compared += 1
        for ledger in (False, True):
            S = cm.sparse_jacobian(t, y, th, ledger=ledger).toarray()
            if ledger:
                z = np.concatenate([y, np.zeros(n_q)])
                D = np.asarray(cm._aug_jac(np.float64(t), z, cm._th(th), cm.geom))
            else:
                D = np.asarray(cm.jacobian(t, y, th))
            if S.shape != D.shape:
                out.append(f"{label}, ledger={ledger}: shapes {S.shape} vs {D.shape}")
                continue
            same = (np.isnan(S) & np.isnan(D)) | (np.isinf(S) & (S == D))
            err = np.where(same, 0.0, np.abs(S - D))
            big = float(np.max(np.abs(np.where(np.isfinite(D), D, 0.0)), initial=0.0))
            worst = float(np.max(err, initial=0.0))
            if not np.isfinite(worst) or worst > V7_TOL * max(1.0, big):
                r, c = np.unravel_index(
                    np.nanargmax(np.where(np.isfinite(err), err, np.inf)), err.shape
                )
                out.append(
                    f"{label}, ledger={ledger}: max |sparse - dense| = {worst:.3g} "
                    f"(scale {big:.3g}) at row {r}, column {c}: {S[r, c]!r} vs {D[r, c]!r}"
                )
    return (
        out
        if compared
        else "the right-hand side or its derivative is not finite at the states tried"
    )


_V9 = {
    "jax": 0,
    "sparse": 0,
    "scipy": {},
    "replayed": 0,
    "replay_skipped": 0,
    "replay_worst": 0.0,
    "gradients": 0,
    "gradient_finer": 0,
    "gradient_skipped": {},
    "gradient_worst": 0.0,
}
# solves on the JAX backend, those left on scipy (why), and the --replay check


REPLAY_TOLERANCES = 1e-6  # the replay takes its values from the record: round-off only
REPLAY_MAX_STATES = 500  # recording repeats the solve: the 802-state three-axis toy takes
# ~15 s per JAX solve, which with --replay passes the per-test timeout


def _check_replay(cm, t_span, options):
    """Record the run (no ledger), replay it, and compare. The replay re-solves each step to
    full Newton convergence where the run stopped at its Newton tolerance, so they agree to
    within the solver's tolerance: the difference is measured in units of atol + rtol x the
    state's largest value, and must stay below REPLAY_TOLERANCES of them."""
    import numpy as np

    from tenqsp.backends import jax_replay
    from tenqsp.backends.scipy_backend import SolverError

    if cm.n_states > REPLAY_MAX_STATES:
        _V9["replay_skipped"] += 1
        return
    keep = (
        "times",
        "params",
        "theta",
        "rtol",
        "atol",
        "max_step",
        "regimen",
        "initial",
        "max_steps",
    )
    opts = {k: v for k, v in options.items() if k in keep}
    try:
        rec = jax_replay.record(cm, t_span, **opts)
    except SolverError:  # without the ledger the run can differ, and fail where it did not
        _V9["replay_skipped"] += 1
        return
    ys = np.asarray(jax_replay.replay(cm, rec))
    rtol = float(opts.get("rtol", 1e-6))
    atol = np.broadcast_to(np.asarray(opts.get("atol", 1e-6), dtype=float), (cm.n_states,))
    tolerance = atol + rtol * np.max(np.abs(rec.ys), axis=0)
    worst = float(np.max(np.abs(ys - rec.ys) / tolerance))
    _V9["replayed"] += 1
    _V9["replay_worst"] = max(_V9["replay_worst"], worst)
    if not worst <= REPLAY_TOLERANCES:
        raise AssertionError(
            f"replay of '{cm.ir.name}' differs from the recorded run by {worst:.3g} tolerances"
        )


GRADIENT_MAX_STATES = 200  # each check is a recorded run and four more solves
GRADIENT_TOL = 1e-4  # relative to the outputs' size, as V8's finite-difference bound
GRADIENT_STEP = 1e-4  # relative change of every parameter for the central differences
_GRADIENT_SEEN: set = set()
_GRADIENT_FORWARD = False  # --forward: check the forward sensitivities, not the replay


def _check_gradient(cm, t_span, options):
    """V8 on every model (step 4.4d): the derivative of the run along a random direction of
    all parameters at once (each changed in proportion to its value), from the replay
    (jax.jvp; with --forward, carried along the run as a forward sensitivity), against
    central differences of full solves, both at tight tolerances (rtol 1e-10, atol / 1e4).
    Once per distinct model and options.

    - Differences between solves are noisy (each run chooses its own steps): their error is
      estimated from two step sizes, and the derivative must agree within GRADIENT_TOL of
      the outputs' size or 10x that estimate.
    - An output time on a boundary that moves (a dose, a switch or an event at that very
      time) is a kink: the derivative there is one-sided (after the boundary, as outputs are
      reported) and central differences average both sides. Such times (where forward and
      backward differences disagree) are left out.
    - The run's steps are chosen for its derivative along the direction too (``record(...,
      direction=)``, design note step4_4d_api.md §6.3), as ``sim.function`` does.
    - A solution that is itself inaccurate (values far below atol, then regrowing) has an
      equally inaccurate derivative. So a mismatch is checked again with fine steps
      (max_step = span/1000, then span/10000) and atol / 1e4 more: agreeing then, it is
      counted apart ("agrees only with finer steps"); still disagreeing, it fails."""
    import numpy as np

    from tenqsp.backends.scipy_backend import SolverError

    def skip(why):
        _V9["gradient_skipped"][why] = _V9["gradient_skipped"].get(why, 0) + 1

    key = (cm.ir.content_hash(), repr(sorted((k, repr(v)) for k, v in options.items())))
    if key in _GRADIENT_SEEN:
        return
    _GRADIENT_SEEN.add(key)
    if cm.n_states > GRADIENT_MAX_STATES:
        return skip(f"above {GRADIENT_MAX_STATES} states")
    keep = ("times", "params", "theta", "max_step", "regimen", "initial", "max_steps")
    opts = {k: v for k, v in options.items() if k in keep}
    opts["rtol"] = 1e-10
    opts["atol"] = np.asarray(options.get("atol", 1e-6), dtype=float) * 1e-4
    try:
        result = _compare_gradient(cm, t_span, opts)
    except SolverError:
        return skip("a run at tight tolerances fails")
    if isinstance(result, str):
        return skip(result)
    err, noise = result
    if err <= max(GRADIENT_TOL, 10 * noise):
        _V9["gradients"] += 1
        _V9["gradient_worst"] = max(_V9["gradient_worst"], err)
        return
    again = None
    for pieces in (1_000, 10_000):  # a solution at rest also stays at order 1
        finer = {
            **opts,
            "max_step": (t_span[1] - t_span[0]) / pieces,
            "atol": opts["atol"] * 1e-4,
            "max_steps": max(opts.get("max_steps", 100_000), 4 * pieces),
        }
        try:
            again = _compare_gradient(cm, t_span, finer)
        except SolverError:
            again = None
        if isinstance(again, tuple) and again[0] <= max(GRADIENT_TOL, 10 * again[1]):
            _V9["gradient_finer"] += 1
            return
    raise AssertionError(
        f"the derivative of '{cm.ir.name}' differs from finite differences by {err:.3g} of "
        f"the outputs' size (finite-difference noise {noise:.2g}), and with finer steps "
        f"by {again[0] if isinstance(again, tuple) else again}"
    )


def _compare_gradient(cm, t_span, opts):
    """(largest difference, finite-difference noise), both relative to the outputs' size;
    or why it cannot be compared (a string)."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from tenqsp.backends import jax_backend as jb
    from tenqsp.backends import jax_replay
    from tenqsp.backends.scipy_backend import SolverError

    o = {k: v for k, v in opts.items() if k not in ("params", "theta")}
    try:
        th = np.asarray(
            cm._th(cm.theta(opts.get("params")) if opts.get("theta") is None else opts["theta"]),
            dtype=float,
        )
        p = cm.params.size
        rng = np.random.default_rng(int(cm.ir.content_hash()[:8], 16))
        direction = np.zeros_like(th)
        direction[:p] = th[:p] * rng.choice([-1.0, 1.0], p) * rng.uniform(0.5, 1.0, p)
        if not np.any(direction):
            return "no parameter with a value other than 0"
        if _GRADIENT_FORWARD:  # the run carries the derivative along the direction
            run = jb._prepare(
                cm, t_span, o.get("times"), None, th, "BDF", o.get("rtol", 1e-6),
                o.get("atol", 1e-6), o.get("max_step", np.inf), False, o.get("regimen"),
                o.get("initial"),
            )  # fmt: skip
        else:
            rec = jax_replay.record(cm, t_span, theta=th, direction=direction, **o)
    except ValueError:
        return "the run is refused"
    if _GRADIENT_FORWARD:
        prog = jb._program(
            cm, ledger=False, model_doses=run.model_doses,
            max_steps=o.get("max_steps", 100_000), forward=1,
        )  # fmt: skip
        out = prog(**run.args, dot={"theta": jnp.asarray(direction)[None]})
        if int(out["status"]) != 0:
            raise SolverError(f"status {int(out['status'])}")
        G = np.asarray(out["ys"])[: run.t_out.size, cm.n_states :]
        y_run = np.asarray(out["ys"])[: run.t_out.size, : cm.n_states]
    else:
        _, tangent = jax.jvp(
            lambda x: jax_replay.replay(cm, rec, x), (jnp.asarray(th),), (jnp.asarray(direction),)
        )
        G = np.asarray(tangent)
        y_run = rec.ys
    h = GRADIENT_STEP
    try:
        ys = {
            k: jax_replay.record(cm, t_span, theta=th + k * h * direction, **o).ys
            for k in (-2, -1, 1, 2)
        }
    except (SolverError, ValueError):
        return "a run with the parameters moved fails"
    fd = [(ys[1] - ys[-1]) / (2 * h), (ys[2] - ys[-2]) / (4 * h)]
    size = np.max(np.abs(y_run)) + np.max(np.abs(fd[0]))
    if not size > 0:
        return "outputs and their derivatives all 0"
    if not np.all(np.isfinite(G)):
        return "outputs or derivatives not finite"
    # kinks: one-sided differences disagree there. On a smooth output the disagreement is
    # the curvature, h^2 f'' (4x as large at 2h); at a kink it is h x the jump in slope (2x
    # at 2h). A large one is a kink whatever its ratio; a small one that still matters for
    # the comparison (above GRADIENT_TOL / 10) is a kink when it doubles rather than
    # quadruples (a switch time that the direction barely moves).
    d1 = (ys[1] - y_run) - (y_run - ys[-1])
    d2 = (ys[2] - y_run) - (y_run - ys[-2])
    doubles = np.abs(d2 - 2 * d1) < np.abs(d2 - 4 * d1)
    kink = (np.abs(d1) / h > 1e-2 * size) | ((np.abs(d1) / h > 0.1 * GRADIENT_TOL * size) & doubles)
    smooth = ~np.any(kink, axis=1)
    if not np.any(smooth):
        return "every output time on a kink"
    G, fd = G[smooth], [x[smooth] for x in fd]
    noise = float(np.max(np.abs(fd[0] - fd[1]))) / size
    if noise > 1e-2:
        return "finite differences unreliable (a boundary or event moves across)"
    return float(np.max(np.abs(G - fd[0]))) / size, noise


def _use_jax_backend(replay: bool = False, gradients: bool = False):
    """Route every solve through the JAX backend (V9): Model.solve and CompiledModel.solve.
    Runs the JAX backend does not take stay on scipy and are counted: method='Radau' (the
    JAX backend is BDF only) and models it refuses (NotImplementedError: a function with
    no JAX form). Solves on the sparse tier (step 4.5) are counted too."""
    import tenqsp as tq
    from tenqsp.backends import jax_backend
    from tenqsp.compile import CompiledModel

    scipy_solve = CompiledModel.solve

    def on_scipy(why):
        _V9["scipy"][why] = _V9["scipy"].get(why, 0) + 1

    def cm_solve(self, t_span, **options):
        if options.get("method", "BDF") != "BDF":
            on_scipy("method='Radau' asked for")
            return scipy_solve(self, t_span, **options)
        try:
            sol = jax_backend.solve(self, t_span, **options)
        except NotImplementedError as e:
            on_scipy(str(e)[:80])
            return scipy_solve(self, t_span, **options)
        _V9["jax"] += 1
        _V9["sparse"] += jax_backend.tier(self) == "sparse"
        if replay:
            _check_replay(self, t_span, options)
        if gradients:
            _check_gradient(self, t_span, options)
        return sol

    CompiledModel.solve = cm_solve
    model_solve = tq.Model.solve

    @functools.wraps(model_solve)
    def solve(self, t_span, *, backend="scipy", **options):  # both end in CompiledModel.solve
        if backend not in ("scipy", "jax"):  # an invalid name still raises
            return model_solve(self, t_span, backend=backend, **options)
        return model_solve(self, t_span, backend="scipy", **options)

    tq.Model.solve = solve


def pytest_collection_modifyitems(config, items):
    if config.getoption("--backend") != "jax":
        return
    import pytest

    skip = pytest.mark.skip(reason="about the scipy backend itself (--backend=jax)")
    for item in items:
        if "scipy_backend" in item.keywords:
            item.add_marker(skip)


def pytest_configure(config):
    for flag in ("--replay", "--gradients"):
        if config.getoption(flag) and config.getoption("--backend") != "jax":
            raise ValueError(f"{flag} needs --backend=jax")
    if config.getoption("--forward"):
        if not config.getoption("--gradients"):
            raise ValueError("--forward needs --gradients")
        global _GRADIENT_FORWARD
        _GRADIENT_FORWARD = True
    if config.getoption("--tier") != "auto":
        if config.getoption("--backend") != "jax":
            raise ValueError("--tier needs --backend=jax")
        from tenqsp.backends import jax_backend

        jax_backend.SPARSE_FROM_STATES = jax_backend.SPARSE_FROM_STATES_GPU = 0
        # every host solve costs ~100 us of callback overhead: tiny models at tight
        # tolerances take far longer than on the dense tier they would normally use
        base = getattr(config.option, "timeout", None) or config.getini("timeout")
        if base:
            config.option.timeout = 4 * float(base)
    if config.getoption("--gradients"):
        # each run is checked against central finite differences: several more solves (the
        # Keller-Segel growth-rate test takes ~110 s alone, against a 120 s limit)
        base = getattr(config.option, "timeout", None) or config.getini("timeout")
        if base and float(base) > 0:
            config.option.timeout = 2 * float(base)
    if config.getoption("--backend") == "jax":
        _use_jax_backend(config.getoption("--replay"), config.getoption("--gradients"))
    active = [name for name in _CHECKS if config.getoption("--" + name)]
    if not active:
        return
    import tenqsp as tq

    run = {"roundtrip": _round_trip, "jacobian": _v7}
    original = tq.Model.compile
    seen: dict[str, set] = {name: set() for name in active}
    config._model_checks = {name: {"models": [], "failures": [], "skipped": []} for name in active}

    @functools.wraps(original)
    def compile_and_check(self, *args, **kwargs):
        compiled = original(self, *args, **kwargs)  # an invalid model raises here, as before
        ir = self.to_ir()
        key = ir.content_hash()
        for name in active:
            if key in seen[name]:
                continue
            seen[name].add(key)
            info = config._model_checks[name]
            try:
                diff = run[name](self, compiled, ir)
            except Exception as e:  # noqa: BLE001 (reported, not raised)
                diff = [f"{type(e).__name__}: {e}"]
            if isinstance(diff, str):  # could not be checked, and why
                info["skipped"].append([key, diff])
                continue
            info["models"].append(key)
            if diff:
                info["failures"].append((ir.name, diff))
                raise AssertionError(f"{_CHECKS[name]} of '{ir.name}':\n  " + "\n  ".join(diff))
        return compiled

    tq.Model.compile = compile_and_check


_CHECK_DIR = os.path.join(os.path.dirname(__file__), "..", ".pytest_cache", "model_checks")


def pytest_unconfigure(config):
    if config.getoption("--backend") == "jax" and os.environ.get("PYTEST_XDIST_WORKER"):
        import json

        os.makedirs(_CHECK_DIR, exist_ok=True)
        path = os.path.join(_CHECK_DIR, "v9_" + os.environ["PYTEST_XDIST_WORKER"] + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_V9, f)
    info = getattr(config, "_model_checks", None)
    if info is not None and os.environ.get("PYTEST_XDIST_WORKER"):
        import json

        os.makedirs(_CHECK_DIR, exist_ok=True)
        path = os.path.join(_CHECK_DIR, os.environ["PYTEST_XDIST_WORKER"] + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f)


def pytest_sessionstart(session):
    wanted = getattr(session.config, "_model_checks", None) is not None
    wanted = wanted or session.config.getoption("--backend") == "jax"
    if wanted and not os.environ.get("PYTEST_XDIST_WORKER"):
        import shutil

        shutil.rmtree(_CHECK_DIR, ignore_errors=True)


def pytest_terminal_summary(terminalreporter, config):
    import glob
    import json

    if os.environ.get("PYTEST_XDIST_WORKER"):
        # xdist workers run this hook too: adding other workers' files into this worker's
        # counters (saved after it) inflated the replay and gradient counts until 4.5
        return
    if config.getoption("--backend") == "jax":
        on_jax, on_scipy = _V9["jax"], dict(_V9["scipy"])
        for path in glob.glob(os.path.join(_CHECK_DIR, "v9_*.json")):
            with open(path, encoding="utf-8") as f:
                w = json.load(f)
            on_jax += w["jax"]
            _V9["sparse"] += w.get("sparse", 0)
            for why, k in w["scipy"].items():
                on_scipy[why] = on_scipy.get(why, 0) + k
            for k in ("replayed", "replay_skipped", "gradients", "gradient_finer"):
                _V9[k] += w[k]
            _V9["replay_worst"] = max(_V9["replay_worst"], w["replay_worst"])
            _V9["gradient_worst"] = max(_V9["gradient_worst"], w["gradient_worst"])
            for why, k in w["gradient_skipped"].items():
                _V9["gradient_skipped"][why] = _V9["gradient_skipped"].get(why, 0) + k
        left = "; ".join(f"{k} {why}" for why, k in sorted(on_scipy.items())) or "none"
        terminalreporter.write_line(
            f"V9 on the JAX backend: {on_jax} solves on JAX ({_V9['sparse']} on the sparse "
            f"tier); left on scipy: {left}"
        )
        if config.getoption("--replay"):
            terminalreporter.write_line(
                f"Replay (step 4.4a): {_V9['replayed']} runs recorded and replayed, largest "
                f"difference {_V9['replay_worst']:.2g} tolerances; {_V9['replay_skipped']} not "
                f"checked (above {REPLAY_MAX_STATES} states, or the run without the ledger "
                "failed)"
            )
        if config.getoption("--gradients"):
            left = "; ".join(f"{k} {why}" for why, k in sorted(_V9["gradient_skipped"].items()))
            terminalreporter.write_line(
                f"Gradients (step 4.4d, V8, "
                f"{'forward sensitivities' if _GRADIENT_FORWARD else 'replay'}): "
                f"{_V9['gradients']} distinct runs agree with finite "
                f"differences (largest difference {_V9['gradient_worst']:.2g} of the outputs' "
                f"size); {_V9['gradient_finer']} agree only with finer steps (the solution's "
                f"steps too coarse for its derivative); not checked: {left or 'none'}"
            )
    checks = getattr(config, "_model_checks", None)
    if checks is None:
        return

    workers = []
    for path in glob.glob(os.path.join(_CHECK_DIR, "gw*.json")):
        with open(path, encoding="utf-8") as f:
            workers.append(json.load(f))
    for name, info in checks.items():
        models, failures = set(info["models"]), list(info["failures"])
        skipped = {k: why for k, why in info["skipped"]}
        for w in workers:
            models |= set(w[name]["models"])
            failures += w[name]["failures"]
            skipped.update({k: why for k, why in w[name]["skipped"]})
        reasons: dict[str, int] = {}
        for k, why in skipped.items():
            if k not in models:
                reasons[why] = reasons.get(why, 0) + 1
        if name == "roundtrip":
            line = f"V12 round trip: {len(models)} distinct models, {len(failures)} failed"
        else:
            why = "; ".join(f"{v} {k}" for k, v in sorted(reasons.items())) or "none"
            line = (
                f"V7 sparse Jacobian: {len(models)} distinct models checked against dense "
                f"jacfwd, {len(failures)} failed (not checkable: {why})"
            )
        terminalreporter.write_line(line)
        for model, diff in failures:
            terminalreporter.write_line(f"  {model}: {diff[0][:300]}")
