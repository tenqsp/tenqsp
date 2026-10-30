"""Phase 5 gate: the reference models' cost on both backends, the target case re-measured,
and how long each tutorial takes to run (-> docs/benchmarks.md).

Run alone, one case per process, one after another:
python benchmarks/phase5_gate.py              # list the cases
python benchmarks/phase5_gate.py 2            # run case 2
python benchmarks/phase5_gate.py tutorials    # time every tutorial (one process each)
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np
import scipy
from phase1 import with_scheme

from tenqsp.examples import (
    drug_tolerance_model,
    oral_absorption_model,
    plaque_model,
    toy_phage_model,
)

ROOT = Path(__file__).resolve().parents[1]

# (label, build, t_end, solve options)
CASES = [
    ("target: toy phage 100 z x 100 a, erlang, 72 h",
     lambda: with_scheme(toy_phage_model(100, 100), "erlang").ir, 72.0, {}),
    ("plaque fronts 240 x x 40 a, 10 h", lambda: plaque_model(240, 40), 10.0, {"rtol": 1e-6}),
    ("oral absorption 160 g x 50 x, 24 h", oral_absorption_model, 24.0, {}),
    ("drug tolerance 160 x, constant drug, 60 d", drug_tolerance_model, 60.0,
     {"params": {"c0": 1.0}}),
]  # fmt: skip


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def run(k: int) -> None:
    import tenqsp as tq

    label, build, t_end, kw = CASES[k - 1]
    built = build()
    m = built if isinstance(built, tq.Model) else tq.Model.from_ir(built)
    cm, t_compile = _timed(m.compile)
    times = np.linspace(0, t_end, 25)
    a, t_scipy = _timed(lambda: cm.solve((0, t_end), times=times, **kw))
    _, t_jax1 = _timed(lambda: m.solve((0, t_end), times=times, backend="jax", **kw))
    b, t_jax2 = _timed(lambda: m.solve((0, t_end), times=times, backend="jax", **kw))
    gap = max(
        float(np.max(np.abs(a[n] - b[n]) / np.maximum(np.max(np.abs(a[n])), 1e-30)))
        for n in a.states
    )
    tier = "sparse" if cm.n_states >= 250 else "dense"
    print(
        f"| {label} | {cm.n_states:,} | {t_compile:.2f} | {t_scipy:.2f} | "
        f"{a.stats['nfev']:,} | {t_jax1:.2f} | {t_jax2:.2f} ({tier}) | {gap:.0e} |"
    )
    print(
        f"Python {platform.python_version()}, JAX {jax.__version__}, scipy {scipy.__version__}, "
        f"numpy {np.__version__}"
    )


def tutorials() -> None:
    for p in sorted((ROOT / "docs" / "tutorials").glob("[0-9][0-9]_*.py")):
        t0 = time.perf_counter()
        r = subprocess.run(
            [sys.executable, p.name],
            cwd=p.parent,
            env={**os.environ, "MPLBACKEND": "Agg"},
            capture_output=True,
        )
        status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
        print(f"| {p.stem} | {time.perf_counter() - t0:.0f} | {status} |", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        for i, (label, *_) in enumerate(CASES, 1):
            print(i, label)
        print("tutorials")
    elif sys.argv[1] == "tutorials":
        tutorials()
    else:
        run(int(sys.argv[1]))
