"""Step 4.3: the JAX BDF (dense tier) against scipy BDF, on the Phase 1 models and the toy
phage model at several sizes up to the dense-tier limit (1,000 states).

Run alone on an idle machine. Each case runs in its own process, one after another:
python benchmarks/step4_3_jax.py          # all cases
python benchmarks/step4_3_jax.py 3        # one case (index into CASES)
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time

import jax
import numpy as np
import scipy
from phase1 import transit, with_scheme

from tenqsp.backends import jax_backend
from tenqsp.examples import phage_latency_model, toy_phage_model

CASES = [
    ("transit chain, 5 erlang cells", transit),
    (
        "phage latency, 40 age cells, erlang",
        lambda: with_scheme(phage_latency_model(), "erlang"),
    ),
    ("toy phage 10 z x 12 a, erlang", lambda: with_scheme(toy_phage_model(10, 12), "erlang")),
    ("toy phage 16 z x 20 a, erlang", lambda: with_scheme(toy_phage_model(16, 20), "erlang")),
    ("toy phage 22 z x 30 a, erlang", lambda: with_scheme(toy_phage_model(22, 30), "erlang")),
    ("toy phage 24 z x 38 a, erlang", lambda: with_scheme(toy_phage_model(24, 38), "erlang")),
]
T_OUT = np.linspace(0, 24, 25)


def run(i: int) -> dict:
    label, build = CASES[i]
    cm = build()
    cm.solve((0, 24), times=T_OUT)  # scipy: compiles its functions
    walls = []
    for _ in range(3):
        s = cm.solve((0, 24), times=T_OUT)
        walls.append(s.stats["wall_seconds"])
    t0 = time.perf_counter()
    j = jax_backend.solve(cm, (0, 24), times=T_OUT)  # first JAX solve: traces and compiles
    first = time.perf_counter() - t0
    runs = []
    for _ in range(3):
        j = jax_backend.solve(cm, (0, 24), times=T_OUT)
        runs.append(j.stats["wall_seconds"])
    diff = max(
        float(np.max(np.abs(j[k] - s[k])) / max(np.max(np.abs(s[k])), 1e-300))
        for k in cm.unpack(cm.initial_state())
    )
    return {
        "label": label,
        "states": cm.n_states,
        "jax_first_s": first,
        "jax_s": min(runs),
        "scipy_s": min(walls),
        "jax_steps": j.stats["n_steps"],
        "scipy_nfev": s.stats["nfev"],
        "jax_nlu": j.stats["nlu"],
        "scipy_nlu": s.stats["nlu"],
        "jax_njev": j.stats["njev"],
        "scipy_njev": s.stats["njev"],
        "diff": diff,
        "ledger": max(j.ledger.check().values()),
    }


def main():
    if len(sys.argv) > 1:
        print(json.dumps(run(int(sys.argv[1]))))
        return
    rows = []
    for i in range(len(CASES)):
        out = subprocess.run(
            [sys.executable, __file__, str(i)], capture_output=True, text=True, check=True
        )
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
        print(rows[-1], flush=True)
    head = (
        "| model | states | JAX first solve incl. compile (s) | JAX solve (s) | scipy solve (s) | "
        "LU JAX / scipy | Jacobians JAX / scipy | largest difference | ledger unexplained |"
    )
    lines = [head, "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['states']:,} | {r['jax_first_s']:.1f} | {r['jax_s']:.3f} | "
            f"{r['scipy_s']:.3f} | {r['jax_nlu']} / {r['scipy_nlu']} | "
            f"{r['jax_njev']} / {r['scipy_njev']} | {r['diff']:.0e} | {r['ledger']:.0e} |"
        )
    print("\n".join(lines))
    print(
        f"\nPython {platform.python_version()}, JAX {jax.__version__}, scipy {scipy.__version__}, "
        f"numpy {np.__version__}; {platform.processor() or platform.machine()}, CPU JAX"
    )


if __name__ == "__main__":
    main()
