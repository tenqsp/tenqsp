"""Phase 2 step 2.1: the sparse Jacobian at 10^3 - 10^5 states (compare ADR 0003).

Run alone on an idle machine, one size at a time:
python benchmarks/phase2_scale.py            # all sizes
python benchmarks/phase2_scale.py 100 100    # one size (n_z n_a)
"""

from __future__ import annotations

import platform
import resource
import sys
import time

import jax
import numpy as np
import scipy
from phase1 import with_scheme

from tenqsp.examples import toy_phage_model

SIZES = [(30, 40), (60, 60), (100, 100), (170, 170), (316, 316)]


def run(n_z: int, n_a: int) -> dict:
    t0 = time.perf_counter()
    cm = with_scheme(toy_phage_model(n_z, n_a), "erlang")
    S = cm.sparse_structure
    structure = time.perf_counter() - t0
    y = np.asarray(cm.initial_state())
    t0 = time.perf_counter()
    cm.sparse_jacobian(0.0, y, ledger=True)
    jac_first = time.perf_counter() - t0
    walls = []
    for _ in range(3):
        t0 = time.perf_counter()
        J = cm.sparse_jacobian(0.0, y, ledger=True)
        walls.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    sol = cm.solve((0, 24), times=[24])
    first = time.perf_counter() - t0
    sol = cm.solve((0, 24), times=[24])
    s = sol.stats
    return {
        "states": cm.n_states,
        "colours": S["n_colours"],
        "reverse": S["n_row_colours"],
        "nnz": J.nnz,
        "structure_s": structure,
        "jac_first_s": jac_first,
        "jac_ms": min(walls) * 1e3,
        "first_solve_s": first,
        "solve_s": s["wall_seconds"],
        "nfev": s["nfev"],
        "njev": s["njev"],
        "nlu": s["nlu"],
        "ledger": max(sol.ledger.check().values()),
        "rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "phage_24h": float(sol["phage"][-1]),
    }


def main():
    sizes = [tuple(int(v) for v in sys.argv[1:3])] if len(sys.argv) > 2 else SIZES
    for n_z, n_a in sizes:
        r = run(n_z, n_a)
        print(f"{n_z}x{n_a}", r, flush=True)
    print(
        f"Python {platform.python_version()}, JAX {jax.__version__}, scipy {scipy.__version__}, "
        f"numpy {np.__version__}"
    )


if __name__ == "__main__":
    main()
