"""Phase 2 gate: Performance targets and the Phase 2 models (results -> docs/benchmarks.md).

Run alone on an idle machine, one case per process:
python benchmarks/phase2_gate.py            # list the cases
python benchmarks/phase2_gate.py 3          # run case 3
"""

from __future__ import annotations

import platform
import resource
import sys
import time
from pathlib import Path

import jax
import numpy as np
import scipy
from phase1 import with_scheme

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from pbpk_models import pbpk  # noqa: E402
from test_gallery_phase2 import friberg, hiv_age_model  # noqa: E402
from test_transitions_phase2 import two_phage  # noqa: E402

from tenqsp.examples import toy_phage_model, toy_three_axis_model  # noqa: E402


def _pbpk_bolus():
    m = pbpk()
    m.dose(m["Cven"], times=[0], amounts=[100.0])
    return m.compile()


def _hiv():
    m, _ = hiv_age_model(60, A=60.0)
    return m.compile()


# (label, build, t_end, solve kwargs)
CASES = [
    ("target: toy phage 100 z x 100 a, erlang, 72 h",
     lambda: with_scheme(toy_phage_model(100, 100), "erlang"), 72.0, {}),
    ("toy phage 30 z x 40 a, sharp, 24 h", lambda: toy_phage_model().compile(), 24.0, {}),
    ("three-axis toy I[a, z, s] 24 a x 16 z x 2, 12 h",
     lambda: toy_three_axis_model().compile(), 12.0, {}),
    ("two-phage toy B[r1, r2], 24 h", lambda: two_phage()[0].compile(), 24.0, {}),
    ("whole-body PBPK 9 organs x 3 spaces, bolus, 24 h", _pbpk_bolus, 24.0, {}),
    ("HIV infection age, 60 cells, 60 days", _hiv, 60.0, {"rtol": 1e-8, "atol": 1e-12}),
    ("Friberg myelosuppression, 600 h", lambda: friberg(dose=1.8).compile(), 600.0, {}),
]  # fmt: skip


def run(k: int) -> None:
    label, build, t_end, kw = CASES[k - 1]
    t0 = time.perf_counter()
    cm = build()
    sol = cm.solve((0, t_end), times=[t_end], **kw)  # first solve: compiles
    first = time.perf_counter() - t0
    walls = []
    for _ in range(3):
        sol = cm.solve((0, t_end), times=[t_end], **kw)
        walls.append(sol.stats["wall_seconds"])
    rhs_call = cm.warm_up()["rhs call"]
    S = cm.sparse_structure
    s = sol.stats
    ledger = max(sol.ledger.check().values()) if sol.ledger is not None else float("nan")
    print(
        f"| {label} | {cm.n_states:,} | {S['n_colours']} | {first:.1f} | {min(walls):.2f} | "
        f"{rhs_call * 1e3:.3f} | {s['nfev']:,} | {s['njev']} | {s['nlu']} | "
        f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} | {ledger:.0e} |"
    )
    print(
        f"Python {platform.python_version()}, JAX {jax.__version__}, scipy {scipy.__version__}, "
        f"numpy {np.__version__}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        for i, (label, *_) in enumerate(CASES, 1):
            print(i, label)
    else:
        run(int(sys.argv[1]))
