"""Phase 3 gate: the cost of the notation and of what the library tells you, and whether a
model read from text solves exactly like the same model built in Python (-> docs/benchmarks.md).

Run alone on an idle machine, one case per process:
python benchmarks/phase3_gate.py            # list the cases
python benchmarks/phase3_gate.py 3          # run case 3
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import jax
import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from pbpk_models import pbpk  # noqa: E402
from test_gallery_phase2 import friberg  # noqa: E402

import tenqsp as tq  # noqa: E402
from tenqsp.canonical import differences  # noqa: E402
from tenqsp.examples import toy_phage_model, toy_three_axis_model  # noqa: E402


def _pbpk_bolus():
    m = pbpk()
    m.dose(m["Cven"], times=[0], amounts=[100.0])
    return m


# (label, build, t_end)
CASES = [
    ("toy phage 30 z x 40 a, sharp", toy_phage_model, 24.0),
    ("toy phage 100 z x 100 a, sharp", lambda: toy_phage_model(100, 100), 1.0),
    ("three-axis toy I[a, z, s]", toy_three_axis_model, 12.0),
    ("whole-body PBPK 9 organs x 3 spaces", _pbpk_bolus, 24.0),
    ("Friberg myelosuppression", lambda: friberg(dose=1.8), 600.0),
]


def _best(fn, k=3):
    out, walls = None, []
    for _ in range(k):
        t0 = time.perf_counter()
        out = fn()
        walls.append(time.perf_counter() - t0)
    return out, min(walls) * 1e3


def run(k: int) -> None:
    label, build, t_end = CASES[k - 1]
    m = build()
    text, t_write = _best(m.to_text)
    back, t_read = _best(lambda: tq.Model.from_text(text))
    diff, t_cmp = _best(lambda: differences(m.to_ir(), back.to_ir()))
    _, t_explain = _best(m.explain)
    _, t_diag = _best(m.diagnostics)
    _, t_latex = _best(m.to_latex)
    a = m.compile().solve((0, t_end), times=[t_end])
    b = back.compile().solve((0, t_end), times=[t_end])
    gap = max(
        float(np.max(np.abs(a[n] - b[n]) / np.maximum(np.abs(a[n]), 1e-6), initial=0.0))
        for n in a.states
    )
    print(
        f"| {label} | {len(text.splitlines())} | {t_write:.0f} | {t_read:.0f} | {t_cmp:.0f} | "
        f"{'none' if not diff else len(diff)} | {t_explain:.0f} | {t_diag:.0f} | {t_latex:.0f} | "
        f"{gap:.0e} | {a.stats['nfev']:,} / {b.stats['nfev']:,} |"
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
