"""Phase 1 benchmark table (run alone on an idle machine; results -> docs/benchmarks.md).

python benchmarks/phase1.py
"""

from __future__ import annotations

import dataclasses
import platform
import time

import jax
import numpy as np
import scipy

import tenqsp as tq
from tenqsp.compile import CompiledModel
from tenqsp.examples import phage_latency_model, toy_phage_model


def with_scheme(model: tq.Model, scheme: str) -> CompiledModel:
    ir = model.to_ir()
    terms = tuple(
        dataclasses.replace(t, scheme=scheme) if t.kind == "advection" else t for t in ir.terms
    )
    return CompiledModel(dataclasses.replace(ir, terms=terms))


def transit():
    m = tq.Model("transit")
    a = m.axis("a", 0, 2.0, 5, kind="one_way")
    A, C = m.compartment("A"), m.compartment("C")
    X = m.field("X", axes=[a])
    ka, ke = m.params(ka=1.5, ke=0.2)
    m.advect(X, axis=a, velocity=1.0)
    m.inflow(X, axis=a, at="lower", flux=ka * A)
    m.outflow(X, axis=a, at="upper", to=C)
    m.ode(A, -ka * A)
    m.ode(C, -ke * C)
    m.dose(A, times=[0.0], amounts=[100.0])
    return m.compile()


CASES = [
    ("transit chain, 5 erlang cells", transit),
    ("phage latency, 40 age cells, sharp", lambda: phage_latency_model().compile()),
    (
        "phage latency, 40 age cells, erlang",
        lambda: with_scheme(phage_latency_model(), "erlang"),
    ),
    ("toy phage 30 z x 40 a, sharp", lambda: toy_phage_model().compile()),
    ("toy phage 30 z x 40 a, erlang", lambda: with_scheme(toy_phage_model(), "erlang")),
    ("toy phage 60 z x 60 a, erlang", lambda: with_scheme(toy_phage_model(60, 60), "erlang")),
]


def main():
    rows = []
    for label, build in CASES:
        cm = build()
        t0 = time.perf_counter()
        cm.solve((0, 24), times=[24])  # first solve: compiles
        first = time.perf_counter() - t0
        walls = []
        for _ in range(3):
            sol = cm.solve((0, 24), times=[24])
            walls.append(sol.stats["wall_seconds"])
        rhs_call = cm.warm_up()["rhs call"]
        s = sol.stats
        rows.append(
            (
                label,
                cm.n_states,
                len(cm.ledger_entries),
                first,
                min(walls),
                rhs_call * 1e3,
                s["nfev"],
                s["njev"],
                s["nlu"],
                max(sol.ledger.check().values()),
            )
        )
        print(rows[-1], flush=True)
    head = (
        "| model | states | ledger channels | first solve incl. compile (s) | solve (s) | "
        "RHS call (ms) | RHS evals | Jacobians | LU | ledger unexplained |"
    )
    lines = [head, "|" + "---|" * 10]
    for r in rows:
        lines.append(
            f"| {r[0]} | {r[1]:,} | {r[2]} | {r[3]:.1f} | {r[4]:.2f} | {r[5]:.3f} | "
            f"{r[6]:,} | {r[7]} | {r[8]} | {r[9]:.0e} |"
        )
    print("\n".join(lines))
    print(
        f"\nPython {platform.python_version()}, JAX {jax.__version__}, scipy {scipy.__version__}, "
        f"numpy {np.__version__}; {platform.processor() or platform.machine()}, CPU JAX"
    )


if __name__ == "__main__":
    main()
