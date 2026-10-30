"""Step 4.5: the JAX backend's sparse tier (SuperLU on the host) against its dense tier and
the scipy backend, on the toy phage model from 141 to ~10,000 states: forward solves (where
the tiers cross over, which sets ``jax_backend.SPARSE_FROM_STATES``), and a gradient through
``sim.function`` with its memory.

Run alone on an idle machine. Each case runs in its own process, one after another:
python benchmarks/step4_5_sparse.py          # all cases
python benchmarks/step4_5_sparse.py 3        # one case (index into CASES)
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import resource
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import tenqsp as tq
from tenqsp.backends import jax_backend as jb
from tenqsp.examples import toy_phage_model

# (n_z, n_a): 141, 353, 705, 961, 2,481, 5,761 and 10,201 states
SIZES = [(10, 12), (16, 20), (22, 30), (24, 38), (40, 60), (60, 94), (80, 125)]
CASES = SIZES
DENSE_UP_TO = 1_000  # the dense LU above this takes minutes (table 4.3)
GRADIENT_UP_TO = 10_500
T_OUT = np.linspace(0, 24, 25)


def erlang(m: tq.Model) -> tq.Model:
    ir = m.to_ir()
    terms = tuple(
        dataclasses.replace(t, scheme="erlang") if t.kind == "advection" else t for t in ir.terms
    )
    return tq.Model.from_ir(dataclasses.replace(ir, terms=terms))


def best(fn, k=3):
    walls = []
    for _ in range(k):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        walls.append(time.perf_counter() - t0)
    return min(walls)


def run(i: int) -> dict:
    nz, na = CASES[i]
    m = erlang(toy_phage_model(nz, na))
    out = {"label": f"toy phage {nz} z x {na} a"}
    cm = m.compile()
    out["states"] = cm.n_states
    cm.solve((0, 24), times=T_OUT)
    out["scipy_s"] = min(cm.solve((0, 24), times=T_OUT).stats["wall_seconds"] for _ in range(3))
    for tier, threshold in (("dense", 10**9), ("sparse", 0)):
        if tier == "dense" and cm.n_states > DENSE_UP_TO:
            continue
        jb.SPARSE_FROM_STATES = threshold
        t0 = time.perf_counter()
        jb.solve(cm, (0, 24), times=T_OUT)
        out[f"{tier}_first_s"] = time.perf_counter() - t0
        out[f"{tier}_s"] = min(
            jb.solve(cm, (0, 24), times=T_OUT).stats["wall_seconds"] for _ in range(3)
        )
    if cm.n_states <= GRADIENT_UP_TO:  # a gradient on the tier the size picks
        jb.SPARSE_FROM_STATES = 0
        sim = m.simulator(outputs=["P"])
        names = list(cm.params.index)
        sc = tq.Scenario(times=np.linspace(2, 24, 12))
        t0 = time.perf_counter()
        f = sim.function(sc, wrt=names)
        x0 = jnp.asarray(f.x0)
        out["f_build_s"] = time.perf_counter() - t0
        grad = jax.jit(jax.grad(lambda x: jnp.sum(jnp.log(f(x)["P"]) ** 2)))
        t0 = time.perf_counter()
        jax.block_until_ready(grad(x0))
        out["grad_first_s"] = time.perf_counter() - t0
        out["grad_s"] = best(lambda: grad(x0), k=2)
        out["f_s"] = best(lambda: f(x0)["P"], k=2)
        out["params"] = len(names)
        out["capacity"] = f.capacity
    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return out


def main():
    if len(sys.argv) > 1:
        print(json.dumps(run(int(sys.argv[1]))))
        return
    rows = []
    load = os.getloadavg()[0]
    for i in range(len(CASES)):
        res = subprocess.run(
            [sys.executable, __file__, str(i)], capture_output=True, text=True, check=True
        )
        rows.append(json.loads(res.stdout.strip().splitlines()[-1]))
        print(rows[-1], flush=True)

    def f(r, k, fmt="{:.3f}"):
        return fmt.format(r[k]) if k in r else "–"

    head = (
        "| model | states | scipy (s) | JAX dense (s) | JAX sparse (s) | sparse / scipy | "
        "first sparse solve incl. compile (s) | f(x) (s) | grad (s) | grad / scipy | "
        "build f + compile grad (s) | peak memory (GB) |"
    )
    lines = [head, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        ratio = r["sparse_s"] / r["scipy_s"]
        g = f"{r['grad_s'] / r['scipy_s']:.1f}" if "grad_s" in r else "–"
        build = f"{r['f_build_s'] + r['grad_first_s']:.1f}" if "grad_first_s" in r else "–"
        lines.append(
            f"| {r['label']} | {r['states']:,} | {r['scipy_s']:.3f} | {f(r, 'dense_s')} | "
            f"{r['sparse_s']:.3f} | {ratio:.2f} | {r['sparse_first_s']:.1f} | {f(r, 'f_s')} | "
            f"{f(r, 'grad_s')} | {g} | {build} | {r['peak_rss_gb']:.1f} |"
        )
    print("\n".join(lines))
    print(
        f"\nPython {platform.python_version()}, JAX {jax.__version__}; "
        f"{platform.processor() or platform.machine()}, CPU JAX; load average at the start "
        f"{load:.1f}"
    )


if __name__ == "__main__":
    main()
