# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Forward sensitivities or record-and-replay: where one beats the other (sim.function,
derivatives="forward" against "reverse"), by number of states and of names in ``wrt``.

Run alone on an idle machine. Each case runs in its own process, one after another (the
record's room only grows within a process, so cases must not share one):
python benchmarks/forward_or_reverse.py                    # all cases
python benchmarks/forward_or_reverse.py forward 100 20     # one case: mode, states, names

The model is a transit chain of n compartments, a bolus into the first; compartment i
leaves at rate k_{group(i)}, the n compartments split into p groups (p parameters, all in
``wrt``). Per case, rtol 1e-6, 12 output times:
- solve: one call of ``sim.function(..., derivatives=False)``;
- grad: ``jax.jit(jax.grad(sum of squares of the output))``;
- jacobian: ``jax.jit(jax.jacfwd(output))``, every output time by every parameter;
the mean of 5 calls after two first calls (the first compiles, shown apart).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

CASES = [(5, 5), (10, 10), (20, 20), (40, 40), (100, 5), (100, 10), (100, 20), (100, 50),
         (100, 100), (400, 10), (400, 50)]  # fmt: skip


def one(mode: str, n: int, p: int) -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    import numpy as np

    import tenqsp as tq

    jax.config.update("jax_enable_x64", True)
    grp = [i * p // n for i in range(n)]
    vals = [(n / 8) * 2 ** (((j * 7) % 5 - 2) / 4) for j in range(p)]
    lines = ["model chain", "param " + ", ".join(f"k{j + 1}={vals[j]:.4g}" for j in range(p))]
    for i in range(n):
        out = f"k{grp[i] + 1}*x{i + 1}"
        inflow = "" if i == 0 else f"k{grp[i - 1] + 1}*x{i} "
        lines.append(f"d_t(x{i + 1}) = {inflow}- {out}")
    lines.append(f"observable out = x{n} + x{max(1, n // 2)}")
    m = tq.Model.from_text("\n".join(lines) + "\n")
    sim = m.simulator(outputs=["out"], rtol=1e-6, atol=1e-8)
    sc = tq.Scenario(regimen=tq.bolus("x1", 100.0, times=[0.0]), times=np.linspace(1, 24, 12))
    names = [f"k{j + 1}" for j in range(p)]
    f = sim.function(sc, wrt=names, derivatives=mode)
    f0 = sim.function(sc, wrt=names, derivatives=False)
    x = jnp.asarray(np.asarray(f.x0))

    def clock(fn, reps=5):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        first = time.perf_counter() - t0
        jax.block_until_ready(fn(x))
        t0 = time.perf_counter()
        for _ in range(reps):
            jax.block_until_ready(fn(x))
        return 1e3 * (time.perf_counter() - t0) / reps, first

    v, _ = clock(jax.jit(lambda x: f0(x)["out"]))
    g, gc = clock(jax.jit(jax.grad(lambda x: jnp.sum(f(x)["out"] ** 2))))
    J, jc = clock(jax.jit(jax.jacfwd(lambda x: f(x)["out"])))
    print(f"{mode:8s} n={n:4d} p={p:3d}  solve {v:7.1f} ms  grad {g:8.1f} ms  jacobian "
          f"{J:8.1f} ms  (compile {gc:.0f}/{jc:.0f} s)", flush=True)  # fmt: skip


if __name__ == "__main__":
    if len(sys.argv) == 4:
        one(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    else:
        for n, p in CASES:
            for mode in ("forward", "reverse"):
                subprocess.run([sys.executable, __file__, mode, str(n), str(p)], check=True)
