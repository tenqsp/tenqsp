"""Step 4.4: what a derivative costs (sim.function, sim.sensitivity), against a plain forward
solve on the JAX backend, from a two-compartment PK model up to the toy phage model at 353
states.

Run alone on an idle machine. Each case runs in its own process, one after another:
python benchmarks/step4_4_gradients.py          # all cases
python benchmarks/step4_4_gradients.py 3        # one case (index into CASES)

Per case, with every parameter in ``wrt`` and the model's own doses:
- forward: one solve on the JAX backend, no ledger (steps for the solution only);
- f(x): one call of ``f = sim.function(...)``: the run with steps steered by the
  derivative too (design note step4_4d_api.md §6.3), then its replay;
- jacfwd: ``jax.jit(jax.jacfwd(f))``, every output by every parameter (``sim.sensitivity``);
- grad: ``jax.jit(jax.grad(sum of squares of f))``, reverse mode (an estimator's objective);
the best of 3 after a first call, which compiles (shown apart).
"""

from __future__ import annotations

import dataclasses
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import tenqsp as tq
from tenqsp.backends import jax_backend as jb
from tenqsp.examples import phage_latency_model, toy_phage_model
from tenqsp.gradients import _inputs

sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))

PK = """model pk
param ka = 1.2, CL = 4, V = 30, Q = 8, V2 = 60
compartment depot, centr, periph
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr - Q/V*centr + Q/V2*periph
d_t(periph) = Q/V*centr - Q/V2*periph
observable conc = centr/V
dose depot at t=0: 100
dose depot at t=12: 100
"""


def erlang(model: tq.Model) -> tq.Model:
    ir = model.to_ir()
    terms = tuple(
        dataclasses.replace(t, scheme="erlang") if t.kind == "advection" else t for t in ir.terms
    )
    return tq.Model.from_ir(dataclasses.replace(ir, terms=terms))


def friberg():
    from test_gallery_phase2 import friberg

    return friberg()


T24 = np.linspace(1, 24, 24)
CASES = [
    ("two-compartment oral PK, 2 doses", lambda: tq.Model.from_text(PK), T24),
    ("Friberg myelosuppression (at rest)", friberg, np.linspace(25, 500, 20)),
    ("phage latency, 40 age cells", lambda: erlang(phage_latency_model()), T24),
    ("toy phage 10 z x 12 a", lambda: erlang(toy_phage_model(10, 12)), T24),
    ("toy phage 16 z x 20 a", lambda: erlang(toy_phage_model(16, 20)), T24),
]


def best(fn, k=3):
    walls = []
    for _ in range(k):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        walls.append(time.perf_counter() - t0)
    return min(walls)


def first(fn):
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    return time.perf_counter() - t0


def run(i: int) -> dict:
    label, build, times = CASES[i]
    m = build()
    cm = m.compile()
    sc = tq.Scenario(times=times)
    names = list(cm.params.index)
    span = (0.0, float(times[-1]))

    def forward():  # no ledger, as f
        return jb.solve(cm, span, times=times, ledger=False)

    forward()  # compiles
    fwd = best(lambda: forward().y)
    steps_plain = forward().stats["n_steps"]
    # outputs: every state (what an objective would read)
    sim = m.simulator(outputs=list(cm.layout.names))

    t0 = time.perf_counter()
    f = sim.function(sc, wrt=names)
    x0 = jnp.asarray(f.x0)
    build_f = time.perf_counter() - t0
    first_f = first(lambda: f(x0))
    call_f = best(lambda: f(x0))
    # the steered run's steps (the probe f makes, at x0)
    probe = jb._program(cm, ledger=False, model_doses=True, max_steps=100_000, steer=True)
    args, dot = _inputs(x0, f._data, True)
    steps_steered = int(probe(**args, dot=dot)["n_steps"])

    jac = jax.jit(jax.jacfwd(f))
    first_jac = first(lambda: jac(x0))
    call_jac = best(lambda: jac(x0))

    def loss(x):
        return sum(jnp.sum(v**2) for v in f(x).values())

    grad = jax.jit(jax.grad(loss))
    first_grad = first(lambda: grad(x0))
    call_grad = best(lambda: grad(x0))
    # forward and reverse agree (a sanity check of the numbers timed)
    J = jac(x0)
    g_fwd = sum(jnp.einsum("t...,t...p->p", 2 * f(x0)[k], J[k]) for k in J)
    agree = float(jnp.max(jnp.abs(g_fwd - grad(x0))) / jnp.max(jnp.abs(g_fwd)))
    return {
        "label": label,
        "states": cm.n_states,
        "params": len(names),
        "fwd_s": fwd,
        "steps_plain": steps_plain,
        "steps_steered": steps_steered,
        "build_f_s": build_f + first_f,
        "f_s": call_f,
        "jac_first_s": first_jac,
        "jac_s": call_jac,
        "grad_first_s": first_grad,
        "grad_s": call_grad,
        "agree": agree,
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
        "| model | states | params | forward (s) | steps plain / steered | f(x) (s) | "
        "jacfwd (s) | grad (s) | grad / forward | compile f, jacfwd, grad (s) | "
        "forward = reverse |"
    )
    lines = [head, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['states']:,} | {r['params']} | {r['fwd_s']:.4f} | "
            f"{r['steps_plain']} / {r['steps_steered']} | {r['f_s']:.4f} | {r['jac_s']:.4f} | "
            f"{r['grad_s']:.4f} | {r['grad_s'] / r['fwd_s']:.1f} | {r['build_f_s']:.1f}, "
            f"{r['jac_first_s']:.1f}, {r['grad_first_s']:.1f} | {r['agree']:.0e} |"
        )
    print("\n".join(lines))
    print(
        f"\nPython {platform.python_version()}, JAX {jax.__version__}, numpy {np.__version__}; "
        f"{platform.processor() or platform.machine()}, CPU JAX"
    )


if __name__ == "__main__":
    main()
