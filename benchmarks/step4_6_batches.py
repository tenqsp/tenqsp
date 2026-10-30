"""Step 4.6: parameter sets in one call (``jax.vmap``), against one call per set, on both
tiers of the JAX backend, the toy phage model from 141 to 505 states (around the tiers'
threshold, 250), on the CPU, on several CPU cores (``jax.shard_map``) and on a GPU.

- forward: the solve alone (the JAX backend's program, no record; what a population
  simulation needs);
- function: ``sim.function`` (the run steered by the derivative, recorded, then replayed:
  what an estimator calls).

Run alone on an idle machine. Each case runs in its own process, one after another:
python benchmarks/step4_6_batches.py          # every CPU case
python benchmarks/step4_6_batches.py gpu      # every GPU case (an environment with CUDA JAX)
python benchmarks/step4_6_batches.py 3        # one case (index into CPU_CASES)
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

# (n_z, n_a, members, tier, what, cores): 141, 353 and 505 states
CPU_CASES = [
    (10, 12, 64, "dense", "forward", 1),
    (10, 12, 64, "sparse", "forward", 1),
    (16, 20, 64, "dense", "forward", 1),
    (16, 20, 64, "sparse", "forward", 1),
    (18, 26, 64, "dense", "forward", 1),
    (18, 26, 64, "sparse", "forward", 1),
    (10, 12, 64, "dense", "function", 1),
    (10, 12, 64, "sparse", "function", 1),
    (16, 20, 64, "dense", "function", 1),
    (16, 20, 64, "sparse", "function", 1),
    (10, 12, 64, "dense", "function", 64),  # one member per core
    (18, 26, 1000, "sparse", "forward", 1),  # target: 1,000 sets at ~500 states
]
GPU_CASES = [
    (10, 12, 64, "dense", "forward", 1),
    (10, 12, 1024, "dense", "forward", 1),
    (16, 20, 64, "dense", "forward", 1),
    (16, 20, 64, "sparse", "forward", 1),
    (18, 26, 1000, "dense", "forward", 1),  # target
    (10, 12, 64, "dense", "function", 1),
    (16, 20, 64, "dense", "function", 1),
    # the GPU's tier threshold (design decision 2, 2026-10-26): 601, 705, 793 and 913 states
    (20, 28, 64, "dense", "forward", 1),
    (20, 28, 64, "sparse", "forward", 1),
    (22, 30, 64, "dense", "forward", 1),
    (22, 30, 64, "sparse", "forward", 1),
    (22, 34, 64, "dense", "forward", 1),
    (22, 34, 64, "sparse", "forward", 1),
    (24, 36, 64, "dense", "forward", 1),
    (24, 36, 64, "sparse", "forward", 1),
    (18, 26, 64, "dense", "forward", 1),
    (18, 26, 64, "sparse", "forward", 1),
    (20, 28, 1000, "dense", "forward", 1),
    (20, 28, 1000, "sparse", "forward", 1),
]
LOOP_OVER = 16  # members timed one call at a time (per member: the mean)


def run(case) -> dict:
    nz, na, members, tier, what, cores = case
    if cores > 1:
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cores}"
    import jax
    import jax.numpy as jnp
    import numpy as np

    import tenqsp as tq
    from tenqsp.backends import jax_backend as jb
    from tenqsp.examples import toy_phage_model
    from tenqsp.gradients import _inputs

    def erlang(m):
        ir = m.to_ir()
        terms = tuple(
            dataclasses.replace(t, scheme="erlang") if t.kind == "advection" else t
            for t in ir.terms
        )
        return tq.Model.from_ir(dataclasses.replace(ir, terms=terms))

    jb.SPARSE_FROM_STATES = 10**9 if tier == "dense" else 0
    m = erlang(toy_phage_model(nz, na))
    cm = m.compile()
    times = np.linspace(2, 24, 12)
    f = m.simulator(outputs=["P"]).function(tq.Scenario(times=times), wrt=list(cm.params.index))
    x0 = np.asarray(f.x0)
    rng = np.random.default_rng(0)
    X = jnp.asarray(x0 * np.exp(0.2 * rng.standard_normal((members, x0.size))))
    if what == "function":

        def g(x):
            return f(x)["P"]

    else:
        program = jb._program(cm, ledger=False, model_doses=True, max_steps=100_000)

        def g(x):
            args, _ = _inputs(x, f._data, True)
            out = program(**args)
            return jnp.where(out["status"] == 0, out["ys"][: len(times)], jnp.nan)

    out = {
        "device": jax.devices()[0].platform,
        "states": cm.n_states,
        "tier": tier,
        "what": what,
        "members": members,
        "cores": cores,
    }
    if cores > 1:
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        mesh = Mesh(np.array(jax.devices()), ("p",))
        batch = jax.jit(
            jax.shard_map(
                jax.vmap(g), mesh=mesh, in_specs=P("p"), out_specs=P("p"), check_vma=False
            )
        )
    else:
        batch = jax.jit(jax.vmap(g))
    if jax.devices()[0].platform == "cpu":
        one = jax.jit(g)
        one(X[0]).block_until_ready()
        t0 = time.perf_counter()
        for x in X[:LOOP_OVER]:
            one(x).block_until_ready()
        out["loop_per_member"] = (time.perf_counter() - t0) / LOOP_OVER
    t0 = time.perf_counter()
    r = batch(X).block_until_ready()
    out["first_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    r = batch(X).block_until_ready()
    out["batch_s"] = time.perf_counter() - t0
    out["batch_per_member"] = out["batch_s"] / members
    out["failed"] = int(np.isnan(np.asarray(r)).reshape(members, -1).any(axis=1).sum())
    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    if out["device"] == "gpu":
        stats = jax.devices()[0].memory_stats() or {}
        out["peak_gpu_gb"] = stats.get("peak_bytes_in_use", 0) / 1e9
    return out


def main():
    args = sys.argv[1:]
    if args and args[0].isdigit():
        cases = GPU_CASES if len(args) > 1 and args[1] == "gpu" else CPU_CASES
        print(json.dumps(run(cases[int(args[0])])))
        return
    gpu = bool(args) and args[0] == "gpu"
    cases = GPU_CASES if gpu else CPU_CASES
    load = os.getloadavg()[0]
    rows = []
    for i in range(len(cases)):
        cmd = [sys.executable, __file__, str(i)] + (["gpu"] if gpu else [])
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        rows.append(json.loads(res.stdout.strip().splitlines()[-1]))
        print(rows[-1], flush=True)

    def f(r, k, fmt="{:.3f}"):
        return fmt.format(r[k]) if k in r else "–"

    head = (
        "| device | states | tier | what | members | cores | one call per member (s) | "
        "batch per member (s) | batch (s) | first batch incl. compile (s) | failed | "
        "peak memory (GB) |"
    )
    lines = [head, "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        mem = r.get("peak_gpu_gb", r["peak_rss_gb"])
        lines.append(
            f"| {r['device']} | {r['states']:,} | {r['tier']} | {r['what']} | {r['members']:,} | "
            f"{r['cores']} | {f(r, 'loop_per_member')} | {r['batch_per_member']:.3f} | "
            f"{r['batch_s']:.1f} | {r['first_s']:.1f} | {r['failed']} | {mem:.1f} |"
        )
    print("\n".join(lines))
    import jax

    print(
        f"\nPython {platform.python_version()}, JAX {jax.__version__}; "
        f"{platform.processor() or platform.machine()}; load average at the start {load:.1f}"
    )


if __name__ == "__main__":
    main()
