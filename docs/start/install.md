# Install

TenQSP needs Python 3.12 or later. It runs on Linux, macOS and Windows.

```bash
pip install tenqsp                 # the library
pip install "tenqsp[plot]"         # with plots (matplotlib)
pip install "tenqsp[xarray]"       # with xarray output
pip install "tenqsp[plot,xarray]"  # both
```

Keep the quotes (zsh, the default shell on macOS, reads the brackets otherwise) and leave no
space after the comma.

To work on TenQSP itself, from a clone of the repository:

```bash
conda env create -f environment.yml
conda activate tenqsp
python -m pip install -e ".[dev,xarray,plot]"
pytest                             # the fast tests, about a minute
```

## Check the installation

```python
import tenqsp as tq

m = tq.Model.from_text("""
model check
param k = 0.5
d_t(A) = -k*A
A(0) = 1
""")
sol = m.solve((0, 2), times=[2.0])
print(sol["A"][-1])                # exp(-1) = 0.3679
```

## What gets installed

TenQSP is built on [JAX](https://jax.readthedocs.io) (arrays, compilation, derivatives),
[SciPy](https://scipy.org) (the default solver and sparse linear algebra) and
[SymPy](https://www.sympy.org) (reading formulas). Importing TenQSP switches JAX to 64-bit
floating point, which every model needs.

One solve is CPU work: the scipy solver, and linear algebra too small for a GPU to speed up.
A GPU pays for a batch of scenarios run as one JAX function, with
[`sim.function([sc1, sc2, ...], wrt=...)`](../guide/gradients.md#many-scenarios-at-once) or,
to simulate a population,
[`derivatives=False`](../guide/gradients.md#simulating-populations-derivativesfalse). On the
[Performance](../background/performance.md) page, 1,024 scenarios of a 141-state model took
0.011 s each on an RTX A6000, against 0.15 s each one by one on a CPU core. Many CPU cores
close much of that gap (64 scenarios with derivatives on 64 cores: 0.02 s each), so a GPU is
an option, not a requirement.

To use an NVIDIA GPU (Linux, or WSL2 on Windows, with a driver for CUDA 12), install TenQSP
and then:

```bash
pip install -U "jax[cuda12]"
```

Without a GPU, JAX prints a one-line note that it falls back to the CPU; that note needs no
action.
