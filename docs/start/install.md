# Install

TenQSP needs Python 3.12 or later. It runs on Linux, macOS and Windows.

```bash
pip install tenqsp                 # the library
pip install "tenqsp[plot]"         # with plots (matplotlib)
pip install "tenqsp[xarray]"       # with xarray output
```

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

TenQSP runs on the CPU. A GPU helps only for large batches of small models (see
[Simulation and exploration](../guide/simulation.md)); to use one, install a CUDA build of
`jaxlib` as the JAX documentation describes. Without one, JAX prints a one-line note that it
falls back to the CPU.
