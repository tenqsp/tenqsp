# TenQSP

[![CI](https://github.com/tenqsp/tenqsp/actions/workflows/ci.yml/badge.svg)](https://github.com/tenqsp/tenqsp/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Structured-population PDEs for quantitative systems pharmacology, in JAX.**

Pharmacometric and QSP models describe populations: drug molecules, bacteria, blood cells,
tumour cells. Often a population is not uniform. Cells differ in age, maturation, dormancy,
depth in a tissue or genotype, and that difference decides how they respond to a drug. The
usual remedy is a chain of compartments, which hides the structure in a choice of how many
compartments to use.

TenQSP lets you write the structure itself. A population becomes a **field** over **axes**,
continuous (age, maturation, depth) or categorical (phenotype, organ), next to ordinary
compartments, and the model is written as equations, the way a paper writes them:

```tq
model delayed_pk
axis z in [0, 1] n=20 kind=one_way      # progress through the gut: 0 swallowed, 1 absorbed
param MTT=1.5, ka=2, CL=4, V=30
field G(z)                               # drug in transit, by progress
d_t(G) = -d_z(G/MTT)  using sharp        # moves along z at speed 1/MTT
    at z=1: outflow
d_t(depot) = G(z=1)/MTT - ka*depot
d_t(centr) = ka*depot - CL/V*centr
dose G at t=0 to 48 every 12: 100*point(z, at=0)
observable conc = centr/V
```

```python
import tenqsp as tq

m = tq.load("delayed_pk.tq")
sol = m.solve((0, 72), times=range(73))
sol["conc"]
```

TenQSP turns the model into a system of ODEs by the finite-volume method, works out which
states affect which (the sparse Jacobian), and solves it with a stiff solver: scipy's, or
TenQSP's own in JAX. The same model can be built step by step in Python, run over many
scenarios, and differentiated with respect to its parameters, dose amounts and dose times.

- **Equations as in a paper**, with boundary conditions under their equation;
  `m.to_latex()` prints them for an article, `m.explain()` says what each term means.
- **Books that balance**: a ledger accounts for every cell, molecule or phage.
- **Nothing hidden**: no value is clipped or capped behind your back.
- **One build, many runs**: changing parameters, doses or output times never rebuilds.
- **Exact gradients** through JAX, for estimation and sensitivity analysis.

## Install

```bash
pip install tenqsp
```

TenQSP needs Python 3.12 or later. For plots add `tenqsp[plot]`; for labelled array output,
`tenqsp[xarray]`.

## Documentation

**<https://tenqsp.github.io/tenqsp/>**: start here, the user guide, twelve tutorials, and
the notation and Python references.

## Cite

If you use TenQSP in your work, please cite it: see [`CITATION.cff`](CITATION.cff), or
"Cite this repository" on GitHub.

## Licence and credit

TenQSP is free and open source under the [Apache License 2.0](LICENSE). It was created by
Dongwoo Chae, MD, PhD, Yonsei University College of Medicine
([ORCID 0000-0002-7675-3821](https://orcid.org/0000-0002-7675-3821)); copyright 2026 Yonsei
University Health System (see [`NOTICE`](NOTICE)). Its code was written with AI assistance
(Claude Code, Anthropic) under the author's design, review and testing.

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md).
