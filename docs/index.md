# TenQSP

**Structured-population PDEs for quantitative systems pharmacology, in JAX.**

Pharmacometric and QSP models describe populations: drug molecules, bacteria, blood cells,
tumour cells. Often a population is not uniform. Cells differ in **age**, **maturation**,
**dormancy**, **depth** in a tissue, or **genotype**, and that difference decides how they
respond to a drug. The usual remedy is a chain of compartments, one per stage, which hides
the structure in a choice of how many compartments to use.

TenQSP lets you write the structure itself. A population becomes a **field** over **axes**:

- **continuous axes**: age since infection, maturation, dormancy level, depth into a
  biofilm, time since swallowing;
- **categorical axes**: phenotype, susceptibility to each drug or phage, organ, tissue space.

Fields sit next to ordinary compartments (plasma, a depot, free phage), and the model is
written as equations, the way a paper writes them:

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

TenQSP turns this into a system of ODEs by the finite-volume method, works out which states
affect which (the sparse Jacobian), and solves it with a stiff solver. The same model can be
built step by step in Python, solved with scipy or with TenQSP's own solver in JAX, run over
many scenarios, and differentiated with respect to its parameters, dose amounts and dose
times.

## What it gives you

- **Equations as in a paper.** Transport `-d_z(v*F)`, diffusion `d_z(D*d_z(F))`, losses,
  transitions between categories, and boundary conditions written under their equation.
  `m.to_latex()` prints the equations for an article; `m.explain()` says in words what each
  term means and how it is computed.
- **Books that balance.** Every cell, molecule or phage is accounted for: a ledger records
  what entered, left, died and was dosed, and closes to rounding.
- **Nothing hidden.** No value is clipped or capped behind your back; every numerical
  safeguard is listed by `explain()`.
- **One build, many runs.** Parameter values, dose amounts and times, and output times are
  inputs: changing them never rebuilds the model.
- **Gradients.** Exact derivatives of the computed solution with respect to parameters and
  dosing, through JAX, for estimation and sensitivity analysis.

## How to read this site

| If you want to | Read |
|---|---|
| install TenQSP and run a model | [Install](start/install.md), then [A first model](start/first-model.md) |
| know why a QSP model needs axes at all | [Why PDEs in QSP](start/why-pdes.md) |
| write your own model | the [User guide](guide/index.md), one chapter per task |
| see complete models with checks against known results | the [Tutorials](tutorials/index.md) |

TenQSP was created by Dongwoo Chae, Yonsei University College of Medicine
([About](project/about.md)). Its code was written with AI assistance (Claude Code,
Anthropic) under the author's design, review and testing.
