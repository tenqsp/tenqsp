# About TenQSP

TenQSP was created by **Dongwoo Chae, MD, PhD**, Yonsei University College of Medicine,
Seoul, Korea ([ORCID 0000-0002-7675-3821](https://orcid.org/0000-0002-7675-3821)).

## Why it exists

Pharmacometrics and quantitative systems pharmacology grew up on ordinary differential
equations. NONMEM, Monolix and the ODE solvers behind them made compartment models the
common language of the field, and that language has served drug development well.

But the populations these models describe are structured. Cells have an age, a maturation
stage, a level of dormancy, a position in a tissue, a genotype. When the structure matters,
the ODE modeller's only tool is a chain of compartments, and the chain hides the structure
in a choice of how many compartments to use. The equations that describe structure directly,
partial differential equations, have been used for decades in mathematical biology, but they
stayed out of reach of most pharmacometric work: writing them meant writing the numerics
too.

TenQSP was built to close that gap. It lets a modeller write a structured model the way a
paper writes it, as equations over axes, and does the numerical work: the grid, the fluxes,
the boundary conditions, the bookkeeping, the stiff solver and the derivatives. Its aim is
to let ODE modellers step into PDE modelling without first becoming numerical analysts, and
to make structured-population models an everyday part of pharmacometrics.

## Free for everyone

TenQSP is open source under the Apache License 2.0: free to use, modify and redistribute,
in academia and industry, at no cost. Copyright 2026 Yonsei University Health System. Contributions are welcome (see the repository's
contributing guide).

## How to cite

If you use TenQSP in your work, please cite it. Each release has a DOI on Zenodo; the
repository's `CITATION.cff` gives the reference in the form your reference manager needs.

## How it was made

TenQSP's code was written with AI assistance (Claude Code, Anthropic) under the author's
design, review and testing. Every numerical method is verified against analytical
solutions, hand-written ODE models or published results, and those checks run with every
change.
