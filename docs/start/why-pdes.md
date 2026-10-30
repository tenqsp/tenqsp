# Why PDEs in QSP

## Populations have structure

A dose of drug, a culture of bacteria, a patient's neutrophils: each is a population, and
its members are not all alike.

- An infected bacterium bursts some time after infection. Whether phage are released now
  depends on **how long ago** each cell was infected.
- A neutrophil precursor matures for days before it reaches the blood. A cytotoxic drug that
  hits the marrow today shows in the blood **after the maturation time**.
- A persister cell wakes slowly from deep dormancy and quickly from shallow dormancy. An
  antibiotic that kills only growing cells clears a population at the pace set by **how
  dormancy is distributed**.
- A drug diffusing into a biofilm reaches the surface first. Cells **deep inside** see less
  of it, and later.
- A bacterium resistant to one phage may still be killed by another. With two phages, each
  cell has a **genotype** on two axes.

Each of these is a variable that describes a member of the population, besides time: age
since infection, maturation, dormancy level, depth, genotype. A model that keeps track of
the population **along** such a variable is a structured-population model, and its
equations are partial differential equations (PDEs) in time and that variable.

## Compartment chains are PDEs in disguise

Pharmacometrics already handles structure, with chains of compartments. Take the transit
compartments of an absorption delay (Savic et al. 2007): \(n\) compartments, each passing
its content to the next at rate \(k_{tr}\),

$$
\frac{dA_i}{dt} = k_{tr} A_{i-1} - k_{tr} A_i .
$$

Now take one quantity \(u(t, z)\) spread along an axis \(z\) of progress, moving at
speed \(v\):

$$
\frac{\partial u}{\partial t} = -\frac{\partial (v\,u)}{\partial z} .
$$

Cut \(z\) into cells of width \(\Delta z\) and let each cell pass on what it holds (the
simplest "upwind" finite-volume scheme). The equation of cell \(i\) is then exactly the
transit chain, with \(k_{tr} = v/\Delta z\). **A chain of \(n\) transit compartments is the
transport equation computed on \(n\) cells.**

Seeing it this way separates two things the chain mixes together:

- **the biology**: the delay has a mean (the mean transit time) and a spread;
- **the numerics**: the number of cells.

In the chain, the number of compartments sets the spread (the delay is gamma distributed
with coefficient of variation \(1/\sqrt{n}\)). So "how many transit compartments?" is
really "how variable is the delay?", answered by changing the computation. With the PDE you
state the spread you mean, as a diffusion along the axis or as a sharp delay, and refining
the grid then only makes the answer more accurate. [Tutorial 1](../tutorials/01_transit_compartments.ipynb)
shows both.

## What the equations add

Written as a PDE, a structured model can say things a chain says only awkwardly:

- **rates that depend on the structure variable**: a lysis hazard that rises with infection
  age, a kill rate that falls with dormancy depth;
- **renewal at an end**: newly infected cells enter at age 0, at the rate infection happens;
- **diffusion**: drug spreading into tissue, cells drifting in and out of dormancy at random;
- **several axes at once**: bacteria by dormancy level × genotype; infected cells by
  infection age × the host's genotype; a drug by organ × tissue space;
- **boundary conditions**: a surface held at the bulk concentration, an exchange with the
  surrounding liquid, a sealed base.

These are the equations of published models: age-structured cell populations, the
McKendrick–von Foerster equation, reaction–diffusion in tissue, Fokker–Planck descriptions of
cell-to-cell variability.

## Why this needs a library

A PDE model on a grid has many more states than a compartment model: 20 cells of dormancy
× 4 genotypes × 30 cells of infection age is already thousands. Writing it by hand in
NONMEM, Monolix or an ODE package means writing the grid, the fluxes between cells, the
boundary conditions and the bookkeeping yourself, and getting them all right.

TenQSP does that part:

- you write the equations, as in a paper; TenQSP builds the finite-volume system, in a form
  that keeps every cell's books balanced;
- it finds which states affect which (most do not), so a stiff solver can handle tens of
  thousands of states;
- it compiles the model once; new parameter values and doses never rebuild it;
- it gives exact derivatives of the solution through JAX, for estimation and sensitivity.
