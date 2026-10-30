# How TenQSP is checked

A simulator is only useful if its answers can be trusted. This page says, in plain words, what
evidence there is that TenQSP solves the equations you write, and solves them correctly. The
evidence is a test suite: a fast tier of about 1,200 tests (1,203 at the time of writing),
plus 66 longer ones (convergence studies, long runs and the tutorials).
Almost every test compares TenQSP with something that does not come from TenQSP's own code: a
formula from the mathematics, an independently written ODE model, a published result, or a
second method.

## Exact solutions

Where an equation has a solution in closed form, the computed solution is compared with it.
Among them:

| Problem | What is compared |
|---|---|
| decay, logistic growth, repeated oral doses (Bateman superposition), infusions | the whole time course, to the solver's tolerance |
| one-compartment oral PK with a lag time and bioavailability (NONMEM's ALAG and F) | concentrations, and their derivatives with respect to every parameter |
| ageing cells with a constant inflow at age 0 and a constant death rate (McKendrick–von Foerster) | the steady age profile \((J/v)\,e^{-\mu a / v}\) |
| an age-structured population with births (renewal) | the stable age profile \(e^{-(\mu + r)a}\) and the growth rate \(r\), the root of the Euler–Lotka equation |
| ageing with an age-dependent death rate | the profile along the characteristics, \(n_0(a - t)\,S(a)/S(a - t)\), on refined grids |
| a cosine mode diffusing between sealed ends | its decay rate \(D(\pi/L)^2\) |
| drift and diffusion with sealed ends (Fokker–Planck), chemotaxis up a fixed attractant | the zero-flux steady states, \(n \propto e^{\int a/D}/D\) and the Boltzmann profile \(e^{\chi S/D}\) |
| a reacting slab (drug penetrating a biofilm) | the cosh profile and the effectiveness factor; with exchange to a bulk liquid, the series-resistance steady state |
| advection–diffusion–decay with exchange at the ends | the two-exponential steady profile |
| Fisher–KPP spreading | the exact travelling wave \((1 + e^{(x - ct)/\sqrt 6})^{-2}\) and its speed \(c = 5/\sqrt 6\) |
| Keller–Segel aggregation | the growth rate of a perturbation, the largest eigenvalue of a 2 × 2 matrix |
| switching between active and persister cells under a drug | the biphasic kill curve, from the matrix exponential |
| mutation at division | the mutant count, supplied at the **parent's** division rate |
| an extinction threshold | the crossing time \(\ln(N_0/\text{threshold})/k\) of an exponential decay, and the regrowth of a population fed from \(10^{-12}\) |
| a tumour population on a continuous resistance trait, with selection, drift and phenotypic diffusion | the mean, variance and size of its Gaussian profile, from their own ODEs |

## Manufactured solutions and order of convergence

Closed forms exist for few problems, and rarely with stretched grids, varying coefficients or
every kind of boundary. The **method of manufactured solutions** fills the gap. Choose a
smooth function, for example \(u^*(x) = 1 + \tfrac12\sin 2x + x^2\) with a diffusivity
\(D(x) = 0.2(1 + x)\); put it into the equation; whatever is left over is the source term that
makes \(u^*\) the exact steady solution. That source is derived symbolically, added to the
model, and the model is solved on grids of 10, 20, 40, … cells.

The error should then fall at the scheme's **order**: halving the cells divides it by 2 for a
first-order scheme and by 4 for a second-order one. The **observed order**, measured from
successive grids, is compared with the design order:

- diffusion: order 2, with held values, exchange (Robin) and flux conditions at the ends, on
  uniform and stretched grids;
- transport with `erlang`: order 1;
- transport with `sharp`: order 2 in the average error, 1 in the largest (the limiter and the
  outflow cell are first order by design);
- transport plus diffusion, with exchange at the ends, and chemotaxis velocities: the order of
  their transport scheme.

These run through the whole model, for every combination of operator, boundary condition and
scheme, not only on the operators in isolation. The same idea checks the solver itself: each
BDF order from 1 to 5 must converge at its order.

## Hand-written ODE models, side by side

Many structured models have an equivalent written as ordinary ODEs. The two are solved side by
side, the ODE version with scipy at tight tolerances:

- an `erlang` axis of \(N\) cells and an \(N\)-compartment transit chain (to \(10^{-10}\)), and the
  exit-time distribution, a gamma with \(N\) stages;
- the whole-body PBPK model (9 organs × 3 tissue spaces) and the same model written by hand as
  30 ODEs in amounts;
- a model with two phages and resistance on two axes, written by hand as a flat ODE system;
- the toy phage model's right-hand side, assembled by hand, entry by entry;
- the infection-age HIV model with constant rates and the standard three-ODE
  target-cell-limited model.

## The conservation ledger

The [ledger](../guide/explain.md#the-ledger-where-every-amount-came-from) books every amount
that enters or leaves each field and compartment: each reaction term, each flux through an
end, each dose, each extinction. Change in amount minus the sum of the channels should be
zero. The tests check that this unexplained remainder stays at the level of rounding (below
\(10^{-10}\) relative, typically about \(10^{-15}\)) on models of every kind: transport, diffusion, transitions, PBPK,
doses into fields, extinctions, conditions. They also check individual channels against values
written by hand from the equations. Every solution carries its own ledger, and
`m.diagnostics(sol)` reports the largest unexplained amount for your own runs.

## The sparse Jacobian against a dense one

The Jacobian the solver uses is sparse, with a pattern TenQSP derives from the model's
equations (see [The numerical method](method.md#the-sparse-jacobian)). A missed dependency
would put a derivative in the wrong place. So the whole test suite can run with a check on
**every model any test builds**, up to 2,000 states: the sparse Jacobian must equal the dense
Jacobian computed by JAX directly, at the starting state and at a random positive state, with
and without the ledger, to \(10^{-10}\) relative.

## Two solvers that agree

The JAX solver is a port of scipy's BDF, so on the same model the two must take the same
steps. The tests check that they do (the same number of steps, Jacobians and factorisations)
and that the answers agree, to about \(10^{-13}\) on smooth models. The whole fast test suite is
also run with every solve on the JAX solver: every check of every test must then pass with
the JAX solver too.

## A model is the same as its text

Every model can be written as text (`m.to_text()`) and read back (`tq.Model.from_text`). The
two must be the **same model**, compared in meaning: the same equations, boundary conditions,
doses and values, whatever the order or the way a formula was written. The test suite can run
with this check on every model any test builds; writing the text a second time must give the
same text. Forty-one models were also written by hand in the notation, from its description
alone, and must be identical to their Python versions. Read from text, the benchmark models
give the same solutions as their Python versions (bit for bit, or to the last digits) with the
same number of solver steps.

## Derivatives against finite differences

Derivatives with respect to the parameters ([Gradients and sensitivities](../guide/gradients.md))
are checked four ways:

- against closed-form PK solutions differentiated exactly;
- against central finite differences of full solves;
- forward mode against reverse mode, which must agree to rounding;
- identities that must hold whatever the model's values: for one-compartment PK,
  \(\partial \log C/\partial \log CL + \partial \log C/\partial \log V = -1\).

With the JAX solver, the whole suite can also be run with every run recorded and replayed (the
replay must reproduce the run to rounding) and with the derivative of every distinct model of
up to 200 states checked against finite differences. Dose times, lag times and switch times
are among the parameters checked.

## Published models reproduced

A gallery of models from the pharmacometric and mathematical-biology literature is rebuilt in
TenQSP and checked against the published equations or their known results:

| Model | Checked against |
|---|---|
| Savic et al. 2007, transit-compartment absorption | the chain of transit compartments and Savic's analytic input rate (an Erlang density); its integral, the incomplete gamma function |
| Friberg et al. 2002, myelosuppression | the published five-ODE model, through the nadir and the rebound; the steady state without drug |
| Krzyzanski et al. 1999, lifespan-based indirect response | the ramp-then-plateau response to a step in production |
| Compartmental absorption and transit (Yu & Amidon) | 7 cells as the 7 CAT compartments and its fraction absorbed \(1 - (1 + K_a/K_t)^{-7}\); more cells tend to plug flow |
| dispersion model of absorption (Ni et al. 1980), Danckwerts ends | the closed-form fraction absorbed |
| Sangren & Sheppard 1953, capillary–tissue exchange | the mean transit time on every grid; the variance converging at second order with `sharp` |
| physiological oral absorption with Weibull gastric emptying | the Weibull emptying curve; regional absorption falling along the intestine |
| whole-body PBPK, permeability-limited | infusion steady state = rate / hepatic clearance; AUC = dose / clearance; partition equilibria; the perfusion-limited model as permeability grows |
| Nelson et al. 2004, HIV infection age | the basic reproduction number by quadrature, and the early growth rate as the root of the renewal equation |
| McKendrick–von Foerster age structure | the stable age distribution and the Euler–Lotka growth rate |
| Balaban et al. 2004, persisters | the biphasic kill curve |
| Nielsen et al. 2007, drug-induced resistance switching | a switch rate that depends on the total population |
| phage plaque fronts (Yin & McCaskill limit and with latency) | the well-mixed growth rate, and front speeds from the linear theory at the leading edge, \(c^* = 2\sqrt{Dr}\) without latency |
| Lorenzi, Chisholm & Clairambault 2016, tumour drug tolerance | the moment equations of a Gaussian population |

The parameter values in these checks are illustrative unless the paper gives them: the tests
check the mathematics, not a fit.

## Other checks

- **Compile once.** A hundred parameter sets, dose amounts included, compile every function
  exactly once.
- **Messages.** Mistakes in a model (a misspelt name, a flux that does not balance, a missing
  boundary condition) give plain messages that say what to do; the messages are tested.
- **References complete.** Every statement and function of the notation, and every public
  name of the Python API, is documented in the [reference](../reference/notation.md).
- **The site's examples.** Every code example on this site runs as part of the test suite,
  page by page, and each pair of Notation and Python tabs must build the same model. Every
  [tutorial](../tutorials/index.md) runs too, in the longer tier; each checks its own results
  against closed forms or published models.

Before each release the suite runs on Linux, Windows and macOS with Python 3.12 and 3.13, with
the round-trip and Jacobian checks on every model, the longer tests, and the fast tests on
the JAX solver with every run replayed.
