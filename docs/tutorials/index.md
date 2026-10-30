# Tutorials

Each tutorial builds one model from a published or textbook problem and checks every step
against a known result: a closed-form solution, an ODE written out by hand, or a published
figure. They are ordered from the familiar (transit compartments) to the less familiar
(fronts and continua).

| | Tutorial | What it shows |
|---|---|---|
| 1 | [Transit compartments](01_transit_compartments.ipynb) | a transit chain *is* an age axis; what the number of compartments controls |
| 2 | [Where in the gut](02_where_in_the_gut.ipynb) | stomach by time since swallowing, small intestine by position; absorption that depends on where the drug is |
| 3 | [Myelosuppression](03_myelosuppression.ipynb) | the Friberg model, its transit chain as a maturation axis |
| 4 | [Lifespan response](04_lifespan_response.ipynb) | lifespan-based indirect response (Krzyzanski et al.): cells that live a fixed time |
| 5 | [Hepatic dispersion](05_hepatic_dispersion.ipynb) | between the well-stirred and parallel-tube models: transport plus dispersion along the sinusoid |
| 6 | [Whole-body PBPK](06_whole_body_pbpk.ipynb) | organs and tissue spaces as categorical axes, tables, fluxes |
| 7 | [Infection age](07_infection_age.ipynb) | HIV dynamics with an eclipse phase: infected cells by time since infection, renewal at age 0 |
| 8 | [Age-structured population](08_age_structured_population.ipynb) | the McKendrick–von Foerster equation |
| 9 | [Persisters and dormancy](09_persisters_and_dormancy.ipynb) | biphasic kill curves; from a two-state switch to a dormancy axis |
| 10 | [Biofilm penetration](10_biofilm_penetration.ipynb) | diffusion into depth, exchange with the liquid |
| 11 | [Phage plaque fronts](11_phage_plaque_fronts.ipynb) | a plaque's front speed; infection age × space, the latent period slows the front |
| 12 | [Drug tolerance continuum](12_drug_tolerance_continuum.ipynb) | tolerance as a continuum; selection and drug-driven drift |

Each tutorial is a plain Python file in the repository (`docs/tutorials/`), run by the test
suite, so it cannot go out of date. To open one as a notebook:
`python docs/tutorials/build.py 01` writes `docs/tutorials/01_transit_compartments.ipynb`.
