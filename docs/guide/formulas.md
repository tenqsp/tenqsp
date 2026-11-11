# Models given by formulas

Some models need no differential equations: the outputs are known formulas of time. Examples
are a disease-progression line, a closed-form concentration after a single dose, or an
exponential tumour growth curve. TenQSP evaluates such a model at the output times directly,
with no solver. This is what NONMEM calls `$PRED`.

## Writing one

A model with no equation `d_t(...)` is a model given by formulas. Its outputs are observables,
which may use the parameters, `t`, named formulas, conditions (`if ... else`), `pi`, and
parameters per category.

=== "Notation"

    ```tq
    model progression
    param E0 = 50, prog = 2
    observable E = E0 + prog*t
    ```

=== "Python"

    ```python
    import tenqsp as tq

    m = tq.Model("progression")
    m.param("E0", 50)
    m.param("prog", 2)
    m.observable("E", "E0 + prog*t")
    ```

It runs like any model:

```python
sol = m.solve((0, 52), times=[0, 4, 12, 26, 52])
sol["E"]            # 50, 58, 74, 102, 154
```

`m.solve`, `tq.Simulator`, `sim.function(..., wrt=...)`, batches and `sim.sensitivity` all work
as for a model with equations. The derivatives are exact, and `jax.jit` and `jax.vmap` work.
Tolerances are accepted and change nothing.

## Doses

TenQSP adds nothing PK-specific, so a formula model has no compartments to dose. A dose
enters as a parameter of the formula. For a single oral dose:

```tq
model theoph_sd
param D = 320, ka = 1.5, V = 0.5, ke = 0.09
observable conc = D*ka/(V*(ka - ke))*(exp(-ke*t) - exp(-ka*t))
```

To simulate another dose, change `D` (`params={"D": 160}`). When the model is fitted,
the fit sets `D` from a data column, one value per subject. Several doses are written into the
formula, for example a second dose at 12 h with a condition:

```tq
model theoph_two_doses
param D = 320, ka = 1.5, V = 0.5, ke = 0.09
a = D*ka/(V*(ka - ke))
first = a*(exp(-ke*t) - exp(-ka*t))
second = a*(exp(-ke*(t - 12)) - exp(-ka*(t - 12))) if t >= 12 else 0
observable conc = first + second
```

Doses (`dose`, `tq.bolus`), starting
values `X(0)`, lags, bioavailability and extinction groups all need a state, so a formula model
refuses them, with a message saying what to do instead. `sim.pre_run` is refused too: there is
no state to run ahead to.

## Reading it back

`m.explain()` lists the outputs and says that no solver runs. `m.to_latex()` writes each output
as an equation, $E(t) = E_0 + \mathrm{prog}\,t$. `m.to_text()` writes the model as above.

A model needs something to compute: with no equation and no observable, it is refused.
