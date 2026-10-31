# Reading and writing text

A model written in the notation is read with `tq.Model.from_text` (a string) or `tq.load`
(a `.tq` file); any model is written back with `m.to_text()` and typeset with
`m.to_latex()` (methods of [`tq.Model`](model.md)).

A single formula is read with `tq.formula`, with the same grammar, functions and messages as
in a model, for programs that use TenQSP's formula language outside a model (for example a
fitting part's covariate formulas). It lists the names the formula uses and gives a JAX
function of values for them:

```python
f = tq.formula("CL_pop * (WT/70)^0.75")
f.names                                 # ('CL_pop', 'WT')
g = f.jax(["CL_pop", "WT"])             # arguments in this order
g(0.04, jnp.array([60.0, 80.0]))        # works with jax.grad, jax.jit and jax.vmap
```

Outside a model every name is one value, so what works over a model's axes (shape functions,
`grad`, `k[s]`, a field at one end) is refused with a message.

!!! note "New in 0.7.0 (not yet released)"
    `tq.formula`, `f.names` and `f.jax` come with TenQSP 0.7.0.

::: tenqsp.load

::: tenqsp.table

::: tenqsp.formula
