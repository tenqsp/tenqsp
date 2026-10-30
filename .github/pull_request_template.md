**What this changes and why**

**How it is tested**

<!-- For numerics: what known result (analytical solution, hand-written ODE model, published
result) the new test compares against. -->

- [ ] `ruff check src tests && ruff format --check src tests` passes
- [ ] `pytest` and `pytest --backend=jax` pass
- [ ] every commit is signed off (`git commit -s`; see CONTRIBUTING.md)
- [ ] the documentation and CHANGELOG.md are updated, if users will notice the change
