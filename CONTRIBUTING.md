# Contributing to TenQSP

Thank you for helping. TenQSP is free for everyone, and contributions of every size are
welcome: a report of something that confused you, a fix to the documentation, a new example,
or a change to the library.

## Reporting a problem or asking for something

Open an issue. For a problem, include the smallest model that shows it (the notation text or
the Python code), what you expected, what happened, and the output of
`python -c "import tenqsp, jax; print(tenqsp.__version__, jax.__version__)"`.

## Setting up

```bash
git clone https://github.com/tenqsp/tenqsp
cd tenqsp
conda env create -f environment.yml
conda activate tenqsp
python -m pip install -e ".[dev,xarray,plot,docs]"
pytest
```

## Before you open a pull request

```bash
ruff check src tests && ruff format --check src tests
pytest                       # the fast tests, a few minutes
pytest --backend=jax         # the same on TenQSP's own solver
mkdocs build --strict        # if you changed the documentation
```

- **Every change to the numerics comes with a test against something known**: an
  analytical solution, an equivalent ODE model written by hand, or a published result. A
  test that only repeats what the code computes does not count.
- **Error messages are for modellers.** Say what is wrong in the model's own terms, on which
  line, and what to write instead.
- **Nothing hidden.** No value may be clipped, capped or switched behind the user's back;
  any numerical safeguard must appear in `explain()`.
- New source files start with the licence header:
  ```python
  # SPDX-FileCopyrightText: 2026 Yonsei University Health System
  # SPDX-License-Identifier: Apache-2.0
  ```

## Sign your commits (Developer Certificate of Origin)

TenQSP uses the [Developer Certificate of Origin](https://developercertificate.org/) (DCO)
instead of a contributor licence agreement. By adding a `Signed-off-by` line to a commit,
you certify that you wrote the change, or otherwise have the right to submit it, under the
project's licence (Apache-2.0). Git adds the line for you:

```bash
git commit -s -m "Explain what the change does"
```

which ends the message with `Signed-off-by: Your Name <your.email@example.org>`, using your
real name. Pull requests whose commits are not all signed off cannot be merged. To sign off
commits you already made: `git rebase --signoff main`, then push again.

## Licence and credit

Contributions are licensed under the Apache License 2.0, like the rest of TenQSP.
Contributors are credited in the changelog and in the git history. TenQSP was created by
Dongwoo Chae; that credit, in the `NOTICE` file, stays with every copy.

## Code of conduct

Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).
