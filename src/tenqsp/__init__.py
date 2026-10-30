# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""TenQSP: structured-population tensor PDEs for QSP, in JAX.

Importing the package enables JAX float64, which every TenQSP model requires.

    import tenqsp as tq
    m = tq.Model("my_model")
"""

import jax

jax.config.update("jax_enable_x64", True)

from . import plot  # noqa: E402  (matplotlib only when a plot is drawn)
from .backends.scipy_backend import SolverError  # noqa: E402
from .explore import BatchResult, Grid, PreRun, grid  # noqa: E402
from .expr import FormulaError  # noqa: E402
from .functions import (  # noqa: E402
    abs,
    acos,
    asin,
    atan,
    atan2,
    beta,
    ceil,
    cos,
    cosh,
    erf,
    exp,
    exponential,
    factln,
    factorial,
    floor,
    gamma,
    gammaln,
    grad,
    log,
    log10,
    logit,
    lognormal,
    max,
    min,
    normal,
    normcdf,
    point,
    probit,
    rem,
    sigmoid,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    uniform,
)
from .ir import ModelIR  # noqa: E402
from .lower import CompileError, MissingValues  # noqa: E402
from .model import Condition, Model, Slice, cases, eq, ne  # noqa: E402
from .notation import NotationError, load, table  # noqa: E402
from .regimen import Bolus, Infusion, Regimen, bolus, infusion, regimen_from_table  # noqa: E402
from .simulate import Scenario, SimResult, Simulator  # noqa: E402
from .validate import ModelError, validate  # noqa: E402

__version__ = "0.6.0"

__all__ = [
    "__version__",
    "Bolus",
    "Infusion",
    "Regimen",
    "Scenario",
    "BatchResult",
    "Grid",
    "PreRun",
    "grid",
    "plot",
    "SimResult",
    "Simulator",
    "bolus",
    "infusion",
    "regimen_from_table",
    "abs",
    "acos",
    "asin",
    "atan",
    "atan2",
    "beta",
    "cases",
    "ceil",
    "CompileError",
    "MissingValues",
    "Condition",
    "cos",
    "cosh",
    "eq",
    "erf",
    "exp",
    "exponential",
    "factln",
    "factorial",
    "floor",
    "FormulaError",
    "gamma",
    "gammaln",
    "grad",
    "log",
    "log10",
    "logit",
    "lognormal",
    "max",
    "min",
    "load",
    "Model",
    "ModelError",
    "ModelIR",
    "ne",
    "normal",
    "normcdf",
    "NotationError",
    "point",
    "probit",
    "rem",
    "sigmoid",
    "sin",
    "sinh",
    "Slice",
    "table",
    "SolverError",
    "sqrt",
    "tan",
    "tanh",
    "uniform",
    "validate",
]
