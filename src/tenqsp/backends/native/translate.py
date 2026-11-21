# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""JAX programs as numba functions (ADR 0011): the model's right-hand side, its Jacobians,
its initial state, its doses and its outputs are written once, in JAX; the native backend
translates each one's jaxpr into Python source that numba compiles.

- **Constants are folded.** An equation whose inputs do not depend on the function's inputs
  (the model's geometry, literals, indices built from them) is evaluated once, with JAX, at
  translation; its value is a constant of the generated code.
- **Gathers and scatter-adds with constant indices** (indexing by fixed cells: the only kind
  in TenQSP's programs) become index maps: the source of each output element (a gather of
  ``0, 1, 2, ...``), or the target of each update (the gather that is the scatter's
  transpose), applied by a loop.
- **Rank-0 values are numba scalars;** arrays are numpy arrays, broadcast explicitly.
- A primitive not listed raises ``NotTranslatable`` with its name: the native backend is then
  not available for that model, and the JAX backend runs it.
"""

from __future__ import annotations

import itertools

import jax
import jax.extend as jex
import jax.numpy as jnp
import numpy as np

__all__ = ["NotTranslatable", "source", "translate"]


class NotTranslatable(NotImplementedError):
    """A JAX program the native backend cannot translate."""


_BINARY = {"add": "+", "add_any": "+", "sub": "-", "mul": "*"}
_UFUNC = {
    "max": "np.maximum", "min": "np.minimum", "lt": "np.less", "le": "np.less_equal",
    "gt": "np.greater", "ge": "np.greater_equal", "eq": "np.equal", "ne": "np.not_equal",
    "and": "np.logical_and", "or": "np.logical_or", "pow": "np.power",
    "atan2": "np.arctan2", "rem": "np.fmod", "nextafter": "np.nextafter",
}  # fmt: skip
_UNARY = {
    "neg": "-({})", "exp": "np.exp({})", "log": "np.log({})", "sqrt": "np.sqrt({})",
    "abs": "np.abs({})", "sign": "np.sign({})", "floor": "np.floor({})", "ceil": "np.ceil({})",
    "sin": "np.sin({})", "cos": "np.cos({})", "tan": "np.tan({})", "tanh": "np.tanh({})",
    "log1p": "np.log1p({})", "expm1": "np.expm1({})", "logistic": "(1.0/(1.0+np.exp(-({}))))",
    "rsqrt": "(1.0/np.sqrt({}))", "asin": "np.arcsin({})", "acos": "np.arccos({})",
    "atan": "np.arctan({})", "sinh": "np.sinh({})", "cosh": "np.cosh({})",
    "round": "np.round({})", "is_finite": "np.isfinite({})", "stop_gradient": "{}",
    "copy": "{}", "copy_p": "{}", "square": "(({})*({}))",
}  # fmt: skip
_INLINE = ("jit", "pjit", "closed_call", "core_call", "remat", "checkpoint", "custom_jvp_call",
           "custom_vjp_call", "custom_vjp_call_jaxpr")  # fmt: skip


def _dtype(aval) -> str:
    k = np.dtype(aval.dtype).kind
    return "np.bool_" if k == "b" else "np.int64" if k in "iu" else "np.float64"


class _Writer:
    def __init__(self):
        self.lines: list[str] = []
        self.consts: dict[str, np.ndarray] = {}
        self.names = (f"v{i}" for i in itertools.count())
        self.cnames = (f"C{i}" for i in itertools.count())
        self.uses: dict = {}

    def const(self, value) -> str:
        a = np.asarray(value)
        if a.ndim == 0:  # a literal of the source
            if a.dtype.kind == "b":
                return "True" if bool(a) else "False"
            if a.dtype.kind in "iu":
                return f"np.int64({int(a)})"
            v = float(a)
            return repr(v) if np.isfinite(v) else ("np.inf" if v > 0 else "-np.inf" if v < 0
                                                   else "np.nan")  # fmt: skip
        name = next(self.cnames)
        self.consts[name] = np.ascontiguousarray(a)
        return name

    def emit(self, line: str) -> None:
        self.lines.append("    " + line)


def _const_value(v, env):
    if isinstance(v, jex.core.Literal):
        return np.asarray(v.val)
    got = env.get(v)
    return got[1] if got is not None and got[0] == "const" else None


def _run(jaxpr, consts, ins, w: _Writer, env_in=None):
    """Translate ``jaxpr`` with ``consts`` (values) and ``ins`` (``("code", name)`` or
    ``("const", value)``); returns the outputs in the same form."""
    env: dict = {} if env_in is None else env_in
    for v, c in zip(jaxpr.constvars, consts, strict=True):
        env[v] = ("const", np.asarray(c))
    for v, x in zip(jaxpr.invars, ins, strict=True):
        env[v] = x

    def get(v):
        if isinstance(v, jex.core.Literal):
            return ("const", np.asarray(v.val))
        return env[v]

    # how often each result is read: a result read once by an elementwise operation is
    # written into that operation's expression (numba fuses an array expression into one
    # loop, and a chain of numbers needs no array at all)
    for e in jaxpr.eqns:
        for v in e.invars:
            if not isinstance(v, jex.core.Literal):
                w.uses[v] = w.uses.get(v, 0) + 1
    for v in jaxpr.outvars:
        if not isinstance(v, jex.core.Literal):
            w.uses[v] = w.uses.get(v, 0) + 2  # returned: always written

    for e in jaxpr.eqns:
        args = [get(v) for v in e.invars]
        name = e.primitive.name
        if name in _INLINE:
            sub = e.params.get("jaxpr") or e.params.get("call_jaxpr") or e.params.get("fun_jaxpr")
            closed = sub if hasattr(sub, "consts") else None
            inner = closed.jaxpr if closed is not None else sub
            outs = _run(inner, closed.consts if closed is not None else [], args, w)
            for v, o in zip(e.outvars, outs, strict=True):
                env[v] = o
            continue
        if all(a[0] == "const" for a in args):  # fold: evaluate once with JAX
            vals = e.primitive.bind(*[jnp.asarray(a[1]) for a in args], **e.params)
            vals = vals if e.primitive.multiple_results else [vals]
            for v, val in zip(e.outvars, vals, strict=True):
                env[v] = ("const", np.asarray(val))
            continue
        outs = _emit(e, name, args, w)
        for v, o in zip(e.outvars, outs, strict=True):
            env[v] = o
    return [get(v) for v in jaxpr.outvars]


def _code(a, w: _Writer) -> str:
    if a[0] == "slice1":  # one entry of a 1-D array, as a 1-entry view
        return f"{a[1]}[{a[2]}:{a[2] + 1}]"
    if a[0] == "expr":  # an elementwise expression, written where it is read
        return a[1]
    return a[1] if a[0] == "code" else w.const(a[1])


def _num(a, w: _Writer) -> str:
    """``a`` where a number is read: one entry of a 1-D array reads as that entry."""
    if a[0] == "slice1":
        return f"{a[1]}[{a[2]}]"
    return _code(a, w)


def _array(a, aval, w: _Writer) -> str:
    """``a`` as an array expression (a rank-0 value as a 1-element array is never needed:
    callers handle rank 0 themselves)."""
    return _code(a, w)


def _elementwise(e, name, args, ins, out, w: _Writer) -> str | None:
    """The expression of an elementwise operation (operands of one shape, or numbers), or
    None for another kind of operation."""
    shape = tuple(out.shape)
    same = all(tuple(a.shape) in (shape, ()) for a in ins)
    if not same:
        return None
    c = [_num(a, w) if not av.shape else _code(a, w) for a, av in zip(args, ins, strict=True)]
    if name in _BINARY:
        return f"({c[0]} {_BINARY[name]} {c[1]})"
    if name == "div":
        op = "//" if np.dtype(out.dtype).kind in "iu" else "/"
        return f"({c[0]} {op} {c[1]})"
    if name in _UFUNC:
        return f"{_UFUNC[name]}({c[0]}, {c[1]})"
    if name in _UNARY:
        return "(" + _UNARY[name].format(c[0], c[0]) + ")"
    if name == "integer_pow":
        return f"(({c[0]}) ** {int(e.params['y'])})"
    if name == "select_n" and len(c) == 3:
        if not shape and not ins[0].shape:
            return f"({c[2]} if {c[0]} else {c[1]})"
        return f"np.where({c[0]}, {c[2]}, {c[1]})"
    if name == "convert_element_type" and not shape and _dtype(out) == "np.float64":
        return f"float({c[0]})"
    return None


def _emit(e, name, args, w: _Writer):
    out = e.outvars[0].aval if e.outvars else None
    ins = [v.aval for v in e.invars]
    if out is not None and len(e.outvars) == 1:
        ex = _elementwise(e, name, args, ins, out, w)
        if ex is not None:
            if w.uses.get(e.outvars[0], 0) == 1:
                return [("expr", ex)]
            r = next(w.names)
            w.emit(f"{r} = {ex}")
            return [("code", r)]
    c = [_code(a, w) for a in args]
    r = next(w.names)
    shape = tuple(out.shape) if out is not None else ()
    if name in _BINARY:
        w.emit(f"{r} = {c[0]} {_BINARY[name]} {c[1]}")
    elif name == "div":
        if np.dtype(out.dtype).kind in "iu":
            w.emit(f"{r} = {c[0]} // {c[1]}")
        else:
            w.emit(f"{r} = {c[0]} / {c[1]}")
    elif name in _UFUNC:
        w.emit(f"{r} = {_UFUNC[name]}({c[0]}, {c[1]})")
    elif name == "not":
        w.emit(f"{r} = np.logical_not({c[0]})" if np.dtype(out.dtype).kind == "b"
               else f"{r} = ~{c[0]}")  # fmt: skip
    elif name in _UNARY:
        w.emit(f"{r} = " + _UNARY[name].format(c[0], c[0]))
    elif name == "integer_pow":
        w.emit(f"{r} = ({c[0]}) ** {int(e.params['y'])}")
    elif name == "convert_element_type":
        dt = _dtype(out)
        if not shape and dt == "np.float64":
            w.emit(f"{r} = float({c[0]})")
        else:
            w.emit(f"{r} = {dt}({c[0]})" if not shape else f"{r} = ({c[0]}).astype({dt})")
    elif name == "select_n":
        pred, cases = c[0], c[1:]
        if len(cases) != 2:
            raise NotTranslatable(f"select_n with {len(cases)} cases")
        if not shape and not ins[0].shape:
            w.emit(f"{r} = {cases[1]} if {pred} else {cases[0]}")
        else:
            w.emit(f"{r} = np.where({pred}, {cases[1]}, {cases[0]})")
    elif name == "broadcast_in_dim":
        dims = e.params["broadcast_dimensions"]
        src = tuple(ins[0].shape)
        if not src:
            w.emit(f"{r} = np.full({shape}, {c[0]}, dtype={_dtype(out)})")
        else:
            keep = [1] * len(shape)
            for k, d in enumerate(dims):
                keep[d] = src[k]
            w.emit(f"{r} = np.ascontiguousarray(np.broadcast_to("
                   f"np.reshape(np.ascontiguousarray({c[0]}), {tuple(keep)}), "
                   f"{shape}))")  # fmt: skip
    elif name in ("reshape", "squeeze", "expand_dims"):
        if not shape and not ins[0].shape:  # a number stays a number
            return [args[0]]
        if not shape and args[0][0] == "slice1":
            w.emit(f"{r} = {args[0][1]}[{args[0][2]}]")
        elif not shape:
            w.emit(f"{r} = np.ascontiguousarray({c[0]}).ravel()[0]")
        elif not ins[0].shape:
            w.emit(f"{r} = np.full({shape}, {c[0]}, dtype={_dtype(out)})")
        else:
            w.emit(f"{r} = np.reshape(np.ascontiguousarray({c[0]}), {shape})")
    elif name == "slice":
        st, lim = e.params["start_indices"], e.params["limit_indices"]
        strides = e.params.get("strides") or (1,) * len(st)
        if len(st) == 1 and lim[0] - st[0] == 1 and args[0][0] == "code":
            return [("slice1", args[0][1], int(st[0]))]  # read as a number where it is one
        idx = ", ".join(f"{a}:{b}:{s}" for a, b, s in zip(st, lim, strides, strict=True))
        w.emit(f"{r} = {c[0]}[{idx}]")
    elif name == "concatenate":
        ax = e.params["dimension"]
        if len(shape) == 1:  # one array, filled part by part
            w.emit(f"{r} = np.empty({shape[0]}, dtype={_dtype(out)})")
            at = 0
            for a, av, raw in zip(c, ins, args, strict=True):
                k = int(av.shape[0]) if av.shape else 1
                if raw[0] == "slice1":
                    w.emit(f"{r}[{at}] = {raw[1]}[{raw[2]}]")
                elif not av.shape:
                    w.emit(f"{r}[{at}] = {a}")
                elif k == 1:
                    w.emit(f"{r}[{at}] = {a}[0]")
                else:
                    w.emit(f"{r}[{at}:{at + k}] = {a}")
                at += k
        else:
            parts = []
            for a, av in zip(c, ins, strict=True):
                parts.append(a if av.shape else f"np.full((1,), {a})")
            w.emit(f"{r} = np.concatenate(({', '.join(parts)},), {ax})")
    elif name == "transpose":
        w.emit(
            f"{r} = np.ascontiguousarray(np.transpose({c[0]}, {tuple(e.params['permutation'])}))"
        )
    elif name in ("reduce_sum", "reduce_max", "reduce_min", "reduce_prod", "argmax"):
        fn = {"reduce_sum": "np.sum", "reduce_max": "np.max", "reduce_min": "np.min",
              "reduce_prod": "np.prod"}.get(name)  # fmt: skip
        if fn is None:
            raise NotTranslatable(name)
        axes = sorted(e.params["axes"], reverse=True)
        if len(axes) == len(ins[0].shape):
            w.emit(f"{r} = {fn}({c[0]})")
        else:
            cur = c[0]
            for ax in axes:
                nxt = next(w.names)
                w.emit(f"{nxt} = {fn}({cur}, axis={ax})")
                cur = nxt
            w.emit(f"{r} = {cur}")
    elif name == "dot_general":
        _dot(e, c, ins, out, r, w)
    elif name == "gather":
        _gather(e, args, ins, out, r, w)
    elif name in ("scatter-add", "scatter_add"):
        _scatter_add(e, args, ins, out, r, w)
    elif name == "split":
        ax, sizes = e.params["axis"], e.params["sizes"]
        outs, start = [], 0
        for k in sizes:
            o = next(w.names)
            idx = ", ".join(f"{start}:{start + k}" if d == ax else ":"
                            for d in range(len(ins[0].shape)))  # fmt: skip
            w.emit(f"{o} = np.ascontiguousarray({c[0]}[{idx}])")
            outs.append(("code", o))
            start += k
        return outs
    elif name == "clamp":
        w.emit(f"{r} = np.minimum(np.maximum({c[1]}, {c[0]}), {c[2]})")
    elif name == "cumsum":
        if len(ins[0].shape) != 1 or e.params.get("reverse"):
            raise NotTranslatable("cumsum beyond one axis")
        w.emit(f"{r} = np.cumsum({c[0]})")
    else:
        raise NotTranslatable(f"the JAX primitive '{name}'")
    return [("code", r)]


def _dot(e, c, ins, out, r, w: _Writer):
    (lc, rc), (lb, rb) = e.params["dimension_numbers"]
    ls, rs = tuple(ins[0].shape), tuple(ins[1].shape)
    lf = [d for d in range(len(ls)) if d not in lc and d not in lb]
    rf = [d for d in range(len(rs)) if d not in rc and d not in rb]
    B = int(np.prod([ls[d] for d in lb])) if lb else 1
    K = int(np.prod([ls[d] for d in lc])) if lc else 1
    M = int(np.prod([ls[d] for d in lf])) if lf else 1
    N = int(np.prod([rs[d] for d in rf])) if rf else 1
    a, b = next(w.names), next(w.names)
    w.emit(f"{a} = np.reshape(np.ascontiguousarray(np.transpose({c[0]}, "
           f"{tuple(lb) + tuple(lf) + tuple(lc)})), ({B}, {M}, {K}))")  # fmt: skip
    w.emit(f"{b} = np.reshape(np.ascontiguousarray(np.transpose({c[1]}, "
           f"{tuple(rb) + tuple(rc) + tuple(rf)})), ({B}, {K}, {N}))")  # fmt: skip
    o = next(w.names)
    w.emit(f"{o} = np.empty(({B}, {M}, {N}))")
    w.emit(f"for _b in range({B}):")
    w.emit(f"    {o}[_b] = np.dot(np.ascontiguousarray({a}[_b]), np.ascontiguousarray({b}[_b]))")
    shape = tuple(out.shape)
    if shape:
        w.emit(f"{r} = np.reshape({o}, {shape})")
    else:
        w.emit(f"{r} = {o}[0, 0, 0]")


def _gather(e, args, ins, out, r, w: _Writer):
    if args[1][0] != "const":
        raise NotTranslatable("a gather at indices that depend on the inputs")
    op_shape = tuple(ins[0].shape)
    size = int(np.prod(op_shape)) if op_shape else 1
    iota = jnp.arange(size, dtype=jnp.float64).reshape(op_shape)
    params = dict(e.params)
    params["fill_value"] = -1.0
    src = np.asarray(e.primitive.bind(iota, jnp.asarray(args[1][1]), **params))
    src = np.rint(src).astype(np.int64).ravel()
    mode = str(e.params.get("mode"))
    fill = e.params.get("fill_value")
    fill = np.nan if fill is None else float(fill)
    if "CLIP" in mode or "PROMISE" in mode:
        src = np.clip(src, 0, size - 1)
    m = w.const(src)
    x = _code(args[0], w)
    shape = tuple(out.shape)
    flat = next(w.names)
    w.emit(
        f"{flat} = np.ascontiguousarray({x}).ravel()"
        if op_shape
        else f"{flat} = np.full((1,), {x})"
    )
    g = next(w.names)
    if np.all(src >= 0):
        w.emit(f"{g} = {flat}[{m}]")
    else:
        w.emit(f"{g} = np.empty({m}.size)")
        w.emit(f"for _k in range({m}.size):")
        w.emit(f"    {g}[_k] = {flat}[{m}[_k]] if {m}[_k] >= 0 else {fill!r}")
    if shape:
        w.emit(f"{r} = np.reshape({g}, {shape})")
    else:
        w.emit(f"{r} = {g}[0]")


def _scatter_add(e, args, ins, out, r, w: _Writer):
    if args[1][0] != "const":
        raise NotTranslatable("a scatter-add at indices that depend on the inputs")
    op_shape, up_shape = tuple(ins[0].shape), tuple(ins[2].shape)
    n_out = int(np.prod(op_shape)) if op_shape else 1
    idx = jnp.asarray(args[1][1])

    def scatter(u):
        return e.primitive.bind(jnp.zeros(op_shape), idx, u, **e.params)

    gather = jax.linear_transpose(scatter, jax.ShapeDtypeStruct(up_shape, jnp.float64))
    tgt = np.asarray(gather(jnp.arange(1, n_out + 1, dtype=jnp.float64).reshape(op_shape))[0])
    tgt = np.rint(tgt).astype(np.int64).ravel() - 1  # -1: an update that is dropped
    m = w.const(tgt)
    x, u = _code(args[0], w), _code(args[2], w)
    flat, uf = next(w.names), next(w.names)
    w.emit(f"{flat} = np.ascontiguousarray({x}).ravel().copy()" if op_shape
           else f"{flat} = np.full((1,), {x})")  # fmt: skip
    w.emit(
        f"{uf} = np.ascontiguousarray({u}).ravel()" if up_shape else f"{uf} = np.full((1,), {u})"
    )
    w.emit(f"for _k in range({m}.size):")
    w.emit(f"    if {m}[_k] >= 0:")
    w.emit(f"        {flat}[{m}[_k]] += {uf}[_k]")
    shape = tuple(out.shape)
    w.emit(f"{r} = np.reshape({flat}, {shape})" if shape else f"{r} = {flat}[0]")


def source(fn, *examples, static=(), name="f", prefix="C"):
    """``fn(*args)`` (a JAX function) as the Python source of a function of the same
    arguments for numba, rank-0 arguments as numbers and arrays as numpy arrays; ``static``:
    the positions of arguments that are constants of the translation (folded into the code:
    the model's geometry, say). Returns ``(source, constants)``: the source defines ``name``,
    which returns a tuple of the outputs (flattened pytree leaves), and uses the arrays of
    ``constants`` (names starting with ``prefix``) as globals."""
    dyn = [i for i in range(len(examples)) if i not in static]

    def flat_fn(*d):
        full = list(examples)
        for i, v in zip(dyn, d, strict=True):
            full[i] = v
        return jax.tree.leaves(fn(*full))

    closed = jax.make_jaxpr(flat_fn)(*[examples[i] for i in dyn])
    closed = _dce(closed)
    try:  # straight-line scalar code when the kernel is small enough
        src = _scalar_source(closed, [tuple(np.shape(examples[i])) for i in dyn], name)
    except NotTranslatable:
        src = None
    if src is not None:
        return src, {}
    # loops over flat buffers (array code took minutes to compile: a kernel loop mode cannot
    # write raises, and the model stays on the JAX backend)
    return _loop_source(closed, [tuple(np.shape(examples[i])) for i in dyn], name, prefix)
    w = _Writer()
    w.cnames = (f"{prefix}{i}" for i in itertools.count())
    argnames = [f"a{i}" for i in range(len(dyn))]
    ins = [("code", a) for a in argnames]
    outs = _run(closed.jaxpr, closed.consts, ins, w)
    rets = []
    for o, v in zip(outs, closed.jaxpr.outvars, strict=True):
        shape = tuple(v.aval.shape)
        if o[0] == "const":
            val = np.broadcast_to(np.asarray(o[1], dtype=np.float64), shape)
            rets.append(w.const(np.array(val)) if shape else repr(float(val)))
        else:
            rets.append(_code(o, w))
    head = f"def {name}({', '.join(argnames)}):"
    body = w.lines or ["    pass"]
    return "\n".join([head, *body, f"    return ({', '.join(rets)},)"]), w.consts


def _dce(closed):
    """Equations whose results are not used dropped (JAX's own pass, which also enters
    nested calls): ``vmap`` of ``jvp`` computes the primal too, which a sensitivity does not
    return. The constants become leading inputs for the pass, which keeps every input."""
    from jax._src.interpreters import partial_eval as pe

    k = len(closed.jaxpr.constvars)
    try:
        flat = pe.convert_constvars_jaxpr(closed.jaxpr)
        jaxpr, _ = pe.dce_jaxpr(flat, [True] * len(flat.outvars), instantiate=True)
    except Exception:  # noqa: BLE001 - an internal API: without it the code is only slower
        return closed
    if len(jaxpr.invars) != len(flat.invars):  # pragma: no cover - instantiate keeps them
        return closed
    jaxpr = jaxpr.replace(constvars=jaxpr.invars[:k], invars=jaxpr.invars[k:])
    return jex.core.ClosedJaxpr(jaxpr, closed.consts)


# ----------------------------------------------------------------------------- scalar mode
# A small kernel (every array at most SCALAR_MAX entries, at most SCALAR_OPS operations in all)
# is written as straight-line scalar code: each array is a symbolic array of its entries'
# expressions at translation, structural operations (reshape, broadcast, slice, gather,
# scatter, ...) only rearrange them, and each elementwise result is one line per entry.
# numba compiles such code in seconds and runs it without making arrays; array code with
# broadcasting took minutes to compile for an 8-state model.
SCALAR_MAX = 2048
SCALAR_OPS = 3_000  # straight-line code beyond this takes numba minutes to compile

_PY = {
    "add": lambda a, b: a + b, "add_any": lambda a, b: a + b, "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b, "max": np.maximum, "min": np.minimum, "lt": np.less,
    "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal, "eq": np.equal,
    "ne": np.not_equal, "and": np.logical_and, "or": np.logical_or, "pow": np.power,
    "neg": np.negative, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
    "sign": np.sign, "floor": np.floor, "ceil": np.ceil, "sin": np.sin, "cos": np.cos,
    "tan": np.tan, "tanh": np.tanh, "log1p": np.log1p, "expm1": np.expm1,
    "logistic": lambda a: 1.0 / (1.0 + np.exp(-a)), "rsqrt": lambda a: 1.0 / np.sqrt(a),
    "not": np.logical_not, "is_finite": np.isfinite, "stop_gradient": lambda a: a,
    "copy": lambda a: a, "copy_p": lambda a: a,
}  # fmt: skip


class _TooBig(Exception):
    pass


class _S:
    """Scalar-mode writer: lines, and fresh names."""

    def __init__(self):
        self.lines: list[str] = []
        self.k = itertools.count()
        self.ops = 0

    def new(self, expr: str) -> str:
        self.ops += 1
        if self.ops > SCALAR_OPS:
            raise _TooBig
        name = f"s{next(self.k)}"
        self.lines.append(f"    {name} = {expr}")
        return name


def _is_num(x) -> bool:
    return not isinstance(x, str)


def _lit(x) -> str:
    if isinstance(x, str):
        return x
    x = np.asarray(x)
    if x.dtype.kind == "b":
        return "True" if bool(x) else "False"
    if x.dtype.kind in "iu":
        return f"np.int64({int(x)})"
    v = float(x)
    if np.isnan(v):
        return "np.nan"
    if np.isinf(v):
        return "np.inf" if v > 0 else "-np.inf"
    return repr(v)


def _sym(value) -> np.ndarray:
    """A constant as a symbolic array (entries are numbers)."""
    a = np.asarray(value)
    out = np.empty(a.shape, dtype=object)
    for idx in np.ndindex(a.shape):
        out[idx] = a[idx].item() if hasattr(a[idx], "item") else a[idx]
    return out


def _elem(name, params, args, out_aval, w: _S):
    """One elementwise operation on entries (numbers fold)."""
    if all(_is_num(a) for a in args):
        if name == "integer_pow":
            return np.power(np.float64(args[0]) if not isinstance(args[0], (int, np.integer))
                            else args[0], int(params["y"])).item()  # fmt: skip
        if name == "div":
            a, b = args
            if np.dtype(out_aval.dtype).kind in "iu":
                return int(a) // int(b)
            return float(np.float64(a) / np.float64(b))
        if name == "select_n":
            return args[2] if bool(args[0]) else args[1]
        if name == "convert_element_type":
            k = np.dtype(out_aval.dtype).kind
            return bool(args[0]) if k == "b" else int(args[0]) if k in "iu" else float(args[0])
        f = _PY.get(name)
        if f is not None:
            v = f(*[np.float64(a) if isinstance(a, float) else a for a in args])
            return v.item() if hasattr(v, "item") else v
    c = [_lit(a) for a in args]
    if name in ("add", "add_any"):
        if _is_num(args[0]) and args[0] == 0 and not isinstance(args[0], bool):
            return args[1]
        if _is_num(args[1]) and args[1] == 0 and not isinstance(args[1], bool):
            return args[0]
        return w.new(f"{c[0]} + {c[1]}")
    if name == "sub":
        return w.new(f"{c[0]} - {c[1]}")
    if name == "mul":
        if _is_num(args[0]) and args[0] == 1:
            return args[1]
        if _is_num(args[1]) and args[1] == 1:
            return args[0]
        return w.new(f"{c[0]} * {c[1]}")
    if name == "div":
        op = "//" if np.dtype(out_aval.dtype).kind in "iu" else "/"
        return w.new(f"{c[0]} {op} {c[1]}")
    if name in _UFUNC:
        return w.new(f"{_UFUNC[name]}({c[0]}, {c[1]})")
    if name == "not":
        return w.new(f"not {c[0]}" if np.dtype(out_aval.dtype).kind == "b" else f"~{c[0]}")
    if name in _UNARY:
        return w.new(_UNARY[name].format(c[0], c[0]))
    if name == "integer_pow":
        return w.new(f"{c[0]} ** {int(params['y'])}")
    if name == "select_n":
        if len(c) != 3:
            raise NotTranslatable(f"select_n with {len(c) - 1} cases")
        return w.new(f"{c[2]} if {c[0]} else {c[1]}")
    if name == "convert_element_type":
        k = np.dtype(out_aval.dtype).kind
        fn = "bool" if k == "b" else "np.int64" if k in "iu" else "float"
        return w.new(f"{fn}({c[0]})")
    if name == "clamp":
        return w.new(f"np.minimum(np.maximum({c[1]}, {c[0]}), {c[2]})")
    raise NotTranslatable(f"the JAX primitive '{name}'")


_ELEMENTWISE = set(_PY) | set(_UFUNC) | set(_UNARY) | {"div", "integer_pow", "select_n",
                                                       "convert_element_type", "clamp"}  # fmt: skip


def _sum(entries, w: _S):
    nums = [e for e in entries if _is_num(e)]
    names = [e for e in entries if not _is_num(e)]
    total = float(sum(nums)) if nums else 0.0
    if not names:
        return total
    expr = " + ".join(names) + (f" + {_lit(total)}" if nums and total != 0 else "")
    return w.new(expr) if len(names) > 1 or (nums and total != 0) else names[0]


def _scalar_eqn(e, name, args, w: _S):
    out = e.outvars[0].aval if e.outvars else None
    shape = tuple(out.shape) if out is not None else ()
    if name in _ELEMENTWISE:
        b = [np.broadcast_to(a, shape) for a in args]
        res = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            res[idx] = _elem(name, e.params, [x[idx] for x in b], out, w)
        return [res]
    if name == "broadcast_in_dim":
        src = args[0]
        keep = [1] * len(shape)
        for k, d in enumerate(e.params["broadcast_dimensions"]):
            keep[d] = src.shape[k]
        return [np.array(np.broadcast_to(src.reshape(keep), shape))]
    if name in ("reshape", "squeeze", "expand_dims"):
        return [args[0].reshape(shape)]
    if name == "slice":
        st, lim = e.params["start_indices"], e.params["limit_indices"]
        strides = e.params.get("strides") or (1,) * len(st)
        return [args[0][tuple(slice(a, b, c) for a, b, c in zip(st, lim, strides, strict=True))]]
    if name == "concatenate":
        return [np.concatenate(args, axis=e.params["dimension"])]
    if name == "transpose":
        return [np.transpose(args[0], e.params["permutation"])]
    if name == "split":
        ax, outs, at = e.params["axis"], [], 0
        for k in e.params["sizes"]:
            idx = [slice(None)] * args[0].ndim
            idx[ax] = slice(at, at + k)
            outs.append(args[0][tuple(idx)])
            at += k
        return outs
    if name in ("reduce_sum", "reduce_max", "reduce_min"):
        axes = tuple(e.params["axes"])
        a = np.moveaxis(args[0], axes, tuple(range(-len(axes), 0)))
        flat = a.reshape(shape + (-1,))
        res = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            ent = list(flat[idx])
            if name == "reduce_sum":
                res[idx] = _sum(ent, w)
            else:
                f = "np.maximum" if name == "reduce_max" else "np.minimum"
                acc = ent[0]
                for x in ent[1:]:
                    acc = _elem("max" if name == "reduce_max" else "min", {}, [acc, x], out, w)
                res[idx] = acc
                _ = f
        return [res]
    if name == "dot_general":
        (lc, rc), (lb, rb) = e.params["dimension_numbers"]
        L, R = args
        lf = [d for d in range(L.ndim) if d not in lc and d not in lb]
        rf = [d for d in range(R.ndim) if d not in rc and d not in rb]
        Lp = np.transpose(L, tuple(lb) + tuple(lf) + tuple(lc))
        Rp = np.transpose(R, tuple(rb) + tuple(rf) + tuple(rc))
        nb = int(np.prod([L.shape[d] for d in lb])) if lb else 1
        nk = int(np.prod([L.shape[d] for d in lc])) if lc else 1
        Lp = Lp.reshape(nb, -1, nk)
        Rp = Rp.reshape(nb, -1, nk)
        res = np.empty((nb, Lp.shape[1], Rp.shape[1]), dtype=object)
        for bi in range(nb):
            for i in range(Lp.shape[1]):
                for j in range(Rp.shape[1]):
                    terms = [_elem("mul", {}, [Lp[bi, i, k], Rp[bi, j, k]], out, w)
                             for k in range(nk)]  # fmt: skip
                    res[bi, i, j] = _sum(terms, w)
        return [res.reshape(shape)]
    if name == "gather":
        if not all(_is_num(x) for x in args[1].ravel()):
            raise NotTranslatable("a gather at indices that depend on the inputs")
        op_shape = tuple(e.invars[0].aval.shape)
        size = int(np.prod(op_shape)) if op_shape else 1
        iota = jnp.arange(size, dtype=jnp.float64).reshape(op_shape)
        params = dict(e.params)
        params["fill_value"] = -1.0
        idx = jnp.asarray(np.array(args[1].tolist(), dtype=np.int64).reshape(args[1].shape))
        src = np.rint(np.asarray(e.primitive.bind(iota, idx, **params))).astype(np.int64)
        fill = e.params.get("fill_value")
        fill = np.nan if fill is None else float(fill)
        flat = args[0].ravel()
        res = np.empty(src.shape, dtype=object)
        for k in np.ndindex(src.shape):
            res[k] = flat[src[k]] if src[k] >= 0 else fill
        return [res.reshape(shape)]
    if name in ("scatter-add", "scatter_add"):
        if not all(_is_num(x) for x in args[1].ravel()):
            raise NotTranslatable("a scatter-add at indices that depend on the inputs")
        op_shape, up_shape = tuple(e.invars[0].aval.shape), tuple(e.invars[2].aval.shape)
        n_out = int(np.prod(op_shape)) if op_shape else 1
        idx = jnp.asarray(np.array(args[1].tolist(), dtype=np.int64).reshape(args[1].shape))

        def scatter(u):
            return e.primitive.bind(jnp.zeros(op_shape), idx, u, **e.params)

        tr = jax.linear_transpose(scatter, jax.ShapeDtypeStruct(up_shape, jnp.float64))
        tgt = np.asarray(tr(jnp.arange(1, n_out + 1, dtype=jnp.float64).reshape(op_shape))[0])
        tgt = np.rint(tgt).astype(np.int64).ravel() - 1
        acc = [[x] for x in args[0].ravel()]
        for k, t in enumerate(tgt):
            if t >= 0:
                acc[t].append(args[2].ravel()[k])
        res = np.empty(n_out, dtype=object)
        for i, ent in enumerate(acc):
            res[i] = _sum(ent, w) if len(ent) > 1 else ent[0]
        return [res.reshape(shape)]
    if name == "cumsum":
        if args[0].ndim != 1 or e.params.get("reverse"):
            raise NotTranslatable("cumsum beyond one axis")
        res = np.empty(shape, dtype=object)
        acc = 0.0
        for i, x in enumerate(args[0]):
            acc = _elem("add", {}, [acc, x], out, w)
            res[i] = acc
        return [res]
    raise NotTranslatable(f"the JAX primitive '{name}'")


def _scalar_run(jaxpr, consts, ins, w: _S):
    env: dict = {}
    for v, c in zip(jaxpr.constvars, consts, strict=True):
        env[v] = _sym(c)
    for v, x in zip(jaxpr.invars, ins, strict=True):
        env[v] = x

    def get(v):
        if isinstance(v, jex.core.Literal):
            return _sym(v.val)
        return env[v]

    for e in jaxpr.eqns:
        name = e.primitive.name
        args = [get(v) for v in e.invars]
        if any(a.size > SCALAR_MAX for a in args) or any(
            int(np.prod(v.aval.shape)) > SCALAR_MAX for v in e.outvars
        ):
            raise _TooBig
        if name in _INLINE:
            sub = e.params.get("jaxpr") or e.params.get("call_jaxpr") or e.params.get("fun_jaxpr")
            closed = sub if hasattr(sub, "consts") else None
            inner = closed.jaxpr if closed is not None else sub
            outs = _scalar_run(inner, closed.consts if closed is not None else [], args, w)
        elif all(_is_num(x) for a in args for x in a.ravel()):  # fold: evaluate with JAX
            vals = e.primitive.bind(
                *[jnp.asarray(np.array(a.tolist(), dtype=np.dtype(v.aval.dtype)).reshape(a.shape))
                  for a, v in zip(args, e.invars, strict=True)],
                **e.params,
            )  # fmt: skip
            vals = vals if e.primitive.multiple_results else [vals]
            outs = [_sym(np.asarray(x)) for x in vals]
        else:
            outs = _scalar_eqn(e, name, args, w)
        for v, o in zip(e.outvars, outs, strict=True):
            env[v] = o
    return [get(v) for v in jaxpr.outvars]


def _scalar_source(closed, argshapes, name):
    """The scalar-mode source of ``closed`` (inputs of ``argshapes``), or None when too big."""
    w = _S()
    ins = []
    for i, shp in enumerate(argshapes):
        a = np.empty(shp, dtype=object)
        for idx in np.ndindex(shp):
            a[idx] = f"a{i}[{', '.join(map(str, idx))}]" if shp else f"a{i}"
        ins.append(a)
    try:
        outs = _scalar_run(closed.jaxpr, closed.consts, ins, w)
    except _TooBig:
        return None
    body = list(w.lines)
    rets = []
    for k, (o, v) in enumerate(zip(outs, closed.jaxpr.outvars, strict=True)):
        shp = tuple(v.aval.shape)
        if not shp:
            rets.append(_lit(o[()] if isinstance(o, np.ndarray) else o))
            continue
        r = f"r{k}"
        body.append(f"    {r} = np.empty({shp}, dtype={_dtype(v.aval)})")
        for idx in np.ndindex(shp):
            body.append(f"    {r}[{', '.join(map(str, idx))}] = {_lit(o[idx])}")
        rets.append(r)
    head = f"def {name}({', '.join(f'a{i}' for i in range(len(argshapes)))}):"
    return "\n".join([head, *(body or ["    pass"]), f"    return ({', '.join(rets)},)"])


# ----------------------------------------------------------------------------- loop mode
# A kernel too big for scalar mode: every array is a flat buffer (C order) with its shape at
# translation; reshape is free, structural operations are integer index maps (computed once,
# by applying the operation to 0, 1, 2, ...) applied by a loop, and each chain of elementwise
# operations is one loop. numba compiles plain loops over integer maps quickly.


class _L:
    def __init__(self):
        self.lines: list[str] = []
        self.k = itertools.count()
        self.consts: dict[str, np.ndarray] = {}
        self.uses: dict = {}

    def name(self, p="b"):
        return f"{p}{next(self.k)}"

    def const(self, a, prefix) -> str:
        a = np.ascontiguousarray(a)
        nm = f"{prefix}{len(self.consts)}"
        self.consts[nm] = a
        return nm

    def emit(self, line, depth=1):
        self.lines.append("    " * depth + line)


def _shape(v):
    return tuple(v.aval.shape)


def _size(shape):
    return int(np.prod(shape)) if shape else 1


def _lval(kind, code, shape, dtype="np.float64"):
    return {"kind": kind, "code": code, "shape": tuple(shape), "dtype": dtype}


def _at(x, idx="k"):
    """``x`` read at flat index ``idx`` (a number reads as itself)."""
    if x["kind"] == "num":
        return x["code"]
    if x["kind"] == "expr":
        return x["code"].replace("[k]", f"[{idx}]") if idx != "k" else x["code"]
    return f"{x['code']}[{idx}]"


def _materialize(x, w: _L, prefix):
    """``x`` as a buffer (an expression is written out by one loop; a number given a shape,
    by ``np.full``)."""
    if x["kind"] == "buf" or (x["kind"] == "num" and not x["shape"]):
        return x
    if x["kind"] == "num":
        nm = w.name()
        w.emit(f"{nm} = np.full({_size(x['shape'])}, {x['code']})")
        return _lval("buf", nm, x["shape"], x["dtype"])
    nm = w.name()
    n = _size(x["shape"])
    w.emit(f"{nm} = np.empty({n}, dtype={x['dtype']})")
    w.emit(f"for k in range({n}):")
    w.emit(f"{nm}[k] = {x['code']}", 2)
    return _lval("buf", nm, x["shape"], x["dtype"])


def _from_const(a, w: _L, prefix):
    a = np.asarray(a)
    if a.ndim == 0:
        return _lval("num", _lit(a), (), "np.float64")
    dt = "np.bool_" if a.dtype.kind == "b" else "np.int64" if a.dtype.kind in "iu" else "np.float64"
    return _lval("buf", w.const(a.ravel(), prefix), a.shape, dt)


def _map(x, m, shape, w: _L, prefix, dtype=None):
    """Output entry k = x[m[k]] (``m``: int map; identity maps are free)."""
    m = np.asarray(m, dtype=np.int64).ravel()
    n_in = _size(x["shape"])
    if x["kind"] == "num":
        return _lval("num", x["code"], shape, x["dtype"]) if not shape else \
            _lval("expr", x["code"], shape, x["dtype"])  # fmt: skip
    if m.size == n_in and np.array_equal(m, np.arange(n_in)):
        return dict(x, shape=tuple(shape))
    xb = _materialize(x, w, prefix)
    mc = w.const(m, prefix + "M")
    return _lval("expr", f"{xb['code']}[{mc}[k]]", shape, dtype or x["dtype"])


def _buf(x, shape, w: _L, prefix):
    """``x`` as a buffer of the shape JAX gives it (a number: filled)."""
    if x["kind"] == "num":
        return _materialize(dict(x, shape=tuple(shape) or (1,)), w, prefix)
    return _materialize(x, w, prefix)


def _scalar_of(x, w: _L, prefix, dt="np.float64"):
    """A rank-0 result: a number stays one; a buffer's only entry is read."""
    if x["kind"] == "num":
        return dict(x, shape=())
    xb = _materialize(x, w, prefix)
    nm = w.name("s")
    w.emit(f"{nm} = {xb['code']}[0]")
    return _lval("num", nm, (), dt)


def _loop_eqn(e, name, args, w: _L, prefix):
    out = e.outvars[0].aval if e.outvars else None
    shape = tuple(out.shape) if out is not None else ()
    n = _size(shape)
    dt = _dtype(out) if out is not None else "np.float64"
    if name in _ELEMENTWISE:
        ops = []
        for a, v in zip(args, e.invars, strict=True):
            vs = tuple(v.aval.shape)
            if a["kind"] == "num" or vs == shape:
                ops.append(a)
            elif _size(vs) == 1:  # one entry, read as a number
                ops.append(_scalar_of(a, w, prefix))
            elif len(vs) == len(shape):  # size-1 dimensions broadcast (JAX allows it)
                m = np.broadcast_to(np.arange(_size(vs)).reshape(vs), shape)
                ops.append(_materialize(_map(dict(a, shape=vs), m, shape, w, prefix), w, prefix))
            else:
                raise NotTranslatable(f"an elementwise operation on shapes {vs} and {shape}")
        if n == 1 and not shape:  # numbers: one line
            c = [_at(a) if a["kind"] != "expr" else a["code"] for a in ops]
            c = [cc.replace("[k]", "[0]") for cc in c]
            ex = _elem_text(name, e.params, c, out)
            nm = w.name("s")
            w.emit(f"{nm} = {ex}")
            return [_lval("num", nm, (), dt)]
        c = [_at(a) for a in ops]
        ex = _elem_text(name, e.params, c, out)
        res = _lval("expr", ex, shape, dt)
        if w.uses.get(e.outvars[0], 0) != 1:
            res = _materialize(res, w, prefix)
        return [res]
    if name in ("reshape", "squeeze", "expand_dims"):
        x = args[0]
        if not shape:
            return [_scalar_of(x, w, prefix, dt)]
        return [dict(x, shape=shape)]
    in_shapes = [tuple(v.aval.shape) for v in e.invars]
    if name == "broadcast_in_dim":
        src = in_shapes[0]
        keep = [1] * len(shape)
        for k, d in enumerate(e.params["broadcast_dimensions"]):
            keep[d] = src[k]
        m = np.broadcast_to(np.arange(_size(src)).reshape(keep), shape)
        return [_map(args[0], m, shape, w, prefix)]
    if name == "transpose":
        m = np.transpose(np.arange(_size(in_shapes[0])).reshape(in_shapes[0]),
                         e.params["permutation"])  # fmt: skip
        return [_map(args[0], m, shape, w, prefix)]
    if name == "slice":
        st, lim = e.params["start_indices"], e.params["limit_indices"]
        strides = e.params.get("strides") or (1,) * len(st)
        m = np.arange(_size(in_shapes[0])).reshape(in_shapes[0])[
            tuple(slice(a, b, c) for a, b, c in zip(st, lim, strides, strict=True))
        ]
        res = _map(args[0], m, shape, w, prefix)
        if not shape:
            return [_scalar_of(res, w, prefix, dt)]
        return [res]
    if name == "split":
        ax, outs, at = e.params["axis"], [], 0
        base = np.arange(_size(in_shapes[0])).reshape(in_shapes[0])
        for k, v in zip(e.params["sizes"], e.outvars, strict=True):
            idx = [slice(None)] * len(in_shapes[0])
            idx[ax] = slice(at, at + k)
            outs.append(_map(args[0], base[tuple(idx)], _shape(v), w, prefix))
            at += k
        return outs
    if name == "concatenate":
        ax = e.params["dimension"]
        nm = w.name()
        w.emit(f"{nm} = np.empty({n}, dtype={dt})")
        pos = np.arange(n).reshape(shape)
        at = 0
        for a, shp in zip(args, in_shapes, strict=True):
            k = shp[ax] if shp else 1
            idx = [slice(None)] * len(shape)
            idx[ax] = slice(at, at + k)
            tgt = pos[tuple(idx)].ravel()
            at += k
            if a["kind"] == "num":
                tc = w.const(tgt, prefix + "T")
                w.emit(f"for k in range({tgt.size}):")
                w.emit(f"{nm}[{tc}[k]] = {a['code']}", 2)
                continue
            src = _buf(a, shp, w, prefix)
            tc = w.const(tgt, prefix + "T")
            w.emit(f"for k in range({tgt.size}):")
            w.emit(f"{nm}[{tc}[k]] = {src['code']}[k]", 2)
        return [_lval("buf", nm, shape, dt)]
    if name in ("reduce_sum", "reduce_max", "reduce_min"):
        axes = tuple(e.params["axes"])
        src = in_shapes[0]
        moved = np.moveaxis(np.arange(_size(src)).reshape(src), axes, tuple(range(-len(axes), 0)))
        groups = moved.reshape(n, -1)
        x = _buf(args[0], src, w, prefix)
        gc = w.const(groups, prefix + "G")
        nm = w.name()
        init = "0.0" if name == "reduce_sum" else ("-np.inf" if name == "reduce_max" else "np.inf")
        op = {"reduce_sum": "{a} + {b}", "reduce_max": "np.maximum({a}, {b})",
              "reduce_min": "np.minimum({a}, {b})"}[name]  # fmt: skip
        w.emit(f"{nm} = np.empty({n})")
        w.emit(f"for k in range({n}):")
        w.emit(f"acc = {init}", 2)
        w.emit(f"for j in range({groups.shape[1]}):", 2)
        w.emit("acc = " + op.format(a="acc", b=f"{x['code']}[{gc}[k, j]]"), 3)
        w.emit(f"{nm}[k] = acc", 2)
        res = _lval("buf", nm, shape, "np.float64")
        if not shape:
            sn = w.name("s")
            w.emit(f"{sn} = {nm}[0]")
            return [_lval("num", sn, (), "np.float64")]
        return [res]
    if name == "dot_general":
        (lc, rc), (lb, rb) = e.params["dimension_numbers"]
        ls, rs = in_shapes
        L = np.arange(_size(ls)).reshape(ls)
        R = np.arange(_size(rs)).reshape(rs)
        lf = [d for d in range(len(ls)) if d not in lc and d not in lb]
        rf = [d for d in range(len(rs)) if d not in rc and d not in rb]
        Lp = np.transpose(L, tuple(lb) + tuple(lf) + tuple(lc)).reshape(
            _size([ls[d] for d in lb]), _size([ls[d] for d in lf]), -1)  # fmt: skip
        Rp = np.transpose(R, tuple(rb) + tuple(rf) + tuple(rc)).reshape(
            _size([rs[d] for d in rb]), _size([rs[d] for d in rf]), -1)  # fmt: skip
        li = np.broadcast_to(Lp[:, :, None, :], (Lp.shape[0], Lp.shape[1], Rp.shape[1],
                                                 Lp.shape[2])).reshape(n, -1)  # fmt: skip
        ri = np.broadcast_to(Rp[:, None, :, :], (Lp.shape[0], Lp.shape[1], Rp.shape[1],
                                                 Lp.shape[2])).reshape(n, -1)  # fmt: skip
        a = _materialize(args[0], w, prefix) if args[0]["kind"] != "num" else args[0]
        b = _materialize(args[1], w, prefix) if args[1]["kind"] != "num" else args[1]
        lic, ric = w.const(li, prefix + "D"), w.const(ri, prefix + "D")
        nm = w.name()
        w.emit(f"{nm} = np.empty({n})")
        w.emit(f"for k in range({n}):")
        w.emit("acc = 0.0", 2)
        w.emit(f"for j in range({li.shape[1]}):", 2)
        w.emit(f"acc += {_at(a, f'{lic}[k, j]')} * {_at(b, f'{ric}[k, j]')}", 3)
        w.emit(f"{nm}[k] = acc", 2)
        if not shape:
            sn = w.name("s")
            w.emit(f"{sn} = {nm}[0]")
            return [_lval("num", sn, (), "np.float64")]
        return [_lval("buf", nm, shape, "np.float64")]
    if name == "gather":
        if args[1]["kind"] != "const":
            raise NotTranslatable("a gather at indices that depend on the inputs")
        op_shape = in_shapes[0]
        iota = jnp.arange(_size(op_shape), dtype=jnp.float64).reshape(op_shape)
        params = dict(e.params)
        params["fill_value"] = -1.0
        src = np.rint(np.asarray(e.primitive.bind(iota, jnp.asarray(args[1]["value"]),
                                                  **params))).astype(np.int64).ravel()  # fmt: skip
        if np.all(src >= 0):
            res = _map(args[0], src, shape, w, prefix)
        else:
            fill = e.params.get("fill_value")
            fill = np.nan if fill is None else float(fill)
            x = _materialize(args[0], w, prefix)
            mc = w.const(src, prefix + "M")
            res = _lval("expr", f"({x['code']}[{mc}[k]] if {mc}[k] >= 0 else {_lit(fill)})",
                        shape, dt)  # fmt: skip
        if not shape:
            return [_scalar_of(res, w, prefix, dt)]
        return [res]
    if name in ("scatter-add", "scatter_add"):
        if args[1]["kind"] != "const":
            raise NotTranslatable("a scatter-add at indices that depend on the inputs")
        op_shape, up_shape = in_shapes[0], in_shapes[2]
        n_out = _size(op_shape)
        idx = jnp.asarray(args[1]["value"])

        def scatter(u):
            return e.primitive.bind(jnp.zeros(op_shape), idx, u, **e.params)

        tr = jax.linear_transpose(scatter, jax.ShapeDtypeStruct(up_shape, jnp.float64))
        tgt = np.asarray(tr(jnp.arange(1, n_out + 1, dtype=jnp.float64).reshape(op_shape))[0])
        tgt = np.rint(tgt).astype(np.int64).ravel() - 1
        base = _materialize(args[0], w, prefix) if args[0]["kind"] == "num" and not op_shape \
            else _buf(args[0], op_shape, w, prefix)  # fmt: skip
        u = _buf(args[2], up_shape, w, prefix)
        nm = w.name()
        if base["kind"] == "num":
            w.emit(f"{nm} = np.full({n_out}, {base['code']})")
        else:
            w.emit(f"{nm} = {base['code']}.copy()")
        tc = w.const(tgt, prefix + "T")
        w.emit(f"for k in range({tgt.size}):")
        w.emit(f"if {tc}[k] >= 0:", 2)
        w.emit(f"{nm}[{tc}[k]] += {_at(u)}", 3)
        return [_lval("buf", nm, shape, "np.float64")]
    raise NotTranslatable(f"the JAX primitive '{name}'")


def _elem_text(name, params, c, out):
    if name in ("add", "add_any"):
        return f"({c[0]} + {c[1]})"
    if name == "sub":
        return f"({c[0]} - {c[1]})"
    if name == "mul":
        return f"({c[0]} * {c[1]})"
    if name == "div":
        op = "//" if np.dtype(out.dtype).kind in "iu" else "/"
        return f"({c[0]} {op} {c[1]})"
    if name in _UFUNC:
        return f"{_UFUNC[name]}({c[0]}, {c[1]})"
    if name == "not":
        return f"(not {c[0]})" if np.dtype(out.dtype).kind == "b" else f"(~{c[0]})"
    if name in _UNARY:
        return "(" + _UNARY[name].format(c[0], c[0]) + ")"
    if name == "integer_pow":
        return f"(({c[0]}) ** {int(params['y'])})"
    if name == "select_n":
        if len(c) != 3:
            raise NotTranslatable(f"select_n with {len(c) - 1} cases")
        return f"({c[2]} if {c[0]} else {c[1]})"
    if name == "convert_element_type":
        k = np.dtype(out.dtype).kind
        fn = "bool" if k == "b" else "np.int64" if k in "iu" else "float"
        return f"{fn}({c[0]})"
    if name == "clamp":
        return f"np.minimum(np.maximum({c[1]}, {c[0]}), {c[2]})"
    raise NotTranslatable(f"the JAX primitive '{name}'")


def _loop_run(jaxpr, consts, ins, w: _L, prefix):
    env: dict = {}
    for v, c in zip(jaxpr.constvars, consts, strict=True):
        env[v] = dict(_from_const(c, w, prefix), value=np.asarray(c), kind2="const")
    for v, x in zip(jaxpr.invars, ins, strict=True):
        env[v] = x
    for e in jaxpr.eqns:
        for v in e.invars:
            if not isinstance(v, jex.core.Literal):
                w.uses[v] = w.uses.get(v, 0) + 1
    for v in jaxpr.outvars:
        if not isinstance(v, jex.core.Literal):
            w.uses[v] = w.uses.get(v, 0) + 2

    def get(v):
        if isinstance(v, jex.core.Literal):
            x = _from_const(v.val, w, prefix)
            return dict(x, value=np.asarray(v.val), kind2="const")
        return env[v]

    for e in jaxpr.eqns:
        name = e.primitive.name
        args = [get(v) for v in e.invars]
        if name in _INLINE:
            sub = e.params.get("jaxpr") or e.params.get("call_jaxpr") or e.params.get("fun_jaxpr")
            closed = sub if hasattr(sub, "consts") else None
            inner = closed.jaxpr if closed is not None else sub
            outs = _loop_run(inner, closed.consts if closed is not None else [], args, w, prefix)
        elif all(a.get("kind2") == "const" for a in args):  # fold: evaluate with JAX
            vals = e.primitive.bind(*[jnp.asarray(a["value"]) for a in args], **e.params)
            vals = vals if e.primitive.multiple_results else [vals]
            outs = [dict(_from_const(x, w, prefix), value=np.asarray(x), kind2="const")
                    for x in vals]  # fmt: skip
        else:
            for a in args:  # constants read as constants (gather and scatter indices)
                if a.get("kind2") == "const":
                    a["kind"] = "const" if a["kind"] == "buf" and False else a["kind"]
            cargs = [dict(a, kind="const") if (a.get("kind2") == "const" and k == 1 and name
                                               in ("gather", "scatter-add", "scatter_add"))
                     else a for k, a in enumerate(args)]  # fmt: skip
            outs = _loop_eqn(e, name, cargs, w, prefix)
        for v, o in zip(e.outvars, outs, strict=True):
            env[v] = o
    return [get(v) for v in jaxpr.outvars]


def _loop_source(closed, argshapes, name, prefix):
    w = _L()
    ins = []
    for i, shp in enumerate(argshapes):
        if not shp:
            ins.append(_lval("num", f"a{i}", ()))
        else:
            w.emit(f"f{i} = np.ascontiguousarray(a{i}).ravel()")
            ins.append(_lval("buf", f"f{i}", shp))
    outs = _loop_run(closed.jaxpr, closed.consts, ins, w, prefix)
    rets = []
    for o, v in zip(outs, closed.jaxpr.outvars, strict=True):
        shp = tuple(v.aval.shape)
        if not shp:
            rets.append(
                o["code"] if o["kind"] == "num" else f"{_materialize(o, w, prefix)['code']}[0]"
            )
            continue
        if o["kind"] == "num":
            r = w.name("r")
            w.emit(f"{r} = np.full({shp}, {o['code']})")
            rets.append(r)
            continue
        ob = _materialize(o, w, prefix)
        r = w.name("r")
        w.emit(f"{r} = {ob['code']}.reshape({shp})")
        rets.append(r)
    head = f"def {name}({', '.join(f'a{i}' for i in range(len(argshapes)))}):"
    return "\n".join(
        [head, *(w.lines or ["    pass"]), f"    return ({', '.join(rets)},)"]
    ), w.consts


def translate(fn, *examples, static=(), name="f"):
    """``source``, compiled now (no disk cache): ``(the numba function, its source)``."""
    import numba

    src, consts = source(fn, *examples, static=static, name=name)
    glb = {"np": np, **consts}
    exec(compile(src, f"<tenqsp native {name}>", "exec"), glb)  # noqa: S102 - our own source
    return numba.njit(cache=False)(glb[name]), src
