# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Derivatives of a simulation (step 4.4d; design note ``docs/design/step4_4d_api.md``).

- ``Function`` (``sim.function(scenario, wrt=[...])``): the simulation as a JAX
  function of the numbers named in ``wrt``, ``f(x) -> {output: array}``. At each ``x`` it runs
  the JAX solver (no derivative through its choice of steps), and the derivatives come in
  one of two ways (``derivatives=``; "auto" picks forward up to ``FORWARD_MAX`` numbers in
  ``wrt``, reverse above, design decision 2026-10-31):

  - ``"forward"``: the run carries the derivative of the solution along each ``wrt`` number
    (forward sensitivities, ``jax_backend._program(forward=p)``), and ``jax.jvp``,
    ``jax.jacfwd`` and ``jax.grad`` read them. Cost grows with the number of names; no
    record. Every derivative's error steers the steps, as the solution's does.
  - ``"reverse"``: the run is recorded and replayed (``jax_replay``), and the replay's
    arithmetic carries the derivatives; ``jax.grad`` costs the same whatever the number of
    names. The steps are chosen for the derivatives too (design note §6.3, design decision
    2026-10-25): the run carries the derivative along one direction mixing every ``wrt``
    number (each moved in proportion to its value, with fixed random signs and weights, so
    none cancels another), whose error steers the steps as the solution's does.

  Either way ``f`` is pure JAX: ``jax.grad``, ``jax.jacfwd``, ``jax.jit`` and ``jax.vmap``
  all work on it. A run that fails gives NaN outputs (and NaN derivatives) rather than an
  error, so an optimizer can step back; ``f.status(x)`` says why.
- ``Sensitivity`` (``sim.sensitivity(scenario, wrt=[...])``): d output / d each name at
  every output time, once, at the scenario's values (forward mode); a failing run raises.

``wrt`` names (design note §4): a parameter (``"CL"``; a per-category parameter ``"k"``
expands to every entry, ``"k[R]"`` is one), or a number of the scenario's regimen, counted
from 1 in time order: ``"dose 1 amount"``, ``"dose 1 time"``, ``"infusion 1 rate"``,
``"infusion 1 amount"``, ``"infusion 1 duration"``, ``"infusion 1 start"``. Changing an
infusion's duration keeps its rate (the stop moves) unless its amount is in ``wrt`` too
(then the amount is kept); its rate or amount keep the duration; its start keeps the
duration (both ends move).

Everything structural is fixed by the scenario (outputs, output times, the number of doses
and where they go, the names in ``wrt``); a new scenario of the same shape shares the build.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tenqsp.backends import jax_backend as jb
from tenqsp.backends import jax_replay
from tenqsp.backends.jax_bdf import OK, STATUS_TEXT

DERIVATIVES = ("auto", "forward", "reverse")
FORWARD_MAX = 40  # "auto": forward sensitivities up to this many wrt numbers, reverse above
# (design decision 2026-10-31: forward is cheaper below it for gradients, and for every
# full Jacobian measured; docs/benchmarks.md, "Forward or reverse")
RECORD_ROOM = 4  # the record holds 4x the steps of the run at the scenario's values,
RECORD_ROOM_LARGE = 2  # 2x on models above LARGE_STATES (design decision 2026-10-25, design
LARGE_STATES = 2_000  # note step4_5_sparse_tier.md §5.2: memory grows with states x room)


def record_room(n_states: int) -> int:
    """How many times the steps of the run at x0 the record holds."""
    return RECORD_ROOM_LARGE if n_states > LARGE_STATES else RECORD_ROOM


RECORD_FULL = 100  # status: a run needing more steps than the record holds
_STATUS = {
    **STATUS_TEXT,
    jb.TOO_MANY_EVENTS: f"more than {jb.MAX_EVENTS} switch events (a condition flips back and "
    "forth)",
    jb.TOO_MANY_EXTINCTIONS: f"more than {jb.EXTINCTION_RECORDS} extinctions",
    jb.BAD_ARRIVAL: "a lag or bioavailability below 0 or not finite",
    jb.BAD_REGIMEN: "a dose or infusion moved before the start, or a negative amount, rate or "
    "duration",
    jb.SWITCH_CAPACITY: "more switch times than at the scenario's values (make a new "
    "sim.function(...) at this x)",
    RECORD_FULL: "too many steps for the record (make a new sim.function(...) at this x)",
}
_DOSE = re.compile(r"dose\s+(\d+)\s+(amount|time)")
_INFUSION = re.compile(r"infusion\s+(\d+)\s+(rate|amount|duration|start)")


class Function:
    """The simulation as a JAX function of the numbers named in ``wrt`` (see the module
    docstring). ``f.names``: what each entry of ``x`` is; ``f.x0``: their values in the
    scenario; ``f(x)``: ``{output: array (n_t, ...)}``; ``f.unpack(x)``: ``x`` by name;
    ``f.status(x)``: ``"ok"`` or why the run at ``x`` failed (outputs NaN);
    ``f.derivatives``: ``"forward"``, ``"reverse"`` or ``False``."""

    def __init__(self, sim, scenario=None, *, wrt: Sequence[str], derivatives="auto"):
        from tenqsp.simulate import Scenario, _ModelDoses

        if derivatives is True:  # the default before "auto" existed
            derivatives = "auto"
        if derivatives is not False and derivatives not in DERIVATIVES:
            raise TypeError(
                "derivatives= is 'auto' (the default), 'forward', 'reverse' or False, "
                f"not {derivatives!r}."
            )
        sc = scenario if scenario is not None else Scenario()
        if not isinstance(sc, Scenario):
            raise TypeError(f"The first argument is a tq.Scenario, not {sc!r}.")
        if sc.times is None:
            raise ValueError("Give the output times in the scenario: times=[...].")
        options = dict(sim.options)
        options.pop("backend", None)  # derivatives always come from the JAX solver
        options.pop("ledger", None)  # the ledger is not differentiated (design note §6)
        if options.pop("method", "BDF") != "BDF":
            raise ValueError("Derivatives come from the JAX BDF; the method must be 'BDF'.")
        max_steps = options.pop("max_steps", 100_000)
        self.scenario, self.outputs = sc, sim.outputs
        self.model = sim.model
        cm = self._cm = sim.model.compile()
        regimen = None if isinstance(sc.regimen, _ModelDoses) else sc.regimen
        t0 = 0.0 if sc.start is None else sc.start
        run = jb._prepare(
            cm,
            (t0, sc.times[-1]),
            np.asarray(sc.times),
            sc.params or None,
            None,
            "BDF",
            options.get("rtol", 1e-6),
            options.get("atol", 1e-6),
            options.get("max_step", np.inf),
            False,
            regimen,
            sc.initial_vector(cm),
        )
        self.names, where = _resolve(cm, wrt, regimen)
        self._run, self._regimen = run, regimen
        self.x0 = _x0(run, where, regimen)
        self._data = _data(run, where, regimen)
        self._n_t = run.t_out.size
        if derivatives == "auto":
            derivatives = "forward" if len(self.names) <= FORWARD_MAX else "reverse"
        self.derivatives = derivatives
        if derivatives == "forward":  # sensitivities carried along the run: no record
            probe = jb._program(cm, ledger=False, model_doses=run.model_doses, max_steps=max_steps)
            args, _ = _inputs(jnp.asarray(self.x0), self._data, run.model_doses)
            jb._check(cm, probe(**args), run)  # a scenario that does not run is an error here
            self.capacity = 0
            p = len(self.names)
            self._impl = _sens_program(cm, run.model_doses, max_steps, self.outputs, p)
            return
        if not derivatives:  # the solve alone: no record, no replay (step 4.6, decision 1)
            probe = jb._program(cm, ledger=False, model_doses=run.model_doses, max_steps=max_steps)
            args, _ = _inputs(jnp.asarray(self.x0), self._data, run.model_doses)
            jb._check(cm, probe(**args), run)  # a scenario that does not run is an error here
            self.capacity = 0
            self._impl = _forward_program(cm, run.model_doses, max_steps, self.outputs)
            return
        # the run at the scenario's values (steered as f's) sets the room in the record
        probe = jb._program(
            cm, ledger=False, model_doses=run.model_doses, max_steps=max_steps, steer=True
        )
        args, dot = _inputs(jnp.asarray(self.x0), self._data, run.model_doses)
        out = probe(**args, dot=dot)
        jb._check(cm, out, run)  # a scenario that does not run at all is an error here
        need = max(int(out["n_steps"]), int(out["segments"]))
        # only ever grows, so later scenarios of this shape share the build
        room = record_room(cm.n_states)
        self.capacity = max(jb.slots(room * need), cm._lazy.get("function_capacity", 0))
        cm._lazy["function_capacity"] = self.capacity
        self._impl = _function_program(cm, run.model_doses, self.capacity, max_steps, self.outputs)

    def __call__(self, x) -> dict[str, jax.Array]:
        out, _ = self._impl(jnp.asarray(x, dtype=jnp.float64), self._data)
        return {k: v[: self._n_t] for k, v in out.items()}

    def status(self, x) -> str:
        """``"ok"``, or why the run at ``x`` failed (its outputs are NaN)."""
        _, code = self._impl(jnp.asarray(x, dtype=jnp.float64), self._data)
        code = int(code)
        return "ok" if code == OK else _STATUS.get(code, f"status {code}")

    def unpack(self, x) -> dict[str, float]:
        """``x`` by name."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (len(self.names),):
            raise ValueError(f"x has {len(self.names)} entries ({list(self.names)}).")
        return dict(zip(self.names, (float(v) for v in x), strict=True))

    @property
    def t(self) -> np.ndarray:
        return self._run.t_out

    def __repr__(self) -> str:
        return f"<tq Function of {list(self.names)} -> {list(self.outputs)} at {self._n_t} times>"


class Sensitivity:
    """d output / d each name in ``wrt`` at every output time, at the scenario's values:
    ``s["conc"]["CL"]`` (shape ``(n_t, ...)``), ``s.t``, ``s.values["conc"]`` (the outputs),
    ``s.table()`` (a pandas DataFrame). With ``relative=True`` each is
    ``d y / d theta x theta / y`` ("% change per % change"), NaN where ``y`` is 0."""

    def __init__(self, f: Function, relative: bool = False):
        x0 = jnp.asarray(f.x0)
        values = {k: np.asarray(v) for k, v in f(x0).items()}
        if f.status(x0) != "ok":  # pragma: no cover - the probe in Function raised already
            raise jb.SolverError(f"The run failed: {f.status(x0)}.")
        jac = jax.jacfwd(f)(x0)
        self.t = f.t
        self.names = f.names
        self.relative = relative
        self.values = values
        self._d: dict[str, dict[str, np.ndarray]] = {}
        for out, J in jac.items():
            J = np.asarray(J)  # (n_t, ..., n_x)
            per = {}
            for i, name in enumerate(self.names):
                d = J[..., i]
                if relative:
                    y = values[out]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        d = np.where(y == 0, np.nan, d * float(f.x0[i]) / y)
                per[name] = d
            self._d[out] = per

    def __getitem__(self, output: str) -> dict[str, np.ndarray]:
        if output not in self._d:
            raise KeyError(f"'{output}' is not an output; the outputs are {list(self._d)}.")
        return self._d[output]

    def table(self):
        """One row per output time, one column per (output, name): outputs with one value
        per time (an output over axes is in ``s[output]``)."""
        import pandas as pd

        cols = {}
        for out, per in self._d.items():
            for name, d in per.items():
                if d.ndim == 1:
                    cols[(out, name)] = d
        if not cols:
            raise ValueError("No output has one value per time; read them with s[output].")
        df = pd.DataFrame(cols, index=pd.Index(self.t, name="t"))
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["output", "wrt"])
        return df

    def __repr__(self) -> str:
        kind = " (relative)" if self.relative else ""
        return (
            f"<tq Sensitivity{kind}: d {list(self._d)} / d {list(self.names)} at "
            f"{self.t.size} times>"
        )


# ----------------------------------------------------------------------------- names


def _entries(cm) -> dict[str, int]:
    """Every parameter entry by name, with its place in theta: ``"CL"``, ``"k[R]"``."""
    out = {}
    for name, sl in cm.params.index.items():
        if cm.params.over[name] is None:
            out[name] = sl.start
        else:
            for j, lab in enumerate(cm.params.labels[name]):
                lab = lab if isinstance(lab, str) else ", ".join(lab)
                out[f"{name}[{lab}]"] = sl.start + j
    return out


def _resolve(cm, wrt, regimen) -> tuple[tuple[str, ...], list[tuple]]:
    """The names in ``wrt``, expanded (``"k"`` -> ``"k[S]"``, ``"k[R]"``), and where each
    goes: ``("theta", i)``, ``("dose", j, "amount"|"time")`` or ``("infusion", j, what)``."""
    if isinstance(wrt, str):
        wrt = [wrt]
    wrt = list(wrt)
    if not wrt:
        raise ValueError("wrt= names at least one parameter or regimen number.")
    entries = _entries(cm)
    boluses = regimen.boluses if regimen is not None else ()
    infusions = regimen.infusions if regimen is not None else ()
    names, where = [], []
    for w in wrt:
        if not isinstance(w, str):
            raise TypeError(f"wrt= takes names (text), not {w!r}.")
        text = " ".join(w.split())
        if text in cm.params.index and cm.params.over[text] is not None:
            for lab in cm.params.labels[text]:
                lab = lab if isinstance(lab, str) else ", ".join(lab)
                names.append(f"{text}[{lab}]")
                where.append(("theta", entries[f"{text}[{lab}]"]))
            continue
        key = re.sub(r"\[\s*", "[", re.sub(r"\s*\]", "]", re.sub(r"\s*,\s*", ", ", text)))
        if key in entries:
            names.append(key)
            where.append(("theta", entries[key]))
            continue
        m_d, m_i = _DOSE.fullmatch(text), _INFUSION.fullmatch(text)
        if m_d or m_i:
            if regimen is None:
                raise ValueError(
                    f"wrt: '{w}' is a number of the regimen, and this scenario uses the "
                    "model's own doses. Give the doses as a regimen in the scenario (tq.bolus, "
                    "tq.infusion), or differentiate the parameters the model's doses use."
                )
            k = int((m_d or m_i).group(1))
            have, what = (boluses, "dose") if m_d else (infusions, "infusion")
            if not 1 <= k <= len(have):
                raise ValueError(
                    f"wrt: '{w}': the regimen has {len(have)} {what}(s) (counted from 1, in "
                    "time order; print(scenario.regimen) lists them)."
                )
            names.append(f"{what} {k} {(m_d or m_i).group(2)}")
            where.append((what, k - 1, (m_d or m_i).group(2)))
            continue
        raise ValueError(_unknown(w, cm, entries, boluses, infusions))
    if len(set(names)) != len(names):
        raise ValueError(f"wrt: a name is given twice: {names}.")
    for j in {x[1] for x in where if x[0] == "infusion"}:
        whats = {x[2] for x in where if x[0] == "infusion" and x[1] == j}
        if {"rate", "amount"} <= whats:
            raise ValueError(
                f"wrt: infusion {j + 1}: give its rate or its amount, not both (with the "
                "duration held, one follows from the other)."
            )
    return tuple(names), where


def _unknown(w, cm, entries, boluses, infusions) -> str:
    structural = {
        **{a.name: "an axis (its grid is structural)" for a in cm.ir.axes},
        **{f.name: "a field" for f in cm.ir.fields},
        **{c.name: "a compartment" for c in cm.ir.compartments},
        **{d.name: "a formula (differentiate the parameters it uses)" for d in cm.ir.definitions},
        **{o.name: "an observable" for o in cm.ir.observables},
    }
    valid = [*cm.params.index, *entries]
    valid += [f"dose {k + 1} {q}" for k in range(len(boluses)) for q in ("amount", "time")]
    valid += [
        f"infusion {k + 1} {q}"
        for k in range(len(infusions))
        for q in ("rate", "amount", "duration", "start")
    ]
    hint = [v for v in valid if v.lower() == w.lower()] or difflib.get_close_matches(w, valid, n=3)
    extra = f" Did you mean {', '.join(repr(h) for h in hint)}?" if hint else ""
    if w in structural:
        return f"wrt: '{w}' is {structural[w]}, not something to differentiate.{extra}"
    return (
        f"wrt: '{w}' is not a parameter or a number of the regimen.{extra} Parameters: "
        f"{list(cm.params.index)}; doses: 'dose 1 amount', 'dose 1 time', …; infusions: "
        "'infusion 1 rate' (amount, duration, start)."
    )


def _x0(run, where, regimen) -> np.ndarray:
    out = []
    for w in where:
        if w[0] == "theta":
            out.append(run.th[w[1]])
        elif w[0] == "dose":
            b = regimen.boluses[w[1]]
            out.append(b.amount if w[2] == "amount" else b.time)
        else:
            f = regimen.infusions[w[1]]
            values = {"rate": f.rate, "amount": f.amount, "duration": f.duration, "start": f.start}
            out.append(values[w[2]])
    return np.asarray(out, dtype=np.float64)


def _data(run, where, regimen) -> dict:
    """Everything the program needs that is not structure: the base inputs, and where each
    entry of x goes (index arrays, so scenarios of one shape share the build)."""
    a = run.args

    # a regimen's dose into a field fills several slots (one per cell, each with its share
    # of the amount or rate); an entry of x goes to every slot of its dose
    event = {"dose": run.b_event, "infusion": run.i_event}
    share = {"dose": run.b_share, "infusion": run.i_share}

    def pick(kind, what=None, scaled=False):
        rows = []
        for i, w in enumerate(where):
            if w[0] != kind or (what is not None and w[2] != what):
                continue
            if kind == "theta":
                rows.append((i, w[1], 1.0))
                continue
            for slot in np.flatnonzero(event[kind] == w[1]):
                rows.append((i, int(slot), float(share[kind][slot]) if scaled else 1.0))
        return (
            jnp.asarray(np.array([r[0] for r in rows], dtype=np.int64)),
            jnp.asarray(np.array([r[1] for r in rows], dtype=np.int64)),
            jnp.asarray(np.array([r[2] for r in rows], dtype=np.float64)),
        )

    M = a["i_start"].shape[0]
    n_slots = len(run.i_event) if run.i_event is not None else 0
    dur = np.zeros(M)
    by_amount = np.zeros(M, dtype=bool)
    for slot in range(n_slots):
        dur[slot] = regimen.infusions[run.i_event[slot]].duration
    for w in where:
        if w[0] == "infusion" and w[2] == "amount":
            by_amount[:n_slots][run.i_event == w[1]] = True
    rng, k = np.random.default_rng(20261025), len(where)  # the same direction every time
    return {
        "theta": a["theta"],
        "t0": jnp.asarray(a["t0"], dtype=jnp.float64),
        "t1": jnp.asarray(a["t1"], dtype=jnp.float64),
        "t_out": a["t_out"],
        "b_time": a["b_time"],
        "b_index": a["b_index"],
        "b_amount": a["b_amount"] if a["b_amount"] is not None else jnp.zeros(0),
        "i_start": a["i_start"],
        "i_dur": jnp.asarray(dur),
        "i_index": a["i_index"],
        "i_rate": a["i_rate"],
        "i_by_amount": jnp.asarray(by_amount),
        "sw_capacity": a["sw_capacity"],
        # a given starting state (a pre-run's end; NaN: the model's): fixed, not in x
        "y_start": a.get("y_start", jnp.full(run.n, jnp.nan)),
        "rtol": jnp.asarray(a["rtol"], dtype=jnp.float64),
        "atol": a["atol"],
        "max_step": jnp.asarray(a["max_step"], dtype=jnp.float64),
        "x_theta": pick("theta"),
        "x_dose_amount": pick("dose", "amount", scaled=True),
        "x_dose_time": pick("dose", "time"),
        "x_inf_rate": pick("infusion", "rate", scaled=True),
        "x_inf_amount": pick("infusion", "amount", scaled=True),
        "x_inf_duration": pick("infusion", "duration"),
        "x_inf_start": pick("infusion", "start"),
        # the direction that steers the steps (module docstring): fixed signs and weights
        "steer": jnp.asarray(rng.choice([-1.0, 1.0], k) * rng.uniform(0.5, 1.5, k)),
    }


def _inputs(x, data, model_doses):
    """The program's inputs at ``x``, and the direction (``dot=``) that steers its steps."""

    def scatter(x):
        put = lambda base, idx: (  # noqa: E731
            base.at[idx[1]].set(x[idx[0]] * idx[2]) if idx[0].shape[0] else base
        )
        theta = put(data["theta"], data["x_theta"])
        b_time = put(data["b_time"], data["x_dose_time"])
        b_amount = put(data["b_amount"], data["x_dose_amount"])
        start = put(data["i_start"], data["x_inf_start"])
        dur = put(data["i_dur"], data["x_inf_duration"])
        rate = put(data["i_rate"], data["x_inf_rate"])
        amount = put(data["i_rate"] * data["i_dur"], data["x_inf_amount"])
        by_amount = data["i_by_amount"]
        rate = jnp.where(by_amount, amount / jnp.where(by_amount, dur, 1.0), rate)
        stop = jnp.where(jnp.isfinite(data["i_start"]), start + dur, jnp.inf)
        return {
            "theta": theta,
            "b_time": b_time,
            "b_amount": b_amount,
            "i_start": start,
            "i_stop": stop,
            "i_rate": rate,
        }

    x_sg = jax.lax.stop_gradient(x)
    v = data["steer"] * jnp.where(x_sg != 0, jnp.abs(x_sg), 1.0)
    _, dot = jax.jvp(scatter, (x_sg,), (v,))
    moved = scatter(x)  # with x's derivatives, for the replay
    dot = {k: jnp.where(jnp.isfinite(d), d, 0.0) for k, d in dot.items()}
    if model_doses:
        dot["b_amount"] = None
    args = {
        "theta": moved["theta"],
        "t0": data["t0"],
        "t1": data["t1"],
        "t_out": data["t_out"],
        "b_time": moved["b_time"],
        "b_index": data["b_index"],
        "b_amount": None if model_doses else moved["b_amount"],
        "i_start": moved["i_start"],
        "i_stop": moved["i_stop"],
        "i_index": data["i_index"],
        "i_rate": moved["i_rate"],
        "sw_capacity": data["sw_capacity"],
        "y_start": data["y_start"],
        "rtol": data["rtol"],
        "atol": data["atol"],
        "max_step": data["max_step"],
    }
    return args, dot


# ----------------------------------------------------------------------------- program


def _function_program(cm, model_doses: bool, capacity: int, max_steps: int, outputs):
    """``impl(x, data) -> ({output: (T, ...)}, status)``, jitted once per model structure and
    these options (shared by every scenario of the same shape)."""
    key = ("jax_function", model_doses, capacity, max_steps, tuple(outputs), jb.tier(cm))
    if key in cm._lazy:
        return cm._lazy[key]
    run_program = jb._program(
        cm, ledger=False, model_doses=model_doses, max_steps=max_steps, record=capacity, steer=True
    )
    replay_program = jax_replay._replay_program(cm, model_doses)
    outputs_of = _outputs_of(cm, outputs)

    def impl(x, data):
        args, dot = _inputs(x, data, model_doses)
        theta, b_time = args["theta"], args["b_time"]
        start, stop, rate = args["i_start"], args["i_stop"], args["i_rate"]
        # 1. the run at x, its steps chosen by the solver for the solution and for its
        # derivative along ``dot`` (no derivative through the choice)
        out = run_program(**jax.tree.map(jax.lax.stop_gradient, args), dot=dot)
        full = (out["n_steps"] > capacity) | (out["segments"] > capacity)
        status = jnp.where(out["status"] != OK, out["status"], jnp.where(full, RECORD_FULL, OK))
        # 2. its replay, whose arithmetic carries the derivatives
        ys = replay_program(
            theta,
            out["steps"],
            out["segs"],
            out["out_step"],
            out["n_steps"],
            args["t0"],
            args["t1"],
            args["t_out"],
            b_time,
            args["b_index"],
            args["b_amount"],
            start,
            stop,
            args["i_index"],
            rate,
            args["sw_capacity"],
            args["y_start"],
        )
        # NaN values and NaN derivatives (the second term: a run that failed before its
        # first step replays nothing, so its outputs would not depend on x at all)
        bad = jnp.where(status == OK, 0.0, jnp.nan)
        ys = ys * jnp.where(status == OK, 1.0, jnp.nan) + bad * jnp.sum(x)
        return outputs_of(theta, ys, data["t_out"]), status

    jitted = jax.jit(impl)
    cm._lazy[key] = jitted
    return jitted


_REGIMEN = ("theta", "b_time", "b_amount", "i_start", "i_stop", "i_rate")


def _sens_program(cm, model_doses: bool, max_steps: int, outputs, p: int):
    """``impl(x, data) -> ({output: (T, ...)}, status)`` with forward sensitivities: one run
    carries the derivative of the solution along each entry of ``x`` (``jax_backend._program``,
    ``forward``), and ``jax.jvp``/``jax.grad`` read them (a linear map of the tangent, so
    reverse mode transposes it): no record and no replay. Jitted once per model structure,
    these options and the number of entries of ``x``."""
    key = ("jax_sens_function", model_doses, max_steps, tuple(outputs), p, jb.tier(cm))
    if key in cm._lazy:
        return cm._lazy[key]
    run_program = jb._program(
        cm, ledger=False, model_doses=model_doses, max_steps=max_steps, forward=p
    )
    outputs_of = _outputs_of(cm, outputs)
    n = cm.n_states

    def solve(x, data):
        def moved(x):
            args, _ = _inputs(x, data, model_doses)
            return {k: args[k] for k in _REGIMEN if args[k] is not None}

        args, _ = _inputs(x, data, model_doses)
        # each direction is one entry of x moved in proportion to its value (so the
        # derivatives are on one scale), divided back out below
        size = jnp.where(x != 0, jnp.abs(x), 1.0)
        dot = jax.vmap(lambda e: jax.jvp(moved, (x,), (e,))[1])(jnp.diag(size))
        dot = {k: jnp.where(jnp.isfinite(d), d, 0.0) for k, d in dot.items()}  # padding: inf
        out = run_program(**args, dot=dot)
        ys = out["ys"]
        S = ys[:, n:].reshape(ys.shape[0], p, n) / size[None, :, None]
        # padded output times (+inf, never written: NaN) are cut off by the caller; their
        # zero cotangents would still meet the NaNs in reverse mode (0 x NaN)
        real = jnp.isfinite(data["t_out"])
        S = jnp.where(real[:, None, None], S, 0.0)
        y = jnp.where(real[:, None], ys[:, :n], ys[0, :n])  # (and the outputs read from them)
        return y, S, out["status"]

    @jax.custom_jvp
    def states(x, data):
        ys, _, status = solve(x, data)
        return ys, status

    @states.defjvp
    def states_jvp(primals, tangents):
        x, data = primals
        ys, S, status = solve(x, data)
        dys = jnp.einsum("tpn,p->tn", S, tangents[0])
        return (ys, status), (dys, np.zeros(status.shape, dtype=jax.dtypes.float0))

    def impl(x, data):
        ys, status = states(x, data)
        args, _ = _inputs(x, data, model_doses)  # theta with x's derivatives, for outputs
        bad = jnp.where(status == OK, 0.0, jnp.nan)
        ys = ys * jnp.where(status == OK, 1.0, jnp.nan) + bad * jnp.sum(x)
        return outputs_of(args["theta"], ys, data["t_out"]), status

    jitted = jax.jit(impl)
    cm._lazy[key] = jitted
    return jitted


def _outputs_of(cm, outputs):
    """``outputs_of(theta, ys, t_out) -> {output: (T, ...)}``: the named outputs of a run's
    states at the output times."""
    geom = cm.geom
    out_slots = [i for i in (cm.plan.switch_slot,) if i is not None] + list(cm.floor_slots)
    frame = cm.plan.frame
    states = set(cm.layout.names)

    def outputs_of(theta, ys, t_out):
        th_out = theta.at[jnp.asarray(out_slots, dtype=jnp.int64)].set(jnp.nan)
        result = {}
        unpacked = jax.vmap(cm.layout.unpack)(ys)
        obs = None
        for name in outputs:
            if name in states:
                result[name] = unpacked[name]
                continue
            if obs is None:  # padded times (+inf) as the first: 0 x NaN in reverse mode
                t_safe = jnp.where(jnp.isfinite(t_out), t_out, t_out[0])
                obs = jax.vmap(lambda t, y: cm._obs_impl(t, y, th_out, geom))(t_safe, ys)
            v = obs[name]
            keep = [a for a, k in zip(frame.axes, v.shape[1:], strict=True) if k > 1]
            result[name] = v.reshape((v.shape[0], *(frame.sizes[a] for a in keep)))
        return result

    return outputs_of


@jax.custom_jvp
def _no_derivatives(x):
    """``x``; differentiating through it raises (a function made with derivatives=False)."""
    return x


@_no_derivatives.defjvp
def _no_derivatives_jvp(primals, tangents):
    raise ValueError(
        "This function was made with derivatives=False (no record of its runs, so no "
        "derivatives): make it with sim.function(scenario, wrt=[...]) to differentiate it."
    )


def _forward_program(cm, model_doses: bool, max_steps: int, outputs):
    """``impl(x, data) -> ({output: (T, ...)}, status)`` without derivatives: the solve alone
    (no steering, no record, no replay), jitted once per model structure and these options."""
    key = ("jax_forward_function", model_doses, max_steps, tuple(outputs), jb.tier(cm))
    if key in cm._lazy:
        return cm._lazy[key]
    run_program = jb._program(cm, ledger=False, model_doses=model_doses, max_steps=max_steps)
    outputs_of = _outputs_of(cm, outputs)

    def impl(x, data):
        args, _ = _inputs(_no_derivatives(x), data, model_doses)
        out = run_program(**args)
        status = out["status"]
        ys = out["ys"] * jnp.where(status == OK, 1.0, jnp.nan)
        return outputs_of(args["theta"], ys, data["t_out"]), status

    jitted = jax.jit(impl)
    cm._lazy[key] = jitted
    return jitted


def function(sim, scenario=None, *, wrt, derivatives="auto") -> Function:
    """See ``function``."""
    return Function(sim, scenario, wrt=wrt, derivatives=derivatives)


def sensitivity(sim, scenario=None, *, wrt, relative: bool = False) -> Sensitivity:
    """See ``sensitivity``."""
    if not isinstance(relative, bool):
        raise TypeError("relative= is True or False.")
    return Sensitivity(Function(sim, scenario, wrt=wrt), relative=relative)
