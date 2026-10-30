# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Dosing regimens given at simulation time (design decisions 2026-10-24).

A ``Regimen`` is a list of boluses and zero-order infusions, each into a compartment,
one element of a field (every axis fixed), or a field over continuous axes with a shape
saying where the amount goes (``shape=tq.point("g", at=0)``, as an initial value's shape).
It is a value, separate from the model: a regimen given to a simulation **replaces** the
doses written in the model, never adds to them. An empty ``Regimen()`` means no doses at all.

- ``tq.bolus("depot", 100, times=[0, 12])`` or ``start=0, every=12, n=7``;
- ``tq.infusion("centr", rate=50, duration=2, start=48)``: give two of ``rate``, ``amount``
  and ``duration`` (as NONMEM's RATE/AMT);
- ``a + b`` combines regimens; ``tq.regimen_from_table(df)`` reads NONMEM-style columns.

Amounts are amounts: into a place with a volume (or a weighted field) the value rises by
amount / volume, as for ``m.dose``.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

Where = tuple[tuple[str, str], ...]

_TARGET = re.compile(r"\s*([A-Za-z_]\w*)\s*(?:\[([^\]]*)\])?\s*")


@dataclasses.dataclass(frozen=True)
class Bolus:
    """``amount`` into ``target`` (sliced by ``where``) at ``time``."""

    target: str
    time: float
    amount: float
    where: Where = ()
    shape: str | None = None  # into a field over continuous axes: how the amount is spread

    def __post_init__(self):  # every way in (tq.bolus, JSON, a table) is checked here
        _check_event(self, time=self.time, amount=self.amount)


@dataclasses.dataclass(frozen=True)
class Infusion:
    """``rate`` (amount per time) into ``target`` from ``start`` for ``duration``."""

    target: str
    start: float
    duration: float
    rate: float
    where: Where = ()
    shape: str | None = None  # into a field over continuous axes: how the rate is spread

    def __post_init__(self):
        _check_event(self, start=self.start, duration=self.duration, rate=self.rate)
        if not self.duration > 0:
            raise ValueError(f"An infusion duration must be > 0, not {self.duration:g}.")

    @property
    def stop(self) -> float:
        return self.start + self.duration

    @property
    def amount(self) -> float:
        return self.rate * self.duration


class Regimen:
    """Boluses and infusions; combine with ``+``. ``Regimen()`` is no doses."""

    def __init__(self, events: Iterable[Bolus | Infusion] = ()):
        events = tuple(events)
        for e in events:
            if not isinstance(e, (Bolus, Infusion)):
                raise TypeError(f"A regimen holds boluses and infusions, not {e!r}.")
        key = lambda e: e.time if isinstance(e, Bolus) else e.start  # noqa: E731
        self.events: tuple[Bolus | Infusion, ...] = tuple(sorted(events, key=key))

    def __add__(self, other: Regimen) -> Regimen:
        if not isinstance(other, Regimen):
            return NotImplemented
        return Regimen(self.events + other.events)

    def __iter__(self):
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __eq__(self, other) -> bool:
        return isinstance(other, Regimen) and self.events == other.events

    def __hash__(self) -> int:
        return hash(self.events)

    @property
    def boluses(self) -> tuple[Bolus, ...]:
        return tuple(e for e in self.events if isinstance(e, Bolus))

    @property
    def infusions(self) -> tuple[Infusion, ...]:
        return tuple(e for e in self.events if isinstance(e, Infusion))

    def __repr__(self) -> str:
        if not self.events:
            return "Regimen() (no doses)"
        lines = []
        for e in self.events:
            tgt = _target_text(e.target, e.where) + (f" spread as {e.shape}" if e.shape else "")
            if isinstance(e, Bolus):
                lines.append(f"  bolus {e.amount:g} into {tgt} at t = {e.time:g}")
            else:
                lines.append(
                    f"  infusion {e.rate:g}/time into {tgt} from t = {e.start:g} for "
                    f"{e.duration:g} (amount {e.amount:g})"
                )
        return "Regimen:\n" + "\n".join(lines)

    # ------------------------------------------------------------------ JSON
    def to_list(self) -> list[dict]:
        out = []
        for e in self.events:
            d = dataclasses.asdict(e)
            d["kind"] = "bolus" if isinstance(e, Bolus) else "infusion"
            d["where"] = [list(p) for p in e.where]
            if d["shape"] is None:
                del d["shape"]
            out.append(d)
        return out

    @classmethod
    def from_list(cls, items: Sequence[Mapping]) -> Regimen:
        events = []
        for item in items:
            d = dict(item)
            kind = d.pop("kind", None)
            d["where"] = tuple(tuple(p) for p in d.get("where", ()))
            if kind == "bolus":
                events.append(Bolus(**d))
            elif kind == "infusion":
                events.append(Infusion(**d))
            else:
                raise ValueError(f"A regimen entry needs kind 'bolus' or 'infusion', not {kind!r}.")
        return cls(events)


def bolus(target, amount, *, times=None, start=None, every=None, n=None, shape=None) -> Regimen:
    """Bolus doses: ``bolus("depot", 100, times=[0, 12])``, or repeated,
    ``bolus("depot", 100, start=0, every=12, n=7)`` (seven doses). ``amount`` is one number
    for every time or one per time. The target is a compartment, or one element of a field:
    ``C.at(organ="gut", sub="vascular")`` or ``"C[organ=gut, sub=vascular]"``, or a field over
    continuous axes with ``shape``, which spreads each amount as an initial value's shape
    does: ``bolus("G", 100, times=[0, 12], shape=tq.point("g", at=0))``."""
    name, where = _target(target)
    shape = _shape_text(shape)
    ts = _times(times, start, every, n, "bolus")
    amounts = _per_time(amount, len(ts), "bolus amount")
    for a in amounts:
        if a < 0:
            raise ValueError(f"A bolus amount must be >= 0, not {a:g}.")
    return Regimen(Bolus(name, t, a, where, shape) for t, a in zip(ts, amounts, strict=True))


def infusion(
    target,
    *,
    rate=None,
    amount=None,
    duration=None,
    start=None,
    times=None,
    every=None,
    n=None,
    shape=None,
) -> Regimen:
    """Zero-order infusions: give two of ``rate`` (amount per time), ``amount`` and
    ``duration``, e.g. ``infusion("centr", rate=50, duration=2, start=48)``. Repeated:
    ``start=0, every=24, n=5``, or ``times=[...]`` (the start of each). Into a field over
    continuous axes, ``shape`` spreads the rate as for ``bolus``."""
    name, where = _target(target)
    shape = _shape_text(shape)
    three = (("rate", rate), ("amount", amount), ("duration", duration))
    given = [k for k, v in three if v is not None]
    if len(given) != 2:
        raise ValueError(
            "An infusion needs exactly two of rate, amount and duration; got "
            f"{sorted(given) or 'none'}."
        )
    if times is None and every is None and n is None:
        ts = [0.0 if start is None else _num(start, "infusion start")]
    else:
        ts = _times(times, start, every, n, "infusion")
    if duration is None:
        rate_v, amount_v = _num(rate, "infusion rate"), _num(amount, "infusion amount")
        if rate_v <= 0:
            raise ValueError(
                f"An infusion rate must be > 0 when the duration follows, not {rate_v:g}."
            )
        duration_v = amount_v / rate_v
    else:
        duration_v = _num(duration, "infusion duration")
        if rate is not None:
            rate_v = _num(rate, "infusion rate")
        else:
            rate_v = _num(amount, "infusion amount") / duration_v
    if not duration_v > 0:
        raise ValueError(f"An infusion duration must be > 0, not {duration_v:g}.")
    if rate_v < 0:
        raise ValueError(f"An infusion rate must be >= 0, not {rate_v:g}.")
    return Regimen(Infusion(name, t, duration_v, rate_v, where, shape) for t in ts)


def _shape_text(shape) -> str | None:
    """A shape as text: ``tq.point("g", at=0)`` (a formula) or the same written as text."""
    if shape is None:
        return None
    text = getattr(shape, "text", shape)
    if not isinstance(text, str) or not text.strip():
        raise TypeError(f"shape= takes a shape such as tq.point('g', at=0), not {shape!r}.")
    return text.strip()


def regimen_from_table(table, *, targets: Mapping | None = None) -> Regimen:
    """A regimen from NONMEM-style dose records: a pandas DataFrame or a mapping of columns.

    Columns (case-insensitive): ``TIME``, ``AMT``, ``CMT`` (a compartment name, or a number
    mapped by ``targets={1: "depot", 2: "centr"}``); optional ``RATE`` (0 or missing: a
    bolus; > 0: an infusion of AMT at that rate), ``DUR`` (an infusion of AMT over DUR),
    ``ADDL`` and ``II`` (additional doses every II), ``EVID`` (rows with EVID 1 are doses;
    0 and 2 are not) and ``MDV``. Rows with AMT missing or 0 are not doses. Other columns
    (``DV``, covariates) are not read.

    Not supported, so refused rather than read differently: resets (EVID 3 or 4), steady
    state (SS other than 0), several subjects (ID with more than one value: one regimen per
    subject), a fractional ADDL, and both RATE and DUR on one row."""
    cols = _columns(table)
    need = {"TIME", "AMT", "CMT"} - set(cols)
    if need:
        raise ValueError(f"A dose table needs the columns {sorted(need)}; it has {sorted(cols)}.")
    if "ID" in cols and len({str(v) for v in cols["ID"]}) > 1:
        raise ValueError(
            "The dose table has several subjects (ID); give one subject's rows at a time "
            "(one regimen per subject)."
        )
    nrow = len(cols["TIME"])
    out = Regimen()
    for i in range(nrow):
        row = {k: v[i] for k, v in cols.items()}
        evid = _cell(row.get("EVID"))
        if evid in (3, 4):
            raise ValueError(
                f"Row {i}: EVID {evid:g} (a reset) is not supported; start a new scenario instead."
            )
        ss = _cell(row.get("SS"))
        if ss:
            raise ValueError(
                f"Row {i}: SS {ss:g} (steady state) is not supported; give the doses that lead "
                "to it (ADDL and II), or start from a pre-run state."
            )
        amt = _cell(row.get("AMT"))
        if amt is None or amt == 0:
            continue
        if evid is not None and evid != 1:
            continue
        cmt = row["CMT"]
        if isinstance(cmt, str) and not cmt.strip().lstrip("-").isdigit():
            target = cmt.strip()
        else:
            key = int(float(cmt))
            if targets is None or key not in targets:
                raise ValueError(
                    f"Row {i}: CMT {key} is a number; say which place it is with "
                    f"targets={{{key}: 'name', ...}}."
                )
            target = targets[key]
        t = float(row["TIME"])
        addl, ii = _cell(row.get("ADDL")) or 0, _cell(row.get("II")) or 0
        if addl and not ii > 0:
            raise ValueError(f"Row {i}: ADDL {addl:g} needs II > 0.")
        if addl != int(addl) or addl < 0:
            raise ValueError(f"Row {i}: ADDL must be a whole number >= 0, not {addl:g}.")
        starts = [t + k * ii for k in range(int(addl) + 1)]
        rate, dur = _cell(row.get("RATE")), _cell(row.get("DUR"))
        if rate is not None and rate < 0:
            raise ValueError(
                f"Row {i}: RATE {rate:g} (a rate modelled in the model, as in NONMEM's -1/-2) "
                "is not supported; give RATE or DUR."
            )
        if rate and dur:
            raise ValueError(f"Row {i}: give RATE or DUR, not both (RATE {rate:g}, DUR {dur:g}).")
        if rate:
            out = out + infusion(target, amount=amt, rate=rate, times=starts)
        elif dur:
            out = out + infusion(target, amount=amt, duration=dur, times=starts)
        else:
            out = out + bolus(target, amt, times=starts)
    return out


# ----------------------------------------------------------------------------- helpers


def _target(x) -> tuple[str, Where]:
    """A target: a name, a model handle (``.name``), a slice (``C.at(...)``), or text
    ``"C[organ=gut, sub=vascular]"``."""
    if hasattr(x, "field") and hasattr(x, "where"):  # a Slice
        return x.field, tuple(x.where)
    if hasattr(x, "name") and not isinstance(x, str):
        return x.name, ()
    if isinstance(x, str):
        m = _TARGET.fullmatch(x)
        if not m:
            raise ValueError(
                f"Cannot read the dose target {x!r}; write a name or C[axis=label, ...]."
            )
        where = []
        for part in filter(None, (p.strip() for p in (m.group(2) or "").split(","))):
            if "=" not in part:
                raise ValueError(f"Dose target {x!r}: inside brackets write axis=label.")
            a, lab = (s.strip() for s in part.split("=", 1))
            where.append((a, lab))
        return m.group(1), tuple(where)
    raise TypeError(f"A dose target is a name, a model handle or a slice, not {x!r}.")


def _target_text(name: str, where: Where) -> str:
    return name + (f"[{', '.join(f'{a}={b}' for a, b in where)}]" if where else "")


def _num(x, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"The {what} must be a number, not {x!r}.") from None
    if not math.isfinite(v):
        raise ValueError(f"The {what} must be finite, not {v}.")
    return v


def _times(times, start, every, n, what: str) -> list[float]:
    if times is not None:
        if any(x is not None for x in (start, every, n)):
            raise ValueError(f"{what}: give times=, or start=, every= and n=, not both.")
        ts = [_num(t, f"{what} time") for t in np.ravel(times)]
        if not ts:
            raise ValueError(f"{what}: times= is empty.")
        return ts
    if every is None and n is None:
        if start is None:
            raise ValueError(f"{what}: give times=[...], or start=, every= and n=.")
        return [_num(start, f"{what} start")]
    if every is None or n is None:
        raise ValueError(f"{what}: repeated doses need both every= and n= (and start=, default 0).")
    every_v = _num(every, f"{what} interval")
    if not every_v > 0:
        raise ValueError(f"{what}: every= must be > 0, not {every_v:g}.")
    if int(n) != n or n < 1:
        raise ValueError(f"{what}: n= is the number of doses, a whole number >= 1, not {n!r}.")
    t0 = 0.0 if start is None else _num(start, f"{what} start")
    return [t0 + k * every_v for k in range(int(n))]


def _per_time(x, n: int, what: str) -> list[float]:
    if isinstance(x, (Sequence, np.ndarray)) and not isinstance(x, str):
        vals = [_num(v, what) for v in np.ravel(x)]
        if len(vals) != n:
            raise ValueError(f"{what}: {len(vals)} values for {n} times; give one or one per time.")
        return vals
    return [_num(x, what)] * n


def _columns(table) -> dict[str, list]:
    if hasattr(table, "columns") and hasattr(table, "to_dict"):  # a pandas DataFrame
        data = {str(c): list(table[c]) for c in table.columns}
    elif isinstance(table, Mapping):
        data = {str(k): list(v) for k, v in table.items()}
    else:
        raise TypeError("regimen_from_table takes a pandas DataFrame or a mapping of columns.")
    out = {}
    for k, v in data.items():
        if k.upper() in out:
            raise ValueError(f"Column {k!r} appears twice (columns are case-insensitive).")
        out[k.upper()] = v
    lengths = {len(v) for v in out.values()}
    if len(lengths) > 1:
        raise ValueError("The columns of a dose table must have the same length.")
    return out


def _check_event(event, **numbers) -> None:
    """A dose's fields: a target name, finite numbers, no negative amount or rate."""
    kind = type(event).__name__.lower()
    if not isinstance(event.target, str) or not event.target:
        raise TypeError(f"A {kind} target must be a name, not {event.target!r}.")
    for name, v in numbers.items():
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise TypeError(f"A {kind} {name} must be a number, not {v!r}.")
        if not math.isfinite(v):
            raise ValueError(f"A {kind} {name} must be finite, not {v!r}.")
        if name in ("amount", "rate") and v < 0:
            raise ValueError(f"A {kind} {name} must be >= 0, not {v:g}.")
        object.__setattr__(event, name, float(v))


def _cell(x):
    """A numeric cell, or None if missing (None, NaN, '.' as in NONMEM files)."""
    if x is None or (isinstance(x, str) and x.strip() in ("", ".")):
        return None
    v = float(x)
    return None if math.isnan(v) else v
