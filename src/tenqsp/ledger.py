# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Conservation ledger of a solved model.

For every field and compartment, the ledger splits the change in amount into channels: each
reaction, hazard and transition term, each boundary end, each compartment rate term, routed
outflows and doses. ``initial + sum of channels`` must equal the amount at every time; what is
left over (``Ledger.unexplained``) measures how well the solver kept the books.

The amount of a field is its density times the cell widths and its weights, summed (plain
sums over categories without weights); the amount of a compartment is its value, times its
volume if it has one. Transfers (``m.transfer``) appear on both sides.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class Ledger:
    """Where the amount of each field and compartment came from and went.

    - ``t``: output times;
    - ``initial[name]``: amount at the start, before any dose at the start time;
    - ``amount[name]``: amount at each output time;
    - ``moved[name][channel]``: cumulative amount moved in (+) or out (-) since the start;
    - ``rates[name][channel]``: the rate of each channel at each output time (doses, being
      instantaneous, have no rate);
    - ``kinds[name][channel]``: reaction, hazard, transition, boundary, rate, routed, dose,
      infusion, bioavailability (the part of the doses that does not arrive, "not
      absorbed") or extinction;
    - ``t0``: the start of the solve, where ``initial`` and the cumulative amounts begin.
    """

    t: np.ndarray
    initial: dict[str, float]
    amount: dict[str, np.ndarray]
    moved: dict[str, dict[str, np.ndarray]]
    rates: dict[str, dict[str, np.ndarray]]
    kinds: dict[str, dict[str, str]]
    t0: float | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.amount)

    def unexplained(self, name: str) -> np.ndarray:
        """Amount not accounted for by the channels, at each output time (ideally ~0)."""
        explained = self.initial[name] + sum(self.moved[name].values(), np.zeros_like(self.t))
        return self.amount[name] - explained

    def scale(self, name: str) -> float:
        """Size of the amounts involved, for relative comparisons."""
        parts = [abs(self.initial[name]), float(np.max(np.abs(self.amount[name]), initial=0.0))]
        parts += [float(np.max(np.abs(v), initial=0.0)) for v in self.moved[name].values()]
        return max(parts)

    def check(self) -> dict[str, float]:
        """Largest unexplained amount relative to ``scale``, per field and compartment."""
        out = {}
        for name in self.names:
            s = self.scale(name)
            u = float(np.max(np.abs(self.unexplained(name)), initial=0.0))
            out[name] = 0.0 if s == 0 else u / s
        return out

    def table(self, name: str | None = None) -> str:
        """The books from the start to the last output time, as text."""
        names = self.names if name is None else (name,)
        lines = []
        for n in names:
            start = self.t[0] if self.t0 is None else self.t0
            lines.append(f"{n}: from t = {start:g} (start) to t = {self.t[-1]:g}")
            lines.append(f"  {'amount at the start':<46} {self.initial[n]:>14.6g}")
            for label, v in self.moved[n].items():
                lines.append(f"  {label:<46} {v[-1]:>+14.6g}")
            lines.append(f"  {'amount at the end':<46} {self.amount[n][-1]:>14.6g}")
            rel = self.check()[n]
            lines.append(
                f"  {'unexplained':<46} {self.unexplained(n)[-1]:>+14.3g}  "
                f"(largest, relative: {rel:.1e})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        worst = max(self.check().values(), default=0.0)
        return f"<Ledger: {', '.join(self.names)}; largest relative unexplained {worst:.1e}>"


def build_ledger(
    cm, T, Y, Q, th, y_initial, doses, t0: float | None = None, removed=(), infusions=()
) -> Ledger:
    """Assemble a ``Ledger`` from a solve: output times ``T``, states ``Y``, cumulative
    channel amounts ``Q`` (one column per entry of ``cm.ledger_entries``), the initial state
    before doses, the doses applied as ``(time, target, amount)`` (right-continuous: a dose
    at an output time is counted at that time), the start time ``t0``, and the amounts
    ``removed`` when extinction groups were set to zero, as ``(time, member, -amount)``, and
    the ``infusions`` as ``(start, stop, target, rate)``: exactly rate x time infused.

    A dose or infusion may carry a fifth number, the fraction of it that arrives (a
    bioavailability, ``m.bioavailability``; times are arrival times). Then the dose is
    booked as given and the rest as "not absorbed", so the books close against the
    regimen."""
    initial = {k: float(v) for k, v in cm.amounts(y_initial, th).items()}
    amount = cm.amounts(Y, th)
    moved: dict[str, dict[str, np.ndarray]] = {n: {} for n in cm.layout.names}
    rates: dict[str, dict[str, np.ndarray]] = {n: {} for n in cm.layout.names}
    kinds: dict[str, dict[str, str]] = {n: {} for n in cm.layout.names}
    if cm.ledger_entries:
        R = np.asarray(cm._over_times(cm._ledger, T, Y, th))
    for j, (target, kind, label) in enumerate(cm.ledger_entries):
        moved[target][label] = Q[:, j]
        rates[target][label] = R[:, j]
        kinds[target][label] = kind
    doses = [(*d, 1.0) if len(d) == 3 else tuple(d) for d in doses]
    infusions = [(*f, 1.0) if len(f) == 4 else tuple(f) for f in infusions]
    lost: dict[str, np.ndarray] = {}  # what a bioavailability below 1 keeps out (> 1: adds)
    lost_now: dict[str, np.ndarray] = {}
    for target in dict.fromkeys(tg for _, tg, _, _ in doses):
        given = np.zeros_like(T)
        for td, tg, a, f in doses:
            if tg == target:
                given = given + np.where(T >= td, a, 0.0)
                if f != 1.0:
                    lost[tg] = lost.get(tg, 0.0) + np.where(T >= td, (f - 1.0) * a, 0.0)
        moved[target]["doses"] = given
        kinds[target]["doses"] = "dose"
    for target in dict.fromkeys(tg for _, _, tg, _, _ in infusions):
        given, now = np.zeros_like(T), np.zeros_like(T)
        for ts, te, tg, r, f in infusions:
            if tg == target:
                given = given + r * np.clip(T - ts, 0.0, te - ts)
                on = np.where((T >= ts) & (T < te), r, 0.0)
                now = now + on
                if f != 1.0:
                    lost[tg] = lost.get(tg, 0.0) + (f - 1.0) * r * np.clip(T - ts, 0.0, te - ts)
                    lost_now[tg] = lost_now.get(tg, 0.0) + (f - 1.0) * on
        moved[target]["infusions"] = given
        rates[target]["infusions"] = now
        kinds[target]["infusions"] = "infusion"
    for target, v in lost.items():
        moved[target]["not absorbed"] = v + np.zeros_like(T)
        if target in lost_now:
            rates[target]["not absorbed"] = lost_now[target] + np.zeros_like(T)
        kinds[target]["not absorbed"] = "bioavailability"
    for target in dict.fromkeys(tg for _, tg, _ in removed):
        gone = np.zeros_like(T)
        for te, tg, a in removed:
            if tg == target:
                gone = gone + np.where(T >= te, a, 0.0)
        moved[target]["extinction"] = gone
        kinds[target]["extinction"] = "extinction"
    return Ledger(t=T, initial=initial, amount=amount, moved=moved, rates=rates, kinds=kinds, t0=t0)
