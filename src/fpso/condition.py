"""Condition agent: the historian feed (Phase 3).

Watches daily vibration on pumps, compressors and generators. A machine whose last week
averages above ``ALERT_MM_S``, or ``RISE_MM_S`` above its own usual level, and is still
climbing is flagged as heading for a corrective
work order, and the agent checks at once whether the spares for its model are in stock, so
a missing seal kit is ordered weeks before the breakdown rather than on the day.

Scored against the machines whose corrective work order the clean world holds: how many it
flags, how many it flags wrongly (rises that settle again), and how many days before the
work order starts. Today's baseline is the work order itself, raised ``BASELINE_DAYS``
before it starts.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

from fpso.connectors import Connector

ALERT_MM_S = 4.0
RISE_MM_S = 1.0  # above the machine's own level over the two months before
MIN_SLOPE = 0.05  # mm/s per day over the last two weeks
BASELINE_DAYS = 7
WATCH_DAYS = 90  # a flag counts as right when the work order starts within this


@dataclass
class Alert:
    equipment: str
    as_of: date
    level: float
    slope: float
    spares: list[str] = field(default_factory=list)  # materials for the model with no stock


def _slope(values: list[float]) -> float:
    n = len(values)
    xs = range(n)
    mx, my = (n - 1) / 2, statistics.fmean(values)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, values, strict=True)) / den if den else 0.0


class ConditionAgent:
    def __init__(self, c: dict[str, Connector]):
        self.c = c
        self.series: dict[str, dict[date, float]] = defaultdict(dict)
        for r in c["historian"].iter("vibration"):
            self.series[r.fields["EQUNR"]][date.fromisoformat(r.fields["date"])] = float(
                r.fields["vib_mm_s"]
            )
        models = {r.key: r.fields.get("HERST_TYPBZ", "") for r in c["erp-pm"].iter("equipment")}
        stock = defaultdict(int)
        for r in c["erp-mm"].iter("stock"):
            stock[r.fields["MATNR"]] += int(r.fields["LABST"])
        for r in c["warehouse"].iter("store_balance"):
            if r.fields["WERKS"] == "KST1":
                stock[r.fields["MATNR"]] += int(r.fields["LABST"]) - int(r.fields["RESERVED"] or 0)
        self.missing: dict[str, list[str]] = {}
        by_model = defaultdict(list)
        for r in c["erp-mm"].iter("material"):
            model = r.fields["MAKTX"].rsplit(";", 1)[-1].strip()
            by_model[model].append(r.key)
        for eq, model in models.items():
            self.missing[eq] = sorted(m for m in by_model.get(model, []) if stock[m] <= 0)

    def run(self, as_of: date, since: date | None = None) -> list[Alert]:
        """Alerts raised on each day after ``since`` up to ``as_of``, each day judged only on
        the readings up to it (the historian publishes daily, so the agent runs daily)."""
        out = []
        days = (as_of - since).days if since else 1
        for eq, s in self.series.items():
            for back in range(days - 1, -1, -1):
                day = as_of - timedelta(days=back)
                if alert := self._check(eq, s, day):
                    out.append(alert)
                    break
        return out

    def _check(self, eq: str, s: dict[date, float], day: date) -> Alert | None:
        last = [s[d] for d in sorted(s) if day - timedelta(days=14) < d <= day]
        before = [s[d] for d in s if day - timedelta(days=74) < d <= day - timedelta(days=14)]
        if len(last) < 10 or len(before) < 20:
            return None
        level = statistics.fmean(last[-7:])
        rise = level - statistics.median(before)
        slope = _slope(last)
        if (level >= ALERT_MM_S or rise >= RISE_MM_S) and slope >= MIN_SLOPE:
            return Alert(eq, day, round(level, 2), round(slope, 3), self.missing.get(eq, []))
        return None


def score_alerts(days: dict[date, list[Alert]], failing: dict[str, date]) -> dict:
    """First alert per machine against the corrective work orders that follow."""
    first: dict[str, date] = {}
    for d in sorted(days):
        for a in days[d]:
            first.setdefault(a.equipment, a.as_of)
    right = {
        e: d
        for e, d in first.items()
        if e in failing and d <= failing[e] <= d + timedelta(days=WATCH_DAYS)
    }
    seen_window = [
        e
        for e, start in failing.items()
        if min(days, default=start) <= start <= max(days, default=start)
    ]
    gains = sorted((failing[e] - timedelta(days=BASELINE_DAYS) - d).days for e, d in right.items())
    return {
        "flagged": len(first),
        "right": len(right),
        "precision": round(len(right) / len(first), 4) if first else None,
        "recall": round(len(right) / len(seen_window), 4) if seen_window else None,
        "median_days_before_baseline": gains[len(gains) // 2] if gains else None,
    }
