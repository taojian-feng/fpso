"""Operations ground truth for Phase 3: weather and the equipment historian.

Like the logistics data, built from its own random stream and written clean:

    extracts/
      weather/observed.csv        daily wave height (Hs) and wind at the asset
      weather/forecasts.csv       each morning's forecast for the next ``FORECAST_DAYS`` days
      historian/vibration.csv     daily vibration of rotating equipment, mm/s

Forecast error grows with lead time. Vibration on a machine heading for failure climbs for
weeks before its corrective work order starts; some machines show a rise that settles again
(a false alarm), so a trend alone is not proof.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from fpso.synthetic.truth import World

FORECAST_DAYS = 5
ROTATING = ("P", "K", "G")


@dataclass
class Weather:
    day: date
    hs_m: float
    wind_kn: float


@dataclass
class Operations:
    observed: dict[date, Weather]
    forecasts: dict[date, list[Weather]]  # issue day -> forecast for the next days
    vibration: dict[str, dict[date, float]]  # equipment -> day -> mm/s
    failing: dict[str, date]  # equipment -> planned start of its corrective work order
    false_alarms: set[str]


def _days(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def build_operations(world: World) -> Operations:
    rng = random.Random(f"operations-{world.seed}")
    start, end = world.window_start - timedelta(days=60), world.window_end + timedelta(days=30)
    observed = {}
    hs = 2.5
    for d in _days(start, end):
        winter = 0.5 + 0.5 * math.cos(2 * math.pi * (d.timetuple().tm_yday - 15) / 365)
        mean = 1.6 + 2.4 * winter
        hs = max(0.4, hs + 0.35 * (mean - hs) + rng.gauss(0, 0.8))
        if rng.random() < 0.025 * (0.3 + winter):  # a storm
            hs += rng.uniform(2.5, 5.5)
        wind = max(3.0, hs * rng.uniform(8.5, 12.0) + rng.gauss(0, 4))
        observed[d] = Weather(d, round(hs, 2), round(wind, 1))
    forecasts = {}
    for d in _days(start, end - timedelta(days=FORECAST_DAYS)):
        out = []
        for lead in range(1, FORECAST_DAYS + 1):
            o = observed[d + timedelta(days=lead)]
            err = 0.25 * lead
            out.append(
                Weather(
                    o.day,
                    round(max(0.3, o.hs_m + rng.gauss(0, err)), 2),
                    round(max(2.0, o.wind_kn + rng.gauss(0, 3 * lead)), 1),
                )
            )
        forecasts[d] = out

    failing = {}
    for w in world.work_orders:
        eq = next(e for e in world.equipment if e.equip_no == w.equip_no)
        if w.wo_type == "PM01" and eq.eq_class in ROTATING:
            failing.setdefault(w.equip_no, w.planned_start)
    rotating = [e.equip_no for e in world.equipment if e.eq_class in ROTATING]
    healthy = [e for e in rotating if e not in failing]
    false_alarms = set(rng.sample(healthy, min(len(healthy), max(1, len(failing) // 5))))
    vibration = {}
    for e in rotating:
        base = rng.uniform(1.2, 2.8)
        onset = failing[e] - timedelta(days=rng.randint(15, 45)) if e in failing else None
        bump = start + timedelta(days=rng.randint(30, 200)) if e in false_alarms else None
        series = {}
        for d in _days(start, end):
            v = base + rng.gauss(0, 0.15)
            if onset and onset <= d <= failing[e]:
                v += 4.5 * ((d - onset).days / max(1, (failing[e] - onset).days)) ** 1.5
            if bump and bump <= d <= bump + timedelta(days=24):
                v += 1.8 * math.sin(math.pi * (d - bump).days / 24)
            series[d] = round(v, 2)
        vibration[e] = series
    return Operations(observed, forecasts, vibration, failing, false_alarms)


def _write(path: Path, header: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def export_operations(world: World, extracts: str | Path) -> dict:
    ex = Path(extracts)
    ops = build_operations(world)
    _write(
        ex / "weather" / "observed.csv",
        ["date", "hs_m", "wind_kn"],
        ([w.day.isoformat(), w.hs_m, w.wind_kn] for w in ops.observed.values()),
    )
    _write(
        ex / "weather" / "forecasts.csv",
        ["issued", "valid", "hs_m", "wind_kn"],
        (
            [issued.isoformat(), w.day.isoformat(), w.hs_m, w.wind_kn]
            for issued, ws in ops.forecasts.items()
            for w in ws
        ),
    )
    _write(
        ex / "historian" / "vibration.csv",
        ["EQUNR", "date", "vib_mm_s"],
        ([e, d.isoformat(), v] for e, s in ops.vibration.items() for d, v in s.items()),
    )
    return {
        "weather_days": len(ops.observed),
        "vibration_series": len(ops.vibration),
        "failing": len(ops.failing),
    }
