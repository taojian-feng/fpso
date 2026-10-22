"""Campaign agent (Phase 3).

Outside the turnaround, small jobs on the same system are often done days apart, each with
its own isolation, permit and shipment. The agent groups the non-turnaround work orders of
the coming ``HORIZON_DAYS`` by system (the functional location) where their planned starts
fall within ``WINDOW_DAYS`` of each other, and proposes running each group as one
campaign starting on its latest-ready day. The gain it reports is the separate shipments
and isolations saved; nothing is moved without the planner's approval.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

from fpso.connectors import Connector
from fpso.resolution.normalize import parse_date

HORIZON_DAYS = 56
WINDOW_DAYS = 10


@dataclass
class Campaign:
    system: str
    work_orders: list[str]
    start: date  # proposed common start
    spread_days: int  # how far apart the planned starts were
    isolations_saved: int = 0
    work_order_starts: dict[str, date] = field(default_factory=dict)


def campaigns(c: dict[str, Connector], as_of: date) -> list[Campaign]:
    eq_system = {r.key: r.fields["TPLNR"] for r in c["erp-pm"].iter("equipment")}
    by_system = defaultdict(list)
    for r in c["erp-pm"].iter("work_order"):
        f = r.fields
        start = parse_date(f["GSTRP"])
        if (
            f["AUART"] == "PM03"
            or not start
            or not as_of < start <= as_of + timedelta(days=HORIZON_DAYS)
        ):
            continue
        by_system[eq_system.get(f["EQUNR"], "")].append((start, r.key))
    out = []
    for system, wos in sorted(by_system.items()):
        wos.sort()
        group = [wos[0]] if wos else []
        for start, wo in wos[1:] + [(None, None)]:
            if start and (start - group[0][0]).days <= WINDOW_DAYS:
                group.append((start, wo))
                continue
            if len(group) > 1:
                out.append(
                    Campaign(
                        system,
                        [w for _, w in group],
                        group[-1][0],
                        (group[-1][0] - group[0][0]).days,
                        len(group) - 1,
                        {w: s for s, w in group},
                    )
                )
            group = [(start, wo)] if start else []
    return out
