"""Weather & Disconnect agent (Phase 3).

Reads the morning's met-ocean forecast and marks the sailings and crew-change flights that
the rule cards' weather limits will stop in the next few days. The logistics plan is then
packed again without them, so a part whose boat will not sail shows up as a ``weather``
shortfall days before the boat is cancelled, while a helicopter or a later slot can still
be booked.

On the clean side the same marking uses the weather as it turned out, which is what the
agent's forecast is scored against.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from fpso.connectors import Connector
from fpso.logistics import Network
from fpso.rule_cards import Movement, RuleSet
from fpso.synthetic.operations import FORECAST_DAYS, Operations


@dataclass
class Outlook:
    """Expected wave height and wind per day, as known on one morning."""

    issued: date
    days: dict[date, tuple[float, float]]


def outlook_from_connectors(c: dict[str, Connector], as_of: date) -> Outlook:
    days = {}
    for r in c["weather"].iter("forecast"):
        if r.fields["issued"] == as_of.isoformat():
            days[date.fromisoformat(r.fields["valid"])] = (
                float(r.fields["hs_m"]),
                float(r.fields["wind_kn"]),
            )
    return Outlook(as_of, days)


def outlook_from_truth(ops: Operations, as_of: date) -> Outlook:
    days = {}
    for lead in range(1, FORECAST_DAYS + 1):
        d = as_of + timedelta(days=lead)
        if d in ops.observed:
            days[d] = (ops.observed[d].hs_m, ops.observed[d].wind_kn)
    return Outlook(as_of, days)


def stopped(net: Network, outlook: Outlook, rules: RuleSet) -> tuple[set[str], set[str]]:
    """Sailings and flights within the outlook that the weather limits stop."""
    sailings = {
        s.sailing_id
        for s in net.sailings
        if s.depart in outlook.days
        and rules.violations(Movement("vessel", hs_m=outlook.days[s.depart][0]))
    }
    flights = {
        f.flight_id
        for f in net.flights
        if f.day in outlook.days
        and rules.violations(Movement("helicopter", wind_kn=outlook.days[f.day][1]))
    }
    return sailings, flights


def with_weather(net: Network, outlook: Outlook, rules: RuleSet) -> Network:
    sailings, flights = stopped(net, outlook, rules)
    return Network(net.sailings, net.flights, net.cranes, sailings, flights)


def score_outlook(agent: tuple[set, set], truth: tuple[set, set]) -> dict:
    out = {}
    for name, a, t in (("sailings", agent[0], truth[0]), ("flights", agent[1], truth[1])):
        out[name] = {
            "predicted": len(a),
            "true": len(t),
            "precision": round(len(a & t) / len(a), 4) if a else None,
            "recall": round(len(a & t) / len(t), 4) if t else None,
        }
    return out
