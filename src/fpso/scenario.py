"""Scenario agent (Phase 3): fork the plan, change one thing, see what moves.

The logistics network and the parts on order are copied, one change is applied to the copy,
and the plan is packed again. The answer is the difference: parts that become late, parts
that stop being late, and the cost of delay that adds. The live plan is never touched.

Changes it can apply:

- ``cancel_sailing``: one sailing does not go (weather, a breakdown);
- ``crane_down``: no lifts for some days (a failed crane);
- ``supplier_slip``: every open line of one material is delivered N days later;
- ``storm``: no sailing or flight for some days.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date, timedelta

from fpso.logistics import Item, Network, Planner, Shortfall
from fpso.rule_cards import RuleSet


@dataclass
class Scenario:
    kind: str  # cancel_sailing, crane_down, supplier_slip, storm
    ref: str = ""  # sailing id or material
    start: date | None = None
    days: int = 0


@dataclass
class Outcome:
    scenario: Scenario
    newly_late: list[str] = field(default_factory=list)
    no_longer_late: list[str] = field(default_factory=list)
    extra_late_days: int = 0
    extra_cost_of_delay_usd: float = 0.0


def apply(net: Network, items: list[Item], sc: Scenario) -> tuple[Network, list[Item]]:
    net, items = copy.deepcopy(net), copy.deepcopy(items)
    span = {sc.start + timedelta(days=i) for i in range(sc.days)} if sc.start is not None else set()
    if sc.kind == "cancel_sailing":
        net.cancelled.add(sc.ref)
    elif sc.kind == "crane_down":
        for d in span:
            _, bunkering = net.cranes.get(d, (1, False))
            net.cranes[d] = (0, bunkering)
    elif sc.kind == "supplier_slip":
        for it in items:
            if it.material == sc.ref:
                it.ready += timedelta(days=sc.days)
    elif sc.kind == "storm":
        net.cancelled |= {s.sailing_id for s in net.sailings if s.depart in span}
        net.grounded |= {f.flight_id for f in net.flights if f.day in span}
    else:
        raise ValueError(f"unknown scenario {sc.kind!r}")
    return net, items


def what_if(
    net: Network,
    items: list[Item],
    rules: RuleSet,
    sc: Scenario,
    cost_per_day: dict[str, float] | None = None,
) -> Outcome:
    _, base = Planner(net, rules).plan(items)
    f_net, f_items = apply(net, items, sc)
    _, forked = Planner(f_net, rules).plan(f_items)
    return compare(sc, base, forked, cost_per_day or {})


def compare(sc: Scenario, base: list[Shortfall], forked: list[Shortfall], cost_per_day) -> Outcome:
    b = {s.component: s for s in base}
    f = {s.component: s for s in forked}
    out = Outcome(sc, sorted(f.keys() - b.keys()), sorted(b.keys() - f.keys()))
    for comp in f.keys() | b.keys():
        extra = (f[comp].late_days if comp in f else 0) - (b[comp].late_days if comp in b else 0)
        out.extra_late_days += extra
        out.extra_cost_of_delay_usd += extra * cost_per_day.get(comp, 0.0)
    out.extra_cost_of_delay_usd = round(out.extra_cost_of_delay_usd, 2)
    return out
