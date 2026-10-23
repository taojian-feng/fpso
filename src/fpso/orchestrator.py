"""Orchestrator (Phase 3, autonomy level 3: act within limits).

The Phase 2 agents each propose fixes on their own. The orchestrator puts them on one scale,
cost of delay, and settles who gets the scarce things (helicopter payload, extra deck space,
a spare part) when two parts want the same one:

1. every late part gets a cost of delay per day: the deferred production of a turnaround day
   when the Schedule agent says it moves the turnaround end, otherwise a day rate by the
   equipment's criticality;
2. parts are settled in order of the cost of delay they put at risk, largest first, and each
   takes the cheapest option still available that lands it in time, so a cheap part never
   holds the last helicopter seat a costly one needed;
3. an option whose cost is under the rule cards' approval limit, whose saving is larger than
   its cost, and whose facts are complete (every link settled, every delay notice read) runs
   at once (a spare only when the store keeps one more free and also covers every part of
   the same material needed by then whose supply the agents could not confirm, since such a
   part may turn out to need that spare first); anything else goes to the planner. So do
   every purchase requisition, every fix for a ``capacity`` shortfall (whether a boat is
   full depends on cargo the agents could not trace, packed on an estimated date) and every
   fix for a ``weather`` shortfall (a forecast days out is wrong often enough that a person should
   weigh it).

Each run is checked against every rule card. An action that ran without approval and turns
out wrong, or breaks a card, is counted as an incident; the Phase 3 gate is no incidents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from fpso.inventory import StockProposal
from fpso.logistics import Item, Network, Planner, Shortfall
from fpso.procurement import Recommendation
from fpso.rule_cards import Movement, RuleSet
from fpso.writeback import TRANSFER_USD, Action, heli_cost, slot_cost

TURNAROUND_DAY_USD = 350_000.0  # mock deferred production per turnaround day
DAY_USD = {"A": 20_000.0, "B": 5_000.0, "C": 1_000.0}
SLOT_MAX_SHARE = 0.03  # extra deck a sailing can give, as a share of its deck
AUTO_KINDS = {"book_helicopter_cargo", "request_vessel_slot", "transfer_stock"}
# What today's planner pays when the shortfall is found at the baseline notice: a charter.
EMERGENCY_HELI_USD, EMERGENCY_BOAT_USD = 22_000.0, 48_000.0


@dataclass
class Choice:
    component: str
    cost_of_delay_usd: float  # if nothing is done
    action: Action | None
    auto: bool = False
    saved_usd: float = 0.0


@dataclass
class Settlement:
    choices: list[Choice] = field(default_factory=list)

    def actions(self) -> list[Action]:
        return [c.action for c in self.choices if c.action]


def day_rate(component: str, criticality: dict[str, str], turnaround: dict[str, int]) -> float:
    if turnaround.get(component, 0) > 0:
        return TURNAROUND_DAY_USD
    return DAY_USD.get(criticality.get(component, ""), DAY_USD["C"])


def settle(
    as_of: date,
    items: list[Item],
    shortfalls: list[Shortfall],
    proposals: list[StockProposal],
    recs: list[Recommendation],
    net: Network,
    rules: RuleSet,
    rate: dict[str, float],
    qty: dict[str, int],
    prices: dict[str, float],
    doubtful: set[str] = frozenset(),
    rivals: dict[str, int] | None = None,
) -> Settlement:
    """Settle the day's late parts; ``doubtful`` parts (a link still open, a delay notice
    not read) never run without approval, whatever the cost. ``rivals`` gives, per part, the
    units of its material that unconfirmed parts need by its need date; a spare is taken
    without approval only when the store keeps that many (and at least one) free."""
    rivals = rivals or {}

    def spare_ok(p: StockProposal) -> bool:
        return p.left >= max(1, rivals.get(p.component, 0))

    planner = Planner(net, rules)
    limit = rules.param("auto_limit_usd", 0.0)
    heli_used = {f.flight_id: f.cargo_booked_kg + f.pax * planner.pax_kg for f in net.flights}
    slot_left = {s.sailing_id: s.deck_m2 * SLOT_MAX_SHARE for s in net.sailings}
    stock = {p.component: p for p in proposals}
    item = {i.component: i for i in items}
    out = Settlement()

    for s in sorted(shortfalls, key=lambda s: (-rate.get(s.component, 0) * s.late_days, s.need)):
        it = item.get(s.component)
        cod = rate.get(s.component, 0) * s.late_days
        if not s.sure:  # a full boat resting on estimated dates: a warning, no booking
            out.choices.append(Choice(s.component, cod, None))
            continue
        options = []  # (cost, kind, payload)
        if s.component in stock:
            p = stock[s.component]
            options.append(
                (
                    TRANSFER_USD[p.action],
                    "transfer_stock",
                    {"store": p.store, "material": p.material, "qty": p.qty, "move": p.action},
                )
            )
        if it:
            for f in net.flights:
                if f.flight_id in net.grounded or not it.ready <= f.day <= it.need:
                    continue
                m = Movement(
                    "helicopter",
                    it.weight_kg,
                    payload_used_kg=heli_used[f.flight_id] + it.weight_kg,
                    max_payload_kg=f.max_payload_kg,
                )
                if it.weight_kg <= planner.heli_item and not rules.violations(m):
                    options.append(
                        (
                            heli_cost(it.weight_kg),
                            "book_helicopter_cargo",
                            {"ref": f.flight_id, "on_board": f.day, "weight_kg": it.weight_kg},
                        )
                    )
                    break
            if s.cause != "supply":
                earliest = it.ready + timedelta(days=planner.shore)
                for sl in net.sailings:
                    if sl.depart < earliest or sl.sailing_id in net.cancelled:
                        continue
                    on_board = planner.lift_day(sl.depart + timedelta(days=planner.transit))
                    if on_board and on_board <= it.need and slot_left[sl.sailing_id] >= it.area_m2:
                        options.append(
                            (
                                slot_cost(it.area_m2),
                                "request_vessel_slot",
                                {"ref": sl.sailing_id, "on_board": on_board, "area_m2": it.area_m2},
                            )
                        )
                        break
        options.sort(key=lambda o: o[0])
        if not options:
            out.choices.append(Choice(s.component, cod, None))
            continue
        cost, kind, payload = options[0]
        if kind == "book_helicopter_cargo":
            heli_used[payload["ref"]] += it.weight_kg
        elif kind == "request_vessel_slot":
            slot_left[payload["ref"]] -= it.area_m2
        auto = (
            kind in AUTO_KINDS
            and cost < limit
            and cod > cost
            and s.component not in doubtful
            and s.cause not in ("capacity", "weather")
            and (kind != "transfer_stock" or spare_ok(stock[s.component]))
        )
        action = Action(
            kind,
            s.component,
            as_of,
            {**payload, "need": s.need},
            f"{s.cause} shortfall, {s.late_days} days late; cost of delay {cod:,.0f} USD",
            cost,
            "orchestrator",
        )
        out.choices.append(Choice(s.component, cod, action, auto, round(cod - cost, 2)))

    late = {s.component for s in shortfalls}
    for p in proposals:  # stock that saves a purchase even where nothing is late on board
        if p.component in late:
            continue
        a = Action(
            "transfer_stock",
            p.component,
            as_of,
            {"store": p.store, "material": p.material, "qty": p.qty, "move": p.action},
            f"spare stock covers {p.replaces}",
            TRANSFER_USD[p.action],
            "inventory",
        )
        out.choices.append(
            Choice(
                p.component,
                0.0,
                a,
                a.cost_usd < limit and spare_ok(p) and p.component not in doubtful,
                p.value_usd - a.cost_usd,
            )
        )
    for r in recs:
        if r.action == "order_now" and r.component not in stock:
            n = qty.get(r.component, 1)
            a = Action(
                "create_purchase_requisition",
                r.component,
                as_of,
                {"material": r.material, "qty": n, "need": r.need, "work_order": r.work_order},
                f"latest order date {r.order_by} has passed",
                round(prices.get(r.material, 0.0) * n, 2),
                "procurement",
            )
            out.choices.append(Choice(r.component, 0.0, a, False))
    return out


def unconfirmed_demand(facts: list, as_of: date) -> dict[str, int]:
    """Per part: units of its material that other parts need between today and its need
    date while their supply is unconfirmed (no PO line, or a link still open)."""
    open_ = [f for f in facts if f.need >= as_of and (f.po_line is None or f.unsettled)]
    out = {}
    for f in facts:
        out[f.component] = sum(
            o.qty
            for o in open_
            if o.material == f.material and o.component != f.component and o.need <= f.need
        )
    return out


def check(action: Action, rules: RuleSet, auto: bool) -> list[str]:
    """Rule-card violations of one action as it would run."""
    mode = {"book_helicopter_cargo": "helicopter", "request_vessel_slot": "vessel"}.get(
        action.kind, "transfer"
    )
    m = Movement(
        mode, float(action.payload.get("weight_kg", 0.0)), cost_usd=action.cost_usd, auto=auto
    )
    return rules.violations(m)


def emergency_cost(sf: Shortfall, rules: RuleSet) -> float:
    return (
        EMERGENCY_HELI_USD
        if sf.weight_kg <= rules.param("heli_max_item_kg", 0.0)
        else EMERGENCY_BOAT_USD
    )
