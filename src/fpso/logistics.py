"""Logistics agent (Phase 2, autonomy level 2: act with approval).

Every part already on order has to reach the asset: delivery to the shore base, a day there,
a sailing, a crane lift on arrival. Deck space on each sailing is mostly booked by other
cargo, cranes go down for maintenance, and nothing is lifted while the asset bunkers. The
agent packs the parts due in the next ``LOOKAHEAD_DAYS`` onto sailings, soonest need first
and turnaround parts first on the same day, and reports each part that will land after its
need date, with why:

- ``supply``: even the first sailing after delivery lands too late (the Procurement agent's
  expedite covers this; logistics can only help by helicopter);
- ``capacity``: a sailing that would make it is full;
- ``crane``: the boat arrives in time but no lift is possible until after the need date.

For each it proposes the cheapest fix the rule cards allow: helicopter cargo for a light
part, otherwise a request for extra deck space on a sailing that makes it. Proposals are
actions the planner approves (``fpso.writeback``); nothing is booked without that.

The packing is first-fit by need date: simple, explainable, and the same function runs on
the clean world for scoring, so differences come from what the agent could link and read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from fpso.connectors import Connector
from fpso.readiness import Facts, ReadinessAgent
from fpso.resolution.normalize import leading_int, parse_date
from fpso.rule_cards import Movement, RuleSet
from fpso.synthetic.replay import MANIFEST_LEAD_DAYS
from fpso.synthetic.truth import World
from fpso.writeback import heli_cost, slot_cost

LOOKAHEAD_DAYS = 42
UNTRACED_SLACK_DAYS = 14  # median gap between delivery and need, for lines with no work order
CAUSES = ("supply", "capacity", "crane", "weather")


def cargo_area_m2(weight_kg: float) -> float:
    """Deck area a line of parts takes: a box, a pallet, or one basket per 2 t."""
    if weight_kg <= 25:
        return 0.25
    if weight_kg <= 500:
        return 1.5
    return 6.0 * math.ceil(weight_kg / 2000)


@dataclass
class Item:
    component: str
    work_order: str
    material: str
    qty: int
    weight_kg: float
    ready: date  # at the shore base
    need: date
    priority: int = 0  # higher goes first on the same need date
    po_line: str | None = None
    report: bool = True  # False: packed for the deck space it takes, not reported on

    @property
    def area_m2(self) -> float:
        return cargo_area_m2(self.weight_kg)


@dataclass
class Sailing:
    sailing_id: str
    depart: date
    deck_m2: float
    deck_free_m2: float
    payload_free_kg: float


@dataclass
class Flight:
    flight_id: str
    day: date
    pax: int
    max_payload_kg: float
    cargo_booked_kg: float


@dataclass
class Network:
    sailings: list[Sailing]
    flights: list[Flight]
    cranes: dict[date, tuple[int, bool]]  # day -> (cranes available, bunkering)
    cancelled: set[str] = field(default_factory=set)  # sailings weather stops (Phase 3)
    grounded: set[str] = field(default_factory=set)  # flights weather stops (Phase 3)

    @classmethod
    def from_connectors(cls, c: dict[str, Connector]) -> Network:
        cap = {r.key: r.fields for r in c["vessel"].iter("sailing_capacity")}
        sailings = []
        for r in c["vessel"].iter("sailing"):
            k = cap.get(r.key)
            if not k:
                continue
            deck = float(k["deck_m2"])
            sailings.append(
                Sailing(
                    r.key,
                    parse_date(r.fields["depart_date"]),
                    deck,
                    deck - float(k["deck_booked_m2"]),
                    (float(k["payload_t"]) - float(k["payload_booked_t"])) * 1000,
                )
            )
        flights = [
            Flight(
                r.key,
                parse_date(r.fields["date"]),
                int(r.fields["pax_booked"]),
                float(r.fields["max_payload_kg"]),
                float(r.fields["cargo_booked_kg"]),
            )
            for r in c["helicopter"].iter("flight")
        ]
        cranes = {
            parse_date(r.key): (int(r.fields["cranes_available"]), r.fields["bunkering"] == "Y")
            for r in c["marine-ops"].iter("crane_day")
        }
        return cls(
            sorted(sailings, key=lambda s: s.depart), sorted(flights, key=lambda f: f.day), cranes
        )


@dataclass
class Leg:
    component: str
    mode: str  # vessel, helicopter
    ref: str  # sailing or flight
    depart: date
    on_board: date


@dataclass
class Shortfall:
    component: str
    work_order: str
    material: str
    need: date
    on_board: date | None  # planned, None when no sailing in the plan reaches it
    cause: str  # supply, capacity, crane, weather
    option: str | None = None  # helicopter, vessel_slot
    option_ref: str | None = None
    option_on_board: date | None = None
    weight_kg: float = 0.0
    area_m2: float = 0.0
    po_line: str | None = None
    sure: bool = True  # False: the full boat rests on estimated dates of untraced cargo

    @property
    def late_days(self) -> int:
        return (self.on_board - self.need).days if self.on_board else LOOKAHEAD_DAYS


class Planner:
    """Packs items onto the network under the rule cards."""

    def __init__(self, net: Network, rules: RuleSet):
        self.net, self.rules = net, rules
        self.shore = rules.param("shore_handling_days", 1)
        self.transit = rules.param("transit_days", 1)
        self.reserve = rules.param("deck_reserve_pct", 0.0) / 100
        self.heli_item = rules.param("heli_max_item_kg", 0.0)
        self.pax_kg = rules.param("pax_kg", 110.0)

    def lift_day(self, arrive: date, limit: int = 30) -> date | None:
        d = arrive
        for _ in range(limit):
            cranes, bunkering = self.net.cranes.get(d, (1, False))
            lift = Movement("vessel", cranes_available=cranes, bunkering=bunkering)
            if not self.rules.violations(lift):
                return d
            d += timedelta(days=1)
        return None

    def plan(self, items: list[Item]) -> tuple[list[Leg], list[Shortfall]]:
        deck = {s.sailing_id: s.deck_free_m2 - s.deck_m2 * self.reserve for s in self.net.sailings}
        load = {s.sailing_id: s.payload_free_kg for s in self.net.sailings}
        heli = {f.flight_id: f.cargo_booked_kg + f.pax * self.pax_kg for f in self.net.flights}
        legs, short = [], []
        for it in sorted(items, key=lambda i: (i.need, -i.priority, i.component)):
            earliest = it.ready + timedelta(days=self.shore)
            eligible = [s for s in self.net.sailings if s.depart >= earliest]
            if not eligible:
                continue
            ideal = eligible[0].depart + timedelta(days=self.transit)
            chosen = next(
                (
                    s
                    for s in eligible
                    if s.sailing_id not in self.net.cancelled
                    and deck[s.sailing_id] >= it.area_m2
                    and load[s.sailing_id] >= it.weight_kg
                ),
                None,
            )
            on_board = None
            if chosen:
                deck[chosen.sailing_id] -= it.area_m2
                load[chosen.sailing_id] -= it.weight_kg
                on_board = self.lift_day(chosen.depart + timedelta(days=self.transit))
                legs.append(
                    Leg(
                        it.component,
                        "vessel",
                        chosen.sailing_id,
                        chosen.depart,
                        on_board or date.max,
                    )
                )
            if not it.report or (on_board and on_board <= it.need):
                continue
            if ideal > it.need:
                cause = "supply"
            elif eligible[0].sailing_id in self.net.cancelled:
                cause = "weather"
            elif chosen is None or chosen is not eligible[0]:
                cause = "capacity"
            else:
                cause = "crane"
            sf = Shortfall(
                it.component,
                it.work_order,
                it.material,
                it.need,
                on_board,
                cause,
                weight_kg=it.weight_kg,
                area_m2=it.area_m2,
                po_line=it.po_line,
            )
            self._option(sf, it, eligible, deck, heli)
            short.append(sf)
        return legs, short

    def _option(self, sf: Shortfall, it: Item, eligible, deck, heli) -> None:
        """The cheaper of helicopter cargo and an extra deck slot that lands it in time."""
        options = []
        for f in self.net.flights:
            if f.day < it.ready or f.day > it.need or f.flight_id in self.net.grounded:
                continue
            m = Movement(
                "helicopter",
                it.weight_kg,
                payload_used_kg=heli[f.flight_id] + it.weight_kg,
                max_payload_kg=f.max_payload_kg,
            )
            if it.weight_kg <= self.heli_item and not self.rules.violations(m):
                options.append((heli_cost(it.weight_kg), "helicopter", f.flight_id, f.day))
                break
        if sf.cause != "supply":
            for s in eligible:
                if s.sailing_id in self.net.cancelled:
                    continue
                on_board = self.lift_day(s.depart + timedelta(days=self.transit))
                if on_board and on_board <= it.need:
                    options.append((slot_cost(it.area_m2), "vessel_slot", s.sailing_id, on_board))
                    break
        if options:
            _, sf.option, sf.option_ref, sf.option_on_board = min(options)
            if sf.option == "helicopter":
                heli[sf.option_ref] += it.weight_kg


def agent_plan(planner: Planner, items: list[Item]) -> tuple[list[Leg], list[Shortfall]]:
    """The agent's plan over all cargo on order, untraced lines included.

    Whether a boat is full depends on every line, so untraced lines are packed on their
    estimated need date, and every capacity shortfall this gives is reported. Only those that
    also appear when just the parts traced to a work order are packed (the least the boats
    are sure to carry) are marked ``sure``; the rest rest on the estimate, so they go to the
    planner as warnings and do not claim spare stock."""
    legs, short = planner.plan(items)
    _, sure = planner.plan([i for i in items if not i.component.startswith("po:")])
    traced = {s.component for s in sure if s.cause == "capacity"}
    for s in short:
        s.sure = s.cause != "capacity" or s.component in traced
    return legs, short


def _priority(in_turnaround: bool, criticality: str) -> int:
    return 2 * in_turnaround + (criticality == "A")


def expected_delivery(f: Facts) -> date | None:
    expected = f.delivery
    for d, _ in f.slips:
        if expected is None or d > expected:
            expected = d
    return expected


def _reported(need: date, as_of: date) -> bool:
    return as_of <= need <= as_of + timedelta(days=LOOKAHEAD_DAYS)


def agent_items(facts: list[Facts], chain: ReadinessAgent, as_of: date) -> list[Item]:
    """Everything on order that will sail within the lookahead and has not sailed yet.

    Parts the agent can trace to a work order carry their need date and are reported on.
    PO lines it cannot trace still take deck space: they are packed with an estimated need
    (``UNTRACED_SLACK_DAYS`` after delivery) and never reported."""
    c = chain.c
    weight = {r.key: float(r.fields.get("BRGEW") or 0) for r in c["erp-mm"].iter("material")}
    wos = {r.key: r.fields for r in c["erp-pm"].iter("work_order")}
    crit = {r.key: r.fields.get("ABCKZ", "") for r in c["erp-pm"].iter("equipment")}
    horizon = as_of + timedelta(days=LOOKAHEAD_DAYS)
    out = []
    for f in facts:
        expected = expected_delivery(f)
        if f.po_line is None or expected is None or (expected > horizon and f.need > horizon):
            continue  # a part due in the window but arriving after it is late for sure
        if any(d <= as_of for d, _, _ in f.shipments):
            continue
        if "shipment" in f.unsettled and (
            expected < as_of
            or any(
                (d := chain.sailings.get(c["vessel"].get("manifest_line", ml).fields["sailing_id"]))
                and d <= as_of
                for ml in chain.open_targets("po_line_manifest_line", f.po_line)
            )
        ):
            continue  # it may be on a manifest line nobody could link yet, already sailed
        wo = wos.get(f.work_order, {})
        out.append(
            Item(
                f.component,
                f.work_order,
                f.material,
                f.qty,
                weight.get(f.material, 0.0) * f.qty,
                max(expected, as_of),
                f.need,
                _priority(wo.get("AUART") == "PM03", crit.get(wo.get("EQUNR", ""), "")),
                f.po_line,
                _reported(f.need, as_of),
            )
        )
    traced = {f.po_line for f in facts if f.po_line}
    for key, r in chain.po_lines.items():
        delivery = parse_date(r.fields["EINDT"])
        if key in traced or delivery is None or delivery > horizon:
            continue
        if delivery < as_of or any(
            (d := chain.sailings.get(c["vessel"].get("manifest_line", ml).fields["sailing_id"]))
            and d <= as_of
            for ml in chain.linked_from("po_line_manifest_line", key)
        ):
            continue  # sailed, or delivered with no way to tell where it is
        qty = leading_int(r.fields["MENGE"]) or 1
        out.append(
            Item(
                f"po:{key}",
                "",
                r.fields["MATNR"],
                qty,
                weight.get(r.fields["MATNR"], 0.0) * qty,
                delivery,
                delivery + timedelta(days=UNTRACED_SLACK_DAYS),
                po_line=key,
                report=False,
            )
        )
    return out


def truth_items(world: World, as_of: date) -> list[Item]:
    pl_by_comp = {pl.component_key: pl for pl in world.po_lines}
    mat = {m.mat_no: m for m in world.materials}
    wo = {w.wo_no: w for w in world.work_orders}
    eq = {e.equip_no: e for e in world.equipment}
    depart = {s.sailing_id: s.depart for s in world.sailings}
    sailed = {
        k for ml in world.manifest_lines if depart[ml.sailing_id] <= as_of for k in ml.po_line_keys
    }
    slips: dict[str, list[date]] = {}
    for e in world.emails:
        if e.kind == "slip" and e.sent <= as_of:
            for k in e.po_line_keys:
                slips.setdefault(k, []).append(e.new_date)
    out = []
    for c in world.components:
        pl = pl_by_comp.get(c.key)
        if not pl or pl.order_date > as_of or pl.key in sailed:
            continue
        expected = max([pl.delivery_date, *slips.get(pl.key, [])])
        horizon = as_of + timedelta(days=LOOKAHEAD_DAYS)
        if expected > horizon and c.req_date > horizon:
            continue
        w = wo[c.wo_no]
        out.append(
            Item(
                c.key,
                c.wo_no,
                c.mat_no,
                c.qty,
                mat[c.mat_no].unit_weight_kg * c.qty,
                max(expected, as_of),
                c.req_date,
                _priority(w.in_turnaround, eq[w.equip_no].criticality),
                pl.key,
                _reported(c.req_date, as_of),
            )
        )
    return out


def baseline_notice(sf: Shortfall, net: Network, transit: int) -> date:
    """When a planner notices today: the manifest of the last sailing that could make the
    need date is issued without the part."""
    useful = [s for s in net.sailings if s.depart + timedelta(days=transit) <= sf.need]
    if not useful:
        return sf.need - timedelta(days=14)
    return useful[-1].depart - timedelta(days=MANIFEST_LEAD_DAYS)


def score_shortfalls(agent: list[Shortfall], truth: list[Shortfall]) -> dict:
    a = {(s.component, s.cause) for s in agent}
    t = {(s.component, s.cause) for s in truth}
    ac, tc = {s.component for s in agent}, {s.component for s in truth}
    t_opt = {s.component: s.option for s in truth}
    both = ac & tc
    return {
        "reported": len(a),
        "true": len(t),
        "precision": round(len(ac & tc) / len(ac), 4) if ac else None,
        "recall": round(len(ac & tc) / len(tc), 4) if tc else None,
        "cause_right": round(len(a & t) / len(both), 4) if both else None,
        "option_right": round(
            sum(s.option == t_opt[s.component] for s in agent if s.component in both) / len(both), 4
        )
        if both
        else None,
        "by_cause": {
            k: {
                "reported": sum(s.cause == k for s in agent),
                "true": sum(s.cause == k for s in truth),
                "right": sum(s.cause == k and s.component in tc for s in agent),
            }
            for k in CAUSES
        },
    }


def earlier_shortfalls(
    days: dict[date, tuple[list[Shortfall], list[Shortfall]]], net: Network, transit: int
) -> dict:
    """Days between the agent's first correct flag of a logistics shortfall and today's
    baseline (the last useful manifest issued without it)."""
    first: dict[str, date] = {}
    true_sf: dict[str, Shortfall] = {}
    for d in sorted(days):
        agent, truth = days[d]
        t = {s.component for s in truth if s.cause != "supply"}
        for s in truth:
            if s.cause != "supply":
                true_sf.setdefault(s.component, s)
        for s in agent:
            if s.component in t:
                first.setdefault(s.component, d)
    gains = sorted((baseline_notice(true_sf[c], net, transit) - first[c]).days for c in first)
    return {
        "logistics_shortfalls": len(true_sf),
        "detected": len(gains),
        "median_days_earlier": gains[len(gains) // 2] if gains else None,
        "share_before_baseline": round(sum(g > 0 for g in gains) / len(gains), 4)
        if gains
        else None,
    }
