"""Logistics ground truth for Phase 2: deck space, helicopter seats, crane days, other stores.

Built from the world with its own random stream, so adding it changes no byte of the Phase 0
and Phase 1 extracts. Logistics records are plans the operator's own teams publish (vessel
bookings, flight lists, the marine operations calendar, store balances), so they are
exported clean:

    extracts/
      vessel/sailing_capacity.csv      deck and payload per sailing, and what is booked already
      helicopter/flights.csv           daily crew-change flights: seats, passengers, payload
      marine_ops/crane_ops.csv         cranes available per day and bunkering days
      warehouse/stock_locations.csv    stock offshore and on the sister asset, with reservations
      erp_mm/material_prices.csv       unit price per material
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from fpso.synthetic.truth import World

VESSEL_DECK = {"MV Skerry Supporter": (620.0, 2800.0), "MV Northwind Tern": (480.0, 2100.0)}
HELI_SEATS, HELI_PAYLOAD_KG = 19, 2400
SISTER_ASSET = ("MRL1", "FPSO Merlin")
OFFSHORE_STORE = ("KST1", "OFF1")


@dataclass
class SailingCapacity:
    sailing_id: str
    deck_m2: float
    deck_booked_m2: float
    payload_t: float
    payload_booked_t: float


@dataclass
class Flight:
    flight_id: str
    day: date
    seats: int
    pax_booked: int
    max_payload_kg: int
    cargo_booked_kg: int


@dataclass
class CraneDay:
    day: date
    cranes_total: int
    cranes_available: int
    bunkering: bool


@dataclass
class StoreBalance:
    mat_no: str
    plant: str
    location: str
    qty: int
    reserved_qty: int = 0
    reserved_for: str = ""


@dataclass
class Logistics:
    capacity: dict[str, SailingCapacity]
    flights: list[Flight]
    crane_days: dict[date, CraneDay]
    stores: list[StoreBalance]
    prices: dict[str, float] = field(default_factory=dict)


def _in_turnaround_season(world: World, d: date) -> bool:
    return world.turnaround_start - timedelta(days=21) <= d <= world.turnaround_end


def build_logistics(world: World) -> Logistics:
    rng = random.Random(f"logistics-{world.seed}")
    capacity = {}
    for s in world.sailings:
        # The boats serve the whole field; the share left for this asset follows its size.
        deck, payload = (round(x * world.scale, 1) for x in VESSEL_DECK[s.vessel])
        lo, hi = (0.82, 0.97) if _in_turnaround_season(world, s.depart) else (0.55, 0.90)
        share = rng.uniform(lo, hi)
        capacity[s.sailing_id] = SailingCapacity(
            s.sailing_id,
            deck,
            round(deck * share, 1),
            payload,
            round(payload * min(0.99, share + rng.uniform(-0.1, 0.02)), 1),
        )

    flights = []
    d = world.window_start - timedelta(days=30)
    n = 0
    while d <= world.window_end + timedelta(days=30):
        busy = _in_turnaround_season(world, d)
        for _ in range(2 if busy else 1):
            n += 1
            pax = rng.randint(16, 19) if busy else rng.randint(9, 17)
            flights.append(
                Flight(
                    f"HX{n:05d}",
                    d,
                    HELI_SEATS,
                    pax,
                    HELI_PAYLOAD_KG,
                    rng.choice([0, 0, 50, 100, 150, 250, 400]),
                )
            )
        d += timedelta(days=1)

    cranes = [e for e in world.equipment if e.eq_class == "CR"]
    total = max(1, len(cranes))
    down: dict[date, int] = {}
    for w in world.work_orders:
        if any(w.equip_no == c.equip_no for c in cranes):
            day = w.planned_start
            while day <= w.planned_finish:
                down[day] = down.get(day, 0) + 1
                day += timedelta(days=1)
    if not cranes:  # one crane still needs its own maintenance days
        for _ in range(6):
            day = world.window_start + timedelta(days=rng.randint(0, 180))
            down[day] = 1
    bunker = set()
    day = world.window_start - timedelta(days=rng.randint(0, 12))
    while day <= world.window_end + timedelta(days=30):
        bunker.add(day)
        day += timedelta(days=rng.randint(10, 14))
    crane_days = {}
    day = world.window_start - timedelta(days=60)
    while day <= world.window_end + timedelta(days=45):
        crane_days[day] = CraneDay(day, total, max(0, total - down.get(day, 0)), day in bunker)
        day += timedelta(days=1)

    bought = {pl.mat_no for pl in world.po_lines}
    wos = [w.wo_no for w in world.work_orders]
    stores = []
    for m in sorted(world.materials, key=lambda m: m.mat_no):
        for (plant, loc), p in ((OFFSHORE_STORE, 0.08), ((SISTER_ASSET[0], "OFF1"), 0.18)):
            if m.mat_no in bought and rng.random() < p:
                qty = rng.randint(1, 4)
                reserved = rng.randint(1, qty) if rng.random() < 0.3 else 0
                stores.append(
                    StoreBalance(
                        m.mat_no,
                        plant,
                        loc,
                        qty,
                        reserved,
                        rng.choice(wos) if reserved and plant == OFFSHORE_STORE[0] else "",
                    )
                )
    prices = {
        m.mat_no: round(
            m.unit_weight_kg * rng.uniform(40, 120)
            + rng.uniform(200, 4000) * (m.lead_time_days / 60),
            2,
        )
        for m in world.materials
    }
    return Logistics(capacity, flights, crane_days, stores, prices)


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def export_logistics(world: World, extracts: str | Path) -> dict:
    ex = Path(extracts)
    lg = build_logistics(world)
    _write(
        ex / "vessel" / "sailing_capacity.csv",
        ["sailing_id", "deck_m2", "deck_booked_m2", "payload_t", "payload_booked_t"],
        [
            [c.sailing_id, c.deck_m2, c.deck_booked_m2, c.payload_t, c.payload_booked_t]
            for c in lg.capacity.values()
        ],
    )
    _write(
        ex / "helicopter" / "flights.csv",
        ["flight_id", "date", "route", "seats", "pax_booked", "max_payload_kg", "cargo_booked_kg"],
        [
            [
                f.flight_id,
                f.day.isoformat(),
                "HELIBASE-KST",
                f.seats,
                f.pax_booked,
                f.max_payload_kg,
                f.cargo_booked_kg,
            ]
            for f in lg.flights
        ],
    )
    _write(
        ex / "marine_ops" / "crane_ops.csv",
        ["date", "cranes_total", "cranes_available", "bunkering"],
        [
            [c.day.isoformat(), c.cranes_total, c.cranes_available, "Y" if c.bunkering else "N"]
            for c in lg.crane_days.values()
        ],
    )
    _write(
        ex / "warehouse" / "stock_locations.csv",
        ["MATNR", "WERKS", "LGORT", "LABST", "RESERVED", "RES_AUFNR"],
        [[s.mat_no, s.plant, s.location, s.qty, s.reserved_qty, s.reserved_for] for s in lg.stores],
    )
    _write(
        ex / "erp_mm" / "material_prices.csv",
        ["MATNR", "PRICE", "WAERS"],
        [[m, p, "USD"] for m, p in sorted(lg.prices.items())],
    )
    return {
        "sailing_capacity": len(lg.capacity),
        "flights": len(lg.flights),
        "crane_days": len(lg.crane_days),
        "store_balances": len(lg.stores),
    }
