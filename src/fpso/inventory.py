"""Inventory agent (Phase 2, autonomy level 2: act with approval).

Before a part is bought or flown out, it looks for a spare one already owned: in the asset's
own offshore store, or on the sister asset, whose same-model equipment uses the same spares.
Stock reserved for other work orders is left alone. Parts are matched soonest need first,
so two shortages never claim the same spare.

- ``use_offshore_stock``: free stock in the offshore store covers the need now;
- ``transfer_from_sister``: free stock on the sister asset arrives in time (the transfer time
  comes from the rule cards).

Each proposal names the store balance it draws on and the purchase or expedite it avoids.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from fpso.connectors import Connector
from fpso.logistics import Shortfall
from fpso.procurement import Recommendation
from fpso.rule_cards import RuleSet
from fpso.synthetic.logistics import OFFSHORE_STORE, SISTER_ASSET, Logistics


@dataclass
class Balance:
    material: str
    plant: str
    location: str
    free: int


@dataclass
class StockProposal:
    component: str
    work_order: str
    material: str
    qty: int
    need: date
    action: str  # use_offshore_stock, transfer_from_sister
    store: str  # plant-location drawn on
    replaces: str  # order_now, expedite, or the shortfall cause
    value_usd: float = 0.0  # price of the parts not bought
    left: int = 0  # free units of the material still in that store after this claim


def balances_from_connectors(c: dict[str, Connector]) -> list[Balance]:
    return [
        Balance(
            r.fields["MATNR"],
            r.fields["WERKS"],
            r.fields["LGORT"],
            int(r.fields["LABST"]) - int(r.fields["RESERVED"] or 0),
        )
        for r in c["warehouse"].iter("store_balance")
    ]


def balances_from_truth(lg: Logistics) -> list[Balance]:
    return [Balance(s.mat_no, s.plant, s.location, s.qty - s.reserved_qty) for s in lg.stores]


def find_stock(
    recs: list[Recommendation],
    shortfalls: list[Shortfall],
    qty: dict[str, int],
    balances: list[Balance],
    prices: dict[str, float],
    rules: RuleSet,
    as_of: date,
) -> list[StockProposal]:
    """Spare stock for each part about to be bought late, expedited or short on board.

    ``qty`` is the quantity each component needs."""
    transfer = timedelta(days=rules.param("transfer_days", 7))
    free = {(b.material, b.plant): b.free for b in balances if b.free > 0}
    wants: dict[str, tuple] = {}
    for r in recs:
        if r.action in ("order_now", "expedite"):
            wants[r.component] = (r.work_order, r.material, r.need, r.action)
    for s in shortfalls:
        if not s.sure:
            continue  # a full boat that rests on estimated dates; the planner decides
        wants.setdefault(s.component, (s.work_order, s.material, s.need, s.cause))
    out = []
    for comp, (wo, mat, need, why) in sorted(wants.items(), key=lambda kv: (kv[1][2], kv[0])):
        n = qty.get(comp, 1)
        for plant, action, in_time in (
            (OFFSHORE_STORE[0], "use_offshore_stock", True),
            (SISTER_ASSET[0], "transfer_from_sister", as_of + transfer <= need),
        ):
            if in_time and free.get((mat, plant), 0) >= n:
                free[(mat, plant)] -= n
                value = round(prices.get(mat, 0.0) * n, 2)
                out.append(
                    StockProposal(
                        comp,
                        wo,
                        mat,
                        n,
                        need,
                        action,
                        f"{plant}-OFF1",
                        why,
                        value,
                        free[(mat, plant)],
                    )
                )
                break
    return out


def score_proposals(agent: list[StockProposal], truth: list[StockProposal]) -> dict:
    a = {(p.component, p.action) for p in agent}
    t = {(p.component, p.action) for p in truth}
    return {
        "reported": len(a),
        "true": len(t),
        "precision": round(len(a & t) / len(a), 4) if a else None,
        "recall": round(len(a & t) / len(t), 4) if t else None,
        "value_usd": round(sum(p.value_usd for p in agent if (p.component, p.action) in t), 2),
    }
