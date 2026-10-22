import json
from datetime import timedelta

from fpso.fleet.a2a import (
    Bus,
    LiaisonClient,
    Reliability,
    SupplierAgent,
    build_bus,
    order_book,
    score_exchanges,
)
from fpso.fleet.assets import MERLIN, asset_world

SCALE = 0.05


def _world():
    return asset_world(MERLIN, SCALE)


def _open_po(world):
    """A PO, its vendor and a day it is open on."""
    pl = world.po_lines[len(world.po_lines) // 2]
    return pl, pl.order_date + timedelta(days=8)


def _bus(world, reliability):
    pl, _ = _open_po(world)
    vendor = next(v for v in world.vendors if v.vendor_id == pl.vendor_id)
    bus = Bus()
    bus.register(SupplierAgent(vendor, {"MRL": world}, reliability))
    return bus


def test_agent_card_lists_skills_and_is_plain_json():
    world = _world()
    card = _bus(world, Reliability()).cards()[0]
    assert {s["id"] for s in card["skills"]} == {"order_status", "expedite"}
    assert card["url"].startswith("a2a://") and json.loads(json.dumps(card)) == card


def test_order_status_answers_from_the_order_book():
    world = _world()
    pl, day = _open_po(world)
    client = LiaisonClient(_bus(world, Reliability()), "MRL")
    ex = client.ask(day, {"po": pl.po_no, "vendor": pl.vendor_id, "kind": "no_ack"})
    assert ex.channel == "a2a" and ex.answer["acknowledged"]
    line = next(x for x in ex.answer["lines"] if x["line"] == pl.key)
    assert line["promised"] == order_book(world, pl, day)[0].isoformat()


def test_unknown_order_fails_and_falls_back_to_email():
    world = _world()
    pl, day = _open_po(world)
    client = LiaisonClient(_bus(world, Reliability()), "MRL")
    ex = client.ask(day, {"po": "4599999999", "vendor": pl.vendor_id, "kind": "no_ack"})
    assert ex.channel == "email" and ex.reason == "failed"
    ex = client.ask(day, {"po": pl.po_no, "vendor": "999999", "kind": "no_ack"})
    assert ex.channel == "email" and ex.reason == "no_agent"


def test_silent_and_late_agents_leave_the_email_in_place():
    world = _world()
    pl, day = _open_po(world)
    chase = {"po": pl.po_no, "vendor": pl.vendor_id, "kind": "expedite"}
    silent = LiaisonClient(_bus(world, Reliability(answers=0.0)), "MRL").ask(day, chase)
    assert (silent.channel, silent.reason) == ("email", "no_answer")
    late = LiaisonClient(_bus(world, Reliability(late=1.0, late_hours=48)), "MRL").ask(day, chase)
    assert (late.channel, late.reason) == ("email", "late")
    fast = LiaisonClient(_bus(world, Reliability(late=1.0, late_hours=12)), "MRL").ask(day, chase)
    assert fast.channel == "a2a" and "earliest" in fast.answer["lines"][0]


def test_a_supplier_knows_of_a_slip_before_its_email():
    world = _world()
    e = next(e for e in world.emails if e.kind == "slip")
    pl = next(p for p in world.po_lines if p.key == e.po_line_keys[0])
    before = e.sent - timedelta(days=5)
    assert order_book(world, pl, before)[0] >= e.new_date
    assert order_book(world, pl, e.sent - timedelta(days=30))[0] == pl.delivery_date


def test_stale_answers_lower_date_accuracy_on_slipped_lines():
    world = _world()
    e = next(e for e in world.emails if e.kind == "slip")
    pl = next(p for p in world.po_lines if p.key == e.po_line_keys[0])
    chase = {"po": pl.po_no, "vendor": pl.vendor_id, "kind": "no_ack"}
    good = LiaisonClient(_bus_for(world, pl, Reliability()), "MRL").ask(e.sent, chase)
    stale = LiaisonClient(_bus_for(world, pl, Reliability(stale=1.0)), "MRL").ask(e.sent, chase)
    assert score_exchanges([good], world)["slipped_line_accuracy"] == 1.0
    assert score_exchanges([stale], world)["slipped_line_accuracy"] == 0.0


def _bus_for(world, pl, reliability):
    vendor = next(v for v in world.vendors if v.vendor_id == pl.vendor_id)
    bus = Bus()
    bus.register(SupplierAgent(vendor, {"MRL": world}, reliability))
    return bus


def test_default_fleet_bus_scores_a_mix_of_channels():
    world = _world()
    bus = build_bus({"MRL": world})
    assert 0 < len(bus.agents) < len(world.vendors)
    client = LiaisonClient(bus, "MRL")
    exchanges = []
    for pl in world.po_lines[:60]:
        exchanges.append(
            client.ask(
                pl.order_date + timedelta(days=8),
                {"po": pl.po_no, "vendor": pl.vendor_id, "kind": "no_ack"},
            )
        )
    s = score_exchanges(exchanges, world)
    assert s["questions"] == 60
    assert 0 < s["resolved_by_a2a"] < 60
    assert s["email_reasons"].get("no_agent", 0) > 0
    assert s["promised_date_accuracy"] >= 0.8
