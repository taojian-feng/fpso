import json
from datetime import date, timedelta

import pytest

from fpso.eval.reviewers import SimulatedReviewer
from fpso.inventory import Balance, find_stock
from fpso.logistics import Flight, Item, Network, Planner, Sailing
from fpso.phase2 import run_logistics
from fpso.procurement import Recommendation
from fpso.rule_cards import compile_card, load_cards
from fpso.synthetic import export as export_mod
from fpso.synthetic.export import export
from fpso.synthetic.truth import build_world
from fpso.writeback import Action, ApprovalQueue

D = date(2027, 3, 1)
RULES = load_cards()


def test_every_default_card_compiles_and_passes_its_own_tests():
    assert RULES.cards and not RULES.not_understood
    assert RULES.self_test() == []
    assert RULES.param("heli_max_item_kg", None) == 150


def test_a_sentence_no_template_matches_is_not_guessed():
    assert compile_card("Pumps should be handled with care.") is None
    card = compile_card("No item heavier than 80 kg goes by helicopter.")
    assert card.params["value"] == 80 and card.self_test() == []


def _net(deck_free=50.0, bunker: set[date] = frozenset(), flights=True) -> Network:
    sailings = [
        Sailing(f"S{i}", D + timedelta(days=3 * i), 500.0, deck_free, 50_000) for i in range(6)
    ]
    fl = [Flight(f"H{i}", D + timedelta(days=i), 10, 2400, 0) for i in range(20)] if flights else []
    cranes = {D + timedelta(days=i): (1, D + timedelta(days=i) in bunker) for i in range(40)}
    return Network(sailings, fl, cranes)


def _item(name, need_days, weight=10.0, ready_days=0, priority=0) -> Item:
    return Item(
        name,
        "wo",
        "mat",
        1,
        weight,
        D + timedelta(days=ready_days),
        D + timedelta(days=need_days),
        priority,
    )


def test_a_full_sailing_pushes_the_part_to_the_next_and_flags_capacity():
    # Deck reserve is 2% of 500 m2 = 10 m2, so with 11.5 m2 free only one pallet fits.
    p = Planner(_net(deck_free=11.5, flights=False), RULES)
    legs, short = p.plan([_item("a", 5, weight=100), _item("b", 5, weight=100)])
    assert [leg.ref for leg in legs] == ["S1", "S2"]
    assert [(s.component, s.cause) for s in short] == [("b", "capacity")]


def test_turnaround_parts_go_first_on_the_same_need_date():
    p = Planner(_net(deck_free=11.5, flights=False), RULES)
    _, short = p.plan([_item("a", 5, weight=100), _item("b", 5, weight=100, priority=2)])
    assert [s.component for s in short] == ["a"]


def test_the_cheaper_fix_is_offered():
    p = Planner(_net(deck_free=11.5), RULES)
    _, short = p.plan(
        [_item("a", 5, weight=100), _item("b", 5, weight=20), _item("c", 5, weight=200)]
    )
    fix = {s.component: s.option for s in short}
    assert fix["b"] == "helicopter"  # 20 kg flies for less than a deck slot costs
    assert fix["c"] == "vessel_slot"  # over the 150 kg helicopter limit


def test_no_lift_on_a_bunkering_day():
    p = Planner(_net(bunker={D + timedelta(days=4)}, flights=False), RULES)
    legs, short = p.plan([_item("a", 4)])
    assert legs[0].ref == "S1" and legs[0].on_board == D + timedelta(days=5)
    assert short[0].cause == "crane"


def test_stock_is_found_offshore_first_and_never_claimed_twice():
    recs = [
        Recommendation(c, "wo", "m1", D + timedelta(days=n), "order_now")
        for c, n in (("a", 3), ("b", 10), ("c", 20))
    ]
    balances = [Balance("m1", "KST1", "OFF1", 1), Balance("m1", "MRL1", "OFF1", 1)]
    props = find_stock(recs, [], {"a": 1, "b": 1, "c": 1}, balances, {"m1": 100.0}, RULES, D)
    assert [(p.component, p.action) for p in props] == [
        ("a", "use_offshore_stock"),
        ("b", "transfer_from_sister"),
    ]
    late = find_stock(recs[:1], [], {"a": 1}, balances[1:], {}, RULES, D)
    assert late == []  # the sister asset is 7 days away


def test_only_approved_actions_reach_the_outbox(tmp_path):
    planner = SimulatedReviewer("dana.reyes", "planner", set(), error_rate=0.0, skip_rate=0.0)
    q = ApprovalQueue(tmp_path, planner)
    good = Action("transfer_stock", "a", D)
    bad = Action("book_helicopter_cargo", "b", D)
    decisions = q.review([good, bad], {good.key})
    assert [d.verdict for d in decisions] == ["approved", "rejected"]
    lines = (tmp_path / "outbox" / "warehouse.jsonl").read_text().splitlines()
    assert len(lines) == 1 and not (tmp_path / "outbox" / "helicopter.jsonl").exists()
    later = Action("transfer_stock", "a", D + timedelta(days=7))
    assert (
        q.review([later, Action("book_helicopter_cargo", "b", D + timedelta(days=7))], set()) == []
    )


def test_logistics_extracts_leave_the_earlier_files_unchanged(tmp_path, monkeypatch):
    world = build_world(seed=5, scale=0.05)
    export(world, tmp_path / "with", "A")
    monkeypatch.setattr(export_mod, "export_logistics", lambda w, ex: {})
    export(world, tmp_path / "without", "A")
    old = sorted(
        p.relative_to(tmp_path / "without") for p in (tmp_path / "without/extracts").rglob("*.*")
    )
    for rel in old:
        assert (tmp_path / "with" / rel).read_bytes() == (tmp_path / "without" / rel).read_bytes()
    assert (tmp_path / "with/extracts/helicopter/flights.csv").exists()


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    d = tmp_path_factory.mktemp("kA")
    export(build_world(seed=11, scale=0.1), d, "A")
    return d, run_logistics(d, step_days=14, use_model=False)


def test_replayed_run_scores_every_piece(replayed):
    d, s = replayed
    assert s["cost_usd"] == 0.0 and s["days"] > 5
    assert s["shortfalls"]["by_cause"]["supply"]["precision"] >= 0.9
    # Every full boat is reported, so most are found; in a world this small the estimated
    # packing order of untraced lines is noisy (full size: 86-95% right).
    cap = s["shortfalls"]["by_cause"]["capacity"]
    assert cap["right"] / cap["true"] >= 0.5 and cap["precision"] >= 0.4
    assert s["inventory"]["precision"] >= 0.9
    assert s["approvals"]["precision"] >= 0.7
    assert json.loads((d / "reports/logistics/summary.json").read_text())["days"] == s["days"]


def test_outbox_holds_only_approved_actions(replayed):
    d, s = replayed
    approved = [
        json.loads(line)
        for line in (d / "reports/logistics/decisions.jsonl").read_text().splitlines()
        if json.loads(line)["verdict"] == "approved"
    ]
    written = [
        json.loads(line)["id"]
        for p in (d / "reports/logistics/outbox").glob("*.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert sorted(written) == sorted(a["action"] for a in approved)


def test_profile_b_is_refused(tmp_path):
    export(build_world(seed=11, scale=0.05), tmp_path, "B")
    with pytest.raises(SystemExit):
        run_logistics(tmp_path, use_model=False)
