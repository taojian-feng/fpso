import json
from datetime import date, timedelta

import pytest

from fpso.campaign import WINDOW_DAYS, campaigns
from fpso.condition import ConditionAgent, score_alerts
from fpso.connectors import open_all
from fpso.events import EventBus
from fpso.logistics import Flight, Item, Network, Planner, Sailing
from fpso.orchestrator import settle
from fpso.phase3 import run_orchestration
from fpso.rule_cards import load_cards
from fpso.scenario import Scenario, what_if
from fpso.synthetic.export import export
from fpso.synthetic.truth import build_world
from fpso.weather import Outlook, with_weather

D = date(2027, 3, 1)
RULES = load_cards()


def _net(flights: int = 10, payload: float = 2400) -> Network:
    sailings = [Sailing(f"S{i}", D + timedelta(days=3 * i), 500.0, 60.0, 50_000) for i in range(8)]
    fl = [Flight(f"H{i}", D + timedelta(days=i), 10, payload, 0) for i in range(flights)]
    return Network(sailings, fl, {D + timedelta(days=i): (1, False) for i in range(40)})


def _item(name, need_days, weight=10.0, ready_days=0) -> Item:
    return Item(
        name,
        "wo",
        "mat",
        1,
        weight,
        D + timedelta(days=ready_days),
        D + timedelta(days=need_days),
    )


def test_a_forecast_storm_turns_a_sailing_into_a_weather_shortfall():
    net = _net(flights=0)
    out = Outlook(D, {D + timedelta(days=3): (6.0, 70.0)})
    stormy = with_weather(net, out, RULES)
    assert stormy.cancelled == {"S1"} and not net.cancelled
    _, short = Planner(stormy, RULES).plan([_item("a", 5)])
    assert short[0].cause == "weather"


def test_scenarios_fork_the_plan_and_leave_it_alone():
    net, items = _net(flights=0), [_item("a", 5)]
    o = what_if(net, items, RULES, Scenario("cancel_sailing", "S1"), {"a": 1000.0})
    assert o.newly_late == ["a"] and o.extra_cost_of_delay_usd == 1000.0 * o.extra_late_days
    assert not net.cancelled
    storm = what_if(net, items, RULES, Scenario("storm", start=D, days=10))
    assert storm.newly_late == ["a"]


def test_the_costlier_part_gets_the_last_helicopter_payload():
    # Payload left on the one flight: 2400 - 10 x 110 kg passengers = 1300 kg, one part fits.
    net = _net(flights=1, payload=1300 + 10 * 110)
    net = Network(net.sailings, net.flights, net.cranes, {"S1", "S2", "S3"})
    cheap, dear = _item("cheap", 5, weight=100), _item("dear", 5, weight=100)
    net.flights[0].cargo_booked_kg = 1200
    _, short = Planner(net, RULES).plan([cheap, dear])
    s = settle(D, [cheap, dear], short, [], [], net, RULES, {"cheap": 1e3, "dear": 1e5}, {}, {})
    got = {c.component: c.action.kind if c.action else None for c in s.choices}
    assert got["dear"] == "book_helicopter_cargo" and got["cheap"] != "book_helicopter_cargo"


def test_only_cheap_certain_fixes_run_without_approval():
    # Delivered too late for any boat (supply), so each flies; c's boat is forecast away.
    net = _net(flights=8)
    items = [_item("a", 5, weight=20, ready_days=4), _item("b", 5, weight=20, ready_days=4)]
    _, short = Planner(net, RULES).plan(items)
    stormy = Network(net.sailings, net.flights, net.cranes, {"S1"})
    c = _item("c", 4, weight=20)
    _, c_short = Planner(stormy, RULES).plan([c])
    assert [x.cause for x in short + c_short] == ["supply", "supply", "weather"]
    rate = {k: 50_000.0 for k in "abc"}
    s = settle(D, items + [c], short + c_short, [], [], stormy, RULES, rate, {}, {}, {"b"})
    auto = {c.component: c.auto for c in s.choices}
    assert auto == {"a": True, "b": False, "c": False}
    for c in s.choices:
        assert c.action.cost_usd < RULES.param("auto_limit_usd", 0) or not c.auto


def test_event_bus_delivers_by_prefix_and_logs(tmp_path):
    bus = EventBus(tmp_path / "events.jsonl")
    got = []
    bus.subscribe("plan.*", lambda ev: got.append(ev.topic))
    bus.publish("plan.shortfall", component="a")
    bus.publish("weather.sailing_stopped", sailing="S1")
    assert got == ["plan.shortfall"]
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("kA3")
    export(build_world(seed=11, scale=0.1), d, "A")
    return d


def test_condition_alerts_come_before_the_corrective_work_order(small_run):
    from fpso.readiness import load_world
    from fpso.synthetic.operations import build_operations

    ops = build_operations(load_world(small_run))
    agent = ConditionAgent(open_all(small_run / "extracts"))
    start = date(2027, 1, 4)
    days = {
        start + timedelta(days=i): agent.run(start + timedelta(days=i)) for i in range(0, 180, 3)
    }
    s = score_alerts(days, ops.failing)
    assert s["precision"] >= 0.7 and s["recall"] >= 0.7
    assert s["median_days_before_baseline"] > 0


def test_campaigns_group_one_system_within_the_window(small_run):
    out = campaigns(open_all(small_run / "extracts"), date(2027, 2, 1))
    assert out
    for c in out:
        assert len(c.work_orders) >= 2 and c.spread_days <= WINDOW_DAYS
        assert c.start == max(c.work_order_starts.values())


def test_replayed_orchestration_has_no_incidents(small_run):
    s = run_orchestration(small_run, step_days=14, use_model=False)
    assert s["incidents"]["count"] == 0
    assert s["expedite_spend"]["down_share"] is None or s["expedite_spend"]["down_share"] > 0
    assert s["events"]["source.new"] > 0
    out = small_run / "reports/orchestrator"
    assert json.loads((out / "summary.json").read_text())["days"] == s["days"]
    assert (out / "events.jsonl").exists()


def test_profile_b_runs_only_when_final(tmp_path):
    export(build_world(seed=11, scale=0.05), tmp_path, "B")
    with pytest.raises(SystemExit):
        run_orchestration(tmp_path, use_model=False)
    s = run_orchestration(tmp_path, step_days=28, use_model=False, final=True)
    assert s["run"]["profile"] == "B"
