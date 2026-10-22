from datetime import date, timedelta

from fpso.procurement import Recommendation
from fpso.schedule import (
    Activity,
    forward_pass,
    impact_of,
    part_slip,
    truth_activities,
    turnaround_delay,
)
from fpso.synthetic.truth import build_world

D = date(2027, 4, 10)


def _chain() -> dict[str, Activity]:
    # a (2 days) -> b, starting 3 days after a ends -> c, starting the day after b ends.
    acts = {
        "a": Activity("a", D, D + timedelta(days=1), ["b"], turnaround=True),
        "b": Activity("b", D + timedelta(days=5), D + timedelta(days=6), ["c"], turnaround=True),
        "c": Activity("c", D + timedelta(days=7), D + timedelta(days=9), [], turnaround=True),
        "x": Activity("x", D, D + timedelta(days=20)),  # outside the turnaround
    }
    return acts


def test_a_gap_absorbs_part_of_the_delay():
    acts = _chain()
    assert forward_pass(acts, "a", 2) == {"a": 2}  # a ends day 3; b still starts day 5
    moved = forward_pass(acts, "a", 6)
    assert moved == {"a": 6, "b": 3, "c": 3}
    assert turnaround_delay(acts, moved) == 3


def test_the_turnaround_end_ignores_activities_outside_it():
    acts = _chain()
    assert turnaround_delay(acts, {"x": 30}) == 0


def test_impact_counts_slack_between_need_and_activity_start():
    acts = _chain()
    rec = Recommendation("comp", "wo", "mat", D - timedelta(days=4), "expedite", days_late=10)
    imp = impact_of(rec, 10, "a", acts)
    assert imp.activity_delay == 6  # 4 of the 10 days fall before the activity starts
    assert imp.turnaround_delay == 3
    assert imp.moved == ["a", "b", "c"]
    assert impact_of(rec, 10, None, acts).turnaround_delay == 0


def test_part_slip_for_an_order_placed_today():
    rec = Recommendation("comp", "wo", "mat", D, "order_now")
    assert part_slip(rec, D - timedelta(days=30), lead_days=40) == 14  # 40 + 4 - 30
    assert part_slip(rec, D - timedelta(days=60), lead_days=40) == 0


def test_truth_schedule_has_turnaround_chains():
    world = build_world(seed=11, scale=0.1)
    acts, act_of = truth_activities(world)
    ta = [a for a in acts.values() if a.turnaround]
    assert ta and any(a.successors for a in ta)
    assert max(a.finish for a in ta) <= world.turnaround_end
    assert set(act_of) <= {w.wo_no for w in world.work_orders}


def test_an_overlap_already_in_the_plan_does_not_push_by_itself():
    acts = _chain()
    acts["b"].start = D  # the plan already starts b before a finishes
    assert forward_pass(acts, "a", 1) == {"a": 1, "b": 1, "c": 1}
