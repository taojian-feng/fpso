from datetime import date, timedelta

import pytest

from fpso.procurement import (
    MIN_SHIP_DAYS,
    WARN_DAYS,
    Recommendation,
    earlier_detection,
    recommend,
    run_procurement,
    truth_recommendations,
)
from fpso.readiness import Facts
from fpso.synthetic.export import export
from fpso.synthetic.truth import build_world

TODAY = date(2027, 2, 1)


def _facts(need_days: int, **kw) -> Facts:
    f = Facts("0012-0001", "41000001", "40010001", 1, TODAY + timedelta(days=need_days), 0)
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def test_order_now_once_the_latest_order_date_has_passed():
    rec = recommend(_facts(60), lead_days=60, as_of=TODAY)
    assert rec.action == "order_now"
    assert rec.order_by == TODAY - timedelta(days=MIN_SHIP_DAYS)
    assert rec.days_late == MIN_SHIP_DAYS


def test_order_soon_inside_the_warning_window_and_nothing_beyond_it():
    assert recommend(_facts(70), lead_days=60, as_of=TODAY).action == "order_soon"
    far = MIN_SHIP_DAYS + 60 + WARN_DAYS + 1
    assert recommend(_facts(far), lead_days=60, as_of=TODAY) is None


def test_stock_that_covers_the_need_means_no_order():
    assert recommend(_facts(10, stock=1), lead_days=60, as_of=TODAY) is None


def test_expedite_uses_the_latest_supplier_date():
    need = TODAY + timedelta(days=30)
    on_time = _facts(30, po_line="4500210001-00010", delivery=need - timedelta(days=10))
    assert recommend(on_time, lead_days=60, as_of=TODAY) is None
    slipped = _facts(
        30,
        po_line="4500210001-00010",
        delivery=need - timedelta(days=10),
        slips=[(need + timedelta(days=5), "msg-000001")],
    )
    rec = recommend(slipped, lead_days=60, as_of=TODAY)
    assert rec.action == "expedite"
    assert rec.evidence == "msg-000001"
    assert rec.days_late == 5 + MIN_SHIP_DAYS


def test_truth_flags_only_planned_late_or_slipped_orders():
    world = build_world(seed=11, scale=0.1)
    d = world.window_start + timedelta(days=30)
    late_or_slipped = {
        pl.component_key
        for pl in world.po_lines
        if pl.order_date <= d
        and (
            pl.delivery_date + timedelta(days=MIN_SHIP_DAYS)
            > next(c.req_date for c in world.components if c.key == pl.component_key)
        )
    } | {
        k
        for e in world.emails
        if e.kind == "slip" and e.sent <= d
        for pk in e.po_line_keys
        for k in [next(pl.component_key for pl in world.po_lines if pl.key == pk)]
    }
    for r in truth_recommendations(world, d):
        if r.action == "expedite":
            assert r.component in late_or_slipped


def test_earlier_detection_counts_days_before_the_baseline():
    need = date(2027, 4, 1)
    rec = Recommendation("c1", "w1", "m1", need, "order_now")
    days = {
        date(2027, 2, 1): ([], [rec]),
        date(2027, 2, 8): ([rec], [rec]),
    }
    out = earlier_detection(days)
    assert out["late_parts"] == out["detected"] == 1
    assert out["median_days_earlier"] == (need - timedelta(days=14) - date(2027, 2, 8)).days


@pytest.mark.parametrize("profile", ["A", "C"])
def test_replayed_run_scores_without_a_model(tmp_path, profile):
    export(build_world(seed=11, scale=0.05), tmp_path, profile)
    s = run_procurement(tmp_path, step_days=28, use_model=False)
    assert s["cost_usd"] == 0.0
    assert s["days"] >= 6
    for action in ("order_now", "order_soon"):
        t = s["totals"][action]
        assert t["precision"] is None or t["precision"] >= 0.8
    assert s["schedule"]["turnaround_moves_false"] == 0
    assert s["liaison"]["no_ack_precision"] is None or s["liaison"]["no_ack_precision"] >= 0.9
    assert (tmp_path / "reports" / "procurement" / "summary.json").exists()
    assert not (tmp_path / "reports" / "procurement" / "replay").exists()


def test_held_back_profile_is_refused(tmp_path):
    export(build_world(seed=11, scale=0.05), tmp_path, "B")
    with pytest.raises(SystemExit):
        run_procurement(tmp_path, use_model=False)
