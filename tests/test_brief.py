from datetime import date

from fpso.brief import build_brief, review_brief, save_corrections, score_reviews
from fpso.eval.reviewers import SimulatedReviewer
from fpso.liaison import Chase
from fpso.procurement import Recommendation
from fpso.schedule import Impact

D = date(2027, 3, 1)


def _recs():
    return [
        Recommendation("c-soon", "w1", "m1", date(2027, 5, 1), "order_soon", order_by=D),
        Recommendation("c-late", "w2", "m2", date(2027, 4, 1), "expedite", po_line="4500-00010"),
        Recommendation("c-now", "w3", "m3", date(2027, 3, 20), "order_now", order_by=D),
    ]


def test_turnaround_impact_ranks_first_then_urgency():
    impacts = [Impact("c-late", "w2", 5, "A1", 5, ["A1"], 3)]
    chases = [Chase("4500", "v1", "no_ack", "orders@v.example")]
    items = build_brief(_recs(), impacts, chases)
    assert [x.subject for x in items] == ["c-late", "c-now", "c-soon", "4500"]
    assert "turnaround end moves 3 days" in items[0].text
    assert build_brief(_recs(), impacts, chases, suppressed={"order_now|c-now"})[1].subject == (
        "c-soon"
    )


def test_planner_accepts_right_items_and_edits_or_rejects_the_rest(tmp_path):
    items = build_brief(_recs(), [Impact("c-late", "w2", 5, "A1", 5, ["A1"], 3)], [])
    truth = [
        Recommendation("c-late", "w2", "m2", date(2027, 4, 1), "expedite"),
        Recommendation("c-now", "w3", "m3", date(2027, 3, 20), "order_soon"),
    ]
    planner = SimulatedReviewer("dana.reyes", "planner", set(), error_rate=0, skip_rate=0)
    reviews = review_brief(items, truth, {"c-late": 2}, set(), planner, D)
    verdict = {r.item: r for r in reviews}
    assert verdict["expedite|c-late"].verdict == "edited"  # turnaround moves 2 days, not 3
    assert verdict["expedite|c-late"].correct_turnaround_days == 2
    assert verdict["order_now|c-now"].verdict == "edited"
    assert verdict["order_now|c-now"].correct_kind == "order_soon"
    assert verdict["order_soon|c-soon"].verdict == "rejected"
    path = tmp_path / "corrections.jsonl"
    save_corrections(path, D, items, reviews)
    assert len(path.read_text().splitlines()) == 3
    assert score_reviews(reviews)["accepted_unchanged"] == 0.0
