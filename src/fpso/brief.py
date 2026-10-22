"""Daily planner brief and correction loop (Phase 1, P4 and P5).

The brief merges what the three agents recommend that day into one ranked list: parts
that move the turnaround end first (most days first), then orders to place now, expedites,
orders due soon, and unacknowledged POs. Each item cites its source records.

The simulated planner reviews every item: it accepts it, edits it (right part, wrong action
or wrong turnaround impact) or rejects it. Edits and rejections are kept as correction
cases in ``corrections.jsonl``; an item the planner rejected is not raised again on later
days, so the planner is not asked the same wrong thing twice.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from fpso.eval.reviewers import SimulatedReviewer
from fpso.liaison import Chase
from fpso.procurement import Recommendation
from fpso.schedule import Impact

URGENCY = {"order_now": 0, "expedite": 1, "order_soon": 2, "no_ack": 3}


@dataclass
class BriefItem:
    kind: str  # order_now, expedite, order_soon, no_ack
    subject: str  # the component, or the PO for a chase
    work_order: str = ""
    need: date | None = None
    turnaround_days: int = 0
    text: str = ""
    sources: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.kind}|{self.subject}"


@dataclass
class Review:
    item: str
    verdict: str  # accepted, edited, rejected, skipped
    correct_kind: str | None = None
    correct_turnaround_days: int | None = None


def build_brief(
    recs: list[Recommendation],
    impacts: list[Impact],
    chases: list[Chase],
    suppressed: set[str] = frozenset(),
) -> list[BriefItem]:
    impact = {i.component: i for i in impacts}
    items = []
    for r in recs:
        i = impact.get(r.component)
        days = i.turnaround_delay if i else 0
        when = f"order by {r.order_by}" if r.order_by else f"expected {r.expected}"
        effect = f"; turnaround end moves {days} day{'s' * (days != 1)}" if days else ""
        items.append(
            BriefItem(
                r.action,
                r.component,
                r.work_order,
                r.need,
                days,
                f"{r.action.replace('_', ' ')}: material {r.material} for work order "
                f"{r.work_order}, needed {r.need}, {when}{effect}",
                {"po_line": r.po_line, "email": r.evidence or None, "activity": i and i.activity},
            )
        )
    for c in chases:
        if c.kind == "no_ack":
            items.append(
                BriefItem(
                    "no_ack",
                    c.po,
                    text=f"PO {c.po} not acknowledged by the supplier; chase drafted to {c.to}",
                    sources={"po_lines": c.po_lines},
                )
            )
    items = [x for x in items if x.key not in suppressed]
    return sorted(
        items,
        key=lambda x: (-x.turnaround_days, URGENCY[x.kind], x.need or date.max, x.subject),
    )


def review_brief(
    items: list[BriefItem],
    truth_recs: list[Recommendation],
    truth_turnaround: dict[str, int],
    truth_no_ack: set[str],
    planner: SimulatedReviewer,
    as_of: date,
) -> list[Review]:
    """The simulated planner's verdict on each item, checked against the clean world."""
    true_action = {r.component: r.action for r in truth_recs}
    out = []
    for x in items:
        if x.kind == "no_ack":
            right_kind, right_days = (x.subject in truth_no_ack), None
            correct_kind = None
        else:
            correct_kind = true_action.get(x.subject)
            right_kind = correct_kind == x.kind
            right_days = truth_turnaround.get(x.subject, 0)
        exists = right_kind or (x.kind != "no_ack" and correct_kind is not None)
        answer = planner.confirm_gap(f"{as_of}|{x.key}", exists)
        if answer is None:
            out.append(Review(x.key, "skipped"))
        elif not answer:
            out.append(Review(x.key, "rejected"))
        elif right_kind and (right_days is None or right_days == x.turnaround_days):
            out.append(Review(x.key, "accepted"))
        else:
            out.append(Review(x.key, "edited", correct_kind, right_days))
    return out


def save_corrections(path: Path, as_of: date, items: list[BriefItem], reviews: list[Review]):
    by_key = {x.key: x for x in items}
    with path.open("a") as f:
        for r in reviews:
            if r.verdict in ("edited", "rejected"):
                case = {"as_of": as_of.isoformat(), **asdict(r), "item": asdict(by_key[r.item])}
                f.write(json.dumps(case, default=str) + "\n")


def score_reviews(reviews: list[Review]) -> dict:
    counts = {v: sum(r.verdict == v for r in reviews) for v in ("accepted", "edited", "rejected")}
    answered = sum(counts.values())
    return {
        "items": len(reviews),
        **counts,
        "skipped": len(reviews) - answered,
        "accepted_unchanged": round(counts["accepted"] / answered, 4) if answered else None,
    }


def to_markdown(run: dict, as_of: date, items: list[BriefItem], reviews: list[Review]) -> str:
    verdict = {r.item: r.verdict for r in reviews}
    lines = [
        f"# Planner brief, {as_of.isoformat()}: {run['asset']}",
        "",
        f"{len(items)} items, most turnaround days at risk first. Recommendations only: "
        "nothing has been ordered, changed or sent.",
        "",
        "| # | Item | Turnaround days | Planner |",
        "| ---: | --- | ---: | --- |",
    ]
    for n, x in enumerate(items, 1):
        lines.append(f"| {n} | {x.text} | {x.turnaround_days or '-'} | {verdict.get(x.key, '-')} |")
    return "\n".join(lines) + "\n"
