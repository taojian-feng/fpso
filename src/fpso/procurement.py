"""Lead-time & Procurement agent (Phase 1, autonomy level 1: recommend).

Works backward from each component's need date. A part ordered today arrives one lead time
later and needs ``MIN_SHIP_DAYS`` to reach the next sailing, so its latest order date is::

    order_by = need - MIN_SHIP_DAYS - lead time

For every component of a work order starting up to ``HORIZON_WEEKS`` out, it recommends:

- ``order_now``: no PO line yet, stock does not cover it, and ``order_by`` has passed;
- ``order_soon``: no PO line yet and ``order_by`` falls within ``WARN_DAYS``;
- ``expedite``: the PO line's delivery date, or a later date a supplier email gave, lands
  less than ``MIN_SHIP_DAYS`` before the need.

Components whose PO line link is still open are counted, not recommended on. Links and
email reading come from the Readiness agent's chain walk, so the two agents agree on facts.

The agent runs once per day on the run as it stood that day (``fpso.synthetic.replay``), so
it never sees a PO before it was placed. Scoring compares each day's recommendations with
the same rules applied to the clean world, and measures how much earlier than a planner
would today a late part is flagged. Today's baseline is that a planner notices when the
goods have not arrived ``BASELINE_NOTICE_DAYS`` before the need.

    fpso-procure --run data/kestrel-A --no-model
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

from fpso.connectors import open_all
from fpso.governance import ActionLog
from fpso.models import CachedBackend, ModelClient, client_from_env
from fpso.readiness import Facts, ReadinessAgent, in_horizon, load_world
from fpso.resolution.normalize import leading_int
from fpso.synthetic.replay import as_of_view
from fpso.synthetic.truth import World

ACTIONS = ("order_now", "order_soon", "expedite")
HORIZON_WEEKS = (0, 34)  # the longest lead time (about 30 weeks) plus shipping and warning
MIN_SHIP_DAYS = 4  # delivery to the shore base, then the next sailing
WARN_DAYS = 14
BASELINE_NOTICE_DAYS = 14


@dataclass
class Recommendation:
    component: str
    work_order: str
    material: str
    need: date
    action: str
    order_by: date | None = None
    days_late: int = 0  # past order_by, or how far the expected date overshoots
    po_line: str | None = None
    expected: date | None = None  # delivery date the expedite rests on
    evidence: str = ""  # email the expected date came from, if any


def recommend(f: Facts, lead_days: int | None, as_of: date) -> Recommendation | None:
    """The procurement action for one component, or None when nothing is needed."""
    rec = Recommendation(f.component, f.work_order, f.material, f.need, "")
    if f.po_line is not None:
        expected, evidence = f.delivery, ""
        for d, msg in f.slips:
            if expected is None or d > expected:
                expected, evidence = d, msg
        if expected is None or expected + timedelta(days=MIN_SHIP_DAYS) <= f.need:
            return None
        rec.action, rec.po_line, rec.expected, rec.evidence = (
            "expedite",
            f.po_line,
            expected,
            evidence,
        )
        rec.days_late = (expected + timedelta(days=MIN_SHIP_DAYS) - f.need).days
        return rec
    if f.stock >= f.qty or lead_days is None:
        return None
    rec.order_by = f.need - timedelta(days=MIN_SHIP_DAYS + lead_days)
    if rec.order_by < as_of:
        rec.action, rec.days_late = "order_now", (as_of - rec.order_by).days
    elif rec.order_by <= as_of + timedelta(days=WARN_DAYS):
        rec.action = "order_soon"
    else:
        return None
    return rec


class ProcurementAgent:
    """Recommendations for one day, from the extracts as they stood that day."""

    def __init__(self, extracts: Path, client: ModelClient | None, max_cost_usd: float | None):
        self.connectors = open_all(extracts)
        self.chain = ReadinessAgent(self.connectors, client, max_cost_usd)
        self.chain.horizon = HORIZON_WEEKS
        self.lead = {
            r.key: leading_int(r.fields.get("PLIFZ", ""))
            for r in self.connectors["erp-mm"].iter("material")
        }

    def run(self, as_of: date) -> tuple[list[Recommendation], int]:
        """Recommendations, and how many components wait on an open supply link."""
        recs, open_supply = [], 0
        self.facts = self.chain.facts(as_of)  # kept for the agents that build on this one
        for f in self.facts:
            if f.po_line is None and "supply" in f.unsettled:
                open_supply += 1
                continue
            rec = recommend(f, self.lead.get(f.material), as_of)
            if rec:
                recs.append(rec)
        return recs, open_supply


def truth_recommendations(world: World, as_of: date) -> list[Recommendation]:
    """The same rules on the clean world, with only what had happened by ``as_of``."""
    wos = {w.wo_no for w in world.work_orders if in_horizon(w.planned_start, as_of, HORIZON_WEEKS)}
    pl_by_comp = {pl.component_key: pl for pl in world.po_lines}
    lead = {m.mat_no: m.lead_time_days for m in world.materials}
    slips = defaultdict(list)
    for e in world.emails:
        if e.kind == "slip" and e.sent <= as_of:
            for k in e.po_line_keys:
                slips[k].append((e.new_date, e.message_id))
    out = []
    for c in world.components:
        if c.wo_no not in wos:
            continue
        pl = pl_by_comp.get(c.key)
        f = Facts(c.key, c.wo_no, c.mat_no, c.qty, c.req_date, world.stock.get(c.mat_no, 0))
        if pl and pl.order_date <= as_of:
            f.po_line, f.delivery, f.slips = pl.key, pl.delivery_date, slips[pl.key]
        rec = recommend(f, lead[c.mat_no], as_of)
        if rec:
            out.append(rec)
    return out


def score_day(agent: list[Recommendation], truth: list[Recommendation]) -> dict:
    a = {(r.component, r.action) for r in agent}
    t = {(r.component, r.action) for r in truth}
    out = {}
    for action in ACTIONS:
        aa = {x for x in a if x[1] == action}
        tt = {x for x in t if x[1] == action}
        out[action] = {
            "reported": len(aa),
            "true": len(tt),
            "precision": round(len(aa & tt) / len(aa), 4) if aa else None,
            "recall": round(len(aa & tt) / len(tt), 4) if tt else None,
        }
    return out


def earlier_detection(days: dict[date, tuple[list, list]]) -> dict:
    """Days between the agent's first correct warning and today's baseline, per late part.

    A part is late when the truth ever says order_now or expedite for it. The agent detects
    it on the first day it correctly gives it any action that truth gives it that day."""
    late: dict[str, date] = {}
    first: dict[str, date] = {}
    for d in sorted(days):
        agent, truth = days[d]
        t = {(r.component, r.action) for r in truth}
        for r in truth:
            if r.action in ("order_now", "expedite"):
                late.setdefault(r.component, r.need)
        for r in agent:
            if (r.component, r.action) in t:
                first.setdefault(r.component, d)
    gains = [
        (need - timedelta(days=BASELINE_NOTICE_DAYS) - first[c]).days
        for c, need in late.items()
        if c in first
    ]
    return {
        "late_parts": len(late),
        "detected": len(gains),
        "median_days_earlier": statistics.median(gains) if gains else None,
        "share_before_baseline": round(sum(g > 0 for g in gains) / len(gains), 4)
        if gains
        else None,
    }


def to_markdown(run: dict, as_of: date, recs: list[Recommendation], scores: dict, open_n: int):
    lines = [
        f"# Procurement recommendations, {as_of.isoformat()}: {run['asset']}, "
        f"profile {run['profile']}",
        "",
        f"{len(recs)} recommendations for work orders starting up to {HORIZON_WEEKS[1]} weeks "
        f"out; {open_n} components wait on an open PO line link. Recommend only: nothing is "
        "ordered or changed.",
        "",
        "| Action | Reported | True | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for action, s in scores.items():
        p = "n/a" if s["precision"] is None else f"{s['precision']:.0%}"
        r = "n/a" if s["recall"] is None else f"{s['recall']:.0%}"
        lines.append(f"| {action} | {s['reported']} | {s['true']} | {p} | {r} |")
    lines += [
        "",
        "| Action | Component | Work order | Need | Order by | Expected | Days late | PO line |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    order = {a: i for i, a in enumerate(ACTIONS)}
    for r in sorted(recs, key=lambda r: (order[r.action], r.need, r.component)):
        lines.append(
            f"| {r.action} | {r.component} | {r.work_order} | {r.need} | {r.order_by or '-'} "
            f"| {r.expected or '-'} | {r.days_late} | {r.po_line or '-'} |"
        )
    return "\n".join(lines) + "\n"


def run_procurement(
    run_dir: str | Path,
    start: date | None = None,
    end: date | None = None,
    step_days: int = 7,
    max_cost_usd: float | None = 5.0,
    use_model: bool = True,
    client: ModelClient | None = None,
) -> dict:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "truth" / "run.json").read_text())
    if run["profile"] == "B":
        raise SystemExit("Profile B is held back for the final run.")
    window = [date.fromisoformat(d) for d in run["window"]]
    start, end = start or window[0], end or window[1]
    out_dir = run_dir / "reports" / "procurement"
    out_dir.mkdir(parents=True, exist_ok=True)
    if use_model:
        log = ActionLog(out_dir / "actions.jsonl")
        client = client or client_from_env(log=log)
        client.log = log
        client.backend = CachedBackend(client.backend, out_dir / "answers.jsonl")
    else:
        client = None
    # The other Phase 1 agents and the brief build on this module's recommendations.
    from fpso import brief
    from fpso.eval.reviewers import default_reviewers
    from fpso.liaison import LiaisonAgent, score_chases, truth_unacknowledged
    from fpso.schedule import ScheduleAgent, score_impacts, truth_impacts

    planner = next(
        r
        for r in default_reviewers(set(), log=ActionLog(out_dir / "planner.jsonl"))
        if r.role == "planner"
    )
    corrections = out_dir / "corrections.jsonl"
    corrections.unlink(missing_ok=True)
    suppressed: set[str] = set()  # items the planner rejected stay out of later briefs

    world = load_world(run_dir)  # scoring only
    days: dict[date, tuple[list, list]] = {}
    impact_scores, chase_scores, review_scores = [], [], []
    per_day = []
    d = start
    while d <= end:
        view = out_dir / "replay"
        as_of_view(run_dir, d, view)
        agent = ProcurementAgent(view / "extracts", client, max_cost_usd)
        recs, open_n = agent.run(d)
        impacts = ScheduleAgent(agent.chain).run(recs, d, agent.lead)
        chases = LiaisonAgent(agent.chain).run(d, recs, impacts)
        shutil.rmtree(view)
        truth = truth_recommendations(world, d)
        scores = score_day(recs, truth)
        t_impacts = truth_impacts(world, truth, d)
        t_no_ack = truth_unacknowledged(world, d)
        impact_scores.append(score_impacts(impacts, t_impacts))
        chase_scores.append(score_chases(chases, t_no_ack))
        items = brief.build_brief(recs, impacts, chases, suppressed)
        reviews = brief.review_brief(
            items,
            truth,
            {i.component: i.turnaround_delay for i in t_impacts},
            set(t_no_ack),
            planner,
            d,
        )
        brief.save_corrections(corrections, d, items, reviews)
        suppressed |= {r.item for r in reviews if r.verdict == "rejected"}
        review_scores.append(brief.score_reviews(reviews))
        (out_dir / f"brief-{d.isoformat()}.md").write_text(
            brief.to_markdown(run, d, items, reviews)
        )
        days[d] = (recs, truth)
        (out_dir / f"{d.isoformat()}.md").write_text(to_markdown(run, d, recs, scores, open_n))
        (out_dir / f"{d.isoformat()}.json").write_text(
            json.dumps(
                {
                    "as_of": d,
                    "open_supply": open_n,
                    "scores": scores,
                    "recommendations": [asdict(r) for r in recs],
                    "schedule": {"scores": impact_scores[-1], "impacts": map(asdict, impacts)},
                    "liaison": {"scores": chase_scores[-1], "drafts": map(asdict, chases)},
                    "brief": {"scores": review_scores[-1], "items": map(asdict, items)},
                },
                default=lambda o: list(o) if isinstance(o, map) else str(o),
                indent=2,
            )
        )
        per_day.append(
            {
                "as_of": d.isoformat(),
                "open_supply": open_n,
                **scores,
                "schedule": impact_scores[-1],
                "liaison": chase_scores[-1],
                "brief": review_scores[-1],
            }
        )
        d += timedelta(days=step_days)

    totals = {}
    for action in ACTIONS:
        rep = sum(x[action]["reported"] for x in per_day)
        tru = sum(x[action]["true"] for x in per_day)
        hit = sum(
            len(
                {r.component for r in a if r.action == action}
                & {r.component for r in t if r.action == action}
            )
            for a, t in days.values()
        )
        totals[action] = {
            "reported": rep,
            "true": tru,
            "precision": round(hit / rep, 4) if rep else None,
            "recall": round(hit / tru, 4) if tru else None,
        }
    summary = {
        "run": {k: run[k] for k in ("asset", "profile", "seed", "window")},
        "days": len(per_day),
        "step_days": step_days,
        "cost_usd": round(client.log.total_cost_usd, 4) if client else 0.0,
        "totals": totals,
        "earlier_detection": earlier_detection(days),
        "schedule": _schedule_totals(impact_scores),
        "liaison": _liaison_totals(chase_scores),
        "brief": _brief_totals(review_scores, len(suppressed)),
        "per_day": per_day,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _schedule_totals(scores: list[dict]) -> dict:
    both = sum(s["late_parts"] for s in scores)
    right = sum(round((s["turnaround_right"] or 0) * s["late_parts"]) for s in scores)
    return {
        "late_parts": both,
        "turnaround_right": round(right / both, 4) if both else None,
        **{
            k: sum(s[k] for s in scores)
            for k in (
                "unscheduled",
                "truly_unscheduled",
                "turnaround_moves_true",
                "turnaround_moves_found",
                "turnaround_moves_false",
            )
        },
    }


def _liaison_totals(scores: list[dict]) -> dict:
    rep = sum(s["no_ack_reported"] for s in scores)
    tru = sum(s["no_ack_true"] for s in scores)
    hit_p = sum(round((s["no_ack_precision"] or 0) * s["no_ack_reported"]) for s in scores)
    return {
        "no_ack_reported": rep,
        "no_ack_true": tru,
        "no_ack_precision": round(hit_p / rep, 4) if rep else None,
        "no_ack_recall": round(hit_p / tru, 4) if tru else None,
        "expedite_drafts": sum(s["expedite_drafts"] for s in scores),
    }


def _brief_totals(scores: list[dict], suppressed: int) -> dict:
    out = {k: sum(s[k] for s in scores) for k in ("items", "accepted", "edited", "rejected")}
    answered = out["accepted"] + out["edited"] + out["rejected"]
    out["accepted_unchanged"] = round(out["accepted"] / answered, 4) if answered else None
    out["suppressed_after_rejection"] = suppressed
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Procurement recommendations, replayed day by day.")
    ap.add_argument("--run", required=True, help="a run directory written by fpso-synth")
    ap.add_argument("--from", dest="start", type=date.fromisoformat, help="first day")
    ap.add_argument("--to", dest="end", type=date.fromisoformat, help="last day")
    ap.add_argument("--step", type=int, default=7, help="days between replayed days")
    ap.add_argument("--max-cost", type=float, default=5.0, help="stop new calls above this USD")
    ap.add_argument("--no-model", action="store_true", help="rule links only, no email reading")
    args = ap.parse_args(argv)
    s = run_procurement(
        args.run, args.start, args.end, args.step, args.max_cost, use_model=not args.no_model
    )
    print(
        json.dumps(
            {
                k: s[k]
                for k in (
                    "days",
                    "cost_usd",
                    "totals",
                    "earlier_detection",
                    "schedule",
                    "liaison",
                    "brief",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
