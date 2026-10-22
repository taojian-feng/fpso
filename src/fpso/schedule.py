"""Schedule agent (Phase 1, autonomy level 1: recommend).

For each part the Procurement agent finds late, it works out what the slip does to the
schedule: the work order's P6 activity cannot start until the part is on board, and every
activity that follows it finish-to-start moves with it unless a gap absorbs the delay. The
answer that matters most is whether the turnaround end moves, and by how many days.

The work order to activity link comes from the same chain walk the other agents use. A
late part whose work order has no settled activity link is reported as unscheduled.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

from fpso.procurement import MIN_SHIP_DAYS, Recommendation
from fpso.readiness import ReadinessAgent
from fpso.resolution.normalize import parse_date
from fpso.synthetic.truth import World

TURNAROUND_WBS = ".TA."  # turnaround activities sit under the asset's TA WBS node


@dataclass
class Activity:
    key: str
    start: date
    finish: date
    successors: list[str] = field(default_factory=list)
    turnaround: bool = False


@dataclass
class Impact:
    component: str
    work_order: str
    slip_days: int  # how late the part lands against its need date
    activity: str | None = None
    activity_delay: int = 0
    moved: list[str] = field(default_factory=list)  # activities that move, the first included
    turnaround_delay: int = 0  # days the turnaround end moves


def part_slip(rec: Recommendation, as_of: date, lead_days: int | None) -> int:
    """Days the part lands after its need date if the recommendation is not acted on."""
    if rec.action == "expedite":
        return rec.days_late
    if rec.action == "order_now" and lead_days is not None:
        arrives = as_of + timedelta(days=lead_days + MIN_SHIP_DAYS)
        return max(0, (arrives - rec.need).days)
    return 0


def forward_pass(acts: dict[str, Activity], first: str, delay_days: int) -> dict[str, int]:
    """Days each activity moves when ``first`` cannot start until ``delay_days`` later.

    Finish-to-start: a successor starts no earlier than the day after its predecessor
    finishes, so the gap between them absorbs part of the delay. Where the plan already
    overlaps the two, the plan stands as it is and the gap counts as none."""
    moved = {first: delay_days}
    order = [first]
    while order:
        key = order.pop(0)
        a = acts[key]
        for s in a.successors:
            gap = max(0, (acts[s].start - a.finish).days - 1)
            push = moved[key] - gap
            if push > moved.get(s, 0):
                moved[s] = push
                order.append(s)
    return {k: d for k, d in moved.items() if d > 0}


def turnaround_delay(acts: dict[str, Activity], moved: dict[str, int]) -> int:
    ta = [a for a in acts.values() if a.turnaround]
    if not ta:
        return 0
    end = max(a.finish for a in ta)
    new_end = max(a.finish + timedelta(days=moved.get(a.key, 0)) for a in ta)
    return max(0, (new_end - end).days)


def impact_of(
    rec: Recommendation, slip: int, activity: str | None, acts: dict[str, Activity]
) -> Impact:
    imp = Impact(rec.component, rec.work_order, slip, activity)
    if activity is None or slip <= 0 or activity not in acts:
        return imp
    a = acts[activity]
    # The part is needed for the activity's start; it may already be needed earlier.
    imp.activity_delay = max(0, slip - max(0, (a.start - rec.need).days))
    if imp.activity_delay:
        moved = forward_pass(acts, activity, imp.activity_delay)
        imp.moved = sorted(moved, key=lambda k: (acts[k].start, k))
        imp.turnaround_delay = turnaround_delay(acts, moved)
    return imp


class ScheduleAgent:
    """Slip impacts from the P6 extract and the chain walk's work order links."""

    def __init__(self, chain: ReadinessAgent):
        self.chain = chain
        self.acts: dict[str, Activity] = {}
        for r in chain.c["p6"].iter("activity"):
            f = r.fields
            self.acts[r.key] = Activity(
                r.key,
                parse_date(f["target_start_date"][:10]),
                parse_date(f["target_end_date"][:10]),
                turnaround=TURNAROUND_WBS in f".{f.get('wbs_name', '')}.",
            )
        for r in chain.c["p6"].iter("activity"):
            for p in r.fields.get("predecessors", []):
                if p in self.acts:
                    self.acts[p].successors.append(r.key)

    def activity_of(self, work_orders: set[str]) -> dict[str, str]:
        """Settle work order -> activity links for these work orders (rules, then model)."""
        lt = "work_order_activity"
        self.chain.resolve(self.chain._leading_targets(lt, work_orders))
        out = {}
        for wo in sorted(work_orders):
            linked = self.chain.linked_from(lt, wo)
            if len(linked) == 1:
                out[wo] = linked[0]
        return out

    def run(self, recs: list[Recommendation], as_of: date, lead: dict) -> list[Impact]:
        late = [r for r in recs if r.action in ("order_now", "expedite")]
        act_of = self.activity_of({r.work_order for r in late})
        return [
            impact_of(
                r, part_slip(r, as_of, lead.get(r.material)), act_of.get(r.work_order), self.acts
            )
            for r in late
        ]


def truth_activities(world: World) -> tuple[dict[str, Activity], dict[str, str]]:
    acts = {
        a.act_id: Activity(a.act_id, a.start, a.finish, turnaround=TURNAROUND_WBS in f".{a.wbs}.")
        for a in world.activities
    }
    by_task = {a.task_id: a.act_id for a in world.activities}
    for a in world.activities:
        for p in a.predecessors:
            acts[by_task[p]].successors.append(a.act_id)
    act_of = {a.wo_no: a.act_id for a in world.activities if a.wo_no}
    return acts, act_of


def truth_impacts(world: World, recs: list[Recommendation], as_of: date) -> list[Impact]:
    acts, act_of = truth_activities(world)
    lead = {m.mat_no: m.lead_time_days for m in world.materials}
    return [
        impact_of(r, part_slip(r, as_of, lead.get(r.material)), act_of.get(r.work_order), acts)
        for r in recs
        if r.action in ("order_now", "expedite")
    ]


def score_impacts(agent: list[Impact], truth: list[Impact]) -> dict:
    """Agreement on the turnaround-end impact, over the late parts both sides report."""
    t = {i.component: i for i in truth}
    both = [(a, t[a.component]) for a in agent if a.component in t]
    right = [a for a, b in both if a.turnaround_delay == b.turnaround_delay]
    moves = defaultdict(int)
    for a, b in both:
        moves[(a.turnaround_delay > 0, b.turnaround_delay > 0)] += 1
    return {
        "late_parts": len(both),
        "unscheduled": sum(a.activity is None for a, _ in both),
        "truly_unscheduled": sum(b.activity is None for _, b in both),
        "turnaround_right": round(len(right) / len(both), 4) if both else None,
        "turnaround_moves_true": sum(b.turnaround_delay > 0 for _, b in both),
        "turnaround_moves_found": moves[(True, True)],
        "turnaround_moves_false": moves[(True, False)],
    }
