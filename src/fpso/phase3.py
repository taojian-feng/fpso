"""Phase 3 replay: the orchestrator acting within limits, driven by events.

Each replayed day the bus carries what every source gained since the day before. The
Procurement and Schedule agents run as in Phase 1; the Weather agent takes the boats and
flights the forecast will stop out of the network; the logistics plan is packed again; the
Inventory agent offers spares; the Condition agent reads the historian; the Campaign agent
groups small jobs. The orchestrator then settles every late part by cost of delay: cheap,
clearly worthwhile fixes run at once, everything else goes to the planner.

The same pipeline on the clean world (the weather as it turned out) gives the day's right
answers. The Phase 3 gate is measured:

- incidents: actions run without approval that were wrong or broke a rule card (target 0);
- expedite spend against today's practice, where a late part found at the baseline notice
  needs an emergency charter.

    fpso-orchestrate --run data/kestrel-A --no-model
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from fpso.campaign import campaigns
from fpso.condition import ConditionAgent, score_alerts
from fpso.connectors import open_all
from fpso.eval.reviewers import default_reviewers
from fpso.events import EventBus, snapshot, source_changes
from fpso.governance import ActionLog
from fpso.inventory import balances_from_connectors, find_stock
from fpso.liaison import email_kind
from fpso.logistics import Network, Planner, agent_items, agent_plan, score_shortfalls, truth_items
from fpso.models import CachedBackend, client_from_env
from fpso.orchestrator import check, day_rate, emergency_cost, settle
from fpso.phase2 import ensure_logistics
from fpso.procurement import ProcurementAgent, truth_recommendations
from fpso.readiness import load_world
from fpso.rule_cards import RuleSet, load_cards
from fpso.scenario import Scenario, what_if
from fpso.schedule import ScheduleAgent, truth_impacts
from fpso.synthetic.operations import build_operations, export_operations
from fpso.synthetic.replay import as_of_view
from fpso.weather import (
    outlook_from_connectors,
    outlook_from_truth,
    score_outlook,
    stopped,
    with_weather,
)
from fpso.writeback import ApprovalQueue, score_decisions


def ensure_operations(run_dir: Path, run: dict) -> None:
    if (run_dir / "extracts" / "weather").exists():
        return
    if run["profile"] == "B":
        raise SystemExit("Profile B is held back for the final run.")
    export_operations(load_world(run_dir), run_dir / "extracts")


def doubtful(agent: ProcurementAgent, as_of: date) -> set[str]:
    """Parts whose facts are incomplete: an open link, or a delay notice whose new date was
    not read (without a model, emails are classified but not read)."""
    out = set()
    mail = agent.connectors["supplier-mail"]
    for f in agent.facts:
        if f.unsettled:
            out.add(f.component)
            continue
        if not f.po_line:
            continue
        read = {m for _, m in f.slips}
        for key in agent.chain.linked_from("po_email", f.po_line.rsplit("-", 1)[0]):
            msg = mail.get("message", key).fields
            if (
                key not in read
                and date.fromisoformat(msg["date"]) <= as_of
                and email_kind(msg["subject"], msg["body"]) == "slip"
                and not (agent.chain.email_reads.get(key) or {}).get("new_date")
            ):
                out.add(f.component)
    return out


def run_orchestration(
    run_dir: str | Path,
    start: date | None = None,
    end: date | None = None,
    step_days: int = 7,
    max_cost_usd: float | None = 5.0,
    use_model: bool = True,
    rules: RuleSet | None = None,
) -> dict:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "truth" / "run.json").read_text())
    if run["profile"] == "B":
        raise SystemExit("Profile B is held back for the final run.")
    ensure_logistics(run_dir, run)
    ensure_operations(run_dir, run)
    rules = rules or load_cards()
    if bad := rules.self_test():
        raise SystemExit("rule cards fail their own tests: " + "; ".join(bad))
    window = [date.fromisoformat(d) for d in run["window"]]
    start, end = start or window[0], end or window[1]
    out_dir = run_dir / "reports" / "orchestrator"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = None
    if use_model:
        log = ActionLog(out_dir / "actions.jsonl")
        client = client_from_env(log=log)
        client.backend = CachedBackend(
            client.backend, run_dir / "reports" / "procurement" / "answers.jsonl"
        )

    full = open_all(run_dir / "extracts")
    net = Network.from_connectors(full)
    prices = {r.key: float(r.fields["PRICE"]) for r in full["erp-mm"].iter("material_price")}
    balances = balances_from_connectors(full)
    world = load_world(run_dir)
    ops = build_operations(world)
    truth_qty = {c.key: c.qty for c in world.components}
    wo_crit = {
        w.wo_no: e.criticality
        for w in world.work_orders
        for e in world.equipment
        if e.equip_no == w.equip_no
    }
    t_crit = {c.key: wo_crit[c.wo_no] for c in world.components}
    planner = next(
        r
        for r in default_reviewers(set(), log=ActionLog(out_dir / "planner.jsonl"))
        if r.role == "planner"
    )
    queue = ApprovalQueue(out_dir, planner)
    bus = EventBus(out_dir / "events.jsonl")
    heard: list[str] = []
    bus.subscribe("*", lambda ev: heard.append(ev.topic))

    prev: dict[str, set[str]] = {}
    per_day, alerts = [], {}
    incidents, auto_run = [], 0
    baseline_cost: dict[str, float] = {}  # true shortfall -> emergency charter today
    paid: dict[str, float] = {}  # true shortfall -> what the run paid to fix it
    d = start
    while d <= end:
        view = out_dir / "replay"
        as_of_view(run_dir, d, view)
        agent = ProcurementAgent(view / "extracts", client, max_cost_usd)
        now = snapshot(agent.connectors)
        for source, keys in source_changes(prev, now).items():
            bus.publish("source.new", source=source, count=len(keys), as_of=d)
        prev = now

        recs, _ = agent.run(d)
        impacts = ScheduleAgent(agent.chain).run(recs, d, agent.lead)
        qty = {f.component: f.qty for f in agent.facts}
        crit_of_wo = {
            r.key: agent.connectors["erp-pm"].get("equipment", r.fields["EQUNR"])
            for r in agent.connectors["erp-pm"].iter("work_order")
        }
        crit = {
            f.component: (e.fields.get("ABCKZ", "") if (e := crit_of_wo.get(f.work_order)) else "")
            for f in agent.facts
        }
        rate = {
            f.component: day_rate(
                f.component, crit, {i.component: i.turnaround_delay for i in impacts}
            )
            for f in agent.facts
        }
        outlook = outlook_from_connectors(agent.connectors, d)
        net_w = with_weather(net, outlook, rules)
        for sid in sorted(net_w.cancelled):
            bus.publish("weather.sailing_stopped", sailing=sid, as_of=d)
        items = agent_items(agent.facts, agent.chain, d)
        _, short = agent_plan(Planner(net_w, rules), items)
        for s in short:
            bus.publish("plan.shortfall", component=s.component, cause=s.cause, as_of=d)
        props = find_stock(recs, short, qty, balances, prices, rules, d)
        cond = ConditionAgent(agent.connectors).run(d, d - timedelta(days=step_days))
        alerts[d] = cond
        for a in cond:
            bus.publish("condition.alert", equipment=a.equipment, level=a.level, as_of=d)
        camp = campaigns(agent.connectors, d)
        sett = settle(
            d, items, short, props, recs, net_w, rules, rate, qty, prices, doubtful(agent, d)
        )
        nxt = next(
            (s for s in net_w.sailings if s.depart > d and s.sailing_id not in net_w.cancelled),
            None,
        )
        scen = (
            what_if(net_w, items, rules, Scenario("cancel_sailing", nxt.sailing_id), rate)
            if nxt
            else None
        )
        shutil.rmtree(view)

        # The clean world with the weather as it turned out.
        t_recs = truth_recommendations(world, d)
        t_imp = {i.component: i.turnaround_delay for i in truth_impacts(world, t_recs, d)}
        t_rate = {c: day_rate(c, t_crit, t_imp) for c in t_crit}
        t_out = outlook_from_truth(ops, d)
        t_net = with_weather(net, t_out, rules)
        t_items = truth_items(world, d)
        _, t_short = Planner(t_net, rules).plan(t_items)
        t_props = find_stock(t_recs, t_short, truth_qty, balances, prices, rules, d)
        t_sett = settle(
            d, t_items, t_short, t_props, t_recs, t_net, rules, t_rate, truth_qty, prices
        )
        t_keys = {a.key for a in t_sett.actions()}
        for c in t_sett.choices:
            if (
                c.action
                and c.action.kind != "create_purchase_requisition"
                and c.cost_of_delay_usd > 0
            ):
                sf = next(s for s in t_short if s.component == c.component)
                baseline_cost.setdefault(c.component, emergency_cost(sf, rules))

        # Act: within limits at once, the rest to the planner.
        open_actions = {a.key for a in queue.open(sett.actions())}
        to_review = []
        for c in sett.choices:
            if not c.action or c.action.key not in open_actions:
                continue
            if c.auto:
                auto_run += 1
                broken = check(c.action, rules, auto=True)
                wrong = c.action.key not in t_keys
                if broken or wrong:
                    incidents.append(
                        {"action": c.action.action_id, "rule_breaks": broken, "wrong": wrong}
                    )
                queue.execute(c.action)
                bus.publish("action.executed", id=c.action.action_id, auto=True)
                if not wrong:
                    paid.setdefault(c.component, c.action.cost_usd)
            else:
                to_review.append(c.action)
        decisions = queue.review(to_review, t_keys)
        cost_of = {a.action_id: a for a in to_review}
        for dec in decisions:
            bus.publish("action.reviewed", id=dec.action, verdict=dec.verdict)
            if dec.verdict == "approved" and dec.correct:
                a = cost_of[dec.action]
                paid.setdefault(a.component, a.cost_usd)

        day = {
            "as_of": d.isoformat(),
            "weather": score_outlook(
                stopped(net, outlook, rules), (t_net.cancelled, t_net.grounded)
            ),
            "shortfalls": score_shortfalls(short, t_short),
            "settled": {
                "late_parts": sum(1 for c in sett.choices if c.cost_of_delay_usd > 0),
                "auto": sum(c.auto for c in sett.choices if c.action),
                "to_planner": len(to_review),
                "cost_of_delay_at_risk_usd": round(
                    sum(c.cost_of_delay_usd for c in sett.choices), 2
                ),
                "saved_usd": round(sum(c.saved_usd for c in sett.choices if c.action), 2),
            },
            "approvals": score_decisions(decisions),
            "condition_alerts": len(cond),
            "campaigns": len(camp),
            "scenario_next_sailing_cancelled": asdict(scen) if scen else None,
        }
        per_day.append(day)
        (out_dir / f"{d.isoformat()}.json").write_text(
            json.dumps(
                {
                    **day,
                    "choices": [asdict(c) for c in sett.choices],
                    "alerts": [asdict(a) for a in cond],
                    "campaigns_list": [asdict(c) for c in camp],
                },
                default=str,
                indent=2,
            )
        )
        d += timedelta(days=step_days)

    base = sum(baseline_cost.values())
    spent = sum(paid.get(c, cost) for c, cost in baseline_cost.items())
    summary = {
        "run": {k: run[k] for k in ("asset", "profile", "seed", "window")},
        "days": len(per_day),
        "cost_usd": round(client.log.total_cost_usd, 4) if client else 0.0,
        "incidents": {"auto_run": auto_run, "count": len(incidents), "list": incidents},
        "expedite_spend": {
            "late_parts": len(baseline_cost),
            "baseline_usd": round(base, 2),
            "with_agents_usd": round(spent, 2),
            "down_share": round(1 - spent / base, 4) if base else None,
        },
        "weather": _weather_totals([x["weather"] for x in per_day]),
        "condition": score_alerts(alerts, ops.failing),
        "events": dict(sorted(bus.counts.items())),
        "per_day": per_day,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _weather_totals(days: list[dict]) -> dict:
    out = {}
    for k in ("sailings", "flights"):
        s = [x[k] for x in days]
        pred = sum(x["predicted"] for x in s)
        true = sum(x["true"] for x in s)
        hit = sum(round((x["precision"] or 0) * x["predicted"]) for x in s)
        out[k] = {
            "predicted": pred,
            "true": true,
            "precision": round(hit / pred, 4) if pred else None,
            "recall": round(hit / true, 4) if true else None,
        }
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Phase 3 orchestration, replayed day by day.")
    ap.add_argument("--run", required=True, help="a run directory written by fpso-synth")
    ap.add_argument("--from", dest="start", type=date.fromisoformat, help="first day")
    ap.add_argument("--to", dest="end", type=date.fromisoformat, help="last day")
    ap.add_argument("--step", type=int, default=7, help="days between replayed days")
    ap.add_argument("--max-cost", type=float, default=5.0, help="stop new calls above this USD")
    ap.add_argument("--no-model", action="store_true", help="rule links only, no email reading")
    args = ap.parse_args(argv)
    s = run_orchestration(
        args.run, args.start, args.end, args.step, args.max_cost, use_model=not args.no_model
    )
    print(json.dumps({k: v for k, v in s.items() if k != "per_day"}, indent=2, default=str))


if __name__ == "__main__":
    main()
