"""Phase 2 replay: logistics and inventory agents, with write-back on approval.

Each replayed day runs the Phase 1 Procurement agent, then packs the parts on order onto
sailings (``fpso.logistics``), looks for spare stock before buying or flying anything
(``fpso.inventory``), turns what is left into actions, and puts every action to the planner.
Approved actions land in the mock outbox. The same pipeline on the clean world gives the
day's right answers, so each piece is scored on its own and the Phase 2 gate is measured:

- logistics shortfalls found earlier than a planner finds them today (the last useful
  manifest issued without the part);
- the share of actions the planner approves (target at least 70%).

    fpso-logistics --run data/kestrel-A --no-model
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from fpso.connectors import open_all
from fpso.eval.reviewers import default_reviewers
from fpso.governance import ActionLog
from fpso.inventory import StockProposal, balances_from_connectors, find_stock, score_proposals
from fpso.logistics import (
    CAUSES,
    Network,
    Planner,
    Shortfall,
    agent_items,
    agent_plan,
    earlier_shortfalls,
    score_shortfalls,
    truth_items,
)
from fpso.models import CachedBackend, client_from_env
from fpso.procurement import ProcurementAgent, Recommendation, truth_recommendations
from fpso.readiness import load_world
from fpso.rule_cards import RuleSet, load_cards
from fpso.synthetic.logistics import export_logistics
from fpso.synthetic.replay import as_of_view
from fpso.writeback import (
    TRANSFER_USD,
    Action,
    ApprovalQueue,
    heli_cost,
    score_decisions,
    slot_cost,
)


def ensure_logistics(run_dir: Path, run: dict, final: bool = False) -> None:
    """Add the Phase 2 extracts to a run written before they existed (B only when final)."""
    if (run_dir / "extracts" / "helicopter").exists():
        return
    if run["profile"] == "B" and not final:
        raise SystemExit("Profile B is held back for the final run; pass --final to use it.")
    export_logistics(load_world(run_dir), run_dir / "extracts")


def actions_for(
    as_of: date,
    recs: list[Recommendation],
    shortfalls: list[Shortfall],
    proposals: list[StockProposal],
    qty: dict[str, int],
    prices: dict[str, float],
) -> list[Action]:
    """The day's actions: spare stock first, then buy what is left, then fix shortfalls."""
    out = []
    covered = set()
    for p in proposals:
        covered.add(p.component)
        out.append(
            Action(
                "transfer_stock",
                p.component,
                as_of,
                {"store": p.store, "material": p.material, "qty": p.qty, "move": p.action},
                f"spare stock in {p.store} covers {p.replaces}; "
                f"avoids buying {p.value_usd:,.0f} USD",
                TRANSFER_USD[p.action],
                "inventory",
            )
        )
    for r in recs:
        if r.action == "order_now" and r.component not in covered:
            n = qty.get(r.component, 1)
            out.append(
                Action(
                    "create_purchase_requisition",
                    r.component,
                    as_of,
                    {"material": r.material, "qty": n, "need": r.need, "work_order": r.work_order},
                    f"latest order date {r.order_by} has passed",
                    round(prices.get(r.material, 0.0) * n, 2),
                    "procurement",
                )
            )
    for s in shortfalls:
        if s.component in covered or s.option is None:
            continue
        kind = "book_helicopter_cargo" if s.option == "helicopter" else "request_vessel_slot"
        cost = heli_cost(s.weight_kg) if s.option == "helicopter" else slot_cost(s.area_m2)
        out.append(
            Action(
                kind,
                s.component,
                as_of,
                {
                    "ref": s.option_ref,
                    "on_board": s.option_on_board,
                    "need": s.need,
                    "weight_kg": s.weight_kg,
                    "area_m2": s.area_m2,
                },
                f"{s.cause} shortfall: on board {s.on_board or 'not in plan'}, needed {s.need}",
                cost,
                "logistics",
            )
        )
    return out


def to_markdown(run: dict, as_of: date, shortfalls, actions, decisions) -> str:
    verdict = {d.action: d.verdict for d in decisions}
    lines = [
        f"# Logistics, {as_of.isoformat()}: {run['asset']}, profile {run['profile']}",
        "",
        f"{len(shortfalls)} parts land after their need date as planned; {len(actions)} actions "
        "proposed. Only actions the planner approves are written to the outbox.",
        "",
        "| Component | Need | On board | Cause | Fix |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in sorted(shortfalls, key=lambda s: (s.need, s.component)):
        fix = f"{s.option} {s.option_ref} ({s.option_on_board})" if s.option else "-"
        lines.append(f"| {s.component} | {s.need} | {s.on_board or '-'} | {s.cause} | {fix} |")
    lines += [
        "",
        "| Action | Component | Cost USD | Reason | Planner |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for a in actions:
        lines.append(
            f"| {a.kind} | {a.component} | {a.cost_usd:,.0f} | {a.reason} | "
            f"{verdict.get(a.action_id, 'already done')} |"
        )
    return "\n".join(lines) + "\n"


def run_logistics(
    run_dir: str | Path,
    start: date | None = None,
    end: date | None = None,
    step_days: int = 7,
    max_cost_usd: float | None = 5.0,
    use_model: bool = True,
    rules: RuleSet | None = None,
    final: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "truth" / "run.json").read_text())
    if run["profile"] == "B" and not final:
        raise SystemExit("Profile B is held back for the final run; pass --final to use it.")
    ensure_logistics(run_dir, run, final)
    rules = rules or load_cards()
    if bad := rules.self_test():
        raise SystemExit("rule cards fail their own tests: " + "; ".join(bad))
    window = [date.fromisoformat(d) for d in run["window"]]
    start, end = start or window[0], end or window[1]
    out_dir = run_dir / "reports" / "logistics"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = None
    if use_model:
        log = ActionLog(out_dir / "actions.jsonl")
        client = client_from_env(log=log)
        # Shares the Phase 1 answers, so questions already asked there cost nothing here.
        client.backend = CachedBackend(
            client.backend, run_dir / "reports" / "procurement" / "answers.jsonl"
        )

    full = open_all(run_dir / "extracts")
    net = Network.from_connectors(full)
    planner_net = Planner(net, rules)
    prices = {r.key: float(r.fields["PRICE"]) for r in full["erp-mm"].iter("material_price")}
    balances = balances_from_connectors(full)
    world = load_world(run_dir)
    truth_qty = {c.key: c.qty for c in world.components}
    planner = next(
        r
        for r in default_reviewers(set(), log=ActionLog(out_dir / "planner.jsonl"))
        if r.role == "planner"
    )
    queue = ApprovalQueue(out_dir, planner)
    days, per_day, all_decisions = {}, [], []
    d = start
    while d <= end:
        view = out_dir / "replay"
        as_of_view(run_dir, d, view)
        agent = ProcurementAgent(view / "extracts", client, max_cost_usd)
        recs, _ = agent.run(d)
        qty = {f.component: f.qty for f in agent.facts}
        _, short = agent_plan(planner_net, agent_items(agent.facts, agent.chain, d))
        shutil.rmtree(view)
        props = find_stock(recs, short, qty, balances, prices, rules, d)
        actions = actions_for(d, recs, short, props, qty, prices)

        t_recs = truth_recommendations(world, d)
        _, t_short = planner_net.plan(truth_items(world, d))
        t_props = find_stock(t_recs, t_short, truth_qty, balances, prices, rules, d)
        t_keys = {a.key for a in actions_for(d, t_recs, t_short, t_props, truth_qty, prices)}

        decisions = queue.review(actions, t_keys)
        all_decisions += decisions
        days[d] = (short, t_short)
        day = {
            "as_of": d.isoformat(),
            "shortfalls": score_shortfalls(short, t_short),
            "inventory": score_proposals(props, t_props),
            "approvals": score_decisions(decisions),
        }
        per_day.append(day)
        (out_dir / f"{d.isoformat()}.md").write_text(to_markdown(run, d, short, actions, decisions))
        (out_dir / f"{d.isoformat()}.json").write_text(
            json.dumps(
                {
                    **day,
                    "shortfalls_list": [asdict(s) for s in short],
                    "stock_proposals": [asdict(p) for p in props],
                    "actions": [asdict(a) for a in actions],
                },
                default=str,
                indent=2,
            )
        )
        d += timedelta(days=step_days)

    summary = {
        "run": {k: run[k] for k in ("asset", "profile", "seed", "window")},
        "days": len(per_day),
        "cost_usd": round(client.log.total_cost_usd, 4) if client else 0.0,
        "rule_cards": {
            "compiled": [c.name for c in rules.cards],
            "not_understood": rules.not_understood,
        },
        "shortfalls": _shortfall_totals(per_day),
        "earlier_detection": earlier_shortfalls(days, net, planner_net.transit),
        "inventory": _sum_scores([x["inventory"] for x in per_day], "value_usd"),
        "approvals": score_decisions(all_decisions),
        "outbox": {
            p.stem: sum(1 for _ in p.open()) for p in sorted((out_dir / "outbox").glob("*.jsonl"))
        },
        "per_day": per_day,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _sum_scores(scores: list[dict], *extra: str) -> dict:
    rep = sum(s["reported"] for s in scores)
    tru = sum(s["true"] for s in scores)
    hit = sum(round((s["precision"] or 0) * s["reported"]) for s in scores)
    out = {
        "reported": rep,
        "true": tru,
        "precision": round(hit / rep, 4) if rep else None,
        "recall": round(hit / tru, 4) if tru else None,
    }
    for k in extra:
        out[k] = round(sum(s[k] for s in scores), 2)
    return out


def _shortfall_totals(per_day: list[dict]) -> dict:
    s = [x["shortfalls"] for x in per_day]
    out = _sum_scores(s)
    both = [x for x in s if x["cause_right"] is not None]
    for k in ("cause_right", "option_right"):
        out[k] = round(sum(x[k] for x in both) / len(both), 4) if both else None
    out["by_cause"] = {}
    for k in CAUSES:
        n = {f: sum(x["by_cause"][k][f] for x in s) for f in ("reported", "true", "right")}
        n["precision"] = round(n["right"] / n["reported"], 4) if n["reported"] else None
        out["by_cause"][k] = n
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Phase 2 logistics, replayed day by day.")
    ap.add_argument("--run", required=True, help="a run directory written by fpso-synth")
    ap.add_argument("--from", dest="start", type=date.fromisoformat, help="first day")
    ap.add_argument("--to", dest="end", type=date.fromisoformat, help="last day")
    ap.add_argument("--step", type=int, default=7, help="days between replayed days")
    ap.add_argument("--max-cost", type=float, default=5.0, help="stop new calls above this USD")
    ap.add_argument("--no-model", action="store_true", help="rule links only, no email reading")
    ap.add_argument("--final", action="store_true", help="allow the held-back profile B")
    args = ap.parse_args(argv)
    s = run_logistics(
        args.run,
        args.start,
        args.end,
        args.step,
        args.max_cost,
        use_model=not args.no_model,
        final=args.final,
    )
    print(json.dumps({k: v for k, v in s.items() if k != "per_day"}, indent=2, default=str))


if __name__ == "__main__":
    main()
