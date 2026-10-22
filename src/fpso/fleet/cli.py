"""The Phase 4 check: onboard the sister assets, run the agents on them, ask suppliers first.

    fpso-fleet --out data/fleet --scale 0.25

1. FPSO Kestrel is generated as before (SAP-style, profile A) and registered in the shared
   library as the baseline: connectors and field meanings worked out by hand.
2. FPSO Merlin arrives Maximo-style. No library dialect fits, so the rule mapper drafts the
   mapping; the draft is scored against the generator's truth, a simulated reviewer fixes
   what is wrong, and the confirmed mapping joins the library as the 'maximo' dialect.
3. FPSO Osprey runs the same Maximo set-up with two local renames. The library dialect
   covers most of it; the mapper handles only the rest.
4. Each asset is adapted to the canonical layout and the Readiness and Procurement agents
   (with Schedule and Supplier Liaison) run on it unchanged, without a model, scored
   against that asset's truth.
5. Every chase the Liaison agent drafts goes to the vendor's supplier agent over A2A
   first, falling back to the email when there is no answer in time.

Writes ``reports/fleet/summary.{json,md}``, each asset's mapping draft and the library under
the output directory. The gate: days to onboard Merlin clearly below Kestrel's (under half).
``--model`` also asks the Onboarding agent (``fpso.onboarding``) for its plain-language
draft of each raw source; off by default.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from fpso.fleet.a2a import LiaisonClient, build_bus, score_exchanges
from fpso.fleet.adapter import adapt_run, raw_sources
from fpso.fleet.assets import FLEET, KESTREL, Asset, asset_world
from fpso.fleet.canonical import identity_mapping
from fpso.fleet.dialects import dialect_mapping, export_maximo
from fpso.fleet.mapper import (
    confirm,
    draft_mapping,
    effort_assumptions,
    kestrel_baseline,
    reuse_mapping,
    score_draft,
)
from fpso.library import AssetEntry, Library
from fpso.procurement import run_procurement
from fpso.readiness import report_dates, run_readiness
from fpso.synthetic.export import export

GATE_RATIO = 0.5  # 'clearly below': under half of Kestrel's days


def _slug(asset: Asset) -> str:
    return asset.name.split()[-1].lower()


def onboard(asset: Asset, out: Path, library: Library, scale: float, use_model: bool) -> dict:
    """Generate, map, review and adapt one Maximo-style asset; register it in the library."""
    run_dir = out / _slug(asset)
    raw = run_dir / "raw"
    export_maximo(asset_world(asset, scale), asset, raw)
    sources = raw_sources(raw / "extracts")
    dialect, coverage = library.find_dialect(sources)
    if dialect:
        draft = reuse_mapping(library.mapping(dialect), sources)
    else:
        draft = draft_mapping(sources)
    truth = dialect_mapping(asset)
    score = score_draft(draft, truth)
    confirmed = confirm(truth)
    if dialect is None:
        library.add_dialect(asset.dialect, confirmed)
    adapt_run(raw, confirmed, run_dir)
    rep = out / "reports" / "fleet" / asset.code
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "mapping-draft.json").write_text(json.dumps(draft.as_dict(), indent=2))
    (rep / "mapping-draft.md").write_text(draft_markdown(asset, draft, score))
    if use_model:
        _model_drafts(sources, rep)
    result = {
        "library_dialect": dialect,
        "library_coverage": coverage,
        **score,
    }
    library.add_asset(
        AssetEntry(asset.code, asset.name, asset.dialect, str(run_dir), confirmed.as_dict(), result)
    )
    return result


def _model_drafts(sources: dict, rep: Path) -> None:
    """The Onboarding agent's model draft of each raw source (only with --model)."""
    from fpso import onboarding
    from fpso.governance import ActionLog
    from fpso.models import client_from_env

    client = client_from_env(log=ActionLog(rep / "actions.jsonl"))
    for name, con in sources.items():
        md = onboarding.to_markdown(name, onboarding.onboard(con, client))
        (rep / f"onboarding-{name}.md").write_text(md)


def draft_markdown(asset: Asset, draft, score: dict) -> str:
    lines = [
        f"# Mapping draft: {asset.name} ({asset.dialect})",
        "",
        f"{score['auto_correct']} of {score['fields']} fields right without help "
        f"({score['auto_correct_share']:.0%}); {score['needs_fix']} need a fix. "
        f"Estimated {score['person_hours']} person-hours.",
        "",
        "| Canonical field | Raw table | Raw column | Date format | Confidence | Accepted | From |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for fd in draft.fields:
        lines.append(
            f"| {fd.field} | {fd.table or '-'} | {fd.column or '-'} | {fd.fmt or '-'} "
            f"| {fd.confidence:.2f} | {'yes' if fd.accepted else 'review'} | {fd.source} |"
        )
    return "\n".join(lines) + "\n"


def run_agents(run_dir: Path, step_days: int) -> dict:
    """Readiness and Procurement (with Schedule, Liaison, brief), free mode."""
    run = json.loads((run_dir / "truth" / "run.json").read_text())
    reports = run_readiness(
        run_dir, report_dates(date.fromisoformat(run["turnaround"][0])), use_model=False
    )
    proc = run_procurement(run_dir, step_days=step_days, use_model=False)
    return {
        "readiness": [
            {
                "as_of": r["meta"]["as_of"],
                **{
                    k: r["scores"][k]
                    for k in ("work_packages", "agreement", "gap_precision", "gap_recall")
                },
            }
            for r in reports
        ],
        "procurement": {
            k: proc[k] for k in ("days", "totals", "earlier_detection", "liaison", "brief")
        },
    }


def run_a2a(out: Path, assets: list[Asset], scale: float) -> dict:
    """Send each day's Liaison chases to supplier agents first, per asset."""
    worlds = {a.code: asset_world(a, scale) for a in assets}
    bus = build_bus(worlds)
    per_asset, all_ex = {}, []
    for a in assets:
        client = LiaisonClient(bus, a.code)
        exchanges = []
        proc = out / _slug(a) / "reports" / "procurement"
        for path in sorted(proc.glob("20*.json")):
            day = json.loads(path.read_text())
            d = date.fromisoformat(day["as_of"])
            for chase in day["liaison"]["drafts"]:
                exchanges.append(client.ask(d, chase))
        per_asset[a.code] = score_exchanges(exchanges, worlds[a.code])
        all_ex += exchanges
        (out / "reports" / "fleet" / a.code).mkdir(parents=True, exist_ok=True)
        (out / "reports" / "fleet" / a.code / "a2a.jsonl").write_text(
            "".join(json.dumps(x.__dict__, default=str) + "\n" for x in exchanges)
        )
    n = sum(s["questions"] for s in per_asset.values())
    hit = sum(s["resolved_by_a2a"] for s in per_asset.values())
    lines = sum(s["lines_answered"] for s in per_asset.values())
    right = sum(
        round((s["promised_date_accuracy"] or 0) * s["lines_answered"]) for s in per_asset.values()
    )
    return {
        "agents": len(bus.agents),
        "vendors": len(next(iter(worlds.values())).vendors),
        "cards": bus.cards(),
        "messages": bus.sent,
        "questions": n,
        "resolved_by_a2a": hit,
        "resolved_share": round(hit / n, 4) if n else None,
        "promised_date_accuracy": round(right / lines, 4) if lines else None,
        "per_asset": per_asset,
    }


def run_fleet(
    out: str | Path, scale: float = 0.25, step_days: int = 7, use_model: bool = False
) -> dict:
    out = Path(out)
    library = Library.new()
    kestrel_dir = out / _slug(KESTREL)
    export(asset_world(KESTREL, scale), kestrel_dir, "A")
    library.add_asset(
        AssetEntry(
            KESTREL.code,
            KESTREL.name,
            KESTREL.dialect,
            str(kestrel_dir),
            identity_mapping().as_dict(),
            kestrel_baseline(),
        )
    )
    onboarding = {KESTREL.code: kestrel_baseline()}
    for code in ("MRL", "OSP"):
        onboarding[code] = onboard(FLEET[code], out, library, scale, use_model)
    assets = list(FLEET.values())
    agents = {a.code: run_agents(out / _slug(a), step_days) for a in assets}
    a2a = run_a2a(out, assets, scale)
    library.save(out / "reports" / "fleet" / "library.json")
    k, m = onboarding["KST"]["person_days"], onboarding["MRL"]["person_days"]
    summary = {
        "scale": scale,
        "step_days": step_days,
        "assumptions": effort_assumptions(),
        "onboarding": onboarding,
        "gate": {
            "kestrel_days": k,
            "merlin_days": m,
            "osprey_days": onboarding["OSP"]["person_days"],
            "ratio": round(m / k, 3),
            "passed": m < GATE_RATIO * k,
        },
        "agents": agents,
        "a2a": a2a,
        "library": {
            "connectors": sorted(library.connectors),
            "dialects": sorted(library.dialects),
            "rule_cards": sorted(library.rule_cards),
            "assets": {c: e.dialect for c, e in library.assets.items()},
        },
    }
    rep = out / "reports" / "fleet"
    (rep / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (rep / "summary.md").write_text(to_markdown(summary))
    return summary


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def to_markdown(s: dict) -> str:
    g, a = s["gate"], s["assumptions"]
    lines = [
        "# Phase 4: fleet and suppliers",
        "",
        f"Gate: days to onboard the second asset clearly below the first. FPSO Merlin took "
        f"{g['merlin_days']} person-days against FPSO Kestrel's {g['kestrel_days']} "
        f"({g['ratio']:.0%}); FPSO Osprey, reusing Merlin's dialect, {g['osprey_days']}. "
        f"**{'Passed' if g['passed'] else 'Not passed'}** (under {GATE_RATIO:.0%}).",
        "",
        "## Onboarding",
        "",
        "| Asset | Library dialect | Fields | Right without help | From library "
        "| Flagged, right | Fixed | Silent errors | Person-hours | Person-days |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for code, o in s["onboarding"].items():
        lines.append(
            f"| {code} | {o.get('library_dialect') or '-'} | {o['fields']} "
            f"| {_pct(o['auto_correct_share'])} | {o.get('reused_from_library', 0)} "
            f"| {o.get('flagged_and_right', 0)} | {o['needs_fix']} | {o.get('silent_errors', 0)} "
            f"| {o['person_hours']} | {o['person_days']} |"
        )
    lines += [
        "",
        f"Effort assumptions: Kestrel's {a['sources']} connectors were written by hand "
        f"({a['new_connector_hours_per_source']} h each) and every field worked out by hand "
        f"({a['fix_minutes_per_field']:.0f} min each). A later asset needs "
        f"{a['source_setup_hours']} h per source to set up ({a['library_setup_hours']} h with "
        f"a library dialect), {a['fix_minutes_per_field']:.0f} min per field fixed, "
        f"{a['review_minutes_per_flagged_field']:.0f} min per flagged field confirmed and "
        f"{a['glance_minutes_per_accepted_field']:.0f} min per accepted field; "
        f"{a['hours_per_day']:.0f} h per day.",
        "",
        "## Agents on each asset (no model)",
        "",
        "| Asset | Readiness agreement | Gap precision | Gap recall | order_now P/R "
        "| order_soon P/R | expedite P/R | No-ack precision | Brief accepted |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for code, ag in s["agents"].items():
        r = ag["readiness"]
        t = ag["procurement"]["totals"]
        pr = {k: f"{_pct(v['precision'])} / {_pct(v['recall'])}" for k, v in t.items()}
        lines.append(
            f"| {code} | {' / '.join(_pct(x['agreement']) for x in r)} "
            f"| {' / '.join(_pct(x['gap_precision']) for x in r)} "
            f"| {' / '.join(_pct(x['gap_recall']) for x in r)} "
            f"| {pr['order_now']} | {pr['order_soon']} | {pr['expedite']} "
            f"| {_pct(ag['procurement']['liaison']['no_ack_precision'])} "
            f"| {_pct(ag['procurement']['brief']['accepted_unchanged'])} |"
        )
    x = s["a2a"]
    lines += [
        "",
        "Readiness columns give the two report dates (6 and 5 weeks before the turnaround).",
        "",
        "## Supplier agents over A2A",
        "",
        f"{x['agents']} of {x['vendors']} vendors run an agent. Of {x['questions']} chases the "
        f"Liaison agent drafted, {x['resolved_by_a2a']} ({_pct(x['resolved_share'])}) were "
        f"settled over A2A without an email; promised dates were right on "
        f"{_pct(x['promised_date_accuracy'])} of the lines answered.",
        "",
        "| Asset | Chases | Settled by A2A | No-ack share | Expedite share | Emails still needed "
        "| Why email | Date accuracy | On slipped lines | Slips learned before the email |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for code, p in x["per_asset"].items():
        why = ", ".join(f"{k} {v}" for k, v in p["email_reasons"].items()) or "-"
        lines.append(
            f"| {code} | {p['questions']} | {p['resolved_by_a2a']} "
            f"| {_pct(p['by_kind']['no_ack']['share'])} "
            f"| {_pct(p['by_kind']['expedite']['share'])} | {p['emails_still_needed']} | {why} "
            f"| {_pct(p['promised_date_accuracy'])} "
            f"| {_pct(p['slipped_line_accuracy'])} ({p['slipped_lines_answered']}) "
            f"| {p['slips_learned_before_email']} |"
        )
    lib = s["library"]
    lines += [
        "",
        "## Shared library",
        "",
        f"Connectors: {', '.join(lib['connectors'])}. Dialects: {', '.join(lib['dialects'])}. "
        f"Rule cards: {', '.join(lib['rule_cards'])}. Assets: "
        + ", ".join(f"{c} ({d})" for c, d in lib["assets"].items())
        + ".",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True, help="directory for the fleet's runs and reports")
    ap.add_argument("--scale", type=float, default=0.25, help="1.0 = full size from the plan")
    ap.add_argument("--step", type=int, default=7, help="days between replayed days")
    ap.add_argument("--model", action="store_true", help="also ask the model for source drafts")
    args = ap.parse_args(argv)
    s = run_fleet(args.out, args.scale, args.step, use_model=args.model)
    print(json.dumps({k: s[k] for k in ("gate", "onboarding")}, indent=2))
    print(json.dumps({k: v for k, v in s["a2a"].items() if k != "cards"}, indent=2))


if __name__ == "__main__":
    main()
