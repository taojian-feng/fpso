"""First precision/recall numbers with the model in the loop (plan week 5).

    fpso-adjudicate --run data/kestrel-A --sample 60 --max-cost 5

Samples target records per link type, adjudicates all their candidates, and splits the
targets in two. The simulated reviewers label the calibration half; the calibrator and
auto-accept threshold are fitted on those labels. The test half is scored against the
hidden truth. Writes adjudication.{json,md}, adjudications.jsonl and actions.jsonl
under <run>/reports.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import zlib
from pathlib import Path

from fpso.connectors import open_all
from fpso.eval.harness import load_truth
from fpso.eval.reviewers import default_reviewers
from fpso.governance import ActionLog
from fpso.models import ModelClient, client_from_env
from fpso.resolution.adjudicate import Adjudication, Adjudicator, group_by_target
from fpso.resolution.calibrate import (
    Calibrator,
    choose_threshold,
    expected_calibration_error,
)
from fpso.resolution.candidates import LINK_TYPES, Candidate, CandidateGenerator

TARGET_PRECISION = 0.95
# Which simulated reviewer answers for which link type.
REVIEWER_FOR = {
    "equipment_activity": "dana.reyes",
    "work_order_activity": "dana.reyes",
    "component_po_line": "lee.park",
    "po_line_manifest_line": "lee.park",
    "po_email": "lee.park",
}


def sample_targets(cands: list[Candidate], per_type: int, seed: int = 0) -> list[Candidate]:
    groups = group_by_target(cands)
    keep: set[tuple[str, str]] = set()
    for lt in LINK_TYPES:
        targets = sorted(k for k in groups if k[0] == lt)
        rng = random.Random(f"{seed}|{lt}")
        keep.update(rng.sample(targets, min(per_type, len(targets))))
    return [c for c in cands if (c.link_type, c.to_key) in keep]


def is_calibration_target(link_type: str, to_key: str) -> bool:
    return zlib.crc32(f"{link_type}|{to_key}".encode()) % 2 == 0


def ratio(n: float, d: float) -> float | None:
    return round(n / d, 4) if d else None


def score_link_type(
    labelled: list[tuple[Adjudication, bool]], test: list[Adjudication], truth: set
) -> dict:
    """Fit calibration and threshold on reviewer labels, then score the test rows."""
    cal = Calibrator.fit([a.confidence for a, _ in labelled], [y for _, y in labelled])
    probs_cal = [cal.predict(a.confidence) for a, _ in labelled]
    threshold = choose_threshold(probs_cal, [y for _, y in labelled], TARGET_PRECISION)

    test_targets = {(a.link_type, a.to_key) for a in test}
    test_true = {p for p in truth if (p[0], p[2]) in test_targets}
    probs = [cal.predict(a.confidence) for a in test]
    y = [a.pair in truth for a in test]
    accepted = [a for a, p in zip(test, probs, strict=True) if p >= threshold]
    proposed = {a.pair for a in test if a.verdict != "no_match"}
    rule_acc = [a for a in test if a.rule_score >= 0.9]
    tp = sum(a.pair in truth for a in accepted)
    return {
        "targets": len(test_targets),
        "candidates": len(test),
        "calibration_labels": len(labelled),
        "threshold": round(threshold, 4),
        "test_true_links": len(test_true),
        "auto_accepted": len(accepted),
        "precision": ratio(tp, len(accepted)),
        "recall": ratio(len(proposed & test_true), len(test_true)),
        "auto_resolved": ratio(tp, len(test_true)),
        "ece": round(expected_calibration_error(probs, y), 4) if test else None,
        "raw_match_precision": ratio(
            sum(a.pair in truth for a in test if a.verdict == "match"),
            sum(a.verdict == "match" for a in test),
        ),
        "rule_precision": ratio(sum(a.pair in truth for a in rule_acc), len(rule_acc)),
        "rule_recall": ratio(sum(a.pair in truth for a in rule_acc), len(test_true)),
    }


def add_totals(out: dict[str, dict], cost_usd: float) -> dict[str, dict]:
    resolved = sum(
        round((s["auto_resolved"] or 0) * s["test_true_links"])
        for k, s in out.items()
        if not k.startswith("_")
    )
    out["_total"] = {
        "cost_usd": round(cost_usd, 4),
        "cost_per_resolved_link_usd": round(cost_usd / resolved, 4) if resolved else None,
    }
    return out


def score(
    adjs: list[Adjudication], truth: set, reviewers: dict, cost_usd: float
) -> dict[str, dict]:
    out = {}
    for lt in LINK_TYPES:
        rows = [a for a in adjs if a.link_type == lt]
        calib = [a for a in rows if is_calibration_target(lt, a.to_key)]
        test = [a for a in rows if not is_calibration_target(lt, a.to_key)]
        labelled = [(a, reviewers[REVIEWER_FOR[lt]].label(a.pair)) for a in calib]
        labelled = [(a, y) for a, y in labelled if y is not None]
        out[lt] = score_link_type(labelled, test, truth)
        # This quick experiment reports targets over both halves.
        out[lt]["targets"] = len({a.to_key for a in rows})
        out[lt]["candidates"] = len(rows)
    return add_totals(out, cost_usd)


def to_markdown(run: dict, scores: dict, meta: dict) -> str:
    def pct(v):
        return "n/a" if v is None else f"{v:.1%}"

    total = scores["_total"]
    lines = [
        f"# Adjudication report: {run['asset']}, profile {run['profile']} "
        f"(v{run['profile_version']}), seed {run['seed']}",
        "",
        f"Model: {meta['model']} (backend {meta['backend']}). {meta['sample']} targets sampled "
        f"per link type; half calibrate (labelled by simulated reviewers), half are scored "
        f"against truth. Cost ${total['cost_usd']:.2f}, "
        f"{meta['groups_skipped']} target groups skipped by the cost cap, {meta['seconds']:.0f} s.",
        "",
        "Auto-accept threshold is set for 95% precision on the calibration labels. Recall "
        "counts links the model did not reject; auto-resolved counts true links accepted "
        "above the threshold.",
        "",
        "| Link type | Test true links | Threshold | Precision | Recall | Auto-resolved | ECE "
        "| Rule precision | Rule recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lt in LINK_TYPES:
        s = scores[lt]
        thr = "none" if s["threshold"] > 1 else f"{s['threshold']:.2f}"
        lines.append(
            f"| {lt} | {s['test_true_links']} | {thr} | {pct(s['precision'])} "
            f"| {pct(s['recall'])} | {pct(s['auto_resolved'])} | {pct(s['ece'])} "
            f"| {pct(s['rule_precision'])} | {pct(s['rule_recall'])} |"
        )
    if total["cost_per_resolved_link_usd"] is not None:
        lines += ["", f"Cost per auto-resolved test link: ${total['cost_per_resolved_link_usd']}."]
    return "\n".join(lines) + "\n"


def run_adjudication(
    run_dir: str | Path,
    sample: int = 60,
    max_cost_usd: float | None = 5.0,
    workers: int = 4,
    client: ModelClient | None = None,
    out_dir: str | Path | None = None,
    rescore: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    prior_cost = 0.0
    if rescore and (out / "actions.jsonl").exists():
        for line in (out / "actions.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if rec["action"].startswith("model:"):
                prior_cost += rec.get("cost_usd") or 0.0
    log = ActionLog(out / "actions.jsonl")
    if client is None:
        client = client_from_env(log=log)
    else:
        client.log = log
    run = json.loads((run_dir / "truth" / "run.json").read_text())
    t0 = time.time()
    connectors = open_all(run_dir / "extracts")
    cands = sample_targets(CandidateGenerator(connectors).generate(), sample)
    adjudicator = Adjudicator(connectors, client, max_cost_usd=max_cost_usd, workers=workers)
    saved = out / "adjudications.jsonl"
    if rescore:
        # Re-score the model's saved answers: no model calls, so no cost.
        adjs = [Adjudication(**json.loads(line)) for line in saved.read_text().splitlines()]
    else:
        adjs = adjudicator.adjudicate(cands)
        with saved.open("w") as f:
            for a in adjs:
                f.write(json.dumps(a.as_dict()) + "\n")

    # Only the scoring step below reads the truth.
    truth = load_truth(run_dir)
    reviewers = {r.name: r for r in default_reviewers(truth, log=log)}
    scores = score(adjs, truth, reviewers, prior_cost + client.log.total_cost_usd)
    meta = {
        "model": adjs[0].model if rescore and adjs else client.role_models["adjudicate"],
        "backend": "replayed" if rescore else type(client.backend).__name__,
        "sample": sample,
        "groups_skipped": adjudicator.skipped_groups,
        "seconds": round(time.time() - t0, 1),
    }
    report = {"run": run, "meta": meta, "scores": scores}
    (out / "adjudication.json").write_text(json.dumps(report, indent=2))
    (out / "adjudication.md").write_text(to_markdown(run, scores, meta))
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Adjudicate sampled candidates and score them.")
    ap.add_argument("--run", required=True, help="a run directory written by fpso-synth")
    ap.add_argument("--sample", type=int, default=60, help="targets per link type")
    ap.add_argument("--max-cost", type=float, default=5.0, help="stop new calls above this USD")
    ap.add_argument("--workers", type=int, default=4, help="parallel model calls")
    ap.add_argument("--out", help="report directory (default: <run>/reports)")
    ap.add_argument(
        "--rescore", action="store_true", help="re-score saved adjudications.jsonl, no model calls"
    )
    args = ap.parse_args(argv)
    report = run_adjudication(
        args.run, args.sample, args.max_cost, args.workers, out_dir=args.out, rescore=args.rescore
    )
    print(to_markdown(report["run"], report["scores"], report["meta"]))


if __name__ == "__main__":
    main()
