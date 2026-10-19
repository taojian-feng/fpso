"""Gold set and measured rounds (plan WS6, week 6 onward).

    fpso-round --run data/kestrel-A --round 0

The gold set is a fixed, stratified sample of target records per link type (for example
80 manifest lines), split into dev and held-out. It is written once to
reports/gold.json and reused by every later round, so rounds stay comparable and the
held-out targets are never used for tuning.

Each round adjudicates the gold targets' candidates. On the dev targets, the simulated
reviewers label the candidate pairs: the pairs that decide the threshold (high model
confidence) and a random 20% of the rest get a second reviewer, and pairs where the two
disagree are left out of calibration. The calibrator and auto-accept threshold are fitted
on those labels; the held-out targets are scored against the hidden truth. Results go to
reports/rounds/round-<n>.{json,md}, with reports/rounds/summary.md comparing all rounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

from fpso.connectors import open_all
from fpso.eval.adjudication import REVIEWER_FOR, add_totals, ratio, score_link_type
from fpso.eval.harness import load_truth
from fpso.eval.reviewers import SimulatedReviewer, default_reviewers
from fpso.governance import ActionLog
from fpso.models import ModelClient, client_from_env
from fpso.resolution import adjudicate as adjudicate_mod
from fpso.resolution.adjudicate import Adjudication, Adjudicator, group_by_target
from fpso.resolution.candidates import LINK_TYPES, Candidate, CandidateGenerator

EASY_RULE_SCORE = 0.9  # a target is "easy" when its best candidate already scores this
DOUBLE_LABEL_CONFIDENCE = 0.9  # pairs at or above this always get a second reviewer
DOUBLE_LABEL_SHARE = 0.2  # random share of the remaining pairs that get one too


@dataclass(frozen=True)
class GoldTarget:
    link_type: str
    to_key: str
    split: str  # "dev" or "heldout"
    stratum: str  # "easy" or "hard"


def build_gold(
    cands: list[Candidate], per_type: int = 80, dev_share: float = 0.25, seed: int = 0
) -> list[GoldTarget]:
    """Half easy and half hard targets per link type where possible, split within strata."""
    groups = group_by_target(cands)
    gold = []
    for lt in LINK_TYPES:
        rng = random.Random(f"gold|{seed}|{lt}")
        strata: dict[str, list[str]] = {"easy": [], "hard": []}
        for (t_lt, key), cs in sorted(groups.items()):
            if t_lt == lt:
                best = max(c.score for c in cs)
                strata["easy" if best >= EASY_RULE_SCORE else "hard"].append(key)
        for keys in strata.values():
            rng.shuffle(keys)
        n_easy = min(len(strata["easy"]), per_type // 2)
        n_hard = min(len(strata["hard"]), per_type - n_easy)
        n_easy = min(len(strata["easy"]), per_type - n_hard)
        for stratum, n in (("easy", n_easy), ("hard", n_hard)):
            chosen = strata[stratum][:n]
            n_dev = round(n * dev_share)
            gold += [
                GoldTarget(lt, k, "dev" if i < n_dev else "heldout", stratum)
                for i, k in enumerate(chosen)
            ]
    return gold


def load_or_build_gold(path: Path, cands: list[Candidate], **kwargs) -> list[GoldTarget]:
    if path.exists():
        return [GoldTarget(**g) for g in json.loads(path.read_text())["targets"]]
    gold = build_gold(cands, **kwargs)
    path.write_text(json.dumps({"params": kwargs, "targets": [asdict(g) for g in gold]}, indent=1))
    return gold


def _in_random_share(pair: tuple[str, str, str], share: float, seed: int) -> bool:
    return zlib.crc32(f"double|{seed}|{'|'.join(pair)}".encode()) % 1000 < share * 1000


def label_dev(
    rows: list[Adjudication],
    primary: SimulatedReviewer,
    secondary: SimulatedReviewer,
    truth: set,
    seed: int = 0,
) -> tuple[list[tuple[Adjudication, bool]], dict]:
    """Reviewer labels for calibration, with double-labelling where it matters."""
    labelled = []
    agree = both = dropped = answered = correct = 0
    for a in rows:
        y1 = primary.label(a.pair)
        in_sample = _in_random_share(a.pair, DOUBLE_LABEL_SHARE, seed)
        y2 = None
        if a.confidence >= DOUBLE_LABEL_CONFIDENCE or in_sample:
            y2 = secondary.label(a.pair)
        if y1 is not None:
            answered += 1
            correct += y1 == (a.pair in truth)
        if in_sample and y1 is not None and y2 is not None:
            both += 1
            agree += y1 == y2
        if y1 is not None and y2 is not None and y1 != y2:
            dropped += 1
            continue
        y = y1 if y1 is not None else y2
        if y is not None:
            labelled.append((a, y))
    stats = {
        "labels_used": len(labelled),
        "disagreements_dropped": dropped,
        "agreement_on_random_double_labels": ratio(agree, both),
        "random_double_labels": both,
        "primary_accuracy_vs_truth": ratio(correct, answered),
        "label_precision_vs_truth": ratio(
            sum(a.pair in truth for a, y in labelled if y), sum(y for _, y in labelled)
        ),
    }
    return labelled, stats


def score_round(
    adjs: list[Adjudication], gold: list[GoldTarget], truth: set, reviewers: dict, cost: float
) -> dict[str, dict]:
    split = {(g.link_type, g.to_key): g.split for g in gold}
    out = {}
    for lt in LINK_TYPES:
        rows = [a for a in adjs if a.link_type == lt]
        dev = [a for a in rows if split.get((lt, a.to_key)) == "dev"]
        held = [a for a in rows if split.get((lt, a.to_key)) == "heldout"]
        primary = reviewers[REVIEWER_FOR[lt]]
        secondary = next(r for name, r in reviewers.items() if name != primary.name)
        labelled, stats = label_dev(dev, primary, secondary, truth)
        out[lt] = {**score_link_type(labelled, held, truth), "reviewers": stats}
    return add_totals(out, cost)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def to_markdown(run: dict, n: int, scores: dict, meta: dict) -> str:
    def pct(v):
        return "n/a" if v is None else f"{v:.1%}"

    total = scores["_total"]
    lines = [
        f"# Round {n}: {run['asset']}, profile {run['profile']} (v{run['profile_version']}), "
        f"seed {run['seed']}",
        "",
        f"Model {meta['model']} ({meta['backend']}), prompt {meta['prompt_hash']}, code "
        f"{meta['commit'] or 'unknown'}. Gold set: {meta['gold_dev']} dev and "
        f"{meta['gold_heldout']} held-out targets. Cost ${total['cost_usd']:.2f}, "
        f"{meta['groups_skipped']} target groups skipped by the cost cap.",
        "",
        "Scores are on held-out targets against the hidden truth. The threshold is fitted on "
        "dev labels from the simulated reviewers for 95% precision. Targets are half easy (a "
        "rule candidate already scores at least 0.9) and half hard, so rates are not "
        "population averages.",
        "",
        "| Link type | True links | Threshold | Precision | Recall | Auto-resolved | ECE "
        "| Rule recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lt in LINK_TYPES:
        s = scores[lt]
        thr = "none" if s["threshold"] > 1 else f"{s['threshold']:.2f}"
        lines.append(
            f"| {lt} | {s['test_true_links']} | {thr} | {pct(s['precision'])} "
            f"| {pct(s['recall'])} | {pct(s['auto_resolved'])} | {pct(s['ece'])} "
            f"| {pct(s['rule_recall'])} |"
        )
    lines += [
        "",
        "## Reviewer labels (dev)",
        "",
        "| Link type | Labels used | Disagreements dropped | Agreement (random 20%) "
        "| Primary accuracy | Label precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lt in LINK_TYPES:
        r = scores[lt]["reviewers"]
        lines.append(
            f"| {lt} | {r['labels_used']} | {r['disagreements_dropped']} "
            f"| {pct(r['agreement_on_random_double_labels'])} "
            f"| {pct(r['primary_accuracy_vs_truth'])} | {pct(r['label_precision_vs_truth'])} |"
        )
    if total["cost_per_resolved_link_usd"] is not None:
        lines += ["", f"Cost per auto-resolved link: ${total['cost_per_resolved_link_usd']}."]
    return "\n".join(lines) + "\n"


def write_summary(rounds_dir: Path) -> None:
    reports = sorted(
        (json.loads(p.read_text()) for p in rounds_dir.glob("round-*.json")),
        key=lambda r: r["round"],
    )
    lines = [
        "# Rounds summary",
        "",
        "Auto-resolved share of held-out true links, with precision in brackets.",
        "",
        "| Round | " + " | ".join(LINK_TYPES) + " | Cost |",
        "| ---: | " + " | ".join("---:" for _ in LINK_TYPES) + " | ---: |",
    ]
    for r in reports:
        cells = []
        for lt in LINK_TYPES:
            s = r["scores"][lt]
            ar = "n/a" if s["auto_resolved"] is None else f"{s['auto_resolved']:.0%}"
            pr = "n/a" if s["precision"] is None else f"{s['precision']:.0%}"
            cells.append(f"{ar} ({pr})")
        cost = r["scores"]["_total"]["cost_usd"]
        lines.append(f"| {r['round']} | " + " | ".join(cells) + f" | ${cost:.2f} |")
    (rounds_dir / "summary.md").write_text("\n".join(lines) + "\n")


def run_round(
    run_dir: str | Path,
    n: int = 0,
    per_type: int = 80,
    max_cost_usd: float | None = 5.0,
    workers: int = 4,
    client: ModelClient | None = None,
    rescore: bool = False,
    final: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "truth" / "run.json").read_text())
    if run["profile"] == "B" and not final:
        raise SystemExit("Profile B is held back for the final run; pass --final to use it.")
    reports = run_dir / "reports"
    rounds_dir = reports / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    log = ActionLog(rounds_dir / f"round-{n}.actions.jsonl")
    if client is None:
        client = client_from_env(log=log)
    else:
        client.log = log

    t0 = time.time()
    connectors = open_all(run_dir / "extracts")
    cands = CandidateGenerator(connectors).generate()
    gold = load_or_build_gold(reports / "gold.json", cands, per_type=per_type)
    keep = {(g.link_type, g.to_key) for g in gold}
    gold_cands = [c for c in cands if (c.link_type, c.to_key) in keep]

    saved = rounds_dir / f"round-{n}.adjudications.jsonl"
    adjudicator = Adjudicator(connectors, client, max_cost_usd=max_cost_usd, workers=workers)
    if rescore:
        adjs = [Adjudication(**json.loads(line)) for line in saved.read_text().splitlines()]
        # The cost is the original run's: replaying makes no calls.
        cost = json.loads((rounds_dir / f"round-{n}.json").read_text())["scores"]["_total"][
            "cost_usd"
        ]
    else:
        adjs = adjudicator.adjudicate(gold_cands)
        with saved.open("w") as f:
            for a in adjs:
                f.write(json.dumps(a.as_dict()) + "\n")
        cost = client.log.total_cost_usd

    # Only the scoring below reads the truth.
    truth = load_truth(run_dir)
    reviewers = {r.name: r for r in default_reviewers(truth, log=log)}
    scores = score_round(adjs, gold, truth, reviewers, cost)
    meta = {
        "model": adjs[0].model if adjs else client.role_models["adjudicate"],
        "backend": "replayed" if rescore else type(client.backend).__name__,
        "prompt_hash": hashlib.sha256(adjudicate_mod.SYSTEM.encode()).hexdigest()[:10],
        "commit": _git_commit(),
        "gold_dev": sum(g.split == "dev" for g in gold),
        "gold_heldout": sum(g.split == "heldout" for g in gold),
        "groups_skipped": adjudicator.skipped_groups,
        "seconds": round(time.time() - t0, 1),
    }
    report = {"round": n, "run": run, "meta": meta, "scores": scores}
    (rounds_dir / f"round-{n}.json").write_text(json.dumps(report, indent=2))
    (rounds_dir / f"round-{n}.md").write_text(to_markdown(run, n, scores, meta))
    write_summary(rounds_dir)
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run one measured round on the gold set.")
    ap.add_argument("--run", required=True, help="a run directory written by fpso-synth")
    ap.add_argument("--round", type=int, default=0, help="round number (0 is the baseline)")
    ap.add_argument("--per-type", type=int, default=80, help="gold targets per link type")
    ap.add_argument("--max-cost", type=float, default=5.0, help="stop new calls above this USD")
    ap.add_argument("--workers", type=int, default=4, help="parallel model calls")
    ap.add_argument("--rescore", action="store_true", help="re-score saved answers, no calls")
    ap.add_argument("--final", action="store_true", help="allow the held-back profile B")
    args = ap.parse_args(argv)
    report = run_round(
        args.run,
        args.round,
        args.per_type,
        args.max_cost,
        args.workers,
        rescore=args.rescore,
        final=args.final,
    )
    print(to_markdown(report["run"], report["round"], report["scores"], report["meta"]))


if __name__ == "__main__":
    main()
