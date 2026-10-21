"""Gold set and measured rounds (plan WS6, week 6 onward).

    fpso-round --run data/kestrel-A --round 0

The gold set is a fixed, stratified sample of target records per link type (for example
80 manifest lines), split into dev and held-out. It is written once to
reports/gold.json and reused by every later round, so rounds stay comparable and the
held-out targets are never used for tuning.

Each round adjudicates the gold targets' candidates. On the dev targets, the simulated
reviewers label the candidate pairs: the pairs that decide the threshold (high model
confidence) and a random 20% of the rest get a second reviewer, and pairs where the two
disagree, or where one of the two skipped, are left out of calibration. The calibrator and
auto-accept threshold are fitted on those labels; the held-out targets are scored against
the hidden truth. Results go to reports/rounds/round-<n>.{json,md}, with
reports/rounds/summary.md comparing all rounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
import zlib
from collections import defaultdict
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
from fpso.resolution.induce import evidence_for, examples_for, propose
from fpso.resolution.patterns import Pattern, apply_patterns

EASY_RULE_SCORE = 0.9  # a target is "easy" when its best candidate already scores this
DOUBLE_LABEL_CONFIDENCE = 0.9  # pairs at or above this always get a second reviewer
DOUBLE_LABEL_SHARE = 0.2  # random share of the remaining pairs that get one too
# From round 1 on, a link is auto-accepted only when the model's own (pattern-adjusted)
# confidence is at least this, on top of the calibrated threshold. Round 0 had none.


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
    agree = both = dropped = unconfirmed = answered = correct = 0
    for a in rows:
        y1 = primary.label(a.pair)
        in_sample = _in_random_share(a.pair, DOUBLE_LABEL_SHARE, seed)
        double = a.confidence >= DOUBLE_LABEL_CONFIDENCE or in_sample
        y2 = secondary.label(a.pair) if double else None
        if y1 is not None:
            answered += 1
            correct += y1 == (a.pair in truth)
        if in_sample and y1 is not None and y2 is not None:
            both += 1
            agree += y1 == y2
        if y1 is not None and y2 is not None and y1 != y2:
            dropped += 1
            continue
        if double and (y1 is None) != (y2 is None):
            # One reviewer skipped, so nothing confirms the other's answer. A single wrong
            # "no" here is enough to sink a threshold on a small dev set.
            unconfirmed += 1
            continue
        y = y1 if y1 is not None else y2
        if y is not None:
            labelled.append((a, y))
    stats = {
        "labels_used": len(labelled),
        "disagreements_dropped": dropped,
        "unconfirmed_dropped": unconfirmed,
        "agreement_on_random_double_labels": ratio(agree, both),
        "random_double_labels": both,
        "primary_accuracy_vs_truth": ratio(correct, answered),
        "label_precision_vs_truth": ratio(
            sum(a.pair in truth for a, y in labelled if y), sum(y for _, y in labelled)
        ),
    }
    return labelled, stats


def dev_labels(
    adjs: list[Adjudication], gold: list[GoldTarget], truth: set, reviewers: dict
) -> dict[str, tuple[list[tuple[Adjudication, bool]], dict]]:
    split = {(g.link_type, g.to_key): g.split for g in gold}
    out = {}
    for lt in LINK_TYPES:
        dev = [a for a in adjs if a.link_type == lt and split.get((lt, a.to_key)) == "dev"]
        primary = reviewers[REVIEWER_FOR[lt]]
        secondary = next(r for name, r in reviewers.items() if name != primary.name)
        out[lt] = label_dev(dev, primary, secondary, truth)
    return out


def score_round(
    adjs: list[Adjudication],
    gold: list[GoldTarget],
    truth: set,
    reviewers: dict,
    cost: float,
    min_confidence: float = 0.0,
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
        out[lt] = {
            **score_link_type(labelled, held, truth, min_confidence),
            "reviewers": stats,
        }
    return add_totals(out, cost)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_adjudications(path: Path) -> list[Adjudication]:
    return [Adjudication(**json.loads(line)) for line in path.read_text().splitlines()]


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def group_by_adjudication(adjs: list[Adjudication]) -> dict[tuple[str, str], list[Adjudication]]:
    out: dict[tuple[str, str], list[Adjudication]] = defaultdict(list)
    for a in adjs:
        out[(a.link_type, a.to_key)].append(a)
    return out


def _read_patterns(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def _reusable_round(rounds_dir: Path, n: int, prompt_hash: str) -> int | None:
    """The latest earlier round whose answers came from the same adjudication prompt."""
    for k in range(n - 1, -1, -1):
        report = rounds_dir / f"round-{k}.json"
        answers = rounds_dir / f"round-{k}.adjudications.jsonl"
        if report.exists() and answers.exists():
            if json.loads(report.read_text())["meta"]["prompt_hash"] == prompt_hash:
                return k
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
        "| Link type | Labels used | Disagreements dropped | Unconfirmed dropped "
        "| Agreement (random 20%) | Primary accuracy | Label precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lt in LINK_TYPES:
        r = scores[lt]["reviewers"]
        lines.append(
            f"| {lt} | {r['labels_used']} | {r['disagreements_dropped']} "
            f"| {r['unconfirmed_dropped']} | {pct(r['agreement_on_random_double_labels'])} "
            f"| {pct(r['primary_accuracy_vs_truth'])} | {pct(r['label_precision_vs_truth'])} |"
        )
    if meta.get("answers_from_round") is not None:
        asked = meta.get("groups_asked")
        lines += [
            "",
            f"Model answers are reused from round {meta['answers_from_round']} for every "
            "target whose prompt is unchanged"
            + (
                f"; {asked} targets with changed candidates or signals were asked again."
                if asked
                else "; this round's cost is pattern induction only."
            ),
        ]
    proposed = meta.get("patterns_this_round") or []
    in_force = meta.get("patterns_in_force") or []
    if n >= 1:
        lines += ["", "## Patterns", ""]
        lines.append(f"In force this round: {', '.join(in_force) if in_force else 'none'}.")
        if proposed:
            lines += ["", "| Pattern | Link type | Rule | Reviewer | Decision |"]
            lines.append("| --- | --- | --- | --- | --- |")
            for p in proposed:
                lines.append(
                    f"| {p['id']} | {p['link_type']} | {p['statement']} "
                    f"| {p['reviewer']} | {p['status']} |"
                )
        else:
            lines += ["", "No new patterns were proposed this round."]
        for f in meta.get("patterns_filtered") or []:
            lines.append(f"- Filtered out ({', '.join(f['filtered_because'])}): {f['statement']}")
        notes = {lt: t for lt, t in (meta.get("induce_notes") or {}).items() if t}
        if notes:
            lines += ["", "Inducer notes:", ""]
            lines += [f"- {lt}: {t}" for lt, t in notes.items()]
        unsure = meta.get("induction_unsure_targets")
        if unsure:
            lines += [
                "",
                "Unsure dev targets shown to the inducer: "
                + ", ".join(f"{lt} {k}" for lt, k in unsure.items())
                + ".",
            ]
    if total["cost_per_resolved_link_usd"] is not None:
        lines += ["", f"Cost per auto-resolved link: ${total['cost_per_resolved_link_usd']}."]
    return "\n".join(lines) + "\n"


def write_summary(rounds_dir: Path) -> None:
    reports = sorted(
        (
            json.loads(p.read_text())
            for p in rounds_dir.glob("round-*.json")
            if p.stem.removeprefix("round-").isdigit()
        ),
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
    patterns_from: str | Path | None = None,
) -> dict:
    """One measured round. The final run (profile B) applies the patterns confirmed on
    profile A, read from ``patterns_from``, and induces none: B is never tuned on."""
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
    prompt_hash = hashlib.sha256(adjudicate_mod.SYSTEM.encode()).hexdigest()[:10]
    earlier = _reusable_round(rounds_dir, n, prompt_hash)
    reused_from = None
    if rescore:
        adjs = _read_adjudications(saved)
        prior_report = json.loads((rounds_dir / f"round-{n}.json").read_text())
        prior = prior_report["meta"]
        model_cost = prior.get("model_cost_usd", prior_report["scores"]["_total"]["cost_usd"])
        reused_from = prior.get("answers_from_round")
        groups_asked = prior.get("groups_asked")
    else:
        groups = group_by_target(gold_cands)
        fingerprints = {
            f"{lt}|{k}": hashlib.sha256(adjudicator.prompt(lt, k, g).encode()).hexdigest()[:16]
            for (lt, k), g in groups.items()
        }
        reused: list[Adjudication] = []
        todo = gold_cands
        if earlier is not None:
            # Same system prompt: reuse the earlier answer for every target whose prompt
            # is unchanged, and ask the model only about targets whose candidates or
            # signals changed. Rounds before prompt fingerprints were kept fall back to
            # comparing candidate pairs and rule scores.
            old_fp = _read_json(rounds_dir / f"round-{earlier}.prompts.json")
            old = group_by_adjudication(
                _read_adjudications(rounds_dir / f"round-{earlier}.adjudications.jsonl")
            )
            same = set()
            for (lt, k), g in groups.items():
                key = f"{lt}|{k}"
                if old_fp is not None:
                    if old_fp.get(key) == fingerprints[key]:
                        same.add((lt, k))
                elif sorted((c.pair, c.score) for c in g) == sorted(
                    (a.pair, a.rule_score) for a in old.get((lt, k), [])
                ):
                    same.add((lt, k))
            reused = [a for key in sorted(same) for a in old[key]]
            todo = [c for c in gold_cands if (c.link_type, c.to_key) not in same]
            reused_from = earlier
        adjs = reused + (adjudicator.adjudicate(todo) if todo else [])
        with saved.open("w") as f:
            for a in adjs:
                f.write(json.dumps(a.as_dict()) + "\n")
        (rounds_dir / f"round-{n}.prompts.json").write_text(json.dumps(fingerprints, indent=0))
        groups_asked = len({(c.link_type, c.to_key) for c in todo})
        model_cost = client.log.total_cost_usd

    # Only the scoring, reviewers and pattern confirmation below read the truth.
    truth = load_truth(run_dir)
    reviewers = {r.name: r for r in default_reviewers(truth, log=log)}
    patterns_path = reports / "patterns.json"
    if final and patterns_from and not rescore:
        # Carry profile A's patterns over unchanged, and keep a copy beside B's reports.
        patterns_path.write_text(Path(patterns_from).read_text())
    if final:
        patterns = [Pattern(**p) for p in _read_patterns(patterns_path)]
    else:
        patterns = [Pattern(**p) for p in _read_patterns(patterns_path) if p["round"] < n]
    new_patterns: list[Pattern] = []
    filtered: list[dict] = []
    unsure_targets: dict[str, int] = {}
    induce_notes: dict[str, str] = {}
    if n >= 1 and not rescore and not final:
        current = apply_patterns(adjs, patterns, cands, connectors)
        labels = dev_labels(current, gold, truth, reviewers)
        dev_targets = {(g.link_type, g.to_key) for g in gold if g.split == "dev"}
        for lt in LINK_TYPES:
            evidence = evidence_for(lt, current, labels[lt][0], cands, connectors)
            unsure_targets[lt] = evidence["unsure_targets"]
            for p in propose(
                client, evidence, patterns + new_patterns, n, cands, filtered, induce_notes
            ):
                p.examples = [
                    list(e) for e in examples_for(p, current, cands, connectors, dev_targets)
                ]
                reviewer = reviewers[REVIEWER_FOR[lt]]
                answer = reviewer.confirm_pattern(p.id, [tuple(e) for e in p.examples])
                p.reviewer = reviewer.reviewer_id
                p.wrong_examples = [list(e) for e in reviewer.wrong_examples(p.examples)]
                p.status = {True: "confirmed", False: "rejected", None: "skipped"}[answer]
                new_patterns.append(p)
        patterns_path.write_text(
            json.dumps([p.as_dict() for p in patterns + new_patterns], indent=1)
        )
    elif rescore and not final:
        new_patterns = [Pattern(**p) for p in _read_patterns(patterns_path) if p["round"] == n]
    all_patterns = patterns + new_patterns
    final_adjs = apply_patterns(adjs, all_patterns, cands, connectors)
    if rescore:
        induce_cost = prior.get("induce_cost_usd", 0.0)
    else:
        induce_cost = client.log.total_cost_usd - model_cost
    cost = model_cost + induce_cost
    scores = score_round(final_adjs, gold, truth, reviewers, cost)
    meta = {
        "model": adjs[0].model if adjs else client.role_models["adjudicate"],
        "backend": "replayed" if rescore else type(client.backend).__name__,
        "prompt_hash": prompt_hash,
        "answers_from_round": reused_from,
        "groups_asked": groups_asked,
        "model_cost_usd": round(model_cost, 4),
        "induce_cost_usd": round(induce_cost, 4),
        "patterns_in_force": [p.id for p in all_patterns if p.status == "confirmed"],
        "patterns_this_round": [p.as_dict() for p in new_patterns],
        "patterns_filtered": filtered if not rescore else prior.get("patterns_filtered", []),
        "induce_notes": induce_notes if not rescore else prior.get("induce_notes", {}),
        "induction_unsure_targets": unsure_targets
        if not rescore
        else prior.get("induction_unsure_targets", {}),
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
    ap.add_argument(
        "--patterns-from",
        help="with --final: the profile A patterns.json to apply (no new patterns are induced)",
    )
    args = ap.parse_args(argv)
    report = run_round(
        args.run,
        args.round,
        args.per_type,
        args.max_cost,
        args.workers,
        rescore=args.rescore,
        final=args.final,
        patterns_from=args.patterns_from,
    )
    print(to_markdown(report["run"], report["round"], report["scores"], report["meta"]))


if __name__ == "__main__":
    main()
