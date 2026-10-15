"""Score candidate generation (and later, resolved links) against a run's hidden truth.

    fpso-eval --run data/kestrel-A

Writes reports/candidates.json and reports/candidates.md under the run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from fpso.connectors import open_all
from fpso.resolution.candidates import LINK_TYPES, Candidate, CandidateGenerator

Pair = tuple[str, str, str]
BASELINE_THRESHOLD = 0.9


def load_truth(run_dir: str | Path) -> set[Pair]:
    with (Path(run_dir) / "truth" / "links.csv").open() as f:
        return {(r["link_type"], r["from_key"], r["to_key"]) for r in csv.DictReader(f)}


def score_candidates(cands: list[Candidate], truth: set[Pair]) -> dict[str, dict]:
    got = {c.pair for c in cands}
    accepted = {c.pair for c in cands if c.score >= BASELINE_THRESHOLD}
    per_target = defaultdict(set)
    for c in cands:
        per_target[c.link_type].add(c.to_key)
    n_cands = Counter(c.link_type for c in cands)
    out = {}
    for lt in LINK_TYPES:
        t = {p for p in truth if p[0] == lt}
        g = {p for p in got if p[0] == lt}
        a = {p for p in accepted if p[0] == lt}
        out[lt] = {
            "true_links": len(t),
            "candidates": n_cands[lt],
            "candidate_recall": round(len(t & g) / len(t), 4) if t else None,
            "candidates_per_target": round(n_cands[lt] / len(per_target[lt]), 2)
            if per_target[lt]
            else 0,
            "baseline_precision": round(len(a & t) / len(a), 4) if a else None,
            "baseline_recall": round(len(a & t) / len(t), 4) if t else None,
        }
    return out


def to_markdown(run: dict, scores: dict[str, dict], seconds: float) -> str:
    lines = [
        f"# Candidate report: {run['asset']}, profile {run['profile']} "
        f"(v{run['profile_version']}), seed {run['seed']}",
        "",
        f"Candidate generation took {seconds:.1f} s. The rule baseline accepts every candidate "
        f"scoring at least {BASELINE_THRESHOLD} with no model involved.",
        "",
        "| Link type | True links | Candidates | Candidate recall | Per target "
        "| Baseline precision | Baseline recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def pct(v):
        return "n/a" if v is None else f"{v:.1%}"

    for lt, s in scores.items():
        lines.append(
            f"| {lt} | {s['true_links']} | {s['candidates']} | {pct(s['candidate_recall'])} "
            f"| {s['candidates_per_target']} | {pct(s['baseline_precision'])} "
            f"| {pct(s['baseline_recall'])} |"
        )
    return "\n".join(lines) + "\n"


def run_candidates(run_dir: str | Path, out_dir: str | Path | None = None) -> dict:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "truth" / "run.json").read_text())
    t0 = time.time()
    cands = CandidateGenerator(open_all(run_dir / "extracts")).generate()
    seconds = time.time() - t0
    scores = score_candidates(cands, load_truth(run_dir))
    out = Path(out_dir) if out_dir else run_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    report = {"run": run, "seconds": round(seconds, 1), "scores": scores}
    (out / "candidates.json").write_text(json.dumps(report, indent=2))
    (out / "candidates.md").write_text(to_markdown(run, scores, seconds))
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score candidate generation against truth.")
    ap.add_argument("--run", required=True, help="a run directory written by fpso-synth")
    ap.add_argument("--out", help="report directory (default: <run>/reports)")
    args = ap.parse_args(argv)
    report = run_candidates(args.run, args.out)
    print(to_markdown(report["run"], report["scores"], report["seconds"]))


if __name__ == "__main__":
    main()
