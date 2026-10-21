import csv

import pytest

from fpso.connectors import open_all
from fpso.eval.harness import load_truth, run_candidates, score_candidates
from fpso.eval.reviewers import SimulatedReviewer, default_reviewers
from fpso.resolution.candidates import LINK_TYPES, CandidateGenerator
from fpso.resolution.normalize import (
    containment,
    find_po_numbers,
    find_tags,
    norm_tag,
    parse_date,
    tag_group,
)
from fpso.synthetic.export import export
from fpso.synthetic.truth import build_world


@pytest.fixture(scope="module", params=["A", "B"])
def run_dir(request, tmp_path_factory):
    d = tmp_path_factory.mktemp(f"res{request.param}")
    export(build_world(seed=21, scale=0.1), d, request.param)
    return d


def test_normalizers():
    assert norm_tag("p-2301-a") == norm_tag("P 2301A") == norm_tag("P–2301A") == "P2301A"
    assert tag_group("P-2301B") == "P2301"
    assert find_tags("E-2393A High Vib (WO 41002715) on KST-MNT-2010") == {"E2393A"}
    assert find_po_numbers("PO#004500210013 and 4500210030-10") == {"4500210013", "4500210030"}
    assert str(parse_date("20270104")) == str(parse_date("2027-01-04 07:00")) == "2027-01-04"
    assert containment("STEM PKG SET", "STEM PACKING SET; VALVE") > 0.4


def test_candidate_recall_is_high_on_every_link_type(run_dir):
    cands = CandidateGenerator(open_all(run_dir / "extracts")).generate()
    scores = score_candidates(cands, load_truth(run_dir))
    assert set(scores) == set(LINK_TYPES)
    for lt, s in scores.items():
        assert s["true_links"] > 0, lt
        assert s["candidate_recall"] >= 0.95, (lt, s)


def test_candidates_never_read_truth(run_dir, monkeypatch):
    import builtins

    real_open = builtins.open

    def guarded(path, *a, **k):
        assert "truth" not in str(path), f"candidate generation opened {path}"
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded)
    CandidateGenerator(open_all(run_dir / "extracts")).generate()


def test_harness_writes_reports(run_dir, tmp_path):
    report = run_candidates(run_dir, tmp_path)
    assert (tmp_path / "candidates.md").read_text().startswith("# Candidate report")
    assert report["scores"]["po_email"]["candidate_recall"] >= 0.95


def test_simulated_reviewer_is_noisy_but_mostly_right_and_stable():
    truth = {("t", str(i), "x") for i in range(500)}
    falses = [("t", str(i), "y") for i in range(500)]
    r = SimulatedReviewer("dana.reyes", "planner", truth, error_rate=0.05, skip_rate=0.1)
    answers = [r.label(p) for p in sorted(truth)] + [r.label(p) for p in falses]
    answered = [(a, i < 500) for i, a in enumerate(answers) if a is not None]
    skipped = answers.count(None) / len(answers)
    accuracy = sum(a == correct for a, correct in answered) / len(answered)
    assert 0.05 < skipped < 0.15
    assert 0.92 < accuracy < 0.98
    assert r.label(falses[0]) == r.label(falses[0])
    assert all(rec.agent == "sim:dana.reyes" for rec in r.log.records)


def test_pattern_confirmation_follows_example_quality():
    truth = {("t", str(i), "x") for i in range(20)}
    good = sorted(truth)
    bad = good[:10] + [("t", str(i), "y") for i in range(10)]
    r = SimulatedReviewer("lee.park", "materials", truth, error_rate=0.0, skip_rate=0.0)
    assert r.confirm_pattern("p1", good) is True
    assert r.confirm_pattern("p2", bad) is False
    assert len(default_reviewers(truth)) == 2


def test_reference_signals_point_at_true_links(run_dir):
    cands = CandidateGenerator(open_all(run_dir / "extracts")).generate()
    truth = load_truth(run_dir)

    def precision(fired):
        hits = [c for c in cands if fired(c.signals)]
        assert hits
        return sum(c.pair in truth for c in hits) / len(hits)

    # A supplier's sales-order reference and a manifest's cited PO item name the link.
    assert precision(lambda s: "sales_order" in s) > 0.9
    assert precision(lambda s: s.get("po_item") is True) > 0.9
    assert precision(lambda s: s.get("po_item") is False) < 0.1
    # Weaker: the equipment a manifest names, and exact item quantities in an email.
    assert precision(lambda s: s.get("manifest_work_orders") == 1) > 0.7  # few cases
    assert precision(lambda s: s.get("item_qty_match") == 1.0) > 0.6


def test_profile_c_fields_make_decisive_candidates(tmp_path):
    export(build_world(seed=11, scale=0.05), tmp_path, "C")
    cands = CandidateGenerator(open_all(tmp_path / "extracts")).generate()
    with (tmp_path / "truth" / "links.csv").open() as f:
        truth = {(r["link_type"], r["from_key"], r["to_key"]) for r in csv.DictReader(f)}
    for sig in ("reservation", "po_item_ref"):
        hits = [c for c in cands if sig in c.signals]
        assert hits and all(c.score >= 0.9 for c in hits)
        assert all(c.pair in truth for c in hits), sig
