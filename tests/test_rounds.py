import json

import pytest
from test_adjudication import rule_responder

from fpso.connectors import open_all
from fpso.eval.harness import load_truth
from fpso.eval.reviewers import SimulatedReviewer
from fpso.eval.rounds import build_gold, label_dev, run_round
from fpso.models import ModelClient, StubBackend
from fpso.resolution.adjudicate import Adjudication
from fpso.resolution.candidates import LINK_TYPES, CandidateGenerator
from fpso.synthetic.export import export
from fpso.synthetic.truth import build_world


@pytest.fixture(scope="module")
def run_dirs(tmp_path_factory):
    out = {}
    for profile in ("A", "B"):
        d = tmp_path_factory.mktemp(f"round{profile}")
        export(build_world(seed=13, scale=0.08), d, profile)
        out[profile] = d
    return out


def test_gold_is_stratified_split_and_deterministic(run_dirs):
    cands = CandidateGenerator(open_all(run_dirs["A"] / "extracts")).generate()
    gold = build_gold(cands, per_type=20)
    assert gold == build_gold(cands, per_type=20)
    for lt in LINK_TYPES:
        g = [x for x in gold if x.link_type == lt]
        assert 0 < len(g) <= 20
        assert len({x.to_key for x in g}) == len(g)
        dev = [x for x in g if x.split == "dev"]
        assert 0 < len(dev) < len(g)
    assert {x.stratum for x in gold} == {"easy", "hard"}


def test_double_labels_drop_disagreements_and_report_accuracy():
    truth = {("t", str(i), "x") for i in range(300)}
    rows = [Adjudication("t", str(i), "x", 0.5, "match", 0.95, "") for i in range(300)]
    rows += [Adjudication("t", str(i), "y", 0.1, "no_match", 0.05, "") for i in range(300)]
    a = SimulatedReviewer("a", "planner", truth, error_rate=0.05, skip_rate=0.1)
    b = SimulatedReviewer("b", "materials", truth, error_rate=0.05, skip_rate=0.1)
    labelled, stats = label_dev(rows, a, b, truth)
    # Every high-confidence pair is double-labelled, so wrong "no" answers there mostly
    # drop out instead of reaching calibration.
    high = [y for adj, y in labelled if adj.confidence >= 0.9]
    assert sum(high) / len(high) > 0.995
    assert stats["disagreements_dropped"] > 0
    assert stats["unconfirmed_dropped"] > 0  # a skip leaves a high-confidence pair unconfirmed
    assert 0.85 < stats["agreement_on_random_double_labels"] < 0.97
    assert 0.92 < stats["primary_accuracy_vs_truth"] < 0.98


def test_round_writes_versioned_reports_and_reuses_gold(run_dirs):
    run = run_dirs["A"]
    client = ModelClient(StubBackend(responder=rule_responder))
    r0 = run_round(run, 0, per_type=20, client=client)
    rounds = run / "reports" / "rounds"
    assert (rounds / "round-0.md").read_text().startswith("# Round 0")
    gold_before = (run / "reports" / "gold.json").read_text()
    r1 = run_round(run, 1, per_type=50, client=ModelClient(StubBackend(responder=rule_responder)))
    assert (run / "reports" / "gold.json").read_text() == gold_before  # fixed across rounds
    assert r1["meta"]["gold_heldout"] == r0["meta"]["gold_heldout"]
    summary = (rounds / "summary.md").read_text()
    assert "| 0 |" in summary and "| 1 |" in summary
    truth = load_truth(run)
    for lt in LINK_TYPES:
        s = r0["scores"][lt]
        assert s["test_true_links"] <= sum(p[0] == lt for p in truth)
        assert s["reviewers"]["labels_used"] > 0
    stub = ModelClient(StubBackend())
    again = run_round(run, 0, per_type=20, client=stub, rescore=True)
    assert stub.backend.calls == []
    assert again["scores"] == r0["scores"]
    assert json.loads((rounds / "round-0.json").read_text())["meta"]["backend"] == "replayed"


def test_profile_b_is_held_back(run_dirs):
    with pytest.raises(SystemExit):
        run_round(run_dirs["B"], 0, per_type=5, client=ModelClient(StubBackend()))
