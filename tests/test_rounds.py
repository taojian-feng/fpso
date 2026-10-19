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
from fpso.resolution.induce import decided_for_prompt
from fpso.resolution.patterns import Pattern, resolve
from fpso.synthetic.export import export
from fpso.synthetic.truth import build_world

COMPONENT_RULE = {
    "statement": "Each PO line buys for one work-order component, delivered 5-25 days early.",
    "from_capacity": "one",
    "from_qty_field": "",
    "to_qty_field": "",
    "gap_signal": "need_minus_delivery_days",
    "gap_low": 5,
    "gap_high": 25,
    "rationale": "test",
}


def responder(model, prompt, schema):
    """Rule-score adjudication that leaves look-alikes unsure, and one component pattern."""
    body = json.loads(prompt)
    if "patterns" in schema["properties"]:
        return {"patterns": [COMPONENT_RULE] if body["link_type"] == "component_po_line" else []}
    out = rule_responder(model, prompt, schema)
    for d, c in zip(out["decisions"], body["candidates"], strict=True):
        if 0.35 <= c["rule_score"] < 0.6:
            d["verdict"], d["confidence"] = "unsure", 0.5
    return out


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
    client = ModelClient(StubBackend(responder=responder))
    r0 = run_round(run, 0, per_type=20, client=client)
    rounds = run / "reports" / "rounds"
    assert (rounds / "round-0.md").read_text().startswith("# Round 0")
    gold_before = (run / "reports" / "gold.json").read_text()
    stub1 = StubBackend(responder=responder)
    r1 = run_round(run, 1, per_type=50, client=ModelClient(stub1))
    # Round 1 reuses round 0's answers and only calls the model to induce patterns.
    assert stub1.calls and all(c["model"] == "claude-opus-5-5" for c in stub1.calls)
    assert r1["meta"]["answers_from_round"] == 0
    proposed = r1["meta"]["patterns_this_round"]
    assert [p["link_type"] for p in proposed] == ["component_po_line"]
    assert proposed[0]["status"] in {"confirmed", "rejected", "skipped"}
    assert proposed[0]["examples"]
    wrong = proposed[0]["wrong_examples"]
    assert all(tuple(e) not in load_truth(run) for e in wrong)
    assert all(e in proposed[0]["examples"] for e in wrong)
    assert "## Patterns" in (rounds / "round-1.md").read_text()
    saved = json.loads((run / "reports" / "patterns.json").read_text())
    assert [p["id"] for p in saved] == [p["id"] for p in proposed]
    # The next round's inducer sees the decision and the examples marked wrong, and the
    # same rule is not proposed twice.
    decided = decided_for_prompt([Pattern(**p) for p in saved], "component_po_line")
    assert decided[0]["status"] == proposed[0]["status"]
    assert len(decided[0]["examples_marked_wrong"]) == min(len(wrong), 5)
    stub2 = StubBackend(responder=responder)
    r2 = run_round(run, 2, per_type=50, client=ModelClient(stub2))
    assert stub2.calls and r2["meta"]["patterns_this_round"] == []
    # Re-running round 1 replaces its own patterns instead of piling them up.
    run_round(run, 1, client=ModelClient(StubBackend(responder=responder)))
    assert len(json.loads((run / "reports" / "patterns.json").read_text())) == len(saved)
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


def _assignment_accuracy(run, pattern):
    """Assigned precision and coverage, and how weak (tied) pairs fare against chance."""
    connectors = open_all(run / "extracts")
    cands = [
        c for c in CandidateGenerator(connectors).generate() if c.link_type == pattern.link_type
    ]
    decided = resolve(pattern, cands, connectors)
    truth = load_truth(run)
    weak = {c.pair for c in cands if c.score < 0.9}
    assigned = [p for p, ok in decided.items() if ok]
    weak_assigned = [p for p in assigned if p in weak]
    ruled_out = [p for p, ok in decided.items() if not ok]
    lt_truth = {p for p in truth if p[0] == pattern.link_type}

    def share(pairs):
        return sum(p in truth for p in pairs) / len(pairs)

    return {
        "precision": share(assigned),
        "coverage": len(set(assigned) & lt_truth) / len(lt_truth),
        "weak_chance": share(weak),  # a weak pair picked at random is true this often
        "weak_assigned": share(weak_assigned),
        "ruled_out_true": share(ruled_out),
    }


def _assert_breaks_ties(acc):
    assert acc["precision"] > 0.85 and acc["coverage"] > 0.6
    # On tied look-alikes the pattern does well over twice as well as chance, and the
    # pairs it rules out are true less often than chance.
    assert acc["weak_assigned"] > 2 * acc["weak_chance"]
    assert acc["ruled_out_true"] < acc["weak_chance"]


def test_one_to_one_assignment_breaks_component_ties(run_dirs):
    p = Pattern(
        "t1",
        "component_po_line",
        "test",
        "one",
        gap_signal="need_minus_delivery_days",
        gap_low=5,
        gap_high=25,
    )
    _assert_breaks_ties(_assignment_accuracy(run_dirs["A"], p))


def test_quantity_capacity_breaks_manifest_ties(run_dirs):
    p = Pattern(
        "t2",
        "po_line_manifest_line",
        "test",
        "quantity",
        "MENGE",
        "qty",
        gap_signal="ship_lag_days",
        gap_low=1,
        gap_high=8,
    )
    _assert_breaks_ties(_assignment_accuracy(run_dirs["A"], p))
