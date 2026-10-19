import json

import pytest

from fpso.connectors import open_all
from fpso.eval.adjudication import run_adjudication, sample_targets
from fpso.eval.harness import load_truth
from fpso.models import ModelClient, StubBackend
from fpso.onboarding import onboard, to_markdown
from fpso.resolution.adjudicate import SCHEMA, Adjudicator
from fpso.resolution.calibrate import (
    Calibrator,
    choose_threshold,
    expected_calibration_error,
    isotonic_fit,
)
from fpso.resolution.candidates import LINK_TYPES, CandidateGenerator
from fpso.synthetic.export import export
from fpso.synthetic.truth import build_world


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("adj")
    export(build_world(seed=11, scale=0.08), d, "A")
    return d


def rule_responder(model, prompt, schema):
    """Answers like a model that trusts the rule score: enough to exercise the pipeline."""
    body = json.loads(prompt)
    return {
        "decisions": [
            {
                "id": c["id"],
                "verdict": "match" if c["rule_score"] >= 0.6 else "no_match",
                "confidence": c["rule_score"],
                "evidence": "rule score",
            }
            for c in body["candidates"]
        ]
    }


def test_isotonic_is_monotone_and_pools_ties():
    uppers, values = isotonic_fit([0.1, 0.2, 0.2, 0.3, 0.9], [0, 1, 0, 0, 1])
    assert values == sorted(values)
    assert uppers[-1] == 0.9
    cal = Calibrator.fit([0.1, 0.2, 0.2, 0.3, 0.9], [False, True, False, False, True])
    assert cal.predict(0.9) == 1.0
    assert cal.predict(0.2) == cal.predict(0.3)  # 0.2 (0.5) and 0.3 (0) violate, so pool


def test_threshold_and_ece():
    probs = [0.99, 0.98, 0.97, 0.6, 0.5, 0.1]
    labels = [True, True, True, False, True, False]
    assert choose_threshold(probs, labels, 0.95) == 0.97
    assert choose_threshold([0.9], [False]) > 1
    assert expected_calibration_error([1.0, 0.0], [True, False]) == 0.0
    assert expected_calibration_error([0.9, 0.9], [False, False]) == pytest.approx(0.9)


def test_adjudicator_prompt_and_missing_decisions(run_dir):
    connectors = open_all(run_dir / "extracts")
    cands = [c for c in CandidateGenerator(connectors).generate() if c.link_type == "po_email"]
    group = [c for c in cands if c.to_key == cands[0].to_key]
    stub = StubBackend()
    adj = Adjudicator(connectors, ModelClient(stub)).adjudicate_group(
        "po_email", group[0].to_key, group
    )
    body = json.loads(stub.calls[0]["prompt"])
    assert body["target"]["key"] == group[0].to_key
    assert [c["id"] for c in body["candidates"]] == list(range(len(group)))
    assert all(a.verdict == "unsure" and a.confidence == 0.5 for a in adj)
    assert "default" in SCHEMA  # stripped by real backends, used by the stub


def test_cost_cap_stops_new_calls(run_dir):
    connectors = open_all(run_dir / "extracts")
    cands = sample_targets(CandidateGenerator(connectors).generate(), 5)

    def pricey(model, prompt, schema):
        return rule_responder(model, prompt, schema)

    class Pricey(StubBackend):
        def complete_json(self, model, system, prompt, schema, effort):
            r = super().complete_json(model, system, prompt, schema, effort)
            r.input_tokens, r.output_tokens = 1_000_000, 0  # $2 per call on the mid model
            return r

    adjudicator = Adjudicator(
        connectors, ModelClient(Pricey(responder=pricey)), max_cost_usd=3.0, workers=1
    )
    adjudicator.adjudicate(cands)
    assert adjudicator.skipped_groups == 25 - 2


def test_pipeline_end_to_end_on_stub(run_dir, tmp_path):
    report = run_adjudication(
        run_dir,
        sample=30,
        client=ModelClient(StubBackend(responder=rule_responder)),
        out_dir=tmp_path,
    )
    assert (tmp_path / "adjudication.md").read_text().startswith("# Adjudication report")
    assert (tmp_path / "adjudications.jsonl").read_text().count("\n") > 30
    assert any("sim:" in line for line in (tmp_path / "actions.jsonl").read_text().splitlines())
    truth = load_truth(run_dir)
    for lt in LINK_TYPES:
        s = report["scores"][lt]
        assert s["targets"] > 0 and s["calibration_labels"] > 0, lt
        assert s["test_true_links"] <= sum(p[0] == lt for p in truth)
        assert s["recall"] is None or 0 <= s["recall"] <= 1


def test_onboarding_draft_renders(run_dir):
    connector = open_all(run_dir / "extracts")["erp-mm"]
    draft = {
        "summary": "SAP MM extract.",
        "entities": [
            {
                "entity_type": "po_line",
                "description": "Purchase order line.",
                "likely_keys": ["EBELN", "EBELP"],
                "fields": [{"name": "EINDT", "meaning": "delivery date"}],
            }
        ],
        "link_types": [
            {"link_type": "po_email", "entity_type": "po", "join_hint": "PO number in subject"}
        ],
    }
    stub = StubBackend(responder=lambda m, p, s: draft)
    md = to_markdown("erp-mm", onboard(connector, ModelClient(stub)))
    assert "## po_line" in md and "`EINDT`" in md and "po_email" in md
    assert stub.calls[0]["model"] == "claude-sonnet-5-5"
    sent = json.loads(stub.calls[0]["prompt"])["entity_types"]
    assert {t["entity_type"] for t in sent} == set(connector.entity_types())
