import json
import re
from datetime import date, datetime

import pytest

from fpso.connectors import open_all
from fpso.eval.harness import load_truth
from fpso.eval.reviewers import default_reviewers
from fpso.eval.rounds import run_round
from fpso.models import ModelClient, StubBackend
from fpso.readiness import (
    Facts,
    ReadinessAgent,
    assess,
    load_world,
    report_dates,
    run_readiness,
    score_report,
    truth_facts,
)
from fpso.synthetic.export import export
from fpso.synthetic.truth import build_world


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("readiness")
    export(build_world(seed=11, scale=0.15), d, "A")
    return d


@pytest.fixture(autouse=True)
def no_kept_answers(run_dir):
    """Each test starts with no answers kept from an earlier run."""
    (run_dir / "reports" / "readiness" / "answers.jsonl").unlink(missing_ok=True)


def read_email_like_a_person(prompt):
    """What a careful reader gets from a slip notice; stands in for the extraction model."""
    m = re.search(r"New expected dispatch: (.+)", prompt)
    if not m:
        return {"is_delay": False, "new_date": "", "parts": []}
    new = datetime.strptime(m.group(1).strip(), "%d %b %Y").date().isoformat()
    return {"is_delay": True, "new_date": new, "parts": re.findall(r"^\s*-\s+(\S+)", prompt, re.M)}


def test_gap_order_and_unverified():
    need = date(2027, 4, 10)
    f = Facts("c", "wo", "m", 2, need, stock=0)
    assert assess(f) == "no_supply"
    f.unsettled.add("supply")
    assert assess(f) == "unverified"  # an open link is a question, not a gap
    f = Facts("c", "wo", "m", 2, need, stock=5)
    assert assess(f) is None  # in stock, nothing on order
    f.unsettled |= {"supply", "near_match"}
    assert assess(f) == "unverified"  # the model half-matched a PO line: may be its supply
    f = Facts("c", "wo", "m", 2, need, stock=0, po_line="p", delivery=date(2027, 4, 1))
    assert assess(f) == "not_shipped"
    f.shipments = [(date(2027, 4, 5), 1, "ML-1"), (date(2027, 4, 12), 1, "ML-2")]
    assert assess(f) == "not_shipped"  # the second part sails too late
    f.shipments[1] = (date(2027, 4, 8), 1, "ML-2")
    assert assess(f) is None
    f.slips = [(date(2027, 4, 20), "msg-1")]
    assert assess(f) == "slipped"
    f.delivery = date(2027, 4, 11)
    assert assess(f) == "late_po"


def test_with_true_links_the_agent_matches_the_truth(run_dir):
    """The walk and the truth side agree when links and emails are read right, so the
    readiness metric measures linking and reading, not bookkeeping differences."""
    world = load_world(run_dir)
    truth = load_truth(run_dir)
    agent = ReadinessAgent(open_all(run_dir / "extracts"), None)

    def oracle(targets, urgency=None):
        for lt, k in targets:
            agent.links[(lt, k)] = [(t[1], 1.0, "oracle") for t in truth if t[:3:2] == (lt, k)]

    agent.resolve = oracle
    mail = agent.c["supplier-mail"]
    agent.read_email = lambda key: read_email_like_a_person(mail.get("message", key).fields["body"])
    planner = default_reviewers(set())[0]
    for d in report_dates(world.turnaround_start):
        s = score_report(agent.facts(d), truth_facts(world, d), planner, d)
        assert s["gaps_true"] > 0
        assert s["agreement"] > 0.95 and s["gap_precision"] > 0.9 and s["gap_recall"] > 0.9


def test_reports_are_written_and_profile_b_is_held_back(run_dir, tmp_path):
    def responder(model, prompt, schema):
        if "is_delay" in schema["properties"]:
            return read_email_like_a_person(prompt)
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

    # A round's saved answers are reused where the prompt is unchanged.
    run_round(run_dir, 0, per_type=10, client=ModelClient(StubBackend(responder=responder)))
    stub = StubBackend(responder=responder)
    run = json.loads((run_dir / "truth" / "run.json").read_text())
    dates = report_dates(date.fromisoformat(run["turnaround"][0]))
    reports = run_readiness(run_dir, dates, client=ModelClient(stub))
    assert [r["meta"]["as_of"] for r in reports] == [d.isoformat() for d in dates]
    assert any(c["model"] == "claude-haiku-5-5" for c in stub.calls)  # emails read
    assert reports[0]["meta"]["answers_from_round"] == 0
    assert reports[0]["meta"]["links"]["model_reused"] > 0
    for r in reports:
        s = r["scores"]
        assert s["work_packages"] > 0 and s["gap_precision"] > 0.8
        md = (run_dir / "reports" / "readiness" / f"{r['meta']['as_of']}.md").read_text()
        assert md.startswith("# Readiness report") and "| Work order |" in md
        for g in r["gaps"]:
            assert g["component"] and g["gap"] != "unverified"
    no_model = run_readiness(run_dir, dates[:1], use_model=False)
    assert no_model[0]["meta"]["cost_usd"] == 0
    b = tmp_path / "b"
    export(build_world(seed=11, scale=0.05), b, "B")
    with pytest.raises(SystemExit):
        run_readiness(b, dates[:1], use_model=False)


def test_cost_cap_settles_the_soonest_work_first(run_dir):
    run = json.loads((run_dir / "truth" / "run.json").read_text())
    d = report_dates(date.fromisoformat(run["turnaround"][0]))[0]
    full = run_readiness(run_dir, [d], client=ModelClient(StubBackend()))[0]
    (run_dir / "reports" / "readiness" / "answers.jsonl").unlink()
    capped = run_readiness(run_dir, [d], max_cost_usd=0.02, client=ModelClient(StubBackend()))[0]
    assert capped["meta"]["run_cost_usd"] < full["meta"]["run_cost_usd"]
    settled = capped["meta"]["model_settled_through"]
    assert settled is not None and settled < full["meta"]["model_settled_through"]
    # Whatever the cap left open is a question for the planner, never a guessed gap.
    assert capped["scores"]["unverified"] >= full["scores"]["unverified"]
    assert capped["scores"]["gap_precision"] >= full["scores"]["gap_precision"] - 0.05


def test_the_cost_cap_is_shared_between_report_dates(run_dir):
    run = json.loads((run_dir / "truth" / "run.json").read_text())
    dates = report_dates(date.fromisoformat(run["turnaround"][0]))
    reports = run_readiness(run_dir, dates, max_cost_usd=0.02, client=ModelClient(StubBackend()))
    first, second = (r["meta"] for r in reports)
    assert first["run_cost_usd"] < 0.02 / 2 + 0.005  # the first date keeps to its half
    assert second["cost_usd"] > 0


def test_a_rerun_pays_only_for_new_questions(run_dir):
    run = json.loads((run_dir / "truth" / "run.json").read_text())
    d = report_dates(date.fromisoformat(run["turnaround"][0]))[0]
    capped = run_readiness(run_dir, [d], max_cost_usd=0.02, client=ModelClient(StubBackend()))[0]
    again = run_readiness(run_dir, [d], max_cost_usd=0.02, client=ModelClient(StubBackend()))[0]
    assert again["meta"]["answers_kept"] > 0
    # Kept answers cost nothing, so the same cap now gets through more questions.
    assert again["meta"]["links"]["model_asked"] > capped["meta"]["links"]["model_asked"]
