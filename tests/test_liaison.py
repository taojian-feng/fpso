from datetime import date, timedelta

from fpso.connectors import open_all
from fpso.liaison import Chase, LiaisonAgent, email_kind, score_chases, truth_unacknowledged
from fpso.procurement import Recommendation
from fpso.readiness import ReadinessAgent
from fpso.schedule import Impact
from fpso.synthetic.export import export
from fpso.synthetic.truth import build_world


def test_email_kind_reads_every_generated_email(tmp_path):
    world = build_world(seed=11, scale=0.05)
    export(world, tmp_path, "B")  # B quotes PO numbers and swaps letters most
    kind_of = {e.message_id: e.kind for e in world.emails}
    con = open_all(tmp_path / "extracts")
    for msg in con["supplier-mail"].iter("message"):
        assert email_kind(msg.fields["subject"], msg.fields["body"]) == kind_of[msg.key]


def test_no_ack_chases_match_the_truth_without_a_model(tmp_path):
    world = build_world(seed=11, scale=0.1)
    export(world, tmp_path, "A")
    d = world.window_start + timedelta(days=40)
    agent = LiaisonAgent(ReadinessAgent(open_all(tmp_path / "extracts"), None))
    chases = agent.run(d, [], [])
    s = score_chases(chases, truth_unacknowledged(world, d))
    assert s["no_ack_reported"] > 0
    assert s["no_ack_precision"] >= 0.9
    assert all(c.to.startswith("orders@") and c.po in c.subject for c in chases)


def test_expedite_draft_groups_lines_by_po_and_names_the_turnaround_impact(tmp_path):
    world = build_world(seed=11, scale=0.05)
    export(world, tmp_path, "A")
    pl1, pl2 = next(
        (a, b)
        for a in world.po_lines
        for b in world.po_lines
        if a.po_no == b.po_no and a.item < b.item
    )
    need = date(2027, 4, 1)
    recs = [
        Recommendation(pl.component_key, pl.wo_no, pl.mat_no, need, "expedite", po_line=pl.key)
        for pl in (pl1, pl2)
    ]
    impacts = [Impact(pl1.component_key, pl1.wo_no, 9, "KST-TA-1010", 9, [], 5)]
    agent = LiaisonAgent(ReadinessAgent(open_all(tmp_path / "extracts"), None))
    drafts = [c for c in agent.run(world.window_start, recs, impacts) if c.kind == "expedite"]
    assert len(drafts) == 1
    d: Chase = drafts[0]
    assert d.po == pl1.po_no and d.po_lines == sorted([pl1.key, pl2.key])
    assert "turnaround end moves 5 days" in d.body
