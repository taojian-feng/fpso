import csv
import json
from datetime import timedelta
from pathlib import Path

import pytest

from fpso.connectors.sources import open_all
from fpso.synthetic.export import export
from fpso.synthetic.replay import MANIFEST_LEAD_DAYS, as_of_view
from fpso.synthetic.truth import build_world


def _rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("A")
    world = build_world(seed=11, scale=0.1)
    export(world, d, profile="A")
    return d, world


def test_replay_hides_what_did_not_exist_yet(run, tmp_path):
    d, world = run
    as_of = world.turnaround_start - timedelta(weeks=8)
    as_of_view(d, as_of, tmp_path / "day")
    con = open_all(tmp_path / "day" / "extracts")

    po_lines = {r.key for r in con["erp-mm"].iter("po_line")}
    assert po_lines == {pl.key for pl in world.po_lines if pl.order_date <= as_of}
    assert 0 < len(po_lines) < len(world.po_lines)

    messages = {r.key for r in con["supplier-mail"].iter("message")}
    assert messages == {e.message_id for e in world.emails if e.sent <= as_of}

    depart = {s.sailing_id: s.depart for s in world.sailings}
    for r in con["vessel"].iter("manifest_line"):
        assert depart[r.fields["sailing_id"]] <= as_of + timedelta(days=MANIFEST_LEAD_DAYS)

    # Whole sources stay whole.
    assert len(list(con["erp-pm"].iter("work_order"))) == len(world.work_orders)
    assert len(list(con["vessel"].iter("sailing"))) == len(world.sailings)


def test_replay_truth_only_links_visible_records(run, tmp_path):
    d, world = run
    as_of = world.window_start + timedelta(days=30)
    kept = as_of_view(d, as_of, tmp_path / "day")
    links = _rows(tmp_path / "day" / "truth" / "links.csv")
    po_lines = {
        f"{r['EBELN']}-{r['EBELP']}"
        for r in _rows(tmp_path / "day" / "extracts" / "erp_mm" / "po_lines.csv")
    }
    for lk in links:
        if lk["link_type"] == "component_po_line":
            assert lk["to_key"] in po_lines
    assert kept["links"] == len(links) < len(_rows(d / "truth" / "links.csv"))
    run_json = json.loads((tmp_path / "day" / "truth" / "run.json").read_text())
    assert run_json["as_of"] == as_of.isoformat()
    assert run_json["profile"] == "A"


def test_replay_after_the_window_keeps_everything(run, tmp_path):
    d, world = run
    as_of_view(d, world.window_end + timedelta(days=365), tmp_path / "day")
    for rel in (
        "extracts/erp_mm/po_lines.csv",
        "extracts/vessel/manifest_lines.csv",
        "truth/links.csv",
    ):
        assert _rows(tmp_path / "day" / rel) == _rows(d / rel)
