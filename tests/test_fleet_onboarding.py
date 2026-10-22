import csv
import hashlib
import json
from pathlib import Path

import pytest

from fpso.connectors import open_all
from fpso.fleet.adapter import adapt_run, raw_sources
from fpso.fleet.assets import KESTREL, MERLIN, OSPREY, asset_world
from fpso.fleet.canonical import N_FIELDS, identity_mapping
from fpso.fleet.dialects import dialect_mapping, export_maximo
from fpso.fleet.mapper import (
    draft_mapping,
    kestrel_baseline,
    profile,
    reuse_mapping,
    score_draft,
    words,
)
from fpso.procurement import run_procurement
from fpso.synthetic.export import export
from fpso.synthetic.truth import build_world

SCALE = 0.05


def _digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def merlin(tmp_path_factory):
    raw = tmp_path_factory.mktemp("merlin") / "raw"
    export_maximo(asset_world(MERLIN, SCALE), MERLIN, raw)
    return raw


def test_asset_world_keeps_the_builders_draws():
    plain, merlin = build_world(seed=MERLIN.seed, scale=SCALE), asset_world(MERLIN, SCALE)
    assert [pl.key for pl in plain.po_lines] == [pl.key for pl in merlin.po_lines]
    assert [c.key for c in plain.components] == [c.key for c in merlin.components]
    assert all(a.act_id.startswith("MRL-") for a in merlin.activities)
    assert all(s.sailing_id.startswith("MRL-") for s in merlin.sailings)
    assert all(a.act_id.startswith("KST-") for a in plain.activities)  # the copy only
    assert asset_world(KESTREL, SCALE).activities == build_world(7, SCALE).activities


def test_kestrel_exports_unchanged_by_a_maximo_export(tmp_path):
    for profile_name in ("A", "B", "C"):
        before = tmp_path / f"before-{profile_name}"
        export(build_world(seed=3, scale=SCALE), before, profile_name)
        export_maximo(asset_world(MERLIN, SCALE), MERLIN, tmp_path / f"m-{profile_name}")
        after = tmp_path / f"after-{profile_name}"
        export(build_world(seed=3, scale=SCALE), after, profile_name)
        assert _digest(before) == _digest(after)


def test_merlin_arrives_in_another_convention(merlin):
    ex = merlin / "extracts"
    with (ex / "maximo" / "POLINE.csv").open() as f:
        row = next(csv.DictReader(f))
    assert {"PONUM", "POLINENUM", "ITEMNUM", "VENDELIVERYDATE", "REQDELIVERYDATE"} <= set(row)
    assert row["ORDERDATE"][2] == "-" and row["ORDERDATE"][3:6].isalpha()  # 09-APR-2027
    assert not (ex / "erp_mm").exists()
    xer = next((ex / "p6").glob("*.xer")).read_text()
    assert "MRL-TA27-Merlin" in xer and "MAXIMO WONUM" in xer and "SAP WO" not in xer
    run = json.loads((merlin / "truth" / "run.json").read_text())
    assert run["asset"] == "FPSO Merlin" and run["dialect"] == "maximo"


def test_words_split_raw_names_into_concepts():
    assert words("VENDELIVERYDATE") == ["VENDOR", "DELIVERY", "DATE"]
    assert words("REQUIREDATE") == ["REQUIRED", "DATE"]
    assert words("POLINENUM") == ["PURCHASEORDER", "LINE", "NO"]
    assert words("sailing_id") == ["VOYAGE", "NO"]


def test_profile_finds_kinds_and_date_formats():
    assert profile("t", "d", ["09-APR-2027", "13-MAY-2027"]).fmt == "%d-%b-%Y"
    day_first = profile("t", "d", ["01/02/2027 06:00", "25/02/2027 06:00"])
    assert day_first.kind == "date" and day_first.fmt == "%d/%m/%Y %H:%M"
    assert profile("t", "q", ["2.00", "1.00"]).kind == "int"
    assert profile("t", "w", ["2.5", "1.0"]).kind == "decimal"
    assert profile("t", "k", ["41000101", "41000107"]).kind == "id"
    assert profile("t", "e", ["orders@a.example"]).kind == "email"
    assert profile("t", "g", ["P-2301A", "XV-1105"]).kind == "tag"


def test_rule_mapper_gets_most_of_merlin_right_and_beats_the_baseline(merlin):
    draft = draft_mapping(raw_sources(merlin / "extracts"))
    s = score_draft(draft, dialect_mapping(MERLIN))
    assert s["fields"] == N_FIELDS
    assert s["auto_correct_share"] >= 0.85
    assert s["silent_errors"] <= 2
    by_field = {fd.field: fd for fd in draft.fields}
    assert by_field["po_line.EINDT"].column == "VENDELIVERYDATE"  # not the requested date
    assert by_field["sailing.depart_date"].fmt == "%d/%m/%Y %H:%M"
    assert by_field["p6.udf.SAP WO"].column == "MAXIMO WONUM"
    assert s["person_days"] < 0.5 * kestrel_baseline()["person_days"]


def test_reused_dialect_covers_osprey_and_costs_less(merlin, tmp_path):
    raw = tmp_path / "osprey"
    export_maximo(asset_world(OSPREY, SCALE), OSPREY, raw)
    merlin_score = score_draft(
        draft_mapping(raw_sources(merlin / "extracts")), dialect_mapping(MERLIN)
    )
    draft = reuse_mapping(dialect_mapping(MERLIN), raw_sources(raw / "extracts"))
    s = score_draft(draft, dialect_mapping(OSPREY))
    assert s["reused_from_library"] >= N_FIELDS - 3
    assert {fd.field for fd in draft.fields if fd.source == "mapper"} >= {
        "manifest_line.customer_ref",
        "p6.udf.SAP WO",
    }
    assert s["person_hours"] < merlin_score["person_hours"]


def test_identity_mapping_scores_perfectly_on_kestrel(tmp_path):
    export(build_world(seed=7, scale=SCALE), tmp_path, "A")
    draft = reuse_mapping(identity_mapping(), raw_sources(tmp_path / "extracts"))
    assert score_draft(draft, identity_mapping())["auto_correct_share"] == 1.0


def test_adapted_merlin_reads_through_the_usual_connectors(merlin, tmp_path):
    counts = adapt_run(merlin, dialect_mapping(MERLIN), tmp_path)
    con = open_all(tmp_path / "extracts")
    world = asset_world(MERLIN, SCALE)
    assert counts["po_line"] == len(world.po_lines) == con["erp-mm"].count("po_line")
    pl = world.po_lines[0]
    rec = con["erp-mm"].get("po_line", pl.key).fields
    assert rec["BEDAT"] == pl.order_date.strftime("%Y%m%d") and rec["MENGE"] == str(pl.qty)
    sailing = con["vessel"].get("sailing", world.sailings[0].sailing_id).fields
    assert sailing["depart_date"] == world.sailings[0].depart.isoformat()
    acts = list(con["p6"].iter("activity"))
    assert any(a.fields["udf"].get("SAP WO") for a in acts)
    with (tmp_path / "truth" / "links.csv").open() as f:
        links = list(csv.DictReader(f))
    comps = {r.key for r in con["erp-pm"].iter("component")}
    assert all(lk["from_key"] in comps for lk in links if lk["link_type"] == "component_po_line")


def test_procurement_runs_on_adapted_merlin_without_a_model(merlin, tmp_path):
    adapt_run(merlin, dialect_mapping(MERLIN), tmp_path)
    s = run_procurement(tmp_path, step_days=28, use_model=False)
    assert s["cost_usd"] == 0.0 and s["run"]["asset"] == "FPSO Merlin"
    for action in ("order_now", "order_soon"):
        t = s["totals"][action]
        assert t["precision"] is None or t["precision"] >= 0.8
    assert s["liaison"]["no_ack_precision"] is None or s["liaison"]["no_ack_precision"] >= 0.9
