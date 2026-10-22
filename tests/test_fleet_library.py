import json

from fpso.fleet.adapter import raw_sources
from fpso.fleet.assets import MERLIN, OSPREY, asset_world
from fpso.fleet.cli import main
from fpso.fleet.dialects import dialect_mapping, export_maximo
from fpso.library import AssetEntry, Library
from fpso.synthetic.export import export
from fpso.synthetic.truth import build_world

SCALE = 0.05


def test_library_starts_with_the_shared_pieces():
    lib = Library.new()
    assert set(lib.connectors) == {"erp-pm", "erp-mm", "p6", "vessel", "supplier-mail"}
    assert list(lib.dialects) == ["sap"]
    assert lib.rule_cards["procurement.order_by"]["MIN_SHIP_DAYS"] == 4
    assert "ack" in lib.rule_cards["liaison.email_kind"]


def test_find_dialect_matches_by_columns(tmp_path):
    export(build_world(seed=7, scale=SCALE), tmp_path / "k", "A")
    export_maximo(asset_world(MERLIN, SCALE), MERLIN, tmp_path / "m")
    export_maximo(asset_world(OSPREY, SCALE), OSPREY, tmp_path / "o")
    lib = Library.new()
    assert lib.find_dialect(raw_sources(tmp_path / "k" / "extracts")) == ("sap", 1.0)
    assert lib.find_dialect(raw_sources(tmp_path / "m" / "extracts"))[0] is None
    lib.add_dialect("maximo", dialect_mapping(MERLIN))
    name, cover = lib.find_dialect(raw_sources(tmp_path / "o" / "extracts"))
    assert name == "maximo" and 0.9 <= cover < 1.0  # Osprey renamed one column


def test_library_round_trips_and_keeps_confirmed_patterns(tmp_path):
    run = tmp_path / "run"
    (run / "reports").mkdir(parents=True)
    (run / "reports" / "patterns.json").write_text(
        json.dumps([{"id": "p1", "status": "confirmed"}, {"id": "p2", "status": "proposed"}])
    )
    lib = Library.new()
    lib.add_dialect("maximo", dialect_mapping(MERLIN))
    lib.add_asset(AssetEntry("MRL", "FPSO Merlin", "maximo", str(run), {}, {"person_days": 1}))
    lib.save(tmp_path / "library.json")
    back = Library.load(tmp_path / "library.json")
    assert back.mapping("maximo") == dialect_mapping(MERLIN)
    assert back.assets["MRL"].onboarding == {"person_days": 1}
    assert [p["id"] for p in back.patterns["MRL"]] == ["p1"]


def test_fleet_cli_runs_free_and_passes_the_gate(tmp_path, monkeypatch):
    monkeypatch.delenv("FPSO_MODEL_BACKEND", raising=False)
    main(["--out", str(tmp_path), "--scale", str(SCALE), "--step", "56"])
    rep = tmp_path / "reports" / "fleet"
    s = json.loads((rep / "summary.json").read_text())
    assert s["gate"]["passed"]
    assert s["onboarding"]["OSP"]["person_days"] < s["onboarding"]["MRL"]["person_days"]
    assert s["onboarding"]["OSP"]["library_dialect"] == "maximo"
    assert set(s["agents"]) == {"KST", "MRL", "OSP"}
    assert s["a2a"]["questions"] > 0 and s["a2a"]["resolved_share"] > 0
    assert "Phase 4" in (rep / "summary.md").read_text()
    assert set(Library.load(rep / "library.json").dialects) == {"sap", "maximo"}
    assert (rep / "MRL" / "mapping-draft.md").exists()
    assert not list(tmp_path.rglob("onboarding-*.md"))  # the model path is off by default


def test_model_drafts_are_optional_and_use_the_stub_by_default(tmp_path, monkeypatch):
    from fpso.fleet.cli import _model_drafts

    monkeypatch.delenv("FPSO_MODEL_BACKEND", raising=False)
    export_maximo(asset_world(MERLIN, SCALE), MERLIN, tmp_path / "m")
    (tmp_path / "rep").mkdir()
    _model_drafts(raw_sources(tmp_path / "m" / "extracts"), tmp_path / "rep")
    assert (tmp_path / "rep" / "onboarding-maximo.md").read_text().startswith("# Onboarding")
