import csv
import email
import hashlib
from pathlib import Path

import pytest

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


def _rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    for profile in ("A", "B"):
        d = tmp_path_factory.mktemp(profile)
        export(build_world(seed=11, scale=SCALE), d, profile=profile)
        out[profile] = d
    return out


def test_same_seed_gives_identical_files(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    export(build_world(seed=3, scale=SCALE), a, "A")
    export(build_world(seed=3, scale=SCALE), b, "A")
    assert _digest(a) == _digest(b)


def test_profiles_corrupt_differently(runs):
    a = (runs["A"] / "extracts/vessel/manifest_lines.csv").read_text()
    b = (runs["B"] / "extracts/vessel/manifest_lines.csv").read_text()
    assert a != b


def _source_keys(ex: Path) -> dict[str, set[str]]:
    po_rows = _rows(ex / "erp_mm/po_lines.csv")
    xer = (ex / "p6/kestrel.xer").read_text().splitlines()
    task_fields = xer[xer.index("%T\tTASK") + 1].split("\t")[1:]
    code_col = task_fields.index("task_code")
    in_task, task_codes = False, set()
    for line in xer:
        if line.startswith("%T"):
            in_task = line == "%T\tTASK"
        elif in_task and line.startswith("%R"):
            task_codes.add(line.split("\t")[1:][code_col])
    return {
        "erp-pm": {r["EQUNR"] for r in _rows(ex / "erp_pm/equipment.csv")}
        | {r["AUFNR"] for r in _rows(ex / "erp_pm/work_orders.csv")}
        | {f"{r['RSNUM']}-{r['RSPOS']}" for r in _rows(ex / "erp_pm/wo_components.csv")},
        "erp-mm": {f"{r['EBELN']}-{r['EBELP']}" for r in po_rows} | {r["EBELN"] for r in po_rows},
        "vessel": {r["line_id"] for r in _rows(ex / "vessel/manifest_lines.csv")},
        "supplier-mail": {p.stem for p in (ex / "supplier_mail").glob("*.eml")},
        "p6": task_codes,
    }


def test_truth_keys_exist_in_extracts(runs):
    for d in runs.values():
        keys = _source_keys(d / "extracts")
        links = _rows(d / "truth/links.csv")
        assert links
        for lk in links:
            assert lk["from_key"] in keys[lk["from_source"]], lk
            assert lk["to_key"] in keys[lk["to_source"]], lk


def test_all_five_link_types_present(runs):
    types = {r["link_type"] for r in _rows(runs["A"] / "truth/links.csv")}
    assert types == {
        "equipment_activity",
        "work_order_activity",
        "component_po_line",
        "po_line_manifest_line",
        "po_email",
    }


def test_truth_is_outside_extracts(runs):
    for d in runs.values():
        assert not any("truth" in p.parts for p in (d / "extracts").rglob("*"))


def test_emails_parse(runs):
    for p in sorted((runs["B"] / "extracts/supplier_mail").glob("*.eml"))[:20]:
        msg = email.message_from_bytes(p.read_bytes())
        assert msg["From"] and msg["Subject"] and msg.get_payload()


def test_held_back_profile_merges_some_manifest_lines(tmp_path):
    export(build_world(seed=11, scale=0.3), tmp_path, "B")
    rows = _rows(tmp_path / "extracts/vessel/manifest_lines.csv")
    assert any(r["description"].startswith("ASSORTED") for r in rows)


def test_dev_profile_never_merges(tmp_path):
    export(build_world(seed=11, scale=0.3), tmp_path, "A")
    rows = _rows(tmp_path / "extracts/vessel/manifest_lines.csv")
    assert not any(r["description"].startswith("ASSORTED") for r in rows)
