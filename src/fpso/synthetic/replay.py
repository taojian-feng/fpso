"""Replay clock: a run's extracts as they stood on one day.

A daily planner brief needs the world as it looked that morning, not the whole 6 months at
once. This copies a run directory, keeping only the records that existed by ``as_of``:

- PO lines whose order date (BEDAT) is on or before ``as_of``;
- supplier emails sent on or before ``as_of``;
- manifest lines whose sailing departs within ``MANIFEST_LEAD_DAYS`` after ``as_of``, or earlier
  (the vessel team issues a manifest a few days before the boat sails);
- weather observed and forecasts issued by ``as_of``, and the historian up to ``as_of``.

Masters, work orders, components, the schedule, sailings and stock are kept whole. Truth links
are cut to the records left, so scoring a replayed day only counts what could be seen.

fpso-replay --run data/kestrel-A --as-of 2027-02-01 --out data/kestrel-A@2027-02-01
"""

from __future__ import annotations

import argparse
import csv
import email
import json
import shutil
from datetime import date, timedelta
from email import policy
from email.utils import parsedate_to_datetime
from pathlib import Path

MANIFEST_LEAD_DAYS = 3


def _read(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def _write(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def _sap(d: str) -> date:
    return date(int(d[:4]), int(d[4:6]), int(d[6:8]))


def as_of_view(run_dir: str | Path, as_of: date, out_dir: str | Path) -> dict:
    """Write ``run_dir`` as it stood on ``as_of`` to ``out_dir``; return the counts kept."""
    src, out = Path(run_dir), Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src / "extracts", out / "extracts")
    ex = out / "extracts"

    header, rows = _read(ex / "erp_mm" / "po_lines.csv")
    rows = [r for r in rows if _sap(r["BEDAT"]) <= as_of]
    _write(ex / "erp_mm" / "po_lines.csv", header, rows)
    po_lines = {f"{r['EBELN']}-{r['EBELP']}" for r in rows}
    pos = {r["EBELN"] for r in rows}

    messages = set()
    for path in sorted((ex / "supplier_mail").glob("*.eml")):
        msg = email.message_from_bytes(path.read_bytes(), policy=policy.default)
        if parsedate_to_datetime(msg["Date"]).date() > as_of:
            path.unlink()
        else:
            messages.add(path.stem)

    _, sailings = _read(ex / "vessel" / "sailings.csv")
    issued = {
        s["sailing_id"]
        for s in sailings
        if date.fromisoformat(s["depart_date"]) - timedelta(days=MANIFEST_LEAD_DAYS) <= as_of
    }
    header, rows = _read(ex / "vessel" / "manifest_lines.csv")
    rows = [r for r in rows if r["sailing_id"] in issued]
    _write(ex / "vessel" / "manifest_lines.csv", header, rows)
    manifest = {r["line_id"] for r in rows}

    # Phase 3 operations data: only what was observed or forecast by that morning.
    for name, col in (("observed.csv", "date"), ("forecasts.csv", "issued")):
        path = ex / "weather" / name
        if path.exists():
            header, rows = _read(path)
            _write(path, header, [r for r in rows if date.fromisoformat(r[col]) <= as_of])
    path = ex / "historian" / "vibration.csv"
    if path.exists():
        header, rows = _read(path)
        _write(path, header, [r for r in rows if date.fromisoformat(r["date"]) <= as_of])

    visible = {
        "erp-mm": lambda k: k in po_lines or k in pos,
        "vessel": lambda k: k in manifest,
        "supplier-mail": lambda k: k in messages,
    }

    def seen(source: str, key: str) -> bool:
        return visible.get(source, lambda k: True)(key)

    (out / "truth").mkdir(parents=True, exist_ok=True)
    header, rows = _read(src / "truth" / "links.csv")
    rows = [
        r
        for r in rows
        if seen(r["from_source"], r["from_key"]) and seen(r["to_source"], r["to_key"])
    ]
    _write(out / "truth" / "links.csv", header, rows)

    run = json.loads((src / "truth" / "run.json").read_text())
    kept = {
        "po_lines": len(po_lines),
        "manifest_lines": len(manifest),
        "emails": len(messages),
        "links": len(rows),
    }
    run["as_of"] = as_of.isoformat()
    run["replayed_from"] = str(src)
    run["counts"] = {**run["counts"], **kept}
    (out / "truth" / "run.json").write_text(json.dumps(run, indent=2))
    return kept


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True, help="a run directory written by fpso-synth")
    ap.add_argument("--as-of", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    print(json.dumps(as_of_view(args.run, args.as_of, args.out), indent=2))


if __name__ == "__main__":
    main()
