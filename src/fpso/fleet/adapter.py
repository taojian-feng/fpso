"""Present a mapped asset to the existing agents.

``raw_sources`` opens a new asset's extracts as they arrive, through the same read-only
connector contract: every CSV as its own entity type, the P6 schedule and the supplier
mailbox through their usual connectors (XER and .eml are standard formats). The mapper
profiles these.

``adapt_run`` then applies a confirmed mapping: it writes the extracts in the canonical
layout (canonical column names, dates and quantities in the canonical form, the P6
work-order UDF under its canonical label) next to the run's truth, so ``open_all``, the
replay clock and every Phase 0 and 1 agent work on the new asset unchanged.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from fpso.connectors import Connector
from fpso.connectors.base import read_csv
from fpso.connectors.sources import P6Connector, SupplierMailConnector
from fpso.fleet.canonical import ENTITIES, UDF_WORK_ORDER, Mapping


class RawCsvConnector(Connector):
    """Every CSV in one folder of a raw extract, unmapped: one entity type per file (its
    path under extracts/), records keyed by row number."""

    def __init__(self, root: str | Path, folder: str):
        self.name = folder
        self.folder = folder
        self.description = f"Raw '{folder}' extract, not yet mapped to canonical fields."
        super().__init__(root)

    def load(self) -> None:
        for path in sorted((self.root / self.folder).glob("*.csv")):
            for i, row in enumerate(read_csv(path), 1):
                self._add(f"{self.folder}/{path.name}", str(i), row, path)


def raw_sources(extracts: str | Path) -> dict[str, Connector]:
    """Open a new asset's raw extracts with connectors that assume no field names."""
    ex = Path(extracts)
    out: dict[str, Connector] = {}
    for d in sorted(p for p in ex.iterdir() if p.is_dir()):
        if d.name == "p6":
            out["p6"] = P6Connector(ex)
        elif d.name == "supplier_mail":
            out["supplier-mail"] = SupplierMailConnector(ex)
        elif any(d.glob("*.csv")):
            out[d.name] = RawCsvConnector(ex, d.name)
    return out


def _convert(value: str, kind: str, fmt: str | None, out_fmt: str) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    if kind == "date" and fmt:
        try:
            return datetime.strptime(v, fmt).strftime(out_fmt)
        except ValueError:
            return v
    if kind in ("int", "line") and re.fullmatch(r"\d+\.0+", v):
        return v.split(".")[0]
    return v


def adapt_run(raw_run: str | Path, mapping: Mapping, out_run: str | Path) -> dict:
    """Write ``raw_run`` in the canonical layout under ``out_run``; return rows per entity."""
    raw, out = Path(raw_run), Path(out_run)
    ex_in, ex_out = raw / "extracts", out / "extracts"
    if ex_out.exists():
        shutil.rmtree(ex_out)
    counts = {}
    for e in ENTITIES.values():
        rows = list(read_csv(ex_in / mapping.tables[e.name]))
        path = ex_out / e.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([x.name for x in e.fields])
            for r in rows:
                w.writerow(
                    [
                        _convert(
                            r.get(mapping.fields.get(f"{e.name}.{x.name}") or "", ""),
                            x.kind,
                            mapping.formats.get(f"{e.name}.{x.name}"),
                            x.out_fmt,
                        )
                        for x in e.fields
                    ]
                )
        counts[e.name] = len(rows)
    (ex_out / "p6").mkdir(parents=True, exist_ok=True)
    for xer in (ex_in / "p6").glob("*.xer"):
        text = xer.read_text()
        if mapping.udf and mapping.udf != UDF_WORK_ORDER:
            text = text.replace(f"\tTASK\t{mapping.udf}\n", f"\tTASK\t{UDF_WORK_ORDER}\n")
        (ex_out / "p6" / xer.name).write_text(text)
    shutil.copytree(ex_in / "supplier_mail", ex_out / "supplier_mail")
    (out / "truth").mkdir(parents=True, exist_ok=True)
    shutil.copy(raw / "truth" / "links.csv", out / "truth" / "links.csv")
    run = json.loads((raw / "truth" / "run.json").read_text())
    run["adapted_from"] = str(raw)
    (out / "truth" / "run.json").write_text(json.dumps(run, indent=2))
    return counts
