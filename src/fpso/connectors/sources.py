"""File-drop connectors for the five Phase 0 sources."""

from __future__ import annotations

import email
from email import policy
from email.utils import parsedate_to_datetime
from pathlib import Path

from fpso.connectors.base import Connector, read_csv


class ErpPmConnector(Connector):
    name = "erp-pm"
    description = "ERP plant maintenance: equipment master, work orders, work order components."

    def load(self) -> None:
        d = self.root / "erp_pm"
        for row in read_csv(d / "equipment.csv"):
            self._add("equipment", row["EQUNR"], row, d / "equipment.csv")
        for row in read_csv(d / "work_orders.csv"):
            self._add("work_order", row["AUFNR"], row, d / "work_orders.csv")
        for row in read_csv(d / "wo_components.csv"):
            self._add("component", f"{row['RSNUM']}-{row['RSPOS']}", row, d / "wo_components.csv")


class ErpMmConnector(Connector):
    name = "erp-mm"
    description = "ERP materials: material master, vendors, purchase orders and lines, stock."

    def load(self) -> None:
        d = self.root / "erp_mm"
        for row in read_csv(d / "materials.csv"):
            self._add("material", row["MATNR"], row, d / "materials.csv")
        for row in read_csv(d / "vendors.csv"):
            self._add("vendor", row["LIFNR"], row, d / "vendors.csv")
        pos: dict[str, dict] = {}
        for row in read_csv(d / "po_lines.csv"):
            self._add("po_line", f"{row['EBELN']}-{row['EBELP']}", row, d / "po_lines.csv")
            po = pos.setdefault(
                row["EBELN"],
                {"EBELN": row["EBELN"], "LIFNR": row["LIFNR"], "BEDAT": row["BEDAT"], "lines": []},
            )
            po["lines"].append(row["EBELP"])
            po["BEDAT"] = min(po["BEDAT"], row["BEDAT"])
        for po_no, po in pos.items():
            self._add("po", po_no, po, d / "po_lines.csv")
        for row in read_csv(d / "stock.csv"):
            self._add(
                "stock", f"{row['MATNR']}-{row['WERKS']}-{row['LGORT']}", row, d / "stock.csv"
            )
        if (d / "material_prices.csv").exists():  # Phase 2
            for row in read_csv(d / "material_prices.csv"):
                self._add("material_price", row["MATNR"], row, d / "material_prices.csv")


class P6Connector(Connector):
    name = "p6"
    description = "Primavera P6 schedule (XER): activities with WBS, dates, UDFs, predecessors."

    def load(self) -> None:
        path = next((self.root / "p6").glob("*.xer"))
        tables = parse_xer(path.read_text())
        wbs = {r["wbs_id"]: r for r in tables.get("PROJWBS", [])}
        udf_labels = {r["udf_type_id"]: r["udf_type_label"] for r in tables.get("UDFTYPE", [])}
        udfs: dict[str, dict[str, str]] = {}
        for r in tables.get("UDFVALUE", []):
            udfs.setdefault(r["fk_id"], {})[udf_labels.get(r["udf_type_id"], r["udf_type_id"])] = r[
                "udf_text"
            ]
        code_by_id = {r["task_id"]: r["task_code"] for r in tables.get("TASK", [])}
        preds: dict[str, list[str]] = {}
        for r in tables.get("TASKPRED", []):
            preds.setdefault(r["task_id"], []).append(code_by_id.get(r["pred_task_id"], ""))
        for r in tables.get("TASK", []):
            w = wbs.get(r["wbs_id"], {})
            fields = {
                "task_code": r["task_code"],
                "task_name": r["task_name"],
                "wbs_short_name": w.get("wbs_short_name", ""),
                "wbs_name": w.get("wbs_name", ""),
                "target_start_date": r["target_start_date"],
                "target_end_date": r["target_end_date"],
                "status_code": r["status_code"],
                "udf": udfs.get(r["task_id"], {}),
                "predecessors": preds.get(r["task_id"], []),
            }
            self._add("activity", r["task_code"], fields, path)


def parse_xer(text: str) -> dict[str, list[dict[str, str]]]:
    """Parse the %T / %F / %R table structure of a P6 XER export."""
    tables: dict[str, list[dict[str, str]]] = {}
    current, fields = None, []
    for line in text.splitlines():
        parts = line.split("\t")
        if parts[0] == "%T":
            current = parts[1]
            tables[current] = []
        elif parts[0] == "%F":
            fields = parts[1:]
        elif parts[0] == "%R" and current:
            tables[current].append(dict(zip(fields, parts[1:], strict=False)))
    return tables


class VesselConnector(Connector):
    name = "vessel"
    description = "Vessel scheduling: sailings and cargo manifest lines."

    def load(self) -> None:
        d = self.root / "vessel"
        for row in read_csv(d / "sailings.csv"):
            self._add("sailing", row["sailing_id"], row, d / "sailings.csv")
        for row in read_csv(d / "manifest_lines.csv"):
            self._add("manifest_line", row["line_id"], row, d / "manifest_lines.csv")
        if (d / "sailing_capacity.csv").exists():  # Phase 2
            for row in read_csv(d / "sailing_capacity.csv"):
                self._add("sailing_capacity", row["sailing_id"], row, d / "sailing_capacity.csv")


class SupplierMailConnector(Connector):
    name = "supplier-mail"
    description = "Supplier mailbox export: one message per .eml file."

    def load(self) -> None:
        for path in sorted((self.root / "supplier_mail").glob("*.eml")):
            msg = email.message_from_bytes(path.read_bytes(), policy=policy.default)
            body = msg.get_body(preferencelist=("plain",))
            fields = {
                "message_id": msg["Message-ID"],
                "from": str(msg["From"]),
                "to": str(msg["To"]),
                "date": parsedate_to_datetime(msg["Date"]).date().isoformat(),
                "subject": str(msg["Subject"]),
                "body": body.get_content() if body else "",
            }
            self._add("message", path.stem, fields, path)


class HelicopterConnector(Connector):
    name = "helicopter"
    description = "Helicopter flight list: seats, passengers booked and cargo payload per flight."

    def load(self) -> None:
        path = self.root / "helicopter" / "flights.csv"
        if path.exists():
            for row in read_csv(path):
                self._add("flight", row["flight_id"], row, path)


class MarineOpsConnector(Connector):
    name = "marine-ops"
    description = "Marine operations calendar: deck cranes available and bunkering days."

    def load(self) -> None:
        path = self.root / "marine_ops" / "crane_ops.csv"
        if path.exists():
            for row in read_csv(path):
                self._add("crane_day", row["date"], row, path)


class WarehouseConnector(Connector):
    name = "warehouse"
    description = "Store balances offshore and on the sister asset, with reservations."

    def load(self) -> None:
        path = self.root / "warehouse" / "stock_locations.csv"
        if path.exists():
            for row in read_csv(path):
                key = f"{row['MATNR']}-{row['WERKS']}-{row['LGORT']}"
                self._add("store_balance", key, row, path)


class WeatherConnector(Connector):
    name = "weather"
    description = "Met-ocean service: observed wave height and wind, and daily 5-day forecasts."

    def load(self) -> None:
        d = self.root / "weather"
        if (d / "observed.csv").exists():
            for row in read_csv(d / "observed.csv"):
                self._add("observation", row["date"], row, d / "observed.csv")
            for row in read_csv(d / "forecasts.csv"):
                key = f"{row['issued']}|{row['valid']}"
                self._add("forecast", key, row, d / "forecasts.csv")


class HistorianConnector(Connector):
    name = "historian"
    description = "Process historian: daily vibration readings of rotating equipment."

    def load(self) -> None:
        path = self.root / "historian" / "vibration.csv"
        if path.exists():
            for row in read_csv(path):
                self._add("vibration", f"{row['EQUNR']}|{row['date']}", row, path)


CONNECTORS: dict[str, type[Connector]] = {
    c.name: c
    for c in (
        ErpPmConnector,
        ErpMmConnector,
        P6Connector,
        VesselConnector,
        SupplierMailConnector,
        HelicopterConnector,
        MarineOpsConnector,
        WarehouseConnector,
        WeatherConnector,
        HistorianConnector,
    )
}


def open_all(extracts: str | Path) -> dict[str, Connector]:
    """Open every connector over one extracts directory."""
    return {name: cls(Path(extracts)) for name, cls in CONNECTORS.items()}
