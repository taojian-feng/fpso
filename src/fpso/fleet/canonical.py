"""The canonical roles the existing agents read, in Kestrel's (SAP-style) field names.

The Phase 0 and 1 agents were written against Kestrel's extracts, so Kestrel's column names
are the canonical schema. Each field carries a plain-language role, the kind of value it
holds, generic words for it (not any one vendor's names) and, for a reference, the entity
whose key it points at. The rule mapper uses these to place a new asset's raw fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Field:
    name: str  # canonical column
    role: str
    kind: str  # key, ref, line, date, int, decimal, text, code, tag, email
    words: tuple[str, ...]
    ref: str | None = None  # entity whose key a 'ref' (or 'key' part) points at
    out_fmt: str = "%Y%m%d"  # how a date is written in the canonical extract


@dataclass(frozen=True)
class Entity:
    name: str
    path: str  # canonical extract file under extracts/
    source: str  # connector that reads it
    words: tuple[str, ...]
    fields: tuple[Field, ...]


def _f(name, role, kind, *words, ref=None, out_fmt="%Y%m%d"):
    return Field(name, role, kind, tuple(words), ref, out_fmt)


ENTITIES: dict[str, Entity] = {
    e.name: e
    for e in (
        Entity(
            "equipment",
            "erp_pm/equipment.csv",
            "erp-pm",
            ("EQUIPMENT",),
            (
                _f("EQUNR", "equipment number", "key", "EQUIPMENT", "NO"),
                _f("EQKTX", "equipment description", "text", "DESCRIPTION"),
                _f("TECHIDENT", "equipment tag", "tag", "TAG"),
                _f("EQART", "equipment class", "code", "CLASS", "TYPE"),
                _f("TPLNR", "functional location", "code", "LOCATION"),
                _f("HERST_TYPBZ", "maker and model", "text", "MODEL", "MANUFACTURER"),
                _f("ABCKZ", "criticality", "code", "CRITICALITY", "PRIORITY"),
            ),
        ),
        Entity(
            "work_order",
            "erp_pm/work_orders.csv",
            "erp-pm",
            ("WORKORDER",),
            (
                _f("AUFNR", "work order number", "key", "WORKORDER", "NO"),
                _f("AUART", "work order type", "code", "WORKORDER", "TYPE"),
                _f("EQUNR", "equipment the work is on", "ref", "EQUIPMENT", "NO", ref="equipment"),
                _f("KTEXT", "work order text", "text", "DESCRIPTION"),
                _f("GSTRP", "planned start", "date", "START", "SCHEDULED"),
                _f("GLTRP", "planned finish", "date", "FINISH", "SCHEDULED"),
            ),
        ),
        Entity(
            "component",
            "erp_pm/wo_components.csv",
            "erp-pm",
            ("MATERIAL", "WORKORDER", "RESERVATION"),
            (
                _f("RSNUM", "reservation number", "key", "RESERVATION", "NO"),
                _f("RSPOS", "reservation line", "line", "LINE", "NO"),
                _f("AUFNR", "work order", "ref", "WORKORDER", "NO", ref="work_order"),
                _f("MATNR", "material", "ref", "MATERIAL", "NO", ref="material"),
                _f("BDMNG", "quantity needed", "int", "QUANTITY"),
                _f("MEINS", "unit", "code", "UNIT"),
                _f("BDTER", "date needed", "date", "REQUIRED", "DATE"),
            ),
        ),
        Entity(
            "material",
            "erp_mm/materials.csv",
            "erp-mm",
            ("MATERIAL",),
            (
                _f("MATNR", "material number", "key", "MATERIAL", "NO"),
                _f("MAKTX", "material description", "text", "DESCRIPTION"),
                _f("MFRPN", "supplier part number", "code", "PART", "MANUFACTURER", "NO"),
                _f("LIFNR", "usual supplier", "ref", "VENDOR", "NO", ref="vendor"),
                _f("PLIFZ", "lead time in days", "int", "LEADTIME", "DAYS"),
                _f("BRGEW", "unit weight", "decimal", "WEIGHT"),
                _f("MEINS", "unit", "code", "UNIT"),
            ),
        ),
        Entity(
            "vendor",
            "erp_mm/vendors.csv",
            "erp-mm",
            ("VENDOR",),
            (
                _f("LIFNR", "vendor number", "key", "VENDOR", "NO"),
                _f("NAME1", "vendor name", "text", "NAME"),
                _f("SMTP_ADDR", "order email address", "email", "EMAIL", "ADDRESS"),
            ),
        ),
        Entity(
            "po_line",
            "erp_mm/po_lines.csv",
            "erp-mm",
            ("PURCHASEORDER", "LINE"),
            (
                _f("EBELN", "purchase order number", "key", "PURCHASEORDER", "NO"),
                _f("EBELP", "purchase order line", "line", "LINE", "NO"),
                _f("LIFNR", "vendor", "ref", "VENDOR", "NO", ref="vendor"),
                _f("MATNR", "material", "ref", "MATERIAL", "NO", ref="material"),
                _f("TXZ01", "line text", "text", "DESCRIPTION"),
                _f("MENGE", "quantity ordered", "int", "QUANTITY", "ORDER"),
                _f("BEDAT", "order date", "date", "ORDER", "DATE"),
                _f("EINDT", "supplier delivery date", "date", "VENDOR", "DELIVERY", "DATE"),
                _f("AUFNR", "work order bought for", "ref", "WORKORDER", "NO", ref="work_order"),
            ),
        ),
        Entity(
            "stock",
            "erp_mm/stock.csv",
            "erp-mm",
            ("INVENTORY", "STOCK", "BALANCE"),
            (
                _f("MATNR", "material", "ref", "MATERIAL", "NO", ref="material"),
                _f("WERKS", "plant or site", "code", "SITE"),
                _f("LGORT", "storage location", "code", "LOCATION", "STORAGE"),
                _f("LABST", "quantity on hand", "int", "BALANCE", "QUANTITY"),
            ),
        ),
        Entity(
            "sailing",
            "vessel/sailings.csv",
            "vessel",
            ("SAILING", "VOYAGE"),
            (
                _f("sailing_id", "sailing id", "key", "VOYAGE", "NO"),
                _f("vessel", "vessel name", "text", "VESSEL", "NAME"),
                _f(
                    "depart_date", "departure date", "date", "DEPARTURE", "DATE", out_fmt="%Y-%m-%d"
                ),
                _f("route", "route", "code", "ROUTE", "DESTINATION"),
            ),
        ),
        Entity(
            "manifest_line",
            "vessel/manifest_lines.csv",
            "vessel",
            ("CARGO", "MANIFEST", "LINE"),
            (
                _f("line_id", "manifest line id", "key", "CARGO", "LINE", "NO"),
                _f("sailing_id", "sailing", "ref", "VOYAGE", "NO", ref="sailing"),
                _f("customer_ref", "customer reference", "text", "REFERENCE", "PURCHASEORDER"),
                _f("description", "cargo description", "text", "DESCRIPTION"),
                _f("qty", "quantity", "int", "QUANTITY", "PIECES"),
                _f("gross_kg", "gross weight", "decimal", "WEIGHT", "GROSS"),
                _f("unit_type", "package type", "code", "PACKAGE", "TYPE"),
            ),
        ),
    )
}

# The P6 user-defined field that holds the work order number on an activity.
UDF_WORK_ORDER = "SAP WO"

N_FIELDS = sum(len(e.fields) for e in ENTITIES.values()) + 1  # plus the P6 work-order UDF


def field_ids() -> list[str]:
    """Every canonical field as 'entity.FIELD', plus the P6 work-order UDF."""
    return [f"{e.name}.{f.name}" for e in ENTITIES.values() for f in e.fields] + [
        f"p6.udf.{UDF_WORK_ORDER}"
    ]


@dataclass
class Mapping:
    """Where each canonical field sits in an asset's raw extracts.

    The same shape serves as a dialect's truth, as the rule mapper's draft and as the
    confirmed mapping kept in the library."""

    tables: dict[str, str] = field(default_factory=dict)  # entity -> raw file under extracts/
    fields: dict[str, str | None] = field(default_factory=dict)  # 'entity.FIELD' -> raw column
    formats: dict[str, str] = field(default_factory=dict)  # 'entity.FIELD' -> raw date format
    udf: str | None = None  # raw label of the P6 work-order UDF

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Mapping:
        return cls(**d)


def identity_mapping() -> Mapping:
    """Kestrel's own convention: every field under its canonical name."""
    m = Mapping(udf=UDF_WORK_ORDER)
    for e in ENTITIES.values():
        m.tables[e.name] = e.path
        for f in e.fields:
            m.fields[f"{e.name}.{f.name}"] = f.name
            if f.kind == "date":
                m.formats[f"{e.name}.{f.name}"] = f.out_fmt
    return m
