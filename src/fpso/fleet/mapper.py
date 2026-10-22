"""Onboarding without a model: a rule mapper from a new asset's raw fields to canonical roles.

For every raw column it profiles the values (dates and their format, ids, integers,
decimals, tags, emails, free text, short codes) and splits the name into words through a
small generic glossary (NUM is a number, WO a work order, VEN a vendor, ITEM a material).
Each canonical field (``fpso.fleet.canonical``) is then scored against each column:

    score = name words (Dice overlap) + value kind fit + key overlap (references only)

A reference such as a work order's equipment must also share values with the key it points
at, which tells apart columns a name alone cannot. Tables are matched to entities first,
then fields within each table, best pairs first. A field the mapper is sure of (score and
margin over the runner-up both high) is accepted; the rest are flagged for review.

The draft is scored against the dialect's truth, and turned into person-hours with the
effort model below. Nothing here calls a model.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime

from fpso.connectors import Connector
from fpso.fleet.canonical import ENTITIES, N_FIELDS, UDF_WORK_ORDER, Field, Mapping, field_ids
from fpso.resolution.normalize import TAG_RE

# Generic glossary: raw name fragment -> concept. Not any one system's column names.
GLOSSARY = {
    **dict.fromkeys(["NUM", "NUMBER", "NBR", "NO", "NR", "ID", "IDENT", "KEY"], "NO"),
    **dict.fromkeys(["ASSET", "EQUIP", "EQUIPMENT", "EQ"], "EQUIPMENT"),
    **dict.fromkeys(["WO", "WORKORDER", "WORK", "JOB"], "WORKORDER"),
    **dict.fromkeys(["ITEM", "MATERIAL", "MAT", "SKU"], "MATERIAL"),
    **dict.fromkeys(["PO", "PURCHASEORDER", "PURCHASE"], "PURCHASEORDER"),
    **dict.fromkeys(["LINE", "LN", "POS", "POSITION"], "LINE"),
    **dict.fromkeys(["VEN", "VEND", "VENDOR", "SUPPLIER", "SUPP", "COMPANY"], "VENDOR"),
    **dict.fromkeys(["DESC", "DESCRIPTION", "TEXT", "TXT"], "DESCRIPTION"),
    **dict.fromkeys(["QTY", "QUANTITY"], "QUANTITY"),
    **dict.fromkeys(["PIECES", "PCS"], "PIECES"),
    **dict.fromkeys(["DATE", "DT", "DAY"], "DATE"),
    **dict.fromkeys(["DAYS"], "DAYS"),
    **dict.fromkeys(["START", "BEGIN"], "START"),
    **dict.fromkeys(["FINISH", "END", "COMPLETE"], "FINISH"),
    **dict.fromkeys(["SCHED", "SCHEDULED", "PLANNED", "PLAN", "TARGET"], "SCHEDULED"),
    **dict.fromkeys(["REQ", "REQUIRE", "REQUIRED", "NEED", "NEEDED"], "REQUIRED"),
    **dict.fromkeys(["REQUEST", "RESERVATION", "RESERVE", "RES"], "RESERVATION"),
    **dict.fromkeys(["DEL", "DELIV", "DELIVERY"], "DELIVERY"),
    **dict.fromkeys(["ORDER", "ORD"], "ORDER"),
    **dict.fromkeys(["LEADTIME", "LEAD"], "LEADTIME"),
    **dict.fromkeys(["WEIGHT", "WT", "WGT", "KG"], "WEIGHT"),
    **dict.fromkeys(["GROSS"], "GROSS"),
    **dict.fromkeys(["UNIT", "UOM"], "UNIT"),
    **dict.fromkeys(["EMAIL", "MAIL", "SMTP"], "EMAIL"),
    **dict.fromkeys(["ADDR", "ADDRESS"], "ADDRESS"),
    **dict.fromkeys(["NAME"], "NAME"),
    **dict.fromkeys(["TAG"], "TAG"),
    **dict.fromkeys(["TYPE", "KIND", "CATEGORY"], "TYPE"),
    **dict.fromkeys(["CLASS"], "CLASS"),
    **dict.fromkeys(["LOCATION", "LOC", "FUNCLOC"], "LOCATION"),
    **dict.fromkeys(["STORAGE", "STORE", "BIN", "WAREHOUSE"], "STORAGE"),
    **dict.fromkeys(["SITE", "PLANT"], "SITE"),
    **dict.fromkeys(["MODEL"], "MODEL"),
    **dict.fromkeys(["MANUFACTURER", "MFR", "MAKER", "MAKE"], "MANUFACTURER"),
    **dict.fromkeys(["PART"], "PART"),
    **dict.fromkeys(["PRIORITY", "PRIO"], "PRIORITY"),
    **dict.fromkeys(["CRITICALITY", "CRIT", "ABC"], "CRITICALITY"),
    **dict.fromkeys(["VOYAGE", "SAILING", "TRIP"], "VOYAGE"),
    **dict.fromkeys(["VESSEL", "SHIP", "BOAT"], "VESSEL"),
    **dict.fromkeys(["ETD", "DEPART", "DEPARTURE"], "DEPARTURE"),
    **dict.fromkeys(["ROUTE"], "ROUTE"),
    **dict.fromkeys(["DEST", "DESTINATION"], "DESTINATION"),
    **dict.fromkeys(["CARGO", "MANIFEST", "SHIPMENT"], "CARGO"),
    **dict.fromkeys(["REF", "REFERENCE"], "REFERENCE"),
    **dict.fromkeys(["CUST", "CUSTOMER"], "CUSTOMER"),
    **dict.fromkeys(["PACKAGE", "PKG", "PACKING"], "PACKAGE"),
    **dict.fromkeys(["BAL", "BALANCE", "ONHAND"], "BALANCE"),
    **dict.fromkeys(["CUR", "CURRENT"], "CURRENT"),
    **dict.fromkeys(["STOCK", "INVENTORY", "INV"], "INVENTORY"),
    **dict.fromkeys(["ISSUE"], "ISSUE"),
}

DATE_FORMATS = (
    "%Y%m%d",
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%d-%b-%Y",
    "%d-%b-%y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M",
    "%d.%m.%Y",
)

# How well a column's value kind fits a canonical field's kind.
KIND_FIT = {
    "key": {"id": 1.0, "code": 0.6, "int": 0.5},
    "ref": {"id": 1.0, "code": 0.6, "int": 0.5},
    "line": {"id": 1.0, "int": 0.8, "code": 0.3},
    "date": {"date": 1.0},
    "int": {"int": 1.0, "decimal": 0.5, "id": 0.3},
    "decimal": {"decimal": 1.0, "int": 0.7},
    "text": {"text": 1.0, "code": 0.4, "tag": 0.2},
    "code": {"code": 1.0, "text": 0.4, "id": 0.4, "int": 0.3, "tag": 0.3},
    "tag": {"tag": 1.0, "code": 0.3},
    "email": {"email": 1.0},
}
MIN_SCORE = 0.4  # below this a field is left unmapped
ACCEPT_SCORE = 0.75  # at or above this, with ACCEPT_MARGIN over the runner-up, no review
ACCEPT_MARGIN = 0.1
DATE_SHARE = 0.95  # share of values a date format must parse

# Effort model (assumptions, stated in every report).
HOURS_PER_DAY = 8.0
NEW_CONNECTOR_HOURS = 6.0  # Kestrel: a connector per source written and tested by hand
SOURCE_SETUP_HOURS = 1.0  # receive an extract, point the generic adapter at it, check counts
LIBRARY_SETUP_HOURS = 0.5  # the same when a library dialect already fits
FIX_MINUTES = 20.0  # a field mapped by hand: find the column, check values, set the transform
REVIEW_MINUTES = 3.0  # confirm a field the mapper flagged as unsure
GLANCE_MINUTES = 1.0  # glance at a field the mapper accepted
SOURCES = 5  # erp-pm, erp-mm, p6, vessel, supplier-mail


def words(name: str) -> list[str]:
    """Raw column name -> glossary concepts: 'VENDELIVERYDATE' -> VENDOR, DELIVERY, DATE."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    out: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", spaced.upper()):
        if chunk:
            out += [GLOSSARY.get(t, t) for t in _segment(chunk)]
    return list(dict.fromkeys(out))


def _segment(chunk: str) -> list[str]:
    """Split a run of letters into glossary words, covering as many letters as possible."""
    n = len(chunk)
    best: list[tuple[int, int, list[str]] | None] = [None] * (n + 1)
    best[0] = (0, 0, [])
    for i in range(1, n + 1):
        for j in range(i):
            if best[j] is None:
                continue
            piece = chunk[j:i]
            known = piece in GLOSSARY
            cov, ntok, toks = best[j]
            cand = (cov + (len(piece) if known else 0), ntok - 1, toks + [piece])
            if best[i] is None or cand[:2] > best[i][:2]:
                best[i] = cand
    return best[n][2]


def dice(a: list[str] | set[str], b: tuple[str, ...] | set[str]) -> float:
    a, b = set(a), set(b)
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


# --- Profiling --------------------------------------------------------------------------


@dataclass
class Column:
    table: str
    name: str
    kind: str  # id, int, decimal, date, email, tag, text, code, empty
    fmt: str | None
    unique: float
    values: set[str] = field(repr=False)
    words: list[str] = field(default_factory=list)


def _date_format(vals: list[str]) -> str | None:
    best, share = None, 0.0
    for fmt in DATE_FORMATS:
        ok = 0
        for v in vals:
            try:
                ok += 1990 <= datetime.strptime(v, fmt).year <= 2100  # 41000101 is an id
            except ValueError:
                pass
        if ok / len(vals) > share:
            best, share = fmt, ok / len(vals)
    return best if share >= DATE_SHARE else None


def _mean_len(vals: list[str]) -> float:
    return sum(len(v) for v in vals) / len(vals)


def profile(table: str, name: str, raw: list[str]) -> Column:
    vals = [v.strip() for v in raw if v and v.strip()]
    col = Column(table, name, "empty", None, 0.0, set(vals), words(name))
    if not vals:
        return col
    col.unique = len(set(vals)) / len(vals)
    sample = vals[:400]
    if fmt := _date_format(sample):
        col.kind, col.fmt = "date", fmt
    elif all("@" in v and "." in v for v in sample):
        col.kind = "email"
    elif all(re.fullmatch(r"\d+(\.\d+)?", v) for v in sample):
        if any("." in v and not re.fullmatch(r"\d+\.0+", v) for v in sample):
            col.kind = "decimal"
        elif all(v.isdigit() for v in sample) and (
            max(len(v) for v in sample) >= 6 or all(v.startswith("0") for v in sample)
        ):
            col.kind = "id"
        else:
            col.kind = "int"
    elif sum(bool(TAG_RE.fullmatch(v)) for v in sample) >= 0.6 * len(sample):
        col.kind = "tag"
    elif _mean_len(sample) >= 12 and sum(" " in v for v in sample) >= 0.5 * len(sample):
        col.kind = "text"
    else:
        col.kind = "code"
    return col


def profile_tables(sources: dict[str, Connector]) -> dict[str, list[Column]]:
    """Raw CSV tables -> profiled columns, from the raw connectors (one entity type per file)."""
    out = {}
    for con in sources.values():
        for table in con.entity_types():
            if not table.endswith(".csv"):
                continue
            cols = con.schema(table)
            rows = [r.fields for r in con.iter(table)]
            out[table] = [profile(table, c, [r.get(c, "") for r in rows]) for c in cols]
    return out


def udf_values(sources: dict[str, Connector]) -> dict[str, set[str]]:
    """P6 user-defined field label -> the values activities hold under it."""
    out: dict[str, set[str]] = defaultdict(set)
    for con in sources.values():
        for t in con.entity_types():
            if t == "activity":
                for r in con.iter(t):
                    for label, v in (r.fields.get("udf") or {}).items():
                        out[label].add(v)
    return out


# --- Scoring a column for a field -------------------------------------------------------


def field_score(f: Field, col: Column, keys: dict[str, set[str]] | None = None) -> float:
    name = 1.0 if col.name == f.name else dice(col.words, f.words)
    kind = KIND_FIT[f.kind].get(col.kind, 0.0)
    if f.kind == "ref" and keys is not None and f.ref in keys and col.values:
        overlap = len(col.values & keys[f.ref]) / len(col.values)
        return 0.45 * name + 0.25 * kind + 0.30 * overlap
    return 0.6 * name + 0.4 * kind


@dataclass
class FieldDraft:
    field: str  # 'entity.FIELD'
    table: str | None
    column: str | None
    fmt: str | None
    confidence: float
    margin: float
    accepted: bool  # sure enough to need no review
    source: str = "mapper"  # mapper, library


@dataclass
class Draft:
    mapping: Mapping
    fields: list[FieldDraft]

    def as_dict(self) -> dict:
        return {"mapping": self.mapping.as_dict(), "fields": [asdict(f) for f in self.fields]}


def _assign_fields(
    entity: str, cols: list[Column], keys: dict[str, set[str]] | None, only: set[str] | None = None
) -> list[FieldDraft]:
    fields = [f for f in ENTITIES[entity].fields if only is None or f.name in only]
    scores = {(f.name, c.name): field_score(f, c, keys) for f in fields for c in cols}
    out: dict[str, FieldDraft] = {}
    used: set[str] = set()
    for (fname, cname), s in sorted(scores.items(), key=lambda kv: -kv[1]):
        if fname in out or cname in used or s < MIN_SCORE:
            continue
        runner = max((v for (f, c), v in scores.items() if f == fname and c != cname), default=0)
        col = next(c for c in cols if c.name == cname)
        out[fname] = FieldDraft(
            f"{entity}.{fname}",
            col.table,
            cname,
            col.fmt,
            round(s, 3),
            round(s - runner, 3),
            s >= ACCEPT_SCORE and s - runner >= ACCEPT_MARGIN,
        )
        used.add(cname)
    return [
        out.get(f.name) or FieldDraft(f"{entity}.{f.name}", None, None, None, 0.0, 0.0, False)
        for f in fields
    ]


def _table_score(entity: str, cols: list[Column]) -> float:
    e = ENTITIES[entity]
    stem = cols[0].table.rsplit("/", 1)[-1].rsplit(".", 1)[0] if cols else ""
    best = [max((field_score(f, c) for c in cols), default=0.0) for f in e.fields]
    return 0.3 * dice(words(stem), e.words) + 0.7 * sum(best) / len(best)


def _keys(fields: list[FieldDraft], tables: dict[str, list[Column]]) -> dict[str, set[str]]:
    """Values of each entity's key column, from the fields mapped so far."""
    out = {}
    for fd in fields:
        entity, fname = fd.field.split(".", 1)
        if entity not in ENTITIES or not fd.column:
            continue
        f = next(x for x in ENTITIES[entity].fields if x.name == fname)
        if f.kind == "key":
            out[entity] = next(c.values for c in tables[fd.table] if c.name == fd.column)
    return out


def _map_udf(udfs: dict[str, set[str]], wo_keys: set[str]) -> FieldDraft:
    best, score, runner = None, 0.0, 0.0
    for label, vals in sorted(udfs.items()):
        s = len(vals & wo_keys) / len(vals) if vals else 0.0
        if s > score:
            best, score, runner = label, s, score
        elif s > runner:
            runner = s
    ok = best is not None and score >= 0.5
    return FieldDraft(
        f"p6.udf.{UDF_WORK_ORDER}",
        "p6" if ok else None,
        best if ok else None,
        None,
        round(score, 3),
        round(score - runner, 3),
        ok and score >= 0.9 and score - runner >= ACCEPT_MARGIN,
    )


def draft_mapping(sources: dict[str, Connector]) -> Draft:
    """Map every canonical field onto the raw extracts, from names and values alone."""
    tables = profile_tables(sources)
    pairs = sorted(
        ((_table_score(e, cols), e, t) for e in ENTITIES for t, cols in tables.items()),
        reverse=True,
    )
    table_of: dict[str, str] = {}
    for _, e, t in pairs:
        if e not in table_of and t not in table_of.values():
            table_of[e] = t
    first = [fd for e, t in table_of.items() for fd in _assign_fields(e, tables[t], None)]
    keys = _keys(first, tables)
    fields = [fd for e, t in table_of.items() for fd in _assign_fields(e, tables[t], keys)]
    missing = [e for e in ENTITIES if e not in table_of]
    fields += [
        FieldDraft(f"{e}.{f.name}", None, None, None, 0.0, 0.0, False)
        for e in missing
        for f in ENTITIES[e].fields
    ]
    fields.append(_map_udf(udf_values(sources), keys.get("work_order", set())))
    order = {fid: i for i, fid in enumerate(field_ids())}
    fields.sort(key=lambda fd: order[fd.field])
    return Draft(_to_mapping(table_of, fields), fields)


def _to_mapping(table_of: dict[str, str], fields: list[FieldDraft]) -> Mapping:
    m = Mapping(tables=dict(table_of))
    for fd in fields:
        if fd.field.startswith("p6."):
            m.udf = fd.column
            continue
        m.fields[fd.field] = fd.column
        if fd.fmt:
            m.formats[fd.field] = fd.fmt
    return m


def reuse_mapping(known: Mapping, sources: dict[str, Connector]) -> Draft:
    """Start from a library dialect's confirmed mapping: keep every field whose column is
    still there with values of the same kind, and map only the rest by rules."""
    tables = profile_tables(sources)
    fields: list[FieldDraft] = []
    todo: dict[str, set[str]] = defaultdict(set)
    for e in ENTITIES.values():
        t = known.tables.get(e.name)
        cols = {c.name: c for c in tables.get(t, [])}
        for f in e.fields:
            fid = f"{e.name}.{f.name}"
            col = cols.get(known.fields.get(fid) or "")
            fmt_ok = f.kind != "date" or (col is not None and col.fmt == known.formats.get(fid))
            if col is not None and fmt_ok and KIND_FIT[f.kind].get(col.kind, 0) > 0:
                fields.append(FieldDraft(fid, t, col.name, col.fmt, 1.0, 1.0, True, "library"))
            else:
                todo[e.name].add(f.name)
    keys = _keys(fields, tables)
    for entity, names in todo.items():
        t = known.tables.get(entity)
        taken = {fd.column for fd in fields if fd.table == t}
        free = [c for c in tables.get(t, []) if c.name not in taken]
        fields += _assign_fields(entity, free, keys, only=names)
    udfs = udf_values(sources)
    if known.udf in udfs:
        fields.append(
            FieldDraft(f"p6.udf.{UDF_WORK_ORDER}", "p6", known.udf, None, 1.0, 1.0, True, "library")
        )
    else:
        fields.append(_map_udf(udfs, keys.get("work_order", set())))
    order = {fid: i for i, fid in enumerate(field_ids())}
    fields.sort(key=lambda fd: order[fd.field])
    return Draft(_to_mapping(dict(known.tables), fields), fields)


# --- Scoring and effort -----------------------------------------------------------------


def score_draft(draft: Draft, truth: Mapping) -> dict:
    """Which draft fields match the truth (column, and format for dates), and the effort."""
    right, wrong = [], []
    for fd in draft.fields:
        if fd.field.startswith("p6."):
            ok = fd.column == truth.udf
        else:
            entity = fd.field.split(".", 1)[0]
            ok = (
                fd.table == truth.tables[entity]
                and fd.column == truth.fields[fd.field]
                and (fd.field not in truth.formats or fd.fmt == truth.formats[fd.field])
            )
        (right if ok else wrong).append(fd)
    reused = [fd for fd in right if fd.source == "library"]
    accepted = [fd for fd in right if fd.accepted and fd.source != "library"]
    reviewed = [fd for fd in right if not fd.accepted]
    minutes = (
        len(wrong) * FIX_MINUTES + len(reviewed) * REVIEW_MINUTES + len(accepted) * GLANCE_MINUTES
    )
    setup = LIBRARY_SETUP_HOURS if reused else SOURCE_SETUP_HOURS
    hours = SOURCES * setup + minutes / 60
    return {
        "fields": len(draft.fields),
        "auto_correct": len(right),
        "auto_correct_share": round(len(right) / len(draft.fields), 4),
        "reused_from_library": len(reused),
        "accepted_unreviewed": len(accepted),
        "flagged_and_right": len(reviewed),
        "needs_fix": len(wrong),
        "silent_errors": sum(fd.accepted for fd in wrong),  # accepted, yet wrong
        "fixed_fields": [fd.field for fd in wrong],
        "person_hours": round(hours, 1),
        "person_days": round(hours / HOURS_PER_DAY, 2),
    }


def kestrel_baseline() -> dict:
    """Kestrel, onboarded by hand: a connector per source and every field worked out."""
    hours = SOURCES * NEW_CONNECTOR_HOURS + N_FIELDS * FIX_MINUTES / 60
    return {
        "fields": N_FIELDS,
        "auto_correct": 0,
        "auto_correct_share": 0.0,
        "needs_fix": N_FIELDS,
        "person_hours": round(hours, 1),
        "person_days": round(hours / HOURS_PER_DAY, 2),
    }


def effort_assumptions() -> dict:
    return {
        "hours_per_day": HOURS_PER_DAY,
        "new_connector_hours_per_source": NEW_CONNECTOR_HOURS,
        "source_setup_hours": SOURCE_SETUP_HOURS,
        "library_setup_hours": LIBRARY_SETUP_HOURS,
        "fix_minutes_per_field": FIX_MINUTES,
        "review_minutes_per_flagged_field": REVIEW_MINUTES,
        "glance_minutes_per_accepted_field": GLANCE_MINUTES,
        "sources": SOURCES,
    }


def confirm(truth: Mapping) -> Mapping:
    """The mapping after the simulated human review: every wrong or missing field set right
    (the reviewer knows the source), so the result is the dialect's truth."""
    return Mapping.from_dict(truth.as_dict())
