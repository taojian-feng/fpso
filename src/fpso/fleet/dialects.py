"""Export dialects: the source-system convention an asset's extracts arrive in.

Kestrel's extracts are SAP-style (``fpso.synthetic.export``, unchanged here). A sister
asset on a Maximo-style CMMS sends the same kind of facts under other names: ASSETNUM,
WONUM, ITEMNUM, PONUM and POLINENUM instead of EQUNR, AUFNR, MATNR, EBELN and EBELP, dates
written ``09-APR-2027``, quantities as ``2.00``, a logistics provider with its own voyage
and cargo files, a P6 project of its own whose work-order UDF has another label, and a few
columns the agents do not need. Identifier values are kept, so truth keys carry over.

Layout of a Maximo-style run:

    extracts/
      maximo/        ASSET, WORKORDER, WPMATERIAL, ITEM, COMPANIES, POLINE, INVBALANCES (.csv)
      logistics/     voyages.csv, cargo_lines.csv
      p6/            <asset>_ta27.xer
      supplier_mail/ <message-id>.eml
    truth/           links.csv, run.json (never given to agents)

``dialect_mapping`` is the truth of where each canonical field went; the onboarding score
compares the rule mapper's draft with it.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from fpso.fleet.assets import Asset
from fpso.fleet.canonical import ENTITIES, Mapping, identity_mapping
from fpso.synthetic.corrupt import PROFILES, Corruptor
from fpso.synthetic.export import _write_emails, _write_xer
from fpso.synthetic.truth import ASSET_CODE, OPERATOR, World

ERP_DATE = "%d-%b-%Y"  # 09-APR-2027
ETD_DATE = "%d/%m/%Y %H:%M"  # 09/04/2027 06:00, day first
WORK_TYPES = {"PM01": "CM", "PM02": "PM", "PM03": "TA"}
PRIORITY = {"A": "1", "B": "2", "C": "3"}

# canonical entity -> (raw file, {canonical field: raw column})
MAXIMO: dict[str, tuple[str, dict[str, str]]] = {
    "equipment": (
        "maximo/ASSET.csv",
        {
            "EQUNR": "ASSETNUM",
            "EQKTX": "DESCRIPTION",
            "TECHIDENT": "ASSETTAG",
            "EQART": "ASSETTYPE",
            "TPLNR": "LOCATION",
            "HERST_TYPBZ": "MODELNUM",
            "ABCKZ": "PRIORITY",
        },
    ),
    "work_order": (
        "maximo/WORKORDER.csv",
        {
            "AUFNR": "WONUM",
            "AUART": "WORKTYPE",
            "EQUNR": "ASSETNUM",
            "KTEXT": "DESCRIPTION",
            "GSTRP": "SCHEDSTART",
            "GLTRP": "SCHEDFINISH",
        },
    ),
    "component": (
        "maximo/WPMATERIAL.csv",
        {
            "RSNUM": "REQUESTNUM",
            "RSPOS": "LINENUM",
            "AUFNR": "WONUM",
            "MATNR": "ITEMNUM",
            "BDMNG": "ITEMQTY",
            "MEINS": "ORDERUNIT",
            "BDTER": "REQUIREDATE",
        },
    ),
    "material": (
        "maximo/ITEM.csv",
        {
            "MATNR": "ITEMNUM",
            "MAKTX": "DESCRIPTION",
            "MFRPN": "CATALOGCODE",
            "LIFNR": "VENDOR",
            "PLIFZ": "LEADTIME",
            "BRGEW": "WEIGHT",
            "MEINS": "ISSUEUNIT",
        },
    ),
    "vendor": (
        "maximo/COMPANIES.csv",
        {"LIFNR": "COMPANY", "NAME1": "NAME", "SMTP_ADDR": "EMAILADDRESS"},
    ),
    "po_line": (
        "maximo/POLINE.csv",
        {
            "EBELN": "PONUM",
            "EBELP": "POLINENUM",
            "LIFNR": "VENDOR",
            "MATNR": "ITEMNUM",
            "TXZ01": "DESCRIPTION",
            "MENGE": "ORDERQTY",
            "BEDAT": "ORDERDATE",
            "EINDT": "VENDELIVERYDATE",
            "AUFNR": "REFWO",
        },
    ),
    "stock": (
        "maximo/INVBALANCES.csv",
        {"MATNR": "ITEMNUM", "WERKS": "SITEID", "LGORT": "LOCATION", "LABST": "CURBAL"},
    ),
    "sailing": (
        "logistics/voyages.csv",
        {"sailing_id": "VOYAGE_NO", "vessel": "VESSEL_NAME", "depart_date": "ETD", "route": "DEST"},
    ),
    "manifest_line": (
        "logistics/cargo_lines.csv",
        {
            "line_id": "CARGO_ID",
            "sailing_id": "VOYAGE_NO",
            "customer_ref": "PO_REFERENCE",
            "description": "CARGO_DESC",
            "qty": "PIECES",
            "gross_kg": "WEIGHT_KG",
            "unit_type": "PACKAGE",
        },
    ),
}
MAXIMO_UDF = "MAXIMO WONUM"


def dialect_mapping(asset: Asset) -> Mapping:
    """Where each canonical field sits in this asset's raw extracts (the truth)."""
    if asset.dialect == "sap":
        return identity_mapping()
    if asset.dialect != "maximo":
        raise ValueError(f"unknown dialect {asset.dialect!r}")
    m = Mapping(udf=asset.renames.get(f"udf:{MAXIMO_UDF}", MAXIMO_UDF))
    for entity, (path, cols) in MAXIMO.items():
        m.tables[entity] = path
        for f in ENTITIES[entity].fields:
            raw = cols[f.name]
            m.fields[f"{entity}.{f.name}"] = asset.renames.get(f"{path}:{raw}", raw)
            if f.kind == "date":
                m.formats[f"{entity}.{f.name}"] = ETD_DATE if entity == "sailing" else ERP_DATE
    return m


def _d(d: date) -> str:
    return d.strftime(ERP_DATE).upper()


def _qty(n: float) -> str:
    return f"{n:.2f}"


def _write(path: Path, rows: list[dict], renames: dict[str, str], rel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [renames.get(f"{rel}:{k}", k) for k in rows[0]] if rows else []
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([list(r.values()) for r in rows])


def export_maximo(world: World, asset: Asset, out_dir: str | Path, profile: str = "A") -> dict:
    """Write ``world`` as a Maximo-style asset's extracts, with profile ``profile``'s mess."""
    out = Path(out_dir)
    ex = out / "extracts"
    c = Corruptor(profile, world.seed)
    site = f"{asset.code}1"
    mat_by_no = {m.mat_no: m for m in world.materials}
    comp_by_key = {x.key: x for x in world.components}
    eq_by_no = {e.equip_no: e for e in world.equipment}
    wo_by_no = {w.wo_no: w for w in world.work_orders}
    pl_by_key = {pl.key: pl for pl in world.po_lines}
    vendor_by_id = {v.vendor_id: v for v in world.vendors}
    tables: dict[str, list[dict]] = {}

    tables["maximo/ASSET.csv"] = [
        {
            "ASSETNUM": e.equip_no,
            "SITEID": site,
            "DESCRIPTION": c.typo(f"{e.maker_model.split('/')[0]} {e.eq_class} {e.system}"),
            "ASSETTAG": "" if c.chance("erp_techident_missing") else e.tag,
            "ASSETTYPE": e.eq_class,
            "LOCATION": f"{asset.code}-{e.system.replace(' ', '')[:8]}",
            "MODELNUM": e.maker_model,
            "PRIORITY": PRIORITY[e.criticality],
        }
        for e in world.equipment
    ]
    tables["maximo/WORKORDER.csv"] = [
        {
            "WONUM": w.wo_no,
            "SITEID": site,
            "WORKTYPE": WORK_TYPES[w.wo_type],
            "STATUS": "APPR",
            "ASSETNUM": w.equip_no,
            "DESCRIPTION": c.typo(w.description),
            "SCHEDSTART": _d(w.planned_start),
            "SCHEDFINISH": _d(w.planned_finish),
        }
        for w in world.work_orders
    ]
    tables["maximo/WPMATERIAL.csv"] = [
        {
            "WONUM": x.wo_no,
            "REQUESTNUM": x.res_no,
            "LINENUM": x.item,
            "ITEMNUM": x.mat_no,
            "ITEMQTY": _qty(x.qty),
            "ORDERUNIT": "EACH",
            "REQUIREDATE": _d(x.req_date),
        }
        for x in world.components
    ]
    tables["maximo/ITEM.csv"] = [
        {
            "ITEMNUM": m.mat_no,
            "DESCRIPTION": m.description,
            "COMMODITYGROUP": m.eq_class,
            "CATALOGCODE": m.vendor_part,
            "VENDOR": m.vendor_id,
            "LEADTIME": m.lead_time_days,
            "WEIGHT": m.unit_weight_kg,
            "ISSUEUNIT": "EACH",
        }
        for m in world.materials
    ]
    tables["maximo/COMPANIES.csv"] = [
        {
            "COMPANY": v.vendor_id,
            "NAME": v.name,
            "TYPE": "V",
            "EMAILADDRESS": f"orders@{v.domain}",
            "PHONE": f"+44 1224 {500000 + i * 137:06d}",
        }
        for i, v in enumerate(world.vendors)
    ]
    tables["maximo/POLINE.csv"] = [
        {
            "PONUM": pl.po_no,
            "POLINENUM": pl.item,
            "SITEID": site,
            "VENDOR": pl.vendor_id,
            "ITEMNUM": pl.mat_no,
            "DESCRIPTION": mat_by_no[pl.mat_no].description[:40],
            "ORDERQTY": _qty(pl.qty),
            "ORDERDATE": _d(pl.order_date),
            "REQDELIVERYDATE": _d(comp_by_key[pl.component_key].req_date),
            "VENDELIVERYDATE": _d(pl.delivery_date),
            "REFWO": "" if c.chance("erp_po_wo_missing") else pl.wo_no,
        }
        for pl in world.po_lines
    ]
    tables["maximo/INVBALANCES.csv"] = [
        {
            "ITEMNUM": m,
            "SITEID": site,
            "LOCATION": f"{asset.code}-STORE",
            "BINNUM": f"B{int(m) % 40 + 1:02d}",
            "CURBAL": _qty(q),
        }
        for m, q in sorted(world.stock.items())
    ]
    tables["logistics/voyages.csv"] = [
        {
            "VOYAGE_NO": s.sailing_id,
            "VESSEL_NAME": s.vessel,
            "ETD": f"{s.depart.strftime('%d/%m/%Y')} 06:00",
            "DEST": f"{asset.code} FIELD",
            "STATUS": "PLANNED",
        }
        for s in world.sailings
    ]
    cargo = []
    for ml in world.manifest_lines:
        pl = pl_by_key[ml.po_line_keys[0]]
        desc = c.abbreviate(mat_by_no[pl.mat_no].spare)
        if c.rng.random() < 0.35:  # the packer sometimes writes which equipment it is for
            desc += f" FOR {c.tag(eq_by_no[wo_by_no[pl.wo_no].equip_no].tag)}"
        cargo.append(
            {
                "CARGO_ID": ml.line_id,
                "VOYAGE_NO": ml.sailing_id,
                "CONSIGNEE": f"{OPERATOR.upper()} / {asset.name.upper()}",
                "PO_REFERENCE": c.po_ref(pl.po_no, pl.item),
                "CARGO_DESC": c.typo(desc),
                "PIECES": ml.qty,
                "WEIGHT_KG": round(mat_by_no[pl.mat_no].unit_weight_kg * ml.qty * 1.1, 1),
                "PACKAGE": c.rng.choice(["BASKET", "HALF HEIGHT", "CCU", "PALLET"]),
            }
        )
    tables["logistics/cargo_lines.csv"] = cargo
    for rel, rows in tables.items():
        _write(ex / rel, rows, asset.renames, rel)

    xer = ex / "p6" / f"{asset.name.split()[-1].lower()}_ta27.xer"
    _write_xer(world, xer, c, wo_by_no, eq_by_no)
    udf = asset.renames.get(f"udf:{MAXIMO_UDF}", MAXIMO_UDF)
    xer.write_text(
        xer.read_text()
        .replace(f"\t{ASSET_CODE}-TA27\n", f"\t{asset.code}-TA27-{asset.name.split()[-1]}\n")
        .replace("\tTASK\tSAP WO\n", f"\tTASK\t{udf}\n")
    )
    _write_emails(world, ex / "supplier_mail", c, pl_by_key, mat_by_no, vendor_by_id)
    buyer = f"procurement.{asset.name.split()[-1].lower()}".encode()
    for p in (ex / "supplier_mail").glob("*.eml"):
        p.write_bytes(p.read_bytes().replace(b"procurement.kestrel", buyer))

    links = world.links()
    (out / "truth").mkdir(parents=True, exist_ok=True)
    with (out / "truth" / "links.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["link_type", "from_source", "from_key", "to_source", "to_key"])
        w.writerows(links)
    run = {
        "operator": OPERATOR,
        "asset": asset.name,
        "asset_code": asset.code,
        "dialect": asset.dialect,
        "seed": world.seed,
        "scale": world.scale,
        "profile": profile,
        "profile_version": PROFILES[profile]["version"],
        "window": [world.window_start.isoformat(), world.window_end.isoformat()],
        "turnaround": [world.turnaround_start.isoformat(), world.turnaround_end.isoformat()],
        "counts": {
            "equipment": len(world.equipment),
            "materials": len(world.materials),
            "work_orders": len(world.work_orders),
            "components": len(world.components),
            "activities": len(world.activities),
            "po_lines": len(world.po_lines),
            "sailings": len(world.sailings),
            "manifest_lines": len(world.manifest_lines),
            "emails": len(world.emails),
            "links": len(links),
        },
    }
    (out / "truth" / "run.json").write_text(json.dumps(run, indent=2))
    return run
