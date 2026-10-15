"""Write a World as the messy extracts a real operator would send, plus the hidden truth.

Layout under the output directory:

    extracts/            what connectors and agents may read
      erp_pm/            equipment.csv, work_orders.csv, wo_components.csv
      erp_mm/            materials.csv, vendors.csv, po_lines.csv, stock.csv
      p6/                kestrel.xer
      vessel/            sailings.csv, manifest_lines.csv
      supplier_mail/     <message-id>.eml
    truth/               never given to agents: links.csv and run.json
"""

from __future__ import annotations

import csv
import json
import zlib
from dataclasses import replace
from datetime import UTC, date, datetime, time
from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path

from fpso.synthetic.corrupt import PROFILES, Corruptor
from fpso.synthetic.truth import ASSET, ASSET_CODE, OPERATOR, ManifestLine, World

BUYERS = ["d.reyes", "l.park", "procurement.kestrel"]


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _sap_date(d: date) -> str:
    return d.strftime("%Y%m%d")


def export(world: World, out_dir: str | Path, profile: str = "A") -> dict:
    out = Path(out_dir)
    ex = out / "extracts"
    c = Corruptor(profile, world.seed)
    eq_by_no = {e.equip_no: e for e in world.equipment}
    mat_by_no = {m.mat_no: m for m in world.materials}
    vendor_by_id = {v.vendor_id: v for v in world.vendors}
    pl_by_key = {pl.key: pl for pl in world.po_lines}
    wo_by_no = {w.wo_no: w for w in world.work_orders}

    # ERP plant maintenance (SAP-style column names).
    _write_csv(
        ex / "erp_pm" / "equipment.csv",
        ["EQUNR", "EQKTX", "TECHIDENT", "EQART", "TPLNR", "HERST_TYPBZ", "ABCKZ"],
        [
            [
                e.equip_no,
                c.typo(f"{e.maker_model.split('/')[0]} {e.eq_class} {e.system}"),
                "" if c.chance("erp_techident_missing") else e.tag,
                e.eq_class,
                f"{ASSET_CODE}-{e.system.replace(' ', '')[:8]}",
                e.maker_model,
                e.criticality,
            ]
            for e in world.equipment
        ],
    )
    _write_csv(
        ex / "erp_pm" / "work_orders.csv",
        ["AUFNR", "AUART", "EQUNR", "KTEXT", "GSTRP", "GLTRP"],
        [
            [
                w.wo_no,
                w.wo_type,
                w.equip_no,
                c.typo(w.description),
                _sap_date(w.planned_start),
                _sap_date(w.planned_finish),
            ]
            for w in world.work_orders
        ],
    )
    _write_csv(
        ex / "erp_pm" / "wo_components.csv",
        ["RSNUM", "RSPOS", "AUFNR", "MATNR", "BDMNG", "MEINS", "BDTER"],
        [
            [x.res_no, x.item, x.wo_no, x.mat_no, x.qty, "EA", _sap_date(x.req_date)]
            for x in world.components
        ],
    )

    # ERP materials.
    _write_csv(
        ex / "erp_mm" / "materials.csv",
        ["MATNR", "MAKTX", "MFRPN", "LIFNR", "PLIFZ", "BRGEW", "MEINS"],
        [
            [
                m.mat_no,
                m.description,
                m.vendor_part,
                m.vendor_id,
                m.lead_time_days,
                m.unit_weight_kg,
                m.unit,
            ]
            for m in world.materials
        ],
    )
    _write_csv(
        ex / "erp_mm" / "vendors.csv",
        ["LIFNR", "NAME1", "SMTP_ADDR"],
        [[v.vendor_id, v.name, f"orders@{v.domain}"] for v in world.vendors],
    )
    _write_csv(
        ex / "erp_mm" / "po_lines.csv",
        ["EBELN", "EBELP", "LIFNR", "MATNR", "TXZ01", "MENGE", "BEDAT", "EINDT", "AUFNR"],
        [
            [
                pl.po_no,
                pl.item,
                pl.vendor_id,
                pl.mat_no,
                mat_by_no[pl.mat_no].description[:40],
                pl.qty,
                _sap_date(pl.order_date),
                _sap_date(pl.delivery_date),
                "" if c.chance("erp_po_wo_missing") else pl.wo_no,
            ]
            for pl in world.po_lines
        ],
    )
    _write_csv(
        ex / "erp_mm" / "stock.csv",
        ["MATNR", "WERKS", "LGORT", "LABST"],
        [[m, "KST1", "SB01", q] for m, q in sorted(world.stock.items())],
    )

    _write_xer(world, ex / "p6" / "kestrel.xer", c, wo_by_no, eq_by_no)
    manifest = _write_vessel(world, ex / "vessel", c, pl_by_key, mat_by_no, eq_by_no, wo_by_no)
    _write_emails(world, ex / "supplier_mail", c, pl_by_key, mat_by_no, vendor_by_id)

    # Truth uses the manifest lines as exported, since merging changes which lines exist.
    links = [lk for lk in world.links() if lk[0] != "po_line_manifest_line"]
    for ml in manifest:
        for k in ml.po_line_keys:
            links.append(("po_line_manifest_line", "erp-mm", k, "vessel", ml.line_id))
    _write_csv(
        out / "truth" / "links.csv",
        ["link_type", "from_source", "from_key", "to_source", "to_key"],
        [list(lk) for lk in links],
    )
    counts = {
        "equipment": len(world.equipment),
        "materials": len(world.materials),
        "work_orders": len(world.work_orders),
        "components": len(world.components),
        "activities": len(world.activities),
        "po_lines": len(world.po_lines),
        "sailings": len(world.sailings),
        "manifest_lines": len(manifest),
        "emails": len(world.emails),
        "links": len(links),
    }
    run = {
        "operator": OPERATOR,
        "asset": ASSET,
        "seed": world.seed,
        "profile": profile,
        "profile_version": PROFILES[profile]["version"],
        "profile_settings": PROFILES[profile],
        "window": [world.window_start.isoformat(), world.window_end.isoformat()],
        "turnaround": [world.turnaround_start.isoformat(), world.turnaround_end.isoformat()],
        "counts": counts,
    }
    (out / "truth" / "run.json").write_text(json.dumps(run, indent=2))
    return run


def _write_xer(world, path: Path, c: Corruptor, wo_by_no, eq_by_no) -> None:
    """A minimal Primavera P6 XER file: tab-separated %T/%F/%R tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wbs_ids = {a.wbs: 7000 + i for i, a in enumerate({a.wbs: a for a in world.activities}.values())}
    lines = [f"ERMHDR\t21.12\t{world.window_start.isoformat()}\tProject\tadmin\t{OPERATOR}"]

    def table(name: str, fields: list[str], rows: list[list]) -> None:
        lines.append(f"%T\t{name}")
        lines.append("%F\t" + "\t".join(fields))
        for r in rows:
            lines.append("%R\t" + "\t".join(str(v) for v in r))

    table("PROJECT", ["proj_id", "proj_short_name"], [[1, f"{ASSET_CODE}-TA27"]])
    table(
        "PROJWBS",
        ["wbs_id", "proj_id", "wbs_short_name", "wbs_name"],
        [[i, 1, w.split(".")[-1], w.replace("_", " ")] for w, i in wbs_ids.items()],
    )
    task_rows, udf_rows, pred_rows = [], [], []
    for a in world.activities:
        name = a.name
        if a.equip_no:
            eq = eq_by_no[a.equip_no]
            name = name.replace(eq.tag, c.tag(eq.tag), 1)
        if a.wo_no and c.chance("p6_wo_in_name"):
            name = f"{name} (WO {a.wo_no})"
        task_rows.append(
            [
                a.task_id,
                1,
                wbs_ids[a.wbs],
                a.act_id,
                c.typo(name),
                f"{a.start.isoformat()} 07:00",
                f"{a.finish.isoformat()} 19:00",
                "TK_NotStart",
            ]
        )
        if a.wo_no and not c.chance("p6_wo_udf_missing"):
            udf_rows.append([901, a.task_id, a.wo_no])
        for p in a.predecessors:
            pred_rows.append([len(pred_rows) + 1, a.task_id, p, "PR_FS"])
    table(
        "TASK",
        [
            "task_id",
            "proj_id",
            "wbs_id",
            "task_code",
            "task_name",
            "target_start_date",
            "target_end_date",
            "status_code",
        ],
        task_rows,
    )
    table("UDFTYPE", ["udf_type_id", "table_name", "udf_type_label"], [[901, "TASK", "SAP WO"]])
    table("UDFVALUE", ["udf_type_id", "fk_id", "udf_text"], udf_rows)
    table("TASKPRED", ["task_pred_id", "task_id", "pred_task_id", "pred_type"], pred_rows)
    lines.append("%E")
    path.write_text("\n".join(lines) + "\n")


def _write_vessel(world, d: Path, c: Corruptor, pl_by_key, mat_by_no, eq_by_no, wo_by_no):
    _write_csv(
        d / "sailings.csv",
        ["sailing_id", "vessel", "depart_date", "route"],
        [[s.sailing_id, s.vessel, s.depart.isoformat(), "SHOREBASE-KST"] for s in world.sailings],
    )
    # Optionally merge two lines on the same sailing from the same PO into one line.
    exported: list[ManifestLine] = []
    i = 0
    lines = sorted(
        world.manifest_lines,
        key=lambda m: (m.sailing_id, pl_by_key[m.po_line_keys[0]].po_no, m.line_id),
    )
    while i < len(lines):
        ml = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if (
            nxt
            and nxt.sailing_id == ml.sailing_id
            and pl_by_key[nxt.po_line_keys[0]].po_no == pl_by_key[ml.po_line_keys[0]].po_no
            and c.chance("manifest_merge_lines")
        ):
            exported.append(
                replace(ml, po_line_keys=ml.po_line_keys + nxt.po_line_keys, qty=ml.qty + nxt.qty)
            )
            i += 2
            continue
        exported.append(ml)
        i += 1

    rows = []
    for ml in exported:
        pls = [pl_by_key[k] for k in ml.po_line_keys]
        first = pls[0]
        mat = mat_by_no[first.mat_no]
        if len(pls) > 1:
            desc = f"ASSORTED SPARES {first.po_no}"
        else:
            desc = c.abbreviate(mat.spare)
            wo = wo_by_no[first.wo_no]
            if c.rng.random() < 0.35:  # the packer sometimes writes which equipment it is for
                desc += f" FOR {c.tag(eq_by_no[wo.equip_no].tag)}"
            desc = c.typo(desc)
        weight = round(sum(mat_by_no[p.mat_no].unit_weight_kg for p in pls) * ml.qty * 1.1, 1)
        rows.append(
            [
                ml.line_id,
                ml.sailing_id,
                c.po_ref(first.po_no, first.item),
                desc,
                f"{ml.qty} {c.rng.choice(['EA', 'PCS', 'x', ''])}".strip(),
                weight,
                c.rng.choice(["BASKET", "HALF HEIGHT", "CCU", "PALLET"]),
            ]
        )
    _write_csv(
        d / "manifest_lines.csv",
        ["line_id", "sailing_id", "customer_ref", "description", "qty", "gross_kg", "unit_type"],
        rows,
    )
    return exported


def _write_emails(world, d: Path, c: Corruptor, pl_by_key, mat_by_no, vendor_by_id) -> None:
    d.mkdir(parents=True, exist_ok=True)
    fmt = c.p["email_date_format"]
    for e in world.emails:
        v = vendor_by_id[e.vendor_id]
        pls = [pl_by_key[k] for k in e.po_line_keys]
        sales_order = (
            f"SO-{zlib.crc32(f'{e.po_no}{v.vendor_id}{world.seed}'.encode()) % 900000 + 100000}"
        )
        placement = c.email_po_placement()
        item_lines = "\n".join(
            f"  - {mat_by_no[p.mat_no].vendor_part}  {mat_by_no[p.mat_no].spare.title()}  "
            f"qty {p.qty}"
            for p in pls
        )
        if e.kind == "ack":
            subject = "Order acknowledgement"
            body = f"Thank you for your order. We confirm the following items:\n{item_lines}\n"
            body += "Delivery as per your requested dates.\n"
        elif e.kind == "slip":
            subject = "Revised delivery date"
            body = (
                f"Unfortunately the item(s) below are delayed due to sub-supplier capacity:\n"
                f"{item_lines}\nNew expected dispatch: {e.new_date.strftime(fmt)}\n"
            )
        else:
            subject = "Dispatch notification"
            body = f"The following goods have been dispatched to your shore base:\n{item_lines}\n"
        subject = f"{subject} - our ref {sales_order}"
        if placement == "subject":
            po_text = c.rng.choice([e.po_no, f"PO {e.po_no}", f"PO#{e.po_no}", f"P.O. {e.po_no}"])
            subject = f"{po_text} / {subject}"
        elif placement == "body":
            body = f"Your purchase order: {e.po_no}\n\n{body}"
        elif placement == "quoted":
            body += (
                f"\n> From: procurement.kestrel@northwind-offshore.example\n> Our PO {e.po_no}\n"
            )
        msg = EmailMessage()
        msg["Message-ID"] = f"<{e.message_id}@{v.domain}>"
        msg["From"] = f"{v.name} <orders@{v.domain}>"
        msg["To"] = f"{c.rng.choice(BUYERS)}@northwind-offshore.example"
        msg["Date"] = format_datetime(datetime.combine(e.sent, time(9, 30), tzinfo=UTC))
        msg["Subject"] = c.typo(subject)
        msg.set_content(f"Dear Buyer,\n\n{body}\nRegards,\n{v.name}\n")
        (d / f"{e.message_id}.eml").write_bytes(bytes(msg))
