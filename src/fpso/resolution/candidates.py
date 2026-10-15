"""Candidate generation: cheap, high-recall proposals for the five Phase 0 link types.

Reads only through connectors. Each candidate carries the signals that fired and a rough
rule score; the model adjudicates later (plan WS5 step 2). The goal here is recall: a true
link that never becomes a candidate can never be found.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from email.utils import parseaddr

from fpso.connectors import Connector
from fpso.resolution.normalize import (
    containment,
    find_eight_digit_runs,
    find_po_numbers,
    find_six_digit_runs,
    find_tags,
    leading_int,
    norm_tag,
    parse_date,
    tag_group,
)

LINK_TYPES = (
    "equipment_activity",
    "work_order_activity",
    "component_po_line",
    "po_line_manifest_line",
    "po_email",
)

# Which source and entity type sit on each side of a link type.
LINK_ENDS = {
    "equipment_activity": (("erp-pm", "equipment"), ("p6", "activity")),
    "work_order_activity": (("erp-pm", "work_order"), ("p6", "activity")),
    "component_po_line": (("erp-pm", "component"), ("erp-mm", "po_line")),
    "po_line_manifest_line": (("erp-mm", "po_line"), ("vessel", "manifest_line")),
    "po_email": (("erp-mm", "po"), ("supplier-mail", "message")),
}

MAX_PER_ENTITY = 12


@dataclass
class Candidate:
    link_type: str
    from_key: str
    to_key: str
    score: float
    signals: dict = field(default_factory=dict)

    @property
    def pair(self) -> tuple[str, str, str]:
        return (self.link_type, self.from_key, self.to_key)


def _days(a: date | None, b: date | None) -> int | None:
    return None if a is None or b is None else (a - b).days


def _keep_best(cands: list[Candidate], limit: int = MAX_PER_ENTITY) -> list[Candidate]:
    best: dict[tuple, Candidate] = {}
    for c in cands:
        if c.pair not in best or c.score > best[c.pair].score:
            best[c.pair] = c
        else:
            best[c.pair].signals.update(c.signals)
    return sorted(best.values(), key=lambda c: -c.score)[:limit]


class CandidateGenerator:
    def __init__(self, connectors: dict[str, Connector]):
        self.c = connectors
        pm, mm = connectors["erp-pm"], connectors["erp-mm"]
        self.work_orders = {r.key: r for r in pm.iter("work_order")}
        self.equipment = {r.key: r for r in pm.iter("equipment")}
        self.po_lines = {r.key: r for r in mm.iter("po_line")}
        self.materials = {r.key: r for r in mm.iter("material")}

        # Tags per equipment from the equipment master and from work order texts.
        self.tags_by_equipment: dict[str, set[str]] = defaultdict(set)
        for e in self.equipment.values():
            if e.fields.get("TECHIDENT"):
                self.tags_by_equipment[e.key].add(norm_tag(e.fields["TECHIDENT"]))
        for w in self.work_orders.values():
            first = (w.fields.get("KTEXT") or "").split(" ")[0]
            if find_tags(first):
                self.tags_by_equipment[w.fields["EQUNR"]].update(find_tags(first))
        self.equipment_by_tag: dict[str, set[str]] = defaultdict(set)
        self.equipment_by_group: dict[str, set[str]] = defaultdict(set)
        for eq, tags in self.tags_by_equipment.items():
            for t in tags:
                self.equipment_by_tag[t].add(eq)
                self.equipment_by_group[tag_group(t)].add(eq)
        self.wos_by_equipment: dict[str, list] = defaultdict(list)
        for w in self.work_orders.values():
            self.wos_by_equipment[w.fields["EQUNR"]].append(w)

        self.lines_by_po: dict[str, list] = defaultdict(list)
        self.lines_by_material: dict[str, list] = defaultdict(list)
        for pl in self.po_lines.values():
            self.lines_by_po[pl.fields["EBELN"]].append(pl)
            self.lines_by_material[pl.fields["MATNR"]].append(pl)
        self.components_by_wo: dict[str, list] = defaultdict(list)
        for comp in pm.iter("component"):
            self.components_by_wo[comp.fields["AUFNR"]].append(comp)
        self.po_by_suffix: dict[str, set[str]] = defaultdict(set)
        for po in self.lines_by_po:
            self.po_by_suffix[po[-6:]].add(po)

    # Link types 1 and 2 share the activity scan.
    def _activity_candidates(self) -> tuple[list[Candidate], list[Candidate]]:
        wo_out, eq_out = [], []
        for act in self.c["p6"].iter("activity"):
            f = act.fields
            start = parse_date(f["target_start_date"])
            wo_c: list[Candidate] = []
            udf_wo = (f.get("udf") or {}).get("SAP WO", "")
            if udf_wo in self.work_orders:
                wo_c.append(
                    Candidate("work_order_activity", udf_wo, act.key, 0.98, {"udf": udf_wo})
                )
            for n in find_eight_digit_runs(f["task_name"]):
                if n in self.work_orders:
                    wo_c.append(Candidate("work_order_activity", n, act.key, 0.95, {"in_name": n}))
            tags = find_tags(f["task_name"])
            eq_hits: dict[str, tuple[float, dict]] = {}
            for t in tags:
                for eq in self.equipment_by_tag.get(t, ()):
                    eq_hits[eq] = (0.8, {"tag": t})
                for eq in self.equipment_by_group.get(tag_group(t), ()):
                    eq_hits.setdefault(eq, (0.45, {"tag_group": tag_group(t)}))
            for eq, (score, sig) in eq_hits.items():
                eq_out.append(Candidate("equipment_activity", eq, act.key, score, dict(sig)))
                for w in self.wos_by_equipment.get(eq, ()):
                    gap = _days(parse_date(w.fields["GSTRP"]), start)
                    if gap is not None and abs(gap) <= 7:
                        s = score * (0.85 if gap == 0 else 0.6)
                        wo_c.append(
                            Candidate(
                                "work_order_activity",
                                w.key,
                                act.key,
                                s,
                                {**sig, "start_gap_days": gap},
                            )
                        )
            wo_c = _keep_best(wo_c)
            wo_out.extend(wo_c)
            # Equipment through the candidate work orders.
            for c in wo_c:
                eq = self.work_orders[c.from_key].fields["EQUNR"]
                eq_out.append(
                    Candidate(
                        "equipment_activity", eq, act.key, c.score * 0.97, {"via_wo": c.from_key}
                    )
                )
        per_act: dict[str, list[Candidate]] = defaultdict(list)
        for c in eq_out:
            per_act[c.to_key].append(c)
        eq_out = [x for cs in per_act.values() for x in _keep_best(cs)]
        return eq_out, wo_out

    def _component_candidates(self) -> list[Candidate]:
        out = []
        for comp in self.c["erp-pm"].iter("component"):
            f = comp.fields
            need = parse_date(f["BDTER"])
            cands = []
            for pl in self.lines_by_material.get(f["MATNR"], ()):
                g = pl.fields
                sig: dict = {"material": f["MATNR"]}
                score = 0.3
                if g.get("AUFNR") and g["AUFNR"] == f["AUFNR"]:
                    sig["work_order"] = f["AUFNR"]
                    score += 0.5
                elif g.get("AUFNR"):
                    continue  # the PO line is booked to a different work order
                if str(g["MENGE"]) == str(f["BDMNG"]):
                    sig["qty"] = True
                    score += 0.1
                gap = _days(need, parse_date(g["EINDT"]))
                sig["need_minus_delivery_days"] = gap
                if gap is None or not -45 <= gap <= 60:
                    if "work_order" not in sig:
                        continue
                else:
                    score += 0.1 * (1 - abs(gap - 15) / 60)
                cands.append(Candidate("component_po_line", comp.key, pl.key, round(score, 3), sig))
            out.extend(_keep_best(cands))
        return out

    def _manifest_candidates(self) -> list[Candidate]:
        out = []
        sailings = {r.key: r for r in self.c["vessel"].iter("sailing")}
        for ml in self.c["vessel"].iter("manifest_line"):
            f = ml.fields
            depart = (
                parse_date(sailings[f["sailing_id"]].fields["depart_date"])
                if f["sailing_id"] in sailings
                else None
            )
            ref_text = f"{f['customer_ref']} {f['description']}"
            pos = {p for p in find_po_numbers(ref_text) if p in self.lines_by_po}
            if not pos:
                for s in find_six_digit_runs(f["customer_ref"]):
                    pos |= self.po_by_suffix.get(s, set())
            qty = leading_int(f["qty"])
            # Text before any 'FOR <tag>' describes the item; the tag names the equipment.
            item_text = f["description"].split(" FOR ")[0]
            tag_materials = {
                comp.fields["MATNR"]
                for t in find_tags(f["description"])
                for eq in self.equipment_by_tag.get(t, set())
                | self.equipment_by_group.get(tag_group(t), set())
                for w in self.wos_by_equipment.get(eq, ())
                for comp in self.components_by_wo.get(w.key, ())
            }
            pool = (
                [pl for p in pos for pl in self.lines_by_po[p]] if pos else self.po_lines.values()
            )
            cands = []
            for pl in pool:
                g = pl.fields
                lag = _days(depart, parse_date(g["EINDT"]))
                if lag is None or not -5 <= lag <= 45:
                    continue
                mat = self.materials.get(g["MATNR"])
                sim = containment(item_text, mat.fields["MAKTX"] if mat else g["TXZ01"])
                sig = {"ship_lag_days": lag, "desc_sim": round(sim, 2)}
                score = 0.1 + 0.3 * sim + 0.1 * max(0.0, 1 - abs(lag - 3) / 20)
                if g["MATNR"] in tag_materials:
                    sig["tag_material"] = True
                    score += 0.15
                if pos:
                    sig["po_ref"] = g["EBELN"]
                    score += 0.4
                elif sim < 0.4 and "tag_material" not in sig:
                    continue
                if qty is not None and qty <= int(g["MENGE"]):
                    sig["qty_fits"] = True
                    score += 0.05 if qty < int(g["MENGE"]) else 0.1
                cands.append(
                    Candidate("po_line_manifest_line", pl.key, ml.key, round(score, 3), sig)
                )
            out.extend(_keep_best(cands, MAX_PER_ENTITY if pos else 2 * MAX_PER_ENTITY))
        return out

    def _email_candidates(self) -> list[Candidate]:
        out = []
        mm = self.c["erp-mm"]
        vendor_by_domain = {
            r.fields["SMTP_ADDR"].split("@")[-1].lower(): r.key for r in mm.iter("vendor")
        }
        parts_to_material = {r.fields["MFRPN"]: r.key for r in mm.iter("material")}
        pos = {r.key: r for r in mm.iter("po")}
        for msg in self.c["supplier-mail"].iter("message"):
            f = msg.fields
            sent = parse_date(f["date"])
            domain = parseaddr(f["from"])[1].split("@")[-1].lower()
            vendor = vendor_by_domain.get(domain)
            mats = {parts_to_material[p] for p in parts_to_material if p in f["body"]}
            cands = []
            for where in ("subject", "body"):
                for po in find_po_numbers(f[where]):
                    if po in pos:
                        cands.append(Candidate("po_email", po, msg.key, 0.95, {"po_in": where}))
            if vendor:
                for po_no, po in pos.items():
                    if po.fields["LIFNR"] != vendor:
                        continue
                    po_mats = {pl.fields["MATNR"] for pl in self.lines_by_po[po_no]}
                    overlap = len(mats & po_mats) / len(mats | po_mats) if mats else 0.0
                    if overlap == 0:
                        continue
                    gap = _days(sent, parse_date(po.fields["BEDAT"]))
                    if gap is None or gap < -2:
                        continue
                    cands.append(
                        Candidate(
                            "po_email",
                            po_no,
                            msg.key,
                            round(0.3 + 0.5 * overlap, 3),
                            {
                                "vendor": vendor,
                                "parts_overlap": round(overlap, 2),
                                "days_since_order": gap,
                            },
                        )
                    )
            out.extend(_keep_best(cands))
        return out

    def generate(self) -> list[Candidate]:
        eq, wo = self._activity_candidates()
        return (
            eq
            + wo
            + self._component_candidates()
            + self._manifest_candidates()
            + self._email_candidates()
        )
