"""Supplier Liaison agent (Phase 1, autonomy level 1: recommend).

It drafts the messages a buyer would send, and never sends them:

- ``no_ack``: a PO still open (a line not yet due) whose supplier has not acknowledged it
  ``ACK_DAYS`` after it was placed;
- ``expedite``: one message per PO with lines the Procurement agent wants expedited, naming
  each part, the date it is needed on board, and what the delay does to the turnaround.

Which emails belong to a PO comes from the chain walk's PO to email links. What kind of
email each one is (acknowledgement, delay notice, dispatch notice) is read from fixed
phrases in its body, which needs no model; the delay dates themselves come from the
Readiness agent's email reading.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

from fpso.procurement import Recommendation
from fpso.readiness import ReadinessAgent
from fpso.resolution.normalize import parse_date
from fpso.schedule import Impact
from fpso.synthetic.truth import World

ACK_DAYS = 7
KIND_PHRASES = {
    "ack": re.compile(r"\bwe confirm\b|\backnowledg", re.I),
    "slip": re.compile(r"\bdelayed\b|\brevised delivery\b|\bnew expected\b", re.I),
    "asn": re.compile(r"\bdispatched\b|\bdispatch notification\b", re.I),
}


@dataclass
class Chase:
    po: str
    vendor: str
    kind: str  # no_ack, expedite
    to: str = ""
    subject: str = ""
    body: str = ""
    po_lines: list[str] = field(default_factory=list)


def email_kind(subject: str, body: str) -> str | None:
    text = f"{subject}\n{body}"
    for kind in ("slip", "asn", "ack"):  # a delay notice may also 'confirm' items
        if KIND_PHRASES[kind].search(text):
            return kind
    return None


class LiaisonAgent:
    def __init__(self, chain: ReadinessAgent):
        self.chain = chain
        mm = chain.c["erp-mm"]
        self.vendors = {r.key: r.fields for r in mm.iter("vendor")}
        self.pos = {r.key: r.fields for r in mm.iter("po")}

    def _open_pos(self, as_of: date) -> list[str]:
        """POs with a line not yet due, placed at least ACK_DAYS ago."""
        out = []
        for po, f in self.pos.items():
            placed = parse_date(f["BEDAT"])
            lines = [self.chain.po_lines[f"{po}-{item}"].fields for item in f["lines"]]
            due = max((parse_date(x["EINDT"]) or date.min for x in lines), default=date.min)
            if placed and placed + timedelta(days=ACK_DAYS) <= as_of and due >= as_of:
                out.append(po)
        return sorted(out)

    def unacknowledged(self, as_of: date) -> list[str]:
        pos = self._open_pos(as_of)
        self.chain.resolve(self.chain._leading_targets("po_email", set(pos)))
        # An email nobody could link may be this PO's acknowledgement: don't chase on a guess.
        maybe = self.chain._open_sources("po_email", set(pos))
        out = []
        for po in pos:
            if po in maybe:
                continue
            kinds = set()
            for key in self.chain.linked_from("po_email", po):
                msg = self.chain.c["supplier-mail"].get("message", key).fields
                if parse_date(msg["date"]) <= as_of:
                    kinds.add(email_kind(msg["subject"], msg["body"]))
            if not kinds & {"ack", "slip", "asn"}:  # any reply shows the order was received
                out.append(po)
        return out

    def run(self, as_of: date, recs: list[Recommendation], impacts: list[Impact]) -> list[Chase]:
        chases = []
        for po in self.unacknowledged(as_of):
            f = self.pos[po]
            v = self.vendors.get(f["LIFNR"], {})
            chases.append(
                Chase(
                    po,
                    f["LIFNR"],
                    "no_ack",
                    v.get("SMTP_ADDR", ""),
                    f"PO {po}: please acknowledge",
                    f"Dear {v.get('NAME1', 'supplier')},\n\nWe placed purchase order {po} on "
                    f"{f['BEDAT']} and have not had your acknowledgement. Please confirm the "
                    "order and the delivery dates for each line.\n\nRegards,\nNorthwind "
                    "Offshore procurement",
                    [f"{po}-{i}" for i in f["lines"]],
                )
            )
        impact = {i.component: i for i in impacts}
        by_po: dict[str, list[Recommendation]] = defaultdict(list)
        for r in recs:
            if r.action == "expedite" and r.po_line:
                by_po[r.po_line.rsplit("-", 1)[0]].append(r)
        for po, rs in sorted(by_po.items()):
            f = self.pos.get(po, {})
            v = self.vendors.get(f.get("LIFNR", ""), {})
            lines = []
            for r in sorted(rs, key=lambda r: (r.need, r.po_line)):
                i = impact.get(r.component)
                effect = (
                    f"; at that date the turnaround end moves {i.turnaround_delay} "
                    f"day{'s' * (i.turnaround_delay != 1)}"
                    if i and i.turnaround_delay
                    else ""
                )
                lines.append(
                    f"- line {r.po_line.rsplit('-', 1)[1]} (material {r.material}): expected "
                    f"{r.expected}, needed on board by {r.need}{effect}"
                )
            chases.append(
                Chase(
                    po,
                    f.get("LIFNR", ""),
                    "expedite",
                    v.get("SMTP_ADDR", ""),
                    f"PO {po}: delivery needed earlier",
                    f"Dear {v.get('NAME1', 'supplier')},\n\nThe following lines on purchase "
                    f"order {po} are due too late for our offshore work:\n"
                    + "\n".join(lines)
                    + "\n\nPlease tell us the earliest date you can deliver each line, and "
                    "whether a part shipment is possible.\n\nRegards,\nNorthwind Offshore "
                    "procurement",
                    sorted(r.po_line for r in rs),
                )
            )
        return chases


def truth_unacknowledged(world: World, as_of: date) -> list[str]:
    lines_by_po: dict[str, list] = defaultdict(list)
    for pl in world.po_lines:
        lines_by_po[pl.po_no].append(pl)
    replied = {e.po_no for e in world.emails if e.sent <= as_of}
    out = []
    for po, lines in lines_by_po.items():
        placed = min(pl.order_date for pl in lines)
        due = max(pl.delivery_date for pl in lines)
        if placed + timedelta(days=ACK_DAYS) <= as_of and due >= as_of and po not in replied:
            out.append(po)
    return sorted(out)


def score_chases(chases: list[Chase], truth_no_ack: list[str]) -> dict:
    a = {c.po for c in chases if c.kind == "no_ack"}
    t = set(truth_no_ack)
    return {
        "no_ack_reported": len(a),
        "no_ack_true": len(t),
        "no_ack_precision": round(len(a & t) / len(a), 4) if a else None,
        "no_ack_recall": round(len(a & t) / len(t), 4) if t else None,
        "expedite_drafts": sum(c.kind == "expedite" for c in chases),
    }
