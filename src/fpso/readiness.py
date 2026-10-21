"""Readiness agent, observe only (plan WS8, autonomy level 0).

For every work package (work order) starting 4 to 26 weeks after the report date, it
walks the chain component -> PO line -> manifest lines and supplier emails, and lists
the gaps that would leave a component missing on the start date:

- no_supply: no PO line found and not enough stock;
- late_po: the PO line's delivery date is after the need date;
- slipped: a supplier email received by the report date moves dispatch past the need date;
- not_shipped: the sailings booked before the need date do not carry the full quantity.

When a gap would rest on a link the agent could not settle (a component whose candidate
PO lines were all left unresolved, or a shipment it could not confirm), it reports the
component as unverified instead: a question for the planner, not a gap.

Each component gets at most one gap, the first in that order. The work package's score
is the share of its components with no gap. The report lists every gap with its source
records. No proposals, no actions.

Links come from candidate generation: a decisive rule match is accepted as is, and the
model adjudicates the rest (reusing answers from the latest round where the prompt is
unchanged). Supplier emails are read by the small extraction model.

    fpso-readiness --run data/kestrel-A --max-cost 5

The truth side (``truth_facts``) builds the same facts from the clean world, so the
harness measures how well the agent's links and email reading carry through.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from fpso.connectors import Connector, open_all
from fpso.eval.reviewers import default_reviewers
from fpso.governance import ActionLog
from fpso.models import CachedBackend, ModelClient, client_from_env
from fpso.resolution.adjudicate import Adjudication, Adjudicator, group_by_target
from fpso.resolution.candidates import EMAIL_ITEM_RE, Candidate, CandidateGenerator
from fpso.resolution.normalize import leading_int, parse_date
from fpso.resolution.patterns import Pattern, apply_patterns
from fpso.synthetic.truth import World, build_world

GAP_TYPES = ("no_supply", "late_po", "slipped", "not_shipped")
HORIZON_WEEKS = (4, 26)
RULE_ACCEPT = 0.9  # a rule match at or above this, clear of the runner-up, needs no model
RULE_MARGIN = 0.1
BATCH = 40  # groups per model batch, in urgency order
CHUNK = 25  # components settled end to end at a time, soonest need first
MAX_SHOWN = 6  # candidates per group shown to the model
MODEL_ACCEPT = 0.9  # the model's own confidence needed to accept its match
# A match the model leans towards but won't confirm: the part may be on order after all.
NEAR_MATCH = 0.5
LINK_TYPES_USED = ("component_po_line", "po_line_manifest_line", "po_email")

EXTRACT_SYSTEM = """You read one supplier email about a purchase order for an offshore
operator. Say whether it reports a delay to dispatch or delivery. If it does, give the new
expected dispatch or delivery date as YYYY-MM-DD, and list the supplier part numbers the
delay applies to exactly as written (empty if it applies to the whole order). Do not guess
a date that is not in the email."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_delay": {"type": "boolean"},
        "new_date": {"type": "string"},
        "parts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["is_delay", "new_date", "parts"],
    "additionalProperties": False,
    "default": {"is_delay": False, "new_date": "", "parts": []},
}


@dataclass
class Facts:
    """What is known about one component's supply chain on the report date."""

    component: str
    work_order: str
    material: str
    qty: int
    need: date
    stock: int
    po_line: str | None = None
    delivery: date | None = None
    slips: list[tuple[date, str]] = field(default_factory=list)  # new date, email
    shipments: list[tuple[date, int, str]] = field(default_factory=list)  # depart, qty, line
    unsettled: set[str] = field(default_factory=set)  # 'supply', 'shipment', 'email' left open

    def sources(self) -> dict:
        return {
            "component": self.component,
            "work_order": self.work_order,
            "po_line": self.po_line,
            "manifest_lines": [s[2] for s in self.shipments],
            "emails": [s[1] for s in self.slips],
        }


def assess(f: Facts) -> str | None:
    """The component's gap, 'unverified', or None when it should be on board in time."""
    if f.po_line is None:
        if "near_match" in f.unsettled:
            return "unverified"  # stock is shared; a half-matched PO line may be its supply
        if f.stock >= f.qty:
            return None
        return "unverified" if "supply" in f.unsettled else "no_supply"
    if f.delivery is not None and f.delivery > f.need:
        return "late_po"
    if "email" in f.unsettled:
        return "unverified"  # an unread supplier email may report a slip
    if any(d > f.need for d, _ in f.slips):
        return "slipped"
    if sum(q for d, q, _ in f.shipments if d < f.need) < f.qty:
        return "unverified" if "shipment" in f.unsettled else "not_shipped"
    return None


def report_dates(turnaround_start: date) -> list[date]:
    """Two weekly reports while the turnaround is 6 and 5 weeks out."""
    return [turnaround_start - timedelta(weeks=6), turnaround_start - timedelta(weeks=5)]


def in_horizon(start: date, as_of: date) -> bool:
    lo, hi = HORIZON_WEEKS
    return as_of + timedelta(weeks=lo) <= start <= as_of + timedelta(weeks=hi)


# --- Truth side: the same facts from the clean world -----------------------------------


def load_world(run_dir: Path) -> World:
    """Rebuild the run's clean world from its seed (and scale, found from the counts when
    run.json predates recording it)."""
    run = json.loads((run_dir / "truth" / "run.json").read_text())
    scales = [run["scale"]] if "scale" in run else [1.0, 0.5, 0.3, 0.25, 0.2, 0.1, 0.08, 0.05]
    for scale in scales:
        world = build_world(seed=run["seed"], scale=scale)
        if (
            len(world.work_orders) == run["counts"]["work_orders"]
            and len(world.po_lines) == run["counts"]["po_lines"]
        ):
            return world
    raise SystemExit(f"cannot rebuild the world for {run_dir}: unknown scale")


def truth_facts(world: World, as_of: date) -> list[Facts]:
    wos = {w.wo_no: w for w in world.work_orders if in_horizon(w.planned_start, as_of)}
    pl_by_comp = {pl.component_key: pl for pl in world.po_lines}
    depart = {s.sailing_id: s.depart for s in world.sailings}
    out = []
    for c in world.components:
        if c.wo_no not in wos:
            continue
        pl = pl_by_comp.get(c.key)
        f = Facts(c.key, c.wo_no, c.mat_no, c.qty, c.req_date, world.stock.get(c.mat_no, 0))
        if pl:
            f.po_line, f.delivery = pl.key, pl.delivery_date
            f.slips = [
                (e.new_date, e.message_id)
                for e in world.emails
                if e.kind == "slip" and pl.key in e.po_line_keys and e.sent <= as_of
            ]
            f.shipments = [
                (depart[ml.sailing_id], ml.qty, ml.line_id)
                for ml in world.manifest_lines
                if pl.key in ml.po_line_keys
            ]
        out.append(f)
    return out


# --- Agent side ---------------------------------------------------------------------------


class ReadinessAgent:
    def __init__(
        self,
        connectors: dict[str, Connector],
        client: ModelClient | None,
        max_cost_usd: float | None = 5.0,
        workers: int = 4,
        patterns: list[Pattern] | None = None,
        prior_answers: list[Adjudication] | None = None,
        prior_fingerprints: dict[str, str] | None = None,
    ):
        self.c = connectors
        self.client = client
        self.max_cost_usd = max_cost_usd
        self.patterns = [p for p in patterns or [] if p.status == "confirmed"]
        self.gen = CandidateGenerator(connectors)
        self.cands = self.gen.generate()
        self.adjudicator = (
            Adjudicator(connectors, client, max_cost_usd=max_cost_usd, workers=workers)
            if client
            else None
        )
        self.prior = group_by_target_adjs(prior_answers or [])
        self.prior_fp = prior_fingerprints
        self.links: dict[tuple[str, str], list[tuple[str, float, str]]] = {}
        self.stats = defaultdict(int)
        self.near_matches: set[tuple[str, str]] = set()  # (link type, source), see NEAR_MATCH
        self.settled_through: date | None = None
        self.email_reads: dict[str, dict] = {}
        pm, mm = connectors["erp-pm"], connectors["erp-mm"]
        self.work_orders = {r.key: r for r in pm.iter("work_order")}
        self.components = list(pm.iter("component"))
        self.po_lines = {r.key: r for r in mm.iter("po_line")}
        self.stock: dict[str, int] = defaultdict(int)
        for r in mm.iter("stock"):
            self.stock[r.fields["MATNR"]] += leading_int(r.fields["LABST"]) or 0
        self.sailings = {
            r.key: parse_date(r.fields["depart_date"]) for r in connectors["vessel"].iter("sailing")
        }
        self.part_to_material = {
            r.fields["MFRPN"]: r.key for r in mm.iter("material") if r.fields.get("MFRPN")
        }
        self.lines_by_po_material: dict[tuple[str, str], list] = defaultdict(list)
        for r in self.po_lines.values():
            self.lines_by_po_material[(r.fields["EBELN"], r.fields["MATNR"])].append(r)

    # Links ----------------------------------------------------------------------------

    def resolve(
        self, targets: set[tuple[str, str]], urgency: dict[tuple[str, str], date] | None = None
    ) -> None:
        """Accept links for the given (link_type, target) groups not resolved yet.

        Groups go to the model most urgent first (earliest need date), in batches, so a
        run that hits the cost cap has settled the work packages starting soonest."""
        groups = {
            k: g
            for k, g in group_by_target(
                [c for c in self.cands if (c.link_type, c.to_key) in targets]
            ).items()
            if k not in self.links
        }
        to_model: list[Candidate] = []
        adjs: list[Adjudication] = []
        for (lt, to_key), g in sorted(groups.items()):
            ranked = sorted(g, key=lambda c: -c.score)
            margin = ranked[0].score - (ranked[1].score if len(ranked) > 1 else 0.0)
            if ranked[0].score >= RULE_ACCEPT and margin >= RULE_MARGIN:
                self.links[(lt, to_key)] = [(ranked[0].from_key, ranked[0].score, "rule")]
                self.stats["rule"] += 1
            elif self._reusable(lt, to_key, g):
                adjs += self.prior[(lt, to_key)]
                self.stats["model_reused"] += 1
            elif self.adjudicator and not self.over_budget():
                to_model += ranked[:MAX_SHOWN]  # the rest are far behind the leaders
            elif self.adjudicator:
                continue  # past the cost cap: left open, not settled as unresolvable
            else:
                self.links[(lt, to_key)] = []
                self.stats["unresolved"] += 1
        if to_model:
            urgency = urgency or {}
            pending = sorted(
                group_by_target(to_model).items(),
                key=lambda kv: (urgency.get(kv[0], date.max), kv[0]),
            )
            before = self.adjudicator.skipped_groups
            for i in range(0, len(pending), BATCH):
                batch = [c for _, g in pending[i : i + BATCH] for c in g]
                adjs += self.adjudicator.adjudicate(batch)
            self.stats["model_asked"] += len(pending)
            self.stats["skipped_by_cost_cap"] += self.adjudicator.skipped_groups - before
        adjs = apply_patterns(adjs, self.patterns, self.cands, self.c)
        for (lt, to_key), g in group_by_target_adjs(adjs).items():
            best = max(g, key=lambda a: a.confidence)
            ok = best.verdict == "match" and best.confidence >= MODEL_ACCEPT
            self.links[(lt, to_key)] = [(best.from_key, best.confidence, "model")] if ok else []
            if not ok and best.verdict == "match" and best.confidence >= NEAR_MATCH:
                self.near_matches.add((lt, best.from_key))
        for k in groups:
            if k not in self.links and (self.adjudicator is None or not self.over_budget()):
                self.links[k] = []

    def _reusable(self, lt: str, to_key: str, g: list[Candidate]) -> bool:
        old = self.prior.get((lt, to_key))
        if not old:
            return False
        if self.prior_fp is not None and self.adjudicator is not None:
            fp = hashlib.sha256(self.adjudicator.prompt(lt, to_key, g).encode()).hexdigest()[:16]
            return self.prior_fp.get(f"{lt}|{to_key}") == fp
        return sorted((c.pair, c.score) for c in g) == sorted((a.pair, a.rule_score) for a in old)

    def _leading_targets(self, lt: str, sources: set[str]) -> set[tuple[str, str]]:
        """Groups where one of these sources leads or ties for the lead. Only those can
        link it; settling the others would cost model calls and change nothing here."""
        out = set()
        for (t, to_key), g in group_by_target([c for c in self.cands if c.link_type == lt]).items():
            top = max(c.score for c in g)
            if any(c.from_key in sources and c.score >= top - 0.05 for c in g):
                out.add((t, to_key))
        return out

    def _urgency(self, lt: str, need: dict[str, date]) -> dict[tuple[str, str], date]:
        """Each group's earliest need date among the sources it could link."""
        out: dict[tuple[str, str], date] = {}
        for c in self.cands:
            if c.link_type == lt and c.from_key in need:
                k = (lt, c.to_key)
                out[k] = min(need[c.from_key], out.get(k, date.max))
        return out

    def _open_sources(self, lt: str, sources: set[str]) -> set[str]:
        """Sources that lead (or tie for the lead) in a group with no accepted link: left
        unresolved, or never reached because the cost cap stopped the walk."""
        groups = group_by_target(
            [c for c in self.cands if c.link_type == lt and not self.links.get((lt, c.to_key))]
        )
        out = set()
        for g in groups.values():
            top = max(c.score for c in g)
            out |= {c.from_key for c in g if c.from_key in sources and c.score >= top - 0.05}
        return out

    def linked_from(self, lt: str, from_key: str) -> list[str]:
        """Targets accepted as linked to this source record."""
        return [
            to_key
            for (t, to_key), accepted in self.links.items()
            if t == lt
            for f, _, _ in accepted
            if f == from_key
        ]

    # Emails ---------------------------------------------------------------------------

    def read_email(self, key: str) -> dict | None:
        if key in self.email_reads:
            return self.email_reads[key]
        msg = self.c["supplier-mail"].get("message", key)
        if self.client is None:
            out = dict(EXTRACT_SCHEMA["default"])  # no model: emails are not read
        elif self.over_budget():
            return None  # unread: the slip, if any, stays unknown
        else:
            out = self.client.complete_json(
                agent="readiness",
                role="extract",
                system=EXTRACT_SYSTEM,
                prompt=f"Subject: {msg.fields['subject']}\n\n{msg.fields['body']}",
                schema=EXTRACT_SCHEMA,
                effort="low",
            )
            self.stats["emails_read"] += 1
        self.email_reads[key] = out
        return out

    def _slip_fits_line(self, msg, pl: dict) -> bool | None:
        """Whether a delay email about this PO's material concerns this line.

        A PO can order one material on several lines. The email's quoted quantity then
        tells them apart; None when it cannot."""
        same = self.lines_by_po_material[(pl["EBELN"], pl["MATNR"])]
        if len(same) < 2:
            return True
        quoted = {
            int(q)
            for part, q in EMAIL_ITEM_RE.findall(msg.fields["body"])
            if self.part_to_material.get(part) == pl["MATNR"]
        }
        fits = [r for r in same if (leading_int(r.fields["MENGE"]) or 0) in quoted]
        if len(fits) == 1:
            return fits[0].fields is pl
        return None

    # The walk -------------------------------------------------------------------------

    def set_cap(self, usd: float) -> None:
        self.max_cost_usd = usd
        if self.adjudicator:
            self.adjudicator.max_cost_usd = usd

    def over_budget(self) -> bool:
        return (
            self.client is not None
            and self.max_cost_usd is not None
            and self.client.log.total_cost_usd >= self.max_cost_usd
        )

    def _settle(self, comps: list, as_of: date) -> None:
        """Settle the whole chain for these components: their PO lines, then those POs'
        supplier emails and reading them, then the lines' manifest lines. Emails come
        before manifests: past the cost cap an unread email leaves the component
        unverified, while an open manifest group often does not decide it."""
        keys = {c.key for c in comps}
        need = {c.key: parse_date(c.fields["BDTER"]) or date.max for c in comps}
        self.resolve(
            self._leading_targets("component_po_line", keys),
            self._urgency("component_po_line", need),
        )
        po_line_of = self._po_lines_of(keys)
        lines = set(po_line_of.values())
        pos = {self.po_lines[k].fields["EBELN"] for k in lines}
        self.resolve(self._leading_targets("po_email", pos))
        for po in sorted(pos):
            for msg_key in self.linked_from("po_email", po):
                msg = self.c["supplier-mail"].get("message", msg_key)
                if parse_date(msg.fields["date"]) <= as_of:
                    self.read_email(msg_key)
        self.resolve(self._leading_targets("po_line_manifest_line", lines))

    def _po_lines_of(self, comp_keys: set[str]) -> dict[str, str]:
        out = {}
        for (lt, to_key), accepted in self.links.items():
            if lt == "component_po_line":
                for from_key, _, _ in accepted:
                    if from_key in comp_keys:
                        out[from_key] = to_key
        return out

    def facts(self, as_of: date) -> list[Facts]:
        scope = {
            k: w
            for k, w in self.work_orders.items()
            if (d := parse_date(w.fields["GSTRP"])) and in_horizon(d, as_of)
        }
        comps = sorted(
            (c for c in self.components if c.fields["AUFNR"] in scope),
            key=lambda c: (parse_date(c.fields["BDTER"]) or date.max, c.key),
        )
        # Work through the chain in order of need, a few components at a time, so a run
        # that hits the cost cap has settled the soonest work packages end to end.
        self.settled_through = None
        for i in range(0, len(comps), CHUNK):
            # Past the cost cap this still settles what rule matches decide for free.
            chunk = comps[i : i + CHUNK]
            self._settle(chunk, as_of)
            if not self.over_budget():
                self.settled_through = parse_date(chunk[-1].fields["BDTER"])
        comp_keys = {c.key for c in comps}
        po_line_of = self._po_lines_of(comp_keys)
        open_supply = self._open_sources("component_po_line", comp_keys)
        open_shipment = self._open_sources("po_line_manifest_line", set(po_line_of.values()))
        manifest = self.c["vessel"]
        out = []
        for c in comps:
            f = c.fields
            pl_key = po_line_of.get(c.key)
            facts = Facts(
                c.key,
                f["AUFNR"],
                f["MATNR"],
                leading_int(f["BDMNG"]) or 0,
                parse_date(f["BDTER"]),
                self.stock.get(f["MATNR"], 0),
            )
            if c.key in open_supply and not pl_key:
                facts.unsettled.add("supply")  # with a PO line accepted, supply is known
                if ("component_po_line", c.key) in self.near_matches:
                    facts.unsettled.add("near_match")
            if pl_key:
                if pl_key in open_shipment:
                    facts.unsettled.add("shipment")
                pl = self.po_lines[pl_key].fields
                facts.po_line, facts.delivery = pl_key, parse_date(pl["EINDT"])
                for ml_key in self.linked_from("po_line_manifest_line", pl_key):
                    ml = manifest.get("manifest_line", ml_key).fields
                    depart = self.sailings.get(ml["sailing_id"])
                    if depart:
                        facts.shipments.append((depart, leading_int(ml["qty"]) or 0, ml_key))
                for msg_key in self.linked_from("po_email", pl["EBELN"]):
                    msg = self.c["supplier-mail"].get("message", msg_key)
                    if parse_date(msg.fields["date"]) > as_of:
                        continue  # not received yet on the report date
                    read = self.read_email(msg_key)
                    if read is None:
                        facts.unsettled.add("email")
                        continue
                    new = _iso(read.get("new_date", ""))
                    if not read.get("is_delay") or new is None:
                        continue
                    mats = {self.part_to_material.get(p) for p in read.get("parts", [])}
                    if read.get("parts") and pl["MATNR"] not in mats:
                        continue
                    match = self._slip_fits_line(msg, pl)
                    if match is None:
                        facts.unsettled.add("email")  # which of the look-alike lines is late?
                    elif match:
                        facts.slips.append((new, msg_key))
            out.append(facts)
        return out


def group_by_target_adjs(adjs: list[Adjudication]) -> dict[tuple[str, str], list[Adjudication]]:
    out: dict[tuple[str, str], list[Adjudication]] = defaultdict(list)
    for a in adjs:
        out[(a.link_type, a.to_key)].append(a)
    return out


def _iso(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


# --- Scoring and reports ----------------------------------------------------------------


def gap_list(facts: list[Facts]) -> dict[str, dict[str, str]]:
    """Work order -> {component: gap}, for every work order with materials to supply."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for f in facts:
        out.setdefault(f.work_order, {})
        gap = assess(f)
        if gap and gap != "unverified":
            out[f.work_order][f.component] = gap
    return out


def score_report(agent: list[Facts], truth: list[Facts], planner, as_of: date) -> dict:
    a_gaps, t_gaps = gap_list(agent), gap_list(truth)
    wps = sorted(set(a_gaps) | set(t_gaps))
    agree = [wp for wp in wps if a_gaps.get(wp, {}) == t_gaps.get(wp, {})]
    a_set = {(wp, c, g) for wp, gs in a_gaps.items() for c, g in gs.items()}
    t_set = {(wp, c, g) for wp, gs in t_gaps.items() for c, g in gs.items()}
    # The simulated planner marks each reported gap right or wrong, with some noise.
    marks = [planner.confirm_gap(f"{as_of}|{c}|{g}", (wp, c, g) in t_set) for wp, c, g in a_set]
    by_type = {}
    for g in GAP_TYPES:
        a = {x for x in a_set if x[2] == g}
        t = {x for x in t_set if x[2] == g}
        by_type[g] = {
            "reported": len(a),
            "true": len(t),
            "precision": round(len(a & t) / len(a), 4) if a else None,
            "recall": round(len(a & t) / len(t), 4) if t else None,
        }
    answered = [m for m in marks if m is not None]
    return {
        "components": len(agent),
        "unverified": sum(assess(f) == "unverified" for f in agent),
        "work_packages": len(wps),
        "agreement": round(len(agree) / len(wps), 4) if wps else None,
        "gaps_reported": len(a_set),
        "gaps_true": len(t_set),
        "gap_precision": round(len(a_set & t_set) / len(a_set), 4) if a_set else None,
        "gap_recall": round(len(a_set & t_set) / len(t_set), 4) if t_set else None,
        "planner_marked_right": round(sum(answered) / len(answered), 4) if answered else None,
        "by_type": by_type,
    }


def to_markdown(run: dict, as_of: date, facts: list[Facts], scores: dict, meta: dict) -> str:
    def pct(v):
        return "n/a" if v is None else f"{v:.0%}"

    by_wp: dict[str, list[Facts]] = defaultdict(list)
    for f in facts:
        by_wp[f.work_order].append(f)
    lines = [
        f"# Readiness report, {as_of.isoformat()}: {run['asset']}, profile {run['profile']}",
        "",
        f"Work packages starting {HORIZON_WEEKS[0]} to {HORIZON_WEEKS[1]} weeks out: "
        f"{len(by_wp)}, with {len(facts)} components. Observe only: no proposals, no actions.",
        "",
        f"Agreement with the true gap list: {pct(scores['agreement'])} of work packages "
        f"(target 80%). Gap precision {pct(scores['gap_precision'])}, recall "
        f"{pct(scores['gap_recall'])}; the simulated planner marked "
        f"{pct(scores['planner_marked_right'])} of reported gaps right. "
        f"{scores['unverified']} components are unverified: a link they depend on is open.",
        "",
        "| Gap | Reported | True | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for g, s in scores["by_type"].items():
        lines.append(
            f"| {g} | {s['reported']} | {s['true']} | {pct(s['precision'])} | {pct(s['recall'])} |"
        )
    lines += [
        "",
        f"Links so far this run: {meta['links']}. Cost for this date ${meta['cost_usd']:.2f} "
        f"(run total ${meta['run_cost_usd']:.2f})."
        + (
            f" The model settled links for components needed up to "
            f"{meta['model_settled_through']}; later ones rely on rule links where the cost "
            "cap stopped it."
            if meta.get("model_settled_through")
            else ""
        ),
        "",
        "## Work packages with gaps",
        "",
        "| Work order | Start | Score | Component | Gap | PO line | Manifest lines | Emails |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for wp, fs in sorted(by_wp.items(), key=lambda kv: (min(f.need for f in kv[1]), kv[0])):
        gaps = [(f, assess(f)) for f in fs]
        bad = [(f, g) for f, g in gaps if g and g != "unverified"]
        if not bad:
            continue
        score = 1 - len(bad) / len(fs)
        for f, g in bad:
            src = f.sources()
            lines.append(
                f"| {wp} | {f.need.isoformat()} | {score:.0%} | {f.component} | {g} "
                f"| {src['po_line'] or '-'} | {', '.join(src['manifest_lines']) or '-'} "
                f"| {', '.join(src['emails']) or '-'} |"
            )
    return "\n".join(lines) + "\n"


def run_readiness(
    run_dir: str | Path,
    as_of: list[date] | None = None,
    max_cost_usd: float | None = 5.0,
    workers: int = 4,
    client: ModelClient | None = None,
    use_model: bool = True,
    final: bool = False,
) -> list[dict]:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "truth" / "run.json").read_text())
    if run["profile"] == "B" and not final:
        raise SystemExit("Profile B is held back for the final run; pass --final to use it.")
    out_dir = run_dir / "reports" / "readiness"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = ActionLog(out_dir / "actions.jsonl")
    if use_model:
        if client is None:
            client = client_from_env(log=log)
        else:
            client.log = log
        # Answers are kept, so a re-run with a bigger cap pays only for new questions.
        client.backend = CachedBackend(client.backend, out_dir / "answers.jsonl")
    else:
        client = None
    t0 = time.time()
    rounds = run_dir / "reports" / "rounds"
    latest = max(
        (
            int(p.name.split(".")[0].removeprefix("round-"))
            for p in rounds.glob("round-*.adjudications.jsonl")
        ),
        default=None,
    )
    prior, prior_fp = [], None
    if latest is not None:
        prior = [
            Adjudication(**json.loads(line))
            for line in (rounds / f"round-{latest}.adjudications.jsonl").read_text().splitlines()
        ]
        fp = rounds / f"round-{latest}.prompts.json"
        prior_fp = json.loads(fp.read_text()) if fp.exists() else None
    patterns_path = run_dir / "reports" / "patterns.json"
    patterns = (
        [Pattern(**p) for p in json.loads(patterns_path.read_text())]
        if patterns_path.exists()
        else []
    )
    agent = ReadinessAgent(
        open_all(run_dir / "extracts"),
        client,
        max_cost_usd,
        workers,
        patterns=patterns,
        prior_answers=prior,
        prior_fingerprints=prior_fp,
    )
    world = None
    reports = []
    spent = 0.0
    dates = as_of or []
    for i, d in enumerate(dates):
        if max_cost_usd is not None:
            # Each report date may spend up to its share of the cap, plus what earlier
            # dates left unspent, so a later report can still read its new emails.
            agent.set_cap(max_cost_usd * (i + 1) / len(dates))
        facts = agent.facts(d)
        total = client.log.total_cost_usd if client else 0.0
        # Only the scoring below reads the truth.
        world = world or load_world(run_dir)
        truth = truth_facts(world, d)
        planner = next(r for r in default_reviewers(set(), log=log) if r.role == "planner")
        scores = score_report(facts, truth, planner, d)
        meta = {
            "as_of": d.isoformat(),
            "links": dict(agent.stats),  # cumulative over this run's report dates
            "emails_read": agent.stats.get("emails_read", 0),
            "cost_usd": round(total - spent, 4),  # this report date only
            "run_cost_usd": round(total, 4),
            "model_settled_through": agent.settled_through.isoformat()
            if agent.settled_through and client
            else None,
            "seconds": round(time.time() - t0, 1),
            "answers_from_round": latest,
            "answers_kept": client.backend.hits if client else 0,  # reused, not paid for
        }
        report = {
            "run": run,
            "meta": meta,
            "scores": scores,
            "gaps": [
                {**f.sources(), "need": f.need.isoformat(), "gap": assess(f)}
                for f in facts
                if assess(f) not in (None, "unverified")
            ],
            "unverified": [
                {**f.sources(), "need": f.need.isoformat(), "open": sorted(f.unsettled)}
                for f in facts
                if assess(f) == "unverified"
            ],
            # On-track components too, so a missed gap can be traced to the link behind it.
            "on_track": [
                {
                    **f.sources(),
                    "need": f.need.isoformat(),
                    "shipped": [str(s[0]) for s in f.shipments],
                }
                for f in facts
                if assess(f) is None
            ],
        }
        (out_dir / f"{d.isoformat()}.json").write_text(json.dumps(report, indent=2, default=str))
        (out_dir / f"{d.isoformat()}.md").write_text(to_markdown(run, d, facts, scores, meta))
        reports.append(report)
        spent = total
    return reports


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Weekly readiness reports (observe only).")
    ap.add_argument("--run", required=True, help="a run directory written by fpso-synth")
    ap.add_argument(
        "--as-of",
        action="append",
        type=date.fromisoformat,
        help="report date, repeatable (default: the turnaround 6 and 5 weeks out)",
    )
    ap.add_argument("--max-cost", type=float, default=5.0, help="stop new calls above this USD")
    ap.add_argument("--workers", type=int, default=4, help="parallel model calls")
    ap.add_argument("--no-model", action="store_true", help="rule links only, no email reading")
    ap.add_argument("--final", action="store_true", help="allow the held-back profile B")
    args = ap.parse_args(argv)
    run = json.loads((Path(args.run) / "truth" / "run.json").read_text())
    dates = args.as_of or report_dates(date.fromisoformat(run["turnaround"][0]))
    for r in run_readiness(
        args.run, dates, args.max_cost, args.workers, use_model=not args.no_model, final=args.final
    ):
        s = r["scores"]
        print(
            f"{r['meta']['as_of']}: {s['work_packages']} work packages, agreement "
            f"{s['agreement']}, gap precision {s['gap_precision']}, recall {s['gap_recall']}, "
            f"cost ${r['meta']['cost_usd']}"
        )


if __name__ == "__main__":
    main()
