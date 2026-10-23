"""Adjudication (plan WS5 step 2): Claude decides which candidate links are real.

Candidates are grouped by their target record (the right-hand side of the link type, such
as one manifest line). One model call sees the target, every candidate for it, and the
signals that fired, and returns a verdict, a confidence and the evidence for each. Seeing
the alternatives side by side is cheaper than one call per pair and lets the model compare.
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from fpso.connectors import Connector, Record
from fpso.models import ModelClient
from fpso.resolution.candidates import LINK_ENDS, Candidate

VERDICTS = ("match", "no_match", "unsure")
MAX_FIELD_CHARS = 1500

LINK_MEANING = {
    "equipment_activity": "the P6 schedule activity is work on this ERP equipment item",
    "work_order_activity": "the P6 schedule activity executes this ERP maintenance work order",
    "component_po_line": "the purchase order line buys the material this work order "
    "component (reservation) needs",
    "po_line_manifest_line": "the vessel manifest line ships goods from this purchase order line",
    "po_email": "the supplier email is about this purchase order",
}

SYSTEM = """You link records across an offshore operator's systems: ERP maintenance (SAP PM),
ERP materials (SAP MM), the Primavera P6 schedule, vessel manifests and supplier email.
The data is messy: tags drift in case and separators, references go missing, text is
abbreviated, numbers are zero-padded or truncated, dates change format.

You get one target record and several candidate records that might link to it, with the
matching signals a cheap rule pass found. For every candidate decide whether the link is
real. A target normally links to at most one candidate, so compare them against each other.
Judge from the records themselves; the signals are hints and can be wrong.

For each candidate return:
- verdict: match, no_match or unsure
- confidence: your probability, from 0 to 1, that the link is real
- evidence: one short sentence naming the fields that decided it"""

SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": ["id", "verdict", "confidence", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
    "default": {"decisions": []},
}


@dataclass
class Adjudication:
    link_type: str
    from_key: str
    to_key: str
    rule_score: float
    verdict: str
    confidence: float
    evidence: str
    model: str | None = None

    @property
    def pair(self) -> tuple[str, str, str]:
        return (self.link_type, self.from_key, self.to_key)

    def as_dict(self) -> dict:
        return asdict(self)


def compact_record(record: Record | None) -> dict:
    if record is None:
        return {"missing": True}
    fields = {}
    for k, v in record.fields.items():
        if v in (None, "", [], {}):
            continue
        s = v if isinstance(v, str) else json.dumps(v, default=str)
        fields[k] = s if len(s) <= MAX_FIELD_CHARS else s[:MAX_FIELD_CHARS] + " …"
    return {"source": record.source, "type": record.entity_type, "key": record.key, **fields}


def group_by_target(cands: list[Candidate]) -> dict[tuple[str, str], list[Candidate]]:
    groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in cands:
        groups[(c.link_type, c.to_key)].append(c)
    return dict(groups)


class Adjudicator:
    def __init__(
        self,
        connectors: dict[str, Connector],
        client: ModelClient,
        max_cost_usd: float | None = None,
        workers: int = 4,
    ):
        self.c = connectors
        self.client = client
        self.max_cost_usd = max_cost_usd
        self.workers = workers
        self.skipped_groups = 0

    def _record(self, side: tuple[str, str], key: str) -> Record | None:
        source, entity_type = side
        return self.c[source].get(entity_type, key)

    def prompt(self, link_type: str, to_key: str, cands: list[Candidate]) -> str:
        from_side, to_side = LINK_ENDS[link_type]
        body = {
            "link_type": link_type,
            "a_real_link_means": LINK_MEANING[link_type],
            "target": compact_record(self._record(to_side, to_key)),
            "candidates": [
                {
                    "id": i,
                    "record": compact_record(self._record(from_side, c.from_key)),
                    "signals": c.signals,
                    "rule_score": c.score,
                }
                for i, c in enumerate(cands)
            ],
        }
        return json.dumps(body, indent=1, default=str)

    def adjudicate_group(
        self, link_type: str, to_key: str, cands: list[Candidate]
    ) -> list[Adjudication]:
        data = self.client.complete_json(
            agent="resolver",
            role="adjudicate",
            system=SYSTEM,
            prompt=self.prompt(link_type, to_key, cands),
            schema=SCHEMA,
            effort="low",
        )
        return self._parse(link_type, cands, data)

    def _parse(self, link_type: str, cands: list[Candidate], data: dict) -> list[Adjudication]:
        model = self.client.role_models["adjudicate"]
        by_id = {d.get("id"): d for d in data.get("decisions", []) if isinstance(d, dict)}
        out = []
        for i, c in enumerate(cands):
            d = by_id.get(i)
            if d is None or d.get("verdict") not in VERDICTS:
                verdict, conf, evidence = "unsure", 0.5, "no decision returned"
            else:
                verdict = d["verdict"]
                conf = min(1.0, max(0.0, float(d.get("confidence", 0.5))))
                evidence = str(d.get("evidence", ""))
            out.append(
                Adjudication(
                    link_type, c.from_key, c.to_key, c.score, verdict, conf, evidence, model
                )
            )
        return out

    def adjudicate(self, cands: list[Candidate]) -> list[Adjudication]:
        """Adjudicate every candidate, stopping early if the cost cap is reached.

        With a batching backend the groups go out as one batch, so the cap is checked once
        before it is sent and can be passed by up to that batch's cost."""
        groups = sorted(group_by_target(cands).items())
        if self.client.batches:
            if self.max_cost_usd is not None and (
                self.client.log.total_cost_usd >= self.max_cost_usd
            ):
                self.skipped_groups += len(groups)
                return []
            answers = self.client.complete_many(
                agent="resolver",
                role="adjudicate",
                system=SYSTEM,
                prompts=[self.prompt(lt, to_key, g) for (lt, to_key), g in groups],
                schema=SCHEMA,
                effort="low",
            )
            return [
                a
                for ((lt, _), g), data in zip(groups, answers, strict=True)
                for a in self._parse(lt, g, data)
            ]

        def run(item):
            (link_type, to_key), group = item
            if self.max_cost_usd is not None and (
                self.client.log.total_cost_usd >= self.max_cost_usd
            ):
                self.skipped_groups += 1
                return []
            return self.adjudicate_group(link_type, to_key, group)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return [a for batch in pool.map(run, groups) for a in batch]
