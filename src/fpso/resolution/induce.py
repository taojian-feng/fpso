"""Pattern induction (plan WS5 step 4): the large model proposes general rules from the
cases the pipeline could not settle, and a reviewer confirms or rejects each rule once.

The model sees, per link type, the dev targets still left unsure, with their candidates,
and summary statistics of the timing signals and source-record reuse among links the
reviewers confirmed. It answers in the executable pattern vocabulary of
``fpso.resolution.patterns``, so a confirmed rule can run with no model at all.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from fpso.connectors import Connector
from fpso.models import ModelClient
from fpso.resolution.adjudicate import LINK_MEANING, Adjudication, compact_record
from fpso.resolution.candidates import LINK_ENDS, Candidate
from fpso.resolution.patterns import CAPACITIES, Pattern, resolve

MAX_GROUPS = 8
MAX_EXAMPLES = 12

SYSTEM = """You improve an entity-resolution pipeline for an offshore operator's supply
chain. For one link type you see the cases the model left unsure: a target record and
several look-alike candidates. You also see statistics from links reviewers confirmed.

Propose at most two general rules that would break these ties, in this vocabulary:
- from_capacity: "one" if a source record can link to only one target, "quantity" if a
  source record's ordered quantity caps the total quantity of its targets, "any" if there
  is no limit. For "quantity", name the quantity fields (from_qty_field on the source
  record, to_qty_field on the target record).
- gap_signal, gap_low, gap_high: a numeric timing signal from the candidates and the
  window true links fall in. Leave gap_signal empty for none.
Only propose a rule the evidence supports; an empty list is a fine answer. Write the
statement as one plain sentence a planner or materials coordinator could confirm."""

SCHEMA = {
    "type": "object",
    "properties": {
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "from_capacity": {"type": "string", "enum": list(CAPACITIES)},
                    "from_qty_field": {"type": "string"},
                    "to_qty_field": {"type": "string"},
                    "gap_signal": {"type": "string"},
                    "gap_low": {"type": "number"},
                    "gap_high": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "statement",
                    "from_capacity",
                    "from_qty_field",
                    "to_qty_field",
                    "gap_signal",
                    "gap_low",
                    "gap_high",
                    "rationale",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["patterns"],
    "additionalProperties": False,
    "default": {"patterns": []},
}


def _quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    v = sorted(values)

    def q(p):
        return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]

    return {"n": len(v), "min": v[0], "p10": q(0.1), "median": q(0.5), "p90": q(0.9), "max": v[-1]}


def evidence_for(
    link_type: str,
    adjs: list[Adjudication],
    labelled: list[tuple[Adjudication, bool]],
    cands: list[Candidate],
    connectors: dict[str, Connector],
) -> dict:
    """What the inducer sees: unsure dev groups and statistics from confirmed links."""
    (from_src, from_type), (to_src, to_type) = LINK_ENDS[link_type]
    by_pair = {c.pair: c for c in cands if c.link_type == link_type}
    confirmed = [a for a, y in labelled if y and a.link_type == link_type]
    signals: dict[str, list[float]] = defaultdict(list)
    for a in confirmed:
        for k, v in by_pair[a.pair].signals.items():
            if isinstance(v, int | float) and not isinstance(v, bool):
                signals[k].append(v)
    reuse = Counter(a.from_key for a in confirmed)
    groups = defaultdict(list)
    for a in adjs:
        if a.link_type == link_type:
            groups[a.to_key].append(a)
    # Unsure: no single confident match, either none or several look-alikes tied.
    unsure = [
        g
        for g in groups.values()
        if sum(a.verdict == "match" and a.confidence >= 0.9 for a in g) != 1
        and any(a.verdict != "no_match" for a in g)
    ]
    unsure.sort(key=lambda g: g[0].to_key)
    return {
        "link_type": link_type,
        "a_real_link_means": LINK_MEANING[link_type],
        "source_fields": connectors[from_src].schema(from_type),
        "target_fields": connectors[to_src].schema(to_type),
        "confirmed_links": len(confirmed),
        "confirmed_signal_stats": {k: _quantiles(v) for k, v in sorted(signals.items())},
        "source_records_in_more_than_one_confirmed_link": sum(n > 1 for n in reuse.values()),
        "unsure_targets": len(unsure),
        "examples": [
            {
                "target": compact_record(connectors[to_src].get(to_type, g[0].to_key)),
                "candidates": [
                    {
                        "record": compact_record(connectors[from_src].get(from_type, a.from_key)),
                        "signals": by_pair[a.pair].signals,
                        "verdict": a.verdict,
                        "confidence": a.confidence,
                        "evidence": a.evidence,
                    }
                    for a in sorted(g, key=lambda a: -a.confidence)[:6]
                ],
            }
            for g in unsure[:MAX_GROUPS]
        ],
    }


def propose(
    client: ModelClient, evidence: dict, existing: list[Pattern], round_no: int
) -> list[Pattern]:
    lt = evidence["link_type"]
    body = {
        **evidence,
        "already_decided": [
            {"statement": p.statement, "status": p.status} for p in existing if p.link_type == lt
        ],
    }
    if not evidence["unsure_targets"]:
        return []
    data = client.complete_json(
        agent="pattern-inducer",
        role="induce",
        system=SYSTEM,
        prompt=json.dumps(body, indent=1, default=str),
        schema=SCHEMA,
        effort="high",
    )
    known_fields = set(evidence["source_fields"]) | set(evidence["target_fields"])
    known_signals = set(evidence["confirmed_signal_stats"])
    seen = {p.signature for p in existing}
    out = []
    for i, d in enumerate(data.get("patterns", [])[:2]):
        p = Pattern(
            id=f"r{round_no}-{lt}-{i + 1}",
            link_type=lt,
            statement=str(d.get("statement", "")),
            from_capacity=d.get("from_capacity", "any"),
            from_qty_field=str(d.get("from_qty_field", "")),
            to_qty_field=str(d.get("to_qty_field", "")),
            gap_signal=str(d.get("gap_signal", "")),
            gap_low=float(d.get("gap_low", 0)),
            gap_high=float(d.get("gap_high", 0)),
            round=round_no,
        )
        valid = (
            p.from_capacity in CAPACITIES
            and (not p.gap_signal or p.gap_signal in known_signals)
            and (p.gap_low <= p.gap_high)
            and (
                p.from_capacity != "quantity" or {p.from_qty_field, p.to_qty_field} <= known_fields
            )
            and (p.from_capacity != "any" or p.gap_signal)
            and p.signature not in seen
        )
        if valid:
            seen.add(p.signature)
            out.append(p)
    return out


def examples_for(
    pattern: Pattern,
    adjs: list[Adjudication],
    cands: list[Candidate],
    connectors: dict[str, Connector],
    dev_targets: set[tuple[str, str]],
) -> list[tuple[str, str, str]]:
    """Dev links the pattern would newly accept: what a reviewer should look at."""
    decided = resolve(pattern, cands, connectors)
    rows = [
        a
        for a in adjs
        if a.link_type == pattern.link_type and (a.link_type, a.to_key) in dev_targets
    ]
    changed = [a.pair for a in rows if decided.get(a.pair) and a.confidence < 0.9]
    if len(changed) < 3:
        changed += [a.pair for a in rows if decided.get(a.pair) and a.pair not in changed]
    return sorted(changed)[:MAX_EXAMPLES]
