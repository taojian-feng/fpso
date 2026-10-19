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
MAX_WRONG = 5

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
A capacity rule alone rarely separates look-alikes that share a part and a quantity;
pair it with a timing window when the confirmed links' signal statistics are tight.
Rules already decided are listed with the example links the reviewer marked wrong and
their signals. Do not propose a rejected rule again in other words: compare the wrong
examples' signals with the confirmed links' statistics and change what let them through,
most often by adding or narrowing a timing window. Say in notes why you proposed what
you did, or why nothing.
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
        },
        "notes": {"type": "string"},
    },
    "required": ["patterns", "notes"],
    "additionalProperties": False,
    "default": {"patterns": [], "notes": ""},
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


def decided_for_prompt(
    existing: list[Pattern], link_type: str, cands: list[Candidate] | None = None
) -> list[dict]:
    """Earlier rules for the link type, with the reviewer's wrong examples and signals."""
    signals = {c.pair: c.signals for c in cands or [] if c.link_type == link_type}
    return [
        {
            "statement": p.statement,
            "from_capacity": p.from_capacity,
            "gap_signal": p.gap_signal,
            "gap_window": [p.gap_low, p.gap_high] if p.gap_signal else None,
            "status": p.status,
            "examples_shown": len(p.examples),
            "examples_marked_wrong": [
                {"link": e[1:], "signals": signals.get(tuple(e), {})}
                for e in p.wrong_examples[:MAX_WRONG]
            ],
        }
        for p in existing
        if p.link_type == link_type
    ]


def propose(
    client: ModelClient,
    evidence: dict,
    existing: list[Pattern],
    round_no: int,
    cands: list[Candidate] | None = None,
    filtered: list[dict] | None = None,
    notes: dict[str, str] | None = None,
) -> list[Pattern]:
    """Validated new rules. Proposals that fail validation go to ``filtered``, with why,
    and the model's own account of its answer goes to ``notes`` under the link type."""
    lt = evidence["link_type"]
    body = {**evidence, "already_decided": decided_for_prompt(existing, lt, cands)}
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
    if notes is not None:
        notes[lt] = str(data.get("notes", ""))
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
        checks = [
            (p.from_capacity in CAPACITIES, "unknown capacity"),
            (not p.gap_signal or p.gap_signal in known_signals, "unknown timing signal"),
            (p.gap_low <= p.gap_high, "empty timing window"),
            (
                p.from_capacity != "quantity" or {p.from_qty_field, p.to_qty_field} <= known_fields,
                "unknown quantity field",
            ),
            (p.from_capacity != "any" or bool(p.gap_signal), "rule constrains nothing"),
            (p.signature not in seen, "same as an earlier rule"),
        ]
        failed = [why for ok, why in checks if not ok]
        if failed:
            if filtered is not None:
                filtered.append({**p.as_dict(), "filtered_because": failed})
            continue
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
