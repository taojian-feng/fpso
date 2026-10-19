"""Onboarding agent, first pass (plan WS7).

Given a connector, it reads the schema and a few sample records of each entity type and
drafts a plain-language description of each type, its likely keys, and the Phase 0 link
types it could take part in. Taojian reviews the draft; the metric is how much of it is
kept unchanged.

    fpso-onboard erp-mm --extracts data/kestrel-A/extracts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fpso.connectors import CONNECTORS, Connector, open_all
from fpso.models import ModelClient, client_from_env
from fpso.resolution.adjudicate import LINK_MEANING, compact_record

SAMPLES = 3

SYSTEM = """You onboard a new data source for an offshore operator's supply chain agents.
You see one source's entity types, their field names, record counts and a few sample
records. Field names are raw (often SAP or Primavera abbreviations); work out what they
mean from names and values. Be concrete and say when you are guessing."""

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string"},
                    "description": {"type": "string"},
                    "likely_keys": {"type": "array", "items": {"type": "string"}},
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "meaning": {"type": "string"},
                            },
                            "required": ["name", "meaning"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["entity_type", "description", "likely_keys", "fields"],
                "additionalProperties": False,
            },
        },
        "link_types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "link_type": {"type": "string"},
                    "entity_type": {"type": "string"},
                    "join_hint": {"type": "string"},
                },
                "required": ["link_type", "entity_type", "join_hint"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "entities", "link_types"],
    "additionalProperties": False,
    "default": {"summary": "(stub backend: no draft)", "entities": [], "link_types": []},
}


def onboard(connector: Connector, client: ModelClient) -> dict:
    body = {
        "source": connector.name,
        "description": connector.description,
        "entity_types": [
            {
                "entity_type": t,
                "count": connector.count(t),
                "fields": connector.schema(t),
                "samples": [compact_record(r) for r in connector.sample(t, SAMPLES)],
            }
            for t in connector.entity_types()
        ],
        "candidate_link_types": LINK_MEANING,
    }
    return client.complete_json(
        agent="onboarding",
        role="onboard",
        system=SYSTEM,
        prompt=json.dumps(body, indent=1, default=str),
        schema=SCHEMA,
        effort="medium",
    )


def to_markdown(source: str, draft: dict) -> str:
    lines = [f"# Onboarding draft: {source}", "", draft.get("summary", ""), ""]
    for e in draft.get("entities", []):
        lines += [f"## {e['entity_type']}", "", e["description"], ""]
        if e["likely_keys"]:
            lines += ["Likely keys: " + ", ".join(f"`{k}`" for k in e["likely_keys"]), ""]
        if e["fields"]:
            lines += ["| Field | Meaning |", "| --- | --- |"]
            lines += [f"| `{f['name']}` | {f['meaning']} |" for f in e["fields"]]
            lines.append("")
    if draft.get("link_types"):
        lines += ["## Proposed link types", "", "| Link type | Entity | How to join |"]
        lines.append("| --- | --- | --- |")
        lines += [
            f"| {lt['link_type']} | {lt['entity_type']} | {lt['join_hint']} |"
            for lt in draft["link_types"]
        ]
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Draft an onboarding description of a source.")
    ap.add_argument("connector", choices=sorted(CONNECTORS))
    ap.add_argument("--extracts", required=True, help="extracts directory of a run")
    ap.add_argument("--out", help="write the Markdown draft here (default: print it)")
    args = ap.parse_args(argv)
    connector = open_all(args.extracts)[args.connector]
    md = to_markdown(args.connector, onboard(connector, client_from_env()))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
