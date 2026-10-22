"""Shared library (Phase 4): what onboarding one asset leaves for the next.

- connectors: the read-only connectors every asset is read through;
- dialects: confirmed field mappings, one per source-system convention (Kestrel's SAP-style
  names are the canonical ones, so its dialect is the identity);
- rule cards: the agents' business rules and their parameters, the same on every asset;
- patterns: confirmed link patterns, kept per asset that confirmed them;
- assets: each onboarded asset with its dialect, run directory and onboarding effort.

A new asset first looks for a dialect whose tables and columns its raw extracts already
have; only what that dialect does not cover goes to the rule mapper and a reviewer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fpso.connectors import CONNECTORS, Connector
from fpso.fleet.canonical import Mapping, identity_mapping

MIN_COVERAGE = 0.8  # share of a dialect's columns a raw extract must have to reuse it


@dataclass
class AssetEntry:
    code: str
    name: str
    dialect: str
    run: str
    mapping: dict = field(default_factory=dict)  # the asset's confirmed mapping
    onboarding: dict = field(default_factory=dict)  # effort and how much the mapper got right


def rule_cards() -> dict[str, dict]:
    """The agents' rules as cards: a statement and the parameters the code runs with."""
    from fpso import liaison, procurement, readiness, schedule

    return {
        "procurement.order_by": {
            "statement": "Latest order date is the need date minus the days to reach a "
            "sailing minus the material's lead time; expedite a line due too late.",
            "MIN_SHIP_DAYS": procurement.MIN_SHIP_DAYS,
            "WARN_DAYS": procurement.WARN_DAYS,
            "HORIZON_WEEKS": list(procurement.HORIZON_WEEKS),
        },
        "readiness.gaps": {
            "statement": "Each component gets at most one gap, in this order.",
            "GAP_TYPES": list(readiness.GAP_TYPES),
            "HORIZON_WEEKS": list(readiness.HORIZON_WEEKS),
        },
        "resolution.rule_accept": {
            "statement": "A rule match this strong, this far clear of the runner-up, needs "
            "no model.",
            "RULE_ACCEPT": readiness.RULE_ACCEPT,
            "RULE_MARGIN": readiness.RULE_MARGIN,
        },
        "liaison.no_ack": {
            "statement": "Chase an open PO with no supplier reply this many days after it "
            "was placed.",
            "ACK_DAYS": liaison.ACK_DAYS,
        },
        "liaison.email_kind": {
            "statement": "What kind a supplier email is, from fixed phrases.",
            **{k: p.pattern for k, p in liaison.KIND_PHRASES.items()},
        },
        "schedule.turnaround": {
            "statement": "Turnaround activities sit under this WBS node.",
            "TURNAROUND_WBS": schedule.TURNAROUND_WBS,
        },
    }


@dataclass
class Library:
    connectors: dict[str, str] = field(default_factory=dict)
    dialects: dict[str, dict] = field(default_factory=dict)
    rule_cards: dict[str, dict] = field(default_factory=dict)
    patterns: dict[str, list[dict]] = field(default_factory=dict)
    assets: dict[str, AssetEntry] = field(default_factory=dict)

    @classmethod
    def new(cls) -> Library:
        return cls(
            connectors={name: c.description for name, c in CONNECTORS.items()},
            dialects={"sap": identity_mapping().as_dict()},
            rule_cards=rule_cards(),
        )

    @classmethod
    def load(cls, path: str | Path) -> Library:
        d = json.loads(Path(path).read_text())
        d["assets"] = {k: AssetEntry(**v) for k, v in d["assets"].items()}
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    def mapping(self, dialect: str) -> Mapping:
        return Mapping.from_dict(self.dialects[dialect])

    def add_dialect(self, name: str, mapping: Mapping) -> None:
        self.dialects[name] = mapping.as_dict()

    def add_asset(self, entry: AssetEntry) -> None:
        """Register an asset, with any link patterns its run has confirmed."""
        self.assets[entry.code] = entry
        path = Path(entry.run) / "reports" / "patterns.json"
        if path.exists():
            self.patterns[entry.code] = [
                p for p in json.loads(path.read_text()) if p.get("status") == "confirmed"
            ]

    def coverage(self, dialect: str, sources: dict[str, Connector]) -> float:
        """Share of the dialect's mapped columns that the raw extracts have."""
        m = self.mapping(dialect)
        have: set[tuple[str, str]] = set()
        for con in sources.values():
            for t in con.entity_types():
                have |= {(t, c) for c in con.schema(t)}
        wanted = [(m.tables[f.split(".", 1)[0]], col) for f, col in m.fields.items() if col]
        return sum(w in have for w in wanted) / len(wanted) if wanted else 0.0

    def find_dialect(self, sources: dict[str, Connector]) -> tuple[str | None, float]:
        """The best-covering dialect for these raw extracts, if it covers enough."""
        cover, best = max((self.coverage(d, sources), d) for d in self.dialects)
        return (best if cover >= MIN_COVERAGE else None), round(cover, 4)
