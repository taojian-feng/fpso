"""Write-back with approval (Phase 2): the first actions agents may take, each one approved.

Agents propose actions; a planner approves or rejects each; only approved actions execute.
In this mock, executing writes to an outbox per target system (``outbox/<system>.jsonl``)
instead of calling a real system, so the whole loop runs and is scored without credentials.

- ``create_purchase_requisition`` (erp-mm): buy a part the Procurement agent says to order now;
- ``request_vessel_slot`` (vessel): extra deck space on a sailing that makes the need date;
- ``book_helicopter_cargo`` (helicopter): a light part on a crew-change flight;
- ``transfer_stock`` (warehouse): reserve and move a spare from another store.

Every action has a key (kind and component), so a part is never actioned twice: an approved
action is not proposed again, and a rejected one is not raised again. Execution is
idempotent on the action id.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from fpso.eval.reviewers import SimulatedReviewer

SYSTEM_OF = {
    "create_purchase_requisition": "erp-mm",
    "request_vessel_slot": "vessel",
    "book_helicopter_cargo": "helicopter",
    "transfer_stock": "warehouse",
}
# Mock unit costs used to compare options; Phase 3's orchestrator weighs them.
HELI_USD_PER_KG, HELI_USD_FIXED = 14.0, 600.0
SLOT_USD_PER_M2, SLOT_USD_MIN = 260.0, 1500.0
TRANSFER_USD = {"use_offshore_stock": 0.0, "transfer_from_sister": 900.0}


@dataclass
class Action:
    kind: str
    component: str
    as_of: date
    payload: dict = field(default_factory=dict)
    reason: str = ""
    cost_usd: float = 0.0
    agent: str = ""

    @property
    def key(self) -> str:
        return f"{self.kind}|{self.component}"

    @property
    def action_id(self) -> str:
        return f"{self.as_of.isoformat()}|{self.key}"

    @property
    def system(self) -> str:
        return SYSTEM_OF[self.kind]


@dataclass
class Decision:
    action: str  # action id
    verdict: str  # approved, rejected, skipped
    correct: bool


class ApprovalQueue:
    """Proposals in, a planner's verdict on each, approved actions out to the outbox."""

    def __init__(self, out_dir: Path, planner: SimulatedReviewer):
        self.out_dir = out_dir
        self.planner = planner
        self.done: set[str] = set()  # keys approved and executed
        self.rejected: set[str] = set()
        self.executed_ids: set[str] = set()
        self.log = out_dir / "decisions.jsonl"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.log.unlink(missing_ok=True)
        for p in (out_dir / "outbox").glob("*.jsonl"):
            p.unlink()

    def open(self, actions: list[Action]) -> list[Action]:
        """Actions not already done or rejected, one per key."""
        seen, out = set(), []
        for a in actions:
            if a.key in self.done or a.key in self.rejected or a.key in seen:
                continue
            seen.add(a.key)
            out.append(a)
        return out

    def review(self, actions: list[Action], truth_keys: set[str]) -> list[Decision]:
        out = []
        for a in self.open(actions):
            correct = a.key in truth_keys
            answer = self.planner.confirm_gap(a.action_id, correct)
            verdict = "skipped" if answer is None else "approved" if answer else "rejected"
            out.append(Decision(a.action_id, verdict, correct))
            if verdict == "approved":
                self.execute(a)
            elif verdict == "rejected":
                self.rejected.add(a.key)
            with self.log.open("a") as f:
                f.write(json.dumps({**asdict(out[-1]), "proposal": asdict(a)}, default=str) + "\n")
        return out

    def execute(self, a: Action) -> None:
        if a.action_id in self.executed_ids:
            return
        self.executed_ids.add(a.action_id)
        self.done.add(a.key)
        box = self.out_dir / "outbox" / f"{a.system}.jsonl"
        box.parent.mkdir(parents=True, exist_ok=True)
        with box.open("a") as f:
            f.write(json.dumps({"id": a.action_id, **asdict(a)}, default=str) + "\n")


def score_decisions(decisions: list[Decision]) -> dict:
    n = {v: sum(d.verdict == v for d in decisions) for v in ("approved", "rejected", "skipped")}
    answered = n["approved"] + n["rejected"]
    return {
        "proposed": len(decisions),
        **n,
        "correct": sum(d.correct for d in decisions),
        "approved_share": round(n["approved"] / answered, 4) if answered else None,
        "precision": round(sum(d.correct for d in decisions) / len(decisions), 4)
        if decisions
        else None,
    }


def heli_cost(weight_kg: float) -> float:
    return round(HELI_USD_FIXED + HELI_USD_PER_KG * weight_kg, 2)


def slot_cost(area_m2: float) -> float:
    return round(max(SLOT_USD_MIN, SLOT_USD_PER_M2 * area_m2), 2)
