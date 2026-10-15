"""Governance: every agent step is recorded with its inputs, reasoning, model and cost."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ActionRecord:
    agent: str
    action: str
    inputs: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class ActionLog:
    """Append-only JSONL log. Mirrors the action_log table so runs work without a database."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.records: list[ActionRecord] = []
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: ActionRecord) -> None:
        with self._lock:
            self.records.append(record)
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(asdict(record), default=str) + "\n")

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd or 0.0 for r in self.records)
