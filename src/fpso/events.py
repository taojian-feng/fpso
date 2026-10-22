"""Event bus (Phase 3): agents react to what changed instead of re-reading on a timer.

In-process publish and subscribe with a JSONL log, so a run can be replayed and audited.
The Phase 3 replay publishes what each source gained since the previous day (new PO lines,
supplier emails, manifest lines, forecasts, vibration readings) and the agents' own
findings; the orchestrator listens to all of it. A live deployment would put the same
topics on a broker; nothing that publishes or subscribes would change.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Event:
    topic: str
    payload: dict[str, Any]


class EventBus:
    def __init__(self, log_path: Path | None = None):
        self.subscribers: dict[str, list[Callable[[Event], None]]] = defaultdict(list)
        self.log_path = log_path
        self.counts: dict[str, int] = defaultdict(int)
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.unlink(missing_ok=True)

    def subscribe(self, topic: str, handler: Callable[[Event], None]) -> None:
        """``topic`` may end in ``*`` to receive every topic with that prefix."""
        self.subscribers[topic].append(handler)

    def publish(self, topic: str, **payload) -> None:
        ev = Event(topic, payload)
        self.counts[topic] += 1
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps({"topic": topic, **payload}, default=str) + "\n")
        for pattern, handlers in list(self.subscribers.items()):
            if pattern == topic or (pattern.endswith("*") and topic.startswith(pattern[:-1])):
                for h in handlers:
                    h(ev)


def source_changes(prev: dict[str, set[str]], now: dict[str, set[str]]) -> dict[str, list[str]]:
    """Keys each source/entity gained since the previous snapshot."""
    return {k: sorted(now[k] - prev.get(k, set())) for k in now if now[k] - prev.get(k, set())}


def snapshot(connectors) -> dict[str, set[str]]:
    return {
        f"{name}/{t}": {r.key for r in c.iter(t)}
        for name, c in connectors.items()
        for t in c.entity_types()
    }
