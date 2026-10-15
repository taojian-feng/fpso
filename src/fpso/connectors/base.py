"""The uniform read-only contract every source connector implements (design layer L1).

Agents only ever see this contract, so a file-drop adapter can later be swapped for a
live API without touching them. Connectors return records as the source holds them:
raw field names and raw values. Working out what the fields mean is the agents' job.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Record:
    source: str
    entity_type: str
    key: str
    fields: dict[str, Any] = field(hash=False)
    extracted_at: str = ""

    def text(self) -> str:
        """All field values joined, for free-text search."""
        return " ".join(str(v) for v in self.fields.values() if v not in (None, ""))

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "entity_type": self.entity_type,
            "key": self.key,
            "fields": self.fields,
            "extracted_at": self.extracted_at,
        }


class Connector:
    """Base class for file-backed connectors. Subclasses fill ``self._records`` in ``load``."""

    name: str = ""
    description: str = ""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._records: dict[str, dict[str, Record]] = {}
        self.load()

    # Subclasses implement this.
    def load(self) -> None:
        raise NotImplementedError

    def _add(self, entity_type: str, key: str, fields: dict[str, Any], path: Path) -> None:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
        bucket = self._records.setdefault(entity_type, {})
        if key in bucket:
            raise ValueError(f"{self.name}: duplicate {entity_type} key {key!r}")
        bucket[key] = Record(self.name, entity_type, key, fields, stamp)

    # The read contract.
    def entity_types(self) -> list[str]:
        return sorted(self._records)

    def count(self, entity_type: str) -> int:
        return len(self._records.get(entity_type, {}))

    def schema(self, entity_type: str) -> list[str]:
        """Field names seen on records of this type, in first-seen order."""
        seen: dict[str, None] = {}
        for r in self._records.get(entity_type, {}).values():
            seen.update(dict.fromkeys(r.fields))
        return list(seen)

    def iter(self, entity_type: str) -> Iterator[Record]:
        return iter(self._records.get(entity_type, {}).values())

    def get(self, entity_type: str, key: str) -> Record | None:
        return self._records.get(entity_type, {}).get(key)

    def sample(self, entity_type: str, n: int = 5) -> list[Record]:
        return list(self._records.get(entity_type, {}).values())[:n]

    def search(self, entity_type: str, query: str, limit: int = 20) -> list[Record]:
        """Records whose key or field text contains every term, case-insensitive."""
        terms = [t.lower() for t in query.split() if t]
        out = []
        for r in self.iter(entity_type):
            text = f"{r.key} {r.text()}".lower()
            if all(t in text for t in terms):
                out.append(r)
                if len(out) >= limit:
                    break
        return out


def read_csv(path: Path) -> Iterable[dict[str, str]]:
    import csv

    with path.open(newline="") as f:
        yield from csv.DictReader(f)
