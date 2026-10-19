"""Confirmed patterns become deterministic signals (plan WS5 step 4).

A pattern is a general rule about how one link type behaves, stated in plain language for
a reviewer and in parameters the code can run. Phase 0 supports one executable kind,
assignment, which covers the ties the model cannot break on its own:

- from_capacity: how many targets one source record can serve. "one" (each PO line
  serves one work-order component), "quantity" (a PO line ships at most its ordered
  quantity across manifest lines), or "any".
- gap_signal and gap_window: a timing signal from candidate generation and the window
  true links fall in (for example a manifest line ships 1 to 7 days after delivery).

The assignment runs over every candidate of the link type, not just the sampled ones,
because a source record taken by one target is unavailable to all the others. Records
already tied by a strong rule score are assigned first, then the rest in order of how
close their timing is to the middle of the window.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace

from fpso.connectors import Connector
from fpso.resolution.adjudicate import Adjudication
from fpso.resolution.candidates import LINK_ENDS, Candidate
from fpso.resolution.normalize import leading_int

CAPACITIES = ("one", "quantity", "any")
STRONG_RULE_SCORE = 0.9


@dataclass
class Pattern:
    id: str
    link_type: str
    statement: str
    from_capacity: str = "any"
    from_qty_field: str = ""
    to_qty_field: str = ""
    gap_signal: str = ""
    gap_low: float = 0.0
    gap_high: float = 0.0
    status: str = "proposed"  # proposed, confirmed, rejected
    round: int = 0
    reviewer: str = ""
    examples: list = field(default_factory=list)
    wrong_examples: list = field(default_factory=list)  # examples the reviewer marked wrong

    @property
    def signature(self) -> tuple:
        window = (self.gap_low, self.gap_high) if self.gap_signal else None
        return (
            self.link_type,
            self.from_capacity,
            self.from_qty_field,
            self.to_qty_field,
            self.gap_signal,
            window,
        )

    def as_dict(self) -> dict:
        return asdict(self)


def _qty(record, field_name: str) -> int | None:
    if record is None or not field_name:
        return None
    return leading_int(str(record.fields.get(field_name, "")))


def resolve(
    pattern: Pattern, cands: list[Candidate], connectors: dict[str, Connector]
) -> dict[tuple[str, str, str], bool]:
    """Pair -> True when the pattern assigns it, False when it rules it out."""
    (from_src, from_type), (to_src, to_type) = LINK_ENDS[pattern.link_type]
    rows = [c for c in cands if c.link_type == pattern.link_type]
    use_gap = bool(pattern.gap_signal)
    mid = (pattern.gap_low + pattern.gap_high) / 2

    def gap(c: Candidate) -> float | None:
        v = c.signals.get(pattern.gap_signal) if use_gap else None
        return v if isinstance(v, int | float) else None

    def in_window(c: Candidate) -> bool:
        g = gap(c)
        return not use_gap or (g is not None and pattern.gap_low <= g <= pattern.gap_high)

    def capacity(from_key: str) -> float:
        if pattern.from_capacity == "one":
            return 1
        if pattern.from_capacity == "quantity":
            q = _qty(connectors[from_src].get(from_type, from_key), pattern.from_qty_field)
            return q if q else float("inf")
        return float("inf")

    def demand(c: Candidate) -> float:
        if pattern.from_capacity != "quantity":
            return 1
        q = _qty(connectors[to_src].get(to_type, c.to_key), pattern.to_qty_field)
        return q or 1

    def order(c: Candidate):
        g = gap(c)
        return (
            c.score < STRONG_RULE_SCORE,
            abs(g - mid) if g is not None else float("inf"),
            -c.score,
            c.from_key,
            c.to_key,
        )

    assigned: dict[str, str] = {}
    used: dict[str, float] = defaultdict(float)
    for c in sorted(rows, key=order):
        if c.to_key in assigned or not in_window(c):
            continue
        if used[c.from_key] + demand(c) > capacity(c.from_key):
            continue
        assigned[c.to_key] = c.from_key
        used[c.from_key] += demand(c)

    out = {}
    for c in rows:
        if assigned.get(c.to_key) == c.from_key:
            out[c.pair] = True
        elif c.to_key in assigned or used[c.from_key] >= capacity(c.from_key):
            out[c.pair] = False
    return out


def apply_patterns(
    adjs: list[Adjudication],
    patterns: list[Pattern],
    cands: list[Candidate],
    connectors: dict[str, Connector],
) -> list[Adjudication]:
    """Cut model confidence on pairs confirmed patterns rule out; calibration decides the rest."""
    out = list(adjs)
    for p in patterns:
        if p.status != "confirmed":
            continue
        decided = resolve(p, cands, connectors)
        adjusted = []
        for a in out:
            d = decided.get(a.pair) if a.link_type == p.link_type else None
            # A pattern only rules look-alikes out. Assigned pairs keep the model's
            # confidence: on tied look-alikes the assignment is right too rarely (about
            # 60-90% on synthetic data) to lift a link to auto-accept by itself.
            if d is True:
                a = replace(a, evidence=f"{a.evidence} [pattern {p.id} assigns]")
            elif d is False:
                a = replace(
                    a,
                    confidence=round(a.confidence * 0.3, 4),
                    evidence=f"{a.evidence} [pattern {p.id} rules out]",
                )
            adjusted.append(a)
        out = adjusted
    return out
