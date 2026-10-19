"""Calibration (plan WS5 step 3): map model confidence to observed accuracy.

One isotonic fit per link type on reviewer-labelled examples, then an auto-accept
threshold: the lowest calibrated probability at which the accepted links still hold the
target precision on those labels. Pure Python, so it needs no numeric libraries.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field


def isotonic_fit(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Pool-adjacent-violators. Returns block upper bounds and their fitted values."""
    by_x: dict[float, list[float]] = {}
    for x, y in zip(xs, ys, strict=True):
        by_x.setdefault(x, []).append(y)
    blocks: list[list[float]] = []  # [sum_y, count, max_x]; ties start as one block
    for x in sorted(by_x):
        blocks.append([sum(by_x[x]), len(by_x[x]), x])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            s, n, mx = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
            blocks[-1][2] = mx
    return [b[2] for b in blocks], [b[0] / b[1] for b in blocks]


@dataclass
class Calibrator:
    uppers: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    @classmethod
    def fit(cls, scores: list[float], labels: list[bool]) -> Calibrator:
        if not scores:
            return cls()
        uppers, values = isotonic_fit(scores, [float(y) for y in labels])
        return cls(uppers, values)

    def predict(self, score: float) -> float:
        if not self.values:
            return score  # nothing to fit on: trust the raw confidence
        # The block whose upper bound is the first one at or above the score.
        i = bisect_left(self.uppers, score)
        return self.values[min(i, len(self.values) - 1)]


def choose_threshold(probs: list[float], labels: list[bool], target: float = 0.95) -> float:
    """Lowest threshold whose accepted set has precision >= target. Above 1 means none."""
    best = 1.01
    for t in sorted(set(probs), reverse=True):
        accepted = [y for p, y in zip(probs, labels, strict=True) if p >= t]
        if sum(accepted) / len(accepted) >= target:
            best = t
        else:
            break
    return best


def expected_calibration_error(probs: list[float], labels: list[bool], bins: int = 10) -> float:
    if not probs:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(labels[i] for i in idx) / len(idx)
            total += len(idx) / len(probs) * abs(conf - acc)
    return total
