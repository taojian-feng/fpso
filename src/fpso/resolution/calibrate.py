"""Calibration (plan WS5 step 3): map model confidence to observed accuracy.

One isotonic fit per link type on reviewer-labelled examples, then an auto-accept
threshold: the lowest calibrated probability at which the accepted links still hold the
target precision on those labels. Pure Python, so it needs no numeric libraries.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

PRIOR_WEIGHT = 3  # labels' worth of trust in the model's own confidence, per block


def _pav(blocks: list[list[float]]) -> list[list[float]]:
    """Pool adjacent violators over [weighted_sum_y, weight, min_x, sum_x, count] blocks."""
    out: list[list[float]] = []
    for b in blocks:
        out.append(list(b))
        while len(out) > 1 and out[-2][0] / out[-2][1] >= out[-1][0] / out[-1][1]:
            last = out.pop()
            for i in (0, 1, 3, 4):
                out[-1][i] += last[i]
    return out


def _raw_blocks(xs: list[float], ys: list[float]) -> list[list[float]]:
    by_x: dict[float, list[float]] = {}
    for x, y in zip(xs, ys, strict=True):
        by_x.setdefault(x, []).append(y)
    # Ties start as one block.
    return [[sum(v), len(v), x, x * len(v), len(v)] for x, v in sorted(by_x.items())]


def isotonic_fit(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Pool-adjacent-violators. Returns block lower bounds and their fitted values."""
    blocks = _pav(_raw_blocks(xs, ys))
    return [b[2] for b in blocks], [b[0] / b[1] for b in blocks]


@dataclass
class Calibrator:
    lowers: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    @classmethod
    def fit(cls, scores: list[float], labels: list[bool], prior_weight: float = PRIOR_WEIGHT):
        """Isotonic fit, with each block shrunk toward its mean model confidence.

        A block with only a few labels, all correct, then reads as likely rather than
        certain: ten right answers at confidence 0.75 calibrate to about 0.94, not 1.0.
        """
        if not scores:
            return cls()
        blocks = _raw_blocks(scores, [float(y) for y in labels])
        shrunk = [
            [b[0] + prior_weight * b[3] / b[4], b[1] + prior_weight, b[2], b[3], b[4]]
            for b in blocks
        ]
        pooled = _pav(shrunk)
        return cls([b[2] for b in pooled], [b[0] / b[1] for b in pooled])

    def predict(self, score: float) -> float:
        if not self.values:
            return score  # nothing to fit on: trust the raw confidence
        # The last block starting at or below the score. A score in a gap between blocks
        # gets the lower block's value, so unseen confidences are never rounded up.
        i = bisect_right(self.lowers, score) - 1
        return self.values[max(i, 0)]


def choose_threshold(probs: list[float], labels: list[bool], target: float = 0.95) -> float:
    """Lowest threshold whose accepted set has precision >= target. Above 1 means none.

    Every accepted link must also be calibrated at or above the target on its own.
    """
    best = 1.01
    for t in sorted({p for p in probs if p >= target}, reverse=True):
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
