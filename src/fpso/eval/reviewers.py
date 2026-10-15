"""Simulated domain reviewers (plan WS1): agents that answer from the ground truth with
deliberate noise, so the feedback loop can run without real people.

Answers are deterministic per question, so re-asking gives the same answer, as with a
person who remembers. Every answer is logged under a 'sim:' reviewer id so it is never
mistaken for real human input.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field

from fpso.governance import ActionLog, ActionRecord

Pair = tuple[str, str, str]


@dataclass
class SimulatedReviewer:
    name: str
    role: str
    truth: set[Pair]
    error_rate: float = 0.05
    skip_rate: float = 0.10
    seed: int = 0
    log: ActionLog = field(default_factory=ActionLog)

    @property
    def reviewer_id(self) -> str:
        return f"sim:{self.name}"

    def _rng(self, question: str) -> random.Random:
        return random.Random(zlib.crc32(f"{self.seed}|{self.name}|{question}".encode()))

    def _answer(self, question: str, correct: bool) -> bool | None:
        rng = self._rng(question)
        if rng.random() < self.skip_rate:
            answer = None
        elif rng.random() < self.error_rate:
            answer = not correct
        else:
            answer = correct
        self.log.write(
            ActionRecord(
                agent=self.reviewer_id,
                action="review",
                inputs={"question": question, "role": self.role},
                reasoning=None if answer is None else ("yes" if answer else "no"),
            )
        )
        return answer

    def label(self, link: Pair) -> bool | None:
        """Is this a true link? None means the reviewer skipped the question."""
        return self._answer("label:" + "|".join(link), link in self.truth)

    def confirm_pattern(
        self, pattern_id: str, examples: list[Pair], bar: float = 0.9
    ) -> bool | None:
        """Confirm a proposed matching rule after looking at its example links.

        A careful reviewer confirms when at least ``bar`` of the examples are right.
        """
        if not examples:
            return None
        share = sum(e in self.truth for e in examples) / len(examples)
        return self._answer(f"pattern:{pattern_id}", share >= bar)


def default_reviewers(truth: set[Pair], seed: int = 0, log: ActionLog | None = None):
    """The two mock reviewers from the plan."""
    log = log or ActionLog()
    return [
        SimulatedReviewer("dana.reyes", "planner", truth, seed=seed, log=log),
        SimulatedReviewer("lee.park", "materials coordinator", truth, seed=seed, log=log),
    ]
