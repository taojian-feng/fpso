"""Rule cards: logistics rules written in plain language and compiled to checks and tests.

A planner writes a card as one English sentence. Each sentence is matched against a small set
of templates; a match yields the rule's parameter (a limit, a number of days) and a check
the logistics plan must pass. Every compiled card also writes its own test cases, one just
inside and one just outside its limit, and ``self_test`` runs them, so a card that compiles
to the wrong check fails before any plan uses it.

A sentence no template matches is reported as not understood rather than guessed at; a model
could draft a template for it later, but nothing it drafts is used until it compiles here.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CARDS = Path(__file__).with_name("rules") / "kestrel.txt"


@dataclass
class Movement:
    """One planned move of cargo, as the checks see it."""

    mode: str  # vessel, helicopter, transfer
    weight_kg: float = 0.0
    shore_days: int | None = None  # days between shore-base delivery and departure
    transit_days: int | None = None
    bunkering: bool = False  # lifted offshore on a bunkering day
    cranes_available: int = 1
    deck_free_after_m2: float | None = None
    deck_m2: float | None = None
    pax: int = 0
    payload_used_kg: float = 0.0
    max_payload_kg: float = 0.0
    hs_m: float = 0.0  # wave height on the day it sails (Phase 3)
    wind_kn: float = 0.0  # wind on the day it flies (Phase 3)
    cost_usd: float = 0.0  # the action that moves it
    auto: bool = False  # executed without a person's approval


@dataclass
class RuleCard:
    text: str
    name: str
    params: dict
    check: Callable[[Movement], str | None]  # a violation message, or None
    cases: list[tuple[Movement, bool]] = field(default_factory=list)  # (movement, passes)

    def self_test(self) -> list[str]:
        out = []
        for m, ok in self.cases:
            if (self.check(m) is None) != ok:
                out.append(f"{self.name}: expected {'pass' if ok else 'fail'} for {m}")
        return out


@dataclass
class RuleSet:
    cards: list[RuleCard]
    not_understood: list[str]

    def param(self, name: str, default):
        for c in self.cards:
            if c.name == name:
                return c.params["value"]
        return default

    def violations(self, m: Movement) -> list[str]:
        return [v for c in self.cards if (v := c.check(m))]

    def self_test(self) -> list[str]:
        return [e for c in self.cards for e in c.self_test()]


def _num(m: re.Match) -> float:
    return float(m.group("n"))


def _heli_item(m):
    n = _num(m)
    return (
        "heli_max_item_kg",
        n,
        lambda x: (
            f"{x.weight_kg:g} kg item by helicopter, limit {n:g} kg"
            if x.mode == "helicopter" and x.weight_kg > n
            else None
        ),
        [(Movement("helicopter", n), True), (Movement("helicopter", n + 1), False)],
    )


def _pax_kg(m):
    n = _num(m)

    def check(x):
        if x.mode != "helicopter":
            return None
        used = x.pax * n + x.payload_used_kg
        return (
            f"flight over payload: {used:g} of {x.max_payload_kg:g} kg"
            if used > x.max_payload_kg
            else None
        )

    full = Movement("helicopter", 0, pax=10, max_payload_kg=10 * n + 100)
    return (
        "pax_kg",
        n,
        check,
        [
            (Movement(**{**full.__dict__, "payload_used_kg": 100}), True),
            (Movement(**{**full.__dict__, "payload_used_kg": 101}), False),
        ],
    )


def _no_bunkering(m):
    return (
        "no_lift_when_bunkering",
        True,
        lambda x: "cargo lifted on a bunkering day" if x.mode == "vessel" and x.bunkering else None,
        [(Movement("vessel"), True), (Movement("vessel", bunkering=True), False)],
    )


def _crane_needed(m):
    return (
        "crane_needed",
        True,
        lambda x: (
            "cargo lifted with no crane available"
            if x.mode == "vessel" and x.cranes_available < 1
            else None
        ),
        [(Movement("vessel"), True), (Movement("vessel", cranes_available=0), False)],
    )


def _shore_days(m):
    n = int(_num(m))
    return (
        "shore_handling_days",
        n,
        lambda x: (
            f"sails {x.shore_days} days after delivery, needs {n}"
            if x.mode == "vessel" and x.shore_days is not None and x.shore_days < n
            else None
        ),
        [(Movement("vessel", shore_days=n), True), (Movement("vessel", shore_days=n - 1), False)],
    )


def _transit(m):
    n = int(_num(m))
    return (
        "transit_days",
        n,
        lambda x: (
            f"planned transit {x.transit_days} days, a sailing takes {n}"
            if x.mode == "vessel" and x.transit_days is not None and x.transit_days < n
            else None
        ),
        [
            (Movement("vessel", transit_days=n), True),
            (Movement("vessel", transit_days=n - 1), False),
        ],
    )


def _transfer(m):
    n = int(_num(m))
    return (
        "transfer_days",
        n,
        lambda x: (
            f"transfer planned in {x.transit_days} days, takes {n}"
            if x.mode == "transfer" and x.transit_days is not None and x.transit_days < n
            else None
        ),
        [
            (Movement("transfer", transit_days=n), True),
            (Movement("transfer", transit_days=n - 1), False),
        ],
    )


def _deck_reserve(m):
    n = _num(m)

    def check(x):
        if x.mode != "vessel" or x.deck_free_after_m2 is None or not x.deck_m2:
            return None
        if x.deck_free_after_m2 < x.deck_m2 * n / 100:
            return f"deck reserve below {n:g}%"
        return None

    return (
        "deck_reserve_pct",
        n,
        check,
        [
            (Movement("vessel", deck_free_after_m2=n, deck_m2=100), True),
            (Movement("vessel", deck_free_after_m2=n - 0.1, deck_m2=100), False),
        ],
    )


def _sail_hs(m):
    n = _num(m)
    return (
        "sailing_max_hs_m",
        n,
        lambda x: (
            f"sails in {x.hs_m:g} m waves, limit {n:g} m"
            if x.mode == "vessel" and x.hs_m > n
            else None
        ),
        [(Movement("vessel", hs_m=n), True), (Movement("vessel", hs_m=n + 0.1), False)],
    )


def _heli_wind(m):
    n = _num(m)
    return (
        "heli_max_wind_kn",
        n,
        lambda x: (
            f"flies in {x.wind_kn:g} kn wind, limit {n:g} kn"
            if x.mode == "helicopter" and x.wind_kn > n
            else None
        ),
        [(Movement("helicopter", wind_kn=n), True), (Movement("helicopter", wind_kn=n + 1), False)],
    )


def _auto_limit(m):
    n = _num(m)
    return (
        "auto_limit_usd",
        n,
        lambda x: (
            f"{x.cost_usd:,.0f} USD run without approval, limit {n:,.0f} USD"
            if x.auto and x.cost_usd >= n
            else None
        ),
        [
            (Movement("vessel", cost_usd=n - 1, auto=True), True),
            (Movement("vessel", cost_usd=n, auto=True), False),
            (Movement("vessel", cost_usd=n * 10), True),
        ],
    )


N = r"(?P<n>\d+(?:\.\d+)?)"
TEMPLATES: list[tuple[re.Pattern, Callable]] = [
    (
        re.compile(
            rf"no (single )?item (over|heavier than) {N} ?kg (goes|may go|is sent) by helicopter",
            re.I,
        ),
        _heli_item,
    ),
    (re.compile(rf"each passenger counts as {N} ?kg", re.I), _pax_kg),
    (re.compile(r"no cargo (is|may be) lifted on a bunkering day", re.I), _no_bunkering),
    (re.compile(r"cargo (is|may be) lifted only when a crane is available", re.I), _crane_needed),
    (
        re.compile(rf"cargo needs {N} days? at the shore base before it (can )?sails?", re.I),
        _shore_days,
    ),
    (re.compile(rf"a sailing takes {N} days? to reach the asset", re.I), _transit),
    (re.compile(rf"a transfer from the sister asset takes {N} days?", re.I), _transfer),
    (re.compile(rf"keep {N} ?% of (the )?deck (space )?free", re.I), _deck_reserve),
    (re.compile(rf"no sailing when the wave height is over {N} ?m", re.I), _sail_hs),
    (
        re.compile(rf"helicopters? (do not|don't) fly when the wind is over {N} ?(kn|knots)", re.I),
        _heli_wind,
    ),
    (re.compile(rf"actions? under {N} ?usd (may|can) run without approval", re.I), _auto_limit),
]


def compile_card(text: str) -> RuleCard | None:
    for pattern, build in TEMPLATES:
        if m := pattern.search(text):
            name, value, check, cases = build(m)
            return RuleCard(text, name, {"value": value}, check, cases)
    return None


def load_cards(path: str | Path = DEFAULT_CARDS) -> RuleSet:
    cards, unknown = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        card = compile_card(line)
        (cards.append(card) if card else unknown.append(line))
    return RuleSet(cards, unknown)
