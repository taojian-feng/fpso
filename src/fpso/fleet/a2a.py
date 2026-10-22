"""Supplier-side agents over a mock, in-process A2A protocol.

Each vendor that takes part publishes an agent card (name, url, skills). Buyers talk to it
in JSON-RPC style messages, serialised to JSON text both ways so nothing but JSON crosses:

    {"jsonrpc": "2.0", "id": 1, "method": "tasks/send",
     "params": {"id": "<task id>", "message": {"role": "user", "parts": [
         {"type": "data", "data": {"skill": "order_status", "buyer": "MRL", "po": "...",
                                   "as_of": "2027-03-01"}}]}}}

and get a task back with a state (``completed``, ``working``, ``failed``) and, once
completed, an artifact with the answer. ``tasks/get`` asks again later for a task that was
still working.

A supplier agent answers from the truth world as its own order book stood that day:
whether it has the order, each line's current promised date and whether it has gone. A
supplier knows of a delay ``KNOWN_AHEAD_DAYS`` before its delay email goes out. Agents can
be unreliable: some tasks get no answer, some are answered only after the buyer's
deadline, and some answers repeat the original date and miss a slip.

``LiaisonClient`` is the Supplier Liaison agent's side: for each chase it would draft, it
asks the vendor's agent first and falls back to the email draft when there is no agent or
no answer by the deadline.
"""

from __future__ import annotations

import json
import random
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from fpso.synthetic.truth import POLine, World

KNOWN_AHEAD_DAYS = 10  # a supplier knows of a delay this long before it emails about it
DEADLINE_HOURS = 24  # how long the buyer waits for an answer before sending the email
LAST_LOOK_HOURS = 7 * 24


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    description: str


@dataclass
class AgentCard:
    name: str
    description: str
    url: str
    skills: list[Skill]
    version: str = "0.1"
    vendor_id: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


SKILLS = [
    Skill(
        "order_status",
        "Order status",
        "Whether the order is acknowledged, each line's promised date, whether it has shipped.",
    ),
    Skill(
        "expedite",
        "Expedite request",
        "The earliest date each line could ship if expedited, and whether the request is taken.",
    ),
]


@dataclass(frozen=True)
class Reliability:
    answers: float = 1.0  # share of tasks answered at all
    late: float = 0.0  # share of answered tasks answered after the buyer's deadline
    late_hours: int = 72
    stale: float = 0.0  # share of answers that give the first promised date, missing slips


class SupplierAgent:
    """One vendor's order desk, answering from its order book for each buyer asset."""

    def __init__(self, vendor, worlds: dict[str, World], reliability: Reliability, seed: int = 0):
        self.vendor = vendor
        self.reliability = reliability
        self.seed = seed
        self.card = AgentCard(
            f"{vendor.name} order desk",
            f"Order status and expedite requests for {vendor.name}.",
            f"a2a://{vendor.domain}/orders",
            SKILLS,
            vendor_id=vendor.vendor_id,
        )
        self.lines: dict[tuple[str, str], list[POLine]] = defaultdict(list)
        self.worlds = worlds
        for buyer, w in worlds.items():
            for pl in w.po_lines:
                if pl.vendor_id == vendor.vendor_id:
                    self.lines[(buyer, pl.po_no)].append(pl)
        self.tasks: dict[str, dict] = {}

    def handle(self, request: dict) -> dict:
        method, params = request.get("method"), request.get("params", {})
        if method == "tasks/send":
            result = self._send(params)
        elif method == "tasks/get":
            result = self._get(params)
        else:
            return _error(request, -32601, f"unknown method {method!r}")
        return {"jsonrpc": "2.0", "id": request.get("id"), "result": result}

    def _send(self, params: dict) -> dict:
        task_id = params["id"]
        data = next((p["data"] for p in params["message"]["parts"] if p.get("type") == "data"), {})
        rng = random.Random(zlib.crc32(f"{self.seed}|{self.vendor.vendor_id}|{task_id}".encode()))
        r = self.reliability
        if rng.random() >= r.answers:
            ready_at = None  # never answered
        elif rng.random() < r.late:
            ready_at = r.late_hours
        else:
            ready_at = 0
        stale = rng.random() < r.stale
        answer = self._answer(data, stale)
        state = "failed" if answer is None else "completed"
        self.tasks[task_id] = {"answer": answer, "ready_at": ready_at, "state": state}
        return self._task(task_id, 0)

    def _get(self, params: dict) -> dict:
        if params["id"] not in self.tasks:
            return {"id": params["id"], "status": {"state": "failed", "message": "unknown task"}}
        return self._task(params["id"], params.get("hours", 0))

    def _task(self, task_id: str, hours: int) -> dict:
        t = self.tasks[task_id]
        if t["ready_at"] is None or hours < t["ready_at"]:
            return {"id": task_id, "status": {"state": "working"}}
        if t["state"] == "failed":
            return {"id": task_id, "status": {"state": "failed", "message": "unknown order"}}
        return {
            "id": task_id,
            "status": {"state": "completed"},
            "artifacts": [{"parts": [{"type": "data", "data": t["answer"]}]}],
        }

    def _answer(self, data: dict, stale: bool) -> dict | None:
        as_of = date.fromisoformat(data["as_of"])
        lines = [
            pl
            for pl in self.lines.get((data.get("buyer"), data.get("po")), [])
            if pl.order_date <= as_of
        ]
        if not lines:
            return None
        world = self.worlds[data["buyer"]]
        out = {"po": data["po"], "acknowledged": True, "lines": []}
        for pl in sorted(lines, key=lambda p: p.item):
            promised, dispatched = order_book(world, pl, as_of)
            if stale:
                promised = pl.delivery_date
            line = {"line": pl.key, "promised": promised.isoformat(), "dispatched": dispatched}
            if data.get("skill") == "expedite":
                gain = zlib.crc32(pl.key.encode()) % 15
                line["earliest"] = max(as_of + timedelta(days=3), promised - timedelta(days=gain))
                line["earliest"] = line["earliest"].isoformat()
            out["lines"].append(line)
        return out


def order_book(world: World, pl: POLine, as_of: date) -> tuple[date, bool]:
    """The supplier's own view of a line on ``as_of``: promised date, and dispatched."""
    promised = pl.delivery_date
    dispatched = False
    for e in world.emails:
        if pl.key not in e.po_line_keys:
            continue
        if e.kind == "slip" and e.sent - timedelta(days=KNOWN_AHEAD_DAYS) <= as_of:
            promised = max(promised, e.new_date)
        if e.kind == "asn" and e.sent <= as_of:
            dispatched = True
    return promised, dispatched


def _error(request: dict, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request.get("id"), "error": {"code": code, "message": message}}


class Bus:
    """In-process transport. Requests and responses cross as JSON text only."""

    def __init__(self):
        self.agents: dict[str, SupplierAgent] = {}
        self.sent = 0

    def register(self, agent: SupplierAgent) -> None:
        self.agents[agent.card.url] = agent

    def cards(self) -> list[dict]:
        return [json.loads(json.dumps(a.card.as_dict())) for a in self.agents.values()]

    def call(self, url: str, request: dict) -> dict:
        self.sent += 1
        agent = self.agents.get(url)
        if agent is None:
            return _error(request, -32000, f"no agent at {url}")
        return json.loads(json.dumps(agent.handle(json.loads(json.dumps(request)))))


@dataclass
class Exchange:
    """One chase the Liaison agent would send, and how it was settled."""

    day: str
    po: str
    vendor: str
    kind: str  # no_ack, expedite
    channel: str  # a2a, email
    reason: str = ""  # why it fell back to email: no_agent, no_answer, late, failed
    answer: dict | None = None
    hours: int = 0


@dataclass
class LiaisonClient:
    """Asks supplier agents first; drafts the email only when they do not answer in time."""

    bus: Bus
    buyer: str
    deadline_hours: int = DEADLINE_HOURS
    directory: dict[str, str] = field(default_factory=dict)  # vendor id -> agent url

    def __post_init__(self):
        if not self.directory:
            self.directory = {c["vendor_id"]: c["url"] for c in self.bus.cards()}

    def ask(self, day: date, chase: dict) -> Exchange:
        po, vendor, kind = chase["po"], chase["vendor"], chase["kind"]
        ex = Exchange(day.isoformat(), po, vendor, kind, "email")
        url = self.directory.get(vendor)
        if url is None:
            ex.reason = "no_agent"
            return ex
        task_id = f"{self.buyer}-{day.isoformat()}-{kind}-{po}"
        skill = "order_status" if kind == "no_ack" else "expedite"
        data = {"skill": skill, "buyer": self.buyer, "po": po, "as_of": day.isoformat()}
        if chase.get("po_lines"):
            data["lines"] = chase["po_lines"]
        msg = {"role": "user", "parts": [{"type": "data", "data": data}]}
        req = {"jsonrpc": "2.0", "id": 1, "method": "tasks/send"}
        resp = self.bus.call(url, {**req, "params": {"id": task_id, "message": msg}})
        task = resp.get("result", {})
        if task.get("status", {}).get("state") == "working":
            again = {"jsonrpc": "2.0", "id": 2, "method": "tasks/get"}
            params = {"id": task_id, "hours": self.deadline_hours}
            task = self.bus.call(url, {**again, "params": params}).get("result", {})
            ex.hours = self.deadline_hours
        state = task.get("status", {}).get("state")
        if state == "completed":
            ex.channel = "a2a"
            ex.answer = task["artifacts"][0]["parts"][0]["data"]
        elif state == "working":
            # The email goes out now; a last look a week later tells late from silent.
            again = {"jsonrpc": "2.0", "id": 3, "method": "tasks/get"}
            params = {"id": task_id, "hours": LAST_LOOK_HOURS}
            last = self.bus.call(url, {**again, "params": params}).get("result", {})
            done = last.get("status", {}).get("state") == "completed"
            ex.reason = "late" if done else "no_answer"
        else:
            ex.reason = "failed"
        return ex


def default_reliability(vendors) -> dict[str, Reliability | None]:
    """Who has an agent and how reliable it is: the first eight vendors (the makers) run
    agents, two of them patchy, one slow and one with a stale order book; the four
    distributors have none and are reached by email only."""
    plan = [
        Reliability(),
        Reliability(answers=0.7),
        Reliability(),
        Reliability(late=0.4),
        Reliability(),
        Reliability(stale=0.3),
        Reliability(answers=0.8),
        Reliability(),
    ]
    return {v.vendor_id: (plan[i] if i < len(plan) else None) for i, v in enumerate(vendors)}


def build_bus(worlds: dict[str, World], seed: int = 0) -> Bus:
    """A bus with one supplier agent per vendor that runs one."""
    any_world = next(iter(worlds.values()))
    bus = Bus()
    for v in any_world.vendors:
        r = default_reliability(any_world.vendors)[v.vendor_id]
        if r is not None:
            bus.register(SupplierAgent(v, worlds, r, seed))
    return bus


def score_exchanges(exchanges: list[Exchange], world: World) -> dict:
    """Share of chases settled over A2A, and how right the promised dates were."""
    pl_by_key = {pl.key: pl for pl in world.po_lines}
    lines = right = early = slipped = slipped_right = 0
    slip_email = defaultdict(list)
    for e in world.emails:
        if e.kind == "slip":
            for k in e.po_line_keys:
                slip_email[k].append(e.sent)
    for ex in exchanges:
        if ex.channel != "a2a":
            continue
        day = date.fromisoformat(ex.day)
        for line in ex.answer["lines"]:
            pl = pl_by_key[line["line"]]
            true, _ = order_book(world, pl, day)
            ok = date.fromisoformat(line["promised"]) == true
            lines += 1
            right += ok
            if true > pl.delivery_date:  # the hard case: the line has slipped
                slipped += 1
                slipped_right += ok
            # A slip the buyer heard of over A2A before any delay email had been sent.
            if true > pl.delivery_date and not any(s <= day for s in slip_email[pl.key]):
                early += ok
    by_kind = {}
    for kind in ("no_ack", "expedite"):
        xs = [x for x in exchanges if x.kind == kind]
        a2a = sum(x.channel == "a2a" for x in xs)
        by_kind[kind] = {
            "questions": len(xs),
            "resolved_by_a2a": a2a,
            "share": round(a2a / len(xs), 4) if xs else None,
        }
    n = len(exchanges)
    a2a = sum(x.channel == "a2a" for x in exchanges)
    reasons = defaultdict(int)
    for x in exchanges:
        if x.channel == "email":
            reasons[x.reason] += 1
    return {
        "questions": n,
        "resolved_by_a2a": a2a,
        "resolved_share": round(a2a / n, 4) if n else None,
        "emails_still_needed": n - a2a,
        "email_reasons": dict(sorted(reasons.items())),
        "by_kind": by_kind,
        "lines_answered": lines,
        "promised_date_accuracy": round(right / lines, 4) if lines else None,
        "slipped_lines_answered": slipped,
        "slipped_line_accuracy": round(slipped_right / slipped, 4) if slipped else None,
        "slips_learned_before_email": early,
    }
