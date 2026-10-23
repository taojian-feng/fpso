"""Model client used by every agent.

Agents ask for a JSON answer that matches a schema, by role ("adjudicate", "extract",
"induce"), not by model name. The backend decides which Claude model serves each role.
Three backends exist:

- ``StubBackend``: no network, for tests and for work before a model account is ready.
- ``AnthropicBackend``: Claude through the Anthropic API with an API key (Phase 0 default).
- ``BedrockBackend``: Claude on Amazon Bedrock in one region (kept for a later move).
- ``AnthropicBatchBackend``: the Anthropic API's Message Batches, at half price. Questions
  asked together (``ModelClient.complete_many``) go out as one batch and come back in
  minutes rather than seconds, so it suits replays, not a planner waiting on an answer.

Pick one with ``client_from_env()``: ``FPSO_MODEL_BACKEND=anthropic`` uses
``ANTHROPIC_API_KEY``; ``FPSO_MODEL_BACKEND=bedrock`` plus ``AWS_REGION`` (and optionally
``AWS_PROFILE``) selects Bedrock; anything else gives the stub. ``FPSO_MODEL_BATCH=1`` with
the Anthropic backend sends grouped questions through Message Batches.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from fpso.governance import ActionLog, ActionRecord

# Which model serves each role. Bulk adjudication uses the mid-size model, high-volume
# email reading the small one, and pattern induction the large one (plan WS5). Source
# onboarding (WS7) uses the mid-size model.
DEFAULT_ROLE_MODELS: dict[str, str] = {
    "adjudicate": "claude-sonnet-5-5",
    "extract": "claude-haiku-5-5",
    "induce": "claude-opus-5-5",
    "onboard": "claude-sonnet-5-5",
}

# USD per million tokens (input, output), Anthropic list prices. An estimate for per-run
# cost tracking, not an invoice.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


@dataclass
class ModelResult:
    data: dict[str, Any]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    price_share: float = 1.0  # 0.5 for Message Batches

    @property
    def cost_usd(self) -> float:
        price_in, price_out = PRICES_PER_MTOK.get(self.model, (0.0, 0.0))
        full = (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000
        return full * self.price_share


class ModelRefused(RuntimeError):
    """The model declined the request (stop_reason == 'refusal')."""


class Backend(Protocol):
    def complete_json(
        self, model: str, system: str, prompt: str, schema: dict[str, Any], effort: str
    ) -> ModelResult: ...


class CachedBackend:
    """Wraps a backend so a question asked before is answered from a file, at no cost.

    A capped run that is re-run with a bigger cap then pays only for the new questions."""

    def __init__(self, inner: Backend, path: str | Path):
        self.inner = inner
        self.path = Path(path)
        self.hits = 0
        self._lock = threading.Lock()
        self._answers: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                self._answers[row["key"]] = row["data"]

    @staticmethod
    def key(model: str, system: str, prompt: str, effort: str) -> str:
        return hashlib.sha256("\x1f".join((model, system, prompt, effort)).encode()).hexdigest()

    def complete_json(self, model, system, prompt, schema, effort):
        k = self.key(model, system, prompt, effort)
        with self._lock:
            if k in self._answers:
                self.hits += 1
                return ModelResult(self._answers[k], model)
        result = self.inner.complete_json(model, system, prompt, schema, effort)
        self._save(k, result)
        return result

    @property
    def batches(self) -> bool:
        return getattr(self.inner, "batches", False)

    def complete_many(self, questions: list[Question]) -> list[ModelResult]:
        keys = [self.key(m, sy, p, e) for m, sy, p, _, e in questions]
        out: list[ModelResult | None] = [None] * len(questions)
        with self._lock:
            for i, k in enumerate(keys):
                if k in self._answers:
                    self.hits += 1
                    out[i] = ModelResult(self._answers[k], questions[i][0])
        todo = [i for i, r in enumerate(out) if r is None]
        for i, result in zip(
            todo, complete_many(self.inner, [questions[i] for i in todo]), strict=True
        ):
            if result.data:  # an empty answer (errored request) is asked again next time
                self._save(keys[i], result)
            out[i] = result
        return out  # type: ignore[return-value]

    def _save(self, k: str, result: ModelResult) -> None:
        with self._lock:
            self._answers[k] = result.data
            with self.path.open("a") as f:
                f.write(json.dumps({"key": k, "data": result.data}) + "\n")


@dataclass
class StubBackend:
    """Deterministic backend. ``responder`` maps (role-model, prompt, schema) to an answer.

    With no responder it returns the schema's ``default`` answer when one is given, so
    pipelines can run end to end with no model at all.
    """

    responder: Callable[[str, str, dict[str, Any]], dict[str, Any]] | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete_json(self, model, system, prompt, schema, effort):
        self.calls.append({"model": model, "system": system, "prompt": prompt})
        if self.responder:
            data = self.responder(model, prompt, schema)
        else:
            data = dict(schema.get("default", {}))
        return ModelResult(data=data, model=model, input_tokens=len(prompt) // 4)


Question = tuple[str, str, str, dict[str, Any], str]  # model, system, prompt, schema, effort


def complete_many(backend: Any, questions: list[Question]) -> list[ModelResult]:
    """Answer several questions, in one batch when the backend can."""
    if hasattr(backend, "complete_many"):
        return backend.complete_many(questions)
    return [backend.complete_json(*q) for q in questions]


class _MessagesBackend:
    """Shared request logic for backends that speak the Anthropic Messages API."""

    client: Any
    model_prefix = ""

    def params(self, model, system, prompt, schema, effort) -> dict[str, Any]:
        json_schema = {k: v for k, v in schema.items() if k != "default"}
        return {
            "model": f"{self.model_prefix}{model}",
            "max_tokens": 16000,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "effort": effort,
                "format": {"type": "json_schema", "schema": json_schema},
            },
        }

    @staticmethod
    def result(model: str, response: Any, price_share: float = 1.0) -> ModelResult:
        if response.stop_reason == "refusal":
            raise ModelRefused(f"{model} refused: {response.stop_details}")
        text = next(b.text for b in response.content if b.type == "text")
        return ModelResult(
            data=json.loads(text),
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            price_share=price_share,
        )

    def complete_json(self, model, system, prompt, schema, effort):
        response = self.client.messages.create(**self.params(model, system, prompt, schema, effort))
        return self.result(model, response)


class AnthropicBackend(_MessagesBackend):
    """Claude through the Anthropic API. Reads ``ANTHROPIC_API_KEY`` from the environment."""

    def __init__(self, client: Any = None):
        if client is None:
            from anthropic import Anthropic

            client = Anthropic()
        self.client = client


class AnthropicBatchBackend(AnthropicBackend):
    """Message Batches: grouped questions at half price, answered when the batch ends.

    A single question still goes through the Messages API at full price. An errored,
    expired or refused request comes back as an empty answer, which the agents treat as
    'no decision' and leave open."""

    batches = True
    PRICE_SHARE = 0.5

    def __init__(self, client: Any = None, poll_seconds: float = 20.0):
        super().__init__(client)
        self.poll_seconds = poll_seconds

    def complete_many(self, questions: list[Question]) -> list[ModelResult]:
        if not questions:
            return []
        batch = self.client.messages.batches.create(
            requests=[
                {"custom_id": f"q{i}", "params": self.params(*q)} for i, q in enumerate(questions)
            ]
        )
        while batch.processing_status != "ended":
            time.sleep(self.poll_seconds)
            batch = self.client.messages.batches.retrieve(batch.id)
        out = [ModelResult({}, q[0]) for q in questions]
        for row in self.client.messages.batches.results(batch.id):
            i = int(row.custom_id[1:])
            if row.result.type != "succeeded":
                continue
            try:
                out[i] = self.result(questions[i][0], row.result.message, self.PRICE_SHARE)
            except (ModelRefused, StopIteration, json.JSONDecodeError):
                pass
        return out


class BedrockBackend(_MessagesBackend):
    """Claude on Amazon Bedrock through the Anthropic SDK's Bedrock Mantle client."""

    model_prefix = "anthropic."

    def __init__(self, region: str, profile: str | None = None, client: Any = None):
        if client is None:
            from anthropic import AnthropicBedrockMantle

            client = AnthropicBedrockMantle(aws_region=region, aws_profile=profile)
        self.client = client


class ModelClient:
    """What agents call. Routes roles to models and logs every call to the action log."""

    def __init__(
        self,
        backend: Backend,
        log: ActionLog | None = None,
        role_models: dict[str, str] | None = None,
    ):
        self.backend = backend
        self.log = log or ActionLog()
        self.role_models = {**DEFAULT_ROLE_MODELS, **(role_models or {})}

    def complete_json(
        self,
        *,
        agent: str,
        role: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        effort: str = "medium",
    ) -> dict[str, Any]:
        model = self.role_models[role]
        result = self.backend.complete_json(model, system, prompt, schema, effort)
        self._log(agent, role, prompt, result)
        return result.data

    @property
    def batches(self) -> bool:
        return getattr(self.backend, "batches", False)

    def complete_many(
        self,
        *,
        agent: str,
        role: str,
        system: str,
        prompts: list[str],
        schema: dict[str, Any],
        effort: str = "medium",
    ) -> list[dict[str, Any]]:
        """Several questions for one role; one batch when the backend sends batches."""
        model = self.role_models[role]
        results = complete_many(self.backend, [(model, system, p, schema, effort) for p in prompts])
        for p, r in zip(prompts, results, strict=True):
            self._log(agent, role, p, r)
        return [r.data for r in results]

    def _log(self, agent: str, role: str, prompt: str, result: ModelResult) -> None:
        self.log.write(
            ActionRecord(
                agent=agent,
                action=f"model:{role}",
                inputs={"prompt_chars": len(prompt)},
                reasoning=result.data.get("reasoning") if isinstance(result.data, dict) else None,
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
            )
        )


def client_from_env(log: ActionLog | None = None) -> ModelClient:
    choice = os.environ.get("FPSO_MODEL_BACKEND", "stub")
    backend: Backend
    if choice == "anthropic":
        batch = os.environ.get("FPSO_MODEL_BATCH", "") not in ("", "0")
        backend = AnthropicBatchBackend() if batch else AnthropicBackend()
    elif choice == "bedrock":
        backend = BedrockBackend(
            region=os.environ.get("AWS_REGION", "us-east-1"),
            profile=os.environ.get("AWS_PROFILE"),
        )
    else:
        backend = StubBackend()
    # FPSO_MODEL_<ROLE> overrides one role's model, e.g. FPSO_MODEL_ADJUDICATE=claude-haiku-5-5.
    role_models = {
        role: os.environ[f"FPSO_MODEL_{role.upper()}"]
        for role in DEFAULT_ROLE_MODELS
        if os.environ.get(f"FPSO_MODEL_{role.upper()}")
    }
    return ModelClient(backend, log=log, role_models=role_models)


def main() -> None:
    """``fpso-model-check``: one tiny call per role to confirm the backend works."""
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
        "default": {"ok": True},
    }
    client = client_from_env()
    print(f"backend: {type(client.backend).__name__}")
    for role, model in client.role_models.items():
        out = client.complete_json(
            agent="model-check",
            role=role,
            system="Reply with the JSON requested.",
            prompt='Return {"ok": true}.',
            schema=schema,
            effort="low",
        )
        print(f"{role:<10} {model:<20} {out}")
    print(f"estimated cost: ${client.log.total_cost_usd:.4f}")
