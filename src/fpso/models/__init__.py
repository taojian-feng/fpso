"""Model client used by every agent.

Agents ask for a JSON answer that matches a schema, by role ("adjudicate", "extract",
"induce"), not by model name. The backend decides which Claude model serves each role.
Three backends exist:

- ``StubBackend``: no network, for tests and for work before a model account is ready.
- ``AnthropicBackend``: Claude through the Anthropic API with an API key (Phase 0 default).
- ``BedrockBackend``: Claude on Amazon Bedrock in one region (kept for a later move).

Pick one with ``client_from_env()``: ``FPSO_MODEL_BACKEND=anthropic`` uses
``ANTHROPIC_API_KEY``; ``FPSO_MODEL_BACKEND=bedrock`` plus ``AWS_REGION`` (and optionally
``AWS_PROFILE``) selects Bedrock; anything else gives the stub.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from fpso.governance import ActionLog, ActionRecord

# Which model serves each role. Bulk adjudication uses the mid-size model, high-volume
# email reading the small one, and pattern induction the large one (plan WS5).
DEFAULT_ROLE_MODELS: dict[str, str] = {
    "adjudicate": "claude-sonnet-5-5",
    "extract": "claude-haiku-5-5",
    "induce": "claude-opus-5-5",
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

    @property
    def cost_usd(self) -> float:
        price_in, price_out = PRICES_PER_MTOK.get(self.model, (0.0, 0.0))
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000


class ModelRefused(RuntimeError):
    """The model declined the request (stop_reason == 'refusal')."""


class Backend(Protocol):
    def complete_json(
        self, model: str, system: str, prompt: str, schema: dict[str, Any], effort: str
    ) -> ModelResult: ...


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


class _MessagesBackend:
    """Shared request logic for backends that speak the Anthropic Messages API."""

    client: Any
    model_prefix = ""

    def complete_json(self, model, system, prompt, schema, effort):
        json_schema = {k: v for k, v in schema.items() if k != "default"}
        response = self.client.messages.create(
            model=f"{self.model_prefix}{model}",
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": json_schema},
            },
        )
        if response.stop_reason == "refusal":
            raise ModelRefused(f"{model} refused: {response.stop_details}")
        text = next(b.text for b in response.content if b.type == "text")
        return ModelResult(
            data=json.loads(text),
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class AnthropicBackend(_MessagesBackend):
    """Claude through the Anthropic API. Reads ``ANTHROPIC_API_KEY`` from the environment."""

    def __init__(self, client: Any = None):
        if client is None:
            from anthropic import Anthropic

            client = Anthropic()
        self.client = client


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
        return result.data


def client_from_env(log: ActionLog | None = None) -> ModelClient:
    choice = os.environ.get("FPSO_MODEL_BACKEND", "stub")
    backend: Backend
    if choice == "anthropic":
        backend = AnthropicBackend()
    elif choice == "bedrock":
        backend = BedrockBackend(
            region=os.environ.get("AWS_REGION", "us-east-1"),
            profile=os.environ.get("AWS_PROFILE"),
        )
    else:
        backend = StubBackend()
    return ModelClient(backend, log=log)


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
