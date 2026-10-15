import json
from types import SimpleNamespace

import pytest

from fpso.governance import ActionLog
from fpso.models import BedrockBackend, ModelClient, ModelRefused, StubBackend, client_from_env

SCHEMA = {
    "type": "object",
    "properties": {"match": {"type": "string"}, "confidence": {"type": "number"}},
    "required": ["match", "confidence"],
    "additionalProperties": False,
    "default": {"match": "unsure", "confidence": 0.0},
}


def test_stub_returns_schema_default_and_logs(tmp_path):
    log = ActionLog(tmp_path / "actions.jsonl")
    client = ModelClient(StubBackend(), log=log)
    out = client.complete_json(
        agent="resolver", role="adjudicate", system="s", prompt="p" * 40, schema=SCHEMA
    )
    assert out == {"match": "unsure", "confidence": 0.0}
    assert log.records[0].model == "claude-sonnet-5-5"
    assert json.loads((tmp_path / "actions.jsonl").read_text())["agent"] == "resolver"


def test_roles_route_to_models():
    stub = StubBackend()
    client = ModelClient(stub)
    for role in ("adjudicate", "extract", "induce"):
        client.complete_json(agent="a", role=role, system="", prompt="", schema=SCHEMA)
    assert [c["model"] for c in stub.calls] == [
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "claude-opus-5-5",
    ]


class FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.stop_reason = stop_reason
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            stop_details=None,
            content=[SimpleNamespace(type="text", text='{"match": "yes", "confidence": 0.9}')],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
        )


def test_bedrock_backend_request_shape_and_cost():
    fake = FakeMessages()
    backend = BedrockBackend(region="us-east-1", client=SimpleNamespace(messages=fake))
    client = ModelClient(backend)
    out = client.complete_json(
        agent="a", role="adjudicate", system="sys", prompt="p", schema=SCHEMA, effort="low"
    )
    assert out == {"match": "yes", "confidence": 0.9}
    assert fake.kwargs["model"] == "anthropic.claude-sonnet-5-5"
    fmt = fake.kwargs["output_config"]["format"]
    assert fmt["type"] == "json_schema" and "default" not in fmt["schema"]
    assert fake.kwargs["output_config"]["effort"] == "low"
    assert client.log.total_cost_usd == pytest.approx((1000 * 2 + 100 * 10) / 1e6)


def test_bedrock_refusal_raises():
    backend = BedrockBackend(
        region="us-east-1", client=SimpleNamespace(messages=FakeMessages("refusal"))
    )
    with pytest.raises(ModelRefused):
        backend.complete_json("claude-sonnet-5-5", "", "", SCHEMA, "low")


def test_env_defaults_to_stub(monkeypatch):
    monkeypatch.delenv("FPSO_MODEL_BACKEND", raising=False)
    assert isinstance(client_from_env().backend, StubBackend)
