"""Contract tests: every connector, over any extract (mock or real), must pass these."""

import asyncio
import inspect

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from fpso.connectors import CONNECTORS, Connector, open_all
from fpso.connectors.mcp_server import build_server
from fpso.synthetic.export import export
from fpso.synthetic.truth import build_world

EXPECTED_TYPES = {
    "erp-pm": {"equipment", "work_order", "component"},
    "erp-mm": {"material", "vendor", "po", "po_line", "stock", "material_price"},
    "p6": {"activity"},
    "vessel": {"sailing", "manifest_line", "sailing_capacity"},
    "supplier-mail": {"message"},
    "helicopter": {"flight"},
    "marine-ops": {"crane_day"},
    "warehouse": {"store_balance"},
    "weather": {"observation", "forecast"},
    "historian": {"vibration"},
}


@pytest.fixture(scope="module", params=["A", "B"])
def connectors(request, tmp_path_factory):
    d = tmp_path_factory.mktemp(f"run{request.param}")
    export(build_world(seed=5, scale=0.05), d, request.param)
    return open_all(d / "extracts")


@pytest.mark.parametrize("name", sorted(CONNECTORS))
def test_contract(connectors, name):
    c = connectors[name]
    assert set(c.entity_types()) == EXPECTED_TYPES[name]
    for t in c.entity_types():
        assert c.count(t) > 0
        assert c.schema(t)
        for r in c.iter(t):
            assert r.source == name and r.entity_type == t and r.key
            assert r.extracted_at
            assert c.get(t, r.key) is r
        first = c.sample(t, 1)[0]
        assert first in c.search(t, first.key, limit=200)
    assert c.get(c.entity_types()[0], "no-such-key") is None


def test_connectors_are_read_only():
    allowed = {"load", "entity_types", "count", "schema", "iter", "get", "sample", "search"}
    for cls in CONNECTORS.values():
        public = {
            n for n, _ in inspect.getmembers(cls, inspect.isfunction) if not n.startswith("_")
        }
        assert public <= allowed, f"{cls.name} exposes {public - allowed}"
        assert issubclass(cls, Connector)


def _payload(result):
    """Structured tool output; the SDK wraps non-object return types under 'result'."""
    sc = result.structured_content
    return sc["result"] if set(sc) == {"result"} else sc


def test_mcp_server_tools_are_read_only_and_work(connectors):
    server = build_server(connectors["erp-pm"])

    async def go():
        tools = await server.list_tools()
        assert {t.name for t in tools} == {"list_entity_types", "describe", "get", "search"}
        assert all(t.annotations and t.annotations.read_only_hint for t in tools)
        counts = _payload(await server.call_tool("list_entity_types", {}))
        assert counts["work_order"] > 0
        wo = connectors["erp-pm"].sample("work_order", 1)[0]
        got = await server.call_tool("get", {"entity_type": "work_order", "key": wo.key})
        assert _payload(got)["fields"] == wo.fields
        found = _payload(
            await server.call_tool("search", {"entity_type": "work_order", "query": wo.key})
        )
        assert wo.key in [r["key"] for r in found]
        with pytest.raises(ToolError):
            await server.call_tool("get", {"entity_type": "nope", "key": "1"})

    asyncio.run(go())
