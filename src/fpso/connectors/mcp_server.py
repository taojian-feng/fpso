"""Expose any connector as a read-only MCP server.

    fpso-mcp erp-pm --extracts data/kestrel-A/extracts

Every tool is annotated read-only. Phase 0 has no write tools at all.
"""

from __future__ import annotations

import argparse
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from fpso.connectors.base import Connector
from fpso.connectors.sources import CONNECTORS

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)


def build_server(connector: Connector) -> MCPServer:
    server = MCPServer(
        name=f"fpso-{connector.name}",
        instructions=f"Read-only access to {connector.description}",
    )

    def _check(entity_type: str) -> None:
        if entity_type not in connector.entity_types():
            raise ToolError(
                f"unknown entity type {entity_type!r}; one of {connector.entity_types()}"
            )

    @server.tool(annotations=READ_ONLY)
    def list_entity_types() -> dict[str, int]:
        """Entity types this source holds, with record counts."""
        return {t: connector.count(t) for t in connector.entity_types()}

    @server.tool(annotations=READ_ONLY)
    def describe(entity_type: str, sample_size: int = 3) -> dict[str, Any]:
        """Field names and a few sample records for one entity type."""
        _check(entity_type)
        return {
            "entity_type": entity_type,
            "count": connector.count(entity_type),
            "fields": connector.schema(entity_type),
            "sample": [r.as_dict() for r in connector.sample(entity_type, min(sample_size, 20))],
        }

    @server.tool(annotations=READ_ONLY)
    def get(entity_type: str, key: str) -> dict[str, Any] | None:
        """One record by its key, or null when there is none."""
        _check(entity_type)
        r = connector.get(entity_type, key)
        return r.as_dict() if r else None

    @server.tool(annotations=READ_ONLY)
    def search(entity_type: str, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Records containing every term of the query (case-insensitive), up to limit."""
        _check(entity_type)
        return [r.as_dict() for r in connector.search(entity_type, query, min(limit, 200))]

    return server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run a read-only connector as an MCP server (stdio).")
    ap.add_argument("connector", choices=sorted(CONNECTORS))
    ap.add_argument("--extracts", required=True, help="path to an extracts/ directory")
    args = ap.parse_args(argv)
    build_server(CONNECTORS[args.connector](args.extracts)).run("stdio")


if __name__ == "__main__":
    main()
