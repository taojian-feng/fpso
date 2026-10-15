"""World model store (L2): entity pointers, links, confirmed patterns and the action log."""

from importlib.resources import files


def schema_sql() -> str:
    return files(__package__).joinpath("schema.sql").read_text()


def apply_schema(conn) -> None:
    """Create or update the schema on an open psycopg connection. Safe to run repeatedly."""
    with conn.cursor() as cur:
        cur.execute(schema_sql())
    conn.commit()
