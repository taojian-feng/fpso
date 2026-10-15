import os
from datetime import UTC, datetime

import pytest

from fpso.world_model import apply_schema

URL = os.environ.get("FPSO_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set FPSO_TEST_DATABASE_URL to run")


@pytest.fixture
def conn():
    import psycopg

    with psycopg.connect(URL) as c:
        with c.cursor() as cur:
            cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        c.commit()
        apply_schema(c)
        apply_schema(c)  # safe to re-run
        yield c


def _ref(cur, source, key, etype):
    cur.execute(
        "INSERT INTO entity_ref (source, source_key, entity_type, extracted_at) "
        "VALUES (%s, %s, %s, %s) RETURNING id",
        (source, key, etype, datetime.now(UTC)),
    )
    return cur.fetchone()[0]


def test_chain_walk_follows_accepted_links_only(conn):
    with conn.cursor() as cur:
        eq = _ref(cur, "erp-pm", "10004001", "equipment")
        act = _ref(cur, "p6", "KST-TA-1010", "activity")
        wo = _ref(cur, "erp-pm", "41000101", "work_order")
        po = _ref(cur, "erp-mm", "4500210001-00010", "po_line")
        for a, b, t, status in [
            (eq, act, "equipment_activity", "confirmed"),
            (wo, act, "work_order_activity", "auto_accepted"),
            (wo, po, "work_order_po_line", "proposed"),
        ]:
            cur.execute(
                "INSERT INTO link (from_ref, to_ref, link_type, confidence, method, status) "
                "VALUES (%s, %s, %s, 0.9, 'rule', %s)",
                (a, b, t, status),
            )
        cur.execute("SELECT ref_id FROM chain_from(%s) ORDER BY depth", (eq,))
        rows = [r[0] for r in cur.fetchall()]
    assert rows == [eq, act, wo]  # the proposed link is not walked


def test_confidence_must_be_a_probability(conn):
    import psycopg

    with conn.cursor() as cur:
        a = _ref(cur, "x", "1", "t")
        b = _ref(cur, "x", "2", "t")
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO link (from_ref, to_ref, link_type, confidence, method) "
                "VALUES (%s, %s, 'x', 1.5, 'rule')",
                (a, b),
            )
