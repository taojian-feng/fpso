-- World model (L2): links and derived state only. Source records stay in their
-- source systems; entity_ref is a pointer, never a copy.

CREATE TABLE IF NOT EXISTS entity_ref (
    id            BIGSERIAL PRIMARY KEY,
    source        TEXT NOT NULL,          -- connector name, e.g. 'erp-pm', 'p6'
    source_key    TEXT NOT NULL,          -- stable key inside that source
    entity_type   TEXT NOT NULL,          -- e.g. 'equipment', 'work_order', 'po_line'
    display       TEXT,                   -- short human label for reports
    extracted_at  TIMESTAMPTZ NOT NULL,
    UNIQUE (source, source_key)
);

CREATE TABLE IF NOT EXISTS pattern (
    id            BIGSERIAL PRIMARY KEY,
    link_type     TEXT NOT NULL,
    description   TEXT NOT NULL,          -- plain-language rule a reviewer confirmed
    rule          JSONB NOT NULL,         -- machine form used by candidate generation
    status        TEXT NOT NULL DEFAULT 'proposed'
                  CHECK (status IN ('proposed', 'confirmed', 'rejected')),
    confirmed_by  TEXT,                   -- reviewer id; simulated reviewers are prefixed 'sim:'
    confirmed_at  TIMESTAMPTZ,
    hits          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS link (
    id            BIGSERIAL PRIMARY KEY,
    from_ref      BIGINT NOT NULL REFERENCES entity_ref(id),
    to_ref        BIGINT NOT NULL REFERENCES entity_ref(id),
    link_type     TEXT NOT NULL,
    confidence    REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    method        TEXT NOT NULL,          -- 'rule', 'model', 'pattern', 'human'
    pattern_id    BIGINT REFERENCES pattern(id),
    evidence      JSONB NOT NULL DEFAULT '{}'::jsonb,
    status        TEXT NOT NULL DEFAULT 'proposed'
                  CHECK (status IN ('proposed', 'auto_accepted', 'confirmed', 'rejected')),
    decided_by    TEXT,
    decided_at    TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (from_ref, to_ref, link_type)
);

CREATE INDEX IF NOT EXISTS link_from_idx ON link (from_ref) WHERE status <> 'rejected';
CREATE INDEX IF NOT EXISTS link_to_idx ON link (to_ref) WHERE status <> 'rejected';

CREATE TABLE IF NOT EXISTS action_log (
    id            BIGSERIAL PRIMARY KEY,
    at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    agent         TEXT NOT NULL,
    action        TEXT NOT NULL,
    inputs        JSONB NOT NULL DEFAULT '{}'::jsonb,
    reasoning     TEXT,
    model         TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      NUMERIC(12, 6)
);

-- Walks accepted links outward from one entity, so a report can show the chain
-- Equipment -> Work order -> Material -> PO line -> Manifest line -> Sailing.
CREATE OR REPLACE FUNCTION chain_from(start_ref BIGINT, max_depth INTEGER DEFAULT 8)
RETURNS TABLE (depth INTEGER, ref_id BIGINT, via_link BIGINT, link_type TEXT, confidence REAL)
LANGUAGE sql STABLE AS $$
    WITH RECURSIVE walk(depth, ref_id, via_link, link_type, confidence, path) AS (
        SELECT 0, start_ref, NULL::BIGINT, NULL::TEXT, 1.0::REAL, ARRAY[start_ref]
        UNION ALL
        SELECT w.depth + 1,
               CASE WHEN l.from_ref = w.ref_id THEN l.to_ref ELSE l.from_ref END,
               l.id, l.link_type, l.confidence,
               w.path || CASE WHEN l.from_ref = w.ref_id THEN l.to_ref ELSE l.from_ref END
        FROM walk w
        JOIN link l ON (l.from_ref = w.ref_id OR l.to_ref = w.ref_id)
                   AND l.status IN ('auto_accepted', 'confirmed')
        WHERE w.depth < max_depth
          AND NOT (CASE WHEN l.from_ref = w.ref_id THEN l.to_ref ELSE l.from_ref END) = ANY (w.path)
    )
    SELECT depth, ref_id, via_link, link_type, confidence FROM walk;
$$;
