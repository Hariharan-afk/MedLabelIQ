from __future__ import annotations

from medlabeliq.db.connection import get_connection


DDL = """
CREATE TABLE IF NOT EXISTS qa_request_log (
    request_log_id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    query_text TEXT NOT NULL,
    drug_filter TEXT,
    family_filter TEXT,
    top_k INTEGER,

    include_evidence BOOLEAN NOT NULL,
    include_diagnostics BOOLEAN NOT NULL,

    final_status TEXT NOT NULL
        CHECK (final_status IN ('answered', 'insufficient_evidence')),

    final_answer TEXT NOT NULL,
    final_citations JSONB NOT NULL DEFAULT '[]'::jsonb,
    final_evidence_summary TEXT NOT NULL,
    safety_note TEXT NOT NULL,

    proposed_status TEXT
        CHECK (
            proposed_status IS NULL
            OR proposed_status IN ('answered', 'insufficient_evidence')
        ),

    verification_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    verification_verdict TEXT
        CHECK (
            verification_verdict IS NULL
            OR verification_verdict IN ('supported', 'insufficient', 'refuted')
        ),
    verification_rationale TEXT,
    verification_evidence_used JSONB NOT NULL DEFAULT '[]'::jsonb,
    verification_overrode_answer BOOLEAN NOT NULL DEFAULT FALSE,

    guardrail_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    guardrail_reason TEXT,

    evidence_count INTEGER NOT NULL DEFAULT 0,
    api_latency_ms DOUBLE PRECISION NOT NULL
);

ALTER TABLE qa_request_log
    ADD COLUMN IF NOT EXISTS requested_drug_filter TEXT;

ALTER TABLE qa_request_log
    ADD COLUMN IF NOT EXISTS resolved_drug_filter TEXT;

ALTER TABLE qa_request_log
    ADD COLUMN IF NOT EXISTS drug_resolution_status TEXT;
    
CREATE INDEX IF NOT EXISTS idx_qa_request_log_created_at
    ON qa_request_log (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_qa_request_log_final_status
    ON qa_request_log (final_status);

CREATE INDEX IF NOT EXISTS idx_qa_request_log_guardrail_triggered
    ON qa_request_log (guardrail_triggered);

CREATE INDEX IF NOT EXISTS idx_qa_request_log_verification_verdict
    ON qa_request_log (verification_verdict);


CREATE TABLE IF NOT EXISTS qa_evidence_log (
    request_log_id UUID NOT NULL
        REFERENCES qa_request_log(request_log_id)
        ON DELETE CASCADE,

    evidence_id TEXT NOT NULL,
    evidence_position INTEGER NOT NULL,

    chunk_id UUID NOT NULL,
    section_id UUID NOT NULL,

    concept_name TEXT NOT NULL,
    retrieval_family TEXT NOT NULL,
    canonical_section_name TEXT,
    nearest_canonical_section_name TEXT,

    heading TEXT NOT NULL,
    set_id TEXT NOT NULL,
    version_number INTEGER NOT NULL,

    hybrid_score DOUBLE PRECISION NOT NULL,
    lexical_rank INTEGER,
    dense_rank INTEGER,

    was_cited BOOLEAN NOT NULL DEFAULT FALSE,

    PRIMARY KEY (request_log_id, evidence_id)
);

CREATE INDEX IF NOT EXISTS idx_qa_evidence_log_request_log_id
    ON qa_evidence_log (request_log_id);

CREATE INDEX IF NOT EXISTS idx_qa_evidence_log_concept_name
    ON qa_evidence_log (concept_name);

CREATE INDEX IF NOT EXISTS idx_qa_evidence_log_retrieval_family
    ON qa_evidence_log (retrieval_family);

CREATE INDEX IF NOT EXISTS idx_qa_evidence_log_was_cited
    ON qa_evidence_log (was_cited);
"""


def main() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)

        conn.commit()

    print("Observability schema created successfully.")


if __name__ == "__main__":
    main()