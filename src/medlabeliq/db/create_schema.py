from __future__ import annotations

from medlabeliq.db.connection import get_connection


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS label_document (
    label_id UUID PRIMARY KEY,
    concept_name TEXT NOT NULL,
    set_id TEXT NOT NULL UNIQUE,
    document_id_root TEXT,
    document_code TEXT,
    document_code_display_name TEXT,
    document_title TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS label_version (
    label_version_id UUID PRIMARY KEY,
    label_id UUID NOT NULL REFERENCES label_document(label_id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    effective_time TEXT,
    source_xml_path TEXT NOT NULL,
    xml_sha256 TEXT,
    is_locked_version BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(label_id, version_number)
);

CREATE TABLE IF NOT EXISTS label_product (
    product_id UUID PRIMARY KEY,
    label_version_id UUID NOT NULL REFERENCES label_version(label_version_id) ON DELETE CASCADE,
    product_index INTEGER NOT NULL,
    product_name TEXT,
    product_code TEXT,
    product_code_system TEXT,
    dosage_form_code TEXT,
    dosage_form_display_name TEXT,
    route_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    active_moieties JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(label_version_id, product_index)
);

CREATE TABLE IF NOT EXISTS product_ingredient (
    ingredient_id UUID PRIMARY KEY,
    product_id UUID NOT NULL REFERENCES label_product(product_id) ON DELETE CASCADE,
    ingredient_index INTEGER NOT NULL,
    class_code TEXT,
    ingredient_name TEXT,
    unii_code TEXT,
    strength_numerator_value TEXT,
    strength_numerator_unit TEXT,
    strength_denominator_value TEXT,
    strength_denominator_unit TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(product_id, ingredient_index)
);

CREATE TABLE IF NOT EXISTS label_section (
    section_id UUID PRIMARY KEY,
    label_version_id UUID NOT NULL REFERENCES label_version(label_version_id) ON DELETE CASCADE,

    source_section_uid TEXT NOT NULL,
    parent_section_id UUID REFERENCES label_section(section_id) ON DELETE SET NULL,
    parent_source_section_uid TEXT,

    order_index INTEGER NOT NULL,
    depth INTEGER NOT NULL,

    loinc_code TEXT,
    code_display_name TEXT,
    raw_title TEXT,
    normalized_title TEXT,

    canonical_section_name TEXT,
    mapping_method TEXT NOT NULL,

    nearest_canonical_section_name TEXT,
    nearest_canonical_source_section_uid TEXT,

    retrieval_family TEXT,
    retrieval_family_source_section_uid TEXT,

    is_unclassified BOOLEAN NOT NULL,
    direct_text TEXT,
    direct_text_sha256 TEXT,
    direct_char_count INTEGER NOT NULL,
    child_count INTEGER NOT NULL,
    heading_path JSONB NOT NULL DEFAULT '[]'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(label_version_id, source_section_uid)
);

CREATE INDEX IF NOT EXISTS idx_label_document_concept_name
ON label_document(concept_name);

CREATE INDEX IF NOT EXISTS idx_label_document_set_id
ON label_document(set_id);

CREATE INDEX IF NOT EXISTS idx_label_version_label_id
ON label_version(label_id);

CREATE INDEX IF NOT EXISTS idx_label_product_label_version_id
ON label_product(label_version_id);

CREATE INDEX IF NOT EXISTS idx_product_ingredient_product_id
ON product_ingredient(product_id);

CREATE INDEX IF NOT EXISTS idx_label_section_label_version_id
ON label_section(label_version_id);

CREATE INDEX IF NOT EXISTS idx_label_section_parent_section_id
ON label_section(parent_section_id);

CREATE INDEX IF NOT EXISTS idx_label_section_canonical_section_name
ON label_section(canonical_section_name);

CREATE INDEX IF NOT EXISTS idx_label_section_retrieval_family
ON label_section(retrieval_family);

CREATE INDEX IF NOT EXISTS idx_label_section_loinc_code
ON label_section(loinc_code);

CREATE INDEX IF NOT EXISTS idx_label_section_direct_text_fts
ON label_section USING GIN (to_tsvector('english', COALESCE(direct_text, '')));
"""


def main() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)

    print("Database schema created successfully.")


if __name__ == "__main__":
    main()