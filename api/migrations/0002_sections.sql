CREATE TABLE section_versions (
    version bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE content_state (
    id boolean PRIMARY KEY DEFAULT true CHECK (id),
    current_version bigint NOT NULL REFERENCES section_versions (version)
);

CREATE TABLE sections (
    version bigint NOT NULL REFERENCES section_versions (version),
    page text NOT NULL,
    section_id text NOT NULL,
    heading text NOT NULL,
    text text NOT NULL,
    position int NOT NULL,
    embedding vector(1536) NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', heading || ' ' || text)) STORED,
    PRIMARY KEY (version, page, section_id)
);

CREATE INDEX sections_tsv_idx ON sections USING gin (tsv);
