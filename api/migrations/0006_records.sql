-- Step 5c: the 30-day question log (fields fixed by SPEC; no answer, JD, token or IP).
CREATE TABLE question_log (
    id uuid PRIMARY KEY,  -- the turn_id sent in `meta`
    created_at timestamptz NOT NULL DEFAULT now(),
    question text NOT NULL,
    answer_language text,
    prompt_version text NOT NULL,
    providers text[] NOT NULL DEFAULT '{}',
    cost_usd numeric(10, 6) NOT NULL DEFAULT 0,
    ip_hash bytea NOT NULL,
    thumbs_down boolean NOT NULL DEFAULT false
);
CREATE INDEX question_log_created_at_idx ON question_log (created_at);

-- Step 5c: the "one question at a time per conversation" lock, shared by all instances.
ALTER TABLE conversations
    ADD COLUMN running_owner uuid,
    ADD COLUMN running_until timestamptz;
