-- Step 6b2: counters for fetching job ads by link, and the 30-day job-match log.

-- kind 'fetch': fetching a job ad from Greenhouse or Lever (counted even when it fails).
ALTER TABLE usage_counters DROP CONSTRAINT usage_counters_kind_check;
ALTER TABLE usage_counters ADD CONSTRAINT usage_counters_kind_check
    CHECK (kind IN ('ask', 'match', 'tts', 'fetch'));

-- One row per job match that got past the daily count and the spending reservation
-- (fields fixed by SPEC). No job ad, requirements, result, token or IP.
CREATE TABLE match_log (
    id uuid PRIMARY KEY,  -- the match_id sent in `result`
    created_at timestamptz NOT NULL DEFAULT now(),
    source text NOT NULL CHECK (source IN ('greenhouse', 'lever', 'text')),
    job_language text,
    prompt_version text NOT NULL,
    providers text[] NOT NULL DEFAULT '{}',
    cost_usd numeric(10, 6) NOT NULL DEFAULT 0,
    ip_hash bytea NOT NULL,
    thumbs_down boolean NOT NULL DEFAULT false
);
CREATE INDEX match_log_created_at_idx ON match_log (created_at);
