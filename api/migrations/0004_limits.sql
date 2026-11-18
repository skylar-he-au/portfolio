-- Step 5a: per-day request counters and the daily key for IP hashes.
-- scope: 'token' (key = sha256 of the session token id), 'ip' (key = daily IP hash),
-- 'site' (key = '\x'::bytea). day is the Australia/Sydney calendar day.
CREATE TABLE usage_counters (
    day date NOT NULL,
    kind text NOT NULL CHECK (kind IN ('ask', 'match', 'tts')),
    scope text NOT NULL CHECK (scope IN ('token', 'ip', 'site')),
    key bytea NOT NULL,
    count integer NOT NULL DEFAULT 0,
    PRIMARY KEY (day, kind, scope, key)
);

-- One random 32-byte key per Sydney day; rows before today are deleted by the cleanup.
CREATE TABLE daily_keys (
    day date PRIMARY KEY,
    key bytea NOT NULL
);
