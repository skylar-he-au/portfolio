-- Step 5b: settled spending per provider per Sydney day (the day the question started).
-- No personal data; kept for the monthly totals.
CREATE TABLE spend_daily (
    day date NOT NULL,
    provider text NOT NULL CHECK (provider IN ('anthropic', 'google', 'openai')),
    spent_usd numeric(12, 6) NOT NULL DEFAULT 0,
    PRIMARY KEY (day, provider)
);

-- Money held by questions still running: one row per question and provider. A row past
-- expires_at (the process died before settling) is booked as spent at its full amount.
CREATE TABLE spend_reservations (
    id uuid NOT NULL,
    provider text NOT NULL CHECK (provider IN ('anthropic', 'google', 'openai')),
    day date NOT NULL,
    usd numeric(12, 6) NOT NULL,
    expires_at timestamptz NOT NULL,
    PRIMARY KEY (id, provider)
);
CREATE INDEX spend_reservations_provider_day_idx ON spend_reservations (provider, day);

-- A provider told us its own limit was reached: the whole AI rests until `until`.
CREATE TABLE ai_pause (
    provider text PRIMARY KEY CHECK (provider IN ('anthropic', 'google', 'openai')),
    until timestamptz NOT NULL,
    reason text NOT NULL
);
