-- Step 8: all patrol asks and matches share one site-wide daily counter.
ALTER TABLE usage_counters DROP CONSTRAINT usage_counters_kind_check;
ALTER TABLE usage_counters ADD CONSTRAINT usage_counters_kind_check
    CHECK (kind IN ('ask', 'match', 'fetch', 'patrol'));
