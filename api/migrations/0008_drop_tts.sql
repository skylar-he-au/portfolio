-- Step 7a: speech moved out of this project; drop the unused 'tts' counter kind.
DELETE FROM usage_counters WHERE kind = 'tts';

ALTER TABLE usage_counters DROP CONSTRAINT usage_counters_kind_check;
ALTER TABLE usage_counters ADD CONSTRAINT usage_counters_kind_check
    CHECK (kind IN ('ask', 'match', 'fetch'));
