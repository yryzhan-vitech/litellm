-- Covering index for the /global/spend admin aggregation (global_spend_per_team).
-- IF NOT EXISTS so this is a no-op on large existing prod deployments where the
-- index is created out-of-band via CREATE INDEX CONCURRENTLY (see PR description).
-- On fresh/small databases it builds quickly inside the normal migrate-deploy path.
CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_startTime_team_id_spend_idx"
  ON "LiteLLM_SpendLogs" ("startTime", "team_id", "spend");
