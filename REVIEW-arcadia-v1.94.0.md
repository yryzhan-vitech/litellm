# Review — `arcadia-v1.94.0` @ `bca9dabfa5`, before the prod cutover

Reviewer: Brendan Smith-Elion · 2026-08-03
Scope: the 33-commit stack on `arcadia-v1.94.0`, CCT PR
[#2382](https://github.com/arcadia/cloud-config-templates/pull/2382), and a comparison
against the patch set running on prd-ai today (`soak-prod-plus-main`, image
`noma-v2-complete-fix`, chart `1.81.12-stable`).

This is the second pass, re-validating the two blockers raised on the first pass.

## Verdict

**Both blockers are closed.** Two code defects found, both fixed in this branch. Three
further items need a decision but are not code changes. Nothing here blocks the ACM on its
own; items 3 and 4 below should be answered before the rollout starts.

---

## Blockers from the first pass — both closed

### ARC-BUG-16, advisor gate — closed

`resolve_advisor_gate_provider()` exists at
`litellm/llms/anthropic/experimental_pass_through/messages/interceptors/advisor.py:352`
and gates on `get_model_list`, not `get_available_deployment`.

That divergence from the production version is correct and worth keeping.
`get_available_deployment` is the selection API: it load-balances, so the gate decision
would differ per request on an alias fronting two providers, and under usage-based routing
it performs synchronous Redis reads inside the request coroutine. The gate needs a
provider, not a pick.

Resolving the **executor** leg as well as the advisor leg is likewise correct and is not
scope creep. Without it, the gate fix would trade a leaked `tool_use` for a hard 400 on the
same request class, because the Bedrock transform builds its URL from whatever model string
it is handed and would default the region.

The routing-param **allowlist** (`aws_region_name`, `vertex_location`, `vertex_project`,
`api_version`) is safe against the config in CCT #2382: every Bedrock deployment there
carries only `aws_region_name`, and the Anthropic ones use `api_key: os.environ/...`, so
the env fallback covers the dropped key. It would break on a deployment using
`aws_role_name` / `aws_profile_name` for cross-account assume-role, or a literal key not
present in the environment — see item 4.

### SRE-3691 / ARC-BUG-20, non-blocking monitor-mode `during_call` — closed

`_is_monitor_only_guardrail` requires `monitor_mode=True` **and** `block_failures=False`
(`litellm/proxy/utils.py:1821`), and the `noma-during-call` guardrail in CCT #2382 sets
exactly that pair, so it detaches. Noma is off the critical path.

Excluding MCP from the detach is the right call: `during_mcp_call` is the only hook that
sees post-rewrite tool arguments, and that matters more once ARIA MCP moves behind the
proxy. ARC-BUG-43's deadline bounds the four hooks that stay awaited, and the two-layer
enforcement is the right shape — a bare float would have replaced the 5s connect budget
with the 10s scan budget and *lengthened* the unreachable-Noma tail.

**For the ACM record:** at the pending-task cap (1024, `utils.py:1932`) monitor tasks are
**dropped**, not queued — the coroutine is closed and a throttled warning is logged. During
a sustained Noma outage at prod RPS that cap is reached quickly and scans are skipped. That
is the right trade (availability over audit), but it should be a documented one rather than
a discovered one.

---

## 1. Fixed here — the advisor tool type was matched by prefix on one leg, exactly on another

ARC-BUG-45 established that Anthropic versions server tools by dated suffix, and introduced
`_ANTHROPIC_ADVISOR_TOOL_PREFIX` so the adapter keeps a future `advisor_20260302` on the
Anthropic-native path. It introduced it **privately**, in `adapters/transformation.py`, so
the interceptor kept comparing against the dated constant at four sites: `can_handle`
(`:52`), the `handle()` lookup (`:73`), the executor-tools rewrite (`:121`) and the gate
(`:386`).

The day the vendor ships the next version, the adapter keeps the tool native while the gate
declines to orchestrate it — a raw `tool_use` reaching the client, which is precisely the
ARC-BUG-44 symptom on a calendar trigger instead of a config one.

**Fix:** one predicate, `is_advisor_tool()`, alongside the constant in
`litellm/types/llms/anthropic.py`; the adapter re-exports the stem rather than deriving its
own, so the two legs cannot disagree. Not a regression from the ARC-BUG-44/45 work — a gap
it left behind.

## 2. Fixed here — ARC-BUG-46's 400 contract stopped one call short

`handle()` validates `model` and `max_uses` as `BadRequestError`, then delegates the advisor
tool's `api_base`/`api_key` to `_resolve_advisor_credentials`, which raised bare
`ValueError` at three caller-input paths: `api_base` without `api_key`, non-https
`api_base`, and TLS verification disabled.

A bare `ValueError` carries no `status_code`, so the proxy's
`getattr(e, "status_code", 500)` reports malformed caller input as a server fault and raises
a High-severity `llm_exceptions` alert — the exact defect ARC-BUG-46 fixed for the two
sibling fields.

Currently **unreachable**: `general_settings.allow_client_side_credentials` is unset in CCT
#2382, so the opt-in guard returns first. That is why the original validation could find
five malformed shapes all returning 400 and no sixth. It becomes reachable, silently, the
day that setting is turned on.

**Fix:** all three raise `BadRequestError`; `model` and `custom_llm_provider` are threaded
in as optional arguments so SDK callers are unaffected. Three existing tests asserting the
old `ValueError` contract are updated.

## 3. Decide — the advisor legs bypass the router, so they get no fallbacks, retries or cooldowns

`_call_messages_handler` goes straight to the provider. CCT #2382 configures
`router_settings.fallbacks` (`claude-sonnet-5 → claude-sonnet-5-anthropic`),
`num_retries: 3` and `cooldown_time: 30` — none of which apply to either advisor leg. A
Bedrock throttle that a normal request survives via fallback becomes a hard error on an
advisor request.

Pre-existing on both branches, so not an upgrade regression — but far more reachable now
that the gate actually fires. Wants a ticket and a line in the ACM risk section.

## 4. Verify before cutover — the advisor model alias must exist in the target's `model_list`

This is the highest-risk pre-cutover check, because it fails differently per environment.

When an advisor alias is absent from `model_list`, `_resolve_advisor_model_via_router`
falls back to the bare alias at **warning** level, and `get_llm_provider("claude-opus-5[1m]")`
then raises `BadRequestError: LLM Provider NOT provided` — the sub-call fails outright
rather than misrouting.

ARIA defaults **both** the executor and the advisor to `claude-opus-5[1m]`
(`aria_bot/shared/config.py:67` and `:98`, `ClaudeModel.OPUS_5_1M`), stamped into
`~/.claude/settings.json` by `aria_bot/shared/bootstrap.py:242`.

Against that:

- CCT #2382's **chart default** `model_list` has no Opus at all — `claude-sonnet-5`,
  `claude-sonnet-4-6`, `claude-haiku-4-5` (+ `-anthropic` fallbacks) and `titan-embed-v2`.
- The newest **dev-ai overlay** I can read locally
  (`cloud-config-infrastructures`, `preprd/dev-ai/litellm-internal/application.yaml`,
  last touched 2026-06-27) defines 28 aliases, none of them `claude-opus-5`,
  `claude-opus-5[1m]`, `claude-sonnet-5` or `claude-sonnet-5[1m]`. Its own comment still
  reads *"claude-opus-4-8[1m] is the ARIA advisor model"*.

That checkout is roughly five weeks stale and I could not reach the deployed ref, so this
needs confirming rather than acting on. But it is the single check most likely to turn an
ARIA validation into a false negative — and I have no visibility at all into the prd-ai
overlay, which is the one that matters for the rollout.

**Confirm on both dev-ai and prd-ai:** `claude-opus-5[1m]`, `claude-opus-5`,
`claude-sonnet-5[1m]`, `claude-sonnet-5`. Same pass covers item 1's allowlist question —
grep the overlays for `aws_role_name`, `aws_profile_name` and any literal (non-`os.environ/`)
`api_key`.

## 5. Verify on dev-ai — the new `callbacks: [prometheus]` chart default

CCT commit `ac540156` adds `prometheus` to `litellm_settings.callbacks`. It is the right
fix for the dead `litellm_guardrail_*` families, but it puts `PrometheusLogger` in a second
registry alongside the existing `success_callback` / `failure_callback` entries.

Worth one before/after counter check that the per-request families
(`litellm_requests_metric`, spend, tokens) are not double-counted. If they are, every
LiteLLM cost and usage dashboard doubles on cutover — which would look like a traffic
change, not a config bug.

---

## Checked against prod and cleared — do not re-litigate

Patches on `soak-prod-plus-main` with no counterpart on `arcadia-v1.94.0`, each confirmed
either absorbed upstream or genuinely obsolete:

| Prod commit | Patch | Status on `arcadia-v1.94.0` |
|---|---|---|
| `cc27b67615` | `/health/readiness` + `/metrics` into `public_routes` (SRE-3692) | **Obsolete.** `/health/readiness` is declared with no `Depends(user_api_key_auth)` — an intentional public probe. `/metrics` goes through `PrometheusAuthMiddleware`, whose `require_auth_for_metrics_endpoint` default is `True` on prod and v1.94.0 alike. No change either way. |
| `01198de641` | SRE-3598 dict usage cost tracking | **Absorbed.** Replaced by v1.94.0's `_get_web_search_requests` rework. |
| `be1d385711` | dead `level` reassignment crashing the effort-capability check | **Absorbed.** The dead line is gone upstream. |
| `bea93125fb` | drop `copy.deepcopy`, it breaks on uvloop/aiohttp | **Preserved.** `data.copy()` at `guardrail_translation/handler.py:266`; no deepcopy in the file. |
| `0d38ccfb37`, `f524b6f28d` | preserve Anthropic server-tools through the pre-call guardrail | **Present and generalised** via `_is_anthropic_native_tool`, covering `ANTHROPIC_HOSTED_TOOLS` plus the advisor stem. |
| `0084a25e1b` | Opus 4.7/4.8 adaptive-thinking on bedrock-invoke | **Not applicable to the chart default** (no Opus deployments). Re-check against the CCI overlays — see item 4. |

Two observations from that sweep, both **pre-existing on prod** and therefore not upgrade
risks, but each worth its own ticket:

- The `/health/readiness` payload is now `{"status", "db"}` unless
  `allow_public_health_readiness_details` is set; the detailed form moved to the
  authenticated `/health/readiness/details`. Anything parsing the legacy payload breaks
  silently.
- `/health/readiness` returns 503 when the DB is unreachable, while
  `general_settings.allow_requests_on_db_unavailable: true` says keep serving. An Aurora
  blip therefore deregisters every ALB target and fails the startupProbe, on a config
  explicitly designed to survive exactly that.

---

## Testing

`aria-bot/local/devai_upgrade_gate.py` covers models, streaming (ARC-BUG-03/05), cache
(ARC-BUG-04) and the advisor gate (ARC-BUG-16). It predates ARC-BUG-44/45/46 and needs
three checks added before it can gate this cutover:

1. advisor tool + a function tool + a hosted tool in one request, through the pre-call
   guardrail (ARC-BUG-44/45)
2. the malformed-advisor 400 contract, including the credential shapes from item 2
3. no `tool_use` leak on the streaming path

Two standing limits: dev-ai runs `store_model_in_db: true` against prod's `false`, so the
DB overlay path is not what prod does; and nothing runnable from dev validates the prd-ai
`model_list`.

## Verification of the changes in this PR

- `tests/.../messages/test_advisor_review_findings.py` — 18 new tests, all passing.
- Full `tests/test_litellm/llms/anthropic/` run is at **parity with the pristine branch**:
  the same 15 pre-existing failures before and after, all of them the proxy's optional
  dependencies (`fastapi_sso`) missing from the local environment rather than anything in
  this diff. Baselined by stashing the changes and re-running.
- `ruff format` clean; `ruff check --select F,E9` clean on every changed file.
- The new tests install the router via `sys.modules` rather than
  `patch("litellm.proxy.proxy_server.llm_router")`, so they run without the proxy extras —
  unlike the existing `test_advisor_gate_provider.py`, which cannot collect without them.
