# +-------------------------------------------------------------+
#
#      Noma Security V2 Guardrail Integration for LiteLLM
#
# +-------------------------------------------------------------+

import asyncio
import contextvars
import enum
import json
import math
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Type, cast
from urllib.parse import urlparse

import httpx
import openai

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS,
    NOMA_CONNECTION_RETRY_CEILING_SECONDS,
    NOMA_CONNECTION_RETRY_MAX,
    NOMA_MAX_SCAN_TIMEOUT_SECONDS,
    NOMA_MIN_SCAN_TIMEOUT_SECONDS,
    NOMA_SCAN_TIMEOUT_SECONDS,
)
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy.guardrails.guardrail_hooks.noma.noma import NomaBlockedMessage
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs, GuardrailStatus

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


# [ARC-BUG-43] Every type a scan can time out as. Three independent hierarchies with no
# common base below Exception, so they must be enumerated:
#   asyncio.TimeoutError     what asyncio.wait_for raises (is builtins.TimeoutError on 3.11+)
#   httpx.TimeoutException   the transport's own read/connect/pool deadlines
#   openai.APITimeoutError   base of litellm.Timeout, which AsyncHTTPHandler.post re-raises
#                            after catching an httpx timeout
# The third is the one that actually reaches us in production: the shared handler converts
# httpx timeouts at seven sites, and because connect is pinned below scan_timeout it fires
# before wait_for for an unreachable Noma. Omitting it left `timed_out` false and the
# DEADLINE EXCEEDED marker silent on 46 real events over 7 days.
#
# The superclass is listed rather than litellm.Timeout itself: litellm.Timeout subclasses
# APITimeoutError, so naming only the subclass would classify it while leaving the parent —
# raised directly elsewhere in the SDK — unrecognised.
_TIMEOUT_EXCEPTIONS: tuple = (
    asyncio.TimeoutError,
    httpx.TimeoutException,
    openai.APITimeoutError,
)

# Fraction of the budget an elapsed time must reach before a timeout-typed exception counts
# as a real deadline expiry. Needed because the type alone does not distinguish an expiry
# from a connection failure the HTTP layer merely labels as a timeout — see the
# classification comment in apply_guardrail. Below 1.0 because a scan cancelled a hair
# early still spent the budget; generous enough that a late-firing ceiling (an event-loop
# stall delays asyncio.wait_for) is not misfiled as a fast failure.
_DEADLINE_ELAPSED_TOLERANCE: float = 0.9

# [ARC-BUG-47] Fraction of the budget below which a timeout-typed failure is read as a broken
# connection rather than an expired deadline, and therefore retried once.
#
# aiohttp raises SocketTimeoutError when a connection breaks WHILE READING, not only when a read
# budget expires, and litellm's transport maps that to httpx.ReadTimeout
# (aiohttp_transport.py: `aiohttp.SocketTimeoutError: httpx.ReadTimeout`). So a pooled keep-alive
# the peer has already closed — Noma sits behind Cloudflare, which recycles idle connections
# sooner than the client pool expires them — surfaces as a timeout that arrives in single-digit
# milliseconds against a 10s budget. Measured on prd-ai: 74 such failures in ~25 minutes,
# elapsed p50 0.036s, max 0.093s, i.e. ~24% of scans never reached the vendor while every
# counter recorded success.
#
# Deliberately an order of magnitude below _DEADLINE_ELAPSED_TOLERANCE rather than adjacent to
# it, leaving the 0.05–0.9 band as neither-retried-nor-counted-as-expiry. A scan that genuinely
# spent a tenth of its budget before failing is doing real work against a real peer; retrying
# that risks doubling latency on a Noma that is merely slow, which is the failure mode
# ARC-BUG-20 and 43 exist to bound. The gap is the safety margin, not an oversight.
_CONNECTION_FAILURE_ELAPSED_FRACTION: float = 0.05

# Why one retry rather than a loop, and where the knobs live: the scan POST is idempotent, so a
# second attempt is safe, and a stale pooled connection is cured by the fresh one the first
# failure already forced. If a newly-established connection also fails this fast, reuse is not
# the problem. NOMA_CONNECTION_RETRY_MAX and NOMA_CONNECTION_RETRY_CEILING_SECONDS live in
# constants.py and read the environment; NOMA_CONNECTION_RETRY_MAX=0 restores the pre-fix
# behaviour whole — retry and classifier both.
#
# ⚠️ NOT hot-reloadable, and an earlier version of this comment overstated it. get_env_int runs at
# module import and this module binds the VALUE, not the module, so changing the variable needs a
# deployment env edit plus a rolling restart of every proxy pod — the same blast radius as an
# image revert, minus only the rebuild. It is also not plumbed today: neither dev-ai nor prd-ai
# declares any NOMA_* env on the proxy deployment (Noma credentials arrive via envFrom secretRef),
# so a first use means authoring an untested overlay change under pressure. Plumb it at its
# default BEFORE it is needed.

# [ARC-BUG-47] Per-request storage for the last scan attempt's own duration, read by the
# ARC-BUG-43 classifier in apply_guardrail so a retried fast failure is not relabelled
# DEADLINE EXCEEDED.
#
# 🔴 A ContextVar, NOT an instance attribute, and the difference is a real bug that a review
# caught. initialize_guardrail_v2 registers ONE NomaV2Guardrail per config as a global callback,
# so every concurrent request shares the instance. The write (in _call_noma_scan) and the read (in
# apply_guardrail's except arm) are separated by an await boundary plus real work — masker
# traversal, OTEL attributes, audit-record build — so a concurrent fast scan landing in that
# window overwrote the value. Reproduced: a genuine 1.05s expiry against a 1.0s budget classified
# on 0.006s and reported timed_out=False, which is precisely the inversion this mechanism exists
# to prevent. With block_failures=False the exception is swallowed, so the misclassification is
# invisible by construction.
#
# asyncio propagates context per task, so each request gets its own value with no cleanup needed.
_LAST_SCAN_ATTEMPT_ELAPSED: contextvars.ContextVar = contextvars.ContextVar(
    "noma_last_scan_attempt_elapsed", default=None
)

_DEFAULT_API_BASE = "https://api.noma.security/"
_AIDR_SCAN_ENDPOINT = "/litellm/guardrail"
_INTERVENED_INPUT_FIELDS = ("texts", "images", "tools", "tool_calls")
_DEFAULT_API_BASE_HOSTNAME = urlparse(_DEFAULT_API_BASE).hostname


class _Action(str, enum.Enum):
    BLOCKED = "BLOCKED"
    NONE = "NONE"
    GUARDRAIL_INTERVENED = "GUARDRAIL_INTERVENED"


def _coerce_end_of_stream_only(value: object) -> bool:
    # [ARC-BUG-14] config values arrive uncoerced (extra="allow"); bare "false" would be truthy
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _coerce_sampling_rate(value: object) -> int:
    # [ARC-BUG-14] unified_guardrail does chunk_counter % rate; 0 crashes mid-stream
    if value is None or value == "":
        return 5
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"streaming_sampling_rate must be an integer >= 1, got {value!r}")
    try:
        rate = int(float(value))
    except ValueError as e:
        raise ValueError(f"streaming_sampling_rate must be an integer >= 1, got {value!r}") from e
    if rate != float(value) or rate < 1:
        raise ValueError(f"streaming_sampling_rate must be an integer >= 1, got {value!r}")
    return rate


def _coerce_scan_timeout(value: object) -> float:
    # [ARC-BUG-43] same uncoerced-config problem as the streaming knobs above.
    # The None branch validates NOMA_SCAN_TIMEOUT_SECONDS through this same path
    # rather than trusting it: get_env_float only rejects non-finite values, so a
    # stray NOMA_SCAN_TIMEOUT_SECONDS=-5 or =0 would otherwise reach httpx and fail
    # every scan in ~0ms. With block_failures=False that is swallowed, so the proxy
    # would serve all traffic with zero Noma coverage while still reporting five
    # guardrails enabled — a security control silently at 0%.
    if value is None or value == "":
        value = NOMA_SCAN_TIMEOUT_SECONDS
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"scan_timeout must be a positive, finite number of seconds, got {value!r}")
    try:
        timeout = float(value)
    except ValueError as e:
        raise ValueError(f"scan_timeout must be a positive, finite number of seconds, got {value!r}") from e
    # inf passes `> 0` and nan fails every comparison, so both slip past a bare
    # `<= 0` guard: inf makes httpx raise OverflowError, nan behaves erratically.
    if not math.isfinite(timeout) or timeout < NOMA_MIN_SCAN_TIMEOUT_SECONDS or timeout > NOMA_MAX_SCAN_TIMEOUT_SECONDS:
        raise ValueError(
            f"scan_timeout must be a finite number of seconds between "
            f"{NOMA_MIN_SCAN_TIMEOUT_SECONDS} and {NOMA_MAX_SCAN_TIMEOUT_SECONDS}, got {value!r}"
        )
    return timeout


class NomaV2Guardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        application_id: Optional[str] = None,
        monitor_mode: Optional[bool] = None,
        block_failures: Optional[bool] = None,
        streaming_end_of_stream_only: bool | None = None,
        streaming_sampling_rate: int | None = None,
        scan_timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)

        self.api_key = api_key or os.environ.get("NOMA_API_KEY")
        self.api_base = (api_base or os.environ.get("NOMA_API_BASE") or _DEFAULT_API_BASE).rstrip("/")
        self.application_id = application_id or os.environ.get("NOMA_APPLICATION_ID")
        if monitor_mode is None:
            self.monitor_mode = os.environ.get("NOMA_MONITOR_MODE", "false").lower() == "true"
        else:
            self.monitor_mode = monitor_mode

        if block_failures is None:
            self.block_failures = os.environ.get("NOMA_BLOCK_FAILURES", "true").lower() == "true"
        else:
            self.block_failures = block_failures

        if self._requires_api_key(api_base=self.api_base) and not self.api_key:
            raise ValueError("Noma v2 guardrail requires api_key when using Noma SaaS endpoint")

        # [ARC-BUG-14] UnifiedLLMGuardrails reads these off the instance via getattr
        self.streaming_end_of_stream_only: bool = _coerce_end_of_stream_only(streaming_end_of_stream_only)
        self.streaming_sampling_rate: int = _coerce_sampling_rate(streaming_sampling_rate)

        # [ARC-BUG-43] bound a single scan; see NOMA_SCAN_TIMEOUT_SECONDS
        self.scan_timeout: float = _coerce_scan_timeout(scan_timeout)

        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))

        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.noma import (
            NomaV2GuardrailConfigModel,
        )

        return NomaV2GuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> List[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.pre_mcp_call,
            GuardrailEventHooks.during_mcp_call,
        ]

    def _get_authorization_header(self) -> str:
        if not self.api_key:
            return ""
        return f"Bearer {self.api_key}"

    @staticmethod
    def _requires_api_key(api_base: str) -> bool:
        parsed = urlparse(api_base)
        return parsed.hostname == _DEFAULT_API_BASE_HOSTNAME

    @staticmethod
    def _get_non_empty_str(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped or None

    def _resolve_action_from_response(
        self,
        response_json: dict,
    ) -> _Action:
        action = response_json.get("action")
        if isinstance(action, str):
            try:
                return _Action(action)
            except ValueError:
                pass

        raise ValueError("Noma v2 response missing valid action")

    def _build_scan_payload(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"],
        application_id: Optional[str],
    ) -> dict:
        payload_request_data = self._sanitize_payload_for_transport(request_data)
        if logging_obj is not None:
            # [ARC-BUG-01] snapshot model_call_details: the async logging handler inserts keys
            # into it on another thread, and it is assigned AFTER the json.dumps round-trip
            # above, so the serializer downstream would iterate a live dict and raise
            # RuntimeError: dictionary changed size during iteration
            model_call_details = getattr(logging_obj, "model_call_details", None)
            if isinstance(model_call_details, dict):
                model_call_details = dict(model_call_details)
            payload_request_data["litellm_logging_obj"] = model_call_details

        payload: dict[str, Any] = {
            "inputs": inputs,
            "request_data": payload_request_data,
            "input_type": input_type,
            "monitor_mode": self.monitor_mode,
        }
        if application_id:
            payload["application_id"] = application_id
        return payload

    @staticmethod
    def _sanitize_payload_for_transport(payload: dict) -> dict:
        def _default(obj: Any) -> Any:
            if hasattr(obj, "model_dump"):
                try:
                    return obj.model_dump()
                except Exception:
                    pass
            return str(obj)

        try:
            json_str = json.dumps(payload, default=_default)
        except (ValueError, TypeError):
            json_str = safe_dumps(payload)

        safe_payload = safe_json_loads(json_str, default={})
        if safe_payload == {} and payload:
            verbose_proxy_logger.warning(
                "Noma v2 guardrail: payload serialization failed, falling back to empty payload"
            )

        if isinstance(safe_payload, dict):
            return safe_payload

        verbose_proxy_logger.warning(
            "Noma v2 guardrail: payload sanitization produced non-dict output (type=%s), falling back to empty payload",
            type(safe_payload).__name__,
        )
        return {}

    async def _call_noma_scan(
        self,
        payload: dict,
    ) -> dict:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        authorization_header = self._get_authorization_header()
        if authorization_header:
            headers["Authorization"] = authorization_header

        endpoint = f"{self.api_base}{_AIDR_SCAN_ENDPOINT}"
        sanitized_payload = self._sanitize_payload_for_transport(payload)
        # [ARC-BUG-43] Two layers, because neither alone bounds the scan:
        #
        # The httpx/aiohttp timeout is PER-I/O-OPERATION, not total. Its read
        # budget resets on every byte received, so a peer that dribbles the
        # response slower than it completes but faster than the budget keeps the
        # scan alive indefinitely — measured at 4x the configured value against a
        # drip server, and reported as a SUCCESS. asyncio.wait_for is what makes
        # scan_timeout an actual wall-clock ceiling.
        #
        # A bare float would also overwrite all four httpx budgets, silently
        # replacing the 5s connect handshake with scan_timeout. For an unreachable
        # Noma that LENGTHENS the tail rather than shortening it, so connect is
        # pinned to the shared client's value.
        request_timeout = httpx.Timeout(
            timeout=self.scan_timeout,
            connect=min(HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS, self.scan_timeout),
        )
        # [ARC-BUG-47] Retry once when a timeout-typed failure arrives far too early to be a real
        # deadline: that shape is a broken pooled connection, and the transport's exception map
        # disguises it as a timeout. See _CONNECTION_FAILURE_ELAPSED_FRACTION.
        #
        # The retry deliberately shares the ORIGINAL budget rather than restarting it. A fresh
        # scan_timeout per attempt would let two attempts spend 2x the ceiling that ARC-BUG-43
        # exists to enforce, which is the opposite of the intent — the whole point of that bug is
        # that scan_timeout is a wall-clock guarantee to the caller, not a per-attempt allowance.
        deadline_budget = self.scan_timeout
        attempt = 0
        while True:
            # The retry gets its OWN ceiling, not the whole remainder. Inheriting the remainder
            # would turn a 36ms failure into a request blocked for the full scan_timeout — see
            # NOMA_CONNECTION_RETRY_CEILING_SECONDS. Still bounded by the shared budget so the
            # two attempts together cannot exceed ARC-BUG-43's wall-clock guarantee.
            attempt_budget = (
                deadline_budget if attempt == 0 else min(deadline_budget, NOMA_CONNECTION_RETRY_CEILING_SECONDS)
            )
            attempt_started = time.monotonic()
            # Reset per attempt. Gated on the retry being enabled so NOMA_CONNECTION_RETRY_MAX=0
            # restores the pre-ARC-BUG-47 classifier exactly, not just the retry — a review found
            # the kill switch left this half active.
            if NOMA_CONNECTION_RETRY_MAX > 0:
                _LAST_SCAN_ATTEMPT_ELAPSED.set(None)
            try:
                response = await asyncio.wait_for(
                    self.async_handler.post(
                        url=endpoint,
                        headers=headers,
                        json=sanitized_payload,
                        timeout=request_timeout,
                    ),
                    timeout=attempt_budget,
                )
                break
            except Exception as e:
                attempt_elapsed = time.monotonic() - attempt_started
                deadline_budget -= attempt_elapsed
                # [ARC-BUG-47] Hand the LAST attempt's own duration to the ARC-BUG-43
                # classifier. Without it that classifier sums both attempts and can relabel a
                # fast connection failure as DEADLINE EXCEEDED — inverting the very marker
                # ARC-BUG-43 exists to make trustworthy. Reproduced at scan_timeout=6.0:
                # attempt 1 at 0.25s plus attempt 2 at 5.2s reported "DEADLINE EXCEEDED after
                # 5.452s" although neither attempt came near the budget.
                if NOMA_CONNECTION_RETRY_MAX > 0:
                    _LAST_SCAN_ATTEMPT_ELAPSED.set(attempt_elapsed)
                looks_like_connection_failure = (
                    isinstance(e, _TIMEOUT_EXCEPTIONS)
                    and attempt_elapsed < self.scan_timeout * _CONNECTION_FAILURE_ELAPSED_FRACTION
                )
                # A retry that cannot complete inside what is left would burn the caller's
                # deadline to return the same failure.
                #
                # Compared against the RETRY's own ceiling, not the connect budget. Comparing
                # against min(connect, scan_timeout) made the whole fix unreachable for any
                # scan_timeout <= 5.0: min() collapsed to scan_timeout, so the test became
                # (scan_timeout - elapsed) > scan_timeout, false for every positive elapsed.
                # Measured cliff: 0.1-5.0 never retried, 5.01+ retried. Two independent reviews
                # found it, and initialize_guardrail_v2 CLAMPS a generic timeout into
                # [0.1, 60.0] rather than rejecting it — so an operator tightening a guardrail
                # to 3s would have silently shipped a pod that reports the fix as active while
                # ~24% of its scans still pass uninspected.
                budget_left_for_a_retry = deadline_budget > min(
                    NOMA_CONNECTION_RETRY_CEILING_SECONDS, self.scan_timeout * _CONNECTION_FAILURE_ELAPSED_FRACTION
                )
                if (
                    not looks_like_connection_failure
                    or attempt >= NOMA_CONNECTION_RETRY_MAX
                    or not budget_left_for_a_retry
                ):
                    raise
                attempt += 1
                # hook=%s to match both sibling ERROR lines: four hooks are default_on and only
                # pre_call sits on the request path, so an operator needs to know which one is
                # retrying to tell whether users are feeling it. Read from the payload rather
                # than self, because _call_noma_scan does not receive input_type and self carries
                # the configured event_hook, which is not the same thing as the leg being scanned.
                verbose_proxy_logger.warning(
                    "Noma v2 scan failed after %.3fs with %s, far below the %ss budget — reading "
                    "this as a broken pooled connection and retrying once on a fresh one. "
                    "guardrail=%s hook=%s remaining_budget=%.3fs retry_ceiling=%ss",
                    attempt_elapsed,
                    type(e).__name__,
                    self.scan_timeout,
                    self.guardrail_name,
                    payload.get("input_type", "unknown"),
                    deadline_budget,
                    NOMA_CONNECTION_RETRY_CEILING_SECONDS,
                )
        verbose_proxy_logger.debug(
            "Noma v2 AIDR response: status_code=%s body=%s",
            response.status_code,
            response.text,
        )
        response.raise_for_status()
        response_json = response.json()
        verbose_proxy_logger.debug(
            "Noma v2 AIDR response parsed: %s",
            json.dumps(response_json, default=str),
        )
        return response_json

    def _add_guardrail_observability(
        self,
        request_data: dict,
        start_time: datetime,
        guardrail_status: GuardrailStatus,
        guardrail_json_response: Any,
    ) -> None:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_provider="noma_v2",
            guardrail_json_response=guardrail_json_response,
            request_data=request_data,
            guardrail_status=guardrail_status,
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            duration=duration,
        )

    def _apply_action(
        self,
        inputs: GenericGuardrailAPIInputs,
        response_json: dict,
        action: _Action,
    ) -> GenericGuardrailAPIInputs:
        if action == _Action.BLOCKED:
            raise NomaBlockedMessage(response_json)

        if action == _Action.GUARDRAIL_INTERVENED:
            updated_inputs = cast(GenericGuardrailAPIInputs, dict(inputs))
            for field in _INTERVENED_INPUT_FIELDS:
                value = response_json.get(field)
                if isinstance(value, list):
                    updated_inputs[field] = value  # type: ignore[literal-required]
            return updated_inputs

        return inputs

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        start_time = datetime.now()
        guardrail_status: GuardrailStatus = "success"
        guardrail_json_response: Any = {}
        dynamic_params = self.get_guardrail_dynamic_request_body_params(request_data)
        if not isinstance(dynamic_params, dict):
            dynamic_params = {}
        response_json: Optional[dict] = None

        # Per-request dynamic params can override configured application context.
        application_id = self._get_non_empty_str(dynamic_params.get("application_id"))

        if application_id is None:
            application_id = self._get_non_empty_str(self.application_id)

        # Fall back to API key alias for per-key traceability in Noma dashboard
        # (ports v1 fallback from PR #16832).
        if application_id is None:
            application_id = self._get_non_empty_str(
                request_data.get("litellm_metadata", {}).get("user_api_key_alias")
            ) or self._get_non_empty_str(request_data.get("metadata", {}).get("user_api_key_alias"))

        try:
            payload = self._build_scan_payload(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                logging_obj=logging_obj,
                application_id=application_id,
            )

            response_json = await self._call_noma_scan(payload=payload)
            if self.monitor_mode:
                action = _Action.NONE
            else:
                action = self._resolve_action_from_response(response_json=response_json)
            guardrail_json_response = response_json
            verbose_proxy_logger.debug(
                "Noma v2 guardrail decision: input_type=%s action=%s",
                input_type,
                action.value,
            )
            processed_inputs = self._apply_action(
                inputs=inputs,
                response_json=response_json,
                action=action,
            )

            guardrail_status = "success" if action == _Action.NONE else "guardrail_intervened"
            return processed_inputs

        except NomaBlockedMessage as e:
            guardrail_status = "guardrail_intervened"
            guardrail_json_response = (
                response_json if isinstance(response_json, dict) else getattr(e, "detail", {"error": "blocked"})
            )
            raise
        except asyncio.CancelledError:
            # [ARC-BUG-43] CancelledError is a BaseException, so without this arm it
            # skips `except Exception` and the `finally` records the initial "success" —
            # a scan killed 0.15s into a 30s hang would be audited, and OTEL-spanned, as
            # having passed. ARC-BUG-20's shutdown drain cancels in-flight monitor tasks
            # on every restart, so this is reached routinely rather than in theory.
            guardrail_status = "guardrail_failed_to_respond"
            guardrail_json_response = {
                "error": "CancelledError",
                "detail": "scan cancelled before completion (shutdown drain or client disconnect)",
                "timed_out": False,
                "scan_timeout_seconds": self.scan_timeout,
            }
            raise
        except Exception as e:
            guardrail_status = "guardrail_failed_to_respond"
            elapsed = (datetime.now() - start_time).total_seconds()
            # [ARC-BUG-47] Classify on the last ATTEMPT's duration, not the sum across a retry.
            # `elapsed` still reports total wall-clock in the log and audit record, which is
            # what a caller waited; only the expiry test uses the per-attempt figure, because
            # "did this scan burn its budget" is a question about one attempt.
            attempt_elapsed_for_classification = _LAST_SCAN_ATTEMPT_ELAPSED.get()
            classify_on = (
                attempt_elapsed_for_classification if attempt_elapsed_for_classification is not None else elapsed
            )
            # [ARC-BUG-43] Classify on MEASURED time, not on the exception type alone.
            #
            # The type is necessary but not sufficient. The shared HTTP handler re-raises
            # litellm.Timeout for anything it considers a timeout, and that includes a dead
            # pooled keep-alive connection which fails in about a millisecond — its own
            # message says so: "Connection timed out ... time taken=0.001 seconds". Keying
            # only on the type marked 74 such events in two hours as deadline expiries, the
            # longest of which took 0.12s against a 10s budget. That is the opposite of
            # useful: it buries the real signal in noise that looks identical.
            #
            # A genuine expiry lands at or just past the budget, so require the elapsed
            # time to be in that neighbourhood. The tolerance is generous because the
            # wall-clock ceiling can fire late if the event loop was blocked (measured: a
            # 7.4s stall from a lazy synchronous import delayed a 2s deadline to 9.4s).
            timeout_type = isinstance(e, _TIMEOUT_EXCEPTIONS)
            timed_out = timeout_type and classify_on >= self.scan_timeout * _DEADLINE_ELAPSED_TOLERANCE
            # [ARC-BUG-43] A dict rather than a bare str(e), for structure: it carries the
            # timed_out flag and the budget that expired, and it keeps an explanation for
            # exceptions whose str() is empty (httpx ConnectTimeout stringifies to '').
            #
            # NOT for sanitization. The payload masker is key-name driven, so a secret in a
            # value under `detail` is no more masked here than it was as a bare string —
            # wrapping it in a dict buys traversal, not redaction. This entry is expected to
            # contain request content: it lands in the spend log and the S3 LLM logs, which
            # are the PHI-bearing audit store by design. What must NOT carry it is the log
            # line below, which goes to CloudWatch.
            guardrail_json_response = {
                "error": type(e).__name__,
                "detail": str(e) or type(e).__name__,
                "timed_out": timed_out,
                "elapsed_seconds": round(elapsed, 4),
                "scan_timeout_seconds": self.scan_timeout,
            }
            if timed_out:
                # Logged at ERROR with a distinct, greppable marker because with
                # block_failures=False this is otherwise INVISIBLE: the exception is
                # swallowed, so _run_guardrail_with_metrics records status="success"
                # and litellm_guardrail_errors_total (gated on status == "error") never
                # increments. Content went unscanned and no metric says so.
                #
                # Both numbers, and the measured one first: an earlier version printed only
                # the configured budget in the slot that reads as elapsed time, so every
                # line claimed "after 10.0s" including the ones that failed in 2ms.
                verbose_proxy_logger.error(
                    "Noma v2 scan DEADLINE EXCEEDED after %.3fs (budget %ss, %s) — content was "
                    "NOT scanned, guardrail=%s hook=%s",
                    elapsed,
                    self.scan_timeout,
                    type(e).__name__,
                    self.guardrail_name,
                    input_type,
                )
            elif timeout_type:
                # A timeout-typed exception that did NOT reach the budget: almost always a
                # connection failure the HTTP layer labels as a timeout (dead pooled
                # keep-alive, refused connect). Still unscanned content, so still ERROR, but
                # it is a connectivity problem rather than a slow Noma and must not be
                # aggregated with real expiries.
                verbose_proxy_logger.error(
                    "Noma v2 scan FAILED FAST after %.3fs (%s, budget %ss) — content was NOT "
                    "scanned; connection-level failure, not a deadline. guardrail=%s hook=%s",
                    elapsed,
                    type(e).__name__,
                    self.scan_timeout,
                    self.guardrail_name,
                    input_type,
                )
            else:
                # The exception TYPE only, deliberately. str(e) here is inherited from the
                # base and can carry request content: a ValueError echoing the prompt, an
                # HTTPStatusError carrying the tenant URL, or a message that happens to
                # include an Authorization header. This line goes to the container's stdout
                # and therefore to CloudWatch, which must stay free of PHI. The full text is
                # kept in the audit entry above, which lands in the spend log and the S3 LLM
                # logs — the store that is intended to hold PHI, replicated to the security
                # account under its own retention. Same information, correct destination.
                verbose_proxy_logger.error(
                    "Noma v2 guardrail failed: %s — guardrail=%s hook=%s (detail in the audit record)",
                    type(e).__name__,
                    self.guardrail_name,
                    input_type,
                )
            if self.block_failures:
                raise
            return inputs
        finally:
            self._add_guardrail_observability(
                request_data=request_data,
                start_time=start_time,
                guardrail_status=guardrail_status,
                guardrail_json_response=guardrail_json_response,
            )
