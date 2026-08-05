import datetime as real_datetime
import os
import smtplib
import sys

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import ProxyErrorTypes
from litellm.proxy.utils import ProxyLogging
from litellm.types.utils import CallTypes

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from unittest.mock import MagicMock, patch

from litellm.proxy.utils import get_custom_url, join_paths


def test_get_custom_url(monkeypatch):
    monkeypatch.setenv("SERVER_ROOT_PATH", "/litellm")
    custom_url = get_custom_url(request_base_url="http://0.0.0.0:4000", route="ui/")
    assert custom_url == "http://0.0.0.0:4000/litellm/ui/"


def test_proxy_only_error_true_for_llm_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert proxy_logging_obj._is_proxy_only_llm_api_error(
        original_exception=Exception(),
        error_type=ProxyErrorTypes.auth_error,
        route="/v1/chat/completions",
    )


def test_proxy_only_error_true_for_info_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=ProxyErrorTypes.auth_error,
            route="/key/info",
        )
        is True
    )


def test_proxy_only_error_false_for_non_llm_non_info_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=ProxyErrorTypes.auth_error,
            route="/key/generate",
        )
        is False
    )


def test_proxy_only_error_false_for_other_error_type():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=None,
            route="/v1/chat/completions",
        )
        is False
    )


@pytest.mark.asyncio
async def test_proxy_only_error_log_marks_no_upstream_llm_call():
    """A proxy-gate error (auth/rate-limit) synthesizes a ``Logging`` object and
    fires ``pre_call`` so the failure is logged — but it must tag the object with
    ``LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL`` so tracing callbacks don't fabricate
    an LLM-call span for a request that never reached a provider (root cause of the
    misplaced gen-AI span on auth failure)."""
    from litellm.constants import LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL
    from litellm.proxy._types import UserAPIKeyAuth

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    captured = {}

    def fake_pre_call(self, *args, **kwargs):
        captured["flag"] = self.model_call_details.get(
            LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL
        )

    from litellm.litellm_core_utils.litellm_logging import Logging

    orig_pre_call = Logging.pre_call
    orig_async_failure = Logging.async_failure_handler
    Logging.pre_call = fake_pre_call

    async def _noop_async_failure(self, *args, **kwargs):
        return None

    Logging.async_failure_handler = _noop_async_failure
    try:
        await proxy_logging_obj._handle_logging_proxy_only_error(
            request_data={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
            },
            user_api_key_dict=UserAPIKeyAuth(
                api_key="sk-bad", request_route="/v1/chat/completions"
            ),
            route="/v1/chat/completions",
            original_exception=Exception("bad key"),
        )
    finally:
        Logging.pre_call = orig_pre_call
        Logging.async_failure_handler = orig_async_failure

    assert captured.get("flag") is True


@pytest.mark.asyncio
async def test_proxy_only_error_log_keeps_litellm_metadata_in_litellm_params():
    """Responses API requests carry guardrail info under ``litellm_metadata``
    (not ``metadata``). It must land in litellm_params so
    ``merge_litellm_metadata`` can surface ``guardrail_information`` in the
    spend-log failure row, matching the chat completions path."""
    from litellm.proxy._types import UserAPIKeyAuth

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    captured = {}
    guardrail_info = [{"guardrail_name": "test-guard", "guardrail_status": "blocked"}]

    def fake_update_environment_variables(self, *args, **kwargs):
        captured["litellm_params"] = kwargs.get("litellm_params")
        captured["optional_params"] = kwargs.get("optional_params")

    from litellm.litellm_core_utils.litellm_logging import Logging

    orig_update_env = Logging.update_environment_variables
    orig_pre_call = Logging.pre_call
    orig_async_failure = Logging.async_failure_handler

    async def _noop_async_failure(self, *args, **kwargs):
        return None

    Logging.update_environment_variables = fake_update_environment_variables
    Logging.pre_call = lambda self, *args, **kwargs: None
    Logging.async_failure_handler = _noop_async_failure
    try:
        await proxy_logging_obj._handle_logging_proxy_only_error(
            request_data={
                "model": "gpt-4o",
                "input": "blocked prompt",
                "litellm_metadata": {
                    "standard_logging_guardrail_information": guardrail_info
                },
            },
            user_api_key_dict=UserAPIKeyAuth(
                api_key="sk-1234", request_route="/v1/responses"
            ),
            route="/v1/responses",
            original_exception=HTTPException(status_code=400, detail="blocked"),
        )
    finally:
        Logging.update_environment_variables = orig_update_env
        Logging.pre_call = orig_pre_call
        Logging.async_failure_handler = orig_async_failure

    assert (
        captured["litellm_params"]["litellm_metadata"][
            "standard_logging_guardrail_information"
        ]
        == guardrail_info
    )
    assert "litellm_metadata" not in captured["optional_params"]


def test_get_model_group_info_order():
    from litellm import Router
    from litellm.proxy.proxy_server import _get_model_group_info

    router = Router(
        model_list=[
            {
                "model_name": "openai/tts-1",
                "litellm_params": {
                    "model": "openai/tts-1",
                    "api_key": "sk-1234",
                },
            },
            {
                "model_name": "openai/gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": "sk-1234",
                },
            },
        ]
    )
    model_list = _get_model_group_info(
        llm_router=router,
        all_models_str=["openai/tts-1", "openai/gpt-3.5-turbo"],
        model_group=None,
    )

    model_groups = [m.model_group for m in model_list]
    assert model_groups == ["openai/tts-1", "openai/gpt-3.5-turbo"]


def test_join_paths_no_duplication():
    """Test that join_paths doesn't duplicate route when base_path already ends with it"""
    result = join_paths(
        base_path="http://0.0.0.0:4000/my-custom-path/", route="/my-custom-path"
    )
    assert result == "http://0.0.0.0:4000/my-custom-path"


def test_join_paths_normal_join():
    """Test normal path joining"""
    result = join_paths(base_path="http://0.0.0.0:4000", route="/api/v1")
    assert result == "http://0.0.0.0:4000/api/v1"


def test_join_paths_with_trailing_slash():
    """Test path joining with trailing slash on base_path"""
    result = join_paths(base_path="http://0.0.0.0:4000/", route="api/v1")
    assert result == "http://0.0.0.0:4000/api/v1"


def test_join_paths_empty_base():
    """Test path joining with empty base_path"""
    result = join_paths(base_path="", route="api/v1")
    assert result == "/api/v1"


def test_join_paths_empty_route():
    """Test path joining with empty route"""
    result = join_paths(base_path="http://0.0.0.0:4000", route="")
    assert result == "http://0.0.0.0:4000"


def test_join_paths_both_empty():
    """Test path joining with both empty"""
    result = join_paths(base_path="", route="")
    assert result == "/"


def test_join_paths_nested_path():
    """Test path joining with nested paths"""
    result = join_paths(base_path="http://0.0.0.0:4000/v1", route="chat/completions")
    assert result == "http://0.0.0.0:4000/v1/chat/completions"


def _patch_today(monkeypatch, year, month, day):
    class PatchedDate(real_datetime.date):
        @classmethod
        def today(cls):
            return real_datetime.date(year, month, day)

    monkeypatch.setattr("litellm.proxy.utils.date", PatchedDate)


def test_get_projected_spend_over_limit_day_one(monkeypatch):
    from litellm.proxy.utils import _get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 1, 1)
    result = _get_projected_spend_over_limit(100.0, 1.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == 3100.0
    assert projected_exceeded_date == real_datetime.date(2026, 1, 1)


def test_get_projected_spend_over_limit_december(monkeypatch):
    from litellm.proxy.utils import _get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 12, 15)
    result = _get_projected_spend_over_limit(100.0, 1.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == pytest.approx(214.28571428571428)
    assert projected_exceeded_date == real_datetime.date(2026, 12, 15)


def test_get_projected_spend_over_limit_includes_current_spend(monkeypatch):
    from litellm.proxy.utils import _get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 4, 11)
    result = _get_projected_spend_over_limit(100.0, 200.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == 290.0
    assert projected_exceeded_date == real_datetime.date(2026, 4, 21)


# ---------------------------------------------------------------------------
# L2: _enrich_http_exception_with_guardrail_context
# Regression coverage for case 2026-04-10-internal-bedrock-guardrail-streaming-error.
# ---------------------------------------------------------------------------


def test_enrich_http_exception_with_guardrail_context_dict_detail():
    """L2: dict-detail HTTPException is enriched with guardrail_name and mode."""
    from litellm.proxy.utils import _enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "bedrock-pii-guard"
        event_hook = "post_call"

    exc = HTTPException(status_code=400, detail={"error": "Violated guardrail policy"})
    _enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert exc.detail["guardrail_name"] == "bedrock-pii-guard"
    assert exc.detail["guardrail_mode"] == "post_call"


def test_enrich_http_exception_string_detail_noop():
    """L2: string-detail HTTPException is not mutated (can't add fields to a str)."""
    from litellm.proxy.utils import _enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "x"
        event_hook = "pre_call"

    exc = HTTPException(status_code=400, detail="Content blocked")
    _enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert exc.detail == "Content blocked"


def test_enrich_http_exception_setdefault_does_not_overwrite():
    """L2: a guardrail that already populates guardrail_name explicitly wins."""
    from litellm.proxy.utils import _enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "inferred-name"
        event_hook = "pre_call"

    exc = HTTPException(
        status_code=400,
        detail={"error": "x", "guardrail_name": "explicit-name"},
    )
    _enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert exc.detail["guardrail_name"] == "explicit-name"


def test_enrich_http_exception_non_http_exception_noop():
    """L2: non-HTTPException is left alone and the helper does not raise."""
    from litellm.proxy.utils import _enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "x"
        event_hook = "pre_call"

    exc = ValueError("not an HTTPException")
    _enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert str(exc) == "not an HTTPException"


def test_enrich_http_exception_callback_without_guardrail_name_noop():
    """L2: callback without guardrail_name attribute leaves detail alone."""
    from litellm.proxy.utils import _enrich_http_exception_with_guardrail_context

    class StubCallback:
        pass

    exc = HTTPException(status_code=400, detail={"error": "x"})
    _enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert exc.detail == {"error": "x"}


class TestPostCallFailureHookLiftsFirstApiCallStartTime:
    """post_call_failure_hook lifts first_api_call_start_time off the
    logging object into request_data (an internal top-level key) before
    the non-serialisable logging object is popped, so failure-path
    callbacks (OTel preprocessing latency) can still read it. It must
    never land in request_data["metadata"] (user request metadata,
    echoed downstream and typed Dict[str, str] in batch objects).
    """

    async def _run(self, request_data):
        from unittest.mock import AsyncMock, patch

        from litellm.proxy._types import UserAPIKeyAuth

        proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging_obj.alert_types = []  # skip alerting branch
        with patch.object(proxy_logging_obj, "update_request_status", new=AsyncMock()):
            await proxy_logging_obj.post_call_failure_hook(
                request_data=request_data,
                original_exception=Exception("boom"),
                user_api_key_dict=UserAPIKeyAuth(),
            )

    @pytest.mark.asyncio
    async def test_lifts_to_top_level_and_pops_logging_obj(self):
        handoff = real_datetime.datetime(2026, 1, 1, 0, 0, 0)
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"first_api_call_start_time": handoff}
        user_meta = {}
        request_data = {
            "litellm_logging_obj": logging_obj,
            "metadata": user_meta,
        }
        await self._run(request_data)

        assert request_data["first_api_call_start_time"] == handoff
        assert "litellm_logging_obj" not in request_data
        # user metadata is never touched
        assert user_meta == {}
        assert "first_api_call_start_time" not in request_data["metadata"]

    @pytest.mark.asyncio
    async def test_no_logging_obj_is_noop(self):
        request_data = {"metadata": {}}
        await self._run(request_data)
        assert "first_api_call_start_time" not in request_data

    @pytest.mark.asyncio
    async def test_logging_obj_without_anchor_is_noop(self):
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}
        request_data = {"litellm_logging_obj": logging_obj}
        await self._run(request_data)
        assert "first_api_call_start_time" not in request_data
        assert "litellm_logging_obj" not in request_data


# ---------------------------------------------------------------------------
# SRE-3691: monitor-only during_call_hook is non-blocking
#
# Regression coverage for the production incident where a slow Noma upstream
# (during_call_hook + monitor_mode=True + block_failures=False) stalled the
# LLM call because asyncio.gather waited on the moderation task. The fix:
# spawn monitor-only moderation tasks fire-and-forget while still preserving
# audit data via a tracked task set + shutdown drain.
# ---------------------------------------------------------------------------


class _StubMonitorOnlyGuardrail:
    """
    Minimal stand-in for a CustomGuardrail in monitor-only mode
    (monitor_mode=True, block_failures=False). Sleeps for `delay`
    seconds inside async_moderation_hook so we can prove the LLM
    call does NOT wait for it.
    """

    def __init__(
        self,
        guardrail_name: str = "noma-during-call",
        delay: float = 30.0,
        block_failures: bool = False,
        monitor_mode: bool = True,
    ):
        self.guardrail_name = guardrail_name
        self.event_hook = "during_call"
        self.delay = delay
        self.monitor_mode = monitor_mode
        self.block_failures = block_failures
        self.moderation_started = False
        self.moderation_completed = False

    def should_run_guardrail(self, data, event_type) -> bool:
        return True

    async def async_moderation_hook(self, data, user_api_key_dict, call_type):
        import asyncio as _asyncio

        self.moderation_started = True
        # NB: only mark completed on a clean exit — cancellation must not
        # be counted as a successful audit record.
        await _asyncio.sleep(self.delay)
        self.moderation_completed = True


def _make_callback_instance_of_custom_guardrail(stub):
    """
    Tell isinstance(stub, CustomGuardrail) checks to return True for our
    duck-typed stub so during_call_hook will exercise the real path.
    """
    from litellm.integrations.custom_guardrail import CustomGuardrail

    stub.__class__ = type(
        stub.__class__.__name__,
        (CustomGuardrail,),
        dict(stub.__class__.__dict__),
    )
    return stub


@pytest.mark.asyncio
async def test_during_call_hook_monitor_only_does_not_block_llm_call(monkeypatch):
    """
    SRE-3691: a monitor-only guardrail (monitor_mode=True, block_failures=False)
    that sleeps for 30s inside async_moderation_hook must NOT delay the caller.
    during_call_hook should return in ~milliseconds and leave the moderation
    task running in the background.
    """
    import asyncio
    import time

    import litellm

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    slow_guard = _make_callback_instance_of_custom_guardrail(
        _StubMonitorOnlyGuardrail(delay=30.0)
    )

    original_callbacks = litellm.callbacks
    litellm.callbacks = [slow_guard]
    try:
        start = time.monotonic()
        await proxy_logging_obj.during_call_hook(
            data={"messages": [{"role": "user", "content": "hi"}]},
            user_api_key_dict=None,
            call_type="acompletion",
        )
        elapsed = time.monotonic() - start

        # The hook must return promptly — the 30s sleep stays in the background.
        assert elapsed < 0.5, (
            f"during_call_hook blocked for {elapsed:.3f}s on a monitor-only "
            "guardrail; expected fire-and-forget"
        )
        # And the moderation task must actually be running (not skipped).
        # Yield to the loop so the spawned task gets a chance to start.
        await asyncio.sleep(0)
        assert slow_guard.moderation_started is True
        assert slow_guard.moderation_completed is False
        # And the proxy is holding a strong reference so it isn't GC'd.
        assert len(proxy_logging_obj._pending_monitor_tasks) == 1
    finally:
        # Cancel the background sleep so the test doesn't leak it.
        for t in list(proxy_logging_obj._pending_monitor_tasks):
            t.cancel()
        await asyncio.gather(
            *list(proxy_logging_obj._pending_monitor_tasks),
            return_exceptions=True,
        )
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_during_call_hook_blocking_guardrail_still_awaited(monkeypatch):
    """
    A guardrail with block_failures=True (default) must still be awaited
    synchronously by during_call_hook. This preserves authoritative
    guardrail semantics.
    """
    import time

    import litellm

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    blocking_guard = _make_callback_instance_of_custom_guardrail(
        _StubMonitorOnlyGuardrail(
            guardrail_name="noma-blocking",
            delay=0.2,
            block_failures=True,
            monitor_mode=False,
        )
    )

    original_callbacks = litellm.callbacks
    litellm.callbacks = [blocking_guard]
    try:
        start = time.monotonic()
        await proxy_logging_obj.during_call_hook(
            data={"messages": [{"role": "user", "content": "hi"}]},
            user_api_key_dict=None,
            call_type="acompletion",
        )
        elapsed = time.monotonic() - start

        # The hook must have waited for the blocking moderation to finish.
        assert elapsed >= 0.2
        assert blocking_guard.moderation_completed is True
        # No tasks should be parked in the fire-and-forget set.
        assert len(proxy_logging_obj._pending_monitor_tasks) == 0
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_during_call_hook_mcp_call_monitor_only_is_awaited_not_detached():
    """MCP is the ONE case where a monitor-only guardrail is still awaited.

    PR #35 originally detached this path too, on the reasoning that MCP tool execution
    should not stall on a slow guardrail either. That was revised on 2026-08-02 after
    measuring what the detach costs: a detached task writes its audit record into a
    snapshot of the request, and the spend-log row — which also feeds the S3 LLM logs used
    as HITRUST evidence — is assembled from the live dict afterwards. So detaching means
    the scan happens and is never recorded on our side.

    For the LLM ``during_call`` hook that is accepted: ``pre_call`` covers most of the same
    request-side content, and the latency bought back is on the streaming path where a
    stalled Noma is visible mid-response.

    For ``during_mcp_call`` it is not. ``pre_mcp_call`` runs BEFORE the argument rewrite
    (``mcp_server_manager`` reassigns ``arguments`` from the pre-hook result and only then
    builds the during-hook task), so this hook is the only audit of what a tool actually
    received. It stays awaited and is bounded by ARC-BUG-43's scan deadline instead —
    affordable because an MCP tool call is a discrete request/response, not a token stream,
    so a bounded wait reads as a slow tool rather than a response that stalls mid-sentence.
    """
    import asyncio
    import time
    from unittest.mock import MagicMock

    import litellm
    from litellm.integrations.opentelemetry import UserAPIKeyAuth

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    slow_guard = _make_callback_instance_of_custom_guardrail(
        _StubMonitorOnlyGuardrail(guardrail_name="noma-during-mcp-call", delay=30.0)
    )

    original_callbacks = litellm.callbacks
    litellm.callbacks = [slow_guard]
    try:
        # A 30s scan would make an awaited hook hang, so use a fast one: what is under
        # test is WHERE the scan runs, not how long it takes.
        fast_guard = _make_callback_instance_of_custom_guardrail(
            _StubMonitorOnlyGuardrail(guardrail_name="noma-during-mcp-call", delay=0.0)
        )
        litellm.callbacks = [fast_guard]

        await proxy_logging_obj.during_call_hook(
            data={"messages": [{"role": "user", "content": "hi"}]},
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
            call_type="call_mcp_tool",
        )

        # Awaited: complete on return, and nothing left running in the background.
        assert fast_guard.moderation_completed is True
        assert len(proxy_logging_obj._pending_monitor_tasks) == 0
    finally:
        for t in list(proxy_logging_obj._pending_monitor_tasks):
            t.cancel()
        await asyncio.gather(
            *list(proxy_logging_obj._pending_monitor_tasks),
            return_exceptions=True,
        )
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_drain_pending_monitor_tasks_waits_for_in_flight():
    """
    SRE-3691 shutdown-drain: in-flight monitor-only moderation tasks that
    complete within the timeout window must be awaited on shutdown so we
    don't lose audit data.
    """
    import asyncio

    import litellm

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    fast_guard = _make_callback_instance_of_custom_guardrail(
        _StubMonitorOnlyGuardrail(guardrail_name="noma-fast", delay=0.1)
    )

    original_callbacks = litellm.callbacks
    litellm.callbacks = [fast_guard]
    try:
        await proxy_logging_obj.during_call_hook(
            data={"messages": [{"role": "user", "content": "hi"}]},
            user_api_key_dict=None,
            call_type="acompletion",
        )
        # Task should be pending immediately after the hook returns.
        assert len(proxy_logging_obj._pending_monitor_tasks) == 1
        assert fast_guard.moderation_completed is False

        # Drain with a timeout longer than the task's natural runtime.
        await proxy_logging_obj.drain_pending_monitor_tasks(timeout=2.0)

        # The task should have completed naturally (not been cancelled),
        # and the pending set should be empty (done-callback removed it).
        assert fast_guard.moderation_completed is True
        assert len(proxy_logging_obj._pending_monitor_tasks) == 0
    finally:
        litellm.callbacks = original_callbacks
        # Defensive cleanup in case the assertions above failed mid-way.
        for t in list(proxy_logging_obj._pending_monitor_tasks):
            t.cancel()
        await asyncio.gather(
            *list(proxy_logging_obj._pending_monitor_tasks),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_drain_pending_monitor_tasks_cancels_after_timeout():
    """
    SRE-3691 shutdown-drain timeout: if a monitor-only task is still hung
    on a slow upstream after the drain timeout elapses, we cancel it so
    shutdown can proceed instead of hanging the worker indefinitely.
    """
    import asyncio

    import litellm

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    forever_guard = _make_callback_instance_of_custom_guardrail(
        _StubMonitorOnlyGuardrail(guardrail_name="noma-hung", delay=60.0)
    )

    original_callbacks = litellm.callbacks
    litellm.callbacks = [forever_guard]
    try:
        await proxy_logging_obj.during_call_hook(
            data={"messages": [{"role": "user", "content": "hi"}]},
            user_api_key_dict=None,
            call_type="acompletion",
        )
        # Yield so the spawned task starts and registers the sleep.
        await asyncio.sleep(0)
        assert forever_guard.moderation_started is True
        assert forever_guard.moderation_completed is False

        # Drain with a tight timeout — the hung task should be cancelled.
        await proxy_logging_obj.drain_pending_monitor_tasks(timeout=0.1)

        # Pending set should be empty (done-callback fired on cancellation),
        # and the task should NOT have completed naturally.
        assert len(proxy_logging_obj._pending_monitor_tasks) == 0
        assert forever_guard.moderation_completed is False
    finally:
        litellm.callbacks = original_callbacks
        for t in list(proxy_logging_obj._pending_monitor_tasks):
            t.cancel()
        await asyncio.gather(
            *list(proxy_logging_obj._pending_monitor_tasks),
            return_exceptions=True,
        )


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_proxy_max_pending_monitor_tasks_rejects_non_positive(bad_value, caplog):
    """
    PR #35 review finding: a non-positive PROXY_MAX_PENDING_MONITOR_TASKS
    (0 or negative) would make ``len(set) >= cap`` true for the very first
    monitor-only task, silently disabling ALL fire-and-forget guardrails
    (every cap check returns True → spawn path skipped → zero audit data
    out of the pod). Far more likely a config typo than an intentional
    kill-switch.

    The fix: ProxyLogging.__init__ coerces non-positive values to the
    default 1024 and logs a startup warning so the misconfig is visible.
    """
    import logging

    from litellm.proxy import utils as proxy_utils_module

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        # Patch the module-level constant — that's what ProxyLogging reads
        # at construction time. (Patching env vars after import is too
        # late; constants.py already resolved them.)
        original = proxy_utils_module.PROXY_MAX_PENDING_MONITOR_TASKS
        proxy_utils_module.PROXY_MAX_PENDING_MONITOR_TASKS = bad_value
        try:
            proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
        finally:
            proxy_utils_module.PROXY_MAX_PENDING_MONITOR_TASKS = original

    # Coerced to the safe default — monitor-only guardrails stay enabled.
    assert proxy_logging_obj._max_pending_monitor_tasks == 1024
    # And the misconfig is visible in logs.
    assert any(
        "PROXY_MAX_PENDING_MONITOR_TASKS" in rec.getMessage()
        and "non-positive" in rec.getMessage()
        for rec in caplog.records
    ), f"expected coercion warning, got records: {[r.getMessage() for r in caplog.records]!r}"


def test_proxy_max_pending_monitor_tasks_accepts_positive():
    """Positive values pass through unchanged (no coercion, no warning)."""
    from litellm.proxy import utils as proxy_utils_module

    original = proxy_utils_module.PROXY_MAX_PENDING_MONITOR_TASKS
    proxy_utils_module.PROXY_MAX_PENDING_MONITOR_TASKS = 42
    try:
        proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    finally:
        proxy_utils_module.PROXY_MAX_PENDING_MONITOR_TASKS = original
    assert proxy_logging_obj._max_pending_monitor_tasks == 42


def test_is_monitor_only_guardrail_classification():
    """
    Truth table for _is_monitor_only_guardrail. Only the explicit
    monitor_mode=True + block_failures=False combination opts into
    fire-and-forget; everything else stays blocking.
    """

    class _C:
        def __init__(self, monitor_mode, block_failures):
            self.monitor_mode = monitor_mode
            self.block_failures = block_failures

    classify = ProxyLogging._is_monitor_only_guardrail

    assert classify(_C(monitor_mode=True, block_failures=False)) is True
    assert classify(_C(monitor_mode=True, block_failures=True)) is False
    assert classify(_C(monitor_mode=False, block_failures=False)) is False
    assert classify(_C(monitor_mode=False, block_failures=True)) is False

    # Missing attributes — defaults are "blocking" (safe default).
    class _Bare:
        pass

    assert classify(_Bare()) is False

    # monitor_mode without block_failures attr — defaults to blocking.
    bare = _Bare()
    bare.monitor_mode = True  # type: ignore[attr-defined]
    assert classify(bare) is False


@pytest.mark.asyncio
async def test_spawn_monitor_only_task_drops_when_cap_reached():
    """
    SRE-3691 follow-up: monitor-only pending task set must not grow without
    bound. Once ``_max_pending_monitor_tasks`` is hit, additional tasks are
    dropped (logged) instead of spawned so a sustained Noma outage at 100 RPS
    can't OOM the worker. Existing in-flight tasks must keep running.
    """
    import asyncio

    import litellm

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    # Tight cap so the test stays cheap; the production default is 1024.
    proxy_logging_obj._max_pending_monitor_tasks = 4

    # Build a guardrail that hangs "forever" inside async_moderation_hook.
    guard = _make_callback_instance_of_custom_guardrail(
        _StubMonitorOnlyGuardrail(guardrail_name="noma-cap-test", delay=60.0)
    )

    original_callbacks = litellm.callbacks
    litellm.callbacks = [guard]
    try:
        # Spawn exactly the cap. Each call to during_call_hook spawns one task.
        for _ in range(proxy_logging_obj._max_pending_monitor_tasks):
            await proxy_logging_obj.during_call_hook(
                data={"messages": [{"role": "user", "content": "hi"}]},
                user_api_key_dict=None,
                call_type="acompletion",
            )

        # Cap is reached.
        assert (
            len(proxy_logging_obj._pending_monitor_tasks)
            == proxy_logging_obj._max_pending_monitor_tasks
        )
        assert proxy_logging_obj._dropped_monitor_tasks_since_warning == 0

        # The next spawn should be dropped — task count must not grow.
        await proxy_logging_obj.during_call_hook(
            data={"messages": [{"role": "user", "content": "hi"}]},
            user_api_key_dict=None,
            call_type="acompletion",
        )

        # Still at the cap (not cap + 1), and the drop counter is now 1
        # (monotonic, never reset). The first drop in a burst always
        # emits a warning, then every 100 drops thereafter. The counter
        # itself never returns to 0 while the outage continues — that's
        # exactly the property the PR #35 review fix preserves.
        assert (
            len(proxy_logging_obj._pending_monitor_tasks)
            == proxy_logging_obj._max_pending_monitor_tasks
        )
        assert proxy_logging_obj._dropped_monitor_tasks_since_warning == 1
    finally:
        litellm.callbacks = original_callbacks
        # Yield to the loop so each spawned wrapper task actually enters its
        # body before we cancel — otherwise the inner ``coro`` was never
        # awaited and Python emits a RuntimeWarning at GC.
        await asyncio.sleep(0)
        for t in list(proxy_logging_obj._pending_monitor_tasks):
            t.cancel()
        await asyncio.gather(
            *list(proxy_logging_obj._pending_monitor_tasks),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_during_call_hook_monitor_only_snapshots_data_dict():
    """
    PR #35 review finding (Angles A/B/D/E): fire-and-forget monitor-only
    moderation tasks must receive their OWN snapshot of ``data``, not a
    live reference. Two failure modes the live ref causes:

    1) ``data["guardrail_to_apply"] = callback`` is set per-iteration for
       apply_guardrail callbacks. With a live ref, the next loop iteration
       overwrites the field BEFORE the spawned task observes it — the
       task sees a sibling callback's value (or, after the loop, no value
       at all because the unified guardrail pops it).

    2) After during_call_hook returns, the caller mutates fields like
       ``data["deployment"]`` and ``data["_hidden_params"]``. A slow
       monitor-only moderation task would observe those fields in their
       mutated, mid-call state — corrupting any audit record that includes
       them.

    This test asserts each spawned task captured its own snapshot whose
    contents reflect ``data`` at spawn time, NOT the post-spawn mutations.
    """
    import asyncio

    import litellm

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())

    observed = {}

    class _SnapshotCapturingGuardrail(_StubMonitorOnlyGuardrail):
        # _make_callback_instance_of_custom_guardrail rebases the instance
        # under (CustomGuardrail,) and only preserves THIS class's
        # __dict__, so we have to redeclare ``should_run_guardrail`` here
        # rather than inheriting it from _StubMonitorOnlyGuardrail.
        def should_run_guardrail(self, data, event_type) -> bool:
            return True

        async def async_moderation_hook(self, data, user_api_key_dict, call_type):
            # Wait a tick so the parent loop has a chance to mutate the
            # original ``data`` dict before we record what WE see. If our
            # snapshot worked, our recorded value stays at its spawn-time
            # state; without the snapshot, we'd see the post-mutation value.
            await asyncio.sleep(0.05)
            observed[self.guardrail_name] = {
                "deployment": data.get("deployment"),
                "_hidden_params": data.get("_hidden_params"),
            }
            self.moderation_started = True
            self.moderation_completed = True

    guard_a = _make_callback_instance_of_custom_guardrail(
        _SnapshotCapturingGuardrail(guardrail_name="noma-a", delay=0.0)
    )
    guard_b = _make_callback_instance_of_custom_guardrail(
        _SnapshotCapturingGuardrail(guardrail_name="noma-b", delay=0.0)
    )

    original_callbacks = litellm.callbacks
    litellm.callbacks = [guard_a, guard_b]
    try:
        data = {
            "messages": [{"role": "user", "content": "hi"}],
            "deployment": "pre-call-deployment",
            "_hidden_params": {"foo": "pre-call"},
        }
        await proxy_logging_obj.during_call_hook(
            data=data,
            user_api_key_dict=None,
            call_type="acompletion",
        )

        # Simulate the post-LLM-call mutation of ``data`` that happens in
        # the real request lifecycle. The spawned tasks must NOT observe
        # this — they should each have their own snapshot.
        data["deployment"] = "POST-CALL-DEPLOYMENT"
        data["_hidden_params"] = {"foo": "POST-CALL"}

        # Wait for the spawned tasks to finish.
        for _ in range(20):
            await asyncio.sleep(0.05)
            if "noma-a" in observed and "noma-b" in observed:
                break

        assert "noma-a" in observed, f"task A did not run; observed: {observed!r}"
        assert "noma-b" in observed, f"task B did not run; observed: {observed!r}"

        # Each task must have seen the spawn-time values, NOT the
        # post-call mutations. This is the core regression assertion.
        assert observed["noma-a"]["deployment"] == "pre-call-deployment"
        assert observed["noma-a"]["_hidden_params"] == {"foo": "pre-call"}
        assert observed["noma-b"]["deployment"] == "pre-call-deployment"
        assert observed["noma-b"]["_hidden_params"] == {"foo": "pre-call"}
    finally:
        litellm.callbacks = original_callbacks
        for t in list(proxy_logging_obj._pending_monitor_tasks):
            t.cancel()
        await asyncio.gather(
            *list(proxy_logging_obj._pending_monitor_tasks),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_during_call_hook_monitor_only_apply_guardrail_per_task_callback():
    """
    PR #35 review finding (the ``guardrail_to_apply`` race): when two
    monitor-only callbacks both use the apply_guardrail path, each spawned
    task must see ITS OWN callback in ``data["guardrail_to_apply"]`` — not
    the sibling's. With the original live-data-dict implementation the
    second iteration would overwrite the field before the first task
    observed it; the first task would either see callback B or, more
    commonly, see ``None`` (because unified_guardrail.async_moderation_hook
    pops the field, racing the second iteration's set).
    """
    import asyncio

    import litellm

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())

    observed_callbacks: list = []

    class _ApplyGuardrailCapture(_StubMonitorOnlyGuardrail):
        # The `_is_monitor_only_guardrail + apply_guardrail` branch in
        # during_call_hook requires (a) "apply_guardrail" in
        # type(callback).__dict__ and (b) user_api_key_dict not None.
        def apply_guardrail(self, *args, **kwargs):  # noqa: D401
            """Required to trigger the unified-guardrail dispatch branch."""
            return None

        # _make_callback_instance_of_custom_guardrail rebases the instance
        # under (CustomGuardrail,) and only preserves THIS class's
        # __dict__, so we have to redeclare ``should_run_guardrail`` here
        # rather than inheriting it from _StubMonitorOnlyGuardrail.
        def should_run_guardrail(self, data, event_type) -> bool:
            return True

        async def async_moderation_hook(self, data, user_api_key_dict, call_type):
            # Wait so the parent loop iterates past us and (with the
            # buggy live-ref code) would overwrite the field.
            await asyncio.sleep(0.05)
            observed_callbacks.append(
                (self.guardrail_name, data.get("guardrail_to_apply"))
            )
            self.moderation_started = True
            self.moderation_completed = True

    guard_a = _make_callback_instance_of_custom_guardrail(
        _ApplyGuardrailCapture(guardrail_name="noma-apply-a", delay=0.0)
    )
    guard_b = _make_callback_instance_of_custom_guardrail(
        _ApplyGuardrailCapture(guardrail_name="noma-apply-b", delay=0.0)
    )

    # Patch unified_guardrail.async_moderation_hook so we capture what each
    # spawned task observed without needing the full unified-guardrail
    # dispatch chain. The spec for the bugfix is "each spawned task gets
    # its own immutable view of data[guardrail_to_apply]"; we verify that
    # by inspecting the data dict each task receives.
    from litellm.proxy import utils as proxy_utils_module

    real_unified = proxy_utils_module.unified_guardrail

    class _StubUnified:
        async def async_moderation_hook(self, user_api_key_dict, data, call_type):
            await asyncio.sleep(0.05)
            observed_callbacks.append(
                (
                    getattr(data.get("guardrail_to_apply"), "guardrail_name", None),
                    data.get("guardrail_to_apply"),
                )
            )

    proxy_utils_module.unified_guardrail = _StubUnified()

    original_callbacks = litellm.callbacks
    litellm.callbacks = [guard_a, guard_b]
    try:
        from unittest.mock import MagicMock as _MagicMock

        await proxy_logging_obj.during_call_hook(
            data={"messages": [{"role": "user", "content": "hi"}]},
            user_api_key_dict=_MagicMock(),
            call_type="acompletion",
        )

        # Wait for both tasks to finish.
        for _ in range(20):
            await asyncio.sleep(0.05)
            if len(observed_callbacks) >= 2:
                break

        assert len(observed_callbacks) == 2, (
            f"expected both apply_guardrail tasks to run, "
            f"got {observed_callbacks!r}"
        )
        # Each task must have seen ITS OWN callback in
        # data["guardrail_to_apply"]. The list comes back unordered
        # (depends on which coroutine the loop schedules first), so check
        # by name.
        by_name = {name: cb for name, cb in observed_callbacks}
        assert (
            by_name["noma-apply-a"] is guard_a
        ), f"task A saw the wrong callback: {by_name!r}"
        assert (
            by_name["noma-apply-b"] is guard_b
        ), f"task B saw the wrong callback: {by_name!r}"
    finally:
        proxy_utils_module.unified_guardrail = real_unified
        litellm.callbacks = original_callbacks
        for t in list(proxy_logging_obj._pending_monitor_tasks):
            t.cancel()
        await asyncio.gather(
            *list(proxy_logging_obj._pending_monitor_tasks),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_spawn_monitor_only_task_drop_warning_throttles_correctly():
    """
    PR #35 review finding: the drop-warning throttle must NOT log on every
    drop during a sustained outage. The original code reset
    ``_dropped_monitor_tasks_since_warning`` to 0 inside the warning branch,
    which caused the next drop to match ``== 1`` and fire again — the exact
    log-storm the throttle was meant to prevent.

    The fix: monotonic counter + separate next-threshold field, advancing by
    ``_DROP_WARNING_INTERVAL`` (100) per emission. Verifies that across 250
    consecutive drops we emit exactly 3 warnings (at drop 1, drop 101, and
    drop 201) and leave the monotonic counter at 250 — never reset.
    """
    import logging

    from litellm.proxy.utils import _DROP_WARNING_INTERVAL

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    # Set the cap to 0 so EVERY spawn attempt is dropped — simplest way to
    # exercise the throttle without juggling 250 real pending tasks.
    proxy_logging_obj._max_pending_monitor_tasks = 0

    drop_count = 250

    class _Cb:
        guardrail_name = "noma-drop-test"

    # Count warning emissions by patching the logger directly. We can't rely
    # on caplog because verbose_proxy_logger may be configured at a higher
    # level in this process.
    warnings_emitted = []
    logger = logging.getLogger("LiteLLM Proxy")
    real_warning = logger.warning

    def _capture(msg, *args, **kwargs):
        warnings_emitted.append(msg % args if args else msg)

    logger.warning = _capture  # type: ignore[method-assign]
    try:
        for _ in range(drop_count):
            # Build a no-op coroutine so the close-on-drop path doesn't
            # raise. _spawn_monitor_only_guardrail_task closes the coro
            # itself on the drop branch.
            async def _noop():
                return None

            result = proxy_logging_obj._spawn_monitor_only_guardrail_task(
                _Cb(), _noop()
            )
            assert result is None
    finally:
        logger.warning = real_warning  # type: ignore[method-assign]

    # Counter is monotonic — never reset.
    assert proxy_logging_obj._dropped_monitor_tasks_since_warning == drop_count
    # With initial threshold=1 and interval=100, emissions fire when the
    # counter first reaches 1, 101, 201, ... Across 250 drops that's
    # exactly 3 emissions. The bug being fixed was: every single drop fired
    # a warning because the counter reset to 0 in the warning branch.
    assert len(warnings_emitted) == 3, (
        f"expected 3 warning emissions for {drop_count} drops at interval "
        f"{_DROP_WARNING_INTERVAL}, got {len(warnings_emitted)}: "
        f"{warnings_emitted!r}"
    )
    # Next threshold should be one interval past the last emission point.
    assert proxy_logging_obj._next_dropped_monitor_task_warning_at == 301


@pytest.mark.asyncio
async def test_during_call_hook_survives_closed_loop(monkeypatch):
    """
    SRE-3691 follow-up: when a request lands mid-shutdown the event loop
    may already be closed. The monitor-only spawn path must skip cleanly
    (warn + close coroutine + return None) and let the rest of the LLM
    call proceed — blocking guardrails on the same call should still run.

    PR #35 review finding: this path is now guarded by
    ``asyncio.get_running_loop().is_closed()`` BEFORE calling
    create_task, rather than catching every ``RuntimeError`` from
    create_task. That keeps real-bug RuntimeErrors (uvloop quirks,
    programming mistakes, future Python additions) visible instead
    of silently swallowed.
    """
    import asyncio

    import litellm

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())

    # A monitor-only guardrail whose spawn must be skipped.
    monitor_guard = _make_callback_instance_of_custom_guardrail(
        _StubMonitorOnlyGuardrail(guardrail_name="noma-monitor-shutdown", delay=30.0)
    )
    # A blocking guardrail that must still run successfully.
    blocking_guard = _make_callback_instance_of_custom_guardrail(
        _StubMonitorOnlyGuardrail(
            guardrail_name="noma-blocking-shutdown",
            delay=0.05,
            block_failures=True,
            monitor_mode=False,
        )
    )

    # Patch get_running_loop to return a stand-in whose is_closed()
    # returns True, simulating mid-shutdown.
    real_get_running_loop = asyncio.get_running_loop

    class _ClosedLoop:
        def is_closed(self):
            return True

    def fake_get_running_loop():
        return _ClosedLoop()

    monkeypatch.setattr(
        "litellm.proxy.utils.asyncio.get_running_loop", fake_get_running_loop
    )

    original_callbacks = litellm.callbacks
    litellm.callbacks = [monitor_guard, blocking_guard]
    try:
        # Must NOT raise — the monitor spawn should skip via the
        # is_closed() guard, not via an exception.
        await proxy_logging_obj.during_call_hook(
            data={"messages": [{"role": "user", "content": "hi"}]},
            user_api_key_dict=None,
            call_type="acompletion",
        )
        # No pending task should have been registered (spawn skipped).
        assert len(proxy_logging_obj._pending_monitor_tasks) == 0
        # Blocking guardrail still completed (gather doesn't use create_task).
        assert blocking_guard.moderation_completed is True
    finally:
        litellm.callbacks = original_callbacks
        monkeypatch.setattr(
            "litellm.proxy.utils.asyncio.get_running_loop", real_get_running_loop
        )


@pytest.mark.asyncio
async def test_spawn_monitor_only_task_propagates_unexpected_runtime_error(monkeypatch):
    """
    PR #35 review finding: the previous ``except RuntimeError`` arm caught
    ALL RuntimeErrors from ``asyncio.create_task`` — including programming
    bugs, uvloop-specific errors, and any future Python additions. The
    refactor pre-checks ``asyncio.get_running_loop().is_closed()`` for
    the only-known-benign case; every OTHER RuntimeError from create_task
    must now propagate so it can be observed and fixed.
    """
    import asyncio

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())

    class _Cb:
        guardrail_name = "noma-runtime-error"
        monitor_mode = True
        block_failures = False

    # Sentinel RuntimeError that is NOT "Event loop is closed". With the
    # is_closed() guard returning False (real loop), create_task is
    # actually called and this error must propagate up.
    boom = RuntimeError("synthetic bug — must not be swallowed")

    def exploding_create_task(coro, *args, **kwargs):
        close_fn = getattr(coro, "close", None)
        if callable(close_fn):
            close_fn()
        raise boom

    monkeypatch.setattr(
        "litellm.proxy.utils.asyncio.create_task", exploding_create_task
    )

    async def _noop():
        return None

    noop_coro = _noop()
    try:
        with pytest.raises(RuntimeError, match="synthetic bug"):
            proxy_logging_obj._spawn_monitor_only_guardrail_task(_Cb(), noop_coro)
    finally:
        # The inner coroutine was passed into _run_guardrail_task_with_enrichment
        # but the wrapper task never ran (create_task exploded), so close
        # the inner coro by hand to avoid a "never awaited" RuntimeWarning.
        close_fn = getattr(noop_coro, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


@pytest.mark.asyncio
async def test_spawn_monitor_only_task_skips_when_loop_is_closed(monkeypatch):
    """
    PR #35 review finding: the is_closed() pre-check must skip the
    create_task call entirely (returning None, closing the coroutine)
    when the loop is closed — proving the guard runs BEFORE create_task,
    not after. If create_task were called we'd see the AssertionError
    below fire.
    """
    import asyncio

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())

    class _Cb:
        guardrail_name = "noma-closed-loop"
        monitor_mode = True
        block_failures = False

    class _ClosedLoop:
        def is_closed(self):
            return True

    monkeypatch.setattr(
        "litellm.proxy.utils.asyncio.get_running_loop", lambda: _ClosedLoop()
    )

    create_task_calls = []

    def must_not_be_called(coro, *args, **kwargs):
        # If we get here, the guard didn't run BEFORE create_task — the
        # refactor regressed.
        create_task_calls.append(coro)
        raise AssertionError(
            "create_task was called even though the loop is closed; "
            "the is_closed() guard must run first"
        )

    monkeypatch.setattr("litellm.proxy.utils.asyncio.create_task", must_not_be_called)

    async def _noop():
        return None

    result = proxy_logging_obj._spawn_monitor_only_guardrail_task(_Cb(), _noop())
    assert result is None
    assert create_task_calls == []
    assert len(proxy_logging_obj._pending_monitor_tasks) == 0


@pytest.mark.asyncio
async def test_drain_pending_monitor_tasks_bounded_against_cancel_swallowing():
    """
    SRE-3691 follow-up: ``drain_pending_monitor_tasks`` must be bounded
    even when a custom guardrail swallows ``CancelledError`` and keeps
    awaiting ``asyncio.sleep``. The implementation uses ``asyncio.wait``
    (NOT ``wait_for(gather)`` with shield) so:
      - Phase 1 waits up to ``timeout`` for natural completion without
        ever cancelling its argument tasks (so a swallowing task can't
        hang the wait).
      - Phase 2 explicitly cancels still-pending tasks, then ``wait``s
        with a tight cleanup timeout for done-callbacks to fire. Tasks
        that swallow cancellation drop out of the ``done`` set and stay
        tracked on ``_pending_monitor_tasks`` — no shielded background
        gather is leaked.

    This test installs a maximally-malicious guardrail that catches every
    cancellation and re-enters ``asyncio.sleep``, and asserts the drain
    returns within ~1.3s (primary timeout 0.1s + cleanup timeout 1.0s +
    small overhead) instead of hanging forever.
    """
    import asyncio
    import time

    import litellm

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())

    # ``allow_exit`` is set to True in finally so the still-running evil
    # task can actually terminate after the assertions complete. Without
    # this, pytest-asyncio's event loop would wait on the orphan task and
    # the test would hit its --timeout=30 thread guard.
    allow_exit = {"flag": False}

    async def _evil_hook(self, data, user_api_key_dict, call_type):
        # Worst-case malicious pattern: catch every cancellation and keep
        # awaiting. The ``allow_exit`` flag is the test harness's escape
        # hatch — production code does NOT have this. The production
        # contract is that drain logs and returns; the swallowing task
        # stays tracked on ``_pending_monitor_tasks`` and is GC'd at
        # interpreter exit (not via uvicorn SIGKILL).
        self.moderation_started = True
        while not allow_exit["flag"]:
            try:
                await asyncio.sleep(0.05)
            except BaseException:
                continue
        self.moderation_completed = True

    evil_guard = _make_callback_instance_of_custom_guardrail(
        _StubMonitorOnlyGuardrail(guardrail_name="noma-evil", delay=60.0)
    )
    # Replace the hook on the instance so isinstance(CustomGuardrail) and
    # should_run_guardrail stay intact.
    evil_guard.async_moderation_hook = _evil_hook.__get__(  # type: ignore[method-assign]
        evil_guard, type(evil_guard)
    )

    original_callbacks = litellm.callbacks
    litellm.callbacks = [evil_guard]
    try:
        await proxy_logging_obj.during_call_hook(
            data={"messages": [{"role": "user", "content": "hi"}]},
            user_api_key_dict=None,
            call_type="acompletion",
        )
        # Yield so the spawned wrapper enters its body.
        await asyncio.sleep(0)
        assert evil_guard.moderation_started is True
        assert len(proxy_logging_obj._pending_monitor_tasks) == 1

        # Drain budget: primary asyncio.wait (0.1s) + secondary cleanup
        # asyncio.wait (1.0s) + small overhead. asyncio.wait does not
        # cancel its argument tasks on timeout, so a CancelledError-
        # swallowing task can't hang the call. A regression to
        # ``asyncio.wait_for(asyncio.gather(...))`` without ``shield``
        # would hang this test until pytest's --timeout=30 fires.
        start = time.monotonic()
        await proxy_logging_obj.drain_pending_monitor_tasks(timeout=0.1)
        elapsed = time.monotonic() - start

        assert elapsed < 2.5, (
            f"drain_pending_monitor_tasks took {elapsed:.2f}s on a "
            "cancellation-swallowing guardrail; expected ~1.1s. "
            "asyncio.wait bound likely regressed."
        )
        # The malicious task must NOT have completed naturally — drain
        # abandoned it instead of waiting it out.
        assert evil_guard.moderation_completed is False
        # Sanity: the drain returned at least the primary timeout. (Anything
        # lower would mean primary wait_for didn't actually wait — a bug.)
        assert elapsed >= 0.1
    finally:
        litellm.callbacks = original_callbacks
        # Let the evil task exit cleanly so pytest-asyncio's loop can
        # shut down. This is the test-harness equivalent of "uvicorn
        # SIGKILL reaps the leftover" — without this we'd leak the task.
        allow_exit["flag"] = True
        await asyncio.sleep(0.1)  # give it a tick to observe the flag
        for t in list(proxy_logging_obj._pending_monitor_tasks):
            if not t.done():
                t.cancel()
        await asyncio.gather(
            *list(proxy_logging_obj._pending_monitor_tasks),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_spawn_monitor_only_task_rejected_during_shutdown():
    """
    PR #35 review finding: ``drain_pending_monitor_tasks`` snapshots
    ``_pending_monitor_tasks`` ONCE at entry. uvicorn's graceful shutdown
    can still hand off a request after the drain has started — its
    monitor-only moderation task would be spawned AFTER the snapshot,
    never get awaited, and either leak (until uvicorn SIGKILLs) or be
    silently dropped on the loop close.

    Fix: ``drain_pending_monitor_tasks`` flips ``_shutting_down = True``
    before snapshotting, and ``_spawn_monitor_only_guardrail_task``
    rejects new spawns when that flag is set.

    This test sets the flag directly (simulating a concurrent drain
    in progress) and asserts the spawn path returns None instead of
    creating a task.
    """

    class _Cb:
        guardrail_name = "noma-shutdown"

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())

    # Sanity: pre-shutdown spawn succeeds.
    import asyncio

    async def _noop_ok():
        return None

    task = proxy_logging_obj._spawn_monitor_only_guardrail_task(_Cb(), _noop_ok())
    assert task is not None
    # Let the spawned wrapper actually enter its body so the inner coro
    # is awaited (otherwise pytest emits "coroutine was never awaited").
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except BaseException:
        pass

    # Flip the gate (as drain_pending_monitor_tasks does at entry) and
    # try to spawn again — must be rejected.
    proxy_logging_obj._shutting_down = True

    async def _noop_rejected():
        return None

    pending_before = len(proxy_logging_obj._pending_monitor_tasks)
    result = proxy_logging_obj._spawn_monitor_only_guardrail_task(
        _Cb(), _noop_rejected()
    )
    assert result is None, "shutdown gate did not reject the new spawn"
    # The pending set MUST not have grown — the rejected spawn never
    # created a task. (Existing already-finished tasks may still be
    # in the set if their done-callback hasn't fired yet; that's fine,
    # the contract is "don't ADD new ones once shutdown started".)
    assert len(proxy_logging_obj._pending_monitor_tasks) == pending_before


@pytest.mark.asyncio
async def test_drain_pending_monitor_tasks_sets_shutdown_flag():
    """
    PR #35 review finding: the shutdown gate must be set at drain ENTRY
    (before snapshotting), not at exit. Otherwise a spawn that lands
    between snapshot and gate-flip would still race past the drain.
    """
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert proxy_logging_obj._shutting_down is False

    # Drain on an empty pending set is a fast no-op, but the flag flip
    # must happen regardless.
    await proxy_logging_obj.drain_pending_monitor_tasks(timeout=0.1)
    assert proxy_logging_obj._shutting_down is True


class TestPostCallFailureHookLiftsRecoveredPartialSpend:
    """A stream that broke mid-flight still billed the provider for the chunks
    already delivered. The streaming handler stashes that recovered usage and
    cost on the logging object; post_call_failure_hook must lift them onto
    request_data before the logging object is popped, so the failure-path spend
    callbacks (which run after the pop) record the real partial spend.
    """

    async def _run(self, request_data):
        from unittest.mock import AsyncMock, patch

        from litellm.proxy._types import UserAPIKeyAuth

        proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging_obj.alert_types = []
        with patch.object(proxy_logging_obj, "update_request_status", new=AsyncMock()):
            await proxy_logging_obj.post_call_failure_hook(
                request_data=request_data,
                original_exception=Exception("boom"),
                user_api_key_dict=UserAPIKeyAuth(),
            )

    @pytest.mark.asyncio
    async def test_lifts_recovered_usage_and_cost(self):
        from litellm.types.utils import Usage

        recovered_usage = Usage(prompt_tokens=30, completion_tokens=1, total_tokens=31)
        logging_obj = MagicMock()
        logging_obj.model_call_details = {
            "combined_usage_object": recovered_usage,
            "response_cost": 3.5e-05,
        }
        request_data = {"litellm_logging_obj": logging_obj, "metadata": {}}
        await self._run(request_data)

        assert request_data["combined_usage_object"] is recovered_usage
        assert request_data["response_cost"] == 3.5e-05
        assert "litellm_logging_obj" not in request_data

    @pytest.mark.asyncio
    async def test_no_recovered_usage_is_noop(self):
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}
        request_data = {"litellm_logging_obj": logging_obj, "metadata": {}}
        await self._run(request_data)
        assert "combined_usage_object" not in request_data
        assert "response_cost" not in request_data


from typing import cast

from litellm.proxy.utils import create_model_info_response
from litellm.types.utils import ModelInfo


def _fake_model_info(**fields: int) -> ModelInfo:
    return cast(ModelInfo, dict(fields))


def _raise_unmapped(model_id: str) -> ModelInfo:
    raise ValueError(f"This model isn't mapped yet: {model_id}")


def test_create_model_info_response_includes_max_tokens_from_lookup():
    response = create_model_info_response(
        model_id="some-model",
        provider="openai",
        llm_router=None,
        get_model_info=lambda _model: _fake_model_info(
            max_input_tokens=128000, max_output_tokens=16384
        ),
    )

    assert response["id"] == "some-model"
    assert response["object"] == "model"
    assert response["max_input_tokens"] == 128000
    assert response["max_output_tokens"] == 16384


def test_create_model_info_response_does_not_call_router_group_info():
    router = MagicMock()
    router.get_configured_token_limits.return_value = (None, None)

    response = create_model_info_response(
        model_id="some-model",
        provider="openai",
        llm_router=router,
        get_model_info=lambda _model: _fake_model_info(
            max_input_tokens=128000, max_output_tokens=16384
        ),
    )

    router.get_model_group_info.assert_not_called()
    assert response["max_input_tokens"] == 128000


def test_create_model_info_response_uses_deployment_limits_when_not_in_cost_map():
    router = MagicMock()
    router.get_configured_token_limits.return_value = (32000, 8000)

    response = create_model_info_response(
        model_id="my-custom-deployment",
        provider="openai",
        llm_router=router,
        get_model_info=_raise_unmapped,
    )

    router.get_model_group_info.assert_not_called()
    assert response["max_input_tokens"] == 32000
    assert response["max_output_tokens"] == 8000


def test_create_model_info_response_deployment_limits_override_cost_map():
    router = MagicMock()
    router.get_configured_token_limits.return_value = (200000, None)

    response = create_model_info_response(
        model_id="gpt-4o",
        provider="openai",
        llm_router=router,
        get_model_info=lambda _model: _fake_model_info(
            max_input_tokens=128000, max_output_tokens=16384
        ),
    )

    assert response["max_input_tokens"] == 200000
    assert response["max_output_tokens"] == 16384


def test_create_model_info_response_survives_malformed_configured_limits():
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "bad-limit-model",
                "litellm_params": {"model": "openai/some-unmapped-model"},
                "model_info": {"max_input_tokens": "128,000"},
            }
        ]
    )

    response = create_model_info_response(
        model_id="bad-limit-model",
        provider="openai",
        llm_router=router,
        get_model_info=_raise_unmapped,
    )

    assert response["id"] == "bad-limit-model"
    assert "max_input_tokens" not in response
    assert "max_output_tokens" not in response


def test_create_model_info_response_emits_integer_token_counts():
    response = create_model_info_response(
        model_id="some-model",
        provider="openai",
        llm_router=None,
        get_model_info=lambda _model: _fake_model_info(
            max_input_tokens=128000, max_output_tokens=16384
        ),
    )

    assert isinstance(response["max_input_tokens"], int)
    assert isinstance(response["max_output_tokens"], int)


def test_create_model_info_response_omits_unknown_individual_limit():
    response = create_model_info_response(
        model_id="some-embedding",
        provider="openai",
        llm_router=None,
        get_model_info=lambda _model: _fake_model_info(max_input_tokens=8191),
    )

    assert response["max_input_tokens"] == 8191
    assert "max_output_tokens" not in response


def test_create_model_info_response_omits_limits_when_lookup_raises():
    response = create_model_info_response(
        model_id="openai/*",
        provider="openai",
        llm_router=None,
        get_model_info=_raise_unmapped,
    )

    assert response["id"] == "openai/*"
    assert "max_input_tokens" not in response
    assert "max_output_tokens" not in response


def test_create_model_info_response_no_router_keeps_base_fields():
    response = create_model_info_response(
        model_id="totally-unknown-model-xyz",
        provider="openai",
        llm_router=None,
        get_model_info=_raise_unmapped,
    )

    assert response == {
        "id": "totally-unknown-model-xyz",
        "object": "model",
        "created": response["created"],
        "owned_by": "openai",
    }


def test_create_model_info_response_reads_real_cost_map():
    response = create_model_info_response(
        model_id="gpt-4o", provider="openai", llm_router=None
    )

    assert isinstance(response["max_input_tokens"], int)
    assert response["max_input_tokens"] > 0
    assert isinstance(response["max_output_tokens"], int)
    assert response["max_output_tokens"] > 0


class TestPostCallFailureHookLLMExceptionAlerting:
    """The llm_exceptions alert is for infra / LLM-API failures, not user
    errors (https://github.com/BerriAI/litellm/issues/3395). Already-normalized
    client errors must be excluded so a guardrail content-policy block never
    pages on-call. ProxyException is such an error; before LIT-3751 only
    HTTPException was excluded, so AIM blocks paged as if the LLM API failed."""

    async def _alerted(self, exc) -> bool:
        import asyncio
        from unittest.mock import AsyncMock

        from litellm.proxy._types import AlertType, UserAPIKeyAuth

        proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging_obj.alert_types = [AlertType.llm_exceptions]
        alerting_handler = AsyncMock()
        with (
            patch.object(proxy_logging_obj, "update_request_status", new=AsyncMock()),
            patch.object(proxy_logging_obj, "alerting_handler", new=alerting_handler),
        ):
            await proxy_logging_obj.post_call_failure_hook(
                request_data={},
                original_exception=exc,
                user_api_key_dict=UserAPIKeyAuth(),
            )
        await asyncio.sleep(0)  # let the fire-and-forget alert task run
        return alerting_handler.called

    @pytest.mark.asyncio
    async def test_proxy_exception_does_not_alert(self):
        from litellm.proxy._types import ProxyException

        exc = ProxyException(
            message="content blocked",
            type="invalid_request_error",
            param=None,
            code=400,
            openai_code="content_policy_violation",
        )
        assert await self._alerted(exc) is False

    @pytest.mark.asyncio
    async def test_http_exception_does_not_alert(self):
        assert (
            await self._alerted(HTTPException(status_code=400, detail="blocked"))
            is False
        )

    @pytest.mark.asyncio
    async def test_genuine_llm_api_error_still_alerts(self):
        assert await self._alerted(Exception("upstream 503")) is True


class TestPostCallFailureHookProxyExceptionLogging:
    """A guardrail block raises a ProxyException; on an LLM route it must still
    drive proxy-only failure logging (_handle_logging_proxy_only_error) so the
    blocked request is recorded, exactly as the old HTTPException did. Before
    LIT-3751 the classifier only matched HTTPException, so switching AIM to
    ProxyException silently dropped the rejected prompt from failure logs."""

    async def _logged(self, exc, *, request_route) -> bool:
        from unittest.mock import AsyncMock

        from litellm.proxy._types import UserAPIKeyAuth

        proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging_obj.alert_types = []
        handle_mock = AsyncMock()
        with (
            patch.object(proxy_logging_obj, "update_request_status", new=AsyncMock()),
            patch.object(
                proxy_logging_obj,
                "_handle_logging_proxy_only_error",
                new=handle_mock,
            ),
        ):
            await proxy_logging_obj.post_call_failure_hook(
                request_data={},
                original_exception=exc,
                user_api_key_dict=UserAPIKeyAuth(
                    api_key="sk-test", request_route=request_route
                ),
            )
        return handle_mock.await_count > 0

    def _block(self):
        from litellm.proxy._types import ProxyException

        return ProxyException(
            message="content blocked",
            type="invalid_request_error",
            param=None,
            code=400,
            openai_code="content_policy_violation",
        )

    @pytest.mark.asyncio
    async def test_proxy_exception_on_llm_route_is_logged(self):
        assert (
            await self._logged(self._block(), request_route="/v1/chat/completions")
            is True
        )

    @pytest.mark.asyncio
    async def test_generic_exception_on_llm_route_is_not_logged(self):
        # A raw provider/unknown exception is logged by the LLM call path, not here.
        assert (
            await self._logged(
                Exception("upstream 503"), request_route="/v1/chat/completions"
            )
            is False
        )


class TestShouldUseSmtpSsl:
    def test_port_465_uses_ssl(self, monkeypatch):
        from litellm.proxy.utils import _should_use_smtp_ssl

        monkeypatch.delenv("SMTP_USE_SSL", raising=False)
        assert _should_use_smtp_ssl(smtp_port=465) is True

    def test_smtp_use_ssl_env_var_forces_ssl_on_any_port(self, monkeypatch):
        from litellm.proxy.utils import _should_use_smtp_ssl

        monkeypatch.setenv("SMTP_USE_SSL", "True")
        assert _should_use_smtp_ssl(smtp_port=2465) is True

    def test_port_587_uses_plain_smtp(self, monkeypatch):
        from litellm.proxy.utils import _should_use_smtp_ssl

        monkeypatch.delenv("SMTP_USE_SSL", raising=False)
        assert _should_use_smtp_ssl(smtp_port=587) is False


class TestCreateSmtpConnection:
    def test_port_465_creates_smtp_ssl_with_verified_context(self, monkeypatch):
        import ssl

        from litellm.proxy.utils import _create_smtp_connection

        monkeypatch.delenv("SMTP_USE_SSL", raising=False)
        with (
            patch("smtplib.SMTP_SSL") as mock_smtp_ssl,
            patch("smtplib.SMTP") as mock_smtp,
        ):
            result = _create_smtp_connection(
                smtp_host="mail.example.com", smtp_port=465
            )

        mock_smtp.assert_not_called()
        assert result is mock_smtp_ssl.return_value
        _, kwargs = mock_smtp_ssl.call_args
        assert kwargs["host"] == "mail.example.com"
        assert kwargs["port"] == 465
        context = kwargs["context"]
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_port_587_creates_plain_smtp(self, monkeypatch):
        from litellm.proxy.utils import _create_smtp_connection

        monkeypatch.delenv("SMTP_USE_SSL", raising=False)
        with (
            patch("smtplib.SMTP_SSL") as mock_smtp_ssl,
            patch("smtplib.SMTP") as mock_smtp,
        ):
            result = _create_smtp_connection(
                smtp_host="mail.example.com", smtp_port=587
            )

        mock_smtp_ssl.assert_not_called()
        assert result is mock_smtp.return_value
        mock_smtp.assert_called_once_with(host="mail.example.com", port=587)


class TestSendEmailStartTls:
    @pytest.mark.asyncio
    async def test_starttls_uses_verified_context(self, monkeypatch):
        import ssl

        from litellm.proxy.utils import send_email

        monkeypatch.setenv("SMTP_HOST", "mail.example.com")
        monkeypatch.setenv("SMTP_PORT", "587")
        monkeypatch.setenv("SMTP_SENDER_EMAIL", "sender@example.com")
        monkeypatch.delenv("SMTP_TLS", raising=False)
        monkeypatch.delenv("SMTP_USE_SSL", raising=False)

        mock_server = MagicMock(spec=smtplib.SMTP)
        with patch(
            "litellm.proxy.utils._create_smtp_connection"
        ) as mock_create_connection:
            mock_create_connection.return_value.__enter__.return_value = mock_server
            await send_email(
                receiver_email="receiver@example.com",
                subject="test",
                html="<p>test</p>",
            )

        _, kwargs = mock_server.starttls.call_args
        context = kwargs["context"]
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True


@pytest.mark.asyncio
async def test_during_call_hook_monitor_only_audit_does_not_reach_the_live_dict():
    """Pins the ACCEPTED trade-off of the monitor-only detach (ARC-BUG-20 / SRE-3691).

    The snapshot that isolates ``guardrail_to_apply`` and the post-LLM-call mutations
    also isolates ``metadata``, where guardrails append
    ``standard_logging_guardrail_information``. The spend log reads that list off the
    LIVE dict, so a monitor-only during_call scan's audit record never reaches it.

    Inherent to fire-and-forget, not an addressing bug: the spend-log row is built
    after the LLM call returns, and a scan we deliberately do not wait for has
    produced nothing by then. Re-pointing the list at the live one was tried and
    rejected — it turns a deterministic, documentable loss into a race that loses the
    record precisely when scans are slow.

    Accepted because during_call adds no unique coverage: pre_call already scans and
    audits the same request-side content, post_call audits the response.

    If someone later makes monitor-only audits land on the live dict, this fails and
    they must confirm they fixed the TIMING too, not just the address.
    """
    import asyncio

    import litellm

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())

    class _AuditWritingGuardrail(_StubMonitorOnlyGuardrail):
        def should_run_guardrail(self, data, event_type):
            return True

        async def async_moderation_hook(self, data, user_api_key_dict, call_type):
            self.moderation_started = True
            # Write the audit record the way a real guardrail does: into whichever
            # metadata dict this task was handed.
            metadata = data.setdefault("metadata", {})
            metadata.setdefault("standard_logging_guardrail_information", []).append(
                {"guardrail_name": self.guardrail_name, "guardrail_status": "success"}
            )
            self.moderation_completed = True
            return data

    callback = _make_callback_instance_of_custom_guardrail(_AuditWritingGuardrail(delay=0.0))
    original_callbacks = litellm.callbacks
    litellm.callbacks = [callback]
    try:
        data = {"model": "gpt-4o-mini", "metadata": {"user_api_key": "sk-test"}}
        live_metadata = data["metadata"]

        await proxy_logging_obj.during_call_hook(
            data=data, user_api_key_dict=None, call_type="acompletion"
        )
        # Give the detached task every chance to finish before we look.
        for _ in range(50):
            await asyncio.sleep(0)
            if callback.moderation_completed:
                break
        await asyncio.sleep(0.05)

        assert callback.moderation_completed, "the scan must actually have run"
        assert "standard_logging_guardrail_information" not in live_metadata
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_during_mcp_call_is_never_detached_even_when_monitor_only():
    """MCP is deliberately excluded from the detach (ARC-BUG-20).

    ``during_mcp_call`` is the only audit of the tool arguments actually sent to an MCP
    server: ``pre_mcp_call`` runs before the rewrite, and the during-hook task is built
    from the rewritten arguments. Detaching it would leave a tool invocation with no
    Arcadia-side audit trail — the same record feeds the S3 LLM logs that HITRUST
    evidence depends on, not just the database.

    The hook stays awaited and is bounded by ARC-BUG-43's scan deadline instead, which is
    affordable because an MCP tool call is a discrete request/response rather than a token
    stream.
    """
    import asyncio

    import litellm

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())

    class _AuditWritingGuardrail(_StubMonitorOnlyGuardrail):
        def should_run_guardrail(self, data, event_type):
            return True

        async def async_moderation_hook(self, data, user_api_key_dict, call_type):
            self.moderation_started = True
            data.setdefault("metadata", {}).setdefault(
                "standard_logging_guardrail_information", []
            ).append({"guardrail_name": self.guardrail_name, "guardrail_status": "success"})
            self.moderation_completed = True
            return data

    callback = _make_callback_instance_of_custom_guardrail(_AuditWritingGuardrail(delay=0.0))
    original_callbacks = litellm.callbacks
    litellm.callbacks = [callback]
    try:
        data = {"model": "gpt-4o-mini", "metadata": {"user_api_key": "sk-test"}}
        live_metadata = data["metadata"]

        # call_mcp_tool routes the hook to during_mcp_call
        await proxy_logging_obj.during_call_hook(
            data=data, user_api_key_dict=None, call_type=CallTypes.call_mcp_tool.value
        )

        # Awaited, not detached: the record is already on the LIVE dict when the hook
        # returns, with no need to wait for a background task.
        assert callback.moderation_completed is True
        entries = live_metadata.get("standard_logging_guardrail_information")
        assert entries and entries[0]["guardrail_name"] == callback.guardrail_name
    finally:
        litellm.callbacks = original_callbacks


def test_detach_is_allowed_for_event_excludes_only_mcp():
    from litellm.types.guardrails import GuardrailEventHooks

    assert (
        ProxyLogging._detach_is_allowed_for_event(
            event_type=GuardrailEventHooks.during_call, guardrail_name="noma-during-call"
        )
        is True
    )
    assert (
        ProxyLogging._detach_is_allowed_for_event(
            event_type=GuardrailEventHooks.during_mcp_call, guardrail_name="noma-during-mcp-call"
        )
        is False
    )


@pytest.mark.asyncio
async def test_shutdown_drains_spend_buffers_before_monitor_tasks():
    """Pins the ORDER of the two shutdown drains.

    The ARC-BUG-20 replay conflicted with the base in exactly one place —
    proxy_shutdown_event, where both sides append a drain — and the resolution kept both.
    That resolution was previously unpinned: swapping the two lines left the whole suite
    green, so the one thing the merge could plausibly get wrong had no guard.

    Spend buffers must flush FIRST. A monitor task cancelled by the later drain cannot
    contribute spend rows (its record goes to the request snapshot), so flushing first
    loses nothing; the reverse order would leave any row the monitor drain produced sitting
    in the buffer past the only flush, with disconnect() next.
    """
    import litellm.proxy.proxy_server as proxy_server_module

    order: list = []

    async def _fake_spend_drain():
        order.append("spend")

    class _FakeProxyLogging:
        async def drain_pending_monitor_tasks(self, timeout: float = 5.0):
            order.append("monitor")

    with patch.object(proxy_server_module, "_drain_spend_buffers_on_shutdown", _fake_spend_drain):
        await proxy_server_module._drain_spend_buffers_on_shutdown()
        await _FakeProxyLogging().drain_pending_monitor_tasks(timeout=5.0)

    assert order == ["spend", "monitor"]

    # And the source itself still calls them in that order — the probe above only proves
    # the assertion works, this proves the shipped code matches it.
    import inspect

    src = inspect.getsource(proxy_server_module.proxy_shutdown_event)
    spend_at = src.index("_drain_spend_buffers_on_shutdown()")
    monitor_at = src.index("drain_pending_monitor_tasks(")
    assert spend_at < monitor_at, "spend buffers must be flushed before the monitor drain"


@pytest.mark.asyncio
async def test_run_guardrail_with_metrics_does_not_report_cancelled_as_success():
    """A cancelled guardrail must not be counted as one that passed.

    CancelledError is a BaseException, so it skips `except Exception` and the `finally`
    would emit the initialised status="success". ARC-BUG-43 fixed this inside the Noma
    guardrail; the same shape lived one layer up here, where it is reached whenever the
    shutdown drain cancels an in-flight monitor task — i.e. on every pod recycle.
    """
    import asyncio

    emitted: list = []

    def _capture(**kwargs):
        emitted.append(kwargs)

    async def _hang():
        await asyncio.sleep(30)

    callback = _StubMonitorOnlyGuardrail(guardrail_name="noma-cancel-probe")

    with patch.object(ProxyLogging, "_emit_guardrail_metrics", staticmethod(_capture)):
        task = asyncio.create_task(
            ProxyLogging._run_guardrail_with_metrics(callback, _hang(), "during_call")
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert emitted, "the finally block must still emit a metric"
    assert emitted[-1]["status"] != "success"
    assert emitted[-1]["error_type"] == "CancelledError"


@pytest.mark.asyncio
async def test_run_guardrail_with_metrics_does_not_report_fail_open_as_success():
    """A fail-open guardrail that never reached the vendor must not be counted as a pass.

    Third instance of the defect the cancelled-status test above fixed, and the same root cause:
    this wrapper derives `status` purely from what escapes the coroutine, so it can only ever learn
    about failures that raise. `block_failures: false` makes a guardrail catch its OWN failure and
    return the inputs unchanged — nothing escapes, and a response that was never scanned is
    recorded as `status="success"`.

    Measured on dev-ai: 9 log lines reading "content was NOT scanned" against
    `litellm_guardrail_requests_total` carrying ONLY `status="success"`. The correct status was
    computed all along and reached the audit record — it just never reached Prometheus. The fix
    publishes it through a ContextVar so the wrapper is told rather than left to infer.
    """
    from litellm.integrations.custom_guardrail import _LAST_GUARDRAIL_STATUS

    emitted: list = []

    async def _fails_open():
        # Exactly what apply_guardrail does under block_failures=False: report, then return.
        _LAST_GUARDRAIL_STATUS.set("guardrail_failed_to_respond")
        return {"texts": ["unchanged"]}

    callback = _StubMonitorOnlyGuardrail(guardrail_name="noma-fail-open-probe")

    with patch.object(ProxyLogging, "_emit_guardrail_metrics", staticmethod(lambda **kw: emitted.append(kw))):
        out = await ProxyLogging._run_guardrail_with_metrics(callback, _fails_open(), "post_call")

    assert out == {"texts": ["unchanged"]}, "fail-open must still return the inputs untouched"
    assert emitted, "the finally block must still emit a metric"
    assert emitted[-1]["status"] == "guardrail_failed_to_respond"
    assert emitted[-1]["error_type"] == "guardrail_failed_to_respond"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reported, expected",
    [
        ("success", "success"),
        (None, "success"),
    ],
    ids=["guardrail-reports-success", "guardrail-reports-nothing"],
)
async def test_run_guardrail_with_metrics_leaves_the_working_paths_alone(reported, expected):
    """The fix must not relabel anything that was already correct.

    A guardrail that succeeds, and one that never calls the audit hook at all (most non-Noma
    guardrails), both have to keep reading `success` — otherwise every clean scan in the fleet
    starts reporting a failure.
    """
    from litellm.integrations.custom_guardrail import _LAST_GUARDRAIL_STATUS

    emitted: list = []

    async def _ok():
        if reported is not None:
            _LAST_GUARDRAIL_STATUS.set(reported)
        return {"texts": ["scanned"]}

    callback = _StubMonitorOnlyGuardrail(guardrail_name="noma-clean-probe")
    with patch.object(ProxyLogging, "_emit_guardrail_metrics", staticmethod(lambda **kw: emitted.append(kw))):
        await ProxyLogging._run_guardrail_with_metrics(callback, _ok(), "pre_call")

    assert emitted[-1]["status"] == expected
    assert emitted[-1]["error_type"] is None


@pytest.mark.asyncio
async def test_an_escaping_exception_outranks_the_self_reported_status():
    """When a guardrail both reports a failure AND raises, the exception is the better signal.

    `error` carries the exception class in `error_type`, which is what an operator needs to tell a
    Noma outage from a bug in our own code. Letting the self-reported string win would flatten every
    such case into one label.
    """
    from litellm.integrations.custom_guardrail import _LAST_GUARDRAIL_STATUS

    emitted: list = []

    async def _reports_then_raises():
        _LAST_GUARDRAIL_STATUS.set("guardrail_failed_to_respond")
        raise RuntimeError("boom")

    callback = _StubMonitorOnlyGuardrail(guardrail_name="noma-raise-probe")
    with patch.object(ProxyLogging, "_emit_guardrail_metrics", staticmethod(lambda **kw: emitted.append(kw))):
        with pytest.raises(RuntimeError):
            await ProxyLogging._run_guardrail_with_metrics(callback, _reports_then_raises(), "post_call")

    assert emitted[-1]["status"] == "error"
    # 🔴 error_type is the load-bearing assertion. status would read "error" either way in some
    # orderings; it is `error_type` that a self-reported string overwrites, and an operator needs the
    # exception CLASS here to tell a Noma outage from a bug in our own code.
    assert emitted[-1]["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_one_hooks_failure_does_not_bleed_into_the_next():
    """pre_call, during_call and post_call run inside ONE request context.

    Without a reset at the top of the wrapper, a clean post_call would read pre_call's failure and
    report a scan that succeeded as one that failed — the mirror image of the bug being fixed, and
    exactly the mistake already recorded inside noma_v2.apply_guardrail for its own ContextVar.
    """
    from litellm.integrations.custom_guardrail import _LAST_GUARDRAIL_STATUS

    emitted: list = []

    async def _fails_open():
        _LAST_GUARDRAIL_STATUS.set("guardrail_failed_to_respond")
        return {"texts": ["x"]}

    async def _never_reports():
        return {"texts": ["x"]}

    callback = _StubMonitorOnlyGuardrail(guardrail_name="noma-bleed-probe")
    with patch.object(ProxyLogging, "_emit_guardrail_metrics", staticmethod(lambda **kw: emitted.append(kw))):
        await ProxyLogging._run_guardrail_with_metrics(callback, _fails_open(), "pre_call")
        # 🔴 Set it OUT OF BAND before the second call. A negative control caught the first version
        # of this test passing with the reset deleted: `_fails_open` sets the var inside its own
        # coroutine, and a bare `await` runs that in the caller's context, so the write was already
        # visible — the reset had nothing to prove. Writing it here reproduces what actually happens
        # in the proxy, where pre_call's value survives into post_call's wrapper invocation.
        _LAST_GUARDRAIL_STATUS.set("guardrail_failed_to_respond")
        await ProxyLogging._run_guardrail_with_metrics(callback, _never_reports(), "post_call")

    assert emitted[0]["status"] == "guardrail_failed_to_respond"
    assert emitted[1]["status"] == "success", "post_call inherited pre_call's failure"


@pytest.mark.asyncio
async def test_the_real_noma_guardrail_reports_an_unscanned_response():
    """End to end through the shipped guardrail, not a stand-in.

    The stub tests above all set the ContextVar by hand, so they would pass even if
    `add_standard_logging_guardrail_information_to_request_data` never published it. This drives the
    real NomaV2Guardrail with a dead scan and asserts both halves at once: fail-open still returns
    the inputs, and the metric no longer says success.
    """
    from litellm.proxy.guardrails.guardrail_hooks.noma.noma_v2 import NomaV2Guardrail

    emitted: list = []
    guardrail = NomaV2Guardrail(
        guardrail_name="noma-post-call",
        api_key="probe-key",
        api_base="https://noma.invalid",
        monitor_mode=True,
        block_failures=False,
    )

    async def _dead_connection(*args, **kwargs):
        raise TimeoutError("connection died")

    guardrail._call_noma_scan = _dead_connection  # type: ignore[method-assign]

    inputs = {"texts": ["some content"]}
    with patch.object(ProxyLogging, "_emit_guardrail_metrics", staticmethod(lambda **kw: emitted.append(kw))):
        out = await ProxyLogging._run_guardrail_with_metrics(
            guardrail,
            guardrail.apply_guardrail(inputs=inputs, request_data={"metadata": {}}, input_type="response"),
            "post_call",
        )

    assert out == inputs, "block_failures=False must keep failing open"
    assert emitted[-1]["status"] != "success", "an unscanned response must not read as a success"
