import time
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

import litellm

from litellm.constants import HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.noma import noma_v2 as nv2
from litellm.proxy.guardrails.guardrail_hooks.noma import (
    NomaV2Guardrail,
    initialize_guardrail,
    initialize_guardrail_v2,
)
from litellm.proxy.guardrails.guardrail_hooks.noma.noma import NomaBlockedMessage
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.types.guardrails import LitellmParams
from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)
from litellm.types.proxy.guardrails.guardrail_hooks.noma import (
    NomaV2GuardrailConfigModel,
)


@pytest.fixture
def noma_v2_guardrail():
    return NomaV2Guardrail(
        api_key="test-api-key",
        api_base="https://api.test.noma.security/",
        application_id="test-app",
        monitor_mode=False,
        block_failures=False,
        guardrail_name="test-noma-v2-guardrail",
        event_hook="pre_call",
        default_on=True,
    )


class TestNomaV2Configuration:
    @pytest.mark.asyncio
    async def test_provider_specific_params_include_noma_v2_fields(self):
        from litellm.proxy.guardrails.guardrail_endpoints import (
            get_provider_specific_params,
        )

        provider_params = await get_provider_specific_params()
        assert "noma_v2" in provider_params

        noma_v2_params = provider_params["noma_v2"]
        assert noma_v2_params["ui_friendly_name"] == "Noma Security v2"
        assert "api_key" in noma_v2_params
        assert "api_base" in noma_v2_params
        assert "application_id" in noma_v2_params
        assert "monitor_mode" in noma_v2_params
        assert "block_failures" in noma_v2_params
        assert "streaming_end_of_stream_only" in noma_v2_params
        assert "streaming_sampling_rate" in noma_v2_params

    def test_init_requires_auth_for_saas_endpoint(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError,
                match="requires api_key when using Noma SaaS endpoint",
            ):
                NomaV2Guardrail()

    def test_init_allows_missing_auth_for_self_managed_endpoint(self):
        with patch.dict(os.environ, {}, clear=True):
            guardrail = NomaV2Guardrail(api_base="https://self-managed.noma.local")
        assert guardrail.api_key is None

    def test_init_defaults_monitor_and_block_failures(self):
        with patch.dict(os.environ, {"NOMA_API_KEY": "test-api-key"}, clear=True):
            guardrail = NomaV2Guardrail()

        assert guardrail.monitor_mode is False
        assert guardrail.block_failures is True

    @pytest.mark.asyncio
    async def test_api_key_auth_path(self, noma_v2_guardrail):
        assert noma_v2_guardrail._get_authorization_header() == "Bearer test-api-key"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"action":"NONE"}'
        mock_response.json.return_value = {
            "action": "NONE",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)

        with patch.object(noma_v2_guardrail.async_handler, "post", mock_post):
            await noma_v2_guardrail._call_noma_scan(
                payload={"inputs": {"texts": []}},
            )

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-api-key"

    @pytest.mark.asyncio
    async def test_self_managed_path_without_api_key_omits_authorization_header(self):
        guardrail = NomaV2Guardrail(
            api_base="https://self-managed.noma.local",
            guardrail_name="test-noma-v2-guardrail",
            event_hook="pre_call",
            default_on=True,
        )
        assert guardrail._get_authorization_header() == ""

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"action":"NONE"}'
        mock_response.json.return_value = {"action": "NONE"}
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)

        with patch.object(guardrail.async_handler, "post", mock_post):
            await guardrail._call_noma_scan(payload={"inputs": {"texts": []}})

        sent_headers = mock_post.call_args.kwargs["headers"]
        assert "Authorization" not in sent_headers

    def test_build_scan_payload_sends_raw_available_data(self, noma_v2_guardrail):
        inputs = {
            "texts": ["hello"],
            "images": ["https://example.com/image.png"],
            "structured_messages": [{"role": "user", "content": "hello"}],
            "tool_calls": [{"id": "tool-1"}],
            "model": "gpt-4o-mini",
        }
        request_data = {
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"headers": {"x-noma-application-id": "header-app"}},
            "litellm_metadata": {"user_api_key_alias": "litellm-alias"},
            "litellm_call_id": "call-id-1",
        }
        payload = noma_v2_guardrail._build_scan_payload(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
            logging_obj=None,
            application_id="dynamic-app",
        )

        assert payload["inputs"] == inputs
        assert payload["request_data"] == request_data
        assert payload["input_type"] == "request"
        assert payload["monitor_mode"] is False
        assert payload["application_id"] == "dynamic-app"
        assert "dynamic_params" not in payload
        assert "x-noma-context" not in payload
        assert "input" not in payload

    def test_build_scan_payload_deep_copies_request_data(self, noma_v2_guardrail):
        request_data = {
            "metadata": {"headers": {"x-noma-application-id": "header-app"}},
            "messages": [{"role": "user", "content": "hello"}],
        }
        payload = noma_v2_guardrail._build_scan_payload(
            inputs={"texts": ["hello"]},
            request_data=request_data,
            input_type="request",
            logging_obj=None,
            application_id="dynamic-app",
        )

        payload["request_data"]["metadata"]["headers"]["x-noma-application-id"] = "mutated-value"
        payload["request_data"]["messages"][0]["content"] = "changed-content"

        assert request_data["metadata"]["headers"]["x-noma-application-id"] == "header-app"
        assert request_data["messages"][0]["content"] == "hello"

    def test_build_scan_payload_snapshots_model_call_details(self, noma_v2_guardrail):
        class _LoggingObj:
            def __init__(self, details):
                self.model_call_details = details

        details = {"model": "gpt-4", "call_id": "abc"}
        logging_obj = _LoggingObj(details)

        payload = noma_v2_guardrail._build_scan_payload(
            inputs={"texts": ["hello"]},
            request_data={"messages": [{"role": "user", "content": "hello"}]},
            input_type="request",
            logging_obj=logging_obj,
            application_id=None,
        )

        embedded = payload["request_data"]["litellm_logging_obj"]
        assert embedded == details
        assert embedded is not details

        details["inserted_by_async_handler"] = "late"
        assert "inserted_by_async_handler" not in embedded

    def test_build_scan_payload_tolerates_non_dict_model_call_details(self, noma_v2_guardrail):
        class _LoggingObj:
            model_call_details = None

        payload = noma_v2_guardrail._build_scan_payload(
            inputs={"texts": ["hello"]},
            request_data={"messages": [{"role": "user", "content": "hello"}]},
            input_type="request",
            logging_obj=_LoggingObj(),
            application_id=None,
        )

        assert payload["request_data"]["litellm_logging_obj"] is None

    def test_build_scan_payload_survives_unpicklable_request_data(self, noma_v2_guardrail):
        """Regression test for NOM-8044: post_call / during_call / during_mcp_call
        used to 500 because request_data contained uvloop.Loop and similar
        C-extension objects whose __reduce__ raises, which crashed deepcopy."""

        class _FakeUvloopObject:
            def __reduce__(self):
                raise TypeError("no default __reduce__ due to non-trivial __cinit__")

            def __repr__(self) -> str:
                return "<fake-uvloop-loop>"

        unpicklable = _FakeUvloopObject()
        request_data = {
            "metadata": {"headers": {"x-noma-application-id": "header-app"}},
            "messages": [{"role": "user", "content": "hello"}],
            "event_loop": unpicklable,
        }

        payload = noma_v2_guardrail._build_scan_payload(
            inputs={"texts": ["hello"]},
            request_data=request_data,
            input_type="response",
            logging_obj=None,
            application_id="dynamic-app",
        )

        assert isinstance(payload["request_data"], dict)
        assert payload["request_data"]["event_loop"] == "<fake-uvloop-loop>"
        assert payload["request_data"]["messages"] == [{"role": "user", "content": "hello"}]

        # Original request_data must not have been mutated by the copy.
        assert request_data["event_loop"] is unpicklable

    def test_build_scan_payload_passes_model_call_details_as_is(self, noma_v2_guardrail):
        class _LoggingObj:
            def __init__(self) -> None:
                self.model_call_details = {
                    "model": "gpt-4.1-mini",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                    "call_type": "acompletion",
                    "litellm_call_id": "call-id-123",
                    "function_id": "fn-id-456",
                    "litellm_trace_id": "trace-id-789",
                    "api_key": "included-as-is",
                }

        request_data = {"litellm_logging_obj": "<Logging object>"}
        payload = noma_v2_guardrail._build_scan_payload(
            inputs={"texts": ["hello"]},
            request_data=request_data,
            input_type="request",
            logging_obj=_LoggingObj(),
            application_id="test-app",
        )

        assert payload["request_data"]["litellm_logging_obj"] == {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "call_type": "acompletion",
            "litellm_call_id": "call-id-123",
            "function_id": "fn-id-456",
            "litellm_trace_id": "trace-id-789",
            "api_key": "included-as-is",
        }
        assert "logging_obj" not in payload
        assert request_data["litellm_logging_obj"] == "<Logging object>"

    @pytest.mark.asyncio
    async def test_call_noma_scan_sanitizes_response_model_dump_object(self, noma_v2_guardrail):
        import json

        class _FakeModelResponse:
            def model_dump(self):
                return {"id": "resp-1", "content": "ok"}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"action":"NONE"}'
        mock_response.json.return_value = {"action": "NONE"}
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)

        payload = {
            "inputs": {"texts": ["hello"]},
            "request_data": {"response": _FakeModelResponse()},
            "input_type": "response",
            "application_id": "test-app",
        }

        with patch.object(noma_v2_guardrail.async_handler, "post", mock_post):
            await noma_v2_guardrail._call_noma_scan(payload=payload)

        sent_payload = mock_post.call_args.kwargs["json"]
        json.dumps(sent_payload)
        assert sent_payload["request_data"]["response"]["id"] == "resp-1"

    def test_sanitize_payload_for_transport_falls_back_to_safe_dumps(self, noma_v2_guardrail):
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.noma.noma_v2.json.dumps",
            side_effect=TypeError("cannot serialize"),
        ):
            with patch(
                "litellm.proxy.guardrails.guardrail_hooks.noma.noma_v2.safe_dumps",
                return_value='{"fallback": true}',
            ) as mock_safe_dumps:
                sanitized = noma_v2_guardrail._sanitize_payload_for_transport({"inputs": {"texts": ["hello"]}})

        mock_safe_dumps.assert_called_once()
        assert sanitized == {"fallback": True}

    def test_sanitize_payload_for_transport_logs_warning_when_payload_becomes_empty(self, noma_v2_guardrail):
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.noma.noma_v2.safe_json_loads",
            return_value={},
        ):
            with patch(
                "litellm.proxy.guardrails.guardrail_hooks.noma.noma_v2.verbose_proxy_logger.warning"
            ) as mock_warning:
                sanitized = noma_v2_guardrail._sanitize_payload_for_transport({"inputs": {"texts": ["hello"]}})

        assert sanitized == {}
        mock_warning.assert_called_once_with(
            "Noma v2 guardrail: payload serialization failed, falling back to empty payload"
        )

    def test_sanitize_payload_for_transport_logs_warning_on_non_dict_output(self, noma_v2_guardrail):
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.noma.noma_v2.safe_json_loads",
            return_value=["not-a-dict"],
        ):
            with patch(
                "litellm.proxy.guardrails.guardrail_hooks.noma.noma_v2.verbose_proxy_logger.warning"
            ) as mock_warning:
                sanitized = noma_v2_guardrail._sanitize_payload_for_transport({"inputs": {"texts": ["hello"]}})

        assert sanitized == {}
        mock_warning.assert_called_once_with(
            "Noma v2 guardrail: payload sanitization produced non-dict output (type=%s), falling back to empty payload",
            "list",
        )

    def test_get_config_model_returns_noma_v2_config_model(self):
        assert NomaV2Guardrail.get_config_model() is NomaV2GuardrailConfigModel


class TestNomaV2ActionBehavior:
    def test_resolve_action_from_response_raises_on_unknown_action(self, noma_v2_guardrail):
        with pytest.raises(ValueError, match="missing valid action"):
            noma_v2_guardrail._resolve_action_from_response({"action": "INVALID"})

    @pytest.mark.asyncio
    async def test_native_action_none(self, noma_v2_guardrail):
        inputs = {"texts": ["hello"]}
        with patch.object(
            noma_v2_guardrail,
            "_call_noma_scan",
            AsyncMock(
                return_value={
                    "action": "NONE",
                }
            ),
        ):
            result = await noma_v2_guardrail.apply_guardrail(
                inputs=inputs,
                request_data={"metadata": {}},
                input_type="request",
            )

        assert result == inputs

    @pytest.mark.asyncio
    async def test_native_action_guardrail_intervened_updates_supported_fields(self, noma_v2_guardrail):
        inputs = {
            "texts": ["Name: Jane"],
            "images": ["https://old.example/image.png"],
            "tools": [{"type": "function", "function": {"name": "old_tool"}}],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "old_tool", "arguments": '{"key":"value"}'},
                }
            ],
        }
        with patch.object(
            noma_v2_guardrail,
            "_call_noma_scan",
            AsyncMock(
                return_value={
                    "action": "GUARDRAIL_INTERVENED",
                    "texts": ["Name: *******"],
                    "images": ["https://new.example/image.png"],
                    "tools": [{"type": "function", "function": {"name": "new_tool"}}],
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "new_tool",
                                "arguments": '{"safe":"true"}',
                            },
                        }
                    ],
                }
            ),
        ):
            result = await noma_v2_guardrail.apply_guardrail(
                inputs=inputs,
                request_data={"metadata": {}},
                input_type="request",
            )

        assert result["texts"] == ["Name: *******"]
        assert result["images"] == ["https://new.example/image.png"]
        assert result["tools"] == [{"type": "function", "function": {"name": "new_tool"}}]
        assert result["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "new_tool", "arguments": '{"safe":"true"}'},
            }
        ]

    @pytest.mark.asyncio
    async def test_native_action_blocked(self, noma_v2_guardrail):
        inputs = {"texts": ["bad"]}
        with patch.object(
            noma_v2_guardrail,
            "_call_noma_scan",
            AsyncMock(
                return_value={
                    "action": "BLOCKED",
                    "blocked_reason": "blocked by policy",
                }
            ),
        ):
            with pytest.raises(NomaBlockedMessage) as exc_info:
                await noma_v2_guardrail.apply_guardrail(
                    inputs=inputs,
                    request_data={"metadata": {}},
                    input_type="request",
                )
        assert exc_info.value.detail["details"]["blocked_reason"] == "blocked by policy"

    @pytest.mark.asyncio
    async def test_intervened_without_modifications_returns_original_inputs(self, noma_v2_guardrail):
        inputs = {"texts": ["Name: Jane"]}
        with patch.object(
            noma_v2_guardrail,
            "_call_noma_scan",
            AsyncMock(
                return_value={
                    "action": "GUARDRAIL_INTERVENED",
                }
            ),
        ):
            result = await noma_v2_guardrail.apply_guardrail(
                inputs=inputs,
                request_data={"metadata": {}},
                input_type="request",
            )
        assert result == inputs

    @pytest.mark.asyncio
    async def test_fail_open_on_technical_scan_failure(self, noma_v2_guardrail):
        inputs = {"texts": ["hello"]}
        with patch.object(
            noma_v2_guardrail,
            "_call_noma_scan",
            AsyncMock(side_effect=Exception("network error")),
        ):
            result = await noma_v2_guardrail.apply_guardrail(
                inputs=inputs,
                request_data={"metadata": {}},
                input_type="request",
            )

        assert result == inputs

    @pytest.mark.asyncio
    async def test_fail_closed_on_technical_scan_failure_when_block_failures_true(self):
        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            block_failures=True,
            guardrail_name="test-noma-v2-guardrail",
            event_hook="pre_call",
            default_on=True,
        )
        with patch.object(
            guardrail,
            "_call_noma_scan",
            AsyncMock(side_effect=Exception("network error")),
        ):
            with pytest.raises(Exception, match="network error"):
                await guardrail.apply_guardrail(
                    inputs={"texts": ["hello"]},
                    request_data={"metadata": {}},
                    input_type="request",
                )

    @pytest.mark.asyncio
    async def test_monitor_mode_ignores_block_action(self):
        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            monitor_mode=True,
            guardrail_name="test-noma-v2-guardrail",
            event_hook="pre_call",
            default_on=True,
        )
        call_mock = AsyncMock(return_value={"action": "BLOCKED"})
        with patch.object(guardrail, "_call_noma_scan", call_mock):
            result = await guardrail.apply_guardrail(
                inputs={"texts": ["hello"]},
                request_data={"metadata": {}},
                input_type="request",
            )

        payload = call_mock.call_args.kwargs["payload"]
        assert payload["monitor_mode"] is True
        assert result == {"texts": ["hello"]}


class TestNomaV2ApplicationIdResolution:
    @pytest.mark.asyncio
    async def test_apply_guardrail_uses_dynamic_application_id(self, noma_v2_guardrail):
        call_mock = AsyncMock(return_value={"action": "NONE"})
        with patch.object(
            noma_v2_guardrail,
            "get_guardrail_dynamic_request_body_params",
            return_value={"application_id": "dynamic-app"},
        ):
            with patch.object(noma_v2_guardrail, "_call_noma_scan", call_mock):
                await noma_v2_guardrail.apply_guardrail(
                    inputs={"texts": ["hello"]},
                    request_data={"metadata": {}},
                    input_type="request",
                )

        payload = call_mock.call_args.kwargs["payload"]
        assert payload["application_id"] == "dynamic-app"

    @pytest.mark.asyncio
    async def test_apply_guardrail_uses_configured_application_id(self, noma_v2_guardrail):
        call_mock = AsyncMock(return_value={"action": "NONE"})
        with patch.object(
            noma_v2_guardrail,
            "get_guardrail_dynamic_request_body_params",
            return_value={},
        ):
            with patch.object(noma_v2_guardrail, "_call_noma_scan", call_mock):
                await noma_v2_guardrail.apply_guardrail(
                    inputs={"texts": ["hello"]},
                    request_data={"metadata": {}},
                    input_type="request",
                )

        payload = call_mock.call_args.kwargs["payload"]
        assert payload["application_id"] == "test-app"

    @pytest.mark.asyncio
    async def test_apply_guardrail_falls_back_to_key_alias_from_litellm_metadata(self, noma_v2_guardrail):
        """When no explicit application_id is set, fall back to user_api_key_alias
        so that each API key gets its own application entry in the Noma dashboard."""
        noma_v2_guardrail.application_id = None
        call_mock = AsyncMock(return_value={"action": "NONE"})
        request_data = {
            "metadata": {},
            "litellm_metadata": {"user_api_key_alias": "test-key-alias"},
        }
        with patch.object(
            noma_v2_guardrail,
            "get_guardrail_dynamic_request_body_params",
            return_value={},
        ):
            with patch.object(noma_v2_guardrail, "_call_noma_scan", call_mock):
                await noma_v2_guardrail.apply_guardrail(
                    inputs={"texts": ["hello"]},
                    request_data=request_data,
                    input_type="request",
                )

        payload = call_mock.call_args.kwargs["payload"]
        assert payload["application_id"] == "test-key-alias"

    @pytest.mark.asyncio
    async def test_apply_guardrail_falls_back_to_key_alias_from_metadata(self, noma_v2_guardrail):
        """user_api_key_alias in metadata (set by proxy_server.py) is also resolved."""
        noma_v2_guardrail.application_id = None
        call_mock = AsyncMock(return_value={"action": "NONE"})
        request_data = {
            "metadata": {"user_api_key_alias": "test-service-key"},
        }
        with patch.object(
            noma_v2_guardrail,
            "get_guardrail_dynamic_request_body_params",
            return_value={},
        ):
            with patch.object(noma_v2_guardrail, "_call_noma_scan", call_mock):
                await noma_v2_guardrail.apply_guardrail(
                    inputs={"texts": ["hello"]},
                    request_data=request_data,
                    input_type="request",
                )

        payload = call_mock.call_args.kwargs["payload"]
        assert payload["application_id"] == "test-service-key"

    @pytest.mark.asyncio
    async def test_apply_guardrail_configured_application_id_takes_precedence_over_key_alias(self, noma_v2_guardrail):
        """Explicit application_id (config/env) wins over key_alias fallback."""
        call_mock = AsyncMock(return_value={"action": "NONE"})
        request_data = {
            "metadata": {"user_api_key_alias": "should-not-be-used"},
        }
        with patch.object(
            noma_v2_guardrail,
            "get_guardrail_dynamic_request_body_params",
            return_value={},
        ):
            with patch.object(noma_v2_guardrail, "_call_noma_scan", call_mock):
                await noma_v2_guardrail.apply_guardrail(
                    inputs={"texts": ["hello"]},
                    request_data=request_data,
                    input_type="request",
                )

        payload = call_mock.call_args.kwargs["payload"]
        assert payload["application_id"] == "test-app"

    @pytest.mark.asyncio
    async def test_apply_guardrail_omits_application_id_when_no_fallback_available(
        self,
    ):
        """When nothing is set — no config, no dynamic params, no key alias — omit entirely."""
        guardrail_no_config = NomaV2Guardrail(
            api_key="test-api-key",
            application_id=None,
            guardrail_name="test-noma-v2-guardrail",
            event_hook="pre_call",
            default_on=True,
        )
        call_mock = AsyncMock(return_value={"action": "NONE"})
        with patch.object(
            guardrail_no_config,
            "get_guardrail_dynamic_request_body_params",
            return_value={},
        ):
            with patch.object(guardrail_no_config, "_call_noma_scan", call_mock):
                await guardrail_no_config.apply_guardrail(
                    inputs={"texts": ["hello"]},
                    request_data={"metadata": {}},
                    input_type="request",
                )

        payload = call_mock.call_args.kwargs["payload"]
        assert "application_id" not in payload


def _streaming_guardrail(**streaming_kwargs):
    return NomaV2Guardrail(
        api_base="https://self-managed.noma.local",
        application_id="test-app",
        guardrail_name="test-noma-v2-streaming",
        event_hook="post_call",
        default_on=True,
        **streaming_kwargs,
    )


def _text_stream(chunk_count):
    async def _stream():
        for index in range(chunk_count):
            yield ModelResponseStream(
                model="gpt-4o",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(content=f"chunk-{index}"),
                        finish_reason="stop" if index == chunk_count - 1 else None,
                    )
                ],
            )

    return _stream()


async def _drain_through_unified_guardrail(guardrail, chunk_count, scanned_texts=None):
    assembled = ModelResponse(choices=[Choices(index=0, message=Message(role="assistant", content="assembled"))])
    scan_mock = AsyncMock(return_value={"action": "NONE"})
    if scanned_texts is not None:
        original_apply = guardrail.apply_guardrail

        async def _record(**kwargs):
            scanned_texts.append("".join(kwargs["inputs"].get("texts") or []))
            return await original_apply(**kwargs)

        guardrail.apply_guardrail = _record
    with (
        patch.object(guardrail, "_call_noma_scan", scan_mock),
        patch(
            "litellm.llms.openai.chat.guardrail_translation.handler.stream_chunk_builder",
            return_value=assembled,
        ),
    ):
        request_data = {
            "messages": [{"role": "user", "content": "hi"}],
            "guardrail_to_apply": guardrail,
            "metadata": {"guardrails": ["test-noma-v2-streaming"]},
        }
        async for _ in UnifiedLLMGuardrails().async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="test", request_route="/chat/completions"),
            response=_text_stream(chunk_count),
            request_data=request_data,
        ):
            pass
    return scan_mock


class TestNomaV2StreamingKnobs:
    def test_streaming_knobs_default_to_framework_defaults(self):
        guardrail = _streaming_guardrail()

        assert guardrail.streaming_end_of_stream_only is False
        assert guardrail.streaming_sampling_rate == 5

    def test_streaming_knobs_accept_explicit_values(self):
        guardrail = _streaming_guardrail(
            streaming_end_of_stream_only=True,
            streaming_sampling_rate=2,
        )

        assert guardrail.streaming_end_of_stream_only is True
        assert guardrail.streaming_sampling_rate == 2

    @pytest.mark.parametrize("invalid_rate", [0, -1, "0", "abc", 1.5, True, False])
    def test_streaming_sampling_rate_rejects_invalid_values(self, invalid_rate):
        with pytest.raises(ValueError, match="streaming_sampling_rate must be an integer >= 1"):
            _streaming_guardrail(streaming_sampling_rate=invalid_rate)

    @pytest.mark.parametrize("raw_rate, expected", [("3", 3), (3, 3), ("", 5), (None, 5)])
    def test_streaming_sampling_rate_coerces_config_values(self, raw_rate, expected):
        guardrail = _streaming_guardrail(streaming_sampling_rate=raw_rate)

        assert guardrail.streaming_sampling_rate == expected

    @pytest.mark.parametrize(
        "raw_flag, expected",
        [("true", True), ("True", True), ("false", False), ("no", False), (True, True), (False, False)],
    )
    def test_streaming_end_of_stream_only_coerces_config_values(self, raw_flag, expected):
        guardrail = _streaming_guardrail(streaming_end_of_stream_only=raw_flag)

        assert guardrail.streaming_end_of_stream_only is expected

    def test_initialize_guardrail_v2_forwards_streaming_knobs(self):
        litellm_params = LitellmParams(
            guardrail="noma_v2",
            mode="post_call",
            api_base="https://self-managed.noma.local",
            streaming_end_of_stream_only=True,
            streaming_sampling_rate=3,
        )

        with patch("litellm.logging_callback_manager.add_litellm_callback") as mock_add:
            guardrail = initialize_guardrail_v2(
                litellm_params=litellm_params,
                guardrail={"guardrail_name": "test-noma-v2-streaming"},
            )

        assert guardrail.streaming_end_of_stream_only is True
        assert guardrail.streaming_sampling_rate == 3
        mock_add.assert_called_once_with(guardrail)

    def test_initialize_guardrail_v2_reads_knobs_from_optional_params_model(self):
        litellm_params = LitellmParams(
            guardrail="noma_v2",
            mode="post_call",
            api_base="https://self-managed.noma.local",
            streaming_end_of_stream_only=False,
            streaming_sampling_rate=9,
            optional_params={
                "streaming_end_of_stream_only": True,
                "streaming_sampling_rate": 1,
            },
        )

        with patch("litellm.logging_callback_manager.add_litellm_callback"):
            guardrail = initialize_guardrail_v2(
                litellm_params=litellm_params,
                guardrail={"guardrail_name": "test-noma-v2-streaming"},
            )

        assert guardrail.streaming_end_of_stream_only is True
        assert guardrail.streaming_sampling_rate == 1

    def test_initialize_guardrail_v2_falls_back_to_top_level_when_optional_params_omits_knob(self):
        litellm_params = LitellmParams(
            guardrail="noma_v2",
            mode="post_call",
            api_base="https://self-managed.noma.local",
            streaming_sampling_rate=7,
        )
        litellm_params.optional_params = {"streaming_end_of_stream_only": True}

        with patch("litellm.logging_callback_manager.add_litellm_callback"):
            guardrail = initialize_guardrail_v2(
                litellm_params=litellm_params,
                guardrail={"guardrail_name": "test-noma-v2-streaming"},
            )

        assert guardrail.streaming_end_of_stream_only is True
        assert guardrail.streaming_sampling_rate == 7

    def test_use_v2_true_routes_through_v2_streaming_wiring(self):
        litellm_params = LitellmParams(
            guardrail="noma",
            mode="post_call",
            api_base="https://self-managed.noma.local",
            use_v2=True,
            streaming_sampling_rate=4,
        )

        with patch("litellm.logging_callback_manager.add_litellm_callback") as mock_add:
            guardrail = initialize_guardrail(
                litellm_params=litellm_params,
                guardrail={"guardrail_name": "test-noma-v2-streaming"},
            )

        assert isinstance(guardrail, NomaV2Guardrail)
        assert guardrail.streaming_sampling_rate == 4
        mock_add.assert_called_once_with(guardrail)

    @pytest.mark.asyncio
    async def test_end_of_stream_only_scans_assembled_response_without_partials(self):
        guardrail = _streaming_guardrail(streaming_end_of_stream_only=True)
        scanned_texts = []

        scan_mock = await _drain_through_unified_guardrail(guardrail, chunk_count=6, scanned_texts=scanned_texts)

        assert scan_mock.call_count == 1
        assert scanned_texts == ["assembled"]

    @pytest.mark.asyncio
    async def test_default_config_scans_a_mid_stream_partial(self):
        guardrail = _streaming_guardrail()
        scanned_texts = []

        await _drain_through_unified_guardrail(guardrail, chunk_count=6, scanned_texts=scanned_texts)

        assert scanned_texts == ["chunk-0chunk-1chunk-2chunk-3chunk-4", "assembled"]

    @pytest.mark.asyncio
    async def test_sampling_rate_controls_which_partials_are_scanned(self):
        guardrail = _streaming_guardrail(streaming_sampling_rate=2)
        scanned_texts = []

        scan_mock = await _drain_through_unified_guardrail(guardrail, chunk_count=5, scanned_texts=scanned_texts)

        assert scan_mock.call_count == 3
        assert scanned_texts == [
            "chunk-0chunk-1",
            "chunk-0chunk-1chunk-2chunk-3",
            "assembled",
        ]

    @pytest.mark.asyncio
    async def test_end_of_stream_only_makes_sampling_rate_irrelevant(self):
        guardrail = _streaming_guardrail(
            streaming_end_of_stream_only=True,
            streaming_sampling_rate=2,
        )
        scanned_texts = []

        await _drain_through_unified_guardrail(guardrail, chunk_count=4, scanned_texts=scanned_texts)

        assert scanned_texts == ["assembled"]


class TestNomaV2ScanTimeout:
    """[ARC-BUG-43] a single scan must carry an explicit deadline.

    Without one the POST inherits the shared httpx client default
    (COMPLETION_HTTP_FALLBACK_SECONDS, 600s). during_call_hook is gathered in
    parallel with the LLM call, so gather waits for the slowest member: a hung
    Noma holds the request open long after the model answered.
    """

    def test_scan_timeout_defaults_to_constant(self, noma_v2_guardrail):
        from litellm.constants import NOMA_SCAN_TIMEOUT_SECONDS

        assert noma_v2_guardrail.scan_timeout == NOMA_SCAN_TIMEOUT_SECONDS

    def test_scan_timeout_accepts_explicit_value(self):
        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            scan_timeout=2.5,
            guardrail_name="test-noma-v2-guardrail",
            event_hook="during_call",
        )
        assert guardrail.scan_timeout == 2.5

    def test_scan_timeout_coerces_config_string(self):
        # config values arrive uncoerced (extra="allow"), same as the streaming knobs
        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            scan_timeout="2.5",
            guardrail_name="test-noma-v2-guardrail",
            event_hook="during_call",
        )
        assert guardrail.scan_timeout == 2.5

    @pytest.mark.parametrize(
        "bad",
        [0, -1, True, "abc", "inf", "nan", "1e400", 0.05, 600, "600"],
        ids=[
            "zero",
            "negative",
            "bool",
            "non-numeric",
            "inf",
            "nan",
            "overflow-to-inf",
            "below-floor",
            "above-ceiling",
            "above-ceiling-str",
        ],
    )
    def test_scan_timeout_rejects_non_positive_non_finite_and_sub_floor(self, bad):
        # inf passes `> 0` and nan fails every comparison, so a bare `<= 0` guard
        # admits both; a sub-floor value fails every scan before it reaches Noma.
        with pytest.raises(ValueError, match="scan_timeout"):
            NomaV2Guardrail(
                api_key="test-api-key",
                api_base="https://api.test.noma.security/",
                scan_timeout=bad,
                guardrail_name="test-noma-v2-guardrail",
                event_hook="during_call",
            )

    @pytest.mark.asyncio
    async def test_scan_passes_timeout_to_http_post(self, noma_v2_guardrail):
        # The payload assertion is the point: a call-count check would pass even
        # if timeout were dropped on the floor.
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "{}"
        mock_response.json.return_value = {"verdict": {"allowed": True}}
        mock_response.raise_for_status.return_value = None
        mock_post = AsyncMock(return_value=mock_response)

        with patch.object(noma_v2_guardrail.async_handler, "post", mock_post):
            await noma_v2_guardrail._call_noma_scan(payload={"probe": "value"})

        # A bare float would set all four httpx budgets, replacing the 5s connect
        # handshake with scan_timeout and LENGTHENING the tail for an unreachable Noma.
        passed = mock_post.await_args.kwargs["timeout"]
        assert isinstance(passed, httpx.Timeout)
        assert passed.read == noma_v2_guardrail.scan_timeout
        assert passed.connect == min(HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS, noma_v2_guardrail.scan_timeout)

    @pytest.mark.asyncio
    async def test_timeout_does_not_block_request_and_is_audited(self):
        """A scan that times out must return the inputs unchanged and still record
        an audit entry, so the spend log carries the timeout instead of silently
        losing the scan."""
        import httpx

        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            monitor_mode=True,
            block_failures=False,
            scan_timeout=1.0,
            guardrail_name="noma-during-call",
            event_hook="during_call",
            default_on=True,
        )
        mock_post = AsyncMock(side_effect=httpx.ReadTimeout("simulated"))
        inputs = {"texts": ["hello"]}
        request_data = {"metadata": {"user_api_key": "sk-test"}}

        with patch.object(guardrail.async_handler, "post", mock_post):
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                input_type="request",
                request_data=request_data,
            )

        assert result == inputs
        entries = request_data["metadata"]["standard_logging_guardrail_information"]
        assert [e["guardrail_status"] for e in entries] == ["guardrail_failed_to_respond"]

    @pytest.mark.asyncio
    async def test_initialize_guardrail_v2_forwards_scan_timeout(self):
        params = LitellmParams(
            guardrail="noma",
            mode="during_call",
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            use_v2=True,
            scan_timeout=3.5,
        )
        callback = initialize_guardrail_v2(
            litellm_params=params,
            guardrail={"guardrail_name": "noma-during-call"},
        )
        assert callback.scan_timeout == 3.5

    @pytest.mark.asyncio
    async def test_scan_timeout_is_a_total_deadline_not_per_read(self):
        """The transport's own timeout is per-I/O-operation: its read budget resets on
        every byte, so a peer that dribbles the response indefinitely keeps the scan
        alive. Measured at 4x the configured value before asyncio.wait_for was added,
        and reported as a SUCCESS."""
        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            scan_timeout=0.3,
            guardrail_name="noma-during-call",
            event_hook="during_call",
        )

        async def _never_finishes(*args, **kwargs):
            await asyncio.sleep(30)
            raise AssertionError("unreachable: the deadline must fire first")

        start = time.monotonic()
        with patch.object(guardrail.async_handler, "post", _never_finishes):
            with pytest.raises(asyncio.TimeoutError):
                await guardrail._call_noma_scan(payload={"probe": "value"})
        assert time.monotonic() - start < 5.0

    def test_env_default_is_validated_not_trusted(self, monkeypatch):
        """get_env_float only rejects non-finite values, so a stray
        NOMA_SCAN_TIMEOUT_SECONDS=-5 would reach the transport and fail every scan in
        ~0ms. With block_failures=False that is swallowed: full traffic, zero AIDR
        coverage, guardrails still reported as enabled."""
        import litellm.proxy.guardrails.guardrail_hooks.noma.noma_v2 as noma_v2_module

        for poisoned in (-5.0, 0.0, 1e-9):
            monkeypatch.setattr(noma_v2_module, "NOMA_SCAN_TIMEOUT_SECONDS", poisoned)
            with pytest.raises(ValueError, match="scan_timeout"):
                noma_v2_module._coerce_scan_timeout(None)

    def test_generic_timeout_param_is_honoured(self):
        """LitellmParams.timeout is the documented generic knob four sibling guardrails
        already consume. An operator who sets it must not get silence."""
        params = LitellmParams(
            guardrail="noma",
            mode="during_call",
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            use_v2=True,
            timeout=2.0,
        )
        callback = initialize_guardrail_v2(litellm_params=params, guardrail={"guardrail_name": "noma-during-call"})
        assert callback.scan_timeout == 2.0

    def test_scan_timeout_wins_over_generic_timeout(self):
        params = LitellmParams(
            guardrail="noma",
            mode="during_call",
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            use_v2=True,
            timeout=2.0,
            scan_timeout=7.0,
        )
        callback = initialize_guardrail_v2(litellm_params=params, guardrail={"guardrail_name": "noma-during-call"})
        assert callback.scan_timeout == 7.0

    def test_default_is_well_under_the_shared_client_budget(self):
        """Pins the value against a regression to the 600s it exists to replace."""
        from litellm.constants import (
            COMPLETION_HTTP_FALLBACK_SECONDS,
            NOMA_SCAN_TIMEOUT_SECONDS,
        )

        assert NOMA_SCAN_TIMEOUT_SECONDS == 10.0
        assert NOMA_SCAN_TIMEOUT_SECONDS < COMPLETION_HTTP_FALLBACK_SECONDS / 10

    @pytest.mark.asyncio
    async def test_timeout_audit_entry_is_a_dict_and_flags_the_timeout(self):
        """With block_failures=False the exception is swallowed, so
        _run_guardrail_with_metrics records status="success" and
        litellm_guardrail_errors_total never increments. The audit entry is the only place a
        timeout is distinguishable, so it must say so explicitly — and be a dict, since
        ConnectTimeout stringifies to '' and str(e) would leave no explanation at all."""
        import httpx

        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            monitor_mode=True,
            block_failures=False,
            scan_timeout=0.2,
            guardrail_name="noma-during-call",
            event_hook="during_call",
        )
        request_data = {"metadata": {"user_api_key": "sk-test"}}

        # Raised only after the budget has elapsed: an instant ConnectTimeout is a
        # connection failure, not an expiry, and is covered separately.
        async def _expire_then_raise(*args, **kwargs):
            await asyncio.sleep(0.25)
            raise httpx.ConnectTimeout("")

        with patch.object(guardrail.async_handler, "post", _expire_then_raise):
            await guardrail.apply_guardrail(
                inputs={"texts": ["hello"]}, input_type="request", request_data=request_data
            )

        entry = request_data["metadata"]["standard_logging_guardrail_information"][0]
        assert entry["guardrail_status"] == "guardrail_failed_to_respond"
        response = entry["guardrail_response"]
        assert isinstance(response, dict)
        assert response["timed_out"] is True
        assert response["scan_timeout_seconds"] == 0.2
        assert response["elapsed_seconds"] >= 0.2
        # The entry must explain itself even for an exception whose str() is empty — here
        # asyncio.wait_for wins the race and raises TimeoutError(""), so `detail` falls back
        # to the type name rather than being blank.
        assert response["detail"] == response["error"]
        assert response["detail"]

    @pytest.mark.asyncio
    async def test_non_timeout_failure_is_not_flagged_as_timed_out(self):
        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            monitor_mode=True,
            block_failures=False,
            guardrail_name="noma-during-call",
            event_hook="during_call",
        )
        request_data = {"metadata": {}}

        with patch.object(guardrail.async_handler, "post", AsyncMock(side_effect=ValueError("bad json"))):
            await guardrail.apply_guardrail(
                inputs={"texts": ["hello"]}, input_type="request", request_data=request_data
            )

        response = request_data["metadata"]["standard_logging_guardrail_information"][0]["guardrail_response"]
        assert response["timed_out"] is False
        assert response["error"] == "ValueError"

    def test_config_model_floor_matches_the_runtime_floor(self):
        """A looser pydantic bound would accept a value that then fails pod startup."""
        import pydantic

        from litellm.constants import NOMA_MIN_SCAN_TIMEOUT_SECONDS

        below = NOMA_MIN_SCAN_TIMEOUT_SECONDS / 2
        with pytest.raises(pydantic.ValidationError):
            NomaV2GuardrailConfigModel(scan_timeout=below)
        with pytest.raises(ValueError):
            NomaV2Guardrail(
                api_key="test-api-key",
                api_base="https://api.test.noma.security/",
                scan_timeout=below,
                guardrail_name="test-noma-v2-guardrail",
                event_hook="during_call",
            )

    @pytest.mark.parametrize(
        "exc_factory",
        [
            pytest.param(lambda: asyncio.TimeoutError(), id="asyncio-TimeoutError"),
            pytest.param(lambda: httpx.ReadTimeout("read"), id="httpx-ReadTimeout"),
            pytest.param(lambda: httpx.ConnectTimeout(""), id="httpx-ConnectTimeout"),
            pytest.param(lambda: httpx.PoolTimeout("pool"), id="httpx-PoolTimeout"),
            pytest.param(
                lambda: litellm.Timeout(message="timed out", model="m", llm_provider="p"),
                id="litellm-Timeout",
            ),
            pytest.param(
                lambda: openai.APITimeoutError(request=httpx.Request("POST", "https://x.invalid")),
                id="openai-APITimeoutError",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_every_timeout_type_is_recognised_as_timeout_shaped(self, exc_factory):
        """All three hierarchies must be recognised, litellm.Timeout included.

        The shared AsyncHTTPHandler catches httpx timeouts and re-raises litellm.Timeout,
        which subclasses NEITHER asyncio.TimeoutError NOR httpx.TimeoutException. Because
        connect is pinned below scan_timeout, that is the type which actually arrives for an
        unreachable Noma, so omitting it left the marker silent on 46 real events.

        Recognition alone does not make it a deadline — see
        test_fast_failure_is_not_classified_as_a_deadline. Here the exception is raised only
        after the budget has elapsed, so both conditions hold.
        """
        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            monitor_mode=True,
            block_failures=False,
            scan_timeout=0.2,
            guardrail_name="noma-pre-call",
            event_hook="pre_call",
        )
        request_data = {"metadata": {}}

        async def _expire_then_raise(*args, **kwargs):
            await asyncio.sleep(0.25)
            raise exc_factory()

        with patch.object(guardrail.async_handler, "post", _expire_then_raise):
            await guardrail.apply_guardrail(
                inputs={"texts": ["hello"]}, input_type="request", request_data=request_data
            )

        entry = request_data["metadata"]["standard_logging_guardrail_information"][0]
        assert entry["guardrail_status"] == "guardrail_failed_to_respond"
        assert entry["guardrail_response"]["timed_out"] is True
        assert entry["guardrail_response"]["elapsed_seconds"] >= 0.2

    @pytest.mark.asyncio
    async def test_fast_failure_is_not_classified_as_a_deadline(self):
        """A timeout-TYPED exception that arrives instantly is a connection failure, not an
        expiry, and must not be filed as one.

        The HTTP layer re-raises litellm.Timeout for a dead pooled keep-alive too — its own
        message says "time taken=0.001 seconds". Keying on the type alone marked 74 such
        events in two hours as deadline expiries, the longest having taken 0.12s against a
        10s budget, which buried the real signal in identical-looking noise.
        """
        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            monitor_mode=True,
            block_failures=False,
            scan_timeout=10.0,
            guardrail_name="noma-pre-call",
            event_hook="pre_call",
        )
        request_data = {"metadata": {}}
        instant = litellm.Timeout(message="Connection timed out. time taken=0.001 seconds", model="m", llm_provider="p")

        with patch.object(guardrail.async_handler, "post", AsyncMock(side_effect=instant)):
            await guardrail.apply_guardrail(
                inputs={"texts": ["hello"]}, input_type="request", request_data=request_data
            )

        response = request_data["metadata"]["standard_logging_guardrail_information"][0]["guardrail_response"]
        assert response["timed_out"] is False, "an instant failure is not a deadline expiry"
        assert response["elapsed_seconds"] < 1.0
        assert response["scan_timeout_seconds"] == 10.0

    @pytest.mark.asyncio
    async def test_cancelled_scan_is_not_audited_as_success(self):
        """CancelledError is a BaseException, so without its own arm it skips
        `except Exception` and the `finally` records the initial "success" — a scan killed
        mid-flight would be audited as having passed. ARC-BUG-20's shutdown drain cancels
        in-flight monitor tasks on every restart, so this is routine."""
        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            monitor_mode=True,
            block_failures=False,
            guardrail_name="noma-during-call",
            event_hook="during_call",
        )
        request_data = {"metadata": {}}

        async def _hang(*args, **kwargs):
            await asyncio.sleep(30)

        with patch.object(guardrail.async_handler, "post", _hang):
            task = asyncio.create_task(
                guardrail.apply_guardrail(inputs={"texts": ["hello"]}, input_type="request", request_data=request_data)
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        entry = request_data["metadata"]["standard_logging_guardrail_information"][0]
        assert entry["guardrail_status"] == "guardrail_failed_to_respond"
        assert entry["guardrail_response"]["error"] == "CancelledError"
        assert entry["guardrail_response"]["timed_out"] is False

    def test_explicit_zero_scan_timeout_raises_rather_than_falling_back(self):
        """`or` would let an explicit 0 fall through to `timeout` (or the default), so a
        typo would silently become a working budget instead of the documented error."""
        params = LitellmParams(
            guardrail="noma",
            mode="during_call",
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            use_v2=True,
            scan_timeout=0,
            timeout=5.0,
        )
        with pytest.raises(ValueError, match="scan_timeout"):
            initialize_guardrail_v2(litellm_params=params, guardrail={"guardrail_name": "noma-during-call"})

    def test_generic_timeout_of_600_does_not_restore_the_600s_budget(self):
        """`timeout: 600` is the exact budget this deadline exists to remove. Superseded by
        the clamp: raising would break an existing, previously-inert config, so the value is
        clamped to the ceiling instead — the 600s budget is still not restored."""
        params = LitellmParams(
            guardrail="noma",
            mode="pre_call",
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            use_v2=True,
            timeout=600.0,
        )
        callback = initialize_guardrail_v2(litellm_params=params, guardrail={"guardrail_name": "noma-pre-call"})
        from litellm.constants import NOMA_MAX_SCAN_TIMEOUT_SECONDS

        assert callback.scan_timeout == NOMA_MAX_SCAN_TIMEOUT_SECONDS
        assert callback.scan_timeout < 600.0

    @pytest.mark.parametrize(
        "generic_timeout,expected",
        [(600.0, 60.0), (0.001, 0.1), (5.0, 5.0)],
        ids=["above-ceiling-clamped", "below-floor-clamped", "in-range-kept"],
    )
    def test_out_of_range_generic_timeout_is_clamped_not_fatal(self, generic_timeout, expected):
        """`timeout` is the shared generic knob, and it was legal-and-inert here before this
        deadline existed. Raising on it would turn an existing config into a pod that never
        becomes ready — and because the guardrail registry re-initialises DB-stored
        guardrails on poll, an Admin-UI edit could break a RUNNING pod. Clamp with a
        warning; keep raising only for `scan_timeout`, which is set deliberately."""
        params = LitellmParams(
            guardrail="noma",
            mode="pre_call",
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            use_v2=True,
            timeout=generic_timeout,
        )
        callback = initialize_guardrail_v2(litellm_params=params, guardrail={"guardrail_name": "noma-pre-call"})
        assert callback.scan_timeout == expected

    def test_non_timeout_failure_log_does_not_carry_the_exception_text(self, caplog):
        """The container log goes to CloudWatch, which must stay free of PHI. str(e) can
        carry request content (a ValueError echoing the prompt, an HTTPStatusError carrying
        the tenant URL), so the log line names the exception TYPE only. The full text stays
        in the audit entry, which lands in the spend log and S3 — the PHI-bearing store."""
        import logging

        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            monitor_mode=True,
            block_failures=False,
            guardrail_name="noma-pre-call",
            event_hook="pre_call",
        )
        secret = "patient MRN 12345 has a diagnosis"
        request_data = {"metadata": {}}

        with caplog.at_level(logging.ERROR):
            with patch.object(guardrail.async_handler, "post", AsyncMock(side_effect=ValueError(secret))):
                asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
                    guardrail.apply_guardrail(inputs={"texts": ["hi"]}, input_type="request", request_data=request_data)
                )

        assert secret not in caplog.text, "PHI must not reach the container log"
        assert "ValueError" in caplog.text
        # ...but the audit record, which goes to the PHI-bearing store, keeps the detail.
        entry = request_data["metadata"]["standard_logging_guardrail_information"][0]
        assert secret in entry["guardrail_response"]["detail"]


class TestNomaV2ConnectionFailureRetry:
    """[ARC-BUG-47] A broken pooled connection reaches us disguised as a timeout.

    aiohttp raises SocketTimeoutError when a connection breaks WHILE READING, and the
    transport maps that to httpx.ReadTimeout. The shape that identifies it is elapsed
    time far below the budget: measured on prd-ai at 2-93ms against 10s, on ~24% of scans.

    These tests pin the discriminator (elapsed vs budget), not the exception type — the
    type is identical for a real expiry, which is exactly why the bug was invisible.
    """

    @staticmethod
    def _ok_response():
        response = MagicMock()
        response.status_code = 200
        response.text = '{"action":"NONE"}'
        response.json.return_value = {"action": "NONE"}
        response.raise_for_status = MagicMock()
        return response

    @pytest.mark.asyncio
    async def test_fast_timeout_is_retried_once_and_succeeds(self, noma_v2_guardrail):
        """The bug's exact shape: instant ReadTimeout, then a healthy Noma on retry."""
        post = AsyncMock(side_effect=[httpx.ReadTimeout("Timeout on reading data from socket"), self._ok_response()])
        with patch.object(noma_v2_guardrail.async_handler, "post", post):
            result = await noma_v2_guardrail._call_noma_scan(payload={"inputs": {"texts": []}})

        assert result == {"action": "NONE"}
        assert post.await_count == 2, "a fast timeout must be retried exactly once"

    @pytest.mark.asyncio
    async def test_retry_is_not_a_loop(self, noma_v2_guardrail):
        """Two fast failures must raise, not retry forever.

        If a freshly-established connection also dies this fast, connection reuse is not
        the problem and further attempts only spend the caller's budget.
        """
        post = AsyncMock(side_effect=httpx.ReadTimeout("Timeout on reading data from socket"))
        with patch.object(noma_v2_guardrail.async_handler, "post", post):
            with pytest.raises(httpx.ReadTimeout):
                await noma_v2_guardrail._call_noma_scan(payload={"inputs": {"texts": []}})

        assert post.await_count == 2, "one retry, then give up"

    @pytest.mark.asyncio
    async def test_real_deadline_expiry_is_not_retried(self, noma_v2_guardrail):
        """The critical negative case, and the reason the threshold exists.

        Same exception type as the bug. Only the elapsed time differs. A scan that
        genuinely burned its budget must NOT be retried — that would double the latency
        ceiling ARC-BUG-43 exists to guarantee, on a Noma that is merely slow.
        """
        # scan_timeout must sit ABOVE the connect budget, or the remaining-budget guard
        # declines the retry first and this test passes for the wrong reason. Measured that
        # exact false pass while building it: at scan_timeout=2.0 the budget guard fired, so
        # widening the threshold to 1.0 left all tests green. Only the threshold can decline
        # the retry at these values.
        noma_v2_guardrail.scan_timeout = HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS * 4

        # Fails at 25% of the budget: far past the 5% connection-failure threshold, and the
        # remaining 75% is comfortably more than one more handshake.
        async def expire_at_a_quarter_of_the_budget(*args, **kwargs):
            await asyncio.sleep(HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS)
            raise httpx.ReadTimeout("real expiry")

        post = AsyncMock(side_effect=expire_at_a_quarter_of_the_budget)
        with patch.object(noma_v2_guardrail.async_handler, "post", post):
            with pytest.raises((httpx.ReadTimeout, asyncio.TimeoutError)):
                await noma_v2_guardrail._call_noma_scan(payload={"inputs": {"texts": []}})

        assert post.await_count == 1, "a genuine expiry must not be retried"

    @pytest.mark.asyncio
    async def test_non_timeout_errors_are_never_retried(self, noma_v2_guardrail):
        """A 500 or a bad payload is not a connection problem. Retrying hides it."""
        post = AsyncMock(side_effect=ValueError("malformed"))
        with patch.object(noma_v2_guardrail.async_handler, "post", post):
            with pytest.raises(ValueError):
                await noma_v2_guardrail._call_noma_scan(payload={"inputs": {"texts": []}})

        assert post.await_count == 1

    @pytest.mark.asyncio
    async def test_all_three_timeout_types_are_covered(self, noma_v2_guardrail):
        """The bug arrives as any of the three hierarchies _TIMEOUT_EXCEPTIONS lists.

        openai.APITimeoutError is the one that actually reached us in production — the
        shared handler converts httpx timeouts before we see them.
        """
        for exc in (
            httpx.ReadTimeout("socket"),
            asyncio.TimeoutError(),
            openai.APITimeoutError(request=httpx.Request("POST", "https://x")),
        ):
            post = AsyncMock(side_effect=[exc, self._ok_response()])
            with patch.object(noma_v2_guardrail.async_handler, "post", post):
                result = await noma_v2_guardrail._call_noma_scan(payload={"inputs": {"texts": []}})
            assert result == {"action": "NONE"}
            assert post.await_count == 2, f"{type(exc).__name__} must be retried"

    @pytest.mark.asyncio
    async def test_retry_shares_the_original_budget(self, noma_v2_guardrail):
        """The retry must NOT restart scan_timeout.

        A fresh budget per attempt would let two attempts spend 2x the wall-clock ceiling
        that ARC-BUG-43 promises the caller. Asserted on the payload the code computes —
        the second attempt's wait_for budget — rather than on a call count, because a
        count cannot distinguish a shared budget from a reset one.
        """
        noma_v2_guardrail.scan_timeout = 10.0
        budgets: list[float] = []
        real_wait_for = asyncio.wait_for

        async def recording_wait_for(aw, timeout):
            budgets.append(timeout)
            return await real_wait_for(aw, timeout)

        post = AsyncMock(side_effect=[httpx.ReadTimeout("Timeout on reading data from socket"), self._ok_response()])
        with patch.object(noma_v2_guardrail.async_handler, "post", post):
            with patch("asyncio.wait_for", recording_wait_for):
                await noma_v2_guardrail._call_noma_scan(payload={"inputs": {"texts": []}})

        assert len(budgets) == 2
        assert budgets[0] == 10.0
        assert budgets[1] < budgets[0], "the retry must inherit the remaining budget, not a fresh one"

    @pytest.mark.asyncio
    async def test_no_retry_once_the_budget_can_no_longer_fit_one(self, noma_v2_guardrail):
        """The budget guard bounds retries even when every failure still looks retryable.

        Only reachable with more than one retry allowed. On the default MAX=1 the guard is
        unreachable by construction: a failure fast enough to qualify (< 5% of the budget)
        always leaves > 95% remaining, which always clears the threshold. Asserting it through
        the default path would assert nothing, so the retry count is raised here to exercise
        the code that actually stops the loop.

        [ARC-BUG-47 review] The first version of this test slept 0.25s against a 0.26s
        threshold — a 9ms margin. Under CPU load the sleep overshot and the THRESHOLD declined
        the retry rather than the budget guard, so on a loaded runner it passed 30/30 even with
        the budget guard deleted. It now drives the clock directly instead of racing it.
        """
        noma_v2_guardrail.scan_timeout = 60.0
        # Each attempt charges 2.5s of budget while staying under the 3.0s fast-fail window
        # (5% of 60s), so the THRESHOLD always says "retryable" and only the budget guard can
        # end the loop. Driving the clock rather than sleeping keeps that exact.
        charge = 2.5
        clock = {"t": 0.0}

        def fake_monotonic():
            return clock["t"]

        async def fail_fast(*args, **kwargs):
            clock["t"] += charge
            raise httpx.ReadTimeout("socket")

        real_monotonic = time.monotonic
        post = AsyncMock(side_effect=fail_fast)
        with patch.object(nv2, "NOMA_CONNECTION_RETRY_MAX", 999):
            with patch.object(nv2.time, "monotonic", fake_monotonic):
                with patch.object(noma_v2_guardrail.async_handler, "post", post):
                    # The guard's observable effect is the exception the CALLER sees. Without it
                    # the loop keeps going until asyncio.wait_for is handed a non-positive budget
                    # and raises TimeoutError itself, burying the real cause. With it, the last
                    # genuine failure propagates. Asserting the exact type is what makes this a
                    # test of the guard rather than of the attempt count — measured: attempts are
                    # 24 either way, so a count assertion cannot tell the two apart.
                    with pytest.raises(httpx.ReadTimeout):
                        await noma_v2_guardrail._call_noma_scan(payload={"inputs": {"texts": []}})
        assert time.monotonic is real_monotonic, "the clock patch must not leak"

        # 60s of budget at 2.5s per attempt: ~24 attempts, then the budget guard stops it —
        # far short of the 999 the counter would allow.
        assert 2 <= post.await_count <= 30, f"budget guard should stop it, got {post.await_count}"

    @pytest.mark.asyncio
    async def test_a_retried_fast_failure_is_not_relabelled_deadline_exceeded(self, noma_v2_guardrail):
        """[ARC-BUG-47 review, finding 3] The retry must not corrupt ARC-BUG-43's own marker.

        apply_guardrail measures elapsed from its own start, so it sums ALL attempts. Left
        alone, enough retried fast failures push the total past 0.9 * scan_timeout and the event
        logs "DEADLINE EXCEEDED" — inverting the exact marker ARC-BUG-43 exists to make
        trustworthy: a connection failure would masquerade as a slow Noma. Classification
        therefore reads the LAST ATTEMPT's duration, while the log and audit record keep total
        wall-clock, which is what the caller actually waited.

        ℹ️ Reaching the defect requires MAX > 1, and that is worth stating rather than hiding:
        at the shipped MAX=1 the sum of two attempts is bounded by 2 * 5% = 10% of the budget,
        which can never reach the 90% expiry line. So the retry ceiling already makes this
        unreachable in production, and this fix is defensive. The test patches the retry count
        to reach the code at all — asserting the invariant rather than today's configuration,
        because a future bump to MAX would otherwise silently reintroduce the relabel.
        """
        noma_v2_guardrail.scan_timeout = 60.0
        noma_v2_guardrail.monitor_mode = True

        # 2.5s per attempt: inside the 3.0s fast-fail window (5% of 60s) so every attempt is
        # judged retryable, and 19+ of them sum past 54.0s (90% of 60s) — the relabel zone.
        clock = {"t": 0.0}

        async def fail_fast_but_expensive(*args, **kwargs):
            clock["t"] += 2.5
            raise httpx.ReadTimeout("socket")

        post = AsyncMock(side_effect=fail_fast_but_expensive)
        request_data: dict = {"metadata": {}}

        # apply_guardrail measures its own elapsed with datetime.now(), NOT time.monotonic —
        # patching only monotonic left the classifier seeing 0.002s and the test passing for
        # the wrong reason, which is exactly what the negative control caught. Both clocks
        # must advance together for the sum to reach the relabel zone.
        class _FakeDatetime(nv2.datetime):
            @classmethod
            def now(cls, tz=None):
                return nv2.datetime.fromtimestamp(clock["t"], tz) if tz else nv2.datetime.fromtimestamp(clock["t"])

        with patch.object(nv2, "NOMA_CONNECTION_RETRY_MAX", 999):
            with patch.object(nv2, "datetime", _FakeDatetime):
                with patch.object(nv2.time, "monotonic", lambda: clock["t"]):
                    with patch.object(noma_v2_guardrail.async_handler, "post", post):
                        await noma_v2_guardrail.apply_guardrail(
                            inputs={"texts": ["hello"]},
                            request_data=request_data,
                            input_type="request",
                            logging_obj=None,
                        )

        assert post.await_count > 18, (
            f"the test must actually reach the relabel zone, only got {post.await_count} attempts"
        )
        entries = request_data["metadata"]["standard_logging_guardrail_information"]
        assert len(entries) == 1, "a retried scan still produces exactly one audit record"
        response = entries[0]["guardrail_response"]
        assert response["timed_out"] is False, (
            "fast failures summing past 0.9x the budget must NOT be filed as a deadline expiry"
        )

    @pytest.mark.asyncio
    async def test_the_retry_is_bounded_by_its_own_ceiling_not_the_remaining_budget(self, noma_v2_guardrail):
        """[ARC-BUG-47 review] The blocker: an unbounded retry is worse than the bug.

        Attempt 1 dies in milliseconds — the bug's signature. If the retry then inherits the
        whole remaining budget and hangs, the request pays the full scan_timeout inside a
        guardrail that today costs it 40ms and lets it through. pre_call is ON the request
        path and cannot be detached, so at scan_timeout=10 that is ~9x the p50 request budget.

        Asserted on the budget handed to the retry, not on wall-clock, so the test states the
        invariant rather than racing a sleep.
        """
        noma_v2_guardrail.scan_timeout = 10.0
        budgets: list[float] = []
        real_wait_for = asyncio.wait_for

        async def recording_wait_for(aw, timeout):
            budgets.append(timeout)
            return await real_wait_for(aw, timeout)

        post = AsyncMock(side_effect=[httpx.ReadTimeout("socket"), self._ok_response()])
        with patch.object(noma_v2_guardrail.async_handler, "post", post):
            with patch("asyncio.wait_for", recording_wait_for):
                await noma_v2_guardrail._call_noma_scan(payload={"inputs": {"texts": []}})

        assert len(budgets) == 2
        assert budgets[0] == 10.0, "attempt 1 gets the full budget"
        assert budgets[1] <= nv2.NOMA_CONNECTION_RETRY_CEILING_SECONDS, (
            f"the retry must be capped at the ceiling, got {budgets[1]}"
        )

    @pytest.mark.asyncio
    async def test_retry_is_reachable_below_the_connect_budget(self, noma_v2_guardrail):
        """[ARC-BUG-47 review] The silent config cliff, found independently by two reviews.

        The first version compared the remaining budget against
        min(HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS, scan_timeout). For any scan_timeout <= 5.0
        that min() collapsed to scan_timeout itself, making the test
        (scan_timeout - elapsed) > scan_timeout — false for every positive elapsed. Measured
        cliff: 0.1-5.0 never retried, 5.01+ retried.

        That matters because initialize_guardrail_v2 CLAMPS a generic timeout into
        [0.1, 60.0] rather than rejecting it, so an operator tightening a guardrail to 3s to
        bound tail latency would have silently shipped a pod reporting the fix as active while
        ~24% of its scans still passed uninspected.
        """
        for scan_timeout in (0.5, 2.0, 3.0, 5.0):
            noma_v2_guardrail.scan_timeout = scan_timeout
            post = AsyncMock(side_effect=[httpx.ReadTimeout("socket"), self._ok_response()])
            with patch.object(noma_v2_guardrail.async_handler, "post", post):
                result = await noma_v2_guardrail._call_noma_scan(payload={"inputs": {"texts": []}})
            assert result == {"action": "NONE"}
            assert post.await_count == 2, f"scan_timeout={scan_timeout} must still retry"

    @pytest.mark.asyncio
    async def test_env_kill_switch_restores_the_prior_behaviour(self, noma_v2_guardrail):
        """[ARC-BUG-47 review] NOMA_CONNECTION_RETRY_MAX=0 must fully disable the retry.

        This ships as a second production change in one day; without a runtime lever the only
        rollback is an image rebuild plus a rolling restart of every proxy pod.
        """
        post = AsyncMock(side_effect=httpx.ReadTimeout("socket"))
        with patch.object(nv2, "NOMA_CONNECTION_RETRY_MAX", 0):
            with patch.object(noma_v2_guardrail.async_handler, "post", post):
                with pytest.raises(httpx.ReadTimeout):
                    await noma_v2_guardrail._call_noma_scan(payload={"inputs": {"texts": []}})

        assert post.await_count == 1, "kill switch must leave exactly the pre-fix behaviour"

    @pytest.mark.asyncio
    async def test_the_retry_is_logged_with_both_numbers(self, noma_v2_guardrail, caplog):
        """Operators need to see WHY a retry happened, and the measured elapsed time.

        ARC-BUG-43 shipped a version that printed only the configured budget in the slot
        that reads as elapsed time, so every line claimed "after 10.0s" including 2ms
        failures. Both numbers, measured one first.
        """
        import logging

        post = AsyncMock(side_effect=[httpx.ReadTimeout("Timeout on reading data from socket"), self._ok_response()])
        with caplog.at_level(logging.WARNING):
            with patch.object(noma_v2_guardrail.async_handler, "post", post):
                await noma_v2_guardrail._call_noma_scan(payload={"inputs": {"texts": []}})

        assert "broken pooled connection" in caplog.text
        assert "ReadTimeout" in caplog.text
        assert noma_v2_guardrail.guardrail_name in caplog.text

        # [ARC-BUG-47 review] Pin the ORDER, not just the presence of numbers. Substring
        # assertions alone let a reviewer swap attempt_elapsed and scan_timeout in the format
        # args -- recreating verbatim the ARC-BUG-43 bug this docstring cites -- with the test
        # still green. The measured time must come first and must not be the budget.
        import re as _re

        m = _re.search(r"scan failed after ([\d.]+)s with (\w+), far below the ([\d.]+)s budget", caplog.text)
        assert m, f"the WARNING must keep its measured-first shape, got: {caplog.text[:300]}"
        measured, exc_name, budget = float(m.group(1)), m.group(2), float(m.group(3))
        assert budget == noma_v2_guardrail.scan_timeout, "the second number must be the budget"
        assert measured < budget, f"measured {measured}s must be below the budget {budget}s"
        assert exc_name == "ReadTimeout"
        # hook, so an operator can tell the on-path pre_call leg from a side hook.
        assert "hook=" in caplog.text

        # [ARC-BUG-47 review] Lock the no-PHI property of this CloudWatch-bound line.
        # This file's own comments state the split explicitly: container logs carry NO PHI,
        # while the audit record deliberately does. A narrow mutation adding str(e) to the log
        # argument passed all 104 tests, so nothing held the property the code already has.
        # str(e) for the three timeout types is 'timed out' / '' / 'Request timed out.', but
        # litellm.Timeout stringifies with the URL and a future exception type could carry more.
        assert "Timeout on reading data from socket" not in caplog.text, (
            "the exception's own message must not reach the container log"
        )


class TestConcurrencySafetyOfTheClassifier:
    """[ARC-BUG-47 review] The per-attempt classifier must be per-REQUEST, not per-instance.

    `initialize_guardrail_v2` registers ONE NomaV2Guardrail per config as a global callback, so
    every concurrent request shares the instance. The first version of this fix parked the last
    attempt's elapsed time on `self`, which two independent reviews flagged as a cross-request
    leak.

    ⚠️ Honest about what these tests do and do not prove. Measured while writing them: the
    end-to-end scenario below passes on the instance-state version TOO, because the write and the
    read are separated only by a synchronous `raise` — no await sits in that window, so a
    concurrent task cannot interleave between them in practice. The leak is real but narrower than
    "the classifier is randomly wrong at concurrency": it needs a caller that awaits between
    `_call_noma_scan` raising and `apply_guardrail` classifying, which this code path does not do
    today and a refactor easily could.

    So the first test asserts the OUTCOME stays correct under concurrency (a regression guard that
    would catch a future refactor introducing that await), and the second asserts the MECHANISM
    directly — that the state lives in a ContextVar and not on the instance, which is the property
    that makes the whole class of bug impossible rather than merely unlikely. Asserting only the
    outcome would have let the instance-state version pass, as it did.
    """

    @pytest.mark.asyncio
    async def test_a_concurrent_fast_failure_does_not_steal_the_slow_scans_classification(self):
        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            monitor_mode=True,
            block_failures=False,
            guardrail_name="noma-pre-call",
            event_hook="pre_call",
            default_on=True,
        )
        guardrail.scan_timeout = 1.0

        async def genuine_expiry(*args, **kwargs):
            await asyncio.sleep(1.05)
            raise httpx.ReadTimeout("real expiry")

        async def fast_connection_break(*args, **kwargs):
            raise httpx.ReadTimeout("socket")

        async def drive(side_effect, request_data):
            with patch.object(guardrail.async_handler, "post", AsyncMock(side_effect=side_effect)):
                await guardrail.apply_guardrail(
                    inputs={"texts": ["x"]},
                    request_data=request_data,
                    input_type="request",
                    logging_obj=None,
                )

        rd_slow: dict = {"metadata": {}}
        rd_fast: dict = {"metadata": {}}

        slow = asyncio.create_task(drive(genuine_expiry, rd_slow))
        await asyncio.sleep(0.4)  # land the fast failure inside the slow scan's window
        await drive(fast_connection_break, rd_fast)
        await slow

        slow_response = rd_slow["metadata"]["standard_logging_guardrail_information"][0]["guardrail_response"]
        assert slow_response["timed_out"] is True, (
            "a genuine deadline expiry was reclassified because a concurrent fast failure "
            "overwrote the shared classifier state"
        )

        fast_response = rd_fast["metadata"]["standard_logging_guardrail_information"][0]["guardrail_response"]
        assert fast_response["timed_out"] is False, "the fast failure must still be a fast failure"

    @pytest.mark.asyncio
    async def test_the_classifier_state_is_context_local_not_instance_state(self):
        """Assert the MECHANISM, because asserting the outcome alone is not enough.

        This is the test that actually fails if the state moves back onto `self` — the outcome
        test above does not, measured. Two properties are checked: the value lands in the
        ContextVar, and it does NOT land on the shared instance where a future refactor could
        leak it across requests.
        """
        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            monitor_mode=True,
            block_failures=False,
            guardrail_name="noma-pre-call",
            event_hook="pre_call",
            default_on=True,
        )
        nv2._LAST_SCAN_ATTEMPT_ELAPSED.set(None)
        post = AsyncMock(side_effect=httpx.ReadTimeout("socket"))
        with patch.object(guardrail.async_handler, "post", post):
            with pytest.raises(httpx.ReadTimeout):
                await guardrail._call_noma_scan(payload={"inputs": {"texts": []}})

        assert nv2._LAST_SCAN_ATTEMPT_ELAPSED.get() is not None, (
            "the attempt's elapsed time must be recorded in the ContextVar"
        )
        assert not hasattr(guardrail, "_last_scan_attempt_elapsed"), (
            "the elapsed time must NOT be parked on the shared guardrail instance — "
            "initialize_guardrail_v2 registers one instance per config for the whole pod"
        )

        # A task started from a COPIED context does not see this request's value — that is the
        # isolation property, and it is what asyncio gives each request handler.
        import contextvars

        fresh = contextvars.Context()
        assert fresh.run(nv2._LAST_SCAN_ATTEMPT_ELAPSED.get) is None, (
            "a fresh context must not inherit another request's attempt time"
        )

    @pytest.mark.asyncio
    async def test_the_kill_switch_restores_the_classifier_too_not_just_the_retry(self):
        """[ARC-BUG-47 review] NOMA_CONNECTION_RETRY_MAX=0 must restore the PRE-fix behaviour whole.

        The first version stopped the retry but kept writing the per-attempt value, so an operator
        pulling the kill switch to escape a retry problem kept the classifier change — and its
        race. Both are now gated on the same switch.
        """
        guardrail = NomaV2Guardrail(
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            monitor_mode=True,
            block_failures=False,
            guardrail_name="noma-pre-call",
            event_hook="pre_call",
            default_on=True,
        )
        nv2._LAST_SCAN_ATTEMPT_ELAPSED.set(None)
        post = AsyncMock(side_effect=httpx.ReadTimeout("socket"))
        with patch.object(nv2, "NOMA_CONNECTION_RETRY_MAX", 0):
            with patch.object(guardrail.async_handler, "post", post):
                with pytest.raises(httpx.ReadTimeout):
                    await guardrail._call_noma_scan(payload={"inputs": {"texts": []}})

        assert post.await_count == 1, "kill switch must disable the retry"
        assert nv2._LAST_SCAN_ATTEMPT_ELAPSED.get() is None, (
            "with the retry disabled, the classifier state must not be written either"
        )
