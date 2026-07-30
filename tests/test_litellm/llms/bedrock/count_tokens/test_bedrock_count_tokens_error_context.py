import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.count_tokens.handler import BedrockCountTokensHandler


@pytest.mark.asyncio
async def test_generic_exception_keeps_class_name_in_message():
    """A bare exception whose str() carries no type (KeyError -> "'messages'") must
    still name its class in the 500, otherwise the failure is unattributable."""
    handler = BedrockCountTokensHandler()

    with patch.object(
        handler, "validate_count_tokens_request", side_effect=KeyError("messages")
    ):
        with pytest.raises(BedrockError) as exc_info:
            await handler.handle_count_tokens_request(
                request_data={"messages": [{"role": "user", "content": "hi"}]},
                litellm_params={},
                resolved_model="anthropic.claude-3-sonnet-20240229-v1:0",
            )

    assert exc_info.value.status_code == 500
    assert "KeyError" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_generic_exception_logs_traceback():
    """The dev-ai case was a litellm.Timeout raised 0.001s into a call Bedrock
    answered with HTTP 200. str(e) named the class but not the raiser, so the log
    must carry the traceback -- i.e. logger.exception, not logger.error."""
    handler = BedrockCountTokensHandler()

    with patch.object(
        handler, "validate_count_tokens_request", side_effect=RuntimeError("boom")
    ):
        with patch(
            "litellm.llms.bedrock.count_tokens.handler.verbose_logger"
        ) as mock_logger:
            with pytest.raises(BedrockError):
                await handler.handle_count_tokens_request(
                    request_data={"messages": [{"role": "user", "content": "hi"}]},
                    litellm_params={},
                    resolved_model="anthropic.claude-3-sonnet-20240229-v1:0",
                )

    mock_logger.exception.assert_called_once()
    logged = mock_logger.exception.call_args[0][0]
    assert "RuntimeError" in logged
    assert "boom" in logged


@pytest.mark.asyncio
async def test_bedrock_error_is_not_reclassified_as_500():
    """The BedrockError branch must still short-circuit: a provider 400 stays a 400,
    so the ARC-BUG-40 undercount path is untouched by this change."""
    handler = BedrockCountTokensHandler()

    with patch.object(
        handler,
        "validate_count_tokens_request",
        side_effect=BedrockError(status_code=400, message="model does not support it"),
    ):
        with pytest.raises(BedrockError) as exc_info:
            await handler.handle_count_tokens_request(
                request_data={"messages": [{"role": "user", "content": "hi"}]},
                litellm_params={},
                resolved_model="anthropic.claude-sonnet-5",
            )

    assert exc_info.value.status_code == 400
    assert "CountTokens processing error" not in str(exc_info.value.message)
