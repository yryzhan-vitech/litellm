"""
Bedrock Token Counter implementation using the CountTokens API.
"""

from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.bedrock.common_utils import BedrockError, get_bedrock_base_model
from litellm.llms.bedrock.count_tokens.handler import BedrockCountTokensHandler
from litellm.types.utils import LlmProviders, TokenCountResponse

# [ARC-BUG-40] Model families whose Bedrock CountTokens support is known-absent, so the call is
# skipped instead of made and thrown away. Matched as a prefix against the resolved model id
# AFTER get_bedrock_base_model() has stripped the bedrock/ and us. prefixes.
#
# Measured against live Bedrock in us-east-1 (aria-arcadia-io, 2026-08-05), both the bare and the
# us.-prefixed id for every Anthropic model in the prd-ai config:
#
#   anthropic.claude-sonnet-5                    ValidationException: doesn't support counting tokens
#   us.anthropic.claude-sonnet-5                 ValidationException: doesn't support counting tokens
#   us.anthropic.claude-sonnet-4-6               ValidationException: doesn't support counting tokens
#   us.anthropic.claude-sonnet-4-5-20250929-v1:0 ValidationException: doesn't support counting tokens
#   us.anthropic.claude-haiku-4-5-20251001-v1:0  ValidationException: doesn't support counting tokens
#   us.anthropic.claude-opus-4-7 / -4-8 / -5     ValidationException: doesn't support counting tokens
#   us.anthropic.claude-fable-5                  ValidationException: doesn't support counting tokens
#
# So NOT ONE Anthropic model this proxy routes to supports it. Every call was guaranteed to 400.
#
# ⚠️ The prefix is not the problem, and that mattered: an earlier branch
# (bugfix/bedrock-count-tokens-inference-profile) tried adding the us. prefix and was correctly
# abandoned — inference-profile ids are rejected too, for every model. Both forms fail
# identically, which the probe above confirms line by line.
#
# Why skipping matters beyond the noise: those 400s are FAST, and a fast failure is exactly what
# the router's cooldown counts. Measured on prd-ai, 33 of 34 successful fallbacks were triggered
# by status=400 — CountTokens, not a provider incident. With allowed_fails armed, three
# token-count requests on one pod would cool a deployment fleet-wide for 30s. The 400s have to
# stop being generated before a circuit breaker can safely be switched on.
#
# Deliberately a denylist, not an allowlist: Bedrock adds CountTokens support over time, and an
# allowlist would silently keep skipping a model the day support arrives. A denylist degrades the
# other way — a newly-supported model starts working on its own, and a newly-launched unsupported
# one costs one wasted call until it is added here.
_COUNT_TOKENS_UNSUPPORTED_PREFIXES: tuple = ("anthropic.claude-",)


class BedrockTokenCounter(BaseTokenCounter):
    """Token counter implementation for AWS Bedrock provider using the CountTokens API."""

    def should_use_token_counting_api(
        self,
        custom_llm_provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> bool:
        """
        Returns True if we should use the Bedrock CountTokens API for token counting.

        [ARC-BUG-40] `model` is optional so existing callers that pass only the provider keep
        working: without it the check degrades to the original provider-only behaviour rather
        than skipping everything.
        """
        if custom_llm_provider != LlmProviders.BEDROCK.value:
            return False
        if model is None:
            return True
        resolved = get_bedrock_base_model(model)
        return not resolved.startswith(_COUNT_TOKENS_UNSUPPORTED_PREFIXES)

    async def count_tokens(
        self,
        model_to_use: str,
        messages: Optional[List[Dict[str, Any]]],
        contents: Optional[List[Dict[str, Any]]],
        deployment: Optional[Dict[str, Any]] = None,
        request_model: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[Any] = None,
    ) -> Optional[TokenCountResponse]:
        """
        Count tokens using AWS Bedrock's CountTokens API.

        This method calls the existing BedrockCountTokensHandler to make an API call
        to Bedrock's token counting endpoint, bypassing the local tiktoken-based counting.

        Args:
            model_to_use: The model identifier
            messages: The messages to count tokens for
            contents: Alternative content format (not used for Bedrock)
            deployment: Deployment configuration containing litellm_params
            request_model: The original request model name

        Returns:
            TokenCountResponse with token count, or None if counting fails
        """
        if not messages:
            return None

        deployment = deployment or {}
        litellm_params = deployment.get("litellm_params", {})

        # Build request data in the format expected by BedrockCountTokensHandler
        request_data: Dict[str, Any] = {
            "model": model_to_use,
            "messages": messages,
        }

        if tools:
            request_data["tools"] = tools

        if system:
            request_data["system"] = system

        # Get the resolved model (strip prefixes like bedrock/, converse/, etc.)
        resolved_model = get_bedrock_base_model(model_to_use)

        try:
            handler = BedrockCountTokensHandler()
            result = await handler.handle_count_tokens_request(
                request_data=request_data,
                litellm_params=litellm_params,
                resolved_model=resolved_model,
            )

            # Transform response to TokenCountResponse
            if result is not None:
                return TokenCountResponse(
                    total_tokens=result.get("input_tokens", 0),
                    request_model=request_model,
                    model_used=model_to_use,
                    tokenizer_type="bedrock_api",
                    original_response=result,
                )
        except BedrockError as e:
            verbose_logger.warning(f"Bedrock CountTokens API error: status={e.status_code}, message={e.message}")
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="bedrock_api",
                error=True,
                error_message=e.message,
                status_code=e.status_code,
            )
        except Exception as e:
            verbose_logger.warning(f"Error calling Bedrock CountTokens API: {e}")
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="bedrock_api",
                error=True,
                error_message=str(e),
                status_code=500,
            )

        return None
