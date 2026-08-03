"""
Advisor Orchestration Handler

Implements the advisor tool loop for providers that don't support
advisor_20260301 natively (i.e. everything except Anthropic direct for now).

How it works:
1. Detects advisor_20260301 in tools + non-native provider → intercepts.
2. Translates the advisor tool to a regular function tool the provider understands.
3. Calls the executor model (non-streaming).
4. If the executor makes a tool_use call named "advisor", runs the advisor model
   and injects the result as a tool_result before re-calling the executor.
5. Repeats until the executor produces a final text response or max_uses is hit.
6. Wraps in FakeAnthropicMessagesStreamIterator if the caller requested streaming.
"""

import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import litellm
import litellm.constants as _c
from litellm._logging import verbose_logger
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.url_utils import validate_url
from litellm.llms.anthropic.common_utils import strip_advisor_blocks_from_messages
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.anthropic import ANTHROPIC_ADVISOR_TOOL_TYPE

ADVISOR_MAX_USES: int = _c.ADVISOR_MAX_USES
ADVISOR_NATIVE_PROVIDERS: frozenset = _c.ADVISOR_NATIVE_PROVIDERS
ADVISOR_TOOL_DESCRIPTION: str = _c.ADVISOR_TOOL_DESCRIPTION

from .base import MessagesInterceptor


class AdvisorMaxIterationsError(Exception):
    """Raised when the advisor loop exceeds max_uses."""


class AdvisorOrchestrationHandler(MessagesInterceptor):
    """Orchestrates the advisor tool loop for non-native providers."""

    def can_handle(
        self,
        tools: Optional[List[Dict]],
        custom_llm_provider: Optional[str],
    ) -> bool:
        if not tools:
            return False
        has_advisor = any(t.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE for t in tools)
        is_non_native = custom_llm_provider not in ADVISOR_NATIVE_PROVIDERS
        return has_advisor and is_non_native

    async def handle(
        self,
        *,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        stream: Optional[bool],
        max_tokens: int,
        custom_llm_provider: Optional[str],
        **kwargs,
    ) -> Union[AnthropicMessagesResponse, AsyncIterator]:
        from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
            FakeAnthropicMessagesStreamIterator,
        )

        # Extract advisor tool config.
        advisor_tool = next(
            (t for t in (tools or []) if t.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE),
            None,
        )
        if advisor_tool is None:
            # Not caller input: can_handle() already found this tool, so its absence here is
            # an internal contract violation and a 500 is the honest answer.
            raise ValueError(f"handle() called but no {ANTHROPIC_ADVISOR_TOOL_TYPE} tool found in tools list")
        # [ARC-BUG-46] Everything below validates CALLER input, so it must be a 400.
        #
        # A bare ValueError carries no `status_code`, so the proxy's
        # `getattr(e, "status_code", 500)` reported a malformed request body as a server
        # fault: HTTP 500 plus a High-severity `llm_exceptions` alert, on every such
        # request. Measured on dev-ai: a request whose advisor tool omits `model` returned
        # 500 and paged, while the same request with `model` present returned 200.
        #
        # This surfaced only once ARC-BUG-44 and ARC-BUG-45 let the advisor tool reach this
        # code at all. Before them it was flattened into a generic tool upstream of the gate
        # and orchestration was skipped, so these checks never ran. Fixing the earlier two
        # is what made this reachable — not a new defect they introduced.
        advisor_model = advisor_tool.get("model")
        if not isinstance(advisor_model, str) or not advisor_model.strip():
            raise BadRequestError(
                message=(
                    "advisor tool definition must include a non-empty string 'model' field naming the "
                    "advisor model. It is required by the Anthropic advisor tool spec and has no default."
                ),
                model=model,
                llm_provider=custom_llm_provider or "anthropic",
            )
        _raw_max_uses = advisor_tool.get("max_uses")
        try:
            max_uses: int = ADVISOR_MAX_USES if _raw_max_uses is None else int(_raw_max_uses)
        except (TypeError, ValueError):
            # `int("abc")` and `int(None)` both raise, and both are caller input.
            raise BadRequestError(
                message=(
                    f"advisor tool 'max_uses' must be an integer, got {type(_raw_max_uses).__name__}: {_raw_max_uses!r}"
                ),
                model=model,
                llm_provider=custom_llm_provider or "anthropic",
            )
        advisor_api_key, advisor_api_base = _resolve_advisor_credentials(advisor_tool)

        # Build the synthetic tool definition the provider will receive.
        synthetic_advisor_tool = _make_synthetic_advisor_tool()

        # Executor tools = all original tools with advisor replaced by the synthetic one.
        executor_tools: List[Dict] = [
            (synthetic_advisor_tool if t.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE else t) for t in (tools or [])
        ]

        # Strip prior advisor blocks from history, preserving advice text as context.
        current_messages: List[Dict] = strip_advisor_blocks_from_messages(
            [dict(m) for m in messages], replace_with_text=True
        )

        parent_request_id: str = str(kwargs.pop("litellm_call_id", None) or uuid.uuid4())
        metadata_base: Dict = dict(kwargs.pop("metadata", None) or {})
        iteration = 0

        # [ARC-BUG-16] Resolve the EXECUTOR alias too, for the same reason as the advisor
        # leg below: this loop reaches _call_messages_handler directly, so nothing between
        # here and the provider re-resolves the alias.
        #
        # This became load-bearing the moment the gate started resolving providers. Before,
        # a Bedrock-routed alias was mis-gated as advisor-native and orchestration was
        # skipped entirely, so this leg never ran with a non-Anthropic provider. Now it
        # does — and dispatching the bare alias with custom_llm_provider="bedrock" makes the
        # Bedrock transform build its URL from the alias string
        # (.../model/claude-sonnet-4-6/invoke), which 400s, and drops the deployment's
        # region so it silently defaults to us-east-1. Fixing the gate without this would
        # trade a leaked tool_use for a hard failure on the same request class.
        resolved_executor_model, executor_routing_params = _resolve_advisor_model_via_router(model)
        # An explicit kwarg from the caller wins over a deployment default.
        executor_routing_params = {k: v for k, v in executor_routing_params.items() if k not in kwargs}

        while True:
            # --- Executor call (always non-streaming) ---
            executor_response: AnthropicMessagesResponse = await _call_messages_handler(
                model=resolved_executor_model,
                messages=current_messages,
                tools=executor_tools,
                stream=False,
                max_tokens=max_tokens,
                custom_llm_provider=custom_llm_provider,
                metadata={
                    **metadata_base,
                    "advisor_sub_call": False,
                    "parent_request_id": parent_request_id,
                },
                **executor_routing_params,
                **kwargs,
            )

            advisor_use_block = _find_advisor_tool_use(executor_response)

            if advisor_use_block is None:
                # No more advisor calls — this is the final response.
                if stream:
                    return FakeAnthropicMessagesStreamIterator(executor_response)
                return executor_response

            iteration += 1
            if iteration > max_uses:
                raise AdvisorMaxIterationsError(
                    f"Advisor orchestration loop exceeded max_uses={max_uses}. "
                    "Increase max_uses in the advisor tool definition or cap the request."
                )

            # --- Build advisor context ---
            advisor_messages = _build_advisor_context(current_messages, executor_response, advisor_use_block)

            # --- Advisor sub-call (always non-streaming, no tools) ---
            # [ARC-BUG-16] Resolve the advisor alias through the router before calling.
            # This leg bypasses the router entirely, so leaving custom_llm_provider=None
            # made litellm name-infer the provider from the alias: "claude-opus-4-8"
            # infers "anthropic" and the sub-call executed against Anthropic-direct even
            # when the alias maps to a Bedrock deployment. Unlike the gate below, this one
            # reproduces on the ordinary proxy path, because nothing upstream pre-resolves
            # it. Falls back to the bare alias when no router is configured (SDK use).
            # The allowlist already excludes api_base/api_key, which are passed explicitly
            # below, so no defensive pop is needed here.
            resolved_advisor_model, advisor_routing_params = _resolve_advisor_model_via_router(advisor_model)
            advisor_response: AnthropicMessagesResponse = await _call_messages_handler(
                model=resolved_advisor_model,
                messages=advisor_messages,
                tools=None,
                stream=False,
                max_tokens=max_tokens,
                custom_llm_provider=None,  # resolved from the model name above
                metadata={
                    **metadata_base,
                    "advisor_sub_call": True,
                    "parent_request_id": parent_request_id,
                },
                api_key=advisor_api_key,
                api_base=advisor_api_base,
                **advisor_routing_params,
            )

            advisor_text = _extract_response_text(advisor_response)

            # --- Inject advisor result and continue loop ---
            current_messages = _inject_advisor_turn(
                current_messages,
                executor_response,
                advisor_use_block,
                advisor_text,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _allow_client_side_advisor_credentials() -> bool:
    """Whether a caller-supplied advisor api_base/api_key may be honored.

    Gated on the proxy's ``allow_client_side_credentials`` opt-in. When the
    interceptor runs outside the proxy (SDK use), there is no admin boundary
    to protect, so client-supplied routing is allowed.
    """
    try:
        from litellm.proxy.proxy_server import general_settings
    except (ImportError, ModuleNotFoundError):
        return True
    return general_settings.get("allow_client_side_credentials") is True


def _resolve_advisor_credentials(advisor_tool: dict) -> tuple[Optional[str], Optional[str]]:
    """Resolve the (api_key, api_base) override for the advisor sub-call.

    A caller-supplied ``api_base`` is only honored alongside a caller-supplied
    ``api_key``: without one, ``AnthropicModelInfo.get_auth_header()`` falls
    back to the proxy's own Anthropic credentials, which would then be sent to
    the caller-chosen ``api_base``. A caller-supplied ``api_base`` is also
    required to be https with TLS verification on, and SSRF-validated so it
    can't target a private/internal/cloud-metadata address, mirroring
    ``proxy.auth.auth_utils.check_complete_credentials``. https with TLS
    verification is required because ``validate_url`` only rewrites the
    connection to a DNS-pinned IP for http, or for https with
    ``litellm.ssl_verify`` disabled; otherwise it returns the URL unchanged
    and relies on certificate validation to block DNS rebinding, so this
    closes the same gap without threading the pinned URL through the whole
    ``anthropic_messages()`` call chain.
    """
    if not _allow_client_side_advisor_credentials():
        return None, None
    api_key: Optional[str] = advisor_tool.get("api_key")
    api_base: Optional[str] = advisor_tool.get("api_base")
    if api_base is None:
        return api_key, None
    if not api_key:
        raise ValueError(
            "advisor tool definition sets 'api_base' without 'api_key'. A "
            "caller-supplied api_base is only honored alongside a "
            "caller-supplied api_key, so the proxy's own credentials are "
            "never sent to a caller-chosen destination."
        )
    if not api_base.startswith("https://"):
        raise ValueError(f"advisor tool definition sets 'api_base'={api_base!r}, which must use the https scheme.")
    if getattr(litellm, "ssl_verify", True) is False:
        raise ValueError(
            "advisor tool definition sets 'api_base' but the proxy has TLS verification "
            "disabled (litellm.ssl_verify=False), so a caller-supplied api_base can't be "
            "safely validated against DNS rebinding."
        )
    if getattr(litellm, "user_url_validation", True):
        validate_url(api_base)
    return api_key, api_base


# Deployment params the advisor sub-call may inherit, as an ALLOWLIST.
#
# A deny-list was tried first and is the wrong default: a deployment's litellm_params also
# carry its credentials, so "everything except the reserved names" forwarded
# aws_access_key_id, aws_secret_access_key, azure_ad_token and vertex_credentials into the
# call kwargs. They do not reach the spend log — StandardLoggingPayload.model_parameters is
# itself allowlisted — but any custom logger reads the raw litellm_params dict, and an
# exception path can stringify them.
#
# So enumerate what the sub-call actually needs to land on the right endpoint: region and
# API version. Credentials are resolved by the provider from the environment or IRSA, the
# same way the executor leg gets them.
_ADVISOR_ROUTING_PARAM_ALLOWLIST = frozenset(
    {
        "aws_region_name",
        "vertex_location",
        "vertex_project",
        "api_version",
    }
)


def _resolve_advisor_model_via_router(advisor_model: str) -> tuple[str, dict]:
    """Resolve a proxy ``model_list`` alias to its real deployment.

    Behind the proxy an alias such as ``"claude-opus-4-8"`` only maps to a concrete
    deployment (``"bedrock/us.anthropic.claude-opus-4-8"`` plus region and credentials)
    through the router. The advisor sub-call goes straight to ``anthropic_messages()`` and
    bypasses the router, so a bare alias either fails ``get_llm_provider()`` outright or —
    worse, and the reason ARC-BUG-16 exists — name-infers to ``anthropic`` and silently
    executes against Anthropic-direct instead of the Bedrock deployment the request was
    routed to.

    Returns ``(model, extra_litellm_params)``, falling back to the original alias and an
    empty dict when no router is configured or the alias is unknown. That keeps SDK and
    native-provider behaviour unchanged, where a bare model resolves on its own.
    """
    try:
        from litellm.proxy.proxy_server import llm_router
    except Exception:
        return advisor_model, {}
    if llm_router is None:
        return advisor_model, {}
    try:
        deployment = llm_router.get_available_deployment(model=advisor_model)
    except Exception as e:
        # Logged, unlike the earlier version: this fallback sends the sub-call to whatever
        # the bare alias name-infers, which is the misrouting ARC-BUG-16 exists to prevent.
        verbose_logger.warning(
            "advisor sub-call: could not resolve %s through the router, falling back to the "
            "bare alias — it may execute against the wrong provider (%s: %s)",
            advisor_model,
            type(e).__name__,
            e,
        )
        return advisor_model, {}
    if not deployment:
        return advisor_model, {}
    litellm_params = dict(deployment.get("litellm_params") or {})
    resolved_model = litellm_params.pop("model", None) or advisor_model
    # Allowlisted routing params only — see _ADVISOR_ROUTING_PARAM_ALLOWLIST for why this
    # is not a deny-list.
    extra = {k: v for k, v in litellm_params.items() if k in _ADVISOR_ROUTING_PARAM_ALLOWLIST and v is not None}
    return resolved_model, extra


def resolve_advisor_gate_provider(
    model: str,
    custom_llm_provider: Optional[str],
    tools: Optional[List[Dict]],
) -> Optional[str]:
    """Provider to gate the advisor interceptor on (``can_handle``).

    A bare proxy alias like ``"claude-sonnet-4-6"`` name-infers to ``"anthropic"`` via
    ``get_llm_provider()`` but may map to a non-Anthropic deployment. Gating on the alias
    treats such a request as advisor-native and skips orchestration for Bedrock-routed
    aliases, leaking the advisor ``tool_use`` back to the client.

    Resolve the alias to its deployment provider when an advisor tool is present and the
    provider is either unset or a name-inferred ``anthropic``; otherwise return
    ``custom_llm_provider`` unchanged.

    Resolution deliberately does NOT go through ``get_available_deployment``. That is the
    selection API: it load-balances, so on an alias fronting deployments from two providers
    the gate decision would flip per request (measured: 23 anthropic / 17 bedrock over 40
    identical calls), and under usage-based routing it performs synchronous Redis reads
    inside this coroutine. The gate needs the alias's provider, not a load-balanced pick, so
    it reads the configured deployment list instead — in-memory, order-independent, and free
    of routing side effects.

    When an alias spans more than one provider and they disagree, resolve to a non-native
    one so orchestration runs. Skipping it is the failure this fix exists to prevent, and
    running it against a provider that supports advisor natively is merely redundant.

    The fallback returns the alias-inferred provider, i.e. it degrades back to the behaviour
    being fixed. That is the safe direction for a routing decision but it is silent, so an
    unexpected failure logs at warning; a simply-absent router stays at debug.
    """
    if custom_llm_provider not in (None, "", "anthropic"):
        return custom_llm_provider
    if not tools or not any(isinstance(t, dict) and t.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE for t in tools):
        return custom_llm_provider
    try:
        from litellm.proxy.proxy_server import llm_router
    except Exception:
        return custom_llm_provider
    if llm_router is None:
        verbose_logger.debug(
            "resolve_advisor_gate_provider: no router configured, gating on the alias provider for %s",
            model,
        )
        return custom_llm_provider
    try:
        deployments = llm_router.get_model_list(model_name=model) or []
        providers = set()
        for deployment in deployments:
            deployment_model = (deployment.get("litellm_params") or {}).get("model")
            if not deployment_model:
                continue
            _, provider, _, _ = litellm.get_llm_provider(model=deployment_model)
            if provider:
                providers.add(provider)
        if not providers:
            return custom_llm_provider
        non_native = sorted(providers - ADVISOR_NATIVE_PROVIDERS)
        if non_native:
            return non_native[0]
        return sorted(providers)[0]
    except Exception as e:
        verbose_logger.warning(
            "resolve_advisor_gate_provider: could not resolve %s, falling back to the alias "
            "provider — advisor orchestration may be skipped for a non-native deployment (%s: %s)",
            model,
            type(e).__name__,
            e,
        )
        return custom_llm_provider


def _make_synthetic_advisor_tool() -> Dict:
    """Build a regular tool definition the executor provider can understand."""
    return {
        "name": "advisor",
        "description": ADVISOR_TOOL_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question or challenge you want guidance on.",
                }
            },
            "required": ["question"],
        },
    }


def _find_advisor_tool_use(response: Any) -> Optional[Dict]:
    """Return the first tool_use block with name='advisor', or None."""
    content = response.get("content") if isinstance(response, dict) else []
    if not isinstance(content, list):
        return None
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "advisor":
            return block
    return None


def _extract_response_text(response: Any) -> str:
    """Extract concatenated text from all text blocks in a response."""
    content = response.get("content") if isinstance(response, dict) else []
    if not isinstance(content, list):
        return ""
    parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(parts).strip()


_PROVIDER_SPECIFIC_KEYS = frozenset({"provider_specific_fields"})


def _build_advisor_context(
    messages: List[Dict],
    executor_response: Any,
    advisor_use_block: Dict,
) -> List[Dict]:
    """
    Build the message list for the advisor sub-call.

    Passes the full conversation + any text the executor produced so far, then
    poses the advisor question as the last user turn.

    tool_use blocks are excluded because Anthropic requires tool_use to be
    immediately followed by tool_result — not the advisor question.
    """
    question = (advisor_use_block.get("input") or {}).get("question") or (
        "Please provide guidance on the current task."
    )
    raw_content = (executor_response.get("content") if isinstance(executor_response, dict) else []) or []
    # Keep only text blocks — strip tool_use and provider-specific fields.
    executor_text_blocks = [
        {k: v for k, v in block.items() if k not in _PROVIDER_SPECIFIC_KEYS}
        for block in raw_content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    result = list(messages)
    if executor_text_blocks:
        result.append({"role": "assistant", "content": executor_text_blocks})
    result.append({"role": "user", "content": question})
    return result


def _inject_advisor_turn(
    messages: List[Dict],
    executor_response: Any,
    advisor_use_block: Dict,
    advisor_text: str,
) -> List[Dict]:
    """
    Append the executor's response (as an assistant turn) and the advisor
    result (as a user tool_result turn) so the executor can continue.
    """
    executor_content = (executor_response.get("content") if isinstance(executor_response, dict) else []) or []
    tool_use_id = advisor_use_block.get("id", "")
    return [
        *messages,
        {"role": "assistant", "content": executor_content},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": advisor_text,
                }
            ],
        },
    ]


def _inject_max_uses_error(
    messages: List[Dict],
    executor_response: Any,
    advisor_use_block: Dict,
) -> List[Dict]:
    """
    Inject a max_uses_exceeded error tool_result so the executor continues
    without further advisor calls (mirrors Anthropic's server-side behaviour).
    """
    executor_content = (executor_response.get("content") if isinstance(executor_response, dict) else []) or []
    tool_use_id = advisor_use_block.get("id", "")
    return [
        *messages,
        {"role": "assistant", "content": executor_content},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": "Advisor unavailable: max_uses limit reached. Continue without advisor guidance.",
                }
            ],
        },
    ]


async def _call_messages_handler(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    stream: bool,
    max_tokens: int,
    custom_llm_provider: Optional[str],
    **kwargs,
) -> Any:
    """
    Call anthropic_messages() — the public async /messages entry point — for
    orchestration sub-calls (executor or advisor).

    Using the public function (decorated with @client) ensures logging, retries,
    and provider resolution all work correctly, identical to a direct user call.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages,
    )

    return await anthropic_messages(
        model=model,
        messages=messages,
        tools=tools,
        stream=stream,
        max_tokens=max_tokens,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )
