"""Review findings against arcadia-v1.94.0 (bca9dabfa5): advisor tool versioning + 400 contract.

Two defects found reviewing the ARC-BUG-16/44/45/46 stack before the prod cutover. Neither
is a regression introduced by that stack; both are gaps it left behind.

FINDING 1 — the advisor tool type is matched by PREFIX on one leg and EXACTLY on another.

ARC-BUG-45 established that Anthropic versions server tools by dated suffix and introduced
``_ANTHROPIC_ADVISOR_TOOL_PREFIX`` so the adapter keeps a future ``advisor_20260302`` on the
Anthropic-native path. It introduced it privately, in adapters/transformation.py, so the
interceptor's gate, ``can_handle`` and ``handle`` went on comparing against the dated
constant. The day the vendor ships the next version, the adapter keeps the tool native and
the gate declines to orchestrate it — which is exactly the ARC-BUG-44 symptom, a raw
``tool_use`` reaching the client, on a calendar trigger rather than a config one.

FINDING 2 — ARC-BUG-46's 400 contract stops one call short.

``handle()`` validates ``model`` and ``max_uses`` as BadRequestError, but delegates the
advisor tool's ``api_base``/``api_key`` to ``_resolve_advisor_credentials``, which raised
bare ``ValueError``. Those carry no ``status_code``, so the proxy's
``getattr(e, "status_code", 500)`` reports malformed caller input as a server fault and
pages. Unreachable while ``allow_client_side_credentials`` is unset — which is why the
original five-shape validation did not find it — and silently reachable the moment it is
turned on.
"""

import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.exceptions import BadRequestError
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    _ANTHROPIC_ADVISOR_TOOL_PREFIX,
)
from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
    AdvisorOrchestrationHandler,
    _resolve_advisor_credentials,
    resolve_advisor_gate_provider,
)
from litellm.types.llms.anthropic import (
    ANTHROPIC_ADVISOR_TOOL_PREFIX,
    ANTHROPIC_ADVISOR_TOOL_TYPE,
    is_advisor_tool,
)

# A plausible next dated version of the advisor tool. The point of every test below is that
# nothing in the interceptor may key on the CURRENT date.
FUTURE_ADVISOR_TOOL_TYPE = "advisor_20260601"


def _bedrock_router():
    """A proxy router whose alias fronts a single Bedrock deployment."""
    router = MagicMock()
    deployment = {"litellm_params": {"model": "bedrock/us.anthropic.claude-sonnet-4-6"}}
    router.get_available_deployment.return_value = deployment
    router.get_model_list.return_value = [deployment]
    return router


@contextmanager
def proxy_router(router):
    """Install ``router`` as the proxy's ``llm_router``.

    Injects a stand-in module rather than patching the real one. The code under test does
    ``from litellm.proxy.proxy_server import llm_router`` at call time, so a sys.modules
    entry is enough — and unlike ``patch("litellm.proxy.proxy_server.llm_router")`` it does
    not require the proxy's optional dependencies (fastapi_sso and friends) to be
    installed. Without that, these tests only run where the full proxy extras are present.
    """
    module = types.ModuleType("litellm.proxy.proxy_server")
    module.llm_router = router  # type: ignore[attr-defined]
    module.general_settings = {}  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": module}):
        yield


# --------------------------------------------------------------------------------------
# FINDING 1 — one predicate, both legs
# --------------------------------------------------------------------------------------


def test_both_legs_derive_the_same_prefix():
    """The adapter's stem and the shared stem must be the same object, not two derivations.

    Two independent ``rsplit`` calls would agree today and could silently diverge later;
    this asserts the adapter re-exports rather than re-derives.
    """
    assert _ANTHROPIC_ADVISOR_TOOL_PREFIX == ANTHROPIC_ADVISOR_TOOL_PREFIX
    assert ANTHROPIC_ADVISOR_TOOL_TYPE.startswith(ANTHROPIC_ADVISOR_TOOL_PREFIX)


@pytest.mark.parametrize(
    "tool, expected",
    [
        ({"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "model": "claude-opus-4-8"}, True),
        ({"type": FUTURE_ADVISOR_TOOL_TYPE, "model": "claude-opus-4-8"}, True),
        ({"type": "web_search_20250305"}, False),
        ({"type": "custom", "name": "advisor"}, False),
        # "advisor" with no dated suffix is not a server tool: the prefix carries the
        # underscore precisely so a plain custom tool named "advisor" cannot match.
        ({"type": "advisor"}, False),
        ({"type": None}, False),
        ({}, False),
        ("not-a-dict", False),
        (None, False),
    ],
)
def test_is_advisor_tool_accepts_any_dated_version_and_no_impostors(tool, expected):
    assert is_advisor_tool(tool) is expected


def test_can_handle_fires_for_a_future_dated_advisor_tool():
    """The gate must orchestrate the next version, not skip it.

    Skipping is what leaks the raw tool_use to the client — ARC-BUG-44's symptom.
    """
    handler = AdvisorOrchestrationHandler()
    tools = [{"type": FUTURE_ADVISOR_TOOL_TYPE, "model": "claude-opus-4-8"}]

    assert handler.can_handle(tools=tools, custom_llm_provider="bedrock") is True
    # ...and still declines when the provider handles advisor natively.
    assert handler.can_handle(tools=tools, custom_llm_provider="anthropic") is False


def test_gate_resolves_the_deployment_provider_for_a_future_dated_advisor_tool():
    """resolve_advisor_gate_provider() must recognise the tool before it will resolve.

    With an exact-match check this returned the alias-inferred "anthropic" for a
    Bedrock-routed alias — the ARC-BUG-16 defect, reintroduced by a date change.
    """
    router = _bedrock_router()
    tools = [{"type": FUTURE_ADVISOR_TOOL_TYPE, "model": "claude-opus-4-8"}]

    with proxy_router(router):
        provider = resolve_advisor_gate_provider(
            model="claude-sonnet-4-6",
            custom_llm_provider="anthropic",
            tools=tools,
        )

    assert provider == "bedrock"


def test_gate_still_ignores_a_request_carrying_no_advisor_tool():
    """Negative control: the loosened predicate must not widen the gate's trigger."""
    router = _bedrock_router()

    with proxy_router(router):
        provider = resolve_advisor_gate_provider(
            model="claude-sonnet-4-6",
            custom_llm_provider="anthropic",
            tools=[{"type": "web_search_20250305"}],
        )

    assert provider == "anthropic"
    router.get_model_list.assert_not_called()


# --------------------------------------------------------------------------------------
# FINDING 2 — the 400 contract covers the credential paths too
# --------------------------------------------------------------------------------------


@pytest.fixture
def client_side_credentials_enabled():
    """Enable the opt-in that makes the credential paths reachable at all."""
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor."
        "_allow_client_side_advisor_credentials",
        return_value=True,
    ):
        yield


def test_api_base_without_api_key_is_a_400(client_side_credentials_enabled):
    with pytest.raises(BadRequestError) as exc:
        _resolve_advisor_credentials(
            {"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "api_base": "https://example.com"},
            model="claude-sonnet-4-6",
            custom_llm_provider="bedrock",
        )

    assert exc.value.status_code == 400
    assert "api_key" in str(exc.value)


def test_non_https_api_base_is_a_400(client_side_credentials_enabled):
    with pytest.raises(BadRequestError) as exc:
        _resolve_advisor_credentials(
            {"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "api_base": "http://example.com", "api_key": "sk-x"},
            model="claude-sonnet-4-6",
            custom_llm_provider="bedrock",
        )

    assert exc.value.status_code == 400
    assert "https" in str(exc.value)


def test_api_base_with_tls_verification_disabled_is_a_400(client_side_credentials_enabled, monkeypatch):
    monkeypatch.setattr(litellm, "ssl_verify", False, raising=False)

    with pytest.raises(BadRequestError) as exc:
        _resolve_advisor_credentials(
            {"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "api_base": "https://example.com", "api_key": "sk-x"},
            model="claude-sonnet-4-6",
            custom_llm_provider="bedrock",
        )

    assert exc.value.status_code == 400
    assert "ssl_verify" in str(exc.value)


def test_credential_paths_stay_unreachable_when_the_opt_in_is_off():
    """The deployed configuration's behaviour: no opt-in, no credential handling at all.

    This is why the original five-shape validation could report a complete 400 contract —
    the sixth through eighth shapes were gated off, not fixed.
    """
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor."
        "_allow_client_side_advisor_credentials",
        return_value=False,
    ):
        assert _resolve_advisor_credentials(
            {"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "api_base": "http://example.com"},
            model="claude-sonnet-4-6",
        ) == (None, None)


def test_helper_keeps_working_without_the_new_arguments(client_side_credentials_enabled):
    """SDK callers pass the tool alone; the added params must stay optional."""
    assert _resolve_advisor_credentials({"type": ANTHROPIC_ADVISOR_TOOL_TYPE}) == (None, None)

    with pytest.raises(BadRequestError):
        _resolve_advisor_credentials({"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "api_base": "https://example.com"})
