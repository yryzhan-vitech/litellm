"""[ARC-BUG-16] The advisor gate and sub-call must resolve a proxy alias to its deployment.

A bare model_list alias such as "claude-sonnet-4-6" name-infers to "anthropic" via
get_llm_provider(), but behind the proxy it may map to a bedrock/... deployment. Two
consequences, both fixed here:

  - the interceptor gate treated such a request as advisor-native and skipped
    orchestration, leaking the advisor tool_use back to the client
  - the advisor sub-call bypasses the router entirely, so it executed against
    Anthropic-direct instead of the Bedrock deployment the request was routed to
"""

from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
    _resolve_advisor_model_via_router,
    resolve_advisor_gate_provider,
)
from litellm.types.llms.anthropic import ANTHROPIC_ADVISOR_TOOL_TYPE

ADVISOR_TOOLS = [{"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "model": "claude-opus-4-8"}]


def _router_returning(model: str, **extra):
    """A stand-in proxy router whose deployment resolves `model`.

    Both APIs are stubbed: the gate reads the configured list (`get_model_list`) while the
    advisor sub-call still needs a concrete pick (`get_available_deployment`).
    """
    router = MagicMock()
    deployment = {"litellm_params": {"model": model, **extra}}
    router.get_available_deployment.return_value = deployment
    router.get_model_list.return_value = [deployment]
    return router


def _router_with_models(*models: str):
    """A router whose alias fronts several deployments."""
    router = MagicMock()
    router.get_model_list.return_value = [{"litellm_params": {"model": m}} for m in models]
    return router


class TestResolveAdvisorGateProvider:
    def test_explicit_non_anthropic_provider_passes_through_untouched(self):
        """An explicit provider is the caller's decision — never second-guess it."""
        assert resolve_advisor_gate_provider("m", "bedrock", ADVISOR_TOOLS) == "bedrock"
        assert resolve_advisor_gate_provider("m", "vertex_ai", ADVISOR_TOOLS) == "vertex_ai"

    @pytest.mark.parametrize(
        "tools",
        [None, [], [{"type": "function", "name": "x"}]],
        ids=["none", "empty", "no-advisor-tool"],
    )
    def test_without_an_advisor_tool_nothing_is_resolved(self, tools):
        """The resolution costs a router lookup, so only pay it when it can matter."""
        assert resolve_advisor_gate_provider("claude-sonnet-4-6", "anthropic", tools) == "anthropic"

    def test_alias_mapping_to_bedrock_resolves_to_bedrock(self):
        """The bug itself: alias says anthropic, deployment says bedrock."""
        with patch(
            "litellm.proxy.proxy_server.llm_router",
            _router_returning("bedrock/us.anthropic.claude-sonnet-4-6", aws_region_name="us-west-2"),
        ):
            assert resolve_advisor_gate_provider("claude-sonnet-4-6", "anthropic", ADVISOR_TOOLS) == "bedrock"

    def test_unset_provider_is_resolved_too(self):
        with patch(
            "litellm.proxy.proxy_server.llm_router",
            _router_returning("bedrock/us.anthropic.claude-sonnet-4-6"),
        ):
            assert resolve_advisor_gate_provider("claude-sonnet-4-6", None, ADVISOR_TOOLS) == "bedrock"

    def test_alias_mapping_to_anthropic_stays_anthropic(self):
        """Resolution must not invent a change where the deployment agrees with the alias."""
        with patch("litellm.proxy.proxy_server.llm_router", _router_returning("anthropic/claude-sonnet-4-6")):
            assert resolve_advisor_gate_provider("claude-sonnet-4-6", "anthropic", ADVISOR_TOOLS) == "anthropic"

    def test_router_failure_falls_back_to_the_alias_provider(self):
        """Fails in the direction of the bug, which is why it logs — but it must not raise:
        a cooled-down deployment would otherwise turn a routing hiccup into a 500."""
        router = MagicMock()
        router.get_model_list.side_effect = RuntimeError("no deployments available")
        with patch("litellm.proxy.proxy_server.llm_router", router):
            assert resolve_advisor_gate_provider("claude-sonnet-4-6", "anthropic", ADVISOR_TOOLS) == "anthropic"

    def test_a_mixed_provider_alias_resolves_deterministically_to_non_native(self):
        """get_available_deployment load-balances, so gating on it flipped the decision per
        request (measured 23 anthropic / 17 bedrock over 40 identical calls). Reading the
        configured list is order-independent, and when providers disagree we pick a
        non-native one: skipping orchestration is the bug, running it against a native
        provider is merely redundant."""
        router = _router_with_models("anthropic/claude-sonnet-4-6", "bedrock/us.anthropic.claude-sonnet-4-6")
        with patch("litellm.proxy.proxy_server.llm_router", router):
            results = {
                resolve_advisor_gate_provider("claude-sonnet-4-6", "anthropic", ADVISOR_TOOLS) for _ in range(40)
            }
        assert results == {"bedrock"}

    def test_the_gate_does_not_use_the_selection_api(self):
        """The selection API performs synchronous Redis reads under usage-based routing and
        would put them inside this request coroutine. The gate needs a provider, not a
        load-balanced pick."""
        router = _router_returning("bedrock/us.anthropic.claude-sonnet-4-6")
        with patch("litellm.proxy.proxy_server.llm_router", router):
            resolve_advisor_gate_provider("claude-sonnet-4-6", "anthropic", ADVISOR_TOOLS)
        assert router.get_model_list.called
        assert not router.get_available_deployment.called

    def test_an_alias_with_no_usable_deployments_falls_back(self):
        router = MagicMock()
        router.get_model_list.return_value = [{"litellm_params": {}}]
        with patch("litellm.proxy.proxy_server.llm_router", router):
            assert resolve_advisor_gate_provider("claude-sonnet-4-6", "anthropic", ADVISOR_TOOLS) == "anthropic"

    def test_an_unimportable_proxy_module_falls_back(self):
        """Outside the proxy the import itself fails; that must not raise."""
        import builtins

        real_import = builtins.__import__

        def _fail(name, *args, **kwargs):
            if name == "litellm.proxy.proxy_server":
                raise ImportError("no proxy here")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", _fail):
            assert resolve_advisor_gate_provider("claude-sonnet-4-6", "anthropic", ADVISOR_TOOLS) == "anthropic"

    def test_no_router_configured_is_left_alone(self):
        """SDK / native use: the bare model resolves on its own, so preserve behaviour."""
        with patch("litellm.proxy.proxy_server.llm_router", None):
            assert resolve_advisor_gate_provider("claude-sonnet-4-6", "anthropic", ADVISOR_TOOLS) == "anthropic"


class TestResolveAdvisorModelViaRouter:
    def test_returns_the_deployment_model_and_routing_params(self):
        with patch(
            "litellm.proxy.proxy_server.llm_router",
            _router_returning("bedrock/us.anthropic.claude-opus-4-8", aws_region_name="us-west-2"),
        ):
            model, extra = _resolve_advisor_model_via_router("claude-opus-4-8")
        assert model == "bedrock/us.anthropic.claude-opus-4-8"
        assert extra == {"aws_region_name": "us-west-2"}

    def test_only_allowlisted_routing_params_are_forwarded(self):
        """A deny-list forwarded the deployment's CREDENTIALS into the call kwargs — they do
        not reach the spend log, but a custom logger reads raw litellm_params and an
        exception path can stringify them. Enumerate what the sub-call needs instead."""
        with patch(
            "litellm.proxy.proxy_server.llm_router",
            _router_returning(
                "bedrock/us.anthropic.claude-opus-4-8",
                aws_region_name="us-west-2",
                api_version="2024-01-01",
                aws_access_key_id="AKIAEXAMPLE",
                aws_secret_access_key="shhh",
                azure_ad_token="tok",
                vertex_credentials="{}",
                api_key="sk-deployment",
                api_base="https://deployment.invalid",
                max_tokens=100,
                stream=True,
                metadata={"x": 1},
                custom_llm_provider="bedrock",
            ),
        ):
            _, extra = _resolve_advisor_model_via_router("claude-opus-4-8")
        assert set(extra) == {"aws_region_name", "api_version"}
        assert not [k for k in extra if any(s in k for s in ("key", "secret", "token", "credential"))]

    def test_none_valued_params_are_dropped(self):
        with patch(
            "litellm.proxy.proxy_server.llm_router",
            _router_returning("bedrock/m", aws_region_name=None, api_version="2024-01-01"),
        ):
            _, extra = _resolve_advisor_model_via_router("alias")
        assert extra == {"api_version": "2024-01-01"}

    @pytest.mark.parametrize("bad", [None, {}], ids=["no-deployment", "empty-deployment"])
    def test_unknown_alias_falls_back(self, bad):
        router = MagicMock()
        router.get_available_deployment.return_value = bad
        with patch("litellm.proxy.proxy_server.llm_router", router):
            assert _resolve_advisor_model_via_router("alias") == ("alias", {})

    def test_no_router_falls_back(self):
        with patch("litellm.proxy.proxy_server.llm_router", None):
            assert _resolve_advisor_model_via_router("alias") == ("alias", {})

    def test_unimportable_proxy_module_falls_back(self):
        """Outside the proxy the import itself raises; the helper must absorb that."""
        import builtins

        real_import = builtins.__import__

        def _fail(name, *args, **kwargs):
            if name == "litellm.proxy.proxy_server":
                raise ImportError("no proxy here")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", _fail):
            assert _resolve_advisor_model_via_router("alias") == ("alias", {})

    def test_a_raising_router_falls_back(self):
        """A cooled-down or misconfigured deployment must not turn the advisor sub-call
        into a 500 — it degrades to the bare alias, which is what SDK use does anyway."""
        router = MagicMock()
        router.get_available_deployment.side_effect = RuntimeError("no deployments available")
        with patch("litellm.proxy.proxy_server.llm_router", router):
            assert _resolve_advisor_model_via_router("alias") == ("alias", {})


class TestBothLegsAreResolved:
    """[ARC-BUG-16] Once the gate resolves providers, BOTH sub-calls must resolve models.

    The loop reaches _call_messages_handler directly, so nothing between it and the provider
    re-resolves an alias. Before the gate fix a Bedrock-routed alias was mis-gated as
    advisor-native and orchestration never ran, which hid this; fixing only the gate would
    have traded a leaked tool_use for a hard 400, because the Bedrock transform builds its
    URL from whatever model string it is handed.
    """

    @staticmethod
    def _router():
        deployments = {
            "claude-sonnet-4-6": {
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-sonnet-4-6-v1:0",
                    "aws_region_name": "us-east-2",
                }
            },
            "claude-opus-4-8": {
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-opus-4-8-v1:0",
                    "aws_region_name": "us-west-2",
                }
            },
        }
        router = MagicMock()
        router.get_available_deployment.side_effect = lambda model: deployments[model]
        router.get_model_list.return_value = [deployments["claude-sonnet-4-6"]]
        return router

    @pytest.mark.asyncio
    async def test_executor_and_advisor_legs_both_carry_a_resolved_model(self):
        import litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor as advisor_module

        calls: list = []

        async def _fake_handler(**kwargs):
            calls.append(
                {
                    "model": kwargs.get("model"),
                    "region": kwargs.get("aws_region_name"),
                    "sub_call": (kwargs.get("metadata") or {}).get("advisor_sub_call"),
                }
            )
            if len(calls) == 1:
                return {
                    "content": [{"type": "tool_use", "id": "t1", "name": "advisor", "input": {"question": "q"}}],
                    "stop_reason": "tool_use",
                }
            return {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}

        handler = advisor_module.AdvisorOrchestrationHandler()
        with (
            patch.object(advisor_module, "_call_messages_handler", _fake_handler),
            patch("litellm.proxy.proxy_server.llm_router", self._router()),
        ):
            await handler.handle(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "model": "claude-opus-4-8"}],
                stream=False,
                max_tokens=64,
                custom_llm_provider="bedrock",
            )

        executor = [c for c in calls if c["sub_call"] is False]
        advisor = [c for c in calls if c["sub_call"] is True]
        assert executor and advisor

        # No leg may dispatch a bare alias — that is what builds the wrong Bedrock URL.
        for call in calls:
            assert call["model"] not in ("claude-sonnet-4-6", "claude-opus-4-8"), call

        assert all(c["model"] == "bedrock/us.anthropic.claude-sonnet-4-6-v1:0" for c in executor)
        assert all(c["region"] == "us-east-2" for c in executor)
        assert all(c["model"] == "bedrock/us.anthropic.claude-opus-4-8-v1:0" for c in advisor)
        # Each leg gets ITS OWN deployment's region, not the other's.
        assert all(c["region"] == "us-west-2" for c in advisor)

    @pytest.mark.asyncio
    async def test_an_explicit_caller_kwarg_beats_a_deployment_default(self):
        import litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor as advisor_module

        seen: list = []

        async def _fake_handler(**kwargs):
            seen.append(kwargs.get("aws_region_name"))
            return {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}

        handler = advisor_module.AdvisorOrchestrationHandler()
        with (
            patch.object(advisor_module, "_call_messages_handler", _fake_handler),
            patch("litellm.proxy.proxy_server.llm_router", self._router()),
        ):
            await handler.handle(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "model": "claude-opus-4-8"}],
                stream=False,
                max_tokens=64,
                custom_llm_provider="bedrock",
                aws_region_name="eu-central-1",
            )

        assert seen == ["eu-central-1"], "a deployment default must not override the caller"


class TestHandlerWiring:
    """The gate has to be wired into the request path, not merely exist."""

    @pytest.mark.asyncio
    async def test_the_interceptor_receives_the_resolved_provider(self):
        """Reverting either the resolve call or the can_handle argument in handler.py left
        the whole suite green, so pin the wiring itself."""
        import litellm.llms.anthropic.experimental_pass_through.messages.handler as handler_module

        seen: dict = {}

        class _Recording:
            def can_handle(self, tools, custom_llm_provider):
                seen["can_handle"] = custom_llm_provider
                return True

            async def handle(self, **kwargs):
                seen["handle"] = kwargs.get("custom_llm_provider")
                return {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}

        router = MagicMock()
        router.get_model_list.return_value = [
            {"litellm_params": {"model": "bedrock/us.anthropic.claude-sonnet-4-6-v1:0"}}
        ]

        with (
            patch.object(handler_module, "get_messages_interceptors", lambda: [_Recording()]),
            patch("litellm.proxy.proxy_server.llm_router", router),
        ):
            await handler_module.anthropic_messages(
                max_tokens=64,
                messages=[{"role": "user", "content": "hi"}],
                model="claude-sonnet-4-6",
                tools=[{"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "model": "claude-opus-4-8"}],
                custom_llm_provider="anthropic",
            )

        assert seen["can_handle"] == "bedrock", "the gate must see the resolved provider"
        assert seen["handle"] == "bedrock", "and the interceptor must be handed it too"
