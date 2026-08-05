"""[ARC-BUG-40] Do not call Bedrock CountTokens for models that cannot answer it.

Measured against live Bedrock in us-east-1 on 2026-08-05, every Anthropic model in the prd-ai
config, in both the bare and the `us.`-prefixed form:

    anthropic.claude-sonnet-5                    ValidationException: doesn't support counting tokens
    us.anthropic.claude-sonnet-5                 ValidationException: doesn't support counting tokens
    us.anthropic.claude-sonnet-4-6               ValidationException: doesn't support counting tokens
    us.anthropic.claude-sonnet-4-5-20250929-v1:0 ValidationException: doesn't support counting tokens
    us.anthropic.claude-haiku-4-5-20251001-v1:0  ValidationException: doesn't support counting tokens
    us.anthropic.claude-opus-4-7 / -4-8 / -5     ValidationException: doesn't support counting tokens
    us.anthropic.claude-fable-5                  ValidationException: doesn't support counting tokens

Not one of them works. Every call was guaranteed to 400: 82 wasted round-trips per pod per 4.5h
and ~246 log lines, about 37% of one prod pod's volume.

⚠️ An earlier version of this docstring claimed those 400s arm the router's cooldown. That is
FALSE and was corrected by review: the CountTokens path never reaches deployment failure
accounting (no logging_obj / failure_callback anywhere under
litellm/llms/bedrock/count_tokens/), and _is_cooldown_required returns False for 400 regardless.
The cooldown trigger is litellm.Timeout at status 408. This patch does not make arming
allowed_fails safer.

Callers see no change: the 400 already fell through to the local tokenizer, so every Anthropic
count on prod comes from it today. This removes the wasted call, not a number anyone relies on.
"""

import pytest

from litellm.llms.bedrock.count_tokens.bedrock_token_counter import BedrockTokenCounter


@pytest.fixture
def counter():
    return BedrockTokenCounter()


class TestUnsupportedAnthropicModelsAreSkipped:
    """Every model id this proxy actually routes to, in the form the router hands over."""

    @pytest.mark.parametrize(
        "model",
        [
            "bedrock/us.anthropic.claude-sonnet-5",
            "bedrock/us.anthropic.claude-opus-5",
            "bedrock/us.anthropic.claude-opus-4-8",
            "bedrock/us.anthropic.claude-opus-4-7",
            "bedrock/us.anthropic.claude-fable-5",
            "bedrock/us.anthropic.claude-sonnet-4-6",
            "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        ],
    )
    def test_no_countokens_call_is_attempted(self, counter, model):
        assert counter.should_use_token_counting_api("bedrock", model=model) is False

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic.claude-sonnet-5",
            "us.anthropic.claude-sonnet-5",
            "bedrock/us.anthropic.claude-sonnet-5",
            "bedrock/converse/us.anthropic.claude-sonnet-5",
            "converse/us.anthropic.claude-sonnet-5",
        ],
    )
    def test_every_prefix_form_of_one_model_is_skipped(self, counter, model):
        """The prefix is NOT the defect, and this pins that.

        An earlier branch (bugfix/bedrock-count-tokens-inference-profile) tried adding the `us.`
        prefix and was correctly abandoned: inference-profile ids are rejected too, for every
        model. Both forms fail identically against live Bedrock, so the skip must not depend on
        which prefix the caller happens to pass.
        """
        assert counter.should_use_token_counting_api("bedrock", model=model) is False


class TestSupportedModelsAreStillAttempted:
    """The skip must be narrow. A denylist that swallows everything is worse than the 400s."""

    @pytest.mark.parametrize(
        "model",
        [
            "bedrock/us.meta.llama3-2-11b-instruct-v1:0",
            "bedrock/amazon.titan-embed-text-v2:0",
            "bedrock/amazon.nova-pro-v1:0",
            "bedrock/mistral.mistral-large-2407-v1:0",
            "bedrock/cohere.command-r-plus-v1:0",
        ],
    )
    def test_non_anthropic_bedrock_models_still_try(self, counter, model):
        assert counter.should_use_token_counting_api("bedrock", model=model) is True

    def test_the_denylist_direction_is_deliberate(self, counter):
        """A denylist, not an allowlist, and the difference matters over time.

        Bedrock adds CountTokens support as models mature. An allowlist would keep silently
        skipping a model the day support arrives, and nobody would notice — the fallback to the
        local tokenizer is invisible except as an undercount. A denylist degrades the other way:
        a newly-supported model starts working on its own, and a newly-launched unsupported one
        costs one wasted call until it is listed.

        Asserted through a hypothetical future family rather than a real id, so this test states
        the intent and does not need editing when Bedrock's catalogue changes.
        """
        assert counter.should_use_token_counting_api("bedrock", model="bedrock/newvendor.somemodel-v1:0") is True


class TestBackwardCompatibility:
    """Six provider implementations share this signature and two call sites use it."""

    def test_provider_only_keeps_the_original_behaviour(self, counter):
        """Without a model the check must NOT skip everything — it degrades to provider-only.

        A caller that has not been updated would otherwise silently stop counting tokens for
        every Bedrock model, which is a much larger regression than the 400s being fixed.
        """
        assert counter.should_use_token_counting_api("bedrock") is True
        assert counter.should_use_token_counting_api("bedrock", model=None) is True

    def test_a_different_provider_is_still_declined(self, counter):
        assert counter.should_use_token_counting_api("openai", model="bedrock/us.anthropic.claude-sonnet-5") is False
        assert counter.should_use_token_counting_api(None) is False

    def test_the_base_signature_carries_the_model_kwarg(self):
        """The widened base signature is what lets the six sibling implementations keep working.

        Asserted on the signature rather than by instantiating: BaseTokenCounter declares both
        methods abstract, so a bare subclass cannot be constructed without implementing the very
        method under test — which would then be testing the stub, not the base.
        """
        import inspect

        from litellm.llms.base_llm.base_utils import BaseTokenCounter

        sig = inspect.signature(BaseTokenCounter.should_use_token_counting_api)
        assert "model" in sig.parameters, "the base must accept the model kwarg"
        assert sig.parameters["model"].default is None, "model must be optional so old callers work"
        assert sig.parameters["custom_llm_provider"].default is None

    @pytest.mark.parametrize(
        "impl_path",
        [
            "litellm.llms.openai.responses.count_tokens.token_counter",
            "litellm.llms.anthropic.count_tokens.token_counter",
            "litellm.llms.gemini.common_utils",
            "litellm.llms.vertex_ai.common_utils",
            "litellm.llms.azure_ai.anthropic.count_tokens.token_counter",
        ],
    )
    def test_every_sibling_accepts_the_model_kwarg_without_raising(self, impl_path):
        """Every other provider's override must survive the widened call, not just tolerate it.

        🔴 This caught a real defect in the first version of this patch. main.py now passes
        `model=` to whichever counter the provider resolves to, and five of the six
        implementations declared only `custom_llm_provider`. The result was NOT an exception the
        caller saw: litellm.acount_tokens swallowed the TypeError and fell through to the local
        tokenizer, so token counts were silently wrong for EVERY provider. Four tests in
        test_count_tokens_public_api.py failed with off-by-a-few counts — 13 instead of 15 — which
        reads like a tokenizer nit rather than a broken call path.

        Asserted by actually CALLING with the kwarg rather than by inspecting the signature. The
        first version of this test only checked that `custom_llm_provider` was still present,
        which every implementation passed while all five were broken.
        """
        import importlib
        import inspect

        mod = importlib.import_module(impl_path)
        counters = [
            obj
            for _, obj in inspect.getmembers(mod, inspect.isclass)
            if hasattr(obj, "should_use_token_counting_api") and obj.__module__ == impl_path
        ]
        assert counters, f"no token counter found in {impl_path}"
        for cls in counters:
            try:
                cls().should_use_token_counting_api("bedrock", model="bedrock/us.anthropic.claude-sonnet-5")
            except TypeError as e:
                pytest.fail(f"{cls.__name__}.should_use_token_counting_api rejects model=: {e}")
            except Exception:
                # Any other exception is the implementation's own business; only the signature
                # is under test here.
                pass
