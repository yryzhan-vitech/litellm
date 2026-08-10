"""The advisor_interception subsystem must be importable AND registrable.

[ARC-BUG-52] All five of its files were absent from this branch while present on
fork `main` and on the production image `a2e1b93d30`, so a config carrying
`callbacks: ["advisor_interception"]` raised on startup rather than registering the
logger. The de-fork dropped the whole subsystem, not a hunk inside it — which is why
no diff-of-shared-files audit found it.

These tests cover the seam the drop broke: the import surface, and the one branch in
`callback_utils` that turns the config string into a live callback. The orchestration
logic itself is upstream's and is covered by its own suite.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm  # noqa: E402
from litellm.proxy.common_utils.callback_utils import initialize_callbacks_on_proxy  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_callbacks():
    """`initialize_callbacks_on_proxy` mutates module-level state."""
    saved = list(litellm.callbacks)
    yield
    litellm.callbacks = saved


def test_public_import_surface_is_intact():
    """What `__init__.py` promises must actually resolve."""
    from litellm.integrations.advisor_interception import (
        AdvisorInterceptionLogger,
        get_litellm_advisor_tool,
        get_litellm_advisor_tool_openai,
        is_advisor_tool,
    )
    from litellm.types.integrations.advisor_interception import AdvisorInterceptionConfig

    assert callable(is_advisor_tool)
    assert callable(get_litellm_advisor_tool)
    assert callable(get_litellm_advisor_tool_openai)
    assert AdvisorInterceptionConfig is not None
    assert hasattr(AdvisorInterceptionLogger, "initialize_from_proxy_config")


def test_the_config_string_registers_the_logger():
    """The regression: this raised before the port, because the module was missing."""
    from litellm.integrations.advisor_interception import AdvisorInterceptionLogger

    initialize_callbacks_on_proxy(
        value=["advisor_interception"],
        premium_user=True,
        config_file_path="",
        litellm_settings={"advisor_interception_params": {"default_advisor_model": "claude-opus-4-8"}},
    )

    assert any(isinstance(cb, AdvisorInterceptionLogger) for cb in litellm.callbacks), (
        "advisor_interception did not register as a callback"
    )


def test_it_stays_off_when_not_configured():
    """Opt-in only — a port must not change behaviour for the clusters that never ask.

    This is the assertion that makes the port safe to ship inside a bundle: prd-ai
    does not name this callback, so nothing about its request path moves.
    """
    from litellm.integrations.advisor_interception import AdvisorInterceptionLogger

    initialize_callbacks_on_proxy(
        value=["websearch_interception"], premium_user=True, config_file_path="", litellm_settings={}
    )

    assert not any(isinstance(cb, AdvisorInterceptionLogger) for cb in litellm.callbacks)


def test_the_messages_path_advisor_is_a_separate_mechanism():
    """Guard against the two advisor paths being conflated in future.

    `/v1/messages` orchestration lives in `interceptors/advisor.py` and is reached via
    `can_handle()`; this subsystem serves `/chat/completions`. They share the tool type
    and nothing else — merging them is how a fix for one silently alters the other.
    """
    from litellm.integrations.advisor_interception import is_advisor_tool as chat_predicate
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors import advisor as messages_path
    from litellm.types.llms.anthropic import is_advisor_tool as messages_predicate

    assert chat_predicate is not messages_predicate
    assert messages_path.AdvisorOrchestrationHandler is not None


def test_both_predicates_agree_on_the_dated_tool_type():
    """Different code, same answer — otherwise one path silently declines the tool.

    Only the shared contract is asserted (the dated `advisor_2*` type), because the two
    predicates legitimately accept different *shapes*: the messages path receives raw
    Anthropic tools, the chat path also handles OpenAI-converted ones.
    """
    from litellm.integrations.advisor_interception import is_advisor_tool as chat_predicate
    from litellm.types.llms.anthropic import is_advisor_tool as messages_predicate

    advisor_tool = {"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-8"}
    assert chat_predicate(advisor_tool) is True
    assert messages_predicate(advisor_tool) is True

    not_advisor = {"type": "web_search_20250305", "name": "web_search"}
    assert chat_predicate(not_advisor) is False
    assert messages_predicate(not_advisor) is False
