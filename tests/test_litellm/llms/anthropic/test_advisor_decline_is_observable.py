"""A declined advisor request must leave a trace.

[ARC-BUG-53] `can_handle()` returns False when a request carries an advisor tool but the
resolved provider is in `ADVISOR_NATIVE_PROVIDERS`. The request then completes as an
ordinary 200 with no advisor output and — until now — no record of the decision anywhere.
The tool was asked for, the orchestration silently did not run, and the model answered on
its own.

That is what users reported as "/advisor does not fire and the model runs its own
adversarial review"; one reporter presented advisor findings and then apologised for
having invented them. The path was invisible BY CONSTRUCTION: it emitted nothing, so no
grep and no metric could tell "the advisor ran" apart from "the advisor was asked for and
skipped". The hypothesis was provable as code and unobservable in traffic.

⚠️ Only the DECLINE is logged. An accept is already observable — it produces advisor
sub-call frames — and logging every one would add a line per request on a proxy already
serving ~1 000 req/pod/h.
"""

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (  # noqa: E402
    AdvisorOrchestrationHandler,
)

ADVISOR_TOOL = {"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-8"}
REGULAR_TOOL = {"name": "get_weather", "description": "w", "input_schema": {"type": "object", "properties": {}}}

MARKER = "[ARC-BUG-53]"


@pytest.fixture
def declines(caplog):
    """Capture the decline marker at WARNING, the level prd-ai can actually see."""
    caplog.set_level(logging.WARNING, logger="LiteLLM")

    def _lines():
        return [r.getMessage() for r in caplog.records if MARKER in r.getMessage()]

    return _lines


@pytest.fixture
def handler():
    return AdvisorOrchestrationHandler()


def test_the_silent_decline_is_now_logged(handler, declines):
    """The regression: advisor asked for, provider native, orchestration skipped.

    This is the only path that used to leave no trace at all.
    """
    assert handler.can_handle(tools=[ADVISOR_TOOL], custom_llm_provider="anthropic") is False

    logged = declines()
    assert len(logged) == 1, f"the decline was not logged: {logged}"
    assert "anthropic" in logged[0], "the log must name the provider that caused the decline"


def test_an_accepted_request_logs_nothing(handler, declines):
    """A non-native provider orchestrates normally — and stays quiet.

    Asserted so that a future edit cannot turn this into a per-request log line on the
    hot path.
    """
    assert handler.can_handle(tools=[ADVISOR_TOOL], custom_llm_provider="bedrock") is True
    assert declines() == []


@pytest.mark.parametrize(
    ("tools", "provider", "why"),
    [
        ([REGULAR_TOOL], "anthropic", "no advisor tool present"),
        ([REGULAR_TOOL], "bedrock", "no advisor tool present"),
        (None, "anthropic", "no tools at all"),
        ([], "bedrock", "empty tool list"),
    ],
)
def test_a_request_without_an_advisor_tool_logs_nothing(handler, declines, tools, provider, why):
    """Only an advisor request can be "declined" — everything else was never a candidate.

    Without this, the marker would fire on every ordinary request to a native provider and
    the signal would be worthless.
    """
    assert handler.can_handle(tools=tools, custom_llm_provider=provider) is False
    assert declines() == [], f"logged a decline when {why}"


def test_the_decline_is_logged_at_warning_not_info(handler, caplog):
    """prd-ai runs `LITELLM_LOG=ERROR`, under which an INFO line does not exist to grep.

    This is the trap that made a `grep -i fallback` return 142 hits which were all
    stack-frame names rather than events: the real lines were `.info()` and had been
    suppressed at source. A decline logged at INFO would be equally ungreppable, so the
    level is part of the fix, not a detail.
    """
    caplog.set_level(logging.DEBUG, logger="LiteLLM")

    handler.can_handle(tools=[ADVISOR_TOOL], custom_llm_provider="anthropic")

    records = [r for r in caplog.records if MARKER in r.getMessage()]
    assert records, "no decline record emitted"
    assert all(r.levelno >= logging.WARNING for r in records), (
        f"the decline must be at WARNING or above to survive LITELLM_LOG=ERROR: {[r.levelname for r in records]}"
    )


def test_a_dated_advisor_tool_variant_still_counts_as_advisor(handler, declines):
    """The predicate is prefix-based, so a future dated type must still be seen.

    ARC-BUG-45 was exactly this: an exact-string comparison stopped matching when the
    dated suffix moved. If that regressed, this decline would go quiet again and the
    instrument would report a false zero.
    """
    future_variant = {"type": "advisor_20991231", "name": "advisor", "model": "claude-opus-4-8"}

    assert handler.can_handle(tools=[future_variant], custom_llm_provider="anthropic") is False
    assert len(declines()) == 1, "a dated advisor variant was not recognised as an advisor tool"


def test_a_malformed_tool_list_does_not_raise(handler):
    """`tools` is caller input — a non-dict entry must not crash the gate."""
    assert handler.can_handle(tools=["not-a-dict", None, 42], custom_llm_provider="anthropic") is False


def test_the_orchestration_sub_calls_cannot_re_trigger_the_decline(handler, declines):
    """The count must be per REQUEST, not per orchestration round.

    Raised by an adversarial review: the advisor loop's sub-calls go back through
    `anthropic_messages()`, which runs this gate again — so an inflated count was a real
    possibility, and a metric that multiplies by round count is worse than no metric.

    It cannot happen, and the reason is structural rather than lucky:
      * the EXECUTOR leg is handed the SYNTHETIC tool, which has a `name` of "advisor" but
        NO `type` — and `is_advisor_tool` matches on the dated `type` prefix, so it reads
        False;
      * the ADVISOR leg is called with `tools=None`, which exits at the first check.

    Asserted rather than trusted, because "the synthetic tool has no type" is exactly the
    kind of invariant a later refactor tidies away.
    """
    synthetic = {"name": "advisor", "description": "the synthetic executor-facing tool", "input_schema": {}}

    assert handler.can_handle(tools=[synthetic], custom_llm_provider="anthropic") is False
    assert handler.can_handle(tools=None, custom_llm_provider="anthropic") is False
    assert declines() == [], "an orchestration sub-call logged a decline and inflated the count"
