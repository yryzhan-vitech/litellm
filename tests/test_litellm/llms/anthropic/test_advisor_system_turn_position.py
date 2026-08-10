"""An advisor sub-call must not 400 on a system turn the caller was allowed to send.

[ARC-BUG-54] Anthropic's `/v1/messages` accepts a `role: "system"` turn inside `messages`
only when it "must precede an 'assistant' message or end the array".
`_build_advisor_context()` APPENDS the executor's answer and the advisor question, so a
system turn that ENDED the array — legal on the way in — stops being last, and the
sub-call is rejected:

    messages.110: role 'system' must precede an 'assistant' message or end the array

Observed live on dev-ai 2026-08-10 15:11:54Z, raised from `advisor.py:229` →
`_call_messages_handler:639`. The advisor orchestration ran; its sub-call failed; the whole
request 400s, so the user loses the executor's answer as well. A mid-conversation directive
therefore made the advisor unusable rather than merely unadvised.

⚠️ These tests assert the RULE, not a fixed output list. Asserting "the system turn was
removed" would pass a fix that removes every system turn including the legal ones, which
would silently strip directives the advisor should see.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (  # noqa: E402
    _build_advisor_context,
)

EXECUTOR_RESPONSE = {"content": [{"type": "text", "text": "the partial answer"}]}
ADVISOR_USE = {"id": "toolu_1", "input": {"question": "review this"}}


def _illegal_system_positions(messages):
    """Anthropic's rule, applied to a role sequence. Returns the offending indices."""
    roles = [m.get("role") for m in messages]
    offending = []
    for index, role in enumerate(roles):
        if role != "system":
            continue
        following = roles[index + 1] if index + 1 < len(roles) else None
        if following is not None and following != "assistant":
            offending.append(index)
    return offending


def _build(messages):
    return _build_advisor_context(messages, EXECUTOR_RESPONSE, ADVISOR_USE)


@pytest.mark.parametrize(
    "messages,why",
    [
        (
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "system", "content": "a mid-conversation directive"},
                {"role": "user", "content": "c"},
            ],
            "the exact live failure: an interior system turn followed by a user turn",
        ),
        (
            [{"role": "user", "content": "a"}, {"role": "system", "content": "trailing directive"}],
            "legal on the way in because it ended the array — the append is what breaks it",
        ),
        (
            [
                {"role": "user", "content": "a"},
                {"role": "system", "content": "one"},
                {"role": "user", "content": "b"},
                {"role": "system", "content": "two"},
            ],
            "several offending turns, one of them trailing",
        ),
    ],
    ids=["interior_before_user", "trailing_becomes_interior", "multiple"],
)
def test_the_advisor_payload_never_violates_the_system_turn_rule(messages, why):
    """The property, asserted directly against Anthropic's stated constraint."""
    assert _illegal_system_positions(_build(messages)) == [], f"would 400 — {why}"


def test_a_system_turn_that_precedes_an_assistant_turn_is_KEPT():
    """🔴 The negative control: over-removal is a silent failure of its own.

    Appending never changes what follows an interior element, so a system turn already
    followed by an assistant turn stays legal and must survive. Dropping it would strip a
    directive the advisor is supposed to act on — and no 400 would ever reveal it.
    """
    messages = [
        {"role": "user", "content": "a"},
        {"role": "system", "content": "keep me"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]

    result = _build(messages)

    assert {"role": "system", "content": "keep me"} in result, "a legal system turn was dropped"
    assert _illegal_system_positions(result) == []


def test_a_conversation_with_no_system_turn_is_unchanged():
    """Guards against a fix that rewrites or reorders ordinary turns."""
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]

    result = _build(messages)

    assert result[: len(messages)] == messages, "the conversation prefix was altered"


def test_the_callers_message_list_is_not_mutated():
    """The executor call reuses this list, and it MUST still carry the directive.

    The fix is scoped to the advisor sub-call precisely so the directive keeps acting on
    the answer the user actually receives. If this list were mutated in place, the executor
    would silently lose it — turning an advisor-only workaround into a behaviour change on
    the main response.
    """
    messages = [
        {"role": "user", "content": "a"},
        {"role": "system", "content": "directive"},
        {"role": "user", "content": "c"},
    ]
    before = [dict(m) for m in messages]

    _build(messages)

    assert messages == before, "the caller's messages were mutated"


def test_the_advisor_question_is_still_the_last_turn():
    """The whole point of the sub-call. A fix that drops turns must not drop this one."""
    messages = [
        {"role": "user", "content": "a"},
        {"role": "system", "content": "directive"},
    ]

    result = _build(messages)

    assert result[-1] == {"role": "user", "content": "review this"}


def test_the_executor_text_still_reaches_the_advisor():
    """The advisor is reviewing the executor's partial answer — it must be present."""
    messages = [{"role": "user", "content": "a"}, {"role": "system", "content": "directive"}]

    result = _build(messages)

    assistant_turns = [m for m in result if m.get("role") == "assistant"]
    assert assistant_turns, "the executor's answer never reached the advisor"
    assert assistant_turns[-1]["content"] == [{"type": "text", "text": "the partial answer"}]


def test_a_malformed_entry_is_passed_through_not_crashed():
    """`messages` is client-supplied. A non-dict must not take the request down."""
    messages = ["not a dict", {"role": "user", "content": "a"}]

    result = _build(messages)

    assert "not a dict" in result
