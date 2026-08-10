"""Characterisation tests for the advisor sub-call context.

These pin down what `_build_advisor_context` actually emits, because two plausible
readings of it are wrong and both were believed during the 2026-08 advisor
investigation:

1. *"It leaves a dangling tool_use."* It does not. `_inject_advisor_turn` always
   appends the `tool_result` turn straight after the assistant turn that holds the
   `tool_use`, so adjacency survives into the next round. The docstring's claim
   about excluding tool_use refers only to the executor's newest response, which it
   does filter.
2. *"Stripping the tool_use envelope would be harmless."* It would not. The prior
   advice is carried *inside* that `tool_result`, so removing the envelope removes
   the advice — the advisor would lose its own history.

Keeping these as tests means the next person to reach for either change gets a
failure instead of a plausible story.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (  # noqa: E402
    _build_advisor_context,
    _inject_advisor_turn,
)

ADVISOR_USE = {
    "type": "tool_use",
    "id": "toolu_advisor_1",
    "name": "advisor",
    "input": {"question": "Is this approach sound?"},
}


def _executor_response(*blocks):
    return {"role": "assistant", "content": list(blocks)}


def _tool_use_pairs(messages):
    """Every (tool_use id, is-it-immediately-followed-by-its-tool_result) in order."""
    out = []
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            nxt_content = (messages[i + 1] if i + 1 < len(messages) else {}).get("content")
            satisfied = isinstance(nxt_content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == block.get("id")
                for b in nxt_content
            )
            out.append((block.get("id"), satisfied))
    return out


def test_round_one_context_carries_no_tool_use():
    """The executor's newest response IS filtered — only its text blocks survive."""
    ctx = _build_advisor_context(
        [{"role": "user", "content": "Review my plan."}],
        _executor_response({"type": "text", "text": "Here is my plan."}, ADVISOR_USE),
        ADVISOR_USE,
    )
    assert _tool_use_pairs(ctx) == []
    assert ctx[-1] == {"role": "user", "content": "Is this approach sound?"}


def test_tool_use_adjacency_survives_into_the_next_round():
    """Regression guard: Anthropic rejects a tool_use not followed by its tool_result.

    Drive the real producer (`_inject_advisor_turn`) rather than hand-writing history,
    so this keeps testing the loop's actual contract.
    """
    after_round1 = _inject_advisor_turn(
        [{"role": "user", "content": "Review my plan."}],
        _executor_response({"type": "text", "text": "Plan v1."}, ADVISOR_USE),
        ADVISOR_USE,
        "ADVICE-1: consider the edge cases.",
    )
    round2_use = {**ADVISOR_USE, "id": "toolu_advisor_2"}
    ctx = _build_advisor_context(
        after_round1, _executor_response({"type": "text", "text": "Plan v2."}, round2_use), round2_use
    )

    unpaired = [tid for tid, satisfied in _tool_use_pairs(ctx) if not satisfied]
    assert unpaired == [], f"tool_use with no adjacent tool_result: {unpaired}"
    assert ctx[-1] == {"role": "user", "content": "Is this approach sound?"}


def test_prior_advice_reaches_the_advisor_sub_call():
    """The advice lives inside the tool_result — dropping that envelope loses it.

    This is the test that makes "just strip the tool_use/tool_result pair" fail
    loudly instead of silently lobotomising multi-round advising.
    """
    after_round1 = _inject_advisor_turn(
        [{"role": "user", "content": "Review my plan."}],
        _executor_response({"type": "text", "text": "Plan v1."}, ADVISOR_USE),
        ADVISOR_USE,
        "ADVICE-1: consider the edge cases.",
    )
    round2_use = {**ADVISOR_USE, "id": "toolu_advisor_2"}
    ctx = _build_advisor_context(
        after_round1, _executor_response({"type": "text", "text": "Plan v2."}, round2_use), round2_use
    )

    carried = [
        b.get("content")
        for m in ctx
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert "ADVICE-1: consider the edge cases." in carried, "round-1 advice did not reach the advisor"


def test_only_user_and_assistant_roles_reach_the_advisor():
    """Anthropic takes `system` as a top-level param, never as a message.

    A mid-array `system` role earns *role 'system' must precede an 'assistant'
    message or end the array*. The public request type only permits user/assistant,
    so a client cannot introduce one — this asserts the orchestration loop does not
    introduce one either.
    """
    after_round1 = _inject_advisor_turn(
        [{"role": "user", "content": "Review my plan."}],
        _executor_response({"type": "text", "text": "Plan."}, ADVISOR_USE),
        ADVISOR_USE,
        "Advice.",
    )
    ctx = _build_advisor_context(after_round1, _executor_response({"type": "text", "text": "Done."}), ADVISOR_USE)
    assert {m["role"] for m in ctx} <= {"user", "assistant"}


def test_provider_specific_fields_are_stripped_from_executor_text():
    """`provider_specific_fields` is not accepted on the wire."""
    ctx = _build_advisor_context(
        [{"role": "user", "content": "Hi."}],
        _executor_response({"type": "text", "text": "Reply.", "provider_specific_fields": {"x": 1}}),
        ADVISOR_USE,
    )
    assert all(
        "provider_specific_fields" not in b
        for m in ctx
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict)
    )


def test_caller_messages_are_not_mutated():
    """The loop reuses `current_messages` after this call, so it must not be touched."""
    inner = [{"type": "text", "text": "kept"}]
    messages = [{"role": "assistant", "content": inner}]
    before = repr(messages)
    _build_advisor_context(messages, _executor_response(), ADVISOR_USE)
    assert repr(messages) == before
    assert messages[0]["content"] is inner


def test_missing_question_falls_back_rather_than_sending_an_empty_turn():
    ctx = _build_advisor_context([{"role": "user", "content": "Hi."}], _executor_response(), {"type": "tool_use"})
    assert ctx[-1] == {"role": "user", "content": "Please provide guidance on the current task."}


@pytest.mark.parametrize(
    "content",
    [None, "plain string", [], ["not-a-dict"], [{"no_type": 1}]],
    ids=["none", "string", "empty", "non-dict-block", "typeless-block"],
)
def test_malformed_history_does_not_raise(content):
    """`messages` is caller input, so a malformed turn must not crash orchestration."""
    ctx = _build_advisor_context([{"role": "assistant", "content": content}], _executor_response(), ADVISOR_USE)
    assert ctx[-1] == {"role": "user", "content": "Is this approach sound?"}
