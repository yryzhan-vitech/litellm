"""The advisor sub-call must not drop the executor's reasoning blocks.

[ARC-BUG-56] Anthropic validates the LATEST assistant message against what it actually
returned:

    messages.N.content.M: `thinking` or `redacted_thinking` blocks in the latest assistant
    message cannot be modified. These blocks must remain as they were in the original
    response.

That was prd-ai's highest-volume client-visible error — 304 alerts between 2026-08-05
11:56Z and 2026-08-10 17:15Z, prd-ai only, still firing when this was written. The whole
request 400s, so the user loses the executor's answer as well as the advice.

🔴 It was twice attributed to the client. The S3 request logs settled it:

  | | failing records | successful thinking-replay records |
  |---|---|---|
  | `advisor` present | 3 of 3 | 0 of 4 |
  | latest assistant turn | `['text']` | `['thinking', ...]` |
  | reasoning-turn vs assistant-turn indices | diverged | matched exactly |

Mechanism: `_build_advisor_context` kept only `type == "text"` blocks, so an
extended-thinking executor response `['thinking', 'text']` became `['text']` — a modified
latest assistant message.

⚠️ These tests assert the reasoning blocks are carried VERBATIM, not merely present.
`signature` is the field Anthropic verifies, so a block that survives with an added or
reordered field is still a 400.
"""

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (  # noqa: E402
    _build_advisor_context,
    _inject_advisor_turn,
    _inject_max_uses_error,
)

THINKING = {"type": "thinking", "thinking": "step one, step two", "signature": "ErUBCkYIAxgCIkDx=="}
REDACTED = {"type": "redacted_thinking", "data": "EroBCoYBGAIiQL2v=="}
TEXT = {"type": "text", "text": "the partial answer"}
TOOL_USE = {"type": "tool_use", "id": "toolu_01A", "name": "Bash", "input": {"command": "ls"}}

ADVISOR_USE = {"id": "toolu_advisor", "input": {"question": "review this"}}
CONVERSATION = [{"role": "user", "content": "the original question"}]


def _latest_assistant_blocks(messages):
    assistant = [m for m in messages if m.get("role") == "assistant"]
    assert assistant, "no assistant turn was produced"
    content = assistant[-1].get("content")
    assert isinstance(content, list), f"expected block list, got {type(content).__name__}"
    return content


@pytest.mark.parametrize(
    "executor_content,expected_reasoning",
    [
        ([copy.deepcopy(THINKING), copy.deepcopy(TEXT)], [THINKING]),
        ([copy.deepcopy(REDACTED), copy.deepcopy(TEXT)], [REDACTED]),
        ([copy.deepcopy(THINKING), copy.deepcopy(REDACTED), copy.deepcopy(TEXT)], [THINKING, REDACTED]),
        ([copy.deepcopy(THINKING)], [THINKING]),
        ([copy.deepcopy(THINKING), copy.deepcopy(TEXT), copy.deepcopy(TOOL_USE)], [THINKING]),
    ],
    ids=["thinking", "redacted", "both", "thinking_only", "with_tool_use"],
)
def test_the_executors_reasoning_reaches_the_advisor_sub_call(executor_content, expected_reasoning):
    """The exact regression: reasoning blocks were filtered out of the latest assistant turn."""
    executor_response = {"content": copy.deepcopy(executor_content)}

    result = _build_advisor_context(copy.deepcopy(CONVERSATION), executor_response, ADVISOR_USE)

    blocks = _latest_assistant_blocks(result)
    reasoning = [b for b in blocks if b.get("type") in ("thinking", "redacted_thinking")]
    assert reasoning == expected_reasoning, (
        f"reasoning blocks were altered or dropped: {reasoning!r} != {expected_reasoning!r}"
    )


def test_the_reasoning_signature_survives_verbatim():
    """`signature` is the field Anthropic verifies — a stripped or rewritten one is a 400."""
    executor_response = {"content": [copy.deepcopy(THINKING), copy.deepcopy(TEXT)]}

    blocks = _latest_assistant_blocks(
        _build_advisor_context(copy.deepcopy(CONVERSATION), executor_response, ADVISOR_USE)
    )

    thinking = next(b for b in blocks if b.get("type") == "thinking")
    assert thinking["signature"] == THINKING["signature"], "the signature was altered"
    assert thinking["thinking"] == THINKING["thinking"], "the reasoning prose was altered"
    assert set(thinking) == set(THINKING), f"the block's field set changed: {sorted(thinking)}"


def test_block_order_within_the_turn_is_preserved():
    """Anthropic compares the turn as returned; a reordered block set is still modified."""
    executor_response = {
        "content": [copy.deepcopy(THINKING), copy.deepcopy(TEXT), copy.deepcopy(REDACTED)]
    }

    blocks = _latest_assistant_blocks(
        _build_advisor_context(copy.deepcopy(CONVERSATION), executor_response, ADVISOR_USE)
    )

    assert [b.get("type") for b in blocks] == ["thinking", "text", "redacted_thinking"]


def test_tool_use_is_still_excluded():
    """🔴 Negative control, and a documented remaining exposure.

    `tool_use` must stay out: Anthropic requires it to be followed immediately by its
    `tool_result`, not by the advisor question. So this exclusion is deliberate — but it is
    also still a block-set change to the latest assistant turn, which can trip the same 400
    on a `['thinking', 'tool_use']` response. The two rules conflict on that shape.

    Pinned so that "fix ARC-BUG-56 by carrying everything" cannot be done accidentally
    without confronting that conflict.
    """
    executor_response = {"content": [copy.deepcopy(THINKING), copy.deepcopy(TOOL_USE)]}

    blocks = _latest_assistant_blocks(
        _build_advisor_context(copy.deepcopy(CONVERSATION), executor_response, ADVISOR_USE)
    )

    assert not any(b.get("type") == "tool_use" for b in blocks), "tool_use must not reach the advisor"
    assert any(b.get("type") == "thinking" for b in blocks), "thinking must still be carried"


def test_provider_specific_fields_are_still_stripped():
    """The one field set that MUST be removed — it is not part of Anthropic's comparison."""
    noisy = {**copy.deepcopy(THINKING), "provider_specific_fields": {"vendor": "junk"}}
    executor_response = {"content": [noisy, copy.deepcopy(TEXT)]}

    blocks = _latest_assistant_blocks(
        _build_advisor_context(copy.deepcopy(CONVERSATION), executor_response, ADVISOR_USE)
    )

    thinking = next(b for b in blocks if b.get("type") == "thinking")
    assert "provider_specific_fields" not in thinking


@pytest.mark.parametrize("inject", [_inject_advisor_turn, _inject_max_uses_error], ids=["advice", "max_uses"])
def test_the_injection_paths_also_keep_the_reasoning(inject):
    """The executor continues from these, so a drop here 400s the NEXT turn instead.

    Verified as already-correct rather than changed: both pass `executor_content` through
    verbatim. Pinned so a future 'consistency' refactor cannot apply the old text-only
    filter here too.
    """
    executor_response = {
        "content": [copy.deepcopy(THINKING), copy.deepcopy(REDACTED), copy.deepcopy(TEXT)]
    }

    kwargs = {"advisor_text": "the advice"} if inject is _inject_advisor_turn else {}
    result = inject(copy.deepcopy(CONVERSATION), executor_response, ADVISOR_USE, **kwargs)

    blocks = _latest_assistant_blocks(result)
    reasoning = [b for b in blocks if b.get("type") in ("thinking", "redacted_thinking")]
    assert reasoning == [THINKING, REDACTED], f"reasoning altered on the injection path: {reasoning!r}"
