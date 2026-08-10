"""A guardrail must not alter a replayed reasoning block.

[ARC-BUG-55] Anthropic rejects a replayed `thinking` / `redacted_thinking` block that differs
from the original response by a single byte:

    messages.N.content.M: `thinking` or `redacted_thinking` blocks in the latest assistant
    message cannot be modified. These blocks must remain as they were in the original response.

That 400 is the highest-volume client-visible error on prd-ai — 304 alerts between
2026-08-05 11:56Z and 2026-08-10 15:41Z, still firing.

🔴 It was attributed to the CLIENT, on the reasoning that the guardrail write-back only touches
blocks carrying a `text` key and reasoning blocks have none. That reasoning was wrong, and this
file is the proof. `_write_back_structured_messages` replaces the ENTIRE `data["messages"]`
array via `anthropic_messages_pt`, and the Anthropic→OpenAI leg MATERIALISES a field that was
never in the request: `adapters/transformation.py` builds both reasoning blocks with
`cache_control=content.get("cache_control", {})`, so an absent key becomes `cache_control: {}`.

The cleanup in `_write_back_structured_messages` existed for exactly this, but matched only
`type == "thinking"` — so `redacted_thinking` reached the vendor with an injected key. A
`thinking` + `text` turn was byte-identical, which is why the asymmetry went unnoticed; add a
`redacted_thinking` block and the request is altered.

⚠️ These tests assert BYTE-IDENTITY of the reasoning blocks, not the absence of one field. A
test keyed on `cache_control` alone would pass any future converter change that injects a
different field, which is the same class of miss that let this ship.
"""

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.anthropic.chat.guardrail_translation.handler import (  # noqa: E402
    AnthropicMessagesHandler,
)

THINKING = {"type": "thinking", "thinking": "the model's private reasoning", "signature": "ErUBCkYIAxgCIkDx=="}
REDACTED = {"type": "redacted_thinking", "data": "EroBCoYBGAIiQL2v=="}
TEXT = {"type": "text", "text": "the visible answer"}


def _round_trip(assistant_content, mutate=None):
    """Push an assistant turn through the guardrail write-back, as the proxy does.

    Returns (original_reasoning_blocks, reasoning_blocks_after).
    """
    handler = AnthropicMessagesHandler()
    data = {
        "model": "claude-sonnet-5",
        "messages": [
            {"role": "user", "content": "question one"},
            {"role": "assistant", "content": copy.deepcopy(assistant_content)},
            {"role": "user", "content": "question two"},
        ],
    }
    original = [b for b in copy.deepcopy(assistant_content) if b.get("type") in ("thinking", "redacted_thinking")]

    structured = list(handler.get_structured_messages(data) or [])
    if mutate is not None:
        structured = mutate(structured)
    handler._write_back_structured_messages(data, structured)

    assistant = next(m for m in data["messages"] if m.get("role") == "assistant")
    content = assistant.get("content")
    after = [
        block
        for block in (content if isinstance(content, list) else [])
        if isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking")
    ]
    return original, after


@pytest.mark.parametrize(
    "assistant_content,why",
    [
        ([copy.deepcopy(REDACTED), copy.deepcopy(TEXT)], "the shape that shipped the 400"),
        ([copy.deepcopy(REDACTED)], "a redacted block with no text block beside it"),
        (
            [copy.deepcopy(THINKING), copy.deepcopy(REDACTED), copy.deepcopy(TEXT)],
            "both reasoning types in one turn",
        ),
        ([copy.deepcopy(THINKING), copy.deepcopy(TEXT)], "the shape that was already correct"),
        ([copy.deepcopy(THINKING)], "a thinking block with no text block beside it"),
    ],
    ids=["redacted_and_text", "redacted_only", "both_types", "thinking_and_text", "thinking_only"],
)
def test_a_replayed_reasoning_block_survives_the_write_back_byte_identical(assistant_content, why):
    """The property Anthropic enforces, asserted directly against the produced request."""
    original, after = _round_trip(assistant_content)

    assert after == original, f"the request was altered ({why}): {after!r} != {original!r}"


def test_a_guardrail_that_rewrites_the_text_still_leaves_reasoning_untouched():
    """🔴 The case that matters: the write-back only runs when the guardrail CHANGED something.

    A guardrail that masks the assistant's visible text is exactly what triggers this path in
    production. The masked text must land, and the reasoning blocks beside it must not move.
    """

    def mask_the_assistant_text(structured):
        for message in structured:
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                message["content"] = "[MASKED BY GUARDRAIL]"
        return structured

    handler = AnthropicMessagesHandler()
    data = {
        "model": "claude-sonnet-5",
        "messages": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": [copy.deepcopy(THINKING), copy.deepcopy(REDACTED), copy.deepcopy(TEXT)],
            },
        ],
    }
    structured = mask_the_assistant_text(list(handler.get_structured_messages(data) or []))

    handler._write_back_structured_messages(data, structured)

    assistant = next(m for m in data["messages"] if m.get("role") == "assistant")
    blocks = assistant["content"]
    reasoning = [b for b in blocks if b.get("type") in ("thinking", "redacted_thinking")]
    texts = [b for b in blocks if b.get("type") == "text"]

    assert reasoning == [THINKING, REDACTED], f"a reasoning block was altered: {reasoning!r}"
    assert texts and texts[0]["text"] == "[MASKED BY GUARDRAIL]", "the guardrail's edit was lost"


def test_a_caller_supplied_cache_control_is_also_stripped_from_reasoning():
    """Keyed on block TYPE, not on the value being empty.

    A real `cache_control` on a replayed reasoning block is equally a modification — Anthropic
    compares against the original response, which cannot have carried a breakpoint the client
    added afterwards. Stripping it costs a caching hint; leaving it costs the whole request.

    Without this, a fix that only popped empty dicts would pass every test above.
    """
    with_breakpoint = {**copy.deepcopy(THINKING), "cache_control": {"type": "ephemeral"}}

    _, after = _round_trip([with_breakpoint, copy.deepcopy(TEXT)])

    assert after, "the reasoning block vanished entirely"
    assert "cache_control" not in after[0], f"a cache breakpoint survived onto a reasoning block: {after[0]!r}"


def test_the_user_turns_are_not_disturbed():
    """Negative control: the fix must not become 'strip fields from everything'."""
    handler = AnthropicMessagesHandler()
    data = {
        "model": "claude-sonnet-5",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "keep me", "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": [copy.deepcopy(REDACTED), copy.deepcopy(TEXT)]},
        ],
    }

    handler._write_back_structured_messages(data, list(handler.get_structured_messages(data) or []))

    user = next(m for m in data["messages"] if m.get("role") == "user")
    user_blocks = user["content"] if isinstance(user["content"], list) else []
    assert any(b.get("cache_control") for b in user_blocks if isinstance(b, dict)), (
        f"a user-turn cache breakpoint was stripped: {user_blocks!r}"
    )
