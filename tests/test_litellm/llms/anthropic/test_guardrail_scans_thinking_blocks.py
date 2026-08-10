"""Extended-thinking content must be SCANNED but never MODIFIED.

[ARC-BUG-49] `_extract_from_content_blocks` dispatched only `text` and `tool_use`, so a
`thinking` block's prose never reached the guardrail — it went to the model uninspected.
The model's reasoning is steered by the prompt, so that is a real coverage hole, not a
theoretical one.

The fix has a hard constraint that makes it easy to get wrong: Anthropic rejects a
request whose `thinking` or `redacted_thinking` block differs from the original response
by a single byte —

    messages.N.content.M: `thinking` or `redacted_thinking` blocks in the latest
    assistant message cannot be modified. These blocks must remain as they were in the
    original response.

— and that 400 is the highest-volume client-visible error on prd-ai (286 events
2026-08-05 → 08-10). So scanning must be strictly read-only: the guardrail sees the text,
and the payload on the wire stays byte-identical.
"""

import asyncio
import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.anthropic.chat.guardrail_translation.handler import (  # noqa: E402
    _UNWRITABLE_CONTENT_IDX,
    AnthropicMessagesHandler,
)

THINKING = {"type": "thinking", "thinking": "the model's private reasoning", "signature": "sig-abc123"}
REDACTED = {"type": "redacted_thinking", "data": "opaque-server-encrypted-blob"}
TEXT = {"type": "text", "text": "the visible answer"}


def _extract(blocks):
    """Run extraction and return (texts, mappings)."""
    texts, images, mappings, tool_calls = [], [], [], []
    AnthropicMessagesHandler()._extract_from_content_blocks(
        response_content=blocks,
        texts_to_check=texts,
        images_to_check=images,
        task_mappings=mappings,
        tool_calls_to_check=tool_calls,
    )
    return texts, mappings


def test_thinking_prose_reaches_the_guardrail():
    """The regression: this text used to be invisible to every guardrail."""
    texts, _ = _extract([TEXT, THINKING])

    assert THINKING["thinking"] in texts, f"thinking content was not scanned: {texts}"
    assert TEXT["text"] in texts, "the ordinary text block must still be scanned"


def test_thinking_is_recorded_as_unwritable():
    """Scanned, but pinned to an index the write-back refuses.

    Two independent guards keep the block immutable — the `type == "text"` check in both
    write-backs, and this sentinel. Belt and braces is deliberate: a future edit that
    loosens the type check must not silently make thinking blocks writable.
    """
    _, mappings = _extract([TEXT, THINKING])

    assert (_UNWRITABLE_CONTENT_IDX, None) in mappings, f"thinking got a writable mapping: {mappings}"
    assert (0, None) in mappings, "the text block must keep its real, writable index"


def test_redacted_thinking_is_not_scanned():
    """Its `data` is an opaque encrypted blob, not prose — scanning it is pure noise.

    Asserted rather than left implicit, so that "why not both?" has an answer in the
    suite instead of being re-litigated.
    """
    texts, mappings = _extract([REDACTED])

    assert texts == [], f"redacted_thinking should not be scanned: {texts}"
    assert mappings == []


def test_a_redacting_guardrail_cannot_alter_the_thinking_block():
    """The whole point: the payload on the wire must stay byte-identical.

    This is the test that fails if anyone makes thinking writable — and a failure here
    means shipping the 400 that is already prd-ai's loudest client-visible error.
    """
    response = {
        "model": "claude-sonnet-5",
        "content": [copy.deepcopy(TEXT), copy.deepcopy(THINKING), copy.deepcopy(REDACTED)],
    }
    original = copy.deepcopy(response["content"])

    handler = AnthropicMessagesHandler()
    texts, mappings = _extract(response["content"])

    # A guardrail that redacts everything it was shown — the dangerous case.
    asyncio.run(
        handler._apply_guardrail_responses_to_output(
            response=response,
            responses=[f"[BLOCKED-{i}]" for i, _ in enumerate(texts)],
            task_mappings=mappings,
        )
    )

    assert response["content"][0]["text"] == "[BLOCKED-0]", "the text block SHOULD be redactable"
    assert response["content"][1] == original[1], "the thinking block was modified"
    assert response["content"][2] == original[2], "the redacted_thinking block was modified"


def test_a_thinking_only_response_writes_nothing():
    """The sentinel is a negative index — it must not address a block from the END.

    With no writable block at all, a naive `content[-1] = ...` would overwrite the
    thinking block itself. This is the case that catches that.
    """
    response = {"model": "claude-sonnet-5", "content": [copy.deepcopy(THINKING)]}
    original = copy.deepcopy(response["content"])

    handler = AnthropicMessagesHandler()
    texts, mappings = _extract(response["content"])
    assert texts == [THINKING["thinking"]], "precondition: the only scanned text is the reasoning"

    asyncio.run(
        handler._apply_guardrail_responses_to_output(response=response, responses=["[BLOCKED]"], task_mappings=mappings)
    )

    assert response["content"] == original, "a negative index wrote into the block list"


@pytest.mark.parametrize(
    "block",
    [
        {"type": "thinking"},
        {"type": "thinking", "thinking": None},
        {"type": "thinking", "thinking": ""},
        {"type": "thinking", "thinking": 42},
        {"type": "thinking", "thinking": {"nested": "dict"}},
    ],
    ids=["missing", "none", "empty", "int", "dict"],
)
def test_malformed_thinking_blocks_are_skipped_not_crashed(block):
    """Response content is provider output — a malformed block must not raise.

    Anything non-string is skipped rather than coerced: `str(42)` would send "42" to the
    guardrail as if it were prose.
    """
    texts, mappings = _extract([block])

    assert texts == [], f"a malformed thinking block was scanned: {texts}"
    assert mappings == []


def test_a_mixed_response_keeps_text_indices_correct():
    """Interleaving must not shift the writable indices.

    The mapping is positional, so a thinking block sitting between two text blocks is
    exactly where an off-by-one would corrupt the wrong one.
    """
    blocks = [
        {"type": "text", "text": "first"},
        copy.deepcopy(THINKING),
        {"type": "text", "text": "second"},
    ]
    response = {"model": "claude-sonnet-5", "content": blocks}
    original = copy.deepcopy(blocks)

    handler = AnthropicMessagesHandler()
    texts, mappings = _extract(blocks)
    asyncio.run(
        handler._apply_guardrail_responses_to_output(
            response=response,
            responses=[f"[R-{i}]" for i, _ in enumerate(texts)],
            task_mappings=mappings,
        )
    )

    assert blocks[0]["text"] == "[R-0]", "the first text block took the wrong response"
    assert blocks[2]["text"] == "[R-2]", "the second text block took the wrong response"
    assert blocks[1] == original[1], "the thinking block between them was modified"


def test_the_thinking_response_does_not_land_on_a_trailing_text_block():
    """The sentinel is -1, so without the write-back guard it addresses the LAST block.

    The shape that exposes it: `thinking` FIRST, a text block LAST. The thinking scan's
    response then overwrites that trailing text — the guardrail's verdict about the
    reasoning gets published as the visible answer, and the real text block's own verdict
    is lost. A mutation that removes the guard survived every other test in this file,
    which is why this case is spelled out.
    """
    blocks = [copy.deepcopy(THINKING), {"type": "text", "text": "the trailing answer"}]
    response = {"model": "claude-sonnet-5", "content": blocks}

    handler = AnthropicMessagesHandler()
    texts, mappings = _extract(blocks)
    assert mappings == [(_UNWRITABLE_CONTENT_IDX, None), (1, None)], f"precondition: {mappings}"

    asyncio.run(
        handler._apply_guardrail_responses_to_output(
            response=response,
            responses=["[VERDICT-ON-REASONING]", "[VERDICT-ON-TEXT]"],
            task_mappings=mappings,
        )
    )

    assert blocks[1]["text"] == "[VERDICT-ON-TEXT]", (
        "the trailing text block took the THINKING scan's response — the -1 sentinel addressed it from the end"
    )
    assert blocks[0] == THINKING, "the thinking block was modified"


def test_redacted_thinking_data_is_never_sent_to_the_guardrail():
    """Asserted on the payload, not just on the mapping.

    A mutation that added `redacted_thinking` to the same branch as `thinking` survived
    the earlier test, because that one only checked a redacted-only response. Mixed with
    a real thinking block, the blob rides along into `texts` — and `data` is an opaque
    encrypted string, so the guardrail scores noise and may block on it.
    """
    texts, _ = _extract([copy.deepcopy(THINKING), copy.deepcopy(REDACTED)])

    assert REDACTED["data"] not in texts, f"the encrypted blob was scanned: {texts}"
    assert texts == [THINKING["thinking"]], f"only the thinking prose should be scanned: {texts}"
