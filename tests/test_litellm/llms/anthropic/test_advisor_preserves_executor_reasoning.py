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


def test_a_carried_tool_use_is_paired_with_a_result():
    """[ARC-BUG-57] Both Anthropic rules hold at once — they were never in real conflict.

    Anthropic requires (a) the latest assistant turn to match what it returned, and (b) a
    `tool_use` to be followed immediately by its `tool_result`. Dropping the `tool_use` broke
    (a) to satisfy (b), which kept the ARC-BUG-56 400 alive for `['thinking', 'tool_use']` —
    the most common extended-thinking shape when the executor calls a tool.

    The fix carries the tool call verbatim and emits a synthetic `tool_result` in the user
    turn immediately after, before the advisor question. This asserts BOTH properties, so a
    future change cannot restore one by sacrificing the other.
    """
    executor_response = {"content": [copy.deepcopy(THINKING), copy.deepcopy(TOOL_USE)]}

    result = _build_advisor_context(copy.deepcopy(CONVERSATION), executor_response, ADVISOR_USE)

    assistant_idx = max(i for i, m in enumerate(result) if m.get("role") == "assistant")
    blocks = result[assistant_idx]["content"]

    # (a) the turn is byte-identical to what the executor returned
    assert blocks == [THINKING, TOOL_USE], f"the returned turn was altered: {blocks!r}"

    # (b) every tool_use is resolved in the very next turn
    following = result[assistant_idx + 1]
    assert following["role"] == "user", "a tool_use must be followed by a user turn"
    results = [b for b in following["content"] if b.get("type") == "tool_result"]
    assert [r["tool_use_id"] for r in results] == [TOOL_USE["id"]], f"unpaired tool_use: {results!r}"

    # the advisor question is still last, in its own turn
    assert result[-1] == {"role": "user", "content": "review this"}


def test_parallel_tool_calls_each_get_their_own_result():
    """A turn can carry several tool calls; every one needs its own result or the API 400s.

    Without this, a fix that emitted a single result would pass the single-tool test above
    and fail on the parallel-call shape Claude Code produces routinely.
    """
    second_tool = {"type": "tool_use", "id": "toolu_02B", "name": "Read", "input": {"path": "/x"}}
    executor_response = {
        "content": [copy.deepcopy(THINKING), copy.deepcopy(TOOL_USE), copy.deepcopy(second_tool)]
    }

    result = _build_advisor_context(copy.deepcopy(CONVERSATION), executor_response, ADVISOR_USE)

    assistant_idx = max(i for i, m in enumerate(result) if m.get("role") == "assistant")
    results = [b for b in result[assistant_idx + 1]["content"] if b.get("type") == "tool_result"]

    assert [r["tool_use_id"] for r in results] == [TOOL_USE["id"], second_tool["id"]]


def test_the_synthetic_result_is_marked_as_an_error():
    """It must not read as 'the tool ran and returned nothing'.

    A neutral empty result would let the advisor reason about a fabricated observation —
    the exact failure class that produced invented advisor findings in the first place.
    """
    executor_response = {"content": [copy.deepcopy(THINKING), copy.deepcopy(TOOL_USE)]}

    result = _build_advisor_context(copy.deepcopy(CONVERSATION), executor_response, ADVISOR_USE)

    assistant_idx = max(i for i, m in enumerate(result) if m.get("role") == "assistant")
    synthetic = next(b for b in result[assistant_idx + 1]["content"] if b.get("type") == "tool_result")

    assert synthetic["is_error"] is True, "a placeholder result must be flagged as not-executed"
    assert "not executed" in synthetic["content"], f"unclear placeholder: {synthetic['content']!r}"


def test_the_advisors_own_invocation_is_still_excluded():
    """🔴 The one exclusion that remains, and it is deliberate.

    The advisor's `server_tool_use` is the block being HANDLED: its result does not exist
    yet, so echoing it back would ask the advisor to review its own pending invocation — and
    it could not be paired with a result either.

    A genuine `server_tool_use` (e.g. web_search) IS carried, so this must key on the
    advisor name, not on the block type.
    """
    advisor_invocation = {
        "type": "server_tool_use",
        "id": "srvtoolu_advisor",
        "name": "advisor",
        "input": {"question": "review this"},
    }
    web_search = {"type": "server_tool_use", "id": "srvtoolu_web", "name": "web_search", "input": {"q": "x"}}
    executor_response = {
        "content": [copy.deepcopy(THINKING), copy.deepcopy(advisor_invocation), copy.deepcopy(web_search)]
    }

    result = _build_advisor_context(copy.deepcopy(CONVERSATION), executor_response, ADVISOR_USE)

    assistant_idx = max(i for i, m in enumerate(result) if m.get("role") == "assistant")
    blocks = result[assistant_idx]["content"]

    assert blocks == [THINKING, web_search], f"wrong blocks carried: {blocks!r}"
    ids = [b["tool_use_id"] for b in result[assistant_idx + 1]["content"]]
    assert ids == [web_search["id"]], f"the advisor's own call was paired with a result: {ids!r}"


def test_no_empty_user_turn_when_there_are_no_tool_calls():
    """A text-only response must not gain a stray empty turn."""
    executor_response = {"content": [copy.deepcopy(THINKING), copy.deepcopy(TEXT)]}

    result = _build_advisor_context(copy.deepcopy(CONVERSATION), executor_response, ADVISOR_USE)

    assert result[-1] == {"role": "user", "content": "review this"}
    assert all(m.get("content") not in ([], None) for m in result), f"an empty turn was added: {result!r}"


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
