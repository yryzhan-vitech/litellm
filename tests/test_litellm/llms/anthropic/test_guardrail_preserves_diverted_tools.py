"""A guardrail must not drop tools it never inspected.

`process_input_messages` translates the request to OpenAI so `apply_guardrail` can
run, then rebuilds `data["tools"]` from what the guardrail returned. But
`_translate_to_openai` *diverts* Anthropic hosted tools that have no OpenAI
tool-shape — `web_search` becomes the `web_search_options` parameter — so they never
appear in `inputs["tools"]`. Rebuilding `data["tools"]` from that list therefore
deletes them from the outbound request.

Client-visible as the tool simply not working, and (when a later turn replays the
tool's own output) as Anthropic rejecting an unknown tool tag.

ARC-BUG-45 fixed the *other* half of this seam — native tools being re-mapped a
second time — and left this half open with a note. These tests close it.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.anthropic.chat.guardrail_translation.handler import (  # noqa: E402
    AnthropicMessagesHandler,
)

WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search"}
REGULAR = {
    "name": "get_weather",
    "description": "Look up the weather",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def _request(*tools):
    return {
        "model": "claude-sonnet-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "what is the weather?"}]}],
        "tools": list(tools),
    }


def _noop_guardrail():
    """A text-moderation guardrail: echoes `inputs` back untouched.

    This is the shape of every guardrail we actually run (Noma included) — it reads
    text and returns the same structure, so any tool loss is the seam's fault, not
    the guardrail's.
    """
    guardrail = MagicMock()
    guardrail.apply_guardrail = AsyncMock(side_effect=lambda inputs, **_: inputs)
    return guardrail


def _tool_ids(tools):
    """Identify tools by name, falling back to type for hosted tools that carry none.

    ``name`` first, not ``type`` first: a reviewed function tool legitimately comes
    back as ``{"type": "custom", "name": "get_weather", ...}`` — ``custom`` is the
    Anthropic shape for a caller-defined tool, so keying on ``type`` would collapse
    every such tool to one indistinguishable id.
    """
    return {t.get("name") or t.get("type") for t in (tools or [])}


def test_translate_to_openai_diverts_web_search_out_of_tools():
    """Precondition for everything below — documents WHY the loss happens."""
    translated = AnthropicMessagesHandler()._translate_to_openai(_request(WEB_SEARCH, REGULAR))

    names = {(t.get("function") or {}).get("name") or t.get("type") for t in (translated.get("tools") or [])}
    assert names == {"get_weather"}, "web_search should be diverted, not translated"
    assert translated.get("web_search_options") == {}, "the divert target"


@pytest.mark.asyncio
async def test_a_noop_guardrail_does_not_drop_the_diverted_web_search_tool():
    """The regression: web_search must still be on the request after the guardrail runs."""
    data = _request(WEB_SEARCH, REGULAR)

    returned = await AnthropicMessagesHandler().process_input_messages(
        data=data,
        guardrail_to_apply=_noop_guardrail(),
        litellm_logging_obj=MagicMock(),
    )

    assert _tool_ids(returned["tools"]) == {"web_search", "get_weather"}, (
        "the guardrail never saw web_search, so it must not be able to remove it"
    )


@pytest.mark.asyncio
async def test_a_web_search_only_request_keeps_its_tool():
    """The worst case: every tool is diverted, so the rebuilt list would be empty.

    `_translate_to_openai` returns `{}` when no regular tools remain, which is a
    different code path from the mixed case above.
    """
    data = _request(WEB_SEARCH)

    returned = await AnthropicMessagesHandler().process_input_messages(
        data=data,
        guardrail_to_apply=_noop_guardrail(),
        litellm_logging_obj=MagicMock(),
    )

    assert _tool_ids(returned["tools"]) == {"web_search"}


@pytest.mark.asyncio
async def test_a_guardrail_that_edits_a_tool_still_wins_on_the_tools_it_saw():
    """Preserving diverted tools must not freeze the ones the guardrail CAN edit.

    Otherwise this fix would silently disable tool masking.
    """
    data = _request(WEB_SEARCH, REGULAR)

    guardrail = MagicMock()

    async def _mask(inputs, **_):
        edited = []
        for tool in inputs.get("tools") or []:
            fn = dict(tool.get("function") or {})
            fn["description"] = "[MASKED]"
            edited.append({**tool, "function": fn})
        return {**inputs, "tools": edited}

    guardrail.apply_guardrail = AsyncMock(side_effect=_mask)

    returned = await AnthropicMessagesHandler().process_input_messages(
        data=data, guardrail_to_apply=guardrail, litellm_logging_obj=MagicMock()
    )

    assert _tool_ids(returned["tools"]) == {"web_search", "get_weather"}
    masked = next(t for t in returned["tools"] if t.get("name") == "get_weather")
    assert masked.get("description") == "[MASKED]", "the guardrail's edit was lost"


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "web_search_20250305", "name": "web_search"},
        {"type": "web_fetch_20250910", "name": "web_fetch"},
        {"type": "code_execution_20250522", "name": "code_execution"},
        {"type": "bash_20250124", "name": "bash"},
        {"type": "text_editor_20250124", "name": "str_replace_editor"},
        {"type": "computer_20250124", "name": "computer", "display_width_px": 1024, "display_height_px": 768},
        {"type": "memory_20250818", "name": "memory"},
        {"type": "tool_search_tool_bm25_20251119", "name": "tool_search"},
        {"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-8"},
        {"type": "mcp_toolset", "name": "mcp"},
        {"type": "some_future_hosted_tool_20991231", "name": "not_yet_invented"},
    ],
    ids=lambda t: t["type"],
)
@pytest.mark.asyncio
async def test_no_hosted_tool_type_is_dropped(tool):
    """Every hosted tool must survive, not just the one that was reported.

    web_search is how this surfaced, but the seam is type-agnostic: anything the
    OpenAI translation cannot express gets diverted or reshaped. The last case is a
    deliberately invented type — the fix works by difference against what the
    translation produced, so a tool type nobody has written down yet is covered too.
    Enumerating known types here would only prove the enumeration.
    """
    data = _request(tool, REGULAR)

    returned = await AnthropicMessagesHandler().process_input_messages(
        data=data, guardrail_to_apply=_noop_guardrail(), litellm_logging_obj=MagicMock()
    )

    assert _tool_ids(returned["tools"]) == {tool["name"], "get_weather"}


@pytest.mark.asyncio
async def test_a_reviewed_tool_carrying_an_explicit_type_is_not_duplicated():
    """A caller tool with BOTH `name` and `type` must be matched on `name`.

    Anthropic's own shape for a caller-defined tool is ``{"type": "custom", "name": ...}``,
    and it is legal on the way IN as well as the way back. `_tool_identity` must read
    `name` before `type`: keyed on `type`, every such tool collapses to the identity
    ``"custom"``, fails to match its reviewed counterpart, and gets re-attached on top of
    it — the request then carries the tool TWICE, once reviewed and once not.

    Regression guard with real provenance: a reviewer's mutation flipped that precedence
    and the whole suite stayed green, because every other case here sends a tool with a
    `name` and no `type` (where both orderings agree). This is the input that separates them.
    """
    typed_regular = {
        "type": "custom",
        "name": "get_weather",
        "description": "Look up the weather",
        "input_schema": {"type": "object", "properties": {}},
    }
    data = _request(WEB_SEARCH, typed_regular)

    returned = await AnthropicMessagesHandler().process_input_messages(
        data=data, guardrail_to_apply=_noop_guardrail(), litellm_logging_obj=MagicMock()
    )

    names = [t.get("name") for t in returned["tools"]]
    assert names.count("get_weather") == 1, f"the reviewed tool was duplicated: {names}"
    assert _tool_ids(returned["tools"]) == {"web_search", "get_weather"}


def _blocking_guardrail():
    """A guardrail that inspects the tools and then REMOVES them all.

    The fix must not resurrect a tool the guardrail deliberately dropped — that is the
    difference between "preserve what was never reviewed" and "override the reviewer".
    """
    guardrail = MagicMock()
    guardrail.apply_guardrail = AsyncMock(side_effect=lambda inputs, **_: {**inputs, "tools": []})
    return guardrail


LONG_NAME = "a" * 70  # over the 64-char cap, so the translation rewrites it


def _long_named_tool():
    return {
        "name": LONG_NAME,
        "description": "a tool whose name the OpenAI translation must truncate",
        "input_schema": {"type": "object", "properties": {"secret": {"type": "string"}}},
    }


@pytest.mark.asyncio
async def test_a_truncated_tool_name_is_not_mistaken_for_a_diverted_tool():
    """Names over 64 chars are REWRITTEN by the translation, not diverted.

    `translate_anthropic_tools_to_openai` rewrites any name longer than 64 characters to
    `<55-char prefix>_<8-char hash>` and records the pair in the mapping it returns. That
    mapping used to be discarded, so the caller-side name never matched the translated
    one and the tool was misread as diverted — arriving TWICE, once reviewed and once raw.
    """
    data = _request(_long_named_tool())

    returned = await AnthropicMessagesHandler().process_input_messages(
        data=data, guardrail_to_apply=_noop_guardrail(), litellm_logging_obj=MagicMock()
    )

    assert len(returned["tools"]) == 1, f"the tool was duplicated across the seam: {returned['tools']}"


@pytest.mark.asyncio
async def test_a_blocked_long_named_tool_is_not_resurrected():
    """The sharpest case: a guardrail removed it, and it must stay removed.

    This is a security property, not a cosmetic one — the tool DID reach the guardrail
    (under its truncated name) and was rejected. Re-attaching it would ship the pristine
    original, schema and all, straight past the decision.
    """
    data = _request(_long_named_tool())

    returned = await AnthropicMessagesHandler().process_input_messages(
        data=data, guardrail_to_apply=_blocking_guardrail(), litellm_logging_obj=MagicMock()
    )

    assert returned["tools"] == [], f"a blocked tool came back: {returned['tools']}"


@pytest.mark.asyncio
async def test_a_tool_with_no_name_and_no_type_is_left_to_the_writeback():
    """An unidentifiable tool cannot be matched — and must not be assumed diverted.

    `kept` holds only truthy identities, so it can never contain `None`. Treating a
    `None` identity as "not in kept" made such a tool re-attach unconditionally: it
    duplicated on the approve path and survived a block.
    """
    nameless = {"description": "carries neither name nor type"}

    approved = await AnthropicMessagesHandler().process_input_messages(
        data=_request(nameless), guardrail_to_apply=_noop_guardrail(), litellm_logging_obj=MagicMock()
    )
    assert len(approved["tools"]) == 1, f"duplicated: {approved['tools']}"

    blocked = await AnthropicMessagesHandler().process_input_messages(
        data=_request(nameless), guardrail_to_apply=_blocking_guardrail(), litellm_logging_obj=MagicMock()
    )
    assert blocked["tools"] == [], f"a blocked tool came back: {blocked['tools']}"


@pytest.mark.asyncio
async def test_a_blocked_reviewed_tool_stays_blocked_while_a_diverted_one_survives():
    """Both halves of the contract in one request, so neither can regress alone."""
    data = _request(WEB_SEARCH, REGULAR)

    returned = await AnthropicMessagesHandler().process_input_messages(
        data=data, guardrail_to_apply=_blocking_guardrail(), litellm_logging_obj=MagicMock()
    )

    assert _tool_ids(returned["tools"]) == {"web_search"}, (
        f"expected the reviewed tool gone and the diverted one kept: {returned['tools']}"
    )


@pytest.mark.asyncio
async def test_a_request_with_no_tools_is_untouched():
    data = {
        "model": "claude-sonnet-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    }

    returned = await AnthropicMessagesHandler().process_input_messages(
        data=data, guardrail_to_apply=_noop_guardrail(), litellm_logging_obj=MagicMock()
    )

    assert "tools" not in returned or returned["tools"] == []
