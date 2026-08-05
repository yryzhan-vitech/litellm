"""[ARC-BUG-48] Two prod patches the de-fork silently dropped, and the guard that stops a third.

Both losses came from ONE fork commit (be5879c8752) whose hunks in advisor.py, handler.py and
guardrail_translation were carried across the rebase while its footprint in these two files was
not. The commit read as "handled" while two of its thirteen files were never diffed, and a ruff
reformat on the clean side made them look recently touched.

The first loss was client-visible in production: a `server_tool_use` block emitted a
`content_block_stop` with no matching `content_block_start`, so its `id` never reached the wire
and a client replaying that turn as history got
`400 messages.N.content.M.server_tool_use.id: String should match pattern '^srvtoolu_...'`.

These tests assert the STREAM SHAPE rather than the presence of a branch, because the defect was
an asymmetry between two loops — one that handles four types and one that stops all of them.
"""

import json

import pytest

from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
    FakeAnthropicMessagesStreamIterator,
)
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    _normalize_anthropic_advisor_tool_models,
)
from litellm.types.llms.anthropic import ANTHROPIC_ADVISOR_TOOL_TYPE


def _events(content_blocks):
    """Drive the iterator and return (event_name, parsed_data) pairs."""
    response = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": content_blocks,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    out = []
    for raw in FakeAnthropicMessagesStreamIterator(response).chunks:
        text = raw.decode()
        name = None
        data = None
        for line in text.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if name:
            out.append((name, data))
    return out


def _starts_and_stops(events):
    starts = {e[1]["index"] for e in events if e[0] == "content_block_start"}
    stops = {e[1]["index"] for e in events if e[0] == "content_block_stop"}
    return starts, stops


class TestServerToolUseSurvivesTheFakeStream:
    """The exact production defect, asserted three ways."""

    BLOCK = {
        "type": "server_tool_use",
        "id": "srvtoolu_01WebSearchAbc123",
        "name": "web_search",
        "input": {"query": "anything"},
    }

    def test_server_tool_use_gets_a_start_not_just_a_stop(self):
        """The defect itself: a stop with no start at that index."""
        events = _events([{"type": "thinking", "thinking": "hm", "signature": "s"}, self.BLOCK])
        starts, stops = _starts_and_stops(events)
        assert stops == {0, 1}, "the caller loop stops every index; that part was never broken"
        assert starts == {0, 1}, f"index 1 must also START, got starts={starts}"

    def test_the_dedicated_branch_shapes_the_block_not_just_the_fail_open_guard(self):
        """Distinguish the restored branch from the fail-open else that also catches this.

        Worth stating plainly: the fail-open guard ALONE already prevents the production 400,
        because it emits a well-formed content_block_start carrying the id. Measured by deleting
        this branch — the stream stays balanced and the id still reaches the wire. That is the
        guard doing exactly its job.

        So without this assertion the branch could be deleted with every other test green. The
        difference is the SHAPE: the dedicated branch emits a curated content_block with just
        type/id/name, matching what the first-party API sends, while the guard passes the raw
        block through including fields like `input` that Anthropic does not put in a start event.
        """
        events = _events([self.BLOCK])
        start = next(d for n, d in events if n == "content_block_start")
        cb = start["content_block"]
        assert set(cb.keys()) == {"type", "id", "name"}, (
            f"the dedicated branch must emit exactly type/id/name, got {sorted(cb.keys())} — "
            "if this fails with 'input' present, the branch was deleted and the fail-open "
            "guard is handling it instead"
        )

    def test_the_srvtoolu_id_reaches_the_wire(self):
        """What the client needs in order to replay the turn as history.

        This is the assertion that maps directly to the production 400: with the branch missing
        the id appeared nowhere in the stream, so a client had nothing to reconstruct.
        """
        events = _events([self.BLOCK])
        blob = json.dumps(events)
        assert self.BLOCK["id"] in blob, "the server_tool_use id must appear somewhere in the stream"
        start = next(d for n, d in events if n == "content_block_start")
        assert start["content_block"]["type"] == "server_tool_use"
        assert start["content_block"]["id"] == self.BLOCK["id"]
        assert start["content_block"]["name"] == "web_search"

    def test_every_index_is_balanced_for_a_realistic_advisor_turn(self):
        """The shape an advisor turn actually produces: thinking, server tool, result, text.

        Asserted as a set equality over all four indices rather than per-block, because the bug
        was an asymmetry between two loops and only a whole-stream check catches that class.
        """
        events = _events(
            [
                {"type": "thinking", "thinking": "reasoning", "signature": "sig"},
                self.BLOCK,
                {
                    "type": "advisor_tool_result",
                    "tool_use_id": self.BLOCK["id"],
                    "content": {"type": "advisor_result", "text": "advice"},
                },
                {"type": "text", "text": "final answer"},
            ]
        )
        starts, stops = _starts_and_stops(events)
        assert starts == stops == {0, 1, 2, 3}, f"unbalanced stream: starts={starts} stops={stops}"


class TestAdvisorToolResultSurvivesTheFakeStream:
    def test_advisor_tool_result_start_carries_the_tool_use_id(self):
        events = _events(
            [
                {
                    "type": "advisor_tool_result",
                    "tool_use_id": "srvtoolu_01AdvisorXyz",
                    "content": {"type": "advisor_result", "text": "the advice"},
                }
            ]
        )
        start = next(d for n, d in events if n == "content_block_start")
        assert start["content_block"]["type"] == "advisor_tool_result"
        assert start["content_block"]["tool_use_id"] == "srvtoolu_01AdvisorXyz"

    def test_advisor_text_is_delivered_as_a_delta(self):
        events = _events(
            [
                {
                    "type": "advisor_tool_result",
                    "tool_use_id": "srvtoolu_01AdvisorXyz",
                    "content": {"type": "advisor_result", "text": "the advice"},
                }
            ]
        )
        deltas = [d for n, d in events if n == "content_block_delta"]
        assert deltas, "advisor text must be streamed, not silently dropped"
        assert deltas[0]["delta"]["text"] == "the advice"
        assert deltas[0]["delta"]["type"] == "advisor_result_delta"

    def test_a_malformed_advisor_content_does_not_raise(self):
        """content is caller-shaped; a string or None where a dict is expected must not crash."""
        for bad in ("just a string", None, 42, []):
            events = _events([{"type": "advisor_tool_result", "tool_use_id": "x", "content": bad}])
            starts, stops = _starts_and_stops(events)
            assert starts == stops == {0}, f"content={bad!r} produced an unbalanced stream"


class TestUnknownBlockTypesFailOpen:
    """The part that was MISSING rather than lost, and the reason this cannot recur silently.

    Anthropic keeps adding content block types. Before this guard, the only two possible shapes
    for an unhandled type were "a start we synthesised" and "a stop with no start" — and the code
    produced the second, which corrupts the caller's history. Failing open degrades instead.
    """

    def test_an_unknown_type_still_gets_a_start(self):
        events = _events([{"type": "some_future_block_20270101", "id": "fut_1", "data": {"k": "v"}}])
        starts, stops = _starts_and_stops(events)
        assert starts == stops == {0}, "an unknown type must not emit a bare stop"

    def test_the_unknown_block_is_passed_through_verbatim(self):
        """Deliberately not reshaped per type.

        Guessing at a shape is what produced the ARC-BUG-44/45 family, where an allowlist
        rebuilt native blocks lossily and stripped fields it did not enumerate.
        """
        block = {"type": "some_future_block_20270101", "id": "fut_1", "nested": {"deep": [1, 2]}}
        events = _events([block])
        start = next(d for n, d in events if n == "content_block_start")
        assert start["content_block"] == block, "the block must survive byte-for-byte"

    @pytest.mark.parametrize(
        "hostile,label",
        [
            ({"type": "web_search_tool_result", "page_age": __import__("datetime").datetime(2026, 8, 5)}, "datetime"),
            ({"type": "mcp_tool_result", "data": b"raw"}, "bytes"),
            ({"type": "mcp_tool_use", "score": float("nan")}, "NaN"),
            ({"type": "mcp_tool_use", "score": float("inf")}, "inf"),
            ({"type": "container_upload", "obj": object()}, "arbitrary object"),
        ],
    )
    def test_an_unserialisable_block_does_not_kill_the_stream(self, hostile, label):
        """🔴 The first version of this guard failed CLOSED, which is worse than the bug it fixes.

        _create_streaming_chunks runs in __init__, so an unguarded json.dumps raise escapes the
        CONSTRUCTOR: the caller never receives an iterator and the client gets a 500 with ZERO SSE
        bytes. That is a third shape the original reasoning did not enumerate — not "a start" or
        "a bare stop" but "no stream at all".

        Reachable on the ALREADY-LIVE websearch path, independent of the advisor:
        websearch_interception injects web_search_tool_result blocks whose page_age comes from
        getattr off a model declaring extra="allow", so a provider transform assigning a datetime
        bypasses validation. Measured as TypeError from __init__ before the guard.

        NaN and inf are the nastier pair: json.dumps ACCEPTS them and emits the bare literals,
        which are not valid JSON — Go encoding/json, serde_json and JSON.parse all reject them.
        That one fails client-side mid-stream, after bytes are committed.
        """
        events = _events([hostile])
        starts, stops = _starts_and_stops(events)
        assert starts == stops == {0}, f"{label}: the stream must stay balanced"
        raw = json.dumps(events)
        assert "NaN" not in raw and "Infinity" not in raw, f"{label}: invalid JSON literal reached the wire"
        start = next(d for n, d in events if n == "content_block_start")
        assert start["content_block"]["type"] == hostile["type"], f"{label}: the type must survive"

    def test_a_degraded_block_says_so_in_the_log(self, caplog):
        """An operator must be able to tell "passed through" from "we could not serialise it"."""
        import logging

        with caplog.at_level(logging.WARNING):
            _events([{"type": "mcp_tool_result", "data": b"raw"}])
        assert "not JSON-serialisable" in caplog.text

    def test_an_unknown_type_is_logged_so_it_gets_a_branch_later(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            _events([{"type": "some_future_block_20270101"}])
        assert "no handler for content block type" in caplog.text
        assert "some_future_block_20270101" in caplog.text

    def test_a_block_with_no_type_at_all_is_survivable(self):
        events = _events([{"text": "no type key"}])
        starts, stops = _starts_and_stops(events)
        assert starts == stops == {0}


class TestAdvisorModelNormalisation:
    """The second lost hunk. Its call site survived the de-fork; only these lines were cut."""

    def test_a_proxy_alias_resolves_to_the_canonical_model(self, monkeypatch):
        """Without this, the alias reaches Anthropic verbatim and earns a 400 that reads as a
        caller mistake but is ours."""

        class _Router:
            model_list = [
                {"model_name": "claude_opus", "litellm_params": {"model": "anthropic/claude-opus-4-6"}},
            ]

        import litellm.proxy.proxy_server as ps

        monkeypatch.setattr(ps, "llm_router", _Router(), raising=False)
        out = _normalize_anthropic_advisor_tool_models([{"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "model": "claude_opus"}])
        assert out[0]["model"] == "claude-opus-4-6", "alias must resolve AND lose the provider prefix"

    def test_a_provider_prefix_is_stripped_even_without_a_router_hit(self, monkeypatch):
        class _Router:
            model_list = []

        import litellm.proxy.proxy_server as ps

        monkeypatch.setattr(ps, "llm_router", _Router(), raising=False)
        out = _normalize_anthropic_advisor_tool_models(
            [{"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "model": "anthropic/claude-opus-4-6"}]
        )
        assert out[0]["model"] == "claude-opus-4-6"

    def test_non_advisor_tools_are_returned_untouched(self):
        """The ARC-BUG-44/45 lesson: never rebuild a tool this function does not own."""
        tools = [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "custom", "name": "Bash", "input_schema": {"type": "object"}},
            "not even a dict",
        ]
        out = _normalize_anthropic_advisor_tool_models(list(tools))
        assert out == tools

    @pytest.mark.parametrize("bad_model", [None, "", "   ", 42, {"nested": 1}])
    def test_a_malformed_advisor_model_is_left_for_the_validator(self, bad_model):
        """Normalisation is not validation. ARC-BUG-46 owns rejecting these with a 400; this
        function must not crash on them and must not invent a value."""
        out = _normalize_anthropic_advisor_tool_models([{"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "model": bad_model}])
        assert out[0].get("model") == bad_model

    def test_the_input_list_is_not_mutated(self):
        """ARC-BUG-44 was caused by in-place mutation of a caller's tools list."""
        original = [{"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "model": "anthropic/claude-opus-4-6"}]
        snapshot = json.loads(json.dumps(original))
        _normalize_anthropic_advisor_tool_models(original)
        assert original == snapshot, "the caller's list must be left alone"


class TestTheNormaliserIsActuallyWired:
    """A restored function nobody calls is still a lost patch.

    The de-fork cut the definitions AND the three lines that invoke them. Testing the function
    in isolation proves the definition came back; only driving transform_anthropic_messages_request
    proves the call site did. Measured while writing this: deleting the call left every other
    test in this file green.
    """

    def test_transform_normalises_the_advisor_model_end_to_end(self, monkeypatch):
        class _Router:
            model_list = [
                {"model_name": "claude_opus", "litellm_params": {"model": "anthropic/claude-opus-4-6"}},
            ]

        import litellm.proxy.proxy_server as ps

        monkeypatch.setattr(ps, "llm_router", _Router(), raising=False)

        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        tools = [{"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "model": "claude_opus"}]
        optional_params = {"tools": tools, "max_tokens": 16}
        AnthropicMessagesConfig().transform_anthropic_messages_request(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "hi"}],
            anthropic_messages_optional_request_params=optional_params,
            litellm_params={},
            headers={},
        )
        sent = optional_params["tools"][0]["model"]
        assert sent == "claude-opus-4-6", (
            f"the alias reached the request body unresolved as {sent!r} — the normaliser is defined but not called"
        )
