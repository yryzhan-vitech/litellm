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
from datetime import datetime

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
        """An operator must be able to tell "passed through" from "we had to coerce it".

        The property is unchanged from round one; only the emitter moved. Round one degraded inside
        this branch and said "not JSON-serialisable"; the coercion now happens in _sse, so the
        signal is its warning. Assert BOTH warnings fire — the type-level one says which block has
        no branch, the frame-level one says its content was rewritten. Losing either leaves an
        operator unable to distinguish a clean passthrough from a lossy one.
        """
        import logging

        with caplog.at_level(logging.WARNING):
            _events([{"type": "mcp_tool_result", "data": b"raw"}])
        assert "no handler for content block type" in caplog.text, "which block lacks a branch"
        assert "coerced it" in caplog.text, "that the content was rewritten, not passed through"
        assert "mcp_tool_result" in caplog.text

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

    # 🔴 Every fixture above is anthropic-backed, and that gap is exactly why the prod version of
    # this function shipped a 3-for-1 regression through 215 tests and two reviews. On prd-ai 20 of
    # 36 groups map to `bedrock/`, so the realistic case was the untested one.
    PRD_SHAPED_ROUTER = [
        {"model_name": "claude-opus-4-6", "litellm_params": {"model": "bedrock/us.anthropic.claude-opus-4-6-v1:0"}},
        {"model_name": "claude-sonnet-5", "litellm_params": {"model": "bedrock/us.anthropic.claude-sonnet-5-v1:0"}},
        {"model_name": "claude-haiku-4-5", "litellm_params": {"model": "bedrock/us.anthropic.claude-haiku-4-5-v1:0"}},
        {"model_name": "claude_opus", "litellm_params": {"model": "anthropic/claude-opus-4-6"}},
        {"model_name": "gpt-ish", "litellm_params": {"model": "openai/gpt-4o"}},
        {"model_name": "vert", "litellm_params": {"model": "vertex_ai/claude-opus-4-6"}},
    ]

    @pytest.fixture
    def prd_router(self, monkeypatch):
        class _Router:
            model_list = TestAdvisorModelNormalisation.PRD_SHAPED_ROUTER

        import litellm.proxy.proxy_server as ps

        monkeypatch.setattr(ps, "llm_router", _Router(), raising=False)

    @pytest.mark.parametrize("bare_alias", ["claude-opus-4-6", "claude-sonnet-5", "claude-haiku-4-5"])
    def test_a_bedrock_backed_alias_is_forwarded_UNCHANGED(self, prd_router, bare_alias):
        """The regression this function shipped, pinned: was ok, became a 400.

        A bare alias is BOTH our `model_name` and a valid Anthropic model id, so it is already
        correct on the wire. The prod hunk resolved it through the router and wrote the result back
        unconditionally, turning `claude-sonnet-5` into `bedrock/us.anthropic.claude-sonnet-5-v1:0`
        — a value Anthropic has never heard of. Forwarding the caller's own string and letting
        Anthropic judge it is the only defensible behaviour.
        """
        out = _normalize_anthropic_advisor_tool_models([{"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "model": bare_alias}])
        assert out[0]["model"] == bare_alias
        assert "/" not in out[0]["model"], "a provider prefix on this field is an instant 400"

    @pytest.mark.parametrize("alias", ["gpt-ish", "vert"])
    def test_a_non_anthropic_backed_alias_is_not_rewritten(self, prd_router, alias):
        """`openai/` and `vertex_ai/` are no more convertible than `bedrock/`.

        Guards the general rule rather than the one provider that bit us: the only safe rewrite is
        one that lands on a bare id, so anything else is passed through untouched.
        """
        out = _normalize_anthropic_advisor_tool_models([{"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "model": alias}])
        assert out[0]["model"] == alias

    def test_an_anthropic_backed_alias_still_resolves(self, prd_router):
        """The fix must not be a no-op — the one genuine win has to survive it.

        Without this the whole function could be reduced to `return tools` and every other test in
        this class would still pass.
        """
        out = _normalize_anthropic_advisor_tool_models([{"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "model": "claude_opus"}])
        assert out[0]["model"] == "claude-opus-4-6"

    def test_a_caller_supplied_bedrock_string_is_left_for_anthropic_to_reject(self, prd_router):
        """Not our job to translate: `us.anthropic.claude-...-v1:0` carries a region prefix and a
        version suffix no Anthropic id has, and reconstructing one by string surgery is the guessing
        that produced the ARC-BUG-44/45 family."""
        raw = "bedrock/us.anthropic.claude-opus-4-6-v1:0"
        out = _normalize_anthropic_advisor_tool_models([{"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "model": raw}])
        assert out[0]["model"] == raw

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


class TestNoCallerShapedValueCanKillTheStream:
    """[ARC-BUG-48, review round 2] The constructor must not raise, whatever the block holds.

    The first version of the fail-open guard hardened only the unknown-type branch, leaving every
    sibling — tool_use.input, text.text, thinking.signature, server_tool_use.id,
    advisor_tool_result.content.text — and the message_start / message_delta usage envelope calling
    json.dumps bare. A review flagged it, and a probe of every branch confirmed each one escaped.

    Why the escape is so much worse than a malformed frame: `_create_streaming_chunks` runs in
    `__init__`, so the raise leaves the CONSTRUCTOR. The caller never receives an iterator and the
    client gets a 500 with ZERO SSE bytes — strictly worse than the 400 this file exists to fix.

    Reachable without the advisor: websearch_interception injects `web_search_tool_result` blocks
    whose `page_age` is read via getattr off a model declaring extra="allow", so a provider
    transform assigning a datetime bypasses validation. This iterator has six call sites.

    Every case below is one the probe caught escaping. They assert BALANCE (starts == stops) and
    STRICT parseability, because those are the two properties a client actually depends on.
    """

    HOSTILE = [
        pytest.param({"type": "text", "text": datetime(2026, 8, 5)}, id="text.text-datetime"),
        pytest.param(
            {"type": "thinking", "thinking": "t", "signature": b"\xff\xfe"},
            id="thinking.signature-bytes",
        ),
        pytest.param(
            {"type": "tool_use", "id": datetime(2026, 8, 5), "name": "n", "input": {}},
            id="tool_use.id-datetime",
        ),
        pytest.param(
            {"type": "tool_use", "id": "toolu_01Keep", "name": "n", "input": {"when": datetime(2026, 8, 5)}},
            id="tool_use.input-datetime-NESTED-dumps",
        ),
        pytest.param(
            {"type": "server_tool_use", "id": "srvtoolu_01Keep", "name": b"web_search"},
            id="server_tool_use.name-bytes",
        ),
        pytest.param(
            {
                "type": "advisor_tool_result",
                "tool_use_id": "srvtoolu_01Adv",
                "content": {"type": "advisor_result", "text": datetime(2026, 8, 5)},
            },
            id="advisor_tool_result.text-datetime",
        ),
        pytest.param(
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01Keep",
                "content": [{"type": "web_search_result", "page_age": datetime(2026, 8, 5)}],
            },
            id="unknown-type-datetime-page_age-THE-LIVE-PATH",
        ),
        pytest.param({"type": datetime(2026, 8, 5), "id": "srvtoolu_01Keep"}, id="the-type-itself-datetime"),
    ]

    @pytest.mark.parametrize("block", HOSTILE)
    def test_the_constructor_does_not_raise_and_the_stream_stays_balanced(self, block):
        events = _events([block])  # _events uses json.loads, so it rejects NaN/Infinity too
        starts, stops = _starts_and_stops(events)
        assert starts == stops, f"unbalanced: starts={starts} stops={stops}"
        assert events[0][0] == "message_start"
        assert events[-1][0] == "message_stop"

    def test_the_nested_partial_json_is_sanitised_not_merely_caught(self):
        """`partial_json` is a JSON string INSIDE a frame, so _sse cannot protect its own argument.

        This is the one site the helper could not cover: json.dumps(input) runs while the payload is
        being BUILT, before _sse is entered. The probe caught it still escaping after the helper
        landed. Assert the value, not just the absence of a raise — a caught-and-dropped `input`
        would satisfy a balance-only check while silently losing the tool arguments.
        """
        events = _events(
            [{"type": "tool_use", "id": "toolu_01Keep", "name": "n", "input": {"when": datetime(2026, 8, 5), "n": 1}}]
        )
        deltas = [e[1] for e in events if e[0] == "content_block_delta"]
        assert len(deltas) == 1
        inner = json.loads(deltas[0]["delta"]["partial_json"])
        assert inner["n"] == 1, "a serialisable sibling field must survive the coercion"
        assert inner["when"] == "2026-08-05 00:00:00", "the datetime is stringified, not dropped"

    @pytest.mark.parametrize(
        "usage, field",
        [
            ({"input_tokens": 1, "output_tokens": float("nan")}, "output_tokens"),
            ({"input_tokens": float("inf"), "output_tokens": 1}, "input_tokens"),
        ],
    )
    def test_non_finite_usage_never_reaches_the_wire_as_a_bare_literal(self, usage, field):
        """json.dumps emits bare NaN / Infinity, which are not JSON.

        Go's encoding/json, serde_json and JSON.parse all reject them, and this one fails
        CLIENT-side mid-stream after bytes are committed — the least recoverable position of all.

        It also proves why `default=str` was not enough: json.dumps never consults `default` for a
        float, so allow_nan=False raises before any coercion hook runs. That is what forced the
        explicit _json_safe walk rather than a one-line dumps argument.
        """
        response = {"id": "m", "role": "assistant", "model": "m", "usage": usage, "content": []}
        for raw in FakeAnthropicMessagesStreamIterator(response).chunks:
            body = raw.decode().split("data: ", 1)[1].strip()
            assert "NaN" not in body and "Infinity" not in body
            json.loads(body)  # strict by default: would raise on a bare literal

    def test_a_str_that_raises_does_not_escape_the_constructor(self):
        """_json_safe was documented as total and was not — str() runs caller code.

        Caught by the probe, not by reasoning: a `__str__` raising RuntimeError went straight out of
        __init__, reproducing the exact failure the sanitiser exists to prevent. This is also why
        both _sse and _json_safe catch bare Exception rather than (TypeError, ValueError) — the set
        of things caller-controlled `__str__` / `__iter__` / property getters can raise is not
        enumerable.
        """

        class Unstringable:
            def __str__(self):
                raise RuntimeError("no str for you")

            def __repr__(self):
                raise RuntimeError("nor repr")

        events = _events([{"type": "weird", "payload": Unstringable()}])
        starts, stops = _starts_and_stops(events)
        assert starts == stops

    def test_the_leaf_guard_saves_the_siblings_of_a_bad_value(self):
        """Two guards cover the escape; only the LEAF one preserves data. That is why both exist.

        Negative controls showed that narrowing _sse's catch OR the leaf catch individually failed
        no test, because either layer alone stops the constructor escape. Redundancy is intended, but
        it made the escape tests unable to justify the leaf guard on its own. The distinguishing
        property is data: when str() raises at a leaf, _json_safe replaces just that leaf and every
        sibling survives, whereas falling through to _sse's minimal fallback discards the entire
        content_block — including the `tool_use_id` a client needs to replay history, which is the
        original ARC-BUG-48 symptom.

        An earlier revision also wrapped the dict comprehension in its own try/except. A control
        proved that coarser net MASKED the leaf guard — it swallowed the RuntimeError, collapsed the
        whole dict to a placeholder, and destroyed exactly the siblings this test protects.
        """

        class Unstringable:
            def __str__(self):
                raise RuntimeError("no str")

            def __repr__(self):
                raise RuntimeError("nor repr")

        events = _events(
            [
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_01Keep",
                    "url": "https://example",
                    "count": 7,
                    "bad": Unstringable(),
                }
            ]
        )
        block = next(d for n, d in events if n == "content_block_start")["content_block"]
        assert block["tool_use_id"] == "srvtoolu_01Keep", "the id must not be collateral damage"
        assert block["url"] == "https://example"
        assert block["count"] == 7
        assert isinstance(block["bad"], str), "only the offending leaf is replaced"

    def test_an_unstringable_dict_key_is_survivable(self):
        """Keys go through str() too, so they need the same treatment as values."""

        class BadKey:
            def __str__(self):
                raise RuntimeError("no str")

            def __repr__(self):
                raise RuntimeError("nor repr")

            def __hash__(self):
                return 1

        events = _events([{"type": "weird", "map": {BadKey(): "v", "good": "w"}}])
        block = next(d for n, d in events if n == "content_block_start")["content_block"]
        assert block["map"]["good"] == "w", "a sane sibling key survives"

    def test_a_self_referential_block_terminates(self):
        """The depth cap, which is the one shape a type-driven walk cannot detect."""
        cyclic: dict = {}
        cyclic["self"] = cyclic
        events = _events([{"type": "weird", "loop": cyclic}])
        starts, stops = _starts_and_stops(events)
        assert starts == stops

    def test_the_unknown_type_guard_now_preserves_the_whole_block(self):
        """A regression pin on the second review round, which REPLACED a lossier degrade path.

        Round one's recovery kept only `type` plus a string `id`/`tool_use_id`. That dropped
        `tool_use_id` for shapes that carry it elsewhere and every field a future block type
        introduces — turning a 500 into the very 400 this file restores the branches to prevent.
        Routing through _sse's str() coercion keeps the whole block with one field stringified,
        which is strictly more than the old path preserved.
        """
        events = _events(
            [
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_01Keep",
                    "content": [{"type": "web_search_result", "url": "u", "page_age": datetime(2026, 8, 5)}],
                }
            ]
        )
        start = next(e[1] for e in events if e[0] == "content_block_start")
        block = start["content_block"]
        assert block["tool_use_id"] == "srvtoolu_01Keep", "the id a client needs for history"
        assert block["content"][0]["url"] == "u", "sibling fields survive, not just the id"
        assert block["content"][0]["page_age"] == "2026-08-05 00:00:00"

    def test_a_clean_response_is_untouched_by_the_guard(self):
        """The fast path must not pay for the slow one — no coercion, no reordering, no extra frames.

        Without this, every assertion above could be satisfied by a sanitiser that rewrites normal
        traffic too, and the coercion would silently become the steady state.
        """
        blocks = [
            {"type": "text", "text": "hello"},
            {"type": "server_tool_use", "id": "srvtoolu_01Real", "name": "web_search"},
        ]
        events = _events(blocks)
        assert [e[0] for e in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        text_delta = next(e[1] for e in events if e[0] == "content_block_delta")
        assert text_delta["delta"]["text"] == "hello"
        stu = [e[1] for e in events if e[0] == "content_block_start"][1]["content_block"]
        assert stu == {"type": "server_tool_use", "id": "srvtoolu_01Real", "name": "web_search"}
