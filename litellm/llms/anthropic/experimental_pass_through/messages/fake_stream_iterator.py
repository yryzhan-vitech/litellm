"""
Fake Streaming Iterator for Anthropic Messages

This module provides a fake streaming iterator that converts non-streaming
Anthropic Messages responses into proper streaming format.

Used when WebSearch interception converts stream=True to stream=False but
the LLM doesn't make a tool call, and we need to return a stream to the user.
"""

import json
import math
from typing import Any, Dict, List, cast

from litellm._logging import verbose_logger
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)


def _json_safe(value: Any, depth: int = 0) -> Any:
    """Coerce an arbitrary value into something json.dumps accepts with allow_nan=False.

    Total by construction — every branch either returns a JSON primitive or recurses. The depth cap
    terminates self-referential structures, which are the one shape a purely type-driven walk cannot
    detect.

    🔴 The str() fallthrough is guarded, because a probe proved the earlier "total" claim false: a
    __str__ that raises RuntimeError escaped straight out of the constructor, which is the exact
    failure mode this whole function exists to prevent. Bare `except Exception` is deliberate — the
    caller controls both the value and its __str__, so the set of escapes is not enumerable.

    Non-finite floats become None rather than a string because they sit in numeric fields
    (`usage.output_tokens`), where a client doing arithmetic on "nan" would fail in a second place.
    JSON has no NaN, so null is the only honest representation of an unrepresentable number.
    """
    if depth > 8:
        return str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        # No try/except at this level on purpose. An earlier revision wrapped the whole
        # comprehension, and a negative control proved that guard MASKED the leaf-level one: a
        # __str__ raising RuntimeError was swallowed here, the dict collapsed to a placeholder
        # string, and the test that should have caught a narrowed leaf catch passed anyway. Since
        # every leaf guards its own str(), the coarse net only ever destroyed sibling data that
        # would otherwise have survived. _sse's outer catch remains the backstop for anything
        # exotic (a dict subclass whose .items() raises).
        return {_json_safe_key(k): _json_safe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, depth + 1) for v in value]
    try:
        return str(value)
    except Exception:
        return f"<unserialisable {type(value).__name__}>"


def _json_safe_key(key: Any) -> str:
    """JSON object keys must be strings, and str() runs caller code, so it needs the same guard."""
    if isinstance(key, str):
        return key
    try:
        return str(key)
    except Exception:
        return f"<unserialisable {type(key).__name__} key>"


def _sse(event: str, payload: Dict[str, Any]) -> bytes:
    """[ARC-BUG-48] Serialise one SSE frame so caller-shaped data cannot kill the stream.

    Every frame in this module runs through here because `_create_streaming_chunks` is called from
    `__init__`: an escaping json.dumps raise means the caller never receives an iterator at all and
    the client gets a 500 with ZERO SSE bytes. A review found that the hardening added for the
    unknown-type branch left every sibling branch — tool_use.input, text.text, thinking.signature,
    server_tool_use.id, advisor_tool_result.content.text — and the message_start / message_delta
    usage envelope still unguarded, so the same datetime or bytes reaching any of them produced the
    identical constructor escape.

    Reachable, not hypothetical: websearch_interception injects blocks whose `page_age` is read via
    getattr off a model declaring extra="allow", so a provider transform assigning a datetime
    bypasses validation.

    allow_nan=False because json.dumps otherwise emits the bare literals NaN / Infinity, which are
    not valid JSON — Go encoding/json, serde_json and JSON.parse all reject them. That failure lands
    client-side mid-stream, after bytes are committed, which is the least recoverable position of
    all — and `default=str` does NOT cover it, because json.dumps never consults `default` for a
    float; allow_nan=False raises before it gets there. So the recovery walks the payload through
    _json_safe instead, which handles both the unserialisable objects and the non-finite numbers.
    Coercing beats dropping: a stringified datetime is far better for a client than a missing event.
    """
    try:
        return f"event: {event}\ndata: {json.dumps(payload, allow_nan=False)}\n\n".encode()
    except Exception:
        # 🔴 Not (TypeError, ValueError). json.dumps calls back into caller-controlled code —
        # __str__, __iter__, dict subclass __getitem__, pydantic property getters — so it can
        # surface any exception type. Narrow catches here left the constructor escape open, which
        # a probe confirmed with a __str__ raising RuntimeError.
        pass
    try:
        body = json.dumps(_json_safe(payload), allow_nan=False)
        verbose_logger.warning(
            "FakeAnthropicMessagesStreamIterator: %s frame held a value JSON cannot represent; "
            "coerced it rather than dropping the frame. The stream stays balanced but one field is "
            "now a string or null — check the provider transform that produced it.",
            event,
        )
    except Exception:
        try:
            # Last resort: keep the frame's shape so index accounting survives even if the
            # content does not. A balanced stream with a thin frame beats no stream.
            # Only reachable via a __str__ that itself raises, since _json_safe is otherwise
            # total. Keep the primitives, which is enough for the frame to parse.
            minimal = {k: v for k, v in payload.items() if isinstance(v, (str, int))}
            minimal["type"] = str(payload.get("type", event))
            body = json.dumps(minimal, allow_nan=False)
        except Exception:
            # Nothing caller-shaped survives. Emit a structurally valid frame of the right
            # event type: the caller's index accounting depends on the frame COUNT, so a
            # content-free frame is recoverable where a missing one is not.
            body = "{}"
        verbose_logger.warning(
            "FakeAnthropicMessagesStreamIterator: %s frame could not be serialised even with "
            "coercion; emitted a reduced frame to keep the stream balanced.",
            event,
        )
    return f"event: {event}\ndata: {body}\n\n".encode()


class FakeAnthropicMessagesStreamIterator:
    """
    Fake streaming iterator for Anthropic Messages responses.

    Used when we need to convert a non-streaming response to a streaming format,
    such as when WebSearch interception converts stream=True to stream=False but
    the LLM doesn't make a tool call.

    This creates a proper Anthropic-style streaming response with multiple events:
    - message_start
    - content_block_start (for each content block)
    - content_block_delta (for text content, chunked)
    - content_block_stop
    - message_delta (for usage)
    - message_stop
    """

    def __init__(self, response: AnthropicMessagesResponse):
        self.response = response
        self.chunks = self._create_streaming_chunks()
        self.current_index = 0

    def _create_content_block_chunks(self, block_dict: Dict[str, Any], index: int) -> List[bytes]:
        """Build SSE chunks for a single content block."""
        chunks = []
        block_type = block_dict.get("type")

        if block_type == "text":
            content_block_start = {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            }
            chunks.append(_sse("content_block_start", content_block_start))
            text = block_dict.get("text", "")
            content_block_delta = {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text},
            }
            chunks.append(_sse("content_block_delta", content_block_delta))

        elif block_type == "thinking":
            content_block_start = {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            }
            chunks.append(_sse("content_block_start", content_block_start))
            thinking_text = block_dict.get("thinking", "")
            if thinking_text:
                content_block_delta = {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": thinking_text},
                }
                chunks.append(_sse("content_block_delta", content_block_delta))
            signature = block_dict.get("signature", "")
            if signature:
                signature_delta = {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "signature_delta", "signature": signature},
                }
                chunks.append(_sse("content_block_delta", signature_delta))

        elif block_type == "redacted_thinking":
            content_block_start = {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "redacted_thinking"},
            }
            chunks.append(_sse("content_block_start", content_block_start))

        elif block_type == "tool_use":
            content_block_start = {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": block_dict.get("id"),
                    "name": block_dict.get("name"),
                    "input": {},
                },
            }
            chunks.append(_sse("content_block_start", content_block_start))
            # 🔴 A nested json.dumps: `partial_json` is a JSON STRING inside a JSON frame, so it
            # is serialised while the payload is being BUILT — before _sse is entered. _sse cannot
            # protect its own arguments, so this site needs the sanitiser applied directly. Found by
            # probing every branch after the helper landed; `{"input": {"when": datetime}}` still
            # escaped the constructor. Anthropic's own wire format requires the double encoding, so
            # flattening it is not an option.
            input_data = _json_safe(block_dict.get("input", {}))
            content_block_delta = {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(input_data, allow_nan=False),
                },
            }
            chunks.append(_sse("content_block_delta", content_block_delta))

        # [ARC-BUG-48] Restored from the pre-de-fork production image a2e1b93d30, where these two
        # branches had been live since fork commit be5879c8752. The de-fork carried that commit's
        # other hunks — interceptors/advisor.py, handler.py, guardrail_translation — and silently
        # dropped its footprint in THIS file (+43 lines) and in messages/transformation.py
        # (+55 lines). The commit read as "handled" while two of its thirteen files were never
        # diffed, and a ruff reformat on the clean side made this file look recently touched.
        #
        # Why the loss was client-visible: the caller loop below iterates EVERY content block with
        # enumerate and appends a content_block_stop for each index unconditionally. A block whose
        # type has no branch here therefore emits a stop with no matching start, and its id never
        # reaches the wire at all. A client applying the standard SSE reconstruction has nothing to
        # attach at that index, so when it replays the turn as history the server_tool_use block
        # comes back without a recoverable id — and Anthropic rejects it:
        #   400 messages.N.content.M.server_tool_use.id: String should match pattern
        #       '^srvtoolu_[a-zA-Z0-9_]+$'
        # Observed on prd-ai 2026-08-05; it stopped the moment the advisor was switched off,
        # because advisor.py:172 is what returns this iterator for streaming callers.
        elif block_type == "server_tool_use":
            content_block_start = {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "server_tool_use",
                    "id": block_dict.get("id"),
                    "name": block_dict.get("name"),
                },
            }
            chunks.append(_sse("content_block_start", content_block_start))

        elif block_type == "advisor_tool_result":
            content_block_start = {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "advisor_tool_result",
                    "tool_use_id": block_dict.get("tool_use_id"),
                    "content": {"type": "advisor_result", "text": ""},
                },
            }
            chunks.append(_sse("content_block_start", content_block_start))
            advisor_content = block_dict.get("content") or {}
            advisor_text = advisor_content.get("text", "") if isinstance(advisor_content, dict) else ""
            if advisor_text:
                content_block_delta = {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "advisor_result_delta", "text": advisor_text},
                }
                chunks.append(_sse("content_block_delta", content_block_delta))

        else:
            # [ARC-BUG-48] Fail OPEN on an unknown block type — the part that was missing rather
            # than merely lost. The enumerate loop guarantees a stop for every index, so an
            # unhandled type otherwise emits a stop with no start, which corrupts the caller's
            # history. Emitting a generic start makes the next unhandled type — and Anthropic
            # keeps adding them — degrade instead of breaking.
            #
            # Passed through unreshaped on purpose: guessing at a per-type shape is what produced
            # the ARC-BUG-44/45 family, where an allowlist rebuilt native blocks lossily.
            #
            # 🔴 Serialisation goes through _sse, and that is what actually makes this fail OPEN.
            # The first version called json.dumps inline and UNGUARDED, which made it fail CLOSED —
            # worse than the bug it fixes. `_create_streaming_chunks` runs in __init__, so a raise
            # here escapes the CONSTRUCTOR: the caller never receives an iterator and the client
            # gets a 500 with zero SSE bytes. That is a third shape the original reasoning did not
            # enumerate — not "a start" or "a bare stop" but "no stream at all".
            #
            # Reachable on the ALREADY-LIVE websearch path, independent of the advisor:
            # websearch_interception injects `web_search_tool_result` blocks whose `page_age` is
            # read via getattr off a model declaring extra="allow", so a provider transform
            # assigning a datetime bypasses validation. Measured: TypeError from __init__ for
            # datetime and for bytes. This iterator has six call sites, so "the advisor is off"
            # covers only one of them.
            #
            # A second review round replaced a hand-rolled degrade here that kept only `type` and
            # any string `id`. _sse coerces with str() instead, so the WHOLE block survives with
            # one field stringified — strictly more than the old path preserved, and it drops
            # neither `tool_use_id` (whose loss reproduced the original ARC-BUG-48 symptom
            # client-side) nor any field a future type introduces.
            chunks.append(
                _sse(
                    "content_block_start",
                    {"type": "content_block_start", "index": index, "content_block": block_dict},
                )
            )
            verbose_logger.warning(
                "FakeAnthropicMessagesStreamIterator: no handler for content block type %r at index %s — "
                "emitting the block in content_block_start so the stream stays well-formed. "
                "Add an explicit branch if this type needs deltas.",
                block_type,
                index,
            )

        content_block_stop = {"type": "content_block_stop", "index": index}
        chunks.append(_sse("content_block_stop", content_block_stop))
        return chunks

    def _create_streaming_chunks(self) -> List[bytes]:
        """Convert the non-streaming response to streaming chunks"""
        chunks = []

        # Cast response to dict for easier access
        response_dict = cast(Dict[str, Any], self.response)

        # 1. message_start event
        usage = response_dict.get("usage", {})
        message_start = {
            "type": "message_start",
            "message": {
                "id": response_dict.get("id"),
                "type": "message",
                "role": response_dict.get("role", "assistant"),
                "model": response_dict.get("model"),
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0) if usage else 0,
                    "output_tokens": 0,
                },
            },
        }
        chunks.append(_sse("message_start", message_start))

        # 2-4. For each content block, send start/delta/stop events
        content_blocks = response_dict.get("content", [])
        for index, block in enumerate(content_blocks):
            block_dict = cast(Dict[str, Any], block)
            chunks.extend(self._create_content_block_chunks(block_dict, index))

        # 5. message_delta event (with final usage and stop_reason)
        # Include cache usage fields so clients that only read message_delta
        # (like Claude Code's SDK) see the full input token breakdown.
        delta_usage: Dict[str, Any] = {
            "output_tokens": usage.get("output_tokens", 0) if usage else 0,
        }
        if usage:
            if usage.get("input_tokens") is not None:
                delta_usage["input_tokens"] = usage["input_tokens"]
            if usage.get("cache_creation_input_tokens") is not None:
                delta_usage["cache_creation_input_tokens"] = usage["cache_creation_input_tokens"]
            if usage.get("cache_read_input_tokens") is not None:
                delta_usage["cache_read_input_tokens"] = usage["cache_read_input_tokens"]
        message_delta = {
            "type": "message_delta",
            "delta": {
                "stop_reason": response_dict.get("stop_reason"),
                "stop_sequence": response_dict.get("stop_sequence"),
            },
            "usage": delta_usage,
        }
        chunks.append(_sse("message_delta", message_delta))

        # 6. message_stop event
        message_stop = {"type": "message_stop", "usage": usage if usage else {}}
        chunks.append(_sse("message_stop", message_stop))

        return chunks

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.current_index >= len(self.chunks):
            raise StopAsyncIteration

        chunk = self.chunks[self.current_index]
        self.current_index += 1
        return chunk

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.chunks):
            raise StopIteration

        chunk = self.chunks[self.current_index]
        self.current_index += 1
        return chunk
