"""
Fake Streaming Iterator for Anthropic Messages

This module provides a fake streaming iterator that converts non-streaming
Anthropic Messages responses into proper streaming format.

Used when WebSearch interception converts stream=True to stream=False but
the LLM doesn't make a tool call, and we need to return a stream to the user.
"""

import json
from typing import Any, Dict, List, cast

from litellm._logging import verbose_logger
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)


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
            chunks.append(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode())
            text = block_dict.get("text", "")
            content_block_delta = {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text},
            }
            chunks.append(f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n".encode())

        elif block_type == "thinking":
            content_block_start = {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            }
            chunks.append(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode())
            thinking_text = block_dict.get("thinking", "")
            if thinking_text:
                content_block_delta = {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": thinking_text},
                }
                chunks.append(f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n".encode())
            signature = block_dict.get("signature", "")
            if signature:
                signature_delta = {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "signature_delta", "signature": signature},
                }
                chunks.append(f"event: content_block_delta\ndata: {json.dumps(signature_delta)}\n\n".encode())

        elif block_type == "redacted_thinking":
            content_block_start = {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "redacted_thinking"},
            }
            chunks.append(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode())

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
            chunks.append(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode())
            input_data = block_dict.get("input", {})
            content_block_delta = {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(input_data),
                },
            }
            chunks.append(f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n".encode())

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
            chunks.append(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode())

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
            chunks.append(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode())
            advisor_content = block_dict.get("content") or {}
            advisor_text = advisor_content.get("text", "") if isinstance(advisor_content, dict) else ""
            if advisor_text:
                content_block_delta = {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "advisor_result_delta", "text": advisor_text},
                }
                chunks.append(f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n".encode())

        else:
            # [ARC-BUG-48] Fail OPEN on an unknown block type, which is the part that was missing
            # rather than merely lost. The enumerate loop guarantees a stop for every index, so the
            # only two possible shapes for an unhandled type are "a start we synthesised" or "a
            # stop with no start". The first is a block a client may not fully understand; the
            # second is a malformed stream that corrupts the caller's history. Emitting a generic
            # start makes the next unhandled type — and Anthropic keeps adding them — degrade
            # instead of breaking, so this exact defect cannot recur silently.
            #
            # Deliberately passes the block through as-is: guessing at a shape per type is what
            # produced the ARC-BUG-44/45 family, where an allowlist rebuilt native blocks lossily.
            content_block_start = {
                "type": "content_block_start",
                "index": index,
                "content_block": block_dict,
            }
            chunks.append(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode())
            verbose_logger.warning(
                "FakeAnthropicMessagesStreamIterator: no handler for content block type %r at index %s — "
                "emitting the block verbatim in content_block_start so the stream stays well-formed. "
                "Add an explicit branch if this type needs deltas.",
                block_type,
                index,
            )

        content_block_stop = {"type": "content_block_stop", "index": index}
        chunks.append(f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n".encode())
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
        chunks.append(f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode())

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
        chunks.append(f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n".encode())

        # 6. message_stop event
        message_stop = {"type": "message_stop", "usage": usage if usage else {}}
        chunks.append(f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n".encode())

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
