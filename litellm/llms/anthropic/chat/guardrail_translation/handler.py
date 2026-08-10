"""
Anthropic Message Handler for Unified Guardrails

This module provides a class-based handler for Anthropic-format messages.
The class methods can be overridden for custom behavior.

Pattern Overview:
-----------------
1. Extract text content from messages/responses (both string and list formats)
2. Create async tasks to apply guardrails to each text segment
3. Track mappings to know where each response belongs
4. Apply guardrail responses back to the original structure
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from litellm._logging import verbose_proxy_logger
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    _ANTHROPIC_ADVISOR_TOOL_PREFIX,
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
    openai_messages_without_system,
    openai_messages_without_tool,
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from litellm.types.llms.anthropic import (
    ANTHROPIC_HOSTED_TOOLS,
    AllAnthropicToolsValues,
    AnthropicMessagesRequest,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionRequest,
    ChatCompletionToolCallChunk,
    ChatCompletionToolParam,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    GenericGuardrailAPIInputs,
    ModelResponse,
)

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import (
        CustomGuardrail,
        ModifyResponseException,
    )
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.anthropic_messages.anthropic_response import (
        AnthropicMessagesResponse,
    )


# [ARC-BUG-49] Recorded in a task mapping for content that is scanned but must NEVER be
# written back — today that is extended-thinking prose, which Anthropic rejects if it comes
# back altered by a single byte.
#
# 🔴 The explicit sentinel check in both write-backs is LOAD-BEARING, not a second line of
# defence. An earlier version of this comment claimed the `type == "text"` gate made a
# mis-index harmless; an adversarial review refuted that by execution and was right. With
# `thinking` FIRST and a text block LAST, `-1` resolves to that text block, whose type IS
# "text" — so the type gate permits the write and the thinking scan's verdict overwrites
# the real answer. Removing either guard is a live corruption, not a tidy-up.
#
# ⚠️ Pinned by `test_the_thinking_verdict_cannot_reach_an_unmapped_trailing_text_block`.
# An earlier version of this comment credited
# `test_the_thinking_response_does_not_land_on_a_trailing_text_block` instead — WRONG, and
# a second review caught it: that test's trailing block has real text, so it earns its own
# mapping whose later write overwrites the corruption, and the test passes with the guard
# neutered. Verified by mutation: neutering the guard left all 13 tests in that file green.
# The killing shape is a trailing text block that earns NO mapping of its own — empty
# `text`, a missing `text` key, or `text: None`, all of which Anthropic emits in practice
# alongside `tool_use`.
#
# -1 rather than a large number because both write-backs bounds-check with
# `>= len(content)`, which a negative index passes silently.
_UNWRITABLE_CONTENT_IDX = -1


def _is_anthropic_native_tool(tool: Any) -> bool:
    """[ARC-BUG-45] Did the forward leg keep this tool in Anthropic form?

    Deliberately delegates to ``translate_anthropic_tools_to_openai``'s own predicate rather
    than restating the type list. The two legs must agree by construction: if a type is
    kept native on the way in and not recognised as native on the way out, it gets rebuilt
    from a shape it never had — which is the ARC-BUG-44 defect running in reverse.
    """
    if not isinstance(tool, dict):
        return False
    tool_type = tool.get("type", "")
    if not isinstance(tool_type, str):
        return False
    return tool_type.startswith(_ANTHROPIC_ADVISOR_TOOL_PREFIX) or any(
        tool_type.startswith(t.value) for t in ANTHROPIC_HOSTED_TOOLS
    )


class AnthropicMessagesHandler(BaseTranslation):
    """
    Handler for processing Anthropic messages with guardrails.

    This class provides methods to:
    1. Process input messages (pre-call hook)
    2. Process output responses (post-call hook)

    Methods can be overridden to customize behavior for different message formats.
    """

    def __init__(self):
        super().__init__()
        self.adapter = LiteLLMAnthropicMessagesAdapter()

    @staticmethod
    def _build_streaming_usage_response(
        responses_so_far: list[Any],
        request_data: Optional[dict],
    ) -> Optional[ModelResponse]:
        chunks = tuple(response for response in responses_so_far if isinstance(response, (str, bytes)))
        if not chunks:
            return None
        try:
            return AnthropicPassthroughLoggingHandler._build_usage_only_response_from_chunks(
                all_chunks=chunks,
                model=str((request_data or {}).get("model") or ""),
            )
        except (AttributeError, TypeError, ValueError):
            return None

    def build_block_sse_chunks(
        self,
        exc: "ModifyResponseException",
        stream_started: bool = False,
        responses_so_far: Optional[list[Any]] = None,
    ) -> list[bytes]:
        """
        Build an Anthropic SSE sequence delivering the guardrail block message
        and terminating the stream cleanly.

        - ``stream_started`` False (buffered / pre-stream): nothing has been
          sent, so emit a complete standalone message (message_start ->
          content_block_* -> message_delta -> message_stop) via
          FakeAnthropicMessagesStreamIterator, the same converter the
          /v1/messages pre-stream block handler uses.
        - ``stream_started`` True (sampling / detect-only end-of-stream): real
          chunks were already sent, so *continue* the in-progress message --
          close the open content block, append the block message as a new text
          block, then end the message. Emitting a second ``message_start`` here
          would make Anthropic clients reject the stream.
        """
        if stream_started:
            return self._block_continuation_chunks(exc, responses_so_far or [])
        return self._standalone_block_chunks(exc)

    def _standalone_block_chunks(self, exc: "ModifyResponseException") -> list[bytes]:
        import uuid

        from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
            FakeAnthropicMessagesStreamIterator,
        )
        from litellm.llms.base_llm.guardrail_translation.utils import (
            blocked_response_usage,
        )
        from litellm.types.utils import AnthropicMessagesResponse

        block_response = AnthropicMessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            type="message",
            role="assistant",
            content=[{"type": "text", "text": exc.message}],
            model=exc.model,
            stop_reason="end_turn",
            usage=blocked_response_usage(getattr(exc, "original_response", None)),
        )
        return list(FakeAnthropicMessagesStreamIterator(response=block_response))

    def _block_continuation_chunks(self, exc: "ModifyResponseException", responses_so_far: list[Any]) -> list[bytes]:
        """Continue an already-started message: close the open content block,
        append the block message as a new text block, then end the message --
        without a second message_start."""

        from litellm.llms.base_llm.guardrail_translation.utils import (
            blocked_response_usage,
        )

        def _sse(event_type: str, payload: dict) -> bytes:
            return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()

        output_tokens = blocked_response_usage(getattr(exc, "original_response", None))["output_tokens"]
        open_index, max_index = self._content_block_state(responses_so_far)
        new_index = (max_index + 1) if max_index is not None else 0
        chunks: list[bytes] = []
        if open_index is not None:
            chunks.append(_sse("content_block_stop", {"type": "content_block_stop", "index": open_index}))
        chunks += [
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": new_index,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": new_index,
                    "delta": {"type": "text_delta", "text": exc.message},
                },
            ),
            _sse("content_block_stop", {"type": "content_block_stop", "index": new_index}),
            _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                },
            ),
            _sse("message_stop", {"type": "message_stop"}),
        ]
        return chunks

    @staticmethod
    def _content_block_state(
        responses_so_far: list[Any],
    ) -> tuple[Optional[int], Optional[int]]:
        """From the SSE chunks already sent to the client, return (open
        content-block index or None, highest content-block index seen or None).

        A single streamed item may bundle multiple SSE events (raw bytes) or be
        an already-parsed event dict, so every event across every item is
        considered -- matching how ``get_streaming_string_so_far`` reads the
        same stream."""
        open_indices: set[int] = set()
        max_index: Optional[int] = None
        for item in responses_so_far:
            for data in AnthropicMessagesHandler._iter_sse_events(item):
                event_type = data.get("type")
                index = data.get("index")
                if not isinstance(index, int):
                    continue
                if event_type == "content_block_start":
                    open_indices.add(index)
                    max_index = index if max_index is None else max(max_index, index)
                elif event_type == "content_block_stop":
                    open_indices.discard(index)
        open_index = max(open_indices) if open_indices else None
        return open_index, max_index

    @staticmethod
    def _iter_sse_events(item: Any) -> list[dict]:
        """Yield the event-data dicts in one stream chunk.

        Handles both formats this stream can carry (see
        ``get_streaming_string_so_far``): raw SSE ``bytes`` -- which may bundle
        several events separated by a blank line -- and an already-parsed event
        ``dict``."""
        if isinstance(item, dict):
            return [item]
        if not isinstance(item, (bytes, bytearray)):
            return []
        events: list[dict] = []
        for block in item.decode("utf-8", errors="replace").split("\n\n"):
            for line in block.split("\n"):
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                try:
                    parsed = json.loads(line[len("data:") :].strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    events.append(parsed)
        return events

    def _translate_to_openai(self, data: dict) -> ChatCompletionRequest:
        """Translate Anthropic request to OpenAI chat completion format."""
        chat_completion_compatible_request, _ = self._translate_to_openai_with_tool_names(data)
        return chat_completion_compatible_request

    @staticmethod
    def _translate_to_openai_with_tool_names(
        data: dict,
    ) -> tuple[ChatCompletionRequest, Mapping[str, str]]:
        """As ``_translate_to_openai``, but also returns ``{truncated: original}``.

        [ARC-BUG-51] Discarding this mapping was itself a defect.
        ``translate_anthropic_tools_to_openai`` rewrites any tool name longer than 64
        characters to ``<55-char prefix>_<8-char hash>``, so the translated tool answers
        to a name the caller never sent. Anything comparing the two sides by name needs
        this to undo the rewrite.
        """
        (
            chat_completion_compatible_request,
            tool_name_mapping,
        ) = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
            anthropic_message_request=cast(AnthropicMessagesRequest, data.copy())
        )
        return chat_completion_compatible_request, dict(tool_name_mapping or {})

    @staticmethod
    def _tool_identity(tool: object, tool_name_mapping: Mapping[str, str]) -> str | None:
        """The identity a tool keeps on both sides of the OpenAI translation.

        A regular tool arrives as ``{"name": ...}`` and comes back wrapped as
        ``{"function": {"name": ...}}``; a hosted tool the adapter keeps native still
        answers to its own ``name``, falling back to ``type`` for the shapes that carry
        no name at all.

        ⚠️ ``name`` before ``type``: Anthropic's own shape for a caller-defined tool is
        ``{"type": "custom", "name": ...}``, so keying on ``type`` collapses every such
        tool to the single identity ``"custom"``.

        ⚠️ A translated name is resolved back through *tool_name_mapping* FIRST. Names
        over 64 characters are rewritten by the translation, so without the reverse
        lookup the two sides can never match, the tool is misread as diverted, and it is
        re-attached — duplicating it on the approve path and RESURRECTING it after a
        guardrail had explicitly removed it.

        Returns ``None`` only for a tool with no identifiable name or type. Callers must
        treat that as "cannot be matched", never as "matched nothing".

        Takes ``object`` rather than a tool type: both sides of this comparison are
        untyped JSON off the wire, and narrowing happens by the ``isinstance`` below.
        """
        if not isinstance(tool, dict):
            return None
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name"):
            translated = str(function["name"])
            return tool_name_mapping.get(translated, translated)
        return str(tool.get("name") or tool.get("type") or "") or None

    @classmethod
    def _diverted_tools(
        cls,
        request_tools: Sequence[object],
        tools_to_check: Sequence[ChatCompletionToolParam],
        tool_name_mapping: Mapping[str, str],
    ) -> tuple[object, ...]:
        """Caller tools that the OpenAI translation did not carry into ``tools``.

        [ARC-BUG-51] Derived by DIFFERENCE against what the translation actually
        produced, rather than by restating which types get diverted. A hard-coded
        list would silently stop covering a type the day the adapter diverts a new
        one — the same drift that made ARC-BUG-44 and -45 two separate bugs on one
        seam. The translation is the authority on what it kept; anything it dropped
        is by definition unreviewable and must be preserved verbatim.

        ⚠️ Nativeness and divertedness are ORTHOGONAL — do not filter on
        ``_is_anthropic_native_tool`` here. ``web_search`` is BOTH native and diverted:
        the writeback loop's native branch only round-trips tools the guardrail
        returned, and a diverted tool was never in that list to begin with. Skipping
        native tools here therefore collected nothing and left the drop in place.

        ⚠️ A tool with NO identity is not diverted either. ``kept`` can never contain
        ``None``, so treating an unidentifiable tool as "not in kept" re-attached it
        unconditionally — duplicating it, and resurrecting it even after a guardrail had
        removed it. Such a tool is left to the writeback loop, the only path that can
        honour what the guardrail decided about it.
        """
        kept = frozenset(
            identity
            for identity in (cls._tool_identity(tool, tool_name_mapping) for tool in tools_to_check or ())
            if identity
        )
        return tuple(
            tool
            for tool in request_tools or ()
            for identity in [cls._tool_identity(tool, tool_name_mapping)]
            if identity is not None and identity not in kept
        )

    def get_structured_messages(self, data: dict) -> Optional[List[AllMessageValues]]:
        """
        Convert Anthropic messages request data to OpenAI-spec structured messages.

        Uses the Anthropic-to-OpenAI adapter to translate message format.
        """
        messages = data.get("messages")
        if messages is None:
            return None
        chat_completion_compatible_request = self._translate_to_openai(data)
        result = cast(
            List[AllMessageValues],
            chat_completion_compatible_request.get("messages", []),
        )
        return result if result else None

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
    ) -> Any:
        """
        Process input messages by applying guardrails to text content.
        """
        messages = data.get("messages")
        if messages is None:
            return data

        skip_system = effective_skip_system_message_for_guardrail(guardrail_to_apply)
        skip_tool = effective_skip_tool_message_for_guardrail(guardrail_to_apply)

        chat_completion_compatible_request, tool_name_mapping = self._translate_to_openai_with_tool_names(data)

        structured_messages = cast(
            List[AllMessageValues],
            chat_completion_compatible_request.get("messages", []),
        )
        if skip_system:
            structured_messages = openai_messages_without_system(structured_messages)
        if skip_tool:
            structured_messages = openai_messages_without_tool(structured_messages)

        texts_to_check: List[str] = []
        images_to_check: List[str] = []
        tools_to_check: List[ChatCompletionToolParam] = chat_completion_compatible_request.get("tools", [])
        task_mappings: List[Tuple[int, Optional[int]]] = []

        # [ARC-BUG-51] Tools the OpenAI translation DIVERTS never reach the guardrail, so
        # they must not be sourced from its reply. `_translate_to_openai` moves Anthropic
        # hosted tools that have no OpenAI tool-shape out of `tools` and into a parameter
        # instead — `web_search` becomes `web_search_options` — so `tools_to_check` is a
        # strict subset of what the caller sent. Rebuilding `data["tools"]` from the
        # guardrail's reply therefore DELETED them from the outbound request, silently:
        # two web_search tools in, one out. ARC-BUG-45 closed the symmetric half of this
        # seam (re-mapping a native tool a second time) and left this half open with a note.
        #
        # Keep the diverted originals aside and re-attach them after the writeback. They
        # are unreviewed by construction, which is not a new gap: they were unreviewed
        # before this fix too — the difference is that they now survive.
        diverted_tools = self._diverted_tools(data.get("tools") or (), tools_to_check, tool_name_mapping)

        # Step 1: Extract all text content and images
        for msg_idx, message in enumerate(messages):
            self._extract_input_text_and_images(
                message=message,
                msg_idx=msg_idx,
                texts_to_check=texts_to_check,
                images_to_check=images_to_check,
                task_mappings=task_mappings,
                skip_system_message=skip_system,
                skip_tool_message=skip_tool,
            )

        # Step 2: Apply guardrail to all texts in batch
        if texts_to_check:
            inputs = GenericGuardrailAPIInputs(texts=texts_to_check)
            if images_to_check:
                inputs["images"] = images_to_check
            if tools_to_check:
                inputs["tools"] = tools_to_check
            original_structured_messages = structured_messages
            if structured_messages:
                inputs["structured_messages"] = structured_messages
            # Include model information if available
            model = data.get("model")
            if model:
                inputs["model"] = model
            guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=data,
                input_type="request",
                logging_obj=litellm_logging_obj,
            )

            guardrailed_texts = guardrailed_inputs.get("texts", [])
            guardrailed_tools = guardrailed_inputs.get("tools")
            if guardrailed_tools is not None:
                # Convert tools back from OpenAI format to Anthropic format
                anthropic_config = AnthropicConfig()
                anthropic_tools: List[AllAnthropicToolsValues] = []
                for tool in guardrailed_tools:
                    # [ARC-BUG-45] A tool the forward leg kept Anthropic-native is already in
                    # Anthropic form — do not send it through the OpenAI-to-Anthropic mapping
                    # a second time. This is the symmetric half of ARC-BUG-44: that fix stopped
                    # `advisor_20260301` being flattened on the way IN, and this stops the way
                    # OUT from rebuilding it from a shape it never had.
                    #
                    # Re-mapping a native tool is not merely redundant, it is lossy and can
                    # fail. `_map_tool_helper` reconstructs each tool through a per-type
                    # allowlist, so any key that type does not enumerate is dropped, and its
                    # advisor branch *validates* — raising `ValueError` where the generic
                    # `custom` branch it used to reach did not. That raise is unreachable by
                    # either of the guardrail's safety valves: `monitor_mode` and
                    # `block_failures` are both checked inside `apply_guardrail`, which has
                    # already returned by the time this loop runs. So it propagated out of
                    # `pre_call_hook` uncaught and, carrying no `status_code`, surfaced as a
                    # **500** on what was really bad caller input — and took the request's
                    # other, valid tools with it.
                    #
                    # Observed on dev-ai minutes after the ARC-BUG-44 rollout: two requests
                    # returned 500 "Advisor tool must have a valid model" plus High-severity
                    # alerts. Skipping the round-trip is preferred over catching the error,
                    # because catching would leave a valid tool silently stripped of the keys
                    # this mapping does not carry.
                    if _is_anthropic_native_tool(tool):
                        # No cast: the loop variable is already Any (the guardrail returns untyped
                        # JSON), so appending is accepted without one. The original cast was pure
                        # noise that cost a LIT006 budget slot — and the gate counts casts rather
                        # than reading a `# cast-ok:` reason, despite its own error text offering one.
                        # The runtime check on the line above is what actually establishes the type.
                        anthropic_tools.append(tool)
                        continue
                    converted_tool, mcp_server = anthropic_config._map_tool_helper(tool)
                    if converted_tool is not None:
                        anthropic_tools.append(converted_tool)
                    # Note: MCP servers are handled separately in the main transformation
                # [ARC-BUG-51] Re-attach the tools the translation diverted (see above).
                # Appended last so a guardrail edit to a reviewed tool still wins.
                anthropic_tools.extend(diverted_tools)
                data["tools"] = anthropic_tools

            guardrailed_structured_messages = guardrailed_inputs.get("structured_messages")
            if (
                guardrailed_structured_messages is not None
                and guardrailed_structured_messages is not original_structured_messages
            ):
                self._write_back_structured_messages(data, guardrailed_structured_messages)
            else:
                # Step 3: Map guardrail responses back to original message structure
                await self._apply_guardrail_responses_to_input(
                    messages=messages,
                    responses=guardrailed_texts,
                    task_mappings=task_mappings,
                )

        verbose_proxy_logger.debug("Anthropic Messages: Processed input messages: %s", messages)

        return data

    @staticmethod
    def _write_back_structured_messages(data: dict, structured_messages: list) -> None:
        """Convert compressed structured_messages back to Anthropic format and write to data."""
        from litellm.litellm_core_utils.prompt_templates.factory import (
            anthropic_messages_pt,
        )

        model = str(data.get("model") or "")
        non_system = [m for m in structured_messages if m.get("role") != "system"]
        converted = anthropic_messages_pt(messages=non_system, model=model, llm_provider="anthropic")
        for msg in converted:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "thinking":
                        block.pop("cache_control", None)
        data["messages"] = converted

    def extract_request_tool_names(self, data: dict) -> List[str]:
        """Extract tool names from Anthropic messages request (tools[].name)."""
        names: List[str] = []
        for tool in data.get("tools") or []:
            if isinstance(tool, dict) and tool.get("name"):
                names.append(str(tool["name"]))
        return names

    def _extract_input_text_and_images(
        self,
        message: Dict[str, Any],
        msg_idx: int,
        texts_to_check: List[str],
        images_to_check: List[str],
        task_mappings: List[Tuple[int, Optional[int]]],
        skip_system_message: bool = False,
        skip_tool_message: bool = False,
    ) -> None:
        """
        Extract text content and images from a message.

        Override this method to customize text/image extraction logic.
        """
        role = str(message.get("role") or "").lower()
        if skip_system_message and role == "system":
            return
        if skip_tool_message and role == "tool":
            return

        content = message.get("content", None)
        tools = message.get("tools", None)
        if content is None and tools is None:
            return

        ## CHECK FOR TEXT + IMAGES
        if content is not None and isinstance(content, str):
            # Simple string content
            texts_to_check.append(content)
            task_mappings.append((msg_idx, None))

        elif content is not None and isinstance(content, list):
            # List content (e.g., multimodal with text and images)
            for content_idx, content_item in enumerate(content):
                # Extract text
                text_str = content_item.get("text", None)
                if text_str is not None:
                    texts_to_check.append(text_str)
                    task_mappings.append((msg_idx, int(content_idx)))

                # Extract images
                if content_item.get("type") == "image":
                    source = content_item.get("source", {})
                    if isinstance(source, dict):
                        # Could be base64 or url
                        data = source.get("data")
                        if data:
                            images_to_check.append(data)

    def _extract_input_tools(
        self,
        tools: List[Dict[str, Any]],
        tools_to_check: List[ChatCompletionToolParam],
    ) -> None:
        """
        Extract tools from a message.
        """
        ## CHECK FOR TOOLS
        if tools is not None and isinstance(tools, list):
            # TRANSFORM ANTHROPIC TOOLS TO OPENAI TOOLS
            openai_tools = self.adapter.translate_anthropic_tools_to_openai(
                tools=cast(List[AllAnthropicToolsValues], tools)
            )
            tools_to_check.extend(openai_tools)  # type: ignore

    async def _apply_guardrail_responses_to_input(
        self,
        messages: List[Dict[str, Any]],
        responses: List[str],
        task_mappings: List[Tuple[int, Optional[int]]],
    ) -> None:
        """
        Apply guardrail responses back to input messages.

        Override this method to customize how responses are applied.
        """
        for task_idx, guardrail_response in enumerate(responses):
            mapping = task_mappings[task_idx]
            msg_idx = cast(int, mapping[0])
            content_idx_optional = cast(Optional[int], mapping[1])

            content = messages[msg_idx].get("content", None)
            if content is None:
                continue

            if isinstance(content, str) and content_idx_optional is None:
                # Replace string content with guardrail response
                messages[msg_idx]["content"] = guardrail_response

            elif isinstance(content, list) and content_idx_optional is not None:
                # Replace specific text item in list content
                messages[msg_idx]["content"][content_idx_optional]["text"] = guardrail_response

    async def process_output_response(
        self,
        response: "AnthropicMessagesResponse",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
        request_data: Optional[dict] = None,
    ) -> Any:
        """
        Process output response by applying guardrails to text content and tool calls.

        Args:
            response: Anthropic MessagesResponse object
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata to pass to guardrails

        Returns:
            Modified response with guardrail applied to content

        Response Format Support:
            - List content: response.content = [
                {"type": "text", "text": "text here"},
                {"type": "tool_use", "id": "...", "name": "...", "input": {...}},
                ...
            ]
        """
        texts_to_check: List[str] = []
        images_to_check: List[str] = []
        tool_calls_to_check: List[ChatCompletionToolCallChunk] = []
        task_mappings: List[Tuple[int, Optional[int]]] = []

        response_content = self._get_response_content(response)
        if not response_content:
            return response

        # Step 1: Extract all text content and tool calls from response
        self._extract_from_content_blocks(
            response_content,
            texts_to_check,
            images_to_check,
            task_mappings,
            tool_calls_to_check,
        )

        # Step 2: Apply guardrail to all texts in batch
        if texts_to_check or tool_calls_to_check:
            request_data = self._prepare_request_data(
                request_data,
                response,
                user_api_key_dict,
                key="response",
            )

            inputs = self._build_guardrail_inputs(
                texts_to_check,
                images_to_check,
                tool_calls_to_check,
                response,
            )

            guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="response",
                logging_obj=litellm_logging_obj,
            )

            guardrailed_texts = guardrailed_inputs.get("texts", [])

            # Step 3: Map guardrail responses back to original response structure
            await self._apply_guardrail_responses_to_output(
                response=response,
                responses=guardrailed_texts,
                task_mappings=task_mappings,
            )

        verbose_proxy_logger.debug("Anthropic Messages: Processed output response: %s", response)

        return response

    async def process_output_streaming_response(
        self,
        responses_so_far: List[Any],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
        request_data: Optional[dict] = None,
    ) -> List[Any]:
        """
        Process output streaming response by applying guardrails to text content.

        Get the string so far, check the apply guardrail to the string so far, and return the list of responses so far.
        """
        from litellm.integrations.custom_guardrail import ModifyResponseException

        has_ended = self._check_streaming_has_ended(responses_so_far)

        # [ARC-BUG-50] Make the skip AUDIBLE. When this returns False the output scan does
        # not run at all, and on prd-ai that is the only output scan a streaming request
        # gets (`streaming_end_of_stream_only: true` plus `noma-during-call` off). Before
        # this, the sole trace was a WARNING about one unparseable frame — nothing said
        # "and therefore nothing was scanned", so the failure was invisible in both logs
        # and metrics. `.error()` deliberately, not `.info()`: prd-ai runs
        # `LITELLM_LOG=ERROR`, under which an info line does not exist to grep for.
        #
        # ⚠️ Rate-limited to ONE line per request, because this method is called once per
        # SAMPLED CHUNK on one of its two call paths — `unified_guardrail.py:988` invokes it
        # from the iterator hook with a growing `responses_so_far`. Measured by an
        # adversarial review: an 805-chunk stream emitted 803 ERROR lines for a single
        # request, scaling linearly with response length. That is a log flood presented as
        # observability, and it would bury the very signal this line exists to surface.
        #
        # The other path (`:1069`) fires ONCE with the whole chunk list, and on the current
        # config it is the only one that runs — `noma-post-call` is the sole hook passing
        # the iterator's `post_call` gate and it sets `streaming_end_of_stream_only: true`.
        # So the de-duplication must NOT be keyed on the chunk count: a
        # `len(responses_so_far) == 1` gate looks like "first call" but would silence
        # exactly the end-of-stream path that matters, on exactly the config we run. That
        # was the first attempt here; it is recorded because it reads as obviously correct.
        #
        # Keyed on `request_data` instead, which is the same dict object for every call in
        # one request — this class is re-instantiated per call
        # (`endpoint_guardrail_translation_mappings[...]()`), so instance state cannot
        # carry. When `request_data` is None there is nothing to key on, so it logs: a
        # duplicate line is a smaller failure than a silent unscanned response.
        skip_marker = "_arc_bug_50_scan_skip_logged"
        already_logged = isinstance(request_data, dict) and request_data.get(skip_marker)
        if not has_ended and responses_so_far and not already_logged:
            if isinstance(request_data, dict):
                request_data[skip_marker] = True
            verbose_proxy_logger.error(
                "Guardrail output scan SKIPPED: no end-of-stream signal found in %d "
                "chunk(s) for guardrail=%s. Neither a message_delta with a non-null "
                "stop_reason nor a message_stop was parsed, so the response was returned "
                "unscanned. Logged once per request — one call path runs per sampled chunk.",
                len(responses_so_far),
                getattr(guardrail_to_apply, "guardrail_name", "unknown"),
            )

        if has_ended:
            # build the model response from the responses_so_far
            built_response = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
                all_chunks=responses_so_far,
                litellm_logging_obj=cast("LiteLLMLoggingObj", litellm_logging_obj),
                model="",
            )

            # Check if model_response is valid and has choices before accessing
            if built_response is not None and hasattr(built_response, "choices") and built_response.choices:
                model_response = cast(ModelResponse, built_response)
                first_choice = cast(Choices, model_response.choices[0])
                tool_calls_list = cast(
                    Optional[List[ChatCompletionMessageToolCall]],
                    first_choice.message.tool_calls,
                )
                string_so_far = first_choice.message.content
                guardrail_inputs = GenericGuardrailAPIInputs()
                if string_so_far:
                    guardrail_inputs["texts"] = [string_so_far]
                if tool_calls_list:
                    guardrail_inputs["tool_calls"] = tool_calls_list

                try:
                    _guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                        inputs=guardrail_inputs,
                        request_data=request_data if request_data is not None else {},
                        input_type="response",
                        logging_obj=litellm_logging_obj,
                    )
                except ModifyResponseException as e:
                    if e.original_response is None:
                        e.original_response = built_response or self._build_streaming_usage_response(
                            responses_so_far, request_data
                        )
                    raise
            else:
                verbose_proxy_logger.debug("Skipping output guardrail - model response has no choices")
            return responses_so_far

        string_so_far = self.get_streaming_string_so_far(responses_so_far)
        try:
            _guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs={"texts": [string_so_far]},
                request_data=request_data if request_data is not None else {},
                input_type="response",
                logging_obj=litellm_logging_obj,
            )
        except ModifyResponseException as e:
            if e.original_response is None:
                e.original_response = self._build_streaming_usage_response(responses_so_far, request_data)
            raise
        return responses_so_far

    def _prepare_request_data(
        self,
        request_data: Optional[dict],
        response: Any,
        user_api_key_dict: Optional[Any],
        key: str,
    ) -> dict:
        """Ensure request_data has the response/responses_so_far key and metadata."""
        if request_data is None:
            request_data = {key: response}
        else:
            if key not in request_data:
                request_data[key] = response

        if "litellm_metadata" not in request_data:
            user_metadata = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
            if user_metadata:
                request_data["litellm_metadata"] = user_metadata
        return request_data

    @staticmethod
    def _get_response_content(response: Any) -> List[Any]:
        """Extract content list from a dict or object response."""
        if isinstance(response, dict):
            return response.get("content", []) or []
        elif hasattr(response, "content"):
            return getattr(response, "content", None) or []
        return []

    def _extract_from_content_blocks(
        self,
        response_content: List[Any],
        texts_to_check: List[str],
        images_to_check: List[str],
        task_mappings: List[Tuple[int, Optional[int]]],
        tool_calls_to_check: List["ChatCompletionToolCallChunk"],
    ) -> None:
        """Extract text, images, and tool calls from content blocks."""
        for content_idx, content_block in enumerate(response_content):
            block_dict: Dict[str, Any] = {}
            if isinstance(content_block, dict):
                block_type = content_block.get("type")
                block_dict = cast(Dict[str, Any], content_block)
            elif hasattr(content_block, "type"):
                block_type = getattr(content_block, "type", None)
                if hasattr(content_block, "model_dump"):
                    block_dict = content_block.model_dump()
                else:
                    block_dict = {
                        "type": block_type,
                        "text": getattr(content_block, "text", None),
                    }
            else:
                continue

            # [ARC-BUG-49] `thinking` joins the dispatch list. Without it the block never
            # reaches `_extract_output_text_and_images`, so its prose was never scanned —
            # the coverage hole this fixes. `redacted_thinking` stays out: its `data` is an
            # opaque server-encrypted blob, not prose, so a scan of it is pure noise.
            if block_type in ["text", "tool_use", "thinking"]:
                self._extract_output_text_and_images(
                    content_block=block_dict,
                    content_idx=content_idx,
                    texts_to_check=texts_to_check,
                    images_to_check=images_to_check,
                    task_mappings=task_mappings,
                    tool_calls_to_check=tool_calls_to_check,
                )

    @staticmethod
    def _build_guardrail_inputs(
        texts_to_check: List[str],
        images_to_check: List[str],
        tool_calls_to_check: List["ChatCompletionToolCallChunk"],
        response: Any,
    ) -> "GenericGuardrailAPIInputs":
        """Build GenericGuardrailAPIInputs with optional images, tool calls, model."""
        inputs = GenericGuardrailAPIInputs(texts=texts_to_check)
        if images_to_check:
            inputs["images"] = images_to_check
        if tool_calls_to_check:
            inputs["tool_calls"] = tool_calls_to_check
        response_model = None
        if isinstance(response, dict):
            response_model = response.get("model")
        elif hasattr(response, "model"):
            response_model = getattr(response, "model", None)
        if response_model:
            inputs["model"] = response_model
        return inputs

    def get_streaming_string_so_far(self, responses_so_far: List[Any]) -> str:
        """
        Parse streaming responses and extract accumulated text content.

        Handles two formats:
        1. Raw bytes in SSE (Server-Sent Events) format from Anthropic API
        2. Parsed dict objects (for backwards compatibility)

        SSE format example:
            b'event: content_block_delta\\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" curious"}}\\n\\n'

        Dict format example:
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": " curious"
                }
            }
        """
        # [ARC-BUG-50] Join the byte chunks before parsing, for the same reason as
        # `_check_streaming_has_ended`: this used to call `_extract_text_from_sse` once
        # PER CHUNK, so a `content_block_delta` split across a transport boundary failed
        # `json.loads` and its text was silently dropped from the scan payload. The
        # guardrail then scored a response with a hole in it — worse than not scanning,
        # because the result looks authoritative.
        #
        # ⚠️ Byte runs are joined and decoded per CONTIGUOUS RUN, not across the whole
        # list, so source order survives when both formats are interleaved. Joining every
        # byte chunk up front and then appending the dicts reorders the text: an
        # adversarial review measured `[dict"A ", bytes"B ", dict"C"]` coming out as
        # "B A C". Same characters, scrambled prose — and a guardrail that scores
        # reordered text returns a verdict about something the user never received.
        # Inert on today's live path, which appends one format only, but the docstring
        # above advertises both, so the next caller would inherit the bug.
        segments: list[str] = []
        byte_run: list[Any] = []

        def _flush_byte_run() -> None:
            if byte_run:
                segments.append(self._extract_text_from_sse(self._join_sse_chunks(byte_run).encode("utf-8")))
                byte_run.clear()

        for response in responses_so_far:
            # Handle already-parsed dict format
            if isinstance(response, dict):
                _flush_byte_run()
                delta = response.get("delta") if response.get("delta") else None
                if delta and delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        segments.append(text)
                continue
            byte_run.append(response)
        _flush_byte_run()

        return "".join(segments)

    def _extract_text_from_sse(self, sse_bytes: bytes) -> str:
        """
        Extract text content from Server-Sent Events (SSE) format.

        Args:
            sse_bytes: Raw bytes in SSE format

        Returns:
            Accumulated text from all content_block_delta events
        """
        text = ""
        try:
            # Decode bytes to string
            sse_string = sse_bytes.decode("utf-8")

            # Split by double newline to get individual events
            events = sse_string.split("\n\n")

            for event in events:
                if not event.strip():
                    continue

                # Parse event lines
                lines = event.strip().split("\n")
                event_type = None
                data_line = None

                for line in lines:
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_line = line[5:].strip()

                # Only process content_block_delta events
                if event_type == "content_block_delta" and data_line:
                    try:
                        data = json.loads(data_line)
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text += delta.get("text", "")
                    except json.JSONDecodeError:
                        verbose_proxy_logger.warning(f"Failed to parse JSON from SSE data: {data_line}")

        except Exception as e:
            verbose_proxy_logger.error(f"Error extracting text from SSE: {e}")

        return text

    def _check_streaming_has_ended(self, responses_so_far: List[Any]) -> bool:
        """
        Check if streaming response has ended by looking for non-null stop_reason.

        Handles two formats:
        1. Raw bytes in SSE (Server-Sent Events) format from Anthropic API
        2. Parsed dict objects (for backwards compatibility)

        SSE format example:
            b'event: message_delta\\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},...}\\n\\n'

        Dict format example:
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "tool_use",
                    "stop_sequence": null
                }
            }

        Returns:
            True if stop_reason is set to a non-null value, indicating stream has ended
        """
        # [ARC-BUG-50] Join the byte chunks BEFORE splitting into events.
        #
        # This used to decode → split → json.loads per raw transport chunk, with no
        # buffering across them. A `message_delta` split across a chunk boundary — which
        # the network is free to do anywhere — failed `json.loads`, logged a warning, and
        # was skipped. Since a non-null `stop_reason` on `message_delta` was the ONLY
        # accepted end-of-stream signal, one split frame made this return False for the
        # whole stream, and `process_output_streaming_response` then never reached
        # `apply_guardrail`: the output scan was skipped with no error and no metric.
        #
        # ⚠️ Consequential on prd-ai specifically, because `noma-post-call` runs
        # `streaming_end_of_stream_only: true` and `noma-during-call` is `default_on:
        # false` (config-decisions.md C1) — so a streaming request gets exactly ONE
        # output scan, at end of stream. That single scan is what a split frame removed,
        # and every guardrail is monitor_mode / block_failures:false, so it failed
        # silently by design.
        #
        # Attribution: `git blame` puts the unbuffered loop on upstream PR #17619, not on
        # an Arcadia commit.
        joined_sse = self._join_sse_chunks(responses_so_far)
        if joined_sse and self._sse_signals_end_of_stream(joined_sse):
            return True

        # Handle already-parsed dict format
        for response in responses_so_far:
            if isinstance(response, dict):
                response_type = response.get("type")
                if response_type == "message_delta":
                    delta = response.get("delta", {})
                    if delta.get("stop_reason") is not None:
                        return True
                # [ARC-BUG-50] `message_stop` is a second, independent end-of-stream
                # signal — see `_sse_signals_end_of_stream`.
                elif response_type == "message_stop":
                    return True

        return False

    @staticmethod
    def _join_sse_chunks(responses_so_far: Sequence[Any]) -> str:
        """Concatenate the raw byte chunks into one SSE string.

        [ARC-BUG-50] Decoded with ``errors="ignore"`` deliberately: a multi-byte UTF-8
        character can itself straddle a chunk boundary, and raising there would recreate
        the very failure this fix removes. The bytes are joined FIRST and decoded once, so
        a split character is whole by the time it is decoded — the ignore is a backstop for
        a genuinely truncated tail, not the normal path.
        """
        chunks = [chunk for chunk in responses_so_far if isinstance(chunk, bytes)]
        if not chunks:
            return ""
        try:
            return b"".join(chunks).decode("utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - defensive
            verbose_proxy_logger.error(f"Error joining SSE chunks: {exc}")
            return ""

    @staticmethod
    def _sse_signals_end_of_stream(joined_sse: str) -> bool:
        """Does this SSE text carry an end-of-stream signal?

        [ARC-BUG-50] TWO signals are accepted, not one:

        * ``message_delta`` with a non-null ``stop_reason`` — the original check;
        * ``message_stop`` — which Anthropic always sends last.

        Accepting only the first made a single unparseable frame suppress the whole
        stream's output scan. With the fallback, a malformed ``message_delta`` costs the
        `stop_reason` detail but no longer costs the scan.

        A parse failure is logged at WARNING and skipped rather than raised: this runs on
        provider output, and a raise here would fail the request instead of the check.
        """
        for event in joined_sse.split("\n\n"):
            if not event.strip():
                continue

            event_type = None
            data_line = None
            for line in event.strip().split("\n"):
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_line = line[5:].strip()

            if event_type == "message_stop":
                return True

            if event_type == "message_delta" and data_line:
                try:
                    delta = json.loads(data_line).get("delta", {})
                except json.JSONDecodeError:
                    # Still reachable: a truncated FINAL frame has no later bytes to be
                    # joined with. It no longer suppresses the scan, because message_stop
                    # is accepted above and the caller falls back to it.
                    verbose_proxy_logger.warning(f"Failed to parse JSON from SSE data: {data_line}")
                    continue
                if isinstance(delta, dict) and delta.get("stop_reason") is not None:
                    return True

        return False

    def _has_text_content(self, response: "AnthropicMessagesResponse") -> bool:
        """
        Check if response has any text content to process.

        Override this method to customize text content detection.
        """
        if isinstance(response, dict):
            response_content = response.get("content", [])
        else:
            response_content = getattr(response, "content", None) or []

        if not response_content:
            return False
        for content_block in response_content:
            # Check if this is a text block by checking the 'type' field
            if isinstance(content_block, dict) and content_block.get("type") == "text":
                content_text = content_block.get("text")
                if content_text and isinstance(content_text, str):
                    return True
        return False

    def _extract_output_text_and_images(
        self,
        content_block: Dict[str, Any],
        content_idx: int,
        texts_to_check: List[str],
        images_to_check: List[str],
        task_mappings: List[Tuple[int, Optional[int]]],
        tool_calls_to_check: Optional[List[ChatCompletionToolCallChunk]] = None,
    ) -> None:
        """
        Extract text content, images, and tool calls from a response content block.

        Override this method to customize text/image/tool extraction logic.
        """
        content_type = content_block.get("type")

        # Extract text content
        if content_type == "text":
            content_text = content_block.get("text")
            if content_text and isinstance(content_text, str):
                # Simple string content
                texts_to_check.append(content_text)
                task_mappings.append((content_idx, None))

        # [ARC-BUG-49] Extended-thinking content is scanned, READ-ONLY.
        #
        # `thinking` carries its prose under a `thinking` key, not `text`, so it was
        # invisible to the branch above and reached the model uninspected — a real
        # coverage hole, and one the caller can steer, since the model's reasoning is
        # shaped by the prompt.
        #
        # ⚠️ The mapping records the SENTINEL index, not the block's real position:
        # Anthropic rejects a request whose `thinking` or `redacted_thinking` block
        # differs by a single byte from the original response ("blocks in the latest
        # assistant message cannot be modified"), and that 400 is the highest-volume
        # client-visible error on prd-ai. So this must never become writable.
        #
        # 🔴 The sentinel is the ONLY thing that makes it unwritable. The `type == "text"`
        # gate in the write-backs does NOT independently save it — with `thinking` first
        # and a text block last, `-1` resolves to that text block, which passes the type
        # gate, and the thinking scan's verdict lands on the real answer. Measured, and
        # covered by `test_the_thinking_response_does_not_land_on_a_trailing_text_block`.
        #
        # `redacted_thinking` is deliberately NOT extracted: its `data` is an opaque
        # server-encrypted blob, not prose, so scanning it yields noise and the
        # guardrail has nothing to act on.
        elif content_type == "thinking":
            thinking_text = content_block.get("thinking")
            if thinking_text and isinstance(thinking_text, str):
                texts_to_check.append(thinking_text)
                task_mappings.append((_UNWRITABLE_CONTENT_IDX, None))

        # Extract tool calls
        elif content_type == "tool_use":
            tool_call = AnthropicConfig.convert_tool_use_to_openai_format(
                anthropic_tool_content=content_block,
                index=content_idx,
            )
            if tool_calls_to_check is None:
                tool_calls_to_check = []
            tool_calls_to_check.append(tool_call)

    async def _apply_guardrail_responses_to_output(
        self,
        response: "AnthropicMessagesResponse",
        responses: List[str],
        task_mappings: List[Tuple[int, Optional[int]]],
    ) -> None:
        """
        Apply guardrail responses back to output response.

        Override this method to customize how responses are applied.
        """
        for task_idx, guardrail_response in enumerate(responses):
            mapping = task_mappings[task_idx]
            content_idx = cast(int, mapping[0])

            # Handle both dict and object responses
            response_content: List[Any] = []
            if isinstance(response, dict):
                response_content = response.get("content", []) or []
            elif hasattr(response, "content"):
                content = getattr(response, "content", None)
                response_content = content or []
            else:
                continue

            if not response_content:
                continue

            # [ARC-BUG-49] Scanned-but-unwritable content (extended thinking) records
            # a sentinel index. Reject it before indexing: a negative index would
            # otherwise address a block from the END of the list and overwrite the
            # wrong one, which for a thinking block earns a hard 400 from Anthropic.
            if content_idx == _UNWRITABLE_CONTENT_IDX:
                continue

            # Get the content block at the index
            if content_idx >= len(response_content):
                continue

            content_block = response_content[content_idx]

            # Verify it's a text block and update the text field
            # Handle both dict and Pydantic object content blocks
            if isinstance(content_block, dict):
                if content_block.get("type") == "text":
                    cast(Dict[str, Any], content_block)["text"] = guardrail_response
            elif hasattr(content_block, "type") and getattr(content_block, "type", None) == "text":
                # Update Pydantic object's text attribute
                if hasattr(content_block, "text"):
                    content_block.text = guardrail_response
