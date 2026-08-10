"""A transport chunk boundary must not suppress the output scan.

[ARC-BUG-50] `_check_streaming_has_ended` and `get_streaming_string_so_far` both parsed
SSE one raw byte chunk at a time, with no buffering across them. The network may split a
frame anywhere, so a `message_delta` cut in half failed `json.loads`, logged a warning,
and was skipped — and since a non-null `stop_reason` on `message_delta` was the ONLY
accepted end-of-stream signal, that single split frame made the whole stream read as
"not finished". `process_output_streaming_response` then never reached `apply_guardrail`:
no scan, no error, no metric.

Consequential on prd-ai in particular, where `noma-post-call` runs
`streaming_end_of_stream_only: true` and `noma-during-call` is off, so a streaming
request gets exactly ONE output scan — the one this removed. All guardrails run
`monitor_mode` / `block_failures: false`, so it failed silently by design.

Attribution: upstream PR #17619, not an Arcadia commit.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.anthropic.chat.guardrail_translation.handler import (  # noqa: E402
    AnthropicMessagesHandler,
)


def _sse(event_type: str, payload: str) -> bytes:
    return f"event: {event_type}\ndata: {payload}\n\n".encode()


MESSAGE_DELTA = _sse(
    "message_delta",
    '{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":9}}',
)
MESSAGE_STOP = _sse("message_stop", '{"type":"message_stop"}')
TEXT_DELTA = _sse(
    "content_block_delta",
    '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"SENSITIVE PHRASE"}}',
)


def _halves(frame: bytes):
    """Split a frame at its midpoint — the network may cut anywhere."""
    mid = len(frame) // 2
    return [frame[:mid], frame[mid:]]


@pytest.fixture
def handler():
    return AnthropicMessagesHandler()


def test_a_whole_message_delta_still_ends_the_stream(handler):
    """Baseline — the path that always worked must keep working."""
    assert handler._check_streaming_has_ended([MESSAGE_DELTA]) is True


def test_a_message_delta_split_across_chunks_still_ends_the_stream(handler):
    """The regression. Same bytes, two chunks — this used to return False.

    A False here means the output scan is skipped for the entire stream, which on prd-ai
    is the only output scan a streaming request receives.
    """
    assert handler._check_streaming_has_ended(_halves(MESSAGE_DELTA)) is True


def test_a_delta_split_at_every_possible_offset_still_ends_the_stream(handler):
    """The boundary can fall anywhere, so assert on every offset, not one.

    A midpoint-only test passes for a fix that happens to work at that offset — e.g. one
    that joins only the last two chunks.
    """
    failures = [
        offset
        for offset in range(1, len(MESSAGE_DELTA))
        if handler._check_streaming_has_ended([MESSAGE_DELTA[:offset], MESSAGE_DELTA[offset:]]) is not True
    ]
    assert failures == [], f"end-of-stream missed when split at offsets {failures[:10]}"


def test_message_stop_alone_ends_the_stream(handler):
    """The second signal, and the reason one bad frame is no longer fatal.

    Anthropic always sends `message_stop` last. Accepting it means a malformed
    `message_delta` costs the `stop_reason` detail but not the scan.
    """
    assert handler._check_streaming_has_ended([MESSAGE_STOP]) is True


def test_a_corrupt_delta_does_not_suppress_a_following_message_stop(handler):
    """The real-world shape: a frame genuinely truncated, then a clean stop.

    Joining cannot repair a frame whose remainder never arrives — the fallback is what
    saves the scan here.
    """
    corrupt = b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_re\n\n'
    assert handler._check_streaming_has_ended([corrupt, MESSAGE_STOP]) is True


def test_a_stream_with_no_end_signal_is_still_reported_as_unfinished(handler):
    """The negative control. Without it, "always return True" would pass everything.

    Returning True too eagerly is its own bug: the scan would run on a partial response
    and the guardrail would score text the user has not finished receiving.
    """
    assert handler._check_streaming_has_ended([TEXT_DELTA]) is False
    assert handler._check_streaming_has_ended([]) is False


def test_a_null_stop_reason_is_not_an_end_signal(handler):
    """`stop_reason: null` appears mid-stream — it must not be read as the end."""
    mid_stream = _sse("message_delta", '{"type":"message_delta","delta":{"stop_reason":null}}')
    assert handler._check_streaming_has_ended([mid_stream]) is False


def test_dict_format_message_stop_ends_the_stream(handler):
    """The already-parsed path needs the same fallback, or the two disagree."""
    assert handler._check_streaming_has_ended([{"type": "message_stop"}]) is True
    assert handler._check_streaming_has_ended([{"type": "message_delta", "delta": {"stop_reason": "end_turn"}}]) is True
    assert handler._check_streaming_has_ended([{"type": "message_delta", "delta": {"stop_reason": None}}]) is False


def test_text_split_across_chunks_reaches_the_scan_payload(handler):
    """The sibling defect: dropped text is worse than no scan.

    A guardrail handed a response with a hole in it returns a verdict that looks
    authoritative. This used to lose the whole delta when it straddled a boundary.
    """
    assert handler.get_streaming_string_so_far([TEXT_DELTA]) == "SENSITIVE PHRASE"
    assert handler.get_streaming_string_so_far(_halves(TEXT_DELTA)) == "SENSITIVE PHRASE"


def test_text_split_at_every_offset_reaches_the_scan_payload(handler):
    failures = [
        offset
        for offset in range(1, len(TEXT_DELTA))
        if handler.get_streaming_string_so_far([TEXT_DELTA[:offset], TEXT_DELTA[offset:]]) != "SENSITIVE PHRASE"
    ]
    assert failures == [], f"text lost when split at offsets {failures[:10]}"


def test_a_multibyte_character_split_across_chunks_survives(handler):
    """A UTF-8 character can itself straddle a boundary.

    Decoding per chunk raises or mangles here; joining first makes the character whole
    before it is decoded. Emoji is deliberate — 4 bytes, so it can be cut three ways.
    """
    frame = _sse(
        "content_block_delta",
        '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"café 🎉 done"}}',
    )
    for offset in range(1, len(frame)):
        assert handler.get_streaming_string_so_far([frame[:offset], frame[offset:]]) == "café 🎉 done", (
            f"multibyte text corrupted when split at offset {offset}"
        )


def test_bytes_and_dict_chunks_mix_without_double_counting(handler):
    """Both formats appear in one stream, and neither may be counted twice.

    The bytes are now joined and parsed once, then the dict chunks are walked — an easy
    way to get this wrong is to keep the old per-chunk call as well and emit the text twice.
    """
    dict_chunk = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "+from-dict"}}
    assert handler.get_streaming_string_so_far([*_halves(TEXT_DELTA), dict_chunk]) == "SENSITIVE PHRASE+from-dict"


def test_an_unrecoverable_truncated_byte_does_not_raise(handler):
    """The `errors="ignore"` backstop must itself be tested.

    A multi-byte character straddling a boundary is repaired by joining — that case is
    covered above. This is the case joining CANNOT repair: the stream's FINAL chunk ends
    on a lead byte whose continuation never arrives, so the joined buffer is genuinely
    undecodable. Strict decoding raises `UnicodeDecodeError` there, and since this runs
    inside the guardrail's output path a raise would fail the request instead of the check
    — turning a truncated tail into a client-visible error.

    Found by mutation testing: dropping `errors="ignore"` left all twelve other tests
    green, because every one of them supplies both halves of any split character.
    """
    truncated_tail = MESSAGE_STOP + "é".encode()[:1]  # lone 0xC3 lead byte, no continuation

    assert handler._check_streaming_has_ended([truncated_tail]) is True, (
        "an undecodable tail must not hide the end-of-stream signal that precedes it"
    )


def test_an_undecodable_chunk_does_not_lose_the_text_before_it(handler):
    """Same backstop on the text-extraction path, which feeds the scan payload.

    A raise here would be worse than on the end-of-stream check: the guardrail would
    receive nothing and the response would go out unscanned.
    """
    truncated_tail = TEXT_DELTA + "🎉".encode()[:2]  # two of four bytes

    assert handler.get_streaming_string_so_far([truncated_tail]) == "SENSITIVE PHRASE"


def test_the_skip_marker_is_logged_at_error_so_it_cannot_be_gated_away():
    """The skip log's LEVEL is part of the fix, and nothing else pins it.

    An adversarial review pointed out that a future edit "helpfully" downgrading this
    `.error()` to `.warning()` would pass every other test in this file — while making the
    marker droppable by a bare `LITELLM_LOG=ERROR` handler. That is not hypothetical: the
    ARC-BUG-53 decline log was written at `.warning()` on exactly that mistaken reasoning,
    and only survives because `json_logs: true` happens to reset the handler level.

    Asserted against the source rather than by executing the branch, because reaching the
    branch requires a full streaming guardrail fixture while the property under test is
    simply "this call site is at ERROR". A source assertion cannot pass for the wrong
    reason here: there is exactly one such call site.
    """
    import inspect

    from litellm.llms.anthropic.chat.guardrail_translation import handler as handler_module

    source = inspect.getsource(handler_module)
    marker = "Guardrail output scan DEGRADED"

    assert source.count(marker) == 1, "more than one skip-log site — update this test"

    # Walk back from the message to the logger call that carries it.
    preamble = source[: source.index(marker)]
    call = preamble[preamble.rindex("verbose_proxy_logger") :]

    assert call.startswith("verbose_proxy_logger.error("), (
        f"the skip marker must be logged at ERROR, found: {call.splitlines()[0]!r}. "
        "A WARNING here is droppable by a LITELLM_LOG=ERROR handler when json_logs is off."
    )
