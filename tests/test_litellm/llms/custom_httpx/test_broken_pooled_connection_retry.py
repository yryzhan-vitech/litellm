"""A dead pooled connection is not a deadline expiry.

[ARC-BUG-47 site C] aiohttp raises ``SocketTimeoutError`` when a keep-alive connection the
peer has already closed breaks WHILE READING — not only when a read budget expires — and
``map_aiohttp_exceptions`` maps that to ``httpx.ReadTimeout``. So a connection recycled by
an intermediary surfaces as a "timeout" in single-digit milliseconds against a
multi-second budget, and the request never reached the peer.

Measured on prd-ai: 74 such failures in ~25 minutes, elapsed p50 0.036 s, max 0.093 s
against a 10 s budget — about 24 % of scans never reached the vendor while every counter
recorded success.

The retry is narrow on purpose. Two broader fixes were rejected and must stay rejected:
reclassifying the exception at ``aiohttp_transport.py:25`` flips 408 → 500 for every
consumer and disables the gate site A already depends on; widening the
``RemoteProtocolError`` arm is an idempotency break, because a ``ReadTimeout`` means the
request WAS delivered. This gate's first condition is what makes a retry safe — a broken
pooled connection means the peer never received it.
"""

import asyncio
import os
import sys

import aiohttp
import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.custom_httpx.aiohttp_transport import map_aiohttp_exceptions  # noqa: E402
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler  # noqa: E402


def _timeout_exc(cause_cls, message: str = "boom"):
    """Build the exception exactly as the transport would, preserving ``__cause__``."""
    try:
        with map_aiohttp_exceptions():
            raise cause_cls(message)
    except Exception as exc:  # noqa: BLE001 - the mapped type is what we want to return
        return exc


def _is_broken(exc, *, stream=False, elapsed=0.03, timeout=10.0):
    return AsyncHTTPHandler._is_broken_pooled_connection(exc=exc, stream=stream, elapsed=elapsed, timeout=timeout)


def test_the_transport_preserves_the_cause_the_gate_depends_on():
    """The whole gate rests on ``raise mapped_exc(message) from exc``.

    Asserted directly, because if upstream ever drops the ``from exc`` the gate silently
    stops matching and every broken connection goes back to being reported as a timeout —
    with no test failing anywhere else.
    """
    exc = _timeout_exc(aiohttp.SocketTimeoutError)

    assert isinstance(exc, httpx.ReadTimeout)
    assert isinstance(exc.__cause__, aiohttp.SocketTimeoutError)


def test_a_broken_pooled_connection_is_retried():
    """The case this fixes: a sub-1 % elapsed SocketTimeoutError on a non-streaming call."""
    assert _is_broken(_timeout_exc(aiohttp.SocketTimeoutError), elapsed=0.03, timeout=10.0) is True


@pytest.mark.parametrize(
    "cause",
    [aiohttp.ServerTimeoutError, asyncio.TimeoutError],
    ids=["server_timeout", "asyncio_timeout"],
)
def test_a_genuine_expiry_is_not_retried(cause):
    """These mean the peer had the request and was too slow.

    Retrying doubles a wait the caller already decided was too long — and on a
    non-idempotent POST it can double the side effect too.
    """
    assert _is_broken(_timeout_exc(cause), elapsed=0.03, timeout=10.0) is False


def test_a_streaming_call_is_never_retried():
    """Bytes may already be on their way to the caller.

    This is also what keeps ARC-BUG-47 site B out of scope: once a response has opened,
    a retry duplicates a partial answer, and that case genuinely needs client-side retry.
    """
    assert _is_broken(_timeout_exc(aiohttp.SocketTimeoutError), stream=True, elapsed=0.001) is False


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [(0.0, True), (0.49, True), (0.5, False), (0.51, False), (5.0, False), (9.99, False)],
)
def test_the_elapsed_fraction_boundary(elapsed, expected):
    """5 % of a 10 s budget is the line, and it is exclusive.

    Below it the connection cannot plausibly have done work; at or above it, it can.
    Pinned as a boundary table rather than one happy value, because a fix that compared
    against the wrong side of the threshold would pass a single-point test.
    """
    assert _is_broken(_timeout_exc(aiohttp.SocketTimeoutError), elapsed=elapsed, timeout=10.0) is expected


@pytest.mark.parametrize("timeout", [None, 0, -1, "30", object()], ids=["none", "zero", "negative", "str", "object"])
def test_an_unknown_budget_is_not_retried(timeout):
    """Fail closed: an unknown budget must not be treated as a large one.

    Without this, ``elapsed < None * 0.05`` either raises or silently permits, and the
    retry would fire on failures nobody has measured. ``"30"`` is included because a
    string budget is truthy and would sail through a `if timeout:` check.
    """
    assert _is_broken(_timeout_exc(aiohttp.SocketTimeoutError), elapsed=0.001, timeout=timeout) is False


def test_a_timeout_with_no_cause_is_not_retried():
    """A bare ``httpx.ReadTimeout`` — raised by code that never went through the transport."""
    assert _is_broken(httpx.ReadTimeout("no cause"), elapsed=0.001, timeout=10.0) is False


def test_an_httpx_timeout_objects_read_budget_is_the_one_measured():
    """``httpx.Timeout`` carries four budgets; ``read`` is what a SocketTimeoutError hits.

    Keyed on ``read`` rather than ``connect``: with connect=1 s and read=30 s, a 0.1 s
    failure is 10 % of connect but 0.3 % of read, and only the second reading is right.
    """
    budget = httpx.Timeout(timeout=30.0, connect=1.0)

    assert _is_broken(_timeout_exc(aiohttp.SocketTimeoutError), elapsed=0.1, timeout=budget) is True
    assert _is_broken(_timeout_exc(aiohttp.SocketTimeoutError), elapsed=2.0, timeout=budget) is False


class TestRemainingBudget:
    """A retry inherits the remaining budget — it never gets a fresh one.

    Otherwise one dead connection doubles the worst-case latency the caller budgeted for,
    which is the same complaint the retry is meant to remove.
    """

    def test_a_float_budget_is_reduced_by_the_elapsed_time(self):
        assert AsyncHTTPHandler._remaining_timeout(timeout=10.0, elapsed=0.03) == pytest.approx(9.97)

    def test_a_nearly_exhausted_budget_keeps_a_one_second_floor(self):
        """A near-zero budget guarantees the retry fails, which is worse than not retrying."""
        assert AsyncHTTPHandler._remaining_timeout(timeout=10.0, elapsed=9.8) == 1.0
        assert AsyncHTTPHandler._remaining_timeout(timeout=10.0, elapsed=99.0) == 1.0

    def test_a_timeout_object_keeps_its_other_budgets(self):
        """Only ``read`` shrinks — resetting connect/write/pool would change unrelated behaviour."""
        original = httpx.Timeout(timeout=30.0, connect=5.0, write=7.0, pool=9.0)

        remaining = AsyncHTTPHandler._remaining_timeout(timeout=original, elapsed=0.05)

        assert isinstance(remaining, httpx.Timeout)
        assert remaining.read == pytest.approx(29.95)
        assert remaining.connect == 5.0
        assert remaining.write == 7.0
        assert remaining.pool == 9.0

    def test_an_unknown_budget_falls_back_without_raising(self):
        assert AsyncHTTPHandler._remaining_timeout(timeout=None, elapsed=0.1) == 1.0


class TestPostRetriesInPlace:
    """End-to-end through ``post()``, not just the predicate.

    The gate tests above prove the DECISION is right; these prove the decision is
    actually wired to a retry — a fix can have a perfect predicate that nothing calls.
    """

    @staticmethod
    def _handler(send_side_effect):
        """An AsyncHTTPHandler whose transport is replaced, so nothing leaves the process.

        The constructor builds its own client, so the swap happens after construction —
        passing one in is not supported.
        """
        from unittest.mock import AsyncMock, MagicMock

        handler = AsyncHTTPHandler(timeout=httpx.Timeout(timeout=10.0))
        handler.client = MagicMock()
        handler.client.build_request = MagicMock(return_value=MagicMock())
        handler.client.send = AsyncMock(side_effect=send_side_effect)
        return handler

    @staticmethod
    def _ok_response():
        from unittest.mock import MagicMock

        response = MagicMock(spec=httpx.Response)
        response.raise_for_status = MagicMock(return_value=None)
        return response

    @pytest.mark.asyncio
    async def test_a_broken_connection_is_retried_and_succeeds(self, monkeypatch):
        """First attempt dies on a closed pooled connection; the retry answers."""
        from unittest.mock import AsyncMock, MagicMock

        ok = self._ok_response()
        handler = self._handler([_timeout_exc(aiohttp.SocketTimeoutError)])

        retried = AsyncMock(return_value=ok)
        monkeypatch.setattr(handler, "single_connection_post_request", retried)
        monkeypatch.setattr(handler, "create_client", MagicMock(return_value=AsyncMock()))

        result = await handler.post(
            url="https://example.invalid/scan", json={"a": 1}, retry_on_broken_pooled_connection=True
        )

        assert result is ok, "the retry's response was not returned"
        assert retried.await_count == 1, "the retry never fired"

    @pytest.mark.asyncio
    async def test_the_retry_inherits_the_remaining_budget(self, monkeypatch):
        """A retry must not be handed a fresh full deadline.

        Asserted on the value passed to ``create_client``, because that is where the
        budget is actually granted — checking the helper alone would not catch a call
        site that ignores it.
        """
        from unittest.mock import AsyncMock, MagicMock

        handler = self._handler([_timeout_exc(aiohttp.SocketTimeoutError)])
        create_client = MagicMock(return_value=AsyncMock())
        monkeypatch.setattr(handler, "create_client", create_client)
        monkeypatch.setattr(handler, "single_connection_post_request", AsyncMock(return_value=self._ok_response()))

        await handler.post(
            url="https://example.invalid/scan",
            json={"a": 1},
            timeout=10.0,
            retry_on_broken_pooled_connection=True,
        )

        granted = create_client.call_args.kwargs["timeout"]
        # `<= 10.0`, not `< 10.0`: a mocked transport fails in ~0 s, so the deduction
        # rounds to zero and the budget is legitimately unchanged. Asserting strictly
        # less would test the mock's clock rather than the inheritance rule — that rule
        # is pinned exactly in TestRemainingBudget, where elapsed is supplied directly.
        assert granted <= 10.0, f"the retry was granted MORE than the original budget: {granted}"
        assert granted >= 1.0, f"the retry budget collapsed below the floor: {granted}"

    @pytest.mark.asyncio
    async def test_the_retry_budget_is_deducted_when_time_actually_passes(self, monkeypatch):
        """Catch a call site that ignores `_remaining_timeout` and passes `timeout` through.

        A mocked transport fails in ~0 s, so the deduction rounds to nothing and a mutant
        that grants a fresh budget looks identical to the real thing. Time has to actually
        move for the two to separate.

        ⚠️ A monotonically advancing clock is used rather than a scripted sequence:
        `time.time()` is called five times inside one `post()` (the handler twice, the
        timing decorator the rest), so a fixed-length iterator raises StopIteration
        mid-request — which surfaces as `RuntimeError: coroutine raised StopIteration` and
        reads like a code bug. Each call advances 2 s, so the handler's own start/end pair
        is 2 s apart at minimum and the deduction is non-zero regardless of call order.
        """
        from unittest.mock import AsyncMock, MagicMock

        import time as time_module

        ticks = {"t": 1000.0}

        def _clock():
            ticks["t"] += 2.0
            return ticks["t"]

        monkeypatch.setattr(time_module, "time", _clock)

        handler = self._handler([_timeout_exc(aiohttp.SocketTimeoutError)])
        create_client = MagicMock(return_value=AsyncMock())
        monkeypatch.setattr(handler, "create_client", create_client)
        monkeypatch.setattr(handler, "single_connection_post_request", AsyncMock(return_value=self._ok_response()))

        await handler.post(
            url="https://example.invalid/scan",
            json={"a": 1},
            timeout=1000.0,
            retry_on_broken_pooled_connection=True,
        )

        granted = create_client.call_args.kwargs["timeout"]
        assert granted < 1000.0, (
            f"the retry was granted the ORIGINAL budget ({granted}) — the elapsed time was "
            "not deducted, so one dead connection doubles the caller's worst case"
        )

    @pytest.mark.asyncio
    async def test_a_genuine_expiry_still_raises_litellm_timeout(self, monkeypatch):
        """The unchanged path: a real deadline expiry must keep its error contract.

        `litellm.Timeout` with the "Timeout passed=… time taken=…" message is what
        downstream classification and the monitoring greps key on, so a fix that
        swallowed it would break both while looking like an improvement.
        """
        from unittest.mock import AsyncMock, MagicMock

        import litellm

        handler = self._handler([_timeout_exc(aiohttp.ServerTimeoutError)])
        retried = AsyncMock()
        monkeypatch.setattr(handler, "single_connection_post_request", retried)
        monkeypatch.setattr(handler, "create_client", MagicMock(return_value=AsyncMock()))

        with pytest.raises(litellm.Timeout) as raised:
            await handler.post(
                url="https://example.invalid/scan",
                json={"a": 1},
                timeout=10.0,
                retry_on_broken_pooled_connection=True,
            )

        assert "Timeout passed=" in str(raised.value)
        assert retried.await_count == 0, "a genuine expiry must NOT be retried"

    @pytest.mark.asyncio
    async def test_a_second_broken_connection_is_not_retried_again(self, monkeypatch):
        """Retried ONCE. A retry loop on a persistently dead pool is an outage amplifier.

        ⚠️ This test hands the mock an already-correct ``litellm.Timeout`` and then checks
        that a ``litellm.Timeout`` comes back, so it proves only the retry COUNT. It
        structurally cannot see the exception SHAPE, because it never runs the real
        ``single_connection_post_request``. An adversarial review used exactly that blind
        spot to find a leak this file thought it covered — see
        ``test_a_second_timeout_is_still_reported_as_litellm_Timeout`` below, which drives
        the real method.
        """
        from unittest.mock import AsyncMock, MagicMock

        import litellm

        handler = self._handler([_timeout_exc(aiohttp.SocketTimeoutError)])
        monkeypatch.setattr(handler, "create_client", MagicMock(return_value=AsyncMock()))
        # The retry itself fails the same way.
        monkeypatch.setattr(
            handler,
            "single_connection_post_request",
            AsyncMock(side_effect=litellm.Timeout(message="second failure", model="m", llm_provider="p")),
        )

        with pytest.raises(litellm.Timeout):
            await handler.post(
                url="https://example.invalid/scan",
                json={"a": 1},
                timeout=10.0,
                retry_on_broken_pooled_connection=True,
            )

    @pytest.mark.asyncio
    async def test_a_second_timeout_is_still_reported_as_litellm_Timeout(self, monkeypatch):
        """🔴 The retry's OWN failure must not escape as a raw ``httpx`` exception.

        ``single_connection_post_request`` has no exception handling of its own, and the
        retry was originally wrapped in ``try/finally`` — which closes the fresh client and
        then re-raises whatever came out. So a retry that also timed out propagated a raw
        ``httpx.ReadTimeout`` out of ``post()``: the one shape no upstream retry, fallback
        or cooldown layer matches on, turning a classified timeout into an unhandled 500.

        Found by an adversarial review, reproduced before fixing, and pinned here.

        The retry client is driven through the REAL ``single_connection_post_request`` —
        mocking that method is precisely what hid this — so the exception type this
        assertion sees is the one production would raise.
        """
        from unittest.mock import AsyncMock, MagicMock

        import litellm

        handler = self._handler([_timeout_exc(aiohttp.SocketTimeoutError)])

        # A fresh client whose send() ALSO times out, unmapped — no litellm typing.
        retry_client = MagicMock()
        retry_client.build_request = MagicMock(return_value=MagicMock())
        retry_client.send = AsyncMock(side_effect=httpx.ReadTimeout("the retry timed out too"))
        retry_client.aclose = AsyncMock()
        monkeypatch.setattr(handler, "create_client", MagicMock(return_value=retry_client))

        with pytest.raises(litellm.Timeout) as raised:
            await handler.post(
                url="https://example.invalid/scan",
                json={"a": 1},
                timeout=10.0,
                retry_on_broken_pooled_connection=True,
            )

        assert "Timeout passed=" in str(raised.value), "not the shared litellm.Timeout shape"
        assert retry_client.send.await_count == 1, "the real retry path never ran"
        assert retry_client.aclose.await_count == 1, "the fresh client leaked"

    @pytest.mark.asyncio
    async def test_a_non_timeout_failure_on_the_retry_is_not_swallowed(self, monkeypatch):
        """The new ``except`` arm must catch ONLY timeouts.

        A ``ConnectError`` on the retry is a different fault with a different remedy;
        relabelling it ``litellm.Timeout`` would send a caller chasing a deadline that
        never expired. Without this test the fix could be widened to a bare ``except``
        and every test above would still pass.
        """
        from unittest.mock import AsyncMock, MagicMock

        handler = self._handler([_timeout_exc(aiohttp.SocketTimeoutError)])

        retry_client = MagicMock()
        retry_client.build_request = MagicMock(return_value=MagicMock())
        retry_client.send = AsyncMock(side_effect=httpx.ConnectError("dns is gone"))
        retry_client.aclose = AsyncMock()
        monkeypatch.setattr(handler, "create_client", MagicMock(return_value=retry_client))

        with pytest.raises(httpx.ConnectError):
            await handler.post(
                url="https://example.invalid/scan",
                json={"a": 1},
                timeout=10.0,
                retry_on_broken_pooled_connection=True,
            )

        assert retry_client.aclose.await_count == 1, "the fresh client leaked on the error path"

    @pytest.mark.asyncio
    async def test_the_reported_elapsed_covers_both_attempts(self, monkeypatch):
        """The ``litellm.Timeout`` must report the wall time the CALLER actually waited.

        Reporting only the first attempt's elapsed would understate it — and that number
        feeds the very elapsed-vs-budget classification this fix exists to keep honest, so
        an understated value could make a real deadline expiry look like a dead connection.

        The clock is patched on the stdlib ``time`` module, because ``http_handler`` does
        ``import time`` and resolves the attribute at call time. Patching an attribute on
        the handler's own module does nothing — a mutant survived for exactly that reason.

        ⚠️ The per-read tick must stay well under ``budget × 0.05`` (here 10.0 × 0.05 =
        0.5s), or the first attempt's own elapsed fails the broken-connection gate, the
        retry never fires, and the test passes for the wrong reason. Drafted at 0.5s and
        caught by this assertion — kept as a warning.
        """
        import time as _time_module
        from unittest.mock import AsyncMock, MagicMock

        import litellm

        # post() reads the clock several times; a fixed-length iterator raises
        # StopIteration mid-call, which asyncio re-labels as RuntimeError. A monotonic
        # generator keeps advancing however many reads the implementation makes.
        tick = 0.01
        state = {"t": 1000.0}

        def fake_time():
            state["t"] += tick
            return state["t"]

        monkeypatch.setattr(_time_module, "time", fake_time)

        handler = self._handler([_timeout_exc(aiohttp.SocketTimeoutError)])
        retry_client = MagicMock()
        retry_client.build_request = MagicMock(return_value=MagicMock())
        retry_client.send = AsyncMock(side_effect=httpx.ReadTimeout("also timed out"))
        retry_client.aclose = AsyncMock()
        monkeypatch.setattr(handler, "create_client", MagicMock(return_value=retry_client))

        with pytest.raises(litellm.Timeout) as raised:
            await handler.post(
                url="https://example.invalid/scan",
                json={"a": 1},
                timeout=10.0,
                retry_on_broken_pooled_connection=True,
            )

        assert retry_client.send.await_count == 1, "the retry never fired — tick too large?"

        # The single-attempt path reads the clock twice (start, end). Re-measuring after the
        # retry adds at least one more read, so the reported elapsed must exceed 2 ticks.
        message = str(raised.value)
        assert "time taken=" in message
        taken = float(message.split("time taken=")[1].split(" ")[0])
        assert taken > 2 * tick, f"elapsed {taken} covers only the first attempt"

    @pytest.mark.asyncio
    async def test_the_retry_is_OFF_by_default(self, monkeypatch):
        """🔴 The whole safety model: this handler is shared by 377 `.post()` call sites.

        Among them are non-idempotent billing writes (`integrations/lago.py`,
        `integrations/openmeter.py`) and Slack alerting. The retry's safety argument — "a
        broken pooled connection means the peer never received it" — holds for the
        client-to-immediate-peer TCP leg, but every consumer sits behind a gateway, and a
        `SocketTimeoutError` says nothing about whether that gateway had already forwarded
        the request upstream. Likeliest is not proven, so the default must not retry.

        Raised by an adversarial review, which found the Lago call site and correctly
        refused to accept "provably safe for every caller" as asserted rather than shown.
        """
        from unittest.mock import AsyncMock, MagicMock

        import litellm

        handler = self._handler([_timeout_exc(aiohttp.SocketTimeoutError)])
        retried = AsyncMock()
        monkeypatch.setattr(handler, "single_connection_post_request", retried)
        monkeypatch.setattr(handler, "create_client", MagicMock(return_value=AsyncMock()))

        with pytest.raises(litellm.Timeout):
            await handler.post(url="https://example.invalid/billing-event", json={"amount": 1}, timeout=10.0)

        assert retried.await_count == 0, (
            "a caller that did not opt in was silently retried — a non-idempotent POST could be delivered twice"
        )


class TestTheRetryCarriesTheSameRequest:
    """🔴 A retry that sends a DIFFERENT body is worse than the timeout it replaces.

    `post()` normalises its arguments through `_prepare_request_data_and_content`, which
    routes a str/bytes `data` into httpx's `content`. The retry originally forwarded the
    caller's RAW `data`, so attempt 1 sent `content=b'...'` and the retry sent
    `content=None` — an empty body. For a guardrail scan that is a POST of nothing that
    returns 200 and looks authoritative.

    Found by a code review, reproduced, then fixed by forwarding the normalised pair.
    """

    @staticmethod
    def _handler_and_recorder(monkeypatch):
        """A handler whose first attempt breaks and whose retry records its build_request."""
        from unittest.mock import AsyncMock, MagicMock

        recorded = []

        def _record(*args, **kwargs):
            # *args because `build_request` is also called positionally ("POST", url) —
            # a keyword-only recorder raises TypeError inside the mock and the test then
            # fails for its own reason rather than the code's.
            recorded.append({key: kwargs.get(key) for key in ("data", "json", "content", "files")})
            return MagicMock()

        handler = AsyncHTTPHandler(timeout=httpx.Timeout(timeout=10.0))
        handler.client = MagicMock()
        handler.client.build_request = MagicMock(side_effect=_record)
        handler.client.send = AsyncMock(side_effect=[_timeout_exc(aiohttp.SocketTimeoutError)])

        retry_client = MagicMock()
        retry_client.build_request = MagicMock(side_effect=_record)
        retry_client.send = AsyncMock(side_effect=httpx.ReadTimeout("the retry timed out too"))
        retry_client.aclose = AsyncMock()
        monkeypatch.setattr(handler, "create_client", MagicMock(return_value=retry_client))
        return handler, recorded

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs,carried",
        [
            ({"content": b'{"scan":"SENSITIVE"}'}, "content"),
            ({"data": b'{"scan":"SENSITIVE"}'}, "content"),
            ({"json": {"scan": "SENSITIVE"}}, "json"),
            ({"data": {"form": "field"}}, "data"),
        ],
        ids=["content_bytes", "data_bytes_routed_to_content", "json_dict", "data_dict_form"],
    )
    async def test_the_retry_sends_the_same_body(self, monkeypatch, kwargs, carried):
        """Every body form the caller can supply must survive the retry byte-for-byte."""
        import litellm

        handler, recorded = self._handler_and_recorder(monkeypatch)

        with pytest.raises(litellm.Timeout):
            await handler.post(
                url="https://example.invalid/scan",
                timeout=10.0,
                retry_on_broken_pooled_connection=True,
                **kwargs,
            )

        assert len(recorded) == 2, f"expected one retry, saw {len(recorded)} attempts"
        first, retry = recorded
        assert first[carried], f"precondition: attempt 1 did not carry {carried}"
        assert retry[carried] == first[carried], (
            f"the retry changed the request body: attempt1 {carried}={first[carried]!r} "
            f"retry {carried}={retry[carried]!r}"
        )

    @pytest.mark.asyncio
    async def test_a_multipart_request_is_not_retried_at_all(self, monkeypatch):
        """`files` cannot be forwarded, so the retry must be DECLINED, not truncated.

        `single_connection_post_request` takes no `files` parameter. Retrying anyway sent a
        multipart request with the file part missing — a partial body that succeeds. A
        request that fails loudly is recoverable; one that succeeds with the wrong payload
        is not. Measured: attempt 1 carried `files`, the retry carried `files=None`.
        """
        import litellm

        handler, recorded = self._handler_and_recorder(monkeypatch)

        with pytest.raises(litellm.Timeout):
            await handler.post(
                url="https://example.invalid/scan",
                data={"field": "value"},
                files={"upload": b"payload"},
                timeout=10.0,
                retry_on_broken_pooled_connection=True,
            )

        assert len(recorded) == 1, "a multipart request must not be retried"
        assert recorded[0]["files"], "precondition: attempt 1 did not carry files"
