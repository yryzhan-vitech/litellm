from typing import Optional

from pydantic import Field

from litellm.constants import (
    NOMA_MAX_SCAN_TIMEOUT_SECONDS,
    NOMA_MIN_SCAN_TIMEOUT_SECONDS,
)

from .base import GuardrailConfigModel


class NomaGuardrailConfigModel(GuardrailConfigModel):
    use_v2: Optional[bool] = Field(
        default=False,
        description="If True and guardrail='noma', route to the new Noma v2 implementation.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="The Noma API key. Reads from NOMA_API_KEY env var if None.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The Noma API base URL. Defaults to https://api.noma.security. Also checks if the NOMA_API_KEY env var is set.",
    )
    application_id: Optional[str] = Field(
        default=None,
        description="The Noma Application ID. Reads from NOMA_APPLICATION_ID env var if None.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Noma Security"


class NomaV2GuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description="The Noma API key. Reads from NOMA_API_KEY env var if None.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The Noma API base URL. Defaults to https://api.noma.security.",
    )
    application_id: Optional[str] = Field(
        default=None,
        description="The Noma Application ID. Reads from NOMA_APPLICATION_ID env var if None.",
    )
    monitor_mode: Optional[bool] = Field(
        default=None,
        description="When true, run guardrail checks in monitor mode.",
    )
    block_failures: Optional[bool] = Field(
        default=None,
        description="When true, fail closed on Noma API errors.",
    )
    # [ARC-BUG-14] declare the shared streaming knobs so they surface in the Admin UI
    streaming_end_of_stream_only: bool | None = Field(
        default=None,
        description="When true, scan the assembled response once at end of stream instead of scanning sampled chunks during the stream.",
    )
    streaming_sampling_rate: int | None = Field(
        default=None,
        ge=1,
        description=(
            "When streaming_end_of_stream_only is False, scan every Nth streamed chunk. "
            "Ignored when streaming_end_of_stream_only is True. Must be an integer >= 1."
        ),
    )
    scan_timeout: float | None = Field(
        default=None,
        # Mirrors the bounds the runtime coercion enforces, so the Admin-UI form rejects
        # an out-of-range value at the point of entry rather than at pod startup.
        #
        # ⚠️ This is NOT a second enforcement layer, despite appearances. Config arrives as
        # `LitellmParams`, which inherits the v1 model and is `extra="allow"`, so this field
        # is not in its MRO and these bounds never fire on the config path — measured:
        # `LitellmParams(scan_timeout=99999)` is accepted. `_coerce_scan_timeout` is the
        # sole gate. Keep them agreed anyway: a looser bound here would let the UI accept a
        # value that then fails startup.
        ge=NOMA_MIN_SCAN_TIMEOUT_SECONDS,
        le=NOMA_MAX_SCAN_TIMEOUT_SECONDS,
        description=(
            "Wall-clock deadline in seconds for a single Noma scan request. Defaults to "
            "NOMA_SCAN_TIMEOUT_SECONDS. Without it the scan inherits the shared httpx "
            "client budget (600s read), which lets a hung Noma hold a request open long "
            "after the model has answered."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Noma Security v2"
