from typing import TYPE_CHECKING

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    NOMA_MAX_SCAN_TIMEOUT_SECONDS,
    NOMA_MIN_SCAN_TIMEOUT_SECONDS,
)
from litellm.types.guardrails import SupportedGuardrailIntegrations

from .noma import NomaGuardrail
from .noma_v2 import NomaV2Guardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _get_config_value(litellm_params: object, optional_params: object, attribute_name: str) -> object | None:
    # [ARC-BUG-14] mirrors generic_guardrail_api: optional_params wins over top-level
    if optional_params is not None:
        value = (
            optional_params.get(attribute_name)
            if isinstance(optional_params, dict)
            else getattr(optional_params, attribute_name, None)
        )
        if value is not None:
            return value
    return getattr(litellm_params, attribute_name, None)


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    use_v2 = getattr(litellm_params, "use_v2", False)
    if isinstance(use_v2, str):
        use_v2 = use_v2.lower() == "true"
    if use_v2:
        return initialize_guardrail_v2(litellm_params=litellm_params, guardrail=guardrail)

    import litellm

    _noma_callback = NomaGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        application_id=litellm_params.application_id,
        monitor_mode=litellm_params.monitor_mode,
        block_failures=litellm_params.block_failures,
        anonymize_input=litellm_params.anonymize_input,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_noma_callback)

    return _noma_callback


def initialize_guardrail_v2(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    optional_params = getattr(litellm_params, "optional_params", None)

    # [ARC-BUG-43] `is None`, not `or`: an explicit `scan_timeout: 0` must reach the
    # coercion and raise the documented startup error, not fall through to `timeout` (or
    # to the default) and silently become a working budget.
    scan_timeout = _get_config_value(litellm_params, optional_params, "scan_timeout")
    if scan_timeout is None:
        # `timeout` is the shared generic knob, not ours, and it was legal-and-inert here
        # before this deadline existed. Raising on an out-of-range value would turn an
        # existing config into a pod that never becomes ready — and for DB-stored
        # guardrails the registry re-initialises on poll, so an Admin-UI edit could break
        # a RUNNING pod. Clamp it into range with a warning instead, and keep raising only
        # for `scan_timeout`, which callers set deliberately for this guardrail.
        generic_timeout = getattr(litellm_params, "timeout", None)
        if generic_timeout is not None:
            try:
                clamped = min(
                    max(float(generic_timeout), NOMA_MIN_SCAN_TIMEOUT_SECONDS),
                    NOMA_MAX_SCAN_TIMEOUT_SECONDS,
                )
            except (TypeError, ValueError):
                verbose_proxy_logger.warning(
                    "Noma guardrail %s: ignoring un-numeric timeout=%r; using the default scan deadline",
                    guardrail.get("guardrail_name", ""),
                    generic_timeout,
                )
            else:
                if clamped != float(generic_timeout):
                    verbose_proxy_logger.warning(
                        "Noma guardrail %s: timeout=%s is outside [%s, %s]; clamped to %ss for the "
                        "scan deadline. Set scan_timeout explicitly to choose a value.",
                        guardrail.get("guardrail_name", ""),
                        generic_timeout,
                        NOMA_MIN_SCAN_TIMEOUT_SECONDS,
                        NOMA_MAX_SCAN_TIMEOUT_SECONDS,
                        clamped,
                    )
                scan_timeout = clamped

    _noma_v2_callback = NomaV2Guardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        application_id=litellm_params.application_id,
        monitor_mode=litellm_params.monitor_mode,
        block_failures=litellm_params.block_failures,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        # [ARC-BUG-14] forward the shared streaming knobs; unwired upstream
        streaming_end_of_stream_only=_get_config_value(litellm_params, optional_params, "streaming_end_of_stream_only"),
        streaming_sampling_rate=_get_config_value(litellm_params, optional_params, "streaming_sampling_rate"),
        # [ARC-BUG-43] per-guardrail scan deadline; falls back to NOMA_SCAN_TIMEOUT_SECONDS.
        # `timeout` is the documented generic knob (LitellmParams.timeout, "Per-request
        # timeout for the guardrail provider API call") that four sibling guardrails
        # already consume, so honour it rather than leaving an operator's `timeout: 2`
        # silently inert. `scan_timeout` wins when both are set, being the specific one.
        scan_timeout=scan_timeout,
    )
    litellm.logging_callback_manager.add_litellm_callback(_noma_v2_callback)

    return _noma_v2_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.NOMA.value: initialize_guardrail,
    SupportedGuardrailIntegrations.NOMA_V2.value: initialize_guardrail_v2,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.NOMA.value: NomaGuardrail,
    SupportedGuardrailIntegrations.NOMA_V2.value: NomaV2Guardrail,
}
