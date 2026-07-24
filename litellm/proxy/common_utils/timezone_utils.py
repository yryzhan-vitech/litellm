from datetime import datetime, timezone

import litellm
from litellm.litellm_core_utils.duration_parser import get_next_standardized_reset_time


def get_budget_reset_timezone():
    """
    Get the budget reset timezone from litellm_settings.
    Falls back to UTC if not specified.

    litellm_settings values are set as attributes on the litellm module
    by proxy_server.py at startup (via setattr(litellm, key, value)).
    """
    return getattr(litellm, "timezone", None) or "UTC"


def get_budget_reset_time(budget_duration: str) -> datetime:
    """
    Get the budget reset time based on the configured timezone.
    Falls back to UTC if not specified.
    """

    reset_at = get_next_standardized_reset_time(
        duration=budget_duration,
        current_time=datetime.now(timezone.utc),
        timezone_str=get_budget_reset_timezone(),
    )
    return reset_at


# [ARC-BUG-06] shared predicate to decide whether a budget-window (re)arm on
# key/user/team update starts a fresh window (upstream PR #34493)
def is_budget_window_newly_armed(
    new_duration: "str | None",
    existing_duration: "str | None",
    existing_reset_at: "datetime | None",
) -> bool:
    if not new_duration:
        return False
    if existing_reset_at is None:
        return True
    now = datetime.now(existing_reset_at.tzinfo)
    if existing_reset_at < now:
        return True
    return new_duration != existing_duration
