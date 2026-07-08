"""FXMacroData helpers for portfolio event-risk analysis."""

from __future__ import annotations

import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

FXMACRODATA_BASE_URL = "https://fxmacrodata.com/api/v1"


def get_release_calendar(
    currency: str = "usd",
    limit: int = 100,
    min_tier: int | None = None,
    api_key: str | None = None,
    base_url: str = FXMACRODATA_BASE_URL,
) -> pd.DataFrame:
    """Fetch an FXMacroData economic release calendar as a DataFrame."""
    limit = max(1, int(limit))
    params = {"limit": limit}
    token = api_key or os.environ.get("FXMACRODATA_API_KEY")
    if token:
        params["api_key"] = token

    url = f"{base_url.rstrip('/')}/calendar/{currency.lower()}?{urlencode(params)}"
    with urlopen(url, timeout=30) as response:  # nosec B310
        payload = json.loads(response.read().decode("utf-8"))

    events = payload.get("data", [])
    if min_tier is not None:
        events = [
            event
            for event in events
            if int(event.get("market_tier") or 99) <= int(min_tier)
        ]

    frame = pd.DataFrame(events[:limit])
    if frame.empty:
        return frame

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.set_index("date").sort_index()
    if "announcement_datetime" in frame.columns:
        frame["announcement_datetime"] = pd.to_datetime(
            frame["announcement_datetime"], unit="s", utc=True, errors="coerce"
        )
    if "announcement_datetime_utc" in frame.columns:
        frame["announcement_datetime_utc"] = pd.to_datetime(
            frame["announcement_datetime_utc"], utc=True, errors="coerce"
        )

    return frame


def event_window_mask(
    index: pd.DatetimeIndex,
    events: pd.DataFrame,
    days_before: int = 1,
    days_after: int = 1,
) -> pd.Series:
    """Return a boolean mask for dates that fall near release-calendar events."""
    normalized_index = pd.DatetimeIndex(index).normalize()
    mask = pd.Series(False, index=index)
    if events.empty:
        return mask

    event_dates = (
        pd.DatetimeIndex(events.index)
        if isinstance(events.index, pd.DatetimeIndex)
        else pd.to_datetime(events["date"], errors="coerce")
    )
    for event_date in event_dates.dropna().normalize():
        window_start = event_date - pd.Timedelta(days=days_before)
        window_end = event_date + pd.Timedelta(days=days_after)
        mask |= (normalized_index >= window_start) & (normalized_index <= window_end)
    return mask
