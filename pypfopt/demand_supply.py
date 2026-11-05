
"""Historical demand and supply zone detection from OHLCV price data.

This module detects historical demand and supply zones using price swings,
ATR-based reactions, and relative volume.

Detection is retrospective. A zone is only confirmed after the required
swing and reaction candles have completed. The pivot date and confirmation
date are both returned to help prevent look-ahead bias in backtests.

This module is independent of portfolio optimisation calculations.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def _validate_window(name: str, value: Any) -> None:
    """Validate a positive integer window parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")

    if value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_threshold(name: str, value: Any) -> None:
    """Validate a finite positive numeric threshold."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")

    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def _validate_ohlcv(
    ohlcv: pd.DataFrame,
    minimum_rows: int,
) -> pd.DataFrame:
    """Validate OHLCV input and return a clean copy."""
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    if not ohlcv.columns.is_unique:
        raise ValueError("OHLCV columns must be unique")

    missing = set(_REQUIRED_COLUMNS).difference(ohlcv.columns)

    if missing:
        raise ValueError(
            "ohlcv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if isinstance(ohlcv.index, pd.MultiIndex):
        raise ValueError("ohlcv must have a one-dimensional index")

    if ohlcv.index.hasnans:
        raise ValueError("ohlcv index cannot contain missing values")

    if not ohlcv.index.is_unique:
        raise ValueError("ohlcv index must not contain duplicate values")

    if not ohlcv.index.is_monotonic_increasing:
        raise ValueError("ohlcv index must be sorted in ascending order")

    data = ohlcv.loc[:, list(_REQUIRED_COLUMNS)].copy()

    try:
        for column in _REQUIRED_COLUMNS:
            data[column] = pd.to_numeric(data[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "OHLCV columns must contain numeric values"
        ) from exc

    if data.isna().any().any():
        raise ValueError(
            "OHLCV data cannot contain NaN or infinite values"
        )

    try:
        values = data.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "OHLCV columns must contain numeric values"
        ) from exc

    if not np.isfinite(values).all():
        raise ValueError(
            "OHLCV data cannot contain NaN or infinite values"
        )

    if len(data) < minimum_rows:
        raise ValueError(
            f"At least {minimum_rows} OHLCV rows are required; "
            f"got {len(data)}"
        )

    prices = data[["Open", "High", "Low", "Close"]]

    if (prices <= 0).any().any():
        raise ValueError("OHLC prices must be positive")

    if (data["Volume"] < 0).any():
        raise ValueError("Volume cannot be negative")

    if (
        data["High"]
        < data[["Open", "Close", "Low"]].max(axis=1)
    ).any():
        raise ValueError(
            "High must be at least Open, Close, and Low on every row"
        )

    if (
        data["Low"]
        > data[["Open", "Close", "High"]].min(axis=1)
    ).any():
        raise ValueError(
            "Low must be at most Open, Close, and High on every row"
        )

    return data


def _average_true_range(
    data: pd.DataFrame,
    window: int = 14,
) -> pd.Series:
    """Calculate ATR using a simple rolling mean of true range."""
    previous_close = data["Close"].shift(1)

    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(
        window=window,
        min_periods=1,
    ).mean().clip(lower=np.finfo(float).eps)


def _zone_strength(
    reaction_atr: float,
    volume_ratio: float,
    age: int,
    length: int,
) -> float:
    """Calculate a heuristic zone strength score between 0 and 100."""
    reaction_score = min(reaction_atr / 4.0, 1.0) * 50.0
    volume_score = min(volume_ratio / 2.0, 1.0) * 30.0

    recency_score = max(
        0.0,
        1.0 - age / max(length, 1),
    ) * 20.0

    return round(
        reaction_score + volume_score + recency_score,
        2,
    )


def _find_zones(
    data: pd.DataFrame,
    zone_type: str,
    swing_window: int,
    reaction_window: int,
    volume_window: int,
    volume_threshold: float,
    min_reaction_atr: float,
) -> list[dict[str, Any]]:
    """Find demand or supply zones in validated OHLCV data."""
    atr = _average_true_range(data, window=14)

    volume_baseline = (
        data["Volume"]
        .rolling(volume_window, min_periods=1)
        .median()
        .shift(1)
    )

    zones: list[dict[str, Any]] = []
    is_demand = zone_type == "demand"

    required_future = max(swing_window, reaction_window)

    for position in range(
        swing_window,
        len(data) - required_future,
    ):
        swing_data = data.iloc[
            position - swing_window:
            position + swing_window + 1
        ]

        pivot = data.iloc[position]

        pivot_price = float(
            pivot["Low"] if is_demand else pivot["High"]
        )

        extremes = (
            swing_data["Low"] if is_demand
            else swing_data["High"]
        )

        extreme = float(
            extremes.min() if is_demand else extremes.max()
        )

        if not np.isclose(
            pivot_price,
            extreme,
            rtol=1e-9,
            atol=1e-12,
        ):
            continue

        extreme_count = np.isclose(
            extremes.to_numpy(dtype=float),
            extreme,
            rtol=1e-9,
            atol=1e-12,
        ).sum()

        if extreme_count != 1:
            continue

        baseline = volume_baseline.iloc[position]

        if not np.isfinite(baseline) or baseline <= 0:
            continue

        volume_ratio = float(pivot["Volume"] / baseline)

        if volume_ratio < volume_threshold:
            continue

        reaction_data = data.iloc[
            position + 1:
            position + reaction_window + 1
        ]

        if is_demand:
            reaction_price = float(reaction_data["High"].max())

            reaction_atr = (
                reaction_price - float(pivot["Close"])
            ) / float(atr.iloc[position])

            lower = float(pivot["Low"])
            upper = float(max(pivot["Open"], pivot["Close"]))

        else:
            reaction_price = float(reaction_data["Low"].min())

            reaction_atr = (
                float(pivot["Close"]) - reaction_price
            ) / float(atr.iloc[position])

            lower = float(min(pivot["Open"], pivot["Close"]))
            upper = float(pivot["High"])

        if reaction_atr < min_reaction_atr:
            continue

        # Check invalidation from the first candle after the pivot.
        subsequent = data.iloc[position + 1:]

        if is_demand:
            invalid = subsequent["Close"] < lower
        else:
            invalid = subsequent["Close"] > upper

        invalidation_date = None

        if invalid.any():
            first_invalid = int(
                np.flatnonzero(invalid.to_numpy())[0]
            )
            invalidation_date = subsequent.index[first_invalid]

        confirmation_position = position + required_future

        age = len(data) - position - 1

        zones.append(
            {
                "type": zone_type,
                "date": data.index[position],
                "confirmation_date": data.index[
                    confirmation_position
                ],
                "lower": round(lower, 4),
                "upper": round(upper, 4),
                "reaction_price": round(reaction_price, 4),
                "reaction_atr": round(reaction_atr, 3),
                "volume_ratio": round(volume_ratio, 3),
                "strength": _zone_strength(
                    reaction_atr,
                    volume_ratio,
                    age,
                    len(data),
                ),
                "invalidated": bool(invalid.any()),
                "invalidation_date": invalidation_date,
            }
        )

    return zones


def detect_demand_supply_zones(
    ohlcv: pd.DataFrame,
    swing_window: int = 3,
    reaction_window: int = 10,
    volume_window: int = 20,
    volume_threshold: float = 1.2,
    min_reaction_atr: float = 1.5,
) -> dict[str, Any]:
    """Detect historical demand and supply zones from OHLCV data.

    A demand zone is a unique local swing low followed by an upward
    reaction of at least ``min_reaction_atr`` ATRs.

    A supply zone is a unique local swing high followed by a downward
    reaction of at least ``min_reaction_atr`` ATRs.

    Both require pivot volume to be at least ``volume_threshold``
    times the preceding rolling median volume.

    A demand zone is invalidated by a subsequent close below its lower
    bound. A supply zone is invalidated by a subsequent close above its
    upper bound. Invalidation is checked from the first candle after
    the pivot, including the reaction period.

    Detection is retrospective. ``date`` identifies the pivot candle,
    while ``confirmation_date`` identifies when the required future
    swing and reaction candles have completed. Backtests must not use
    a zone before its confirmation date.

    The strength score is a heuristic ranking, not a probability or
    a guarantee of future price behaviour. ATR uses a simple rolling
    mean of true range rather than Wilder smoothing.

    Parameters
    ----------
    ohlcv : pandas.DataFrame
        OHLCV data with Open, High, Low, Close, and Volume columns.
        The index must be unique and sorted in ascending order.
    swing_window : int, default 3
        Number of candles on either side used to identify a swing.
    reaction_window : int, default 10
        Number of candles after the pivot used to measure reaction.
    volume_window : int, default 20
        Number of preceding candles used for median volume.
    volume_threshold : float, default 1.2
        Minimum pivot-volume to preceding-median-volume ratio.
    min_reaction_atr : float, default 1.5
        Minimum reaction size measured in ATR units.

    Returns
    -------
    dict
        Contains ``demand_zones``, ``supply_zones``,
        ``preferred_buying_zone``, and ``preferred_selling_zone``.
        Preferred zones are the strongest non-invalidated zones.
    """
    _validate_window("swing_window", swing_window)
    _validate_window("reaction_window", reaction_window)
    _validate_window("volume_window", volume_window)

    _validate_threshold("volume_threshold", volume_threshold)
    _validate_threshold("min_reaction_atr", min_reaction_atr)

    required_future = max(swing_window, reaction_window)
    minimum_rows = swing_window + required_future + 1

    data = _validate_ohlcv(ohlcv, minimum_rows)

    common_args = {
        "swing_window": int(swing_window),
        "reaction_window": int(reaction_window),
        "volume_window": int(volume_window),
        "volume_threshold": float(volume_threshold),
        "min_reaction_atr": float(min_reaction_atr),
    }

    demand_zones = _find_zones(
        data,
        "demand",
        **common_args,
    )

    supply_zones = _find_zones(
        data,
        "supply",
        **common_args,
    )

    valid_demand = [
        zone for zone in demand_zones
        if not zone["invalidated"]
    ]

    valid_supply = [
        zone for zone in supply_zones
        if not zone["invalidated"]
    ]

    def strongest(
        zones: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Return the strongest zone, or None if no valid zone exists."""
        return max(
            zones,
            key=lambda zone: zone["strength"],
            default=None,
        )

    return {
        "demand_zones": demand_zones,
        "supply_zones": supply_zones,
        "preferred_buying_zone": strongest(valid_demand),
        "preferred_selling_zone": strongest(valid_supply),
    }
