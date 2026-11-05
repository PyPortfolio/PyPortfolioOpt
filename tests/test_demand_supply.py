
import numpy as np
import pandas as pd
import pytest

from pypfopt.demand_supply import detect_demand_supply_zones


def make_ohlcv(rows, index=None):
    if index is None:
        index = pd.date_range("2024-01-01", periods=len(rows), freq="D")

    return pd.DataFrame(
        rows,
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=index,
    )


def demand_data():
    return make_ohlcv([
        (100, 101, 99, 100, 100),
        (99, 100, 98, 99, 100),
        (98, 99, 90, 96, 300),
        (97, 104, 96, 103, 110),
        (103, 108, 102, 107, 110),
        (107, 110, 106, 109, 110),
        (109, 111, 108, 110, 110),
        (110, 112, 109, 111, 110),
    ])


def supply_data():
    return make_ohlcv([
        (100, 101, 99, 100, 100),
        (101, 102, 100, 101, 100),
        (103, 110, 102, 104, 300),
        (103, 104, 96, 97, 110),
        (97, 98, 92, 94, 110),
        (94, 96, 89, 91, 110),
        (91, 94, 88, 93, 110),
        (93, 95, 90, 94, 110),
    ])


def detect(data, **overrides):
    parameters = {
        "swing_window": 1,
        "reaction_window": 2,
        "volume_window": 2,
        "min_reaction_atr": 1,
    }
    parameters.update(overrides)
    return detect_demand_supply_zones(data, **parameters)


def test_detects_demand_zone():
    result = detect(demand_data())

    zone = result["preferred_buying_zone"]

    assert zone is not None
    assert zone["type"] == "demand"
    assert zone["lower"] == 90.0
    assert zone["upper"] == 98.0
    assert zone["invalidated"] is False


def test_demand_confirmation_date():
    result = detect(demand_data())

    zone = result["preferred_buying_zone"]

    assert zone["date"] == pd.Timestamp("2024-01-03")
    assert zone["confirmation_date"] == pd.Timestamp("2024-01-05")


def test_detects_supply_zone():
    result = detect(supply_data())

    zone = result["preferred_selling_zone"]

    assert zone is not None
    assert zone["type"] == "supply"
    assert zone["lower"] == 103.0
    assert zone["upper"] == 110.0
    assert zone["invalidated"] is False


def test_supply_confirmation_date():
    result = detect(supply_data())

    zone = result["preferred_selling_zone"]

    assert zone["date"] == pd.Timestamp("2024-01-03")
    assert zone["confirmation_date"] == pd.Timestamp("2024-01-05")


def test_demand_invalidation_during_reaction_window():
    data = make_ohlcv([
        (100, 101, 99, 100, 100),
        (99, 100, 98, 99, 100),
        (98, 99, 90, 96, 300),
        (97, 104, 96, 103, 110),
        (92, 100, 88, 89, 110),
        (89, 95, 87, 90, 110),
        (90, 92, 89, 91, 110),
        (91, 94, 90, 93, 110),
    ])

    result = detect(data, reaction_window=3, min_reaction_atr=0.5)

    zone = result["demand_zones"][0]

    assert zone["invalidated"] is True
    assert zone["invalidation_date"] == pd.Timestamp("2024-01-05")
    assert zone["confirmation_date"] == pd.Timestamp("2024-01-06")
    assert result["preferred_buying_zone"] is None


def test_supply_invalidation_during_reaction_window():
    data = make_ohlcv([
        (100, 101, 99, 100, 100),
        (101, 102, 100, 101, 100),
        (103, 110, 102, 104, 300),
        (103, 104, 96, 97, 110),
        (111, 112, 90, 111, 110),
        (110, 111, 100, 105, 110),
        (105, 108, 101, 106, 110),
        (106, 109, 103, 108, 110),
    ])

    result = detect(data, reaction_window=3, min_reaction_atr=0.5)

    zone = result["supply_zones"][0]

    assert zone["invalidated"] is True
    assert zone["invalidation_date"] == pd.Timestamp("2024-01-05")
    assert result["preferred_selling_zone"] is None


def test_rejects_missing_columns():
    data = pd.DataFrame({"Close": [100] * 20})

    with pytest.raises(ValueError, match="missing required columns"):
        detect_demand_supply_zones(data)


def test_rejects_unsorted_index():
    data = demand_data()
    data = data.iloc[::-1]

    with pytest.raises(ValueError, match="sorted in ascending order"):
        detect(data)


def test_rejects_duplicate_index():
    data = demand_data()
    data.index = [
        "2024-01-01",
        "2024-01-02",
        "2024-01-02",
        "2024-01-04",
        "2024-01-05",
        "2024-01-06",
        "2024-01-07",
        "2024-01-08",
    ]

    with pytest.raises(ValueError, match="duplicate"):
        detect(data)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_values(bad_value):
    data = demand_data().astype(float)
    data.iloc[0, 0] = bad_value

    with pytest.raises(ValueError, match="NaN or infinite"):
        detect(data)


@pytest.mark.parametrize(
    "parameters",
    [
        {"swing_window": 0},
        {"swing_window": 1.5},
        {"swing_window": True},
        {"reaction_window": -1},
        {"volume_window": True},
        {"volume_threshold": np.nan},
        {"volume_threshold": np.inf},
        {"min_reaction_atr": 0},
        {"min_reaction_atr": -1},
    ],
)
def test_rejects_invalid_parameters(parameters):
    data = demand_data()

    with pytest.raises(ValueError):
        detect(data, **parameters)


def test_rejects_insufficient_data():
    data = demand_data().iloc[:3]

    with pytest.raises(ValueError, match="At least"):
        detect(data)


def test_rejects_invalid_ohlc_relationship():
    data = demand_data()
    data.iloc[0, data.columns.get_loc("High")] = 90

    with pytest.raises(ValueError, match="High must be"):
        detect(data)


def test_rejects_negative_volume():
    data = demand_data()
    data.iloc[0, data.columns.get_loc("Volume")] = -100

    with pytest.raises(ValueError, match="Volume cannot be negative"):
        detect(data)


def test_returns_expected_result_keys():
    result = detect(demand_data())

    assert set(result) == {
        "demand_zones",
        "supply_zones",
        "preferred_buying_zone",
        "preferred_selling_zone",
    }


def test_strength_is_within_expected_range():
    result = detect(demand_data())

    zone = result["demand_zones"][0]

    assert 0 <= zone["strength"] <= 100
