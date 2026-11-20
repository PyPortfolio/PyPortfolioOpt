import warnings

import numpy as np
import pandas as pd
import pytest

from pypfopt import expected_returns


@pytest.mark.parametrize("bad_price", [0.0, -1.0])
@pytest.mark.parametrize("log_returns", [False, True])
@pytest.mark.parametrize("as_series", [False, True])
def test_nonpositive_prices_warn_with_asset_and_date(bad_price, log_returns, as_series):
    prices = pd.DataFrame(
        {"WTI": [10.0, bad_price, -2.0]},
        index=pd.date_range("2020-04-20", periods=3),
    )
    if as_series:
        prices = prices["WTI"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.warns(UserWarning, match="Non-positive price.*WTI.*2020-04-21"):
            actual = expected_returns.returns_from_prices(
                prices, log_returns=log_returns
            )
        reference = prices.pct_change(fill_method=None)
        if log_returns:
            reference = np.log(1 + reference)
        reference = reference.dropna(how="all")

    if as_series:
        pd.testing.assert_series_equal(actual, reference)
    else:
        pd.testing.assert_frame_equal(actual, reference)


@pytest.mark.parametrize("log_returns", [False, True])
def test_positive_prices_and_missing_observations_do_not_warn(log_returns):
    prices = pd.DataFrame(
        {"A": [np.nan, 10.0, 11.0, 9.0], "B": [20.0, np.nan, 19.0, 20.0]}
    )
    original = prices.copy(deep=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected_returns.returns_from_prices(prices, log_returns=log_returns)

    assert not caught
    pd.testing.assert_frame_equal(prices, original)


@pytest.mark.parametrize(
    "estimator",
    [expected_returns.mean_historical_return, expected_returns.ema_historical_return],
)
def test_estimators_warn_for_negative_prices(estimator):
    prices = pd.DataFrame({"WTI": [-10.0, -11.0, -12.0]})
    with pytest.warns(UserWarning, match="Non-positive price.*WTI"):
        estimator(prices, frequency=1)


def test_negative_return_inputs_are_not_treated_as_prices():
    returns = pd.DataFrame({"A": [-0.1, 0.0, 0.2]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = expected_returns.mean_historical_return(
            returns, returns_data=True, frequency=1
        )

    assert not caught
    pd.testing.assert_series_equal(actual, (1 + returns).prod() ** (1 / 3) - 1)
