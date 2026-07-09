import numpy as np
import pandas as pd
import pytest

from pypfopt import expected_returns
from tests.utilities_for_tests import get_benchmark_data, get_data


def test_returns_dataframe():
    df = get_data()
    returns_df = expected_returns.returns_from_prices(df)
    assert isinstance(returns_df, pd.DataFrame)
    assert returns_df.shape[1] == 20
    assert len(returns_df) == 7125
    assert not ((returns_df > 1) & returns_df.notnull()).any().any()


def test_prices_from_returns():
    df = get_data()
    returns_df = df.pct_change()  # keep NaN row

    # convert pseudo-price to price
    pseudo_prices = expected_returns.prices_from_returns(returns_df)
    initial_prices = df.bfill().iloc[0]
    test_prices = pseudo_prices * initial_prices

    # check equality, robust to floating point issues
    assert ((test_prices[1:] - df[1:]).fillna(0) < 1e-10).all().all()


def test_prices_from_log_returns():
    df = get_data()
    returns_df = df.pct_change()  # keep NaN row
    log_returns_df = np.log1p(returns_df)

    # convert pseudo-price to price
    pseudo_prices = expected_returns.prices_from_returns(
        log_returns_df, log_returns=True
    )
    initial_prices = df.bfill().iloc[0]
    test_prices = pseudo_prices * initial_prices

    # check equality, robust to floating point issues
    assert ((test_prices[1:] - df[1:]).fillna(0) < 1e-5).all().all()


def test_returns_from_prices():
    df = get_data()
    returns_df = expected_returns.returns_from_prices(df)
    pd.testing.assert_series_equal(returns_df.iloc[-1], df.pct_change().iloc[-1])


def test_returns_warning():
    df = get_data()
    df.iloc[3, :] = 0  # make some prices zero
    with pytest.warns(UserWarning):
        expected_returns.mean_historical_return(df)


def test_log_returns_from_prices():
    df = get_data()
    old_nan = df.isnull().sum(axis=1).sum()
    log_rets = expected_returns.returns_from_prices(df, log_returns=True)
    new_nan = log_rets.isnull().sum(axis=1).sum()
    assert new_nan == old_nan
    np.testing.assert_almost_equal(log_rets.iloc[-1, -1], 0.0001682740081102576)


def test_mean_historical_returns_dummy():
    data = pd.DataFrame(
        [
            [4.0, 2.0, 0.6, -12],
            [4.2, 2.1, 0.59, -13.2],
            [3.9, 2.0, 0.58, -11.3],
            [4.3, 2.1, 0.62, -11.7],
            [4.1, 2.2, 0.63, -10.1],
        ]
    )
    mean = expected_returns.mean_historical_return(data, frequency=1)
    test_answer = pd.Series([0.0061922, 0.0241137, 0.0122722, -0.0421775])
    pd.testing.assert_series_equal(mean, test_answer, rtol=1e-3)

    mean = expected_returns.mean_historical_return(data, compounding=False, frequency=1)
    test_answer = pd.Series([0.0086560, 0.0250000, 0.0128697, -0.03632333])
    pd.testing.assert_series_equal(mean, test_answer, rtol=1e-3)


def test_mean_historical_returns():
    df = get_data()
    mean = expected_returns.mean_historical_return(df)
    assert isinstance(mean, pd.Series)
    assert list(mean.index) == list(df.columns)
    assert mean.notnull().all()
    assert mean.dtype == "float64"
    correct_mean = np.array(
        [
            0.247967,
            0.294304,
            0.284037,
            0.1923164,
            0.371327,
            0.1360093,
            0.0328503,
            0.1200115,
            0.105540,
            0.0423457,
            0.1002559,
            0.1442237,
            -0.0792602,
            0.1430506,
            0.0736356,
            0.238835,
            0.388665,
            0.226717,
            0.1561701,
            0.2318153,
        ]
    )
    np.testing.assert_array_almost_equal(mean.values, correct_mean)


def test_mean_historical_returns_type_warning():
    df = get_data()
    mean = expected_returns.mean_historical_return(df)

    with pytest.warns(RuntimeWarning) as w:
        mean_from_array = expected_returns.mean_historical_return(np.array(df))
        assert len(w) == 1
        assert str(w[0].message) == "prices are not in a dataframe"

    np.testing.assert_array_almost_equal(mean.values, mean_from_array.values, decimal=6)


def test_mean_historical_returns_frequency():
    df = get_data()
    mean = expected_returns.mean_historical_return(df, compounding=False)
    mean2 = expected_returns.mean_historical_return(df, compounding=False, frequency=52)
    np.testing.assert_array_almost_equal(mean / 252, mean2 / 52)


def test_ema_historical_return():
    df = get_data()
    mean = expected_returns.ema_historical_return(df)
    assert isinstance(mean, pd.Series)
    assert list(mean.index) == list(df.columns)
    assert mean.notnull().all()
    assert mean.dtype == "float64"
    # Test the (warning triggering) case that input is not a dataFrame
    with pytest.warns(RuntimeWarning):
        mean_np = expected_returns.ema_historical_return(df.to_numpy())
        mean_np.name = mean.name  # These will differ.
        reset_mean = mean.reset_index(drop=True)  # Index labels would be tickers.
        pd.testing.assert_series_equal(mean_np, reset_mean)


def test_ema_historical_return_frequency():
    df = get_data()
    mean = expected_returns.ema_historical_return(df, compounding=False)
    mean2 = expected_returns.ema_historical_return(df, compounding=False, frequency=52)
    np.testing.assert_array_almost_equal(mean / 252, mean2 / 52)


def test_ema_historical_return_limit():
    df = get_data()
    sma = expected_returns.mean_historical_return(df, compounding=False)
    ema = expected_returns.ema_historical_return(df, compounding=False, span=1e10)
    np.testing.assert_array_almost_equal(ema.values, sma.values)


def test_capm_no_benchmark():
    df = get_data()
    mu = expected_returns.capm_return(df, risk_free_rate=0.02)
    assert isinstance(mu, pd.Series)
    assert list(mu.index) == list(df.columns)
    assert mu.notnull().all()
    assert mu.dtype == "float64"
    correct_mu = np.array(
        [
            0.22148462799238577,
            0.2835429647498704,
            0.14693081977908462,
            0.1488989354304723,
            0.4162399750335195,
            0.22716772604184535,
            0.3970337136813829,
            0.16733214988182069,
            0.31791477659742146,
            0.17279931642386534,
            0.15271750464365566,
            0.351778014382922,
            0.32859883451716376,
            0.1501938182844417,
            0.268295486802897,
            0.31632339201710874,
            0.27753479916328516,
            0.16959588523287855,
            0.3089119447773357,
            0.2558719211959501,
        ]
    )
    np.testing.assert_array_almost_equal(mu.values, correct_mu)
    # Test the (warning triggering) case that input is not a dataFrame
    with pytest.warns(RuntimeWarning):
        mu_np = expected_returns.capm_return(df.to_numpy(), risk_free_rate=0.02)
        mu_np.name = mu.name  # These will differ.
        mu_np.index = mu.index  # Index labels would be tickers.
        pd.testing.assert_series_equal(mu_np, mu)


def test_capm_with_benchmark():
    df = get_data()
    mkt_df = get_benchmark_data()
    mu = expected_returns.capm_return(
        df, market_prices=mkt_df, compounding=True, risk_free_rate=0.02
    )

    assert isinstance(mu, pd.Series)
    assert list(mu.index) == list(df.columns)
    assert mu.notnull().all()
    assert mu.dtype == "float64"
    correct_mu = np.array(
        [
            0.09115799375654746,
            0.09905386632033128,
            0.05676282405265752,
            0.06291827346436336,
            0.13147799781014877,
            0.10239088012000815,
            0.1311567086884512,
            0.07339649698626659,
            0.1301248935078549,
            0.07620949056643983,
            0.07629095442513395,
            0.12163575425541985,
            0.10400070536161658,
            0.0781736030988492,
            0.09185177050469516,
            0.10245700691271296,
            0.11268307946677197,
            0.07870087187919145,
            0.1275598841214107,
            0.09536788741392595,
        ]
    )
    np.testing.assert_array_almost_equal(mu.values, correct_mu)

    mu2 = expected_returns.capm_return(
        df, market_prices=mkt_df, compounding=False, risk_free_rate=0.02
    )
    assert (mu2 >= mu).all()


def test_risk_matrix_and_returns_data():
    # Test the switcher method for simple calls
    df = get_data()

    for method in {"mean_historical_return", "ema_historical_return", "capm_return"}:
        mu = expected_returns.return_model(df, method=method)

        assert isinstance(mu, pd.Series)
        assert list(mu.index) == list(df.columns)
        assert mu.notnull().all()
        assert mu.dtype == "float64"

        mu2 = expected_returns.return_model(
            expected_returns.returns_from_prices(df), method=method, returns_data=True
        )
        pd.testing.assert_series_equal(mu, mu2)


def test_return_model_additional_kwargs():
    df = get_data()
    mkt_prices = get_benchmark_data()

    mu1 = expected_returns.return_model(
        df, method="capm_return", market_prices=mkt_prices, risk_free_rate=0.03
    )
    mu2 = expected_returns.capm_return(
        df, market_prices=mkt_prices, risk_free_rate=0.03
    )
    pd.testing.assert_series_equal(mu1, mu2)


def test_return_model_not_implemented():
    df = get_data()
    with pytest.raises(NotImplementedError):
        expected_returns.return_model(df, method="fancy_new!")


def test_log_return_passthrough():
    # addresses #343
    df = get_data()

    for method in {"mean_historical_return", "ema_historical_return", "capm_return"}:
        mu1 = expected_returns.return_model(df, method=method, log_returns=False)
        mu2 = expected_returns.return_model(df, method=method, log_returns=True)
        try:
            pd.testing.assert_series_equal(mu1, mu2)
        except AssertionError:
            return
        assert False


def test_james_stein_return():
    df = get_data()
    js_return = expected_returns.james_stein_return(df)

    assert isinstance(js_return, pd.Series)
    assert list(js_return.index) == list(df.columns)
    assert js_return.notnull().all()
    assert js_return.dtype == "float64"


def test_james_stein_return_shrinkage():
    df = get_data()

    # With shrinkage=0 we recover the sample estimate (mean_historical_return).
    js_return_0 = expected_returns.james_stein_return(df, shrinkage=0.0)
    sample_mean = expected_returns.mean_historical_return(df)
    np.testing.assert_array_almost_equal(js_return_0.values, sample_mean.values)

    # With shrinkage=1 every asset collapses to the grand mean.
    js_return_1 = expected_returns.james_stein_return(df, shrinkage=1.0)
    assert (js_return_1 == js_return_1.iloc[0]).all()

    # A manual intermediate intensity still yields a valid Series.
    js_return_05 = expected_returns.james_stein_return(df, shrinkage=0.5)
    assert isinstance(js_return_05, pd.Series)


def test_james_stein_return_auto_shrinkage():
    df = get_data()
    js_return_auto = expected_returns.james_stein_return(df, shrinkage=None)
    assert isinstance(js_return_auto, pd.Series)
    assert js_return_auto.notnull().all()


def test_james_stein_return_invalid_shrinkage():
    df = get_data()
    with pytest.raises(ValueError):
        expected_returns.james_stein_return(df, shrinkage=-0.1)
    with pytest.raises(ValueError):
        expected_returns.james_stein_return(df, shrinkage=1.5)


def test_james_stein_return_compounding():
    df = get_data()
    js_compound = expected_returns.james_stein_return(df, compounding=True)
    js_simple = expected_returns.james_stein_return(df, compounding=False)
    assert isinstance(js_compound, pd.Series)
    assert isinstance(js_simple, pd.Series)
    assert js_compound.notnull().all()
    assert js_simple.notnull().all()


def test_james_stein_return_frequency():
    df = get_data()
    js_annual = expected_returns.james_stein_return(
        df, compounding=False, frequency=252
    )
    js_monthly = expected_returns.james_stein_return(
        df, compounding=False, frequency=12
    )
    np.testing.assert_array_almost_equal(js_annual / 252, js_monthly / 12)


def test_james_stein_return_with_return_model():
    df = get_data()
    js_direct = expected_returns.james_stein_return(df, shrinkage=0.3)
    js_dispatch = expected_returns.return_model(
        df, method="james_stein_return", shrinkage=0.3
    )
    pd.testing.assert_series_equal(js_direct, js_dispatch)


def test_james_stein_return_data_warning():
    df = get_data()
    with pytest.warns(RuntimeWarning):
        js_return = expected_returns.james_stein_return(df.to_numpy())
        assert isinstance(js_return, pd.Series)


def test_james_stein_return_log_returns():
    df = get_data()
    js_normal = expected_returns.james_stein_return(df, log_returns=False)
    js_log = expected_returns.james_stein_return(df, log_returns=True)
    assert not (js_normal == js_log).all()


def test_james_stein_return_returns_data():
    df = get_data()
    returns = expected_returns.returns_from_prices(df)
    js_prices = expected_returns.james_stein_return(df)
    js_returns = expected_returns.james_stein_return(returns, returns_data=True)
    pd.testing.assert_series_equal(js_prices, js_returns)
