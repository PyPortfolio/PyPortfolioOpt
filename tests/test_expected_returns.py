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


def _make_ff_test_data(model="ff3", n_periods=120, n_assets=4, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="B")

    factor_cols = ["Mkt-RF", "SMB", "HML"]
    if model == "ff5":
        factor_cols += ["RMW", "CMA"]

    factors = pd.DataFrame(
        {
            "RF": rng.normal(0.0001, 0.00002, n_periods),
            "Mkt-RF": rng.normal(0.0004, 0.008, n_periods),
            "SMB": rng.normal(0.0002, 0.004, n_periods),
            "HML": rng.normal(0.0001, 0.004, n_periods),
        },
        index=dates,
    )

    if model == "ff5":
        factors["RMW"] = rng.normal(0.00015, 0.003, n_periods)
        factors["CMA"] = rng.normal(0.0001, 0.003, n_periods)

    betas = rng.normal(0.6, 0.15, size=(n_assets, len(factor_cols)))
    alphas = rng.normal(0.0002, 0.00005, size=n_assets)

    factor_matrix = factors[factor_cols].to_numpy()
    excess_returns = alphas + factor_matrix @ betas.T
    returns = excess_returns + factors["RF"].to_numpy()[:, None]

    returns_df = pd.DataFrame(
        returns, index=dates, columns=[f"Asset {i+1}" for i in range(n_assets)]
    )
    prices = expected_returns.prices_from_returns(returns_df)
    prices.columns = returns_df.columns

    return prices, returns_df, factors


def test_ff3_return():
    prices, _, factors = _make_ff_test_data(model="ff3")
    mu = expected_returns.ff_return(prices, factors, model="ff3")

    assert isinstance(mu, pd.Series)
    assert list(mu.index) == list(prices.columns)
    assert mu.notnull().all()
    assert mu.dtype == "float64"


def test_ff5_return():
    prices, _, factors = _make_ff_test_data(model="ff5")
    mu = expected_returns.ff_return(prices, factors, model="ff5")

    assert isinstance(mu, pd.Series)
    assert list(mu.index) == list(prices.columns)
    assert mu.notnull().all()
    assert mu.dtype == "float64"


def test_ff_return_missing_columns():
    prices, _, factors = _make_ff_test_data(model="ff3")
    factors = factors.drop(columns=["HML"])

    with pytest.raises(ValueError, match="missing required columns"):
        expected_returns.ff_return(prices, factors, model="ff3")


def test_ff_return_invalid_model():
    prices, _, factors = _make_ff_test_data(model="ff3")

    with pytest.raises(ValueError, match="model must be either"):
        expected_returns.ff_return(prices, factors, model="ff4")


def test_ff_return_no_overlap():
    prices, _, factors = _make_ff_test_data(model="ff3")
    factors.index = pd.date_range("2030-01-01", periods=len(factors), freq="B")

    with pytest.raises(ValueError, match="No overlapping dates"):
        expected_returns.ff_return(prices, factors, model="ff3")


def test_return_model_ff3_return():
    prices, _, factors = _make_ff_test_data(model="ff3")

    mu1 = expected_returns.return_model(prices, method="ff3_return", factor_data=factors)
    mu2 = expected_returns.ff_return(prices, factors)

    pd.testing.assert_series_equal(mu1, mu2)


def test_return_model_ff5_return():
    prices, _, factors = _make_ff_test_data(model="ff5")

    mu1 = expected_returns.return_model(prices, method="ff5_return", factor_data=factors)
    mu2 = expected_returns.ff_return(prices, factors)

    pd.testing.assert_series_equal(mu1, mu2)


def _make_ff_known_data(model="ff3"):
    rng = np.random.default_rng(7)
    dates = pd.date_range("2021-01-01", periods=80, freq="B")

    factors = pd.DataFrame(
        {
            "RF": rng.normal(0.0001, 0.00002, len(dates)),
            "Mkt-RF": rng.normal(0.0004, 0.008, len(dates)),
            "SMB": rng.normal(0.0002, 0.004, len(dates)),
            "HML": rng.normal(0.0001, 0.004, len(dates)),
        },
        index=dates,
    )

    if model == "ff5":
        factors["RMW"] = rng.normal(0.00015, 0.003, len(dates))
        factors["CMA"] = rng.normal(0.00010, 0.003, len(dates))

    factor_cols = ["Mkt-RF", "SMB", "HML"]
    if model == "ff5":
        factor_cols += ["RMW", "CMA"]

    alphas = np.array([0.00020, 0.00010, -0.00005])
    betas = np.array(
        [
            [1.10, 0.40, -0.20, 0.00, 0.00],
            [0.80, -0.10, 0.35, 0.00, 0.00],
            [1.25, 0.05, -0.15, 0.00, 0.00],
        ]
    )

    if model == "ff3":
        betas = betas[:, :3]

    factor_matrix = factors[factor_cols].to_numpy()
    returns = factors["RF"].to_numpy()[:, None] + alphas + factor_matrix @ betas.T

    returns_df = pd.DataFrame(
        returns, index=dates, columns=["Asset 1", "Asset 2", "Asset 3"]
    )
    return returns_df, factors, alphas, betas, factor_cols


def test_ff5_return_missing_columns():
    prices, _, factors = _make_ff_test_data(model="ff5")
    factors = factors.drop(columns=["CMA"])

    with pytest.raises(ValueError, match="missing required columns"):
        expected_returns.ff_return(prices, factors, model="ff5")


def test_ff_return_no_valid_rows_after_dropna():
    prices, _, factors = _make_ff_test_data(model="ff3")
    factors = factors.copy()
    factors[["RF", "Mkt-RF", "SMB", "HML"]] = np.nan

    with pytest.raises(ValueError, match="No valid rows after aligning"):
        expected_returns.ff_return(prices, factors, model="ff3")


def test_ff3_return_recovers_known_linear_model():
    returns_df, factors, alphas, betas, factor_cols = _make_ff_known_data(model="ff3")

    mu = expected_returns.ff_return(
        returns_df,
        factors,
        returns_data=True,
        model="ff3",
        compounding=False,
        frequency=1,
    )

    expected = (
        factors["RF"].mean()
        + alphas
        + betas @ factors[factor_cols].mean().to_numpy()
    )
    expected = pd.Series(expected, index=returns_df.columns, dtype="float64")

    pd.testing.assert_series_equal(mu, expected, check_exact=False, rtol=1e-10, atol=1e-12)


def test_ff5_return_recovers_known_linear_model():
    returns_df, factors, alphas, betas, factor_cols = _make_ff_known_data(model="ff5")

    mu = expected_returns.ff_return(
        returns_df,
        factors,
        returns_data=True,
        model="ff5",
        compounding=False,
        frequency=1,
    )

    expected = (
        factors["RF"].mean()
        + alphas
        + betas @ factors[factor_cols].mean().to_numpy()
    )

    expected = pd.Series(
        expected,
        index=returns_df.columns,
        dtype="float64",
    )

    pd.testing.assert_series_equal(
        mu,
        expected,
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )


def test_ff_return_returns_data():
    prices, _, factors = _make_ff_test_data(model="ff3")

    returns_df = expected_returns.returns_from_prices(prices)

    mu_from_prices = expected_returns.ff_return(
        prices,
        factors,
        model="ff3",
    )

    mu_from_returns = expected_returns.ff_return(
        returns_df,
        factors,
        model="ff3",
        returns_data=True,
    )

    pd.testing.assert_series_equal(mu_from_prices, mu_from_returns)


def test_ff_return_compounding_branch():
    returns_df, factors, alphas, betas, factor_cols = _make_ff_known_data(model="ff3")

    mu = expected_returns.ff_return(
        returns_df,
        factors,
        returns_data=True,
        model="ff3",
        compounding=True,
        frequency=252,
    )

    expected_period_return = (
        factors["RF"].mean()
        + alphas
        + betas @ factors[factor_cols].mean().to_numpy()
    )
    expected = pd.Series(
        (1 + expected_period_return) ** 252 - 1,
        index=returns_df.columns,
        dtype="float64",
    )

    pd.testing.assert_series_equal(mu, expected, check_exact=False, rtol=1e-10, atol=1e-12)


def test_ff_return_ignores_extra_factor_columns():
    prices, _, factors = _make_ff_test_data(model="ff3")
    factors = factors.copy()
    factors["Unused"] = 123.456

    mu_with_extra = expected_returns.ff_return(prices, factors, model="ff3")
    mu_without_extra = expected_returns.ff_return(
        prices, factors.drop(columns=["Unused"]), model="ff3"
    )

    pd.testing.assert_series_equal(mu_with_extra, mu_without_extra)
