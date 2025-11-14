import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pypfopt import EfficientCVaR, expected_returns, objective_functions
from pypfopt.exceptions import OptimizationError
from tests.utilities_for_tests import get_data, setup_efficient_cvar


def test_cvar_example():
    beta = 0.95
    df = get_data()
    mu = expected_returns.mean_historical_return(df)
    historical_rets = expected_returns.returns_from_prices(df).dropna()

    cv = EfficientCVaR(mu, historical_rets, beta=beta)
    w = cv.min_cvar()

    assert isinstance(w, dict)
    assert set(w.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    assert all([i >= -1e-5 for i in w.values()])

    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.17745746040573562, 0.017049502122532853),
        rtol=1e-4,
        atol=1e-4,
    )

    cvar = cv.portfolio_performance()[1]
    portfolio_rets = historical_rets @ cv.weights

    var_hist = portfolio_rets.quantile(1 - beta)
    cvar_hist = -portfolio_rets[portfolio_rets < var_hist].mean()
    np.testing.assert_almost_equal(cvar_hist, cvar, decimal=3)


def test_cvar_no_returns():
    # Issue 324
    df = get_data()
    historical_rets = expected_returns.returns_from_prices(df).dropna()
    assert EfficientCVaR(None, historical_rets)


def test_es_return_sample():
    df = get_data()
    mu = expected_returns.mean_historical_return(df)
    historical_rets = expected_returns.returns_from_prices(df).dropna()

    cv = EfficientCVaR(mu, historical_rets)
    w = cv.efficient_return(0.2)

    assert isinstance(w, dict)
    assert set(w.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    assert all([i >= -1e-5 for i in w.values()])

    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.2, 0.017222),
        rtol=1e-4,
        atol=1e-4,
    )
    # Cover verbose param case
    np.testing.assert_equal(
        cv.portfolio_performance(verbose=True), cv.portfolio_performance()
    )


def test_cvar_example_weekly():
    beta = 0.95
    df = get_data()
    df = df.resample("W").first()
    mu = expected_returns.mean_historical_return(df, frequency=52)
    historical_rets = expected_returns.returns_from_prices(df).dropna()
    cv = EfficientCVaR(mu, historical_rets, beta=beta)
    cv.efficient_return(0.2)
    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.2, 0.03447723250708958),
        rtol=1e-4,
        atol=1e-4,
    )

    cvar = cv.portfolio_performance()[1]
    portfolio_rets = historical_rets @ cv.weights

    var_hist = portfolio_rets.quantile(1 - beta)
    cvar_hist = -portfolio_rets[portfolio_rets < var_hist].mean()
    np.testing.assert_almost_equal(cvar_hist, cvar, decimal=3)


def test_cvar_example_monthly():
    beta = 0.95
    df = get_data()
    df = df.resample("M").first()
    mu = expected_returns.mean_historical_return(df, frequency=12)
    historical_rets = expected_returns.returns_from_prices(df).dropna()
    cv = EfficientCVaR(mu, historical_rets, beta=beta)
    cv.efficient_return(0.2)
    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.2, 0.02343809217822161),
        rtol=1e-4,
        atol=1e-4,
    )

    cvar = cv.portfolio_performance()[1]
    portfolio_rets = historical_rets @ cv.weights

    var_hist = portfolio_rets.quantile(1 - beta)
    cvar_hist = -portfolio_rets[portfolio_rets < var_hist].mean()
    np.testing.assert_almost_equal(cvar_hist, cvar, decimal=3)


def test_cvar_beta():
    # cvar should decrease (i.e higher loss) as beta increases
    cv = setup_efficient_cvar()
    cv._beta = 0.5
    cv.min_cvar()
    cvar = cv.portfolio_performance()[1]

    for beta in np.arange(0.55, 1, 0.05):
        cv = setup_efficient_cvar()
        cv._beta = beta
        cv.min_cvar()
        cvar_test = cv.portfolio_performance()[1]
        assert cvar_test >= cvar
        cvar = cvar_test


def test_cvar_example_short():
    cv = setup_efficient_cvar(weight_bounds=(-1, 1))
    w = cv.efficient_return(0.2, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 0)

    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.2, 0.008481),
        rtol=1e-4,
        atol=1e-4,
    )


def test_min_cvar_extra_constraints():
    cv = setup_efficient_cvar()
    w = cv.min_cvar()
    assert w["GOOG"] < 0.02 and w["AAPL"] > 0.02

    cv = setup_efficient_cvar()
    cv.add_constraint(lambda x: x[0] >= 0.03)
    cv.add_constraint(lambda x: x[1] <= 0.03)
    w = cv.min_cvar()
    assert w["GOOG"] >= 0.025 and w["AAPL"] <= 0.035


@pytest.mark.skipif(
    not _check_soft_dependencies(["ecos"], severity="none"),
    reason="skip test if ecos is not installed in environment",
)
def test_min_cvar_different_solver():
    cv = setup_efficient_cvar(solver="ECOS")
    w = cv.min_cvar()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    test_performance = (0.08447037713814826, 0.017049502122532853)
    np.testing.assert_allclose(
        cv.portfolio_performance(), test_performance, rtol=1e-2, atol=1e-2
    )


def test_min_cvar_tx_costs():
    # Baseline
    cv = setup_efficient_cvar()
    cv.min_cvar()
    w1 = cv.weights

    # Pretend we were initally equal weight
    cv = setup_efficient_cvar()
    prev_w = np.array([1 / cv.n_assets] * cv.n_assets)
    cv.add_objective(objective_functions.transaction_cost, w_prev=prev_w)
    cv.min_cvar()
    w2 = cv.weights

    # TX cost should  pull closer to prev portfolio
    assert np.abs(prev_w - w2).sum() < np.abs(prev_w - w1).sum()


@pytest.mark.skipif(
    not _check_soft_dependencies(["ecos"], severity="none"),
    reason="skip test if ecos is not installed in environment",
)
def test_min_cvar_L2_reg():
    cv = setup_efficient_cvar(solver="ECOS")
    cv.add_objective(objective_functions.L2_reg, gamma=0.1)
    weights = cv.min_cvar()
    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    assert all([i >= 0 for i in weights.values()])

    cv2 = setup_efficient_cvar()
    cv2.min_cvar()

    # L2_reg should pull close to equal weight
    equal_weight = np.full((cv.n_assets,), 1 / cv.n_assets)
    assert (
        np.abs(equal_weight - cv.weights).sum()
        < np.abs(equal_weight - cv2.weights).sum()
    )

    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.08981817616931259, 0.020427209685618623),
        rtol=1e-4,
        atol=1e-4,
    )


def test_min_cvar_sector_constraints():
    sector_mapper = {
        "GOOG": "tech",
        "AAPL": "tech",
        "FB": "tech",
        "AMZN": "tech",
        "BABA": "tech",
        "GE": "utility",
        "AMD": "tech",
        "WMT": "retail",
        "BAC": "fig",
        "GM": "auto",
        "T": "auto",
        "UAA": "airline",
        "SHLD": "retail",
        "XOM": "energy",
        "RRC": "energy",
        "BBY": "retail",
        "MA": "fig",
        "PFE": "pharma",
        "JPM": "fig",
        "SBUX": "retail",
    }

    sector_upper = {
        "tech": 0.2,
        "utility": 0.1,
        "retail": 0.2,
        "fig": 0.4,
        "airline": 0.05,
        "energy": 0.2,
    }
    sector_lower = {"utility": 0.01, "fig": 0.02, "airline": 0.01}

    cv = setup_efficient_cvar()
    cv.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    weights = cv.min_cvar()

    for sector in list(set().union(sector_upper, sector_lower)):
        sector_sum = 0
        for t, v in weights.items():
            if sector_mapper[t] == sector:
                sector_sum += v
        assert sector_sum <= sector_upper.get(sector, 1) + 1e-5
        assert sector_sum >= sector_lower.get(sector, 0) - 1e-5


def test_efficient_risk():
    cv = setup_efficient_cvar()
    w = cv.efficient_risk(0.02)

    assert isinstance(w, dict)
    assert set(w.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    assert all([i >= -1e-5 for i in w.values()])

    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.2267893986249195, 0.02),
        rtol=1e-4,
        atol=1e-4,
    )


def test_efficient_risk_low_risk():
    cv = setup_efficient_cvar()
    cv.min_cvar()
    min_value = cv.portfolio_performance()[1]

    # Should fail below
    with pytest.raises(OptimizationError):
        cv = setup_efficient_cvar()
        cv.efficient_risk(min_value - 0.01)

    cv = setup_efficient_cvar()
    cv.efficient_risk(min_value + 0.01)
    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.363470415007482, min_value + 0.01),
        rtol=1e-4,
        atol=1e-4,
    )


def test_efficient_risk_market_neutral():
    cv = setup_efficient_cvar(weight_bounds=(-1, 1))
    w = cv.efficient_risk(0.025, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 0)
    assert (cv.weights < 1).all() and (cv.weights > -1).all()
    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.5895653670063358, 0.025),
        rtol=1e-4,
        atol=1e-4,
    )


def test_efficient_risk_L2_reg():
    cv = setup_efficient_cvar()
    cv.add_objective(objective_functions.L2_reg, gamma=1)
    weights = cv.efficient_risk(0.03)

    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    np.testing.assert_array_less(np.zeros(len(weights)), cv.weights + 1e-4)
    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.2889961577134966, 0.029393474756427136),
        rtol=1e-4,
        atol=1e-4,
    )

    cv2 = setup_efficient_cvar()
    cv2.efficient_risk(0.19)

    # L2_reg should pull close to equal weight
    equal_weight = np.full((cv.n_assets,), 1 / cv.n_assets)
    assert (
        np.abs(equal_weight - cv.weights).sum()
        < np.abs(equal_weight - cv2.weights).sum()
    )


def test_efficient_return():
    cv = setup_efficient_cvar()
    w = cv.efficient_return(0.25)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    assert all([i >= -1e-5 for i in w.values()])

    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.25, 0.021036631225933487),
        rtol=1e-4,
        atol=1e-4,
    )


def test_efficient_return_short():
    cv = setup_efficient_cvar(weight_bounds=(-3.0, 3.0))
    w = cv.efficient_return(0.26)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.26, 0.01804624747353764),
        rtol=1e-4,
        atol=1e-4,
    )
    cvar = cv.portfolio_performance()[1]

    ef_long_only = setup_efficient_cvar(weight_bounds=(0.0, 1.0))
    ef_long_only.efficient_return(0.26)
    long_only_cvar = ef_long_only.portfolio_performance()[1]

    assert long_only_cvar > cvar


def test_efficient_return_L2_reg():
    cv = setup_efficient_cvar()
    cv.add_objective(objective_functions.L2_reg, gamma=1)
    w = cv.efficient_return(0.25)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    assert all([i >= -1e-5 for i in w.values()])
    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.25, 0.02660410793952383),
        rtol=1e-4,
        atol=1e-4,
    )


def test_cvar_errors():
    df = get_data()
    mu = expected_returns.mean_historical_return(df)
    historical_rets = expected_returns.returns_from_prices(df)

    with pytest.warns(UserWarning):
        EfficientCVaR(mu, historical_rets)

    historical_rets = historical_rets.dropna(axis=0, how="any")
    assert EfficientCVaR(mu, historical_rets)

    cv = setup_efficient_cvar()

    with pytest.raises(NotImplementedError):
        cv.min_volatility()

    with pytest.raises(NotImplementedError):
        cv.max_sharpe()

    with pytest.raises(NotImplementedError):
        cv.max_quadratic_utility()

    with pytest.raises(ValueError):
        # Beta must be between 0 and 1
        cv = EfficientCVaR(mu, historical_rets, 1)

    with pytest.warns(UserWarning):
        cv = EfficientCVaR(mu, historical_rets, 0.1)

    with pytest.raises(OptimizationError):
        # Must be <= max expected return
        cv = EfficientCVaR(mu, historical_rets)
        cv.efficient_return(target_return=np.abs(mu).max() + 0.01)

    with pytest.raises(AttributeError):
        # list not supported.
        EfficientCVaR(mu, historical_rets.to_numpy().tolist())

    historical_rets = historical_rets.iloc[:, :-1]
    with pytest.raises(ValueError):
        EfficientCVaR(mu, historical_rets)


def test_parametrization():
    cv = setup_efficient_cvar()
    cv.efficient_risk(0.19)
    cv.efficient_risk(0.19)

    cv = setup_efficient_cvar()
    cv.efficient_return(0.25)
    cv.efficient_return(0.25)
