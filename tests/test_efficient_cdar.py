import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pypfopt import EfficientCDaR, expected_returns, objective_functions
from pypfopt.exceptions import OptimizationError
from tests.utilities_for_tests import get_data, setup_efficient_cdar


def test_cdar_example():
    beta = 0.95
    cd = setup_efficient_cdar(beta=beta)
    w = cd.min_cdar()
    cdar = cd.portfolio_performance()[1]

    assert isinstance(w, dict)
    assert set(w.keys()) == set(cd.tickers)
    np.testing.assert_almost_equal(cd.weights.sum(), 1)
    assert all([i >= -1e-5 for i in w.values()])

    np.testing.assert_allclose(
        cd.portfolio_performance(),
        (0.14798, 0.056433),
        rtol=1e-4,
        atol=1e-4,
    )

    df = get_data()
    historical_rets = expected_returns.returns_from_prices(df).dropna()
    portfolio_rets = historical_rets @ cd.weights
    cum_rets = portfolio_rets.cumsum(0)
    drawdown = cum_rets.cummax() - cum_rets

    dar_hist = drawdown.quantile(beta)
    cdar_hist = drawdown[drawdown > dar_hist].mean()
    np.testing.assert_almost_equal(cdar_hist, cdar, decimal=3)


def test_es_return_sample():
    cd = setup_efficient_cdar()
    w = cd.efficient_return(0.2)

    assert isinstance(w, dict)
    assert set(w.keys()) == set(cd.tickers)
    np.testing.assert_almost_equal(cd.weights.sum(), 1)
    assert all([i >= -1e-5 for i in w.values()])

    np.testing.assert_allclose(
        cd.portfolio_performance(),
        (0.2, 0.063709),
        rtol=1e-4,
        atol=1e-4,
    )
    # Cover verbose param case
    np.testing.assert_equal(
        cd.portfolio_performance(verbose=True), cd.portfolio_performance()
    )


def test_cdar_example_weekly():
    beta = 0.90
    df = get_data()
    df = df.resample("W").first()
    mu = expected_returns.mean_historical_return(df, frequency=52)
    historical_rets = expected_returns.returns_from_prices(df).dropna()
    cd = EfficientCDaR(mu, historical_rets, beta=beta)
    cd.efficient_return(0.21)
    np.testing.assert_allclose(
        cd.portfolio_performance(),
        (0.21, 0.045085),
        rtol=1e-4,
        atol=1e-4,
    )

    cdar = cd.portfolio_performance()[1]
    portfolio_rets = historical_rets @ cd.weights

    cum_rets = portfolio_rets.cumsum(0)
    drawdown = cum_rets.cummax() - cum_rets

    dar_hist = drawdown.quantile(beta)
    cdar_hist = drawdown[drawdown > dar_hist].mean()
    np.testing.assert_almost_equal(cdar_hist, cdar, decimal=3)


def test_cdar_example_monthly():
    beta = 0.90
    df = get_data()
    df = df.resample("M").first()
    mu = expected_returns.mean_historical_return(df, frequency=12)
    historical_rets = expected_returns.returns_from_prices(df).dropna()
    cd = EfficientCDaR(mu, historical_rets, beta=beta)
    cd.efficient_return(0.23)
    np.testing.assert_allclose(
        cd.portfolio_performance(),
        (0.23, 0.035683),
        rtol=1e-4,
        atol=1e-4,
    )

    cdar = cd.portfolio_performance()[1]
    portfolio_rets = historical_rets @ cd.weights

    cum_rets = portfolio_rets.cumsum(0)
    drawdown = cum_rets.cummax() - cum_rets

    dar_hist = drawdown.quantile(beta)
    cdar_hist = drawdown[drawdown > dar_hist].mean()
    np.testing.assert_almost_equal(cdar_hist, cdar, decimal=3)


def test_cdar_beta():
    # cdar should decrease (i.e higher loss) as beta increases
    cd = setup_efficient_cdar()
    cd._beta = 0.5
    cd.min_cdar()
    cdar = cd.portfolio_performance()[1]

    for beta in np.arange(0.55, 1, 0.05):
        cd = setup_efficient_cdar()
        cd._beta = beta
        cd.min_cdar()
        cdar_test = cd.portfolio_performance()[1]
        assert cdar_test >= cdar
        cdar = cdar_test


def test_cdar_example_short():
    cd = setup_efficient_cdar(weight_bounds=(-1, 1))
    w = cd.efficient_return(0.2, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cd.tickers)
    np.testing.assert_almost_equal(cd.weights.sum(), 0)

    np.testing.assert_allclose(
        cd.portfolio_performance(),
        (0.2, 0.022799),
        rtol=1e-4,
        atol=1e-4,
    )


def test_min_cdar_extra_constraints():
    cd = setup_efficient_cdar()
    w = cd.min_cdar()
    assert w["GOOG"] < 0.02 and w["MA"] > 0.02

    cd = setup_efficient_cdar()
    cd.add_constraint(lambda x: x[0] >= 0.03)
    cd.add_constraint(lambda x: x[16] <= 0.03)
    w = cd.min_cdar()
    assert w["GOOG"] >= 0.025 and w["MA"] <= 0.035


@pytest.mark.skipif(
    not _check_soft_dependencies(["ecos"], severity="none"),
    reason="skip test if ecos is not installed in environment",
)
def test_min_cdar_different_solver():
    cd = setup_efficient_cdar(solver="ECOS")
    w = cd.min_cdar()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cd.tickers)
    np.testing.assert_almost_equal(cd.weights.sum(), 1)

    assert all([i >= -1e-5 for i in w.values()])
    test_performance = (0.14798, 0.056433)
    np.testing.assert_allclose(
        cd.portfolio_performance(), test_performance, rtol=1e-2, atol=1e-2
    )


def test_min_cdar_tx_costs():
    # Baseline
    cd = setup_efficient_cdar()
    cd.min_cdar()
    w1 = cd.weights

    # Pretend we were initally equal weight
    cd = setup_efficient_cdar()
    prev_w = np.array([1 / cd.n_assets] * cd.n_assets)
    cd.add_objective(objective_functions.transaction_cost, w_prev=prev_w)
    cd.min_cdar()
    w2 = cd.weights

    # TX cost should  pull closer to prev portfolio
    assert np.abs(prev_w - w2).sum() < np.abs(prev_w - w1).sum()


@pytest.mark.skipif(
    not _check_soft_dependencies(["ecos"], severity="none"),
    reason="skip test if ecos is not installed in environment",
)
def test_min_cdar_L2_reg():
    cd = setup_efficient_cdar(solver="ECOS")
    cd.add_objective(objective_functions.L2_reg, gamma=0.1)
    weights = cd.min_cdar()
    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(cd.tickers)
    np.testing.assert_almost_equal(cd.weights.sum(), 1)
    assert all([i >= -1e-5 for i in weights.values()])

    cd2 = setup_efficient_cdar()
    cd2.min_cdar()

    # L2_reg should pull close to equal weight
    equal_weight = np.full((cd.n_assets,), 1 / cd.n_assets)
    assert (
        np.abs(equal_weight - cd.weights).sum()
        < np.abs(equal_weight - cd2.weights).sum()
    )

    np.testing.assert_allclose(
        cd.portfolio_performance(),
        (0.135105, 0.060849),
        rtol=1e-4,
        atol=1e-4,
    )


def test_min_cdar_sector_constraints():
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

    cd = setup_efficient_cdar()
    cd.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    weights = cd.min_cdar()

    for sector in list(set().union(sector_upper, sector_lower)):
        sector_sum = 0
        for t, v in weights.items():
            if sector_mapper[t] == sector:
                sector_sum += v
        assert sector_sum <= sector_upper.get(sector, 1) + 1e-5
        assert sector_sum >= sector_lower.get(sector, 0) - 1e-5


def test_efficient_risk():
    cd = setup_efficient_cdar()
    w = cd.efficient_risk(0.08)

    assert isinstance(w, dict)
    assert set(w.keys()) == set(cd.tickers)
    np.testing.assert_almost_equal(cd.weights.sum(), 1)
    assert all([i >= -1e-5 for i in w.values()])

    np.testing.assert_allclose(
        cd.portfolio_performance(),
        (0.261922, 0.08),
        rtol=1e-4,
        atol=1e-4,
    )


def test_efficient_risk_low_risk():
    cd = setup_efficient_cdar()
    cd.min_cdar()
    min_value = cd.portfolio_performance()[1]

    # Should fail below
    with pytest.raises(OptimizationError):
        cd = setup_efficient_cdar()
        cd.efficient_risk(min_value - 0.01)

    cd = setup_efficient_cdar()
    cd.efficient_risk(min_value + 0.01)
    np.testing.assert_allclose(
        cd.portfolio_performance(),
        (0.212772, min_value + 0.01),
        rtol=1e-4,
        atol=1e-4,
    )


def test_efficient_risk_market_neutral():
    cd = setup_efficient_cdar(weight_bounds=(-1, 1))
    w = cd.efficient_risk(0.025, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cd.tickers)
    np.testing.assert_almost_equal(cd.weights.sum(), 0)
    assert (cd.weights < 1).all() and (cd.weights > -1).all()
    np.testing.assert_allclose(
        cd.portfolio_performance(),
        (0.219306, 0.025),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.skip(reason="failing test, unknown reason. See bug report #642.")
def test_efficient_risk_L2_reg():
    cd = setup_efficient_cdar()
    cd.add_objective(objective_functions.L2_reg, gamma=1)
    weights = cd.efficient_risk(0.18)

    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(cd.tickers)
    np.testing.assert_almost_equal(cd.weights.sum(), 1)
    np.testing.assert_array_less(np.zeros(len(weights)), cd.weights + 1e-4)
    np.testing.assert_allclose(
        cd.portfolio_performance(),
        (0.288999, 0.178443),
        rtol=1e-4,
        atol=1e-4,
    )

    cd2 = setup_efficient_cdar()
    cd2.efficient_risk(0.18)

    # L2_reg should pull close to equal weight
    equal_weight = np.full((cd.n_assets,), 1 / cd.n_assets)
    assert (
        np.abs(equal_weight - cd.weights).sum()
        < np.abs(equal_weight - cd2.weights).sum()
    )


def test_efficient_return():
    cd = setup_efficient_cdar()
    w = cd.efficient_return(0.25)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cd.tickers)
    np.testing.assert_almost_equal(cd.weights.sum(), 1)
    assert all([i >= -1e-5 for i in w.values()])

    np.testing.assert_allclose(
        cd.portfolio_performance(),
        (0.25, 0.076193),
        rtol=1e-4,
        atol=1e-4,
    )


def test_efficient_return_short():
    cd = setup_efficient_cdar(weight_bounds=(-3.0, 3.0))
    w = cd.efficient_return(0.28)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cd.tickers)
    np.testing.assert_almost_equal(cd.weights.sum(), 1)
    np.testing.assert_allclose(
        cd.portfolio_performance(),
        (0.28, 0.045999),
        rtol=1e-4,
        atol=1e-4,
    )
    cdar = cd.portfolio_performance()[1]

    ef_long_only = cd = setup_efficient_cdar(weight_bounds=(0.0, 1.0))
    ef_long_only.efficient_return(0.26)
    long_only_cdar = ef_long_only.portfolio_performance()[1]

    assert long_only_cdar > cdar


@pytest.mark.skip(reason="failing test, unknown reason. See bug report #642.")
def test_efficient_return_L2_reg():
    cd = setup_efficient_cdar()
    cd.add_objective(objective_functions.L2_reg, gamma=1)
    w = cd.efficient_return(0.25)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cd.tickers)
    np.testing.assert_almost_equal(cd.weights.sum(), 1)
    assert all([i >= -1e-5 for i in w.values()])
    np.testing.assert_allclose(
        cd.portfolio_performance(),
        (0.25, 0.101115),
        rtol=1e-4,
        atol=1e-4,
    )


def test_cdar_errors():
    df = get_data()
    mu = expected_returns.mean_historical_return(df)
    historical_rets = expected_returns.returns_from_prices(df)

    with pytest.warns(UserWarning):
        EfficientCDaR(mu, historical_rets)

    historical_rets = historical_rets.dropna(axis=0, how="any")
    assert EfficientCDaR(mu, historical_rets)

    cd = setup_efficient_cdar()

    with pytest.raises(NotImplementedError):
        cd.min_volatility()

    with pytest.raises(NotImplementedError):
        cd.max_sharpe()

    with pytest.raises(NotImplementedError):
        cd.max_quadratic_utility()

    with pytest.raises(ValueError):
        # Beta must be between 0 and 1
        cd = EfficientCDaR(mu, historical_rets, 1)

    with pytest.raises(OptimizationError):
        # Must be <= max expected return
        cd = EfficientCDaR(mu, historical_rets)
        cd.efficient_return(target_return=np.abs(mu).max() + 0.01)

    with pytest.raises(TypeError):
        # list not supported.
        EfficientCDaR(mu, historical_rets.to_numpy().tolist())

    historical_rets = historical_rets.iloc[:, :-1]
    with pytest.raises(ValueError):
        EfficientCDaR(mu, historical_rets)


def test_parametrization():
    cd = setup_efficient_cdar()
    cd.efficient_risk(0.08)
    cd.efficient_risk(0.07)

    cd = setup_efficient_cdar()
    cd.efficient_return(0.08)
    cd.efficient_return(0.07)
