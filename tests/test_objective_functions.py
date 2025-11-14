import numpy as np
import pandas as pd

from pypfopt import objective_functions
from pypfopt.expected_returns import mean_historical_return, returns_from_prices
from pypfopt.risk_models import sample_cov
from tests.utilities_for_tests import get_data


def test_volatility_dummy():
    w = np.array([0.4, 0.4, 0.2])
    data = np.diag([0.5, 0.8, 0.9])
    test_var = objective_functions.portfolio_variance(w, data)
    np.testing.assert_almost_equal(test_var, 0.244)


def test_volatility():
    df = get_data()
    S = sample_cov(df)
    w = np.array([1 / df.shape[1]] * df.shape[1])
    var = objective_functions.portfolio_variance(w, S)
    np.testing.assert_almost_equal(var, 0.04498224489292057)


def test_portfolio_return_dummy():
    w = np.array([0.3, 0.1, 0.2, 0.25, 0.15])
    e_rets = pd.Series([0.19, 0.08, 0.09, 0.23, 0.17])

    mu = objective_functions.portfolio_return(w, e_rets, negative=False)
    assert isinstance(mu, float)
    assert mu > 0
    np.testing.assert_almost_equal(mu, w.dot(e_rets))
    np.testing.assert_almost_equal(mu, (w * e_rets).sum())


def test_portfolio_return_real():
    df = get_data()
    e_rets = mean_historical_return(df)
    w = np.array([1 / len(e_rets)] * len(e_rets))
    negative_mu = objective_functions.portfolio_return(w, e_rets)
    assert isinstance(negative_mu, float)
    assert negative_mu < 0
    np.testing.assert_almost_equal(negative_mu, -w.dot(e_rets))
    np.testing.assert_almost_equal(negative_mu, -(w * e_rets).sum())
    np.testing.assert_almost_equal(-e_rets.sum() / len(e_rets), negative_mu)


def test_sharpe_ratio():
    df = get_data()
    e_rets = mean_historical_return(df)
    S = sample_cov(df)
    w = np.array([1 / len(e_rets)] * len(e_rets))

    sharpe = objective_functions.sharpe_ratio(w, e_rets, S, risk_free_rate=0.02)
    assert isinstance(sharpe, float)
    assert sharpe < 0

    sigma = np.sqrt(np.dot(w, np.dot(S, w.T)))
    negative_mu = objective_functions.portfolio_return(w, e_rets)
    np.testing.assert_almost_equal(sharpe * sigma - 0.02, negative_mu)

    # Risk free rate increasing should lead to negative Sharpe increasing.
    assert sharpe < objective_functions.sharpe_ratio(w, e_rets, S, risk_free_rate=0.1)


def test_L2_reg_dummy():
    gamma = 2
    w = np.array([0.1, 0.2, 0.3, 0.4])
    L2_reg = objective_functions.L2_reg(w, gamma=gamma)
    np.testing.assert_almost_equal(L2_reg, gamma * np.sum(w * w))


def test_quadratic_utility():
    df = get_data()
    e_rets = mean_historical_return(df)
    S = sample_cov(df)
    w = np.array([1 / len(e_rets)] * len(e_rets))
    utility = objective_functions.quadratic_utility(w, e_rets, S, risk_aversion=3)
    assert isinstance(utility, float)
    assert utility < 0

    mu = objective_functions.portfolio_return(w, e_rets, negative=False)
    variance = objective_functions.portfolio_variance(w, S)
    np.testing.assert_almost_equal(-utility + 3 / 2 * variance, mu)


def test_transaction_costs():
    old_w = np.array([0.1, 0.2, 0.3])
    new_w = np.array([-0.3, 0.1, 0.2])

    k = 0.1
    tx_cost = k * np.abs(old_w - new_w).sum()
    assert tx_cost == objective_functions.transaction_cost(new_w, old_w, k=k)


def test_ex_ante_tracking_error_dummy():
    bm_w = np.ones(5) / 5
    w = np.array([0.4, 0.4, 0, 0, 0])
    S = pd.DataFrame(np.eye(5))

    te = objective_functions.ex_ante_tracking_error(w, S, bm_w)
    np.testing.assert_almost_equal(te, 0.2)


def test_ex_ante_tracking_error():
    df = get_data()
    n_assets = df.shape[1]
    # Equal weight benchmark
    bm_w = np.ones(n_assets) / n_assets
    portfolio_w = np.zeros(n_assets)
    portfolio_w[:5] = 0.2

    S = sample_cov(df)

    te = objective_functions.ex_ante_tracking_error(portfolio_w, S, bm_w)
    np.testing.assert_almost_equal(te, 0.028297778946639436)


def test_ex_post_tracking_error():
    df = get_data()
    rets = returns_from_prices(df).dropna()
    bm_rets = rets.mean(axis=1)
    w = np.ones((len(df.columns),)) / len(df.columns)

    # TE with the mean should be zero
    te = objective_functions.ex_post_tracking_error(w, rets, bm_rets)
    np.testing.assert_almost_equal(te, 0)

    # Should increase
    prev_te = te
    for mult in range(2, 20, 4):
        bm_rets_new = bm_rets * mult
        te = objective_functions.ex_post_tracking_error(w, rets, bm_rets_new)
        assert te > prev_te
        prev_te = te
