import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest
import scipy.optimize as sco

from pypfopt import (
    EfficientFrontier,
    exceptions,
    expected_returns,
    objective_functions,
    risk_models,
)
from tests.utilities_for_tests import (
    get_cov_matrix,
    get_data,
    setup_efficient_frontier,
    simple_ef_weights,
)


def test_data_source():
    df = get_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 20
    assert len(df) == 7126


def test_returns_dataframe():
    df = get_data()
    returns_df = df.pct_change().dropna(how="all")
    assert isinstance(returns_df, pd.DataFrame)
    assert returns_df.shape[1] == 20
    assert len(returns_df) == 7125
    assert not ((returns_df > 1) & returns_df.notnull()).any().any()


def test_ef_example():
    ef = setup_efficient_frontier()
    ef.efficient_return(0.2)
    np.testing.assert_almost_equal(
        ef.portfolio_performance(risk_free_rate=0.02)[0], 0.2
    )


def test_ef_example_weekly():
    df = get_data()
    prices_weekly = df.resample("W").first()
    mu = expected_returns.mean_historical_return(prices_weekly, frequency=52)
    S = risk_models.sample_cov(prices_weekly, frequency=52)
    ef = EfficientFrontier(mu, S)
    ef.efficient_return(0.2)
    np.testing.assert_almost_equal(
        ef.portfolio_performance(risk_free_rate=0.02)[0], 0.2
    )


def test_efficient_frontier_inheritance():
    ef = setup_efficient_frontier()
    assert ef.clean_weights
    assert ef.n_assets
    assert ef.tickers
    assert isinstance(ef._constraints, list)
    assert isinstance(ef._lower_bounds, np.ndarray)
    assert isinstance(ef._upper_bounds, np.ndarray)


def test_efficient_frontier_expected_returns_list():
    """Cover the edge case that the expected_returns param is a list."""
    ef = setup_efficient_frontier()
    ef.min_volatility()
    ef_r = EfficientFrontier(
        expected_returns=ef.expected_returns.tolist(), cov_matrix=ef.cov_matrix
    )
    ef_r.min_volatility()
    np.testing.assert_equal(
        ef.portfolio_performance(risk_free_rate=0.02),
        ef_r.portfolio_performance(risk_free_rate=0.02),
    )


def test_portfolio_performance():
    ef = setup_efficient_frontier()
    with pytest.raises(ValueError):
        ef.portfolio_performance(risk_free_rate=0.02)
    ef.min_volatility()
    perf = ef.portfolio_performance(risk_free_rate=0.02)
    assert isinstance(perf, tuple)
    assert len(perf) == 3
    assert isinstance(perf[0], float)


def test_min_volatility():
    ef = setup_efficient_frontier()
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])

    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.15056821399482578, 0.15915084514118694, 0.8204054077060996),
    )


def test_min_volatility_different_solver():
    ef = setup_efficient_frontier(solver="ECOS")
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    test_performance = (0.150567, 0.159150, 0.820403)
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02), test_performance, atol=1e-5
    )

    ef = setup_efficient_frontier(solver="OSQP")
    w = ef.min_volatility()
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02), test_performance, atol=1e-5
    )

    ef = setup_efficient_frontier(solver="SCS")
    w = ef.min_volatility()
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02), test_performance, atol=1e-3
    )


def test_min_volatility_no_rets():
    # Should work with no rets, see issue #82
    df = get_data()
    S = risk_models.sample_cov(df)
    ef = EfficientFrontier(None, S)
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_almost_equal(
        ef.portfolio_performance(risk_free_rate=0.02)[1], 0.15915084514118694
    )


def test_min_volatility_tx_costs():
    # Baseline
    ef = setup_efficient_frontier()
    ef.min_volatility()
    w1 = ef.weights

    # Pretend we were initally equal weight
    ef = setup_efficient_frontier()
    prev_w = np.array([1 / ef.n_assets] * ef.n_assets)
    ef.add_objective(objective_functions.transaction_cost, w_prev=prev_w)
    ef.min_volatility()
    w2 = ef.weights

    # TX cost should  pull closer to prev portfolio
    assert np.abs(prev_w - w2).sum() < np.abs(prev_w - w1).sum()


def test_min_volatility_short():
    ef = setup_efficient_frontier(weight_bounds=(None, None))
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.1516319319875544, 0.1555915367269669, 0.8460095886741129),
    )

    # Shorting should reduce volatility
    volatility = ef.portfolio_performance(risk_free_rate=0.02)[1]
    ef_long_only = setup_efficient_frontier()
    ef_long_only.min_volatility()
    long_only_volatility = ef_long_only.portfolio_performance()[1]
    assert volatility < long_only_volatility


def test_min_volatility_L2_reg():
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=1)
    weights = ef.min_volatility()
    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in weights.values()])

    ef2 = setup_efficient_frontier()
    ef2.min_volatility()

    # L2_reg should pull close to equal weight
    equal_weight = np.full((ef.n_assets,), 1 / ef.n_assets)
    assert (
        np.abs(equal_weight - ef.weights).sum()
        < np.abs(equal_weight - ef2.weights).sum()
    )

    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.17356099329164965, 0.1955254118258614, 0.785376140408869),
    )


def test_min_volatility_L2_reg_many_values():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    # Count the number of weights more 1%
    initial_number = sum(ef.weights > 0.01)
    for gamma_multiplier in range(1, 10):
        ef = setup_efficient_frontier()
        ef.add_objective(objective_functions.L2_reg, gamma=0.05 * gamma_multiplier)
        ef.min_volatility()
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        new_number = sum(ef.weights > 0.01)
        # Higher gamma should reduce the number of small weights
        assert new_number >= initial_number
        initial_number = new_number


def test_min_volatility_L2_reg_limit_case():
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=1e10)
    ef.min_volatility()
    equal_weights = np.array([1 / ef.n_assets] * ef.n_assets)
    np.testing.assert_array_almost_equal(ef.weights, equal_weights)


def test_min_volatility_L2_reg_increases_vol():
    # L2 reg should reduce the number of small weights
    # but increase in-sample volatility.
    ef_no_reg = setup_efficient_frontier()
    ef_no_reg.min_volatility()
    vol_no_reg = ef_no_reg.portfolio_performance()[1]
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=2)
    ef.min_volatility()
    vol = ef.portfolio_performance(risk_free_rate=0.02)[1]
    assert vol > vol_no_reg


def test_min_volatility_tx_costs_L2_reg():
    ef = setup_efficient_frontier()
    prev_w = np.array([1 / ef.n_assets] * ef.n_assets)
    ef.add_objective(objective_functions.transaction_cost, w_prev=prev_w)
    ef.add_objective(objective_functions.L2_reg)
    ef.min_volatility()

    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.17363446634404042, 0.1959773703677164, 0.7839398296638683),
    )


def test_min_volatility_cvxpy_vs_scipy():
    # cvxpy
    ef = setup_efficient_frontier()
    ef.min_volatility()
    w1 = ef.weights

    # scipy
    args = (ef.cov_matrix,)
    initial_guess = np.array([1 / ef.n_assets] * ef.n_assets)
    result = sco.minimize(
        objective_functions.portfolio_variance,
        x0=initial_guess,
        args=args,
        method="SLSQP",
        bounds=[(0, 1)] * 20,
        constraints=[{"type": "eq", "fun": lambda x: np.sum(x) - 1}],
    )
    w2 = result["x"]

    cvxpy_var = objective_functions.portfolio_variance(w1, ef.cov_matrix)
    scipy_var = objective_functions.portfolio_variance(w2, ef.cov_matrix)
    assert cvxpy_var <= scipy_var


def test_min_volatility_sector_constraints():
    sector_mapper = {
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

    ef = setup_efficient_frontier(weight_bounds=(None, None))
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

    weights = ef.min_volatility()

    for sector in list(set().union(sector_upper, sector_lower)):
        sector_sum = 0
        for t, v in weights.items():
            if sector_mapper[t] == sector:
                sector_sum += v
        assert sector_sum <= sector_upper.get(sector, 1) + 1e-5
        assert sector_sum >= sector_lower.get(sector, 0) - 1e-5


def test_min_volatility_vs_max_sharpe():
    # Test based on issue #75
    expected_returns_daily = pd.Series(
        [0.043622, 0.120588, 0.072331, 0.056586], index=["AGG", "SPY", "GLD", "HYG"]
    )
    covariance_matrix = pd.DataFrame(
        [
            [0.000859, -0.000941, 0.001494, -0.000062],
            [-0.000941, 0.022400, -0.002184, 0.005747],
            [0.001494, -0.002184, 0.011518, -0.000129],
            [-0.000062, 0.005747, -0.000129, 0.002287],
        ],
        index=["AGG", "SPY", "GLD", "HYG"],
        columns=["AGG", "SPY", "GLD", "HYG"],
    )

    ef = EfficientFrontier(expected_returns_daily, covariance_matrix)
    ef.min_volatility()
    vol_min_vol = ef.portfolio_performance(risk_free_rate=0.00)[1]

    ef = EfficientFrontier(expected_returns_daily, covariance_matrix)
    ef.max_sharpe(risk_free_rate=0.00)
    vol_max_sharpe = ef.portfolio_performance(risk_free_rate=0.00)[1]

    assert vol_min_vol < vol_max_sharpe


def test_min_volatility_nonconvex_objective():
    ef = setup_efficient_frontier()
    ef.add_objective(lambda x: cp.sum((x + 1) / (x + 2) ** 2))
    with pytest.raises(exceptions.OptimizationError):
        ef.min_volatility()


def test_min_volatility_nonlinear_constraint():
    ef = setup_efficient_frontier()
    ef.add_constraint(lambda x: (x + 1) / (x + 2) ** 2 <= 0.5)
    with pytest.raises(exceptions.OptimizationError):
        ef.min_volatility()


def test_min_volatility_large_cov_matrix():
    # test that a large covariance matrix will not fail the PSD check in cvxpy
    cov_matrix = get_cov_matrix()
    mean_return = pd.Series([0.1] * len(cov_matrix), index=cov_matrix.index)
    ef = EfficientFrontier(mean_return, cov_matrix, verbose=True)
    ef.min_volatility()


def test_max_returns():
    ef = setup_efficient_frontier()
    #  In unconstrained case, should equal maximal asset return
    max_ret_idx = ef.expected_returns.argmax()
    pf_max_ret = ef._max_return()
    np.testing.assert_almost_equal(ef.expected_returns[max_ret_idx], pf_max_ret)

    # ... and weights should in the max return asset
    test_res = np.zeros(len(ef.tickers))
    test_res[max_ret_idx] = 1
    np.testing.assert_allclose(ef.weights, test_res, atol=1e-5, rtol=1e-5)

    # Max ret should go down when constrained
    ef = setup_efficient_frontier()
    ef.add_constraint(lambda w: w <= 0.2)
    assert ef._max_return() < pf_max_ret


def test_max_sharpe_error():
    ef = setup_efficient_frontier()
    with pytest.raises(ValueError):
        ef.max_sharpe(risk_free_rate="0.02")

    # An unsupported constraint type, which is incidentally meaningless.
    v = cp.Variable((2, 2), PSD=True)
    ef.add_constraint(lambda _: v >> np.zeros((2, 2)))
    with pytest.raises(TypeError):
        ef.max_sharpe(risk_free_rate=0.02)

    with pytest.raises(ValueError):
        ef.max_sharpe(risk_free_rate=max(ef.expected_returns + 0.01))


def test_max_sharpe_risk_free_warning():
    ef = setup_efficient_frontier()
    with pytest.warns(UserWarning):
        ef.max_sharpe(risk_free_rate=0.03)
        ef.portfolio_performance(risk_free_rate=0.02)


def test_max_sharpe_long_only():
    ef = setup_efficient_frontier()
    w = ef.max_sharpe(risk_free_rate=0.02)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])

    np.testing.assert_allclose(
        ef.portfolio_performance(0.02),
        (0.3047768672819914, 0.22165566922402932, 1.2847714127003216),
    )


def test_max_sharpe_long_weight_bounds():
    ef = setup_efficient_frontier(weight_bounds=(0.03, 0.13))
    ef.max_sharpe(risk_free_rate=0.02)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert ef.weights.min() >= 0.03
    assert ef.weights.max() <= 0.13

    bounds = [(0.01, 0.13), (0.02, 0.11)] * 10
    ef = setup_efficient_frontier(weight_bounds=bounds)
    ef.max_sharpe(risk_free_rate=0.02)
    assert (0.01 <= ef.weights[::2]).all() and (ef.weights[::2] <= 0.13).all()
    assert (0.02 <= ef.weights[1::2]).all() and (ef.weights[1::2] <= 0.11).all()


def test_max_sharpe_explicit_bound():
    ef = setup_efficient_frontier()
    ef.add_constraint(lambda w: w[0] >= 0.2)
    ef.add_constraint(lambda w: w[2] == 0.15)
    ef.add_constraint(lambda w: w[3] + w[4] <= 0.10)

    ef.max_sharpe(risk_free_rate=0.02)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert ef.weights[0] >= 0.2 - 1e-5
    np.testing.assert_almost_equal(ef.weights[2], 0.15)
    assert ef.weights[3] + ef.weights[4] <= 0.10 + 1e-5


def test_max_sharpe_short():
    ef = setup_efficient_frontier(weight_bounds=(None, None))
    w = ef.max_sharpe(risk_free_rate=0.02)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.4937195216716211, 0.29516576454651955, 1.6049270564945908),
    )
    sharpe = ef.portfolio_performance(risk_free_rate=0.02)[2]

    ef_long_only = setup_efficient_frontier()
    ef_long_only.max_sharpe()
    long_only_sharpe = ef_long_only.portfolio_performance()[2]

    assert sharpe > long_only_sharpe


def test_max_sharpe_L2_reg():
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=5)

    with pytest.warns(UserWarning) as w:
        weights = ef.max_sharpe(risk_free_rate=0.02)
        assert len(w) == 1

    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in weights.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.2516854357026833, 0.22043282695478603, 1.051047790401043),
    )

    ef2 = setup_efficient_frontier()
    ef2.max_sharpe()

    # L2_reg should pull close to equal weight
    equal_weight = np.full((ef.n_assets,), 1 / ef.n_assets)
    assert (
        np.abs(equal_weight - ef.weights).sum()
        < np.abs(equal_weight - ef2.weights).sum()
    )


def test_max_sharpe_L2_reg_many_values():
    warnings.filterwarnings("ignore")

    ef = setup_efficient_frontier()
    ef.max_sharpe(risk_free_rate=0.02)
    # Count the number of weights more 1%
    initial_number = sum(ef.weights > 0.01)
    for i in range(1, 20, 2):
        ef = setup_efficient_frontier()
        ef.add_objective(objective_functions.L2_reg, gamma=0.05 * i)
        ef.max_sharpe(risk_free_rate=0.02)
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        new_number = sum(ef.weights > 0.01)
        # Higher gamma should reduce the number of small weights
        assert new_number >= initial_number
        initial_number = new_number


def test_max_sharpe_L2_reg_different_gamma():
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=1)
    ef.max_sharpe(risk_free_rate=0.02)

    ef2 = setup_efficient_frontier()
    ef2.add_objective(objective_functions.L2_reg, gamma=0.01)
    ef2.max_sharpe()

    # Higher gamma should pull close to equal weight
    equal_weight = np.array([1 / ef.n_assets] * ef.n_assets)
    assert (
        np.abs(equal_weight - ef.weights).sum()
        < np.abs(equal_weight - ef2.weights).sum()
    )


def test_max_sharpe_L2_reg_reduces_sharpe():
    # L2 reg should reduce the number of small weights at the cost of Sharpe
    ef_no_reg = setup_efficient_frontier()
    ef_no_reg.max_sharpe()
    sharpe_no_reg = ef_no_reg.portfolio_performance()[2]
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=2)
    ef.max_sharpe(risk_free_rate=0.02)
    sharpe = ef.portfolio_performance(risk_free_rate=0.02)[2]
    assert sharpe < sharpe_no_reg


def test_max_sharpe_L2_reg_with_shorts():
    ef_no_reg = setup_efficient_frontier()
    ef_no_reg.max_sharpe()
    initial_number = sum(ef_no_reg.weights > 0.01)

    ef = setup_efficient_frontier(weight_bounds=(None, None))
    ef.add_objective(objective_functions.L2_reg)
    w = ef.max_sharpe(risk_free_rate=0.02)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.2995338981166366, 0.2234696161770517, 1.2508810052063901),
    )
    new_number = sum(ef.weights > 0.01)
    assert new_number >= initial_number


def test_max_sharpe_risk_free_rate():
    ef = setup_efficient_frontier()
    ef.max_sharpe(risk_free_rate=0.02)
    _, _, initial_sharpe = ef.portfolio_performance(risk_free_rate=0.02)
    ef = setup_efficient_frontier()
    ef.max_sharpe(risk_free_rate=0.10)
    _, _, new_sharpe = ef.portfolio_performance(risk_free_rate=0.10)
    assert new_sharpe <= initial_sharpe

    ef = setup_efficient_frontier()
    ef.max_sharpe(risk_free_rate=0)
    _, _, new_sharpe = ef.portfolio_performance(risk_free_rate=0)
    assert new_sharpe >= initial_sharpe


def test_max_sharpe_risk_free_portfolio_performance():
    # Issue #238 - portfolio perf should use the same rf as
    # max_sharpe
    ef = setup_efficient_frontier()
    ef.max_sharpe(risk_free_rate=0.05)
    with pytest.warns(UserWarning):
        res = ef.portfolio_performance(risk_free_rate=0.02)
        res2 = ef.portfolio_performance(risk_free_rate=0.05)
        np.testing.assert_allclose(res, res2)


def test_min_vol_pair_constraint():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    old_sum = ef.weights[:2].sum()
    ef = setup_efficient_frontier()
    ef.add_constraint(lambda w: (w[1] + w[0] <= old_sum / 2))
    ef.min_volatility()
    new_sum = ef.weights[:2].sum()
    assert new_sum <= old_sum / 2 + 1e-4


def test_max_sharpe_pair_constraint():
    ef = setup_efficient_frontier()
    ef.max_sharpe(risk_free_rate=0.02)
    old_sum = ef.weights[:2].sum()

    ef = setup_efficient_frontier()
    ef.add_constraint(lambda w: (w[1] + w[0] <= old_sum / 2))
    ef.max_sharpe(risk_free_rate=0.02)
    new_sum = ef.weights[:2].sum()
    assert new_sum <= old_sum / 2 + 1e-4


def test_max_sharpe_sector_constraints_manual():
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

    ef = setup_efficient_frontier()
    for sector in sector_upper:
        is_sector = [sector_mapper[t] == sector for t in ef.tickers]
        ef.add_constraint(lambda w: cp.sum(w[is_sector]) <= sector_upper[sector])
    for sector in sector_lower:
        is_sector = [sector_mapper[t] == sector for t in ef.tickers]
        ef.add_constraint(lambda w: cp.sum(w[is_sector]) >= sector_lower[sector])

    weights = ef.max_sharpe(risk_free_rate=0.02)

    for sector in list(set().union(sector_upper, sector_lower)):
        sector_sum = 0
        for t, v in weights.items():
            if sector_mapper[t] == sector:
                sector_sum += v
        assert sector_sum <= sector_upper.get(sector, 1) + 1e-5
        assert sector_sum >= sector_lower.get(sector, 0) - 1e-5


def test_max_sharpe_sector_constraints_auto():
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

    ef = setup_efficient_frontier()
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    weights = ef.max_sharpe(risk_free_rate=0.02)

    for sector in list(set().union(sector_upper, sector_lower)):
        sector_sum = 0
        for t, v in weights.items():
            if sector_mapper[t] == sector:
                sector_sum += v
        assert sector_sum <= sector_upper.get(sector, 1) + 1e-5
        assert sector_sum >= sector_lower.get(sector, 0) - 1e-5


def test_efficient_risk_sector_constraints_manual():
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

    ef = setup_efficient_frontier()

    for sector in sector_upper:
        is_sector = [sector_mapper[t] == sector for t in ef.tickers]
        ef.add_constraint(lambda w: cp.sum(w[is_sector]) <= sector_upper[sector])
    for sector in sector_lower:
        is_sector = [sector_mapper[t] == sector for t in ef.tickers]
        ef.add_constraint(lambda w: cp.sum(w[is_sector]) >= sector_lower[sector])

    weights = ef.efficient_risk(0.19)

    for sector in list(set().union(sector_upper, sector_lower)):
        sector_sum = 0
        for t, v in weights.items():
            if sector_mapper[t] == sector:
                sector_sum += v
        assert sector_sum <= sector_upper.get(sector, 1) + 1e-5
        assert sector_sum >= sector_lower.get(sector, 0) - 1e-5


def test_efficient_risk_sector_constraints_auto():
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
    ef = setup_efficient_frontier()
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    weights = ef.efficient_risk(0.19)
    for sector in list(set().union(sector_upper, sector_lower)):
        sector_sum = 0
        for t, v in weights.items():
            if sector_mapper[t] == sector:
                sector_sum += v
        assert sector_sum <= sector_upper.get(sector, 1) + 1e-5
        assert sector_sum >= sector_lower.get(sector, 0) - 1e-5


def test_max_quadratic_utility():
    ef = setup_efficient_frontier()
    w = ef.max_quadratic_utility(risk_aversion=2)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)

    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.3677732711751504, 0.2921342197778279, 1.1904571516463793),
    )

    ret1, var1, _ = ef.portfolio_performance(risk_free_rate=0.02)
    # increasing risk_aversion should lower both vol and return
    ef2 = setup_efficient_frontier()
    ef2.max_quadratic_utility(10)
    ret2, var2, _ = ef2.portfolio_performance()
    assert ret2 < ret1 and var2 < var1


def test_max_quadratic_utility_with_shorts():
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    ef.max_quadratic_utility()
    np.testing.assert_almost_equal(ef.weights.sum(), 1)

    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (1.4170505733098597, 1.0438577623242156, 1.3383533884915872),
    )


def test_max_quadratic_utility_market_neutral():
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    ef.max_quadratic_utility(market_neutral=True)
    np.testing.assert_almost_equal(ef.weights.sum(), 0)
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (1.248936321062371, 1.0219175004907117, 1.2025787996313317),
    )


def test_max_quadratic_utility_limit():
    # in limit of large risk_aversion, this should approach min variance.
    ef = setup_efficient_frontier()
    ef.max_quadratic_utility(risk_aversion=1e10)

    ef2 = setup_efficient_frontier()
    ef2.min_volatility()
    np.testing.assert_array_almost_equal(ef.weights, ef2.weights)


def test_max_quadratic_utility_L2_reg():
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=5)
    weights = ef.max_quadratic_utility()

    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in weights.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.19774277217586125, 0.2104822672707046, 0.8444548535162986),
    )

    ef2 = setup_efficient_frontier()
    ef2.max_quadratic_utility()

    # L2_reg should pull close to equal weight
    equal_weight = np.full((ef.n_assets,), 1 / ef.n_assets)
    assert (
        np.abs(equal_weight - ef.weights).sum()
        < np.abs(equal_weight - ef2.weights).sum()
    )


def test_max_quadratic_utility_error():
    ef = setup_efficient_frontier()
    with pytest.raises(ValueError):
        ef.max_quadratic_utility(0)
    with pytest.raises(ValueError):
        ef.max_quadratic_utility(-1)


def test_efficient_risk():
    ef = setup_efficient_frontier()
    w = ef.efficient_risk(0.19)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.2552422849133517, 0.19, 1.2381172871434818),
        atol=1e-6,
    )


def test_efficient_risk_limit():
    #  In the limit of high target risk, efficient risk should just maximise return
    ef = setup_efficient_frontier()
    ef.efficient_risk(1)
    w = ef.weights

    ef = setup_efficient_frontier()
    ef._max_return(return_value=False)
    w2 = ef.weights
    np.testing.assert_allclose(w, w2, atol=5)


def test_efficient_risk_error():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    min_possible_vol = ef.portfolio_performance(risk_free_rate=0.02)[1]

    ef = setup_efficient_frontier()
    assert ef.efficient_risk(min_possible_vol + 0.01)

    ef = setup_efficient_frontier()
    with pytest.raises(ValueError):
        # This volatility is too low
        ef.efficient_risk(min_possible_vol - 0.01)

    ef = setup_efficient_frontier()
    with pytest.raises(ValueError):
        # This volatility is negative
        ef.efficient_risk(-0.01)


def test_efficient_risk_many_values():
    for target_risk in np.array([0.16, 0.21, 0.30]):
        ef = setup_efficient_frontier()
        ef.efficient_risk(target_risk)
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        volatility = ef.portfolio_performance(risk_free_rate=0.02)[1]
        assert abs(target_risk - volatility) < 1e-5


def test_efficient_risk_short():
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    w = ef.efficient_risk(0.19)

    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.30035471606347336, 0.19, 1.4755511348079207),
        atol=1e-6,
    )
    sharpe = ef.portfolio_performance(risk_free_rate=0.02)[2]

    ef_long_only = setup_efficient_frontier()
    ef_long_only.efficient_risk(0.19)
    long_only_sharpe = ef_long_only.portfolio_performance()[2]

    assert sharpe > long_only_sharpe


def test_efficient_risk_L2_reg():
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=5)
    weights = ef.efficient_risk(0.19)

    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in weights.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.1931352562313653, 0.18999999989010993, 0.9112381912184281),
        rtol=1e-4,
    )

    ef2 = setup_efficient_frontier()
    ef2.efficient_risk(0.19)

    # L2_reg should pull close to equal weight
    equal_weight = np.full((ef.n_assets,), 1 / ef.n_assets)
    assert (
        np.abs(equal_weight - ef.weights).sum()
        < np.abs(equal_weight - ef2.weights).sum()
    )


def test_efficient_risk_market_neutral():
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    w = ef.efficient_risk(0.21, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 0)
    assert (ef.weights < 1).all() and (ef.weights > -1).all()
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.28640632960825885, 0.21, 1.2686015698590967),
        atol=1e-6,
    )
    sharpe = ef.portfolio_performance(risk_free_rate=0.02)[2]

    ef_long_only = setup_efficient_frontier()
    ef_long_only.efficient_risk(0.21)
    long_only_sharpe = ef_long_only.portfolio_performance()[2]
    assert long_only_sharpe > sharpe


def test_efficient_risk_market_neutral_L2_reg():
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    ef.add_objective(objective_functions.L2_reg)

    w = ef.efficient_risk(0.19, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 0)
    assert (ef.weights < 1).all() and (ef.weights > -1).all()

    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.12790320789339854, 0.1175336636355454, 0.9180621496492316),
        atol=1e-6,
    )


def test_efficient_risk_market_neutral_warning():
    ef = setup_efficient_frontier(solver=cp.ECOS)
    with pytest.warns(RuntimeWarning) as w:
        ef.efficient_risk(0.19, market_neutral=True)
        assert len(w) == 1
        assert (
            str(w[0].message)
            == "Market neutrality requires shorting - bounds have been amended"
        )


def test_efficient_return():
    ef = setup_efficient_frontier()
    target_return = 0.25
    w = ef.efficient_return(target_return)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (target_return, 0.18723269942026335, 1.2284179030274036),
        atol=1e-6,
    )


def test_efficient_return_error():
    ef = setup_efficient_frontier()
    max_ret = ef.expected_returns.max()
    with pytest.raises(ValueError):
        # This return is too high
        ef.efficient_return(max_ret + 0.01)


def test_efficient_frontier_error():
    ef = setup_efficient_frontier()
    with pytest.raises(ValueError):
        EfficientFrontier(ef.expected_returns[:-1], ef.cov_matrix)
    with pytest.raises(TypeError):
        EfficientFrontier(0.02, ef.cov_matrix)
    with pytest.raises(ValueError):
        EfficientFrontier(ef.expected_returns, None)
    with pytest.raises(TypeError):
        EfficientFrontier(ef.expected_returns, 0.01)


def test_efficient_return_many_values():
    ef = setup_efficient_frontier(solver=cp.ECOS)
    for target_return in np.arange(0.25, 0.28, 0.01):
        ef.efficient_return(target_return)
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        assert all([i >= 0 for i in ef.weights])
        mean_return = ef.portfolio_performance(risk_free_rate=0.02)[0]
        np.testing.assert_allclose(target_return, mean_return)


def test_efficient_return_short():
    ef = setup_efficient_frontier(weight_bounds=(None, None))
    target_return = 0.25
    weights_sum = 1.0  # Not market neutral, weights must sum to 1.
    w = ef.efficient_return(target_return)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), weights_sum)

    w_expected = simple_ef_weights(
        ef.expected_returns, ef.cov_matrix, target_return, weights_sum
    )
    np.testing.assert_almost_equal(ef.weights, w_expected)

    vol_expected = np.sqrt(
        objective_functions.portfolio_variance(w_expected, ef.cov_matrix)
    )
    sharpe_expected = objective_functions.sharpe_ratio(
        w_expected,
        ef.expected_returns,
        ef.cov_matrix,
        risk_free_rate=0.02,
        negative=False,
    )
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (target_return, vol_expected, sharpe_expected),
    )

    sharpe = ef.portfolio_performance(risk_free_rate=0.02)[2]

    ef_long_only = setup_efficient_frontier()
    ef_long_only.efficient_return(target_return)
    long_only_sharpe = ef_long_only.portfolio_performance(risk_free_rate=0.02)[2]

    assert sharpe > long_only_sharpe


def test_efficient_return_longshort_target():
    mu = pd.Series([-0.15, -0.12, -0.1, -0.05, -0.01, 0.02, 0.03, 0.04, 0.05])
    cov = pd.DataFrame(np.diag([0.2, 0.2, 0.4, 0.3, 0.1, 0.5, 0.2, 0.3, 0.1]))

    ef = EfficientFrontier(mu, cov, weight_bounds=(-1, 1))
    w = ef.efficient_return(target_return=0.08, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 0)

    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.08, 0.16649041068958137, 0.3603811159542937),
        atol=1e-6,
    )


def test_efficient_return_L2_reg():
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=1)
    w = ef.efficient_return(0.25)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.25, 0.20961660883459776, 1.0972412981906703),
    )


def test_efficient_return_market_neutral():
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    w = ef.efficient_return(0.25, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 0)
    assert (ef.weights < 1).all() and (ef.weights > -1).all()
    np.testing.assert_almost_equal(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.25, 0.1833060046337015, 1.2547324920403273),
    )
    sharpe = ef.portfolio_performance(risk_free_rate=0.02)[2]
    ef_long_only = setup_efficient_frontier()
    ef_long_only.efficient_return(0.25)
    long_only_sharpe = ef_long_only.portfolio_performance(risk_free_rate=0.02)[2]
    assert long_only_sharpe < sharpe


def test_efficient_return_market_neutral_unbounded():
    ef = setup_efficient_frontier(weight_bounds=(None, None))
    target_return = 0.25
    weights_sum = 0.0  # Market neutral so weights must sum to 0.0
    w = ef.efficient_return(target_return, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    w_expected = simple_ef_weights(
        ef.expected_returns, ef.cov_matrix, target_return, weights_sum
    )
    np.testing.assert_almost_equal(ef.weights, w_expected)
    vol_expected = np.sqrt(
        objective_functions.portfolio_variance(w_expected, ef.cov_matrix)
    )
    sharpe_expected = objective_functions.sharpe_ratio(
        w_expected,
        ef.expected_returns,
        ef.cov_matrix,
        risk_free_rate=0.02,
        negative=False,
    )
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (target_return, vol_expected, sharpe_expected),
    )
    sharpe = ef.portfolio_performance(risk_free_rate=0.02)[2]
    ef_long_only = setup_efficient_frontier()
    ef_long_only.efficient_return(target_return)
    long_only_sharpe = ef_long_only.portfolio_performance(risk_free_rate=0.02)[2]
    assert long_only_sharpe < sharpe


def test_efficient_return_market_neutral_warning():
    # This fails
    ef = setup_efficient_frontier(solver=cp.ECOS)
    with pytest.warns(RuntimeWarning) as w:
        ef.efficient_return(0.25, market_neutral=True)
        assert len(w) == 1
        assert (
            str(w[0].message)
            == "Market neutrality requires shorting - bounds have been amended"
        )


def test_max_sharpe_semicovariance():
    df = get_data()
    ef = setup_efficient_frontier()
    ef.cov_matrix = risk_models.semicovariance(df, benchmark=0)
    w = ef.max_sharpe(risk_free_rate=0.02)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.2762965426962885, 0.07372667096108301, 3.476307004714425),
    )


def test_max_sharpe_short_semicovariance():
    df = get_data()
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    ef.cov_matrix = risk_models.semicovariance(df, benchmark=0)
    w = ef.max_sharpe(risk_free_rate=0.02)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.42444834528495234, 0.0898263632679403, 4.50255727350929),
    )


def test_min_volatility_shrunk_L2_reg():
    df = get_data()
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg)

    ef.cov_matrix = risk_models.CovarianceShrinkage(df).ledoit_wolf(
        shrinkage_target="constant_correlation"
    )
    w = ef.min_volatility()

    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.17358178582309983, 0.19563960638632416, 0.7850239972361532),
    )


def test_efficient_return_shrunk():
    df = get_data()
    ef = setup_efficient_frontier()
    ef.cov_matrix = risk_models.CovarianceShrinkage(df).ledoit_wolf(
        shrinkage_target="single_factor"
    )
    w = ef.efficient_return(0.22)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.22, 0.08892163937693882, 2.2491713085967766),
    )


def test_max_sharpe_exp_cov():
    df = get_data()
    ef = setup_efficient_frontier()
    ef.cov_matrix = risk_models.exp_cov(df)
    w = ef.max_sharpe(risk_free_rate=0.02)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.33700887443850647, 0.1807332515488447, 1.7540152225548384),
    )


def test_min_volatility_exp_cov_L2_reg():
    df = get_data()
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg)
    ef.cov_matrix = risk_models.exp_cov(df)
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.1829496087575576, 0.17835412793427002, 0.9136295898775636),
    )


def test_efficient_risk_exp_cov_market_neutral():
    df = get_data()
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    ef.cov_matrix = risk_models.exp_cov(df)
    w = ef.efficient_risk(0.19, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 0)
    assert (ef.weights < 1).all() and (ef.weights > -1).all()
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.3934093962620499, 0.18999999989011893, 1.9653126130421081),
        atol=1e-6,
    )
