import cvxpy as cp
import numpy as np
import pytest

from pypfopt import (
    EfficientFrontier,
    exceptions,
    expected_returns,
    objective_functions,
    risk_models,
)
from pypfopt.base_optimizer import BaseConvexOptimizer
from tests.utilities_for_tests import get_data, setup_efficient_frontier


def test_custom_convex_equal_weights():
    def new_objective(w):
        return cp.sum(w**2)

    ef = setup_efficient_frontier()
    ef.convex_objective(new_objective)
    np.testing.assert_allclose(ef.weights, np.array([1 / 20] * 20))


def test_custom_convex_additional():
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=1)
    w_co = ef.convex_objective(
        objective_functions.portfolio_variance, cov_matrix=ef.cov_matrix
    )

    # Same as test_min_volatility_L2_reg:
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=1)
    w_mv = ef.min_volatility()
    # Weights from custom convex objective match those from min_volatility where
    # an additional objective is applied to both.
    assert dict(w_co) == dict(w_mv)


def test_custom_convex_abs_exposure():
    ef = setup_efficient_frontier(weight_bounds=(None, None))
    ef.add_constraint(lambda x: cp.norm(x, 1) <= 2)
    ef.convex_objective(
        objective_functions.portfolio_variance,
        cov_matrix=ef.cov_matrix,
        weights_sum_to_one=False,
    )


def test_custom_convex_min_var():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    built_in = ef.weights

    # With custom objective
    ef = setup_efficient_frontier()
    ef.convex_objective(
        objective_functions.portfolio_variance, cov_matrix=ef.cov_matrix
    )
    custom = ef.weights
    np.testing.assert_allclose(built_in, custom, atol=1e-7)


def test_custom_convex_objective_market_neutral_efficient_risk():
    target_risk = 0.19
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    ef.efficient_risk(target_risk, market_neutral=True)
    built_in = ef.weights

    # Recreate the market-neutral efficient_risk optimizer using this API
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    ef.add_constraint(lambda x: cp.sum(x) == 0)
    ef.add_constraint(lambda x: cp.quad_form(x, ef.cov_matrix) <= target_risk**2)
    ef.convex_objective(lambda x: -x @ ef.expected_returns, weights_sum_to_one=False)
    custom = ef.weights

    np.testing.assert_allclose(built_in, custom, atol=1e-7)


def test_convex_sharpe_raises_error():
    # With custom objective
    with pytest.raises(exceptions.OptimizationError):
        ef = setup_efficient_frontier()
        ef.convex_objective(
            objective_functions.sharpe_ratio,
            expected_returns=ef.expected_returns,
            cov_matrix=ef.cov_matrix,
        )


def test_custom_tracking_error():
    df = get_data()
    historical_rets = expected_returns.returns_from_prices(df).dropna()
    benchmark_rets = historical_rets["AAPL"]
    historical_rets = historical_rets.drop("AAPL", axis=1)
    S = risk_models.sample_cov(historical_rets, returns_data=True)

    opt = BaseConvexOptimizer(
        n_assets=len(historical_rets.columns),
        tickers=list(historical_rets.columns),
        weight_bounds=(0, 1),
    )

    opt.convex_objective(
        objective_functions.ex_post_tracking_error,
        historic_returns=historical_rets,
        benchmark_returns=benchmark_rets,
    )
    w = opt.clean_weights()

    ef = EfficientFrontier(None, S)
    ef.convex_objective(
        objective_functions.ex_post_tracking_error,
        historic_returns=historical_rets,
        benchmark_returns=benchmark_rets,
    )
    w2 = ef.clean_weights()

    assert w == w2


def test_custom_convex_logarithmic_barrier():
    def logarithmic_barrier(w, cov_matrix, k=0.1):
        log_sum = cp.sum(cp.log(w))
        var = cp.quad_form(w, cov_matrix)
        return var - k * log_sum

    # 60 Years of Portfolio Optimization, Kolm et al (2014)
    ef = setup_efficient_frontier()
    w = ef.convex_objective(logarithmic_barrier, cov_matrix=ef.cov_matrix)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)

    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.17261881638711316, 0.21100848889958182, 0.7232828270702603),
        rtol=1e-4,
    )


def test_custom_convex_deviation_risk_parity_error():
    def deviation_risk_parity(w, cov_matrix):
        n = cov_matrix.shape[0]
        rp = (w @ (cov_matrix @ w)) / cp.quad_form(w, cov_matrix)
        return cp.sum_squares(rp - 1 / n)

    # 60 Years of Portfolio Optimization, Kolm et al (2014)
    ef = setup_efficient_frontier()
    with pytest.raises(exceptions.OptimizationError):
        ef.convex_objective(deviation_risk_parity, cov_matrix=ef.cov_matrix)


def test_custom_convex_kelly():
    def kelly_objective(w, e_returns, cov_matrix, k=3):
        variance = cp.quad_form(w, cov_matrix)
        objective = variance * 0.5 * k - w @ e_returns
        return objective

    lb = 0.01
    ub = 0.3
    ef = setup_efficient_frontier(weight_bounds=(lb, ub))
    weights = ef.convex_objective(
        kelly_objective, e_returns=ef.expected_returns, cov_matrix=ef.cov_matrix
    )

    for w in weights.values():
        assert w >= lb - 1e-8 and w <= ub + 1e-8


def test_custom_nonconvex_min_var():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    original_vol = ef.portfolio_performance()[1]

    # With custom objective
    ef = setup_efficient_frontier()
    ef.nonconvex_objective(
        objective_functions.portfolio_variance, objective_args=ef.cov_matrix
    )
    custom_vol = ef.portfolio_performance()[1]

    # Scipy should be close but not as good for this simple objective
    np.testing.assert_almost_equal(custom_vol, original_vol, decimal=5)
    assert original_vol < custom_vol


def test_custom_nonconvex_logarithmic_barrier():
    def logarithmic_barrier(weights, cov_matrix, k=0.1):
        log_sum = np.sum(np.log(weights))
        portfolio_volatility = np.dot(weights.T, np.dot(cov_matrix, weights))
        return portfolio_volatility - k * log_sum

    # 60 Years of Portfolio Optimization, Kolm et al (2014)
    ef = setup_efficient_frontier()
    w = ef.nonconvex_objective(logarithmic_barrier, objective_args=(ef.cov_matrix, 0.2))
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)


def test_custom_nonconvex_deviation_risk_parity_1():
    def deviation_risk_parity(w, cov_matrix):
        diff = w * np.dot(cov_matrix, w) - (w * np.dot(cov_matrix, w)).reshape(-1, 1)
        return (diff**2).sum().sum()

    # 60 Years of Portfolio Optimization, Kolm et al (2014) - first definition
    ef = setup_efficient_frontier()
    w = ef.nonconvex_objective(deviation_risk_parity, ef.cov_matrix)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)


def test_custom_nonconvex_deviation_risk_parity_2():
    def deviation_risk_parity(w, cov_matrix):
        n = cov_matrix.shape[0]
        rp = (w * (cov_matrix @ w)) / cp.quad_form(w, cov_matrix)
        return cp.sum_squares(rp - 1 / n).value

    # 60 Years of Portfolio Optimization, Kolm et al (2014) - second definition
    ef = setup_efficient_frontier()
    w = ef.nonconvex_objective(deviation_risk_parity, ef.cov_matrix)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)


def test_custom_nonconvex_sharpe():
    ef = setup_efficient_frontier()
    w1 = ef.nonconvex_objective(
        objective_functions.sharpe_ratio,
        objective_args=(ef.expected_returns, ef.cov_matrix, 0.02),
        weights_sum_to_one=True,
    )
    p1 = ef.portfolio_performance(risk_free_rate=0.02)

    ef = setup_efficient_frontier()
    w2 = ef.max_sharpe(risk_free_rate=0.02)
    p2 = ef.portfolio_performance(risk_free_rate=0.02)

    np.testing.assert_allclose(list(w1.values()), list(w2.values()), atol=2e-4)
    assert p2[2] >= p1[2]

    ef = setup_efficient_frontier()
    min_weight, max_weight = 0.01, 0.3
    w3 = ef.nonconvex_objective(
        objective_functions.sharpe_ratio,
        objective_args=(ef.expected_returns, ef.cov_matrix),
        constraints=[
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w: w - min_weight},
            {"type": "ineq", "fun": lambda w: max_weight - w},
        ],
    )
    for w in w3.values():
        assert w >= min_weight - 1e-8 and w <= max_weight + 1e-8


def test_custom_nonconvex_kelly():
    def kelly_objective(w, e_returns, cov_matrix, k=3):
        variance = np.dot(w.T, np.dot(cov_matrix, w))
        objective = variance * 0.5 * k - np.dot(w, e_returns)
        return objective

    lower_bounds, upper_bounds = 0.01, 0.3

    ef = setup_efficient_frontier()
    weights = ef.nonconvex_objective(
        kelly_objective,
        objective_args=(ef.expected_returns, ef.cov_matrix, 3),
        constraints=[
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w: w - lower_bounds},
            {"type": "ineq", "fun": lambda w: upper_bounds - w},
        ],
    )

    for w in weights.values():
        assert w >= lower_bounds - 1e-8 and w <= upper_bounds + 1e-8


def test_custom_nonconvex_utility_objective():
    def utility_obj(weights, mu, cov_matrix, k=1):
        return -weights.dot(mu) + k * np.dot(weights.T, np.dot(cov_matrix, weights))

    ef = setup_efficient_frontier()
    w = ef.nonconvex_objective(
        utility_obj, objective_args=(ef.expected_returns, ef.cov_matrix, 1)
    )
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    vol1 = ef.portfolio_performance()[1]

    # If we increase k, volatility should decrease
    w = ef.nonconvex_objective(
        utility_obj, objective_args=(ef.expected_returns, ef.cov_matrix, 3)
    )
    vol2 = ef.portfolio_performance()[1]
    assert vol2 < vol1


def test_custom_nonconvex_objective_market_neutral_efficient_risk():
    # Recreate the market-neutral efficient_risk optimizer using this API
    target_risk = 0.19
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))

    weight_constr = {"type": "eq", "fun": lambda w: np.sum(w)}
    risk_constr = {
        "type": "eq",
        "fun": lambda w: target_risk**2 - np.dot(w.T, np.dot(ef.cov_matrix, w)),
    }
    constraints = [weight_constr, risk_constr]

    ef.nonconvex_objective(
        lambda w, mu: -w.T.dot(mu),
        objective_args=(ef.expected_returns),
        weights_sum_to_one=False,
        constraints=constraints,
    )
    np.testing.assert_allclose(
        ef.portfolio_performance(risk_free_rate=0.02),
        (0.2591296227818582, target_risk, 1.258574109251818),
        atol=1e-6,
    )
