import numpy as np
import pytest

from pypfopt import risk_models
from pypfopt.cla import CLA
from tests.utilities_for_tests import get_data, setup_cla


def test_portfolio_performance():
    cla = setup_cla()
    with pytest.raises(ValueError):
        cla.portfolio_performance()
    cla.max_sharpe()
    assert cla.portfolio_performance()


def test_cla_inheritance():
    cla = setup_cla()
    assert cla.clean_weights
    assert cla.set_weights


def test_cla_max_sharpe_long_only():
    cla = setup_cla()
    w = cla.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)

    np.testing.assert_allclose(
        cla.portfolio_performance(risk_free_rate=0.02),
        (0.2994470912768992, 0.21764331657015668, 1.283968171780824),
    )


def test_cla_max_sharpe_short():
    cla = setup_cla(weight_bounds=(-1, 1))
    w = cla.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)
    np.testing.assert_allclose(
        cla.portfolio_performance(risk_free_rate=0.02),
        (0.44859872371106785, 0.26762066559448255, 1.601515797589826),
    )
    sharpe = cla.portfolio_performance(risk_free_rate=0.02)[2]

    cla_long_only = setup_cla()
    cla_long_only.max_sharpe()
    long_only_sharpe = cla_long_only.portfolio_performance(risk_free_rate=0.02)[2]

    assert sharpe > long_only_sharpe


def test_cla_max_sharpe_rejects_singular_covariance():
    mu = np.array([0.1, 0.2])
    cov = np.array([[0.01, 0.01], [0.01, 0.01]])

    cla = CLA(mu, cov)

    with pytest.raises(ValueError, match="positive definite covariance matrix"):
        cla.max_sharpe()


def test_cla_custom_bounds():
    bounds = [(0.01, 0.13), (0.02, 0.11)] * 10
    cla = setup_cla(weight_bounds=bounds)
    df = get_data()
    cla.cov_matrix = risk_models.exp_cov(df).values
    w = cla.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)
    assert (0.01 <= cla.weights[::2]).all() and (cla.weights[::2] <= 0.13).all()
    assert (0.02 <= cla.weights[1::2]).all() and (cla.weights[1::2] <= 0.11).all()
    # Test polymorphism of the weight_bounds param.
    bounds2 = ([bounds[0][0], bounds[1][0]] * 10, [bounds[0][1], bounds[1][1]] * 10)
    cla2 = setup_cla(weight_bounds=bounds2)
    cla2.cov_matrix = risk_models.exp_cov(df).values
    w2 = cla2.min_volatility()
    assert dict(w2) == dict(w)


def test_cla_min_volatility():
    cla = setup_cla()
    w = cla.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)
    np.testing.assert_allclose(
        cla.portfolio_performance(risk_free_rate=0.02),
        (0.1505682139948257, 0.15915084514118688, 0.8204054077060994),
    )


def test_cla_error():
    cla = setup_cla()
    w = cla.min_volatility()
    with pytest.raises(NotImplementedError):
        cla.set_weights(w)


def test_cla_two_assets():
    mu = np.array([[0.02569294], [0.16203987]])
    cov = np.array([[0.0012765, -0.00212724], [-0.00212724, 0.01616983]])
    assert CLA(mu, cov)


def test_cla_max_sharpe_semicovariance():
    df = get_data()
    cla = setup_cla()
    cla.cov_matrix = risk_models.semicovariance(df, benchmark=0).values
    w = cla.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)
    np.testing.assert_allclose(
        cla.portfolio_performance(risk_free_rate=0.02),
        (0.2721798377099145, 0.07258537193305141, 3.474251505420551),
        atol=1e-4,
        rtol=1e-4,
    )


def test_cla_max_sharpe_exp_cov():
    df = get_data()
    cla = setup_cla()
    cla.cov_matrix = risk_models.exp_cov(df).values
    w = cla.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)
    np.testing.assert_allclose(
        cla.portfolio_performance(risk_free_rate=0.02),
        (0.32971891062187103, 0.17670121760851704, 1.7527831149871063),
    )


def test_cla_min_volatility_exp_cov_short():
    cla = setup_cla(weight_bounds=(-1, 1))
    df = get_data()
    cla.cov_matrix = risk_models.exp_cov(df).values
    w = cla.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)
    np.testing.assert_allclose(
        cla.portfolio_performance(risk_free_rate=0.02),
        (0.23215576461823062, 0.1325959061825329, 1.6000174569958052),
    )


def test_cla_efficient_frontier():
    cla = setup_cla()

    cla.efficient_frontier()

    mu, sigma, weights = cla.efficient_frontier()
    assert len(mu) == len(sigma) and len(sigma) == len(weights)
    # higher return = higher risk
    assert sigma[-1] < sigma[0] and mu[-1] < mu[0]
    assert weights[0].shape == (20, 1)
