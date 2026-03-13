import numpy as np

from pypfopt import risk_models
from tests.utilities_for_tests import get_data, setup_cla


def test_cvxcla_backend_available():
    """Test that cvxcla backend can be enabled successfully."""
    cla_fast = setup_cla(use_cvxcla=True)
    assert cla_fast.use_cvxcla is True
    assert hasattr(cla_fast, "_cvxcla_engine")


def test_cvxcla_max_sharpe_parity():
    """Test that both backends produce equivalent max Sharpe results."""
    # Original implementation
    cla_original = setup_cla(use_cvxcla=False)
    weights_original = cla_original.max_sharpe()
    perf_original = cla_original.portfolio_performance(risk_free_rate=0.02)

    # cvxcla implementation
    cla_fast = setup_cla(use_cvxcla=True)
    weights_fast = cla_fast.max_sharpe()
    perf_fast = cla_fast.portfolio_performance(risk_free_rate=0.02)

    # Check weights and performance are similar
    weights_orig_array = np.array(list(weights_original.values()))
    weights_fast_array = np.array(list(weights_fast.values()))
    np.testing.assert_allclose(
        weights_orig_array, weights_fast_array, rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(perf_original, perf_fast, rtol=1e-4, atol=1e-6)


def test_cvxcla_min_volatility_parity():
    """Test that both backends produce equivalent min volatility results."""
    # Original implementation
    cla_original = setup_cla(use_cvxcla=False)
    weights_original = cla_original.min_volatility()
    perf_original = cla_original.portfolio_performance(risk_free_rate=0.02)

    # cvxcla implementation
    cla_fast = setup_cla(use_cvxcla=True)
    weights_fast = cla_fast.min_volatility()
    perf_fast = cla_fast.portfolio_performance(risk_free_rate=0.02)

    # Check weights and performance are similar
    weights_orig_array = np.array(list(weights_original.values()))
    weights_fast_array = np.array(list(weights_fast.values()))
    np.testing.assert_allclose(
        weights_orig_array, weights_fast_array, rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(perf_original, perf_fast, rtol=1e-4, atol=1e-6)


def test_cvxcla_cov_matrix_update():
    """Test that covariance matrix updates work correctly in cvxcla backend."""
    df = get_data()
    S2 = risk_models.exp_cov(df)

    # Test cvxcla backend with covariance update
    cla_fast = setup_cla(use_cvxcla=True)
    weights_1 = cla_fast.max_sharpe()

    # Update covariance matrix
    cla_fast.cov_matrix = S2.values
    weights_2 = cla_fast.max_sharpe()

    # Results should be different after covariance update
    weights_1_array = np.array(list(weights_1.values()))
    weights_2_array = np.array(list(weights_2.values()))
    assert not np.allclose(weights_1_array, weights_2_array)

    # Compare with original implementation
    cla_original = setup_cla(use_cvxcla=False)
    cla_original.cov_matrix = S2.values
    weights_original = cla_original.max_sharpe()
    weights_orig_array = np.array(list(weights_original.values()))

    # Should match original implementation
    np.testing.assert_allclose(
        weights_2_array, weights_orig_array, rtol=1e-4, atol=1e-6
    )


def test_cvxcla_efficient_frontier():
    """Test that efficient frontier computation produces similar results."""
    # Original implementation
    cla_original = setup_cla(use_cvxcla=False)
    mu_orig, sigma_orig, weights_orig = cla_original.efficient_frontier(points=50)

    # cvxcla implementation
    cla_fast = setup_cla(use_cvxcla=True)
    mu_fast, sigma_fast, weights_fast = cla_fast.efficient_frontier(points=50)

    # Both should produce non-empty frontiers
    assert len(mu_orig) > 0
    assert len(mu_fast) > 0
    assert len(sigma_orig) == len(mu_orig)
    assert len(sigma_fast) == len(mu_fast)
    assert len(weights_orig) == len(mu_orig)
    assert len(weights_fast) == len(mu_fast)

    # Check that frontier endpoints are similar
    min_return_orig, max_return_orig = min(mu_orig), max(mu_orig)
    min_return_fast, max_return_fast = min(mu_fast), max(mu_fast)

    assert abs(min_return_orig - min_return_fast) < 0.01
    assert abs(max_return_orig - max_return_fast) < 0.01

    # Check that minimum volatility points are similar
    min_vol_orig = min(sigma_orig)
    min_vol_fast = min(sigma_fast)
    assert abs(min_vol_orig - min_vol_fast) < 0.01

    # frontier_values should be set for plotting compatibility
    assert cla_fast.frontier_values is not None
