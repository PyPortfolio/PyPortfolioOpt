"""
Test to demonstrate the bug in global minimum volatility calculation in EfficientFrontier.efficient_risk.
The formula used for global_min_volatility = np.sqrt(1 / np.sum(np.linalg.pinv(cov_matrix))) may be incorrect
for singular covariance matrices or when weight bounds are not taken into account, leading to flawed validation
of target volatility and potential misstatement of portfolio risk.
"""
import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier


def test_global_min_vol_formula():
    """Compare the formula used in efficient_risk with actual minimum volatility via optimization."""
    # Case 1: Simple non‑singular covariance matrix (should match)
    cov = np.array([[1.0, 0.1], [0.1, 1.0]])
    ef = EfficientFrontier(expected_returns=None, cov_matrix=cov, weight_bounds=(-1, 1))
    # Compute global_min_volatility using the internal formula
    global_min_vol = np.sqrt(1 / np.sum(np.linalg.pinv(cov)))
    # Compute actual minimum volatility via optimization
    ef.min_volatility()
    actual_vol = ef.portfolio_performance(verbose=False)[1]
    print(f"Non‑singular case: formula {global_min_vol:.6f}, optimized {actual_vol:.6f}")
    assert np.abs(global_min_vol - actual_vol) < 1e-6, \
        f"Formula disagrees with optimization for non‑singular covariance"
    
    # Case 2: Singular covariance matrix (perfect correlation)
    cov_sing = np.array([[1.0, 1.0], [1.0, 1.0]])
    global_min_vol_sing = np.sqrt(1 / np.sum(np.linalg.pinv(cov_sing)))
    ef_sing = EfficientFrontier(expected_returns=None, cov_matrix=cov_sing, weight_bounds=(-1, 1))
    ef_sing.min_volatility()
    actual_vol_sing = ef_sing.portfolio_performance(verbose=False)[1]
    print(f"Singular case: formula {global_min_vol_sing:.6f}, optimized {actual_vol_sing:.6f}")
    # The true minimum volatility for perfectly correlated assets with shorting allowed is zero.
    # Indeed one can take w = [1, -1] (weights sum to zero) giving zero variance.
    # The formula may not capture this, demonstrating the bug.
    # We accept that the formula may give a non‑zero value, but we highlight the discrepancy.
    if np.abs(global_min_vol_sing - actual_vol_sing) > 1e-6:
        print("  WARNING: formula and optimization differ for singular matrix.")
    
    # Case 3: Realistic singular matrix (rank deficiency) with three assets
    # Construct a covariance matrix where one asset is a linear combination of the others.
    # Use random data to generate a rank‑2 covariance of size 3x3.
    np.random.seed(42)
    returns = np.random.randn(100, 3)
    returns[:, 2] = returns[:, 0] + returns[:, 1]  # third asset is sum of first two
    cov_rank2 = np.cov(returns, rowvar=False)
    global_min_vol_rank2 = np.sqrt(1 / np.sum(np.linalg.pinv(cov_rank2)))
    ef_rank2 = EfficientFrontier(expected_returns=None, cov_matrix=cov_rank2, weight_bounds=(-1, 1))
    ef_rank2.min_volatility()
    actual_vol_rank2 = ef_rank2.portfolio_performance(verbose=False)[1]
    print(f"Rank‑deficient case: formula {global_min_vol_rank2:.6f}, optimized {actual_vol_rank2:.6f}")
    if np.abs(global_min_vol_rank2 - actual_vol_rank2) > 1e-6:
        print("  WARNING: formula and optimization differ for rank‑deficient matrix.")
    
    # Case 4: Impact of weight bounds – the formula ignores bounds, but the actual minimum volatility
    # can be higher when shorting is disallowed. This can cause efficient_risk to accept a target
    # volatility that is actually infeasible, leading to solver failures or misleading results.
    cov = np.array([[0.04, 0.02], [0.02, 0.09]])  # arbitrary
    # Long‑only portfolio (default bounds (0,1))
    ef_long = EfficientFrontier(expected_returns=None, cov_matrix=cov, weight_bounds=(0, 1))
    ef_long.min_volatility()
    actual_vol_long = ef_long.portfolio_performance(verbose=False)[1]
    global_min_vol = np.sqrt(1 / np.sum(np.linalg.pinv(cov)))
    print(f"Long‑only bounds: formula {global_min_vol:.6f}, optimized {actual_vol_long:.6f}")
    # The formula gives the unconstrained minimum, which can be lower than the long‑only minimum.
    # This is not a bug per se, but the validation in efficient_risk should consider bounds.
    if global_min_vol < actual_vol_long - 1e-6:
        print("  WARNING: formula underestimates achievable minimum under long‑only constraints.")
    
    print("\nAll test cases completed. Any warnings indicate potential risks.")


if __name__ == "__main__":
    test_global_min_vol_formula()