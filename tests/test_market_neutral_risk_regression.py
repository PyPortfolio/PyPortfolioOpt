import numpy as np
import pytest

from pypfopt import EfficientFrontier, exceptions


@pytest.mark.parametrize("target", [0.01, 0.05, 0.1])
def test_market_neutral_risk_below_fully_invested_minimum(target):
    # With diagonal 0.04 covariance and weights (a, -a), risk=sqrt(0.08)*a.
    ef = EfficientFrontier([0.1, 0.0], np.eye(2) * 0.04, weight_bounds=(-1, 1))
    weights = np.array(list(ef.efficient_risk(target, market_neutral=True).values()))
    a = target / np.sqrt(0.08)
    np.testing.assert_allclose(weights, [a, -a], atol=2e-5)


def test_market_neutral_risk_updates_parameter():
    ef = EfficientFrontier([0.1, 0.0], np.eye(2) * 0.04, weight_bounds=(-1, 1))
    for target in (0.05, 0.02):
        weights = np.array(
            list(ef.efficient_risk(target, market_neutral=True).values())
        )
        np.testing.assert_allclose(
            weights, [target / np.sqrt(0.08), -target / np.sqrt(0.08)], atol=2e-5
        )


def test_fully_invested_low_risk_is_still_rejected():
    ef = EfficientFrontier([0.1, 0.0], np.eye(2) * 0.04)
    with pytest.raises(ValueError, match="minimum volatility"):
        ef.efficient_risk(0.05)


def test_market_neutral_infeasible_extra_constraint_is_not_accepted():
    ef = EfficientFrontier([0.1, 0.0], np.eye(2) * 0.04, weight_bounds=(-1, 1))
    ef.add_constraint(lambda w: w[0] >= 0.5)
    with pytest.raises(exceptions.OptimizationError):
        ef.efficient_risk(0.05, market_neutral=True)


def test_market_neutral_tight_bounds_allow_risk_below_target():
    ef = EfficientFrontier([0.1, 0.0], np.eye(2) * 0.04, weight_bounds=(-0.02, 0.02))
    weights = np.array(list(ef.efficient_risk(0.05, market_neutral=True).values()))
    np.testing.assert_allclose(weights, [0.02, -0.02], atol=2e-5)
    # efficient_risk imposes an upper bound, not an equality constraint.
    assert np.sqrt(weights @ (np.eye(2) * 0.04) @ weights) < 0.05


def test_market_neutral_preserves_warning_and_shorting_bound_amendment():
    ef = EfficientFrontier([0.1, 0.0], np.eye(2) * 0.04, weight_bounds=(0.1, 1))
    # This is the existing API policy, not a new relaxation introduced here.
    with pytest.warns(RuntimeWarning, match="Market neutrality requires shorting"):
        weights = np.array(list(ef.efficient_risk(0.05, market_neutral=True).values()))
    a = 0.05 / np.sqrt(0.08)
    np.testing.assert_allclose(weights, [a, -a], atol=2e-5)
