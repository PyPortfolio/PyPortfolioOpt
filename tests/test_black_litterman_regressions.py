import numpy as np
import pandas as pd
import pytest

from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns


@pytest.mark.parametrize("risk_free_rate", [0.0, 0.04])
@pytest.mark.parametrize("market_prior", [True, False])
def test_neutral_views_recover_market_weights(risk_free_rate, market_prior):
    tickers = ["A", "B", "C"]
    cov = pd.DataFrame(np.diag([0.04, 0.09, 0.16]), index=tickers, columns=tickers)
    caps = pd.Series([500.0, 300.0, 200.0], index=tickers)
    prior = market_implied_prior_returns(caps, 2.0, cov, risk_free_rate)
    model = BlackLittermanModel(
        cov,
        pi="market" if market_prior else prior,
        market_caps=caps,
        absolute_views=prior,
        risk_aversion=2.0,
        risk_free_rate=risk_free_rate,
    )
    pd.testing.assert_series_equal(model.bl_returns(), prior)
    np.testing.assert_allclose(list(model.bl_weights().values()), [0.5, 0.3, 0.2])


@pytest.mark.parametrize("views", [{"B": 0.06}, {"A": 0.04, "B": 0.06, "C": 0.04}])
def test_singular_covariance_uses_asset_space_system(views):
    # A and C are identical assets. Their total weight is 3/5; B has 2/5.
    cov = pd.DataFrame(
        [[0.04, 0.0, 0.04], [0.0, 0.09, 0.0], [0.04, 0.0, 0.04]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    model = BlackLittermanModel(
        cov, pi=np.array([0.04, 0.06, 0.04]), absolute_views=views, risk_aversion=2
    )
    weights = model.bl_weights()
    np.testing.assert_allclose(list(weights.values()), [0.3, 0.4, 0.3], atol=1e-12)
    np.testing.assert_allclose(list(model.optimize().values()), list(weights.values()))


def test_duplicate_exact_views_match_single_view():
    # A singular view-space system is distinct from a singular asset covariance.
    cov = np.diag([0.04, 0.09, 0.16])
    prior = np.array([0.05, 0.06, 0.07])
    single = BlackLittermanModel(
        cov,
        pi=prior,
        P=np.array([[1.0, 0.0, 0.0]]),
        Q=np.array([0.08]),
        omega=np.zeros((1, 1)),
    )
    duplicate = BlackLittermanModel(
        cov,
        pi=prior,
        P=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        Q=np.array([0.08, 0.08]),
        omega=np.zeros((2, 2)),
    )
    np.testing.assert_allclose(duplicate.bl_returns(), single.bl_returns())
    np.testing.assert_allclose(duplicate.bl_cov(), single.bl_cov())
