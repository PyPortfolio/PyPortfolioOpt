"""Tests for ERCOpt (Equal Risk Contribution / Risk Parity) portfolio.

References
----------
Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of Equally
Weighted Risk Contribution Portfolios. Journal of Portfolio Management, 36(4), 60-70.

Spinu, F. (2013). An Algorithm for Computing Risk Parity Weights. SSRN working paper.
"""

import collections

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pypfopt import ERCOpt
from pypfopt.hierarchical_portfolio import _erc_weights_ccd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _make_returns(n_assets=5, n_obs=252, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((n_obs, n_assets)) * 0.01,
        columns=[f"A{i}" for i in range(n_assets)],
    )


def _make_cov(n_assets=5, seed=0):
    """PD covariance matrix with varied volatilities."""
    ret = _make_returns(n_assets=n_assets, seed=seed)
    return ret.cov()


def _make_cov_with_neg_corr():
    """Covariance matrix that has negative off-diagonal entries."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 4))
    cov_arr = A @ A.T + np.eye(4)
    cols = list("WXYZ")
    return pd.DataFrame(cov_arr, index=cols, columns=cols)


# ---------------------------------------------------------------------------
# _erc_weights_ccd unit tests
# ---------------------------------------------------------------------------


class TestERCWeightsCCD:
    def test_single_asset(self):
        w = _erc_weights_ccd(np.array([[0.04]]))
        assert_allclose(w, [1.0])

    def test_sum_to_one(self):
        cov = np.array([[0.01, 0.0], [0.0, 0.09]])
        w = _erc_weights_ccd(cov)
        assert_allclose(w.sum(), 1.0, atol=1e-12)

    def test_diagonal_analytic_solution(self):
        """For diagonal Σ, ERC gives w_i ∝ 1/σ_i (inverse-vol)."""
        vols = np.array([0.1, 0.3])
        cov = np.diag(vols**2)
        w = _erc_weights_ccd(cov)
        expected = (1 / vols) / (1 / vols).sum()
        assert_allclose(w, expected, atol=1e-8)

    def test_equal_risk_contributions_diagonal(self):
        cov = np.diag([0.01, 0.04, 0.09])
        w = _erc_weights_ccd(cov)
        rc = w * (cov @ w)
        assert_allclose(rc, rc.mean() * np.ones(3), rtol=1e-6)

    def test_equal_risk_contributions_negative_correlation(self):
        """Must converge to correct ERC even with negative off-diagonal entries."""
        rng = np.random.default_rng(1)
        A = rng.standard_normal((4, 4))
        cov = A @ A.T + np.eye(4)
        w = _erc_weights_ccd(cov)
        rc = w * (cov @ w)
        assert_allclose(rc, rc.mean() * np.ones(4), rtol=1e-4)

    def test_less_volatile_asset_gets_higher_weight(self):
        cov = np.array([[0.01, 0.0], [0.0, 0.09]])
        w = _erc_weights_ccd(cov)
        assert w[0] > w[1]  # asset 0 has lower variance → higher ERC weight

    def test_nonnegative_weights(self):
        cov = _make_cov().values
        w = _erc_weights_ccd(cov)
        assert np.all(w >= 0)


# ---------------------------------------------------------------------------
# ERCOpt constructor
# ---------------------------------------------------------------------------


class TestERCOptConstructor:
    def test_from_returns(self):
        ret = _make_returns()
        erc = ERCOpt(returns=ret)
        assert erc.n_assets == 5
        assert erc.tickers == list(ret.columns)

    def test_from_cov_matrix(self):
        cov = _make_cov()
        erc = ERCOpt(cov_matrix=cov)
        assert erc.n_assets == 5
        assert erc.tickers == list(cov.columns)

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="Either returns or cov_matrix"):
            ERCOpt()

    def test_returns_not_dataframe_raises(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            ERCOpt(returns=np.eye(5))

    def test_importable_from_pypfopt(self):
        from pypfopt import ERCOpt as E  # noqa: F401
        assert callable(E)

    def test_in_all(self):
        import pypfopt
        assert "ERCOpt" in pypfopt.__all__


# ---------------------------------------------------------------------------
# ERCOpt.optimize()
# ---------------------------------------------------------------------------


class TestERCOptOptimize:
    def test_returns_ordered_dict(self):
        ret = _make_returns()
        erc = ERCOpt(returns=ret)
        w = erc.optimize()
        assert isinstance(w, collections.OrderedDict)

    def test_weights_sum_to_one(self):
        ret = _make_returns()
        w = ERCOpt(returns=ret).optimize()
        assert_allclose(sum(w.values()), 1.0, atol=1e-12)

    def test_all_weights_nonneg(self):
        ret = _make_returns()
        w = ERCOpt(returns=ret).optimize()
        assert all(v >= 0 for v in w.values())

    def test_equal_risk_contributions(self):
        """Core property: each asset contributes 1/n of total variance."""
        ret = _make_returns(n_assets=5)
        erc = ERCOpt(returns=ret)
        w = erc.optimize()
        cov = ret.cov().values
        w_arr = np.array(list(w.values()))
        rc = w_arr * (cov @ w_arr)
        assert_allclose(rc, rc.mean() * np.ones(len(rc)), rtol=1e-6)

    def test_equal_risk_contributions_negative_corr(self):
        """ERC must hold even for cov with negative off-diagonal entries."""
        cov = _make_cov_with_neg_corr()
        erc = ERCOpt(cov_matrix=cov)
        w = erc.optimize()
        cov_arr = cov.values
        w_arr = np.array(list(w.values()))
        rc = w_arr * (cov_arr @ w_arr)
        assert_allclose(rc, rc.mean() * np.ones(len(rc)), rtol=1e-4)

    def test_tickers_match(self):
        ret = _make_returns(n_assets=4)
        erc = ERCOpt(returns=ret)
        w = erc.optimize()
        assert list(w.keys()) == list(ret.columns)

    def test_from_cov_same_as_from_returns(self):
        """Passing cov_matrix=returns.cov() gives the same weights as passing returns."""
        ret = _make_returns()
        w_ret = ERCOpt(returns=ret).optimize()
        w_cov = ERCOpt(cov_matrix=ret.cov()).optimize()
        assert_allclose(
            list(w_ret.values()), list(w_cov.values()), atol=1e-10
        )

    def test_less_volatile_asset_heavier(self):
        """In a diagonal covariance, the least-volatile asset gets most weight."""
        vols = np.array([0.05, 0.10, 0.20, 0.30])
        cov_arr = np.diag(vols**2)
        tickers = ["L", "M", "H", "V"]
        cov_df = pd.DataFrame(cov_arr, index=tickers, columns=tickers)
        w = ERCOpt(cov_matrix=cov_df).optimize()
        w_vals = np.array(list(w.values()))
        assert w_vals[0] > w_vals[1] > w_vals[2] > w_vals[3]

    def test_identical_assets_get_equal_weight(self):
        """When all assets are identical, ERC gives 1/n to each."""
        n = 4
        cov_arr = np.full((n, n), 0.02) + np.eye(n) * 0.01  # σ² = 0.03, ρ = 2/3
        tickers = [f"X{i}" for i in range(n)]
        cov_df = pd.DataFrame(cov_arr, index=tickers, columns=tickers)
        w = ERCOpt(cov_matrix=cov_df).optimize()
        assert_allclose(list(w.values()), [1 / n] * n, atol=1e-8)

    def test_two_assets_analytic(self):
        """For 2 assets with zero correlation, ERC = inverse-vol weights."""
        vols = np.array([0.10, 0.25])
        cov_arr = np.diag(vols**2)
        cov_df = pd.DataFrame(cov_arr, index=["A", "B"], columns=["A", "B"])
        w = ERCOpt(cov_matrix=cov_df).optimize()
        expected = (1 / vols) / (1 / vols).sum()
        assert_allclose([w["A"], w["B"]], expected, atol=1e-8)

    def test_self_weights_attribute_set(self):
        ret = _make_returns()
        erc = ERCOpt(returns=ret)
        erc.optimize()
        assert erc.weights is not None
        assert len(erc.weights) == erc.n_assets

    def test_clean_weights_runs(self):
        ret = _make_returns()
        erc = ERCOpt(returns=ret)
        erc.optimize()
        cw = erc.clean_weights()
        assert isinstance(cw, dict)

    def test_large_portfolio(self):
        """Optimize a 20-asset portfolio."""
        ret = _make_returns(n_assets=20, n_obs=500)
        w = ERCOpt(returns=ret).optimize()
        assert len(w) == 20
        assert_allclose(sum(w.values()), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# ERCOpt.portfolio_performance()
# ---------------------------------------------------------------------------


class TestERCOptPerformance:
    def test_performance_from_returns(self):
        ret = _make_returns()
        erc = ERCOpt(returns=ret)
        erc.optimize()
        mu, vol, sr = erc.portfolio_performance()
        assert np.isfinite(mu)
        assert vol > 0
        assert np.isfinite(sr)

    def test_performance_from_cov_no_mu(self):
        cov = _make_cov()
        erc = ERCOpt(cov_matrix=cov)
        erc.optimize()
        mu, vol, sr = erc.portfolio_performance()
        assert mu is None
        assert vol > 0

    def test_performance_before_optimize_raises(self):
        ret = _make_returns()
        erc = ERCOpt(returns=ret)
        with pytest.raises(ValueError):
            erc.portfolio_performance()
