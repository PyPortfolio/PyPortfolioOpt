"""
Tests for HERCOpt (Hierarchical Equal Risk Contribution) and
NCOpt (Nested Cluster Optimization).
"""
import collections

import numpy as np
import pandas as pd
import pytest

from pypfopt import HERCOpt, HRPOpt, NCOpt, CovarianceShrinkage
from pypfopt.hierarchical_portfolio import _erc_weights_ccd, _min_var_weights
from tests.utilities_for_tests import get_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_returns():
    df = get_data()
    return df.pct_change().dropna(how="all")


def get_cov():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    return returns.cov()


# ---------------------------------------------------------------------------
# _erc_weights_ccd
# ---------------------------------------------------------------------------

class TestERCWeights:

    def test_single_asset(self):
        cov = np.array([[0.04]])
        w = _erc_weights_ccd(cov)
        np.testing.assert_allclose(w, [1.0])

    def test_two_uncorrelated_equal_vol(self):
        """Equal volatility, no correlation → equal ERC weights."""
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        w = _erc_weights_ccd(cov)
        np.testing.assert_allclose(w, [0.5, 0.5], atol=1e-6)

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(42)
        A = rng.normal(size=(5, 5))
        cov = A @ A.T + np.eye(5) * 0.1
        w = _erc_weights_ccd(cov)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-8)

    def test_weights_non_negative(self):
        rng = np.random.default_rng(7)
        A = rng.normal(size=(6, 6))
        cov = A @ A.T + np.eye(6) * 0.5
        w = _erc_weights_ccd(cov)
        assert np.all(w >= -1e-8)

    def test_equal_risk_contributions(self):
        """All risk contributions should be equal for ERC portfolio."""
        rng = np.random.default_rng(1)
        A = rng.normal(size=(4, 4))
        cov = A @ A.T + np.eye(4)
        w = _erc_weights_ccd(cov)
        rc = w * (cov @ w)
        # All RC equal
        np.testing.assert_allclose(rc, rc.mean() * np.ones(4), rtol=1e-4)

    def test_lower_vol_asset_gets_higher_weight(self):
        """The less volatile asset should receive more weight under ERC."""
        cov = np.array([[0.01, 0.0], [0.0, 0.09]])  # vols 0.1 and 0.3
        w = _erc_weights_ccd(cov)
        assert w[0] > w[1]  # asset 0 is less risky → higher ERC weight


# ---------------------------------------------------------------------------
# _min_var_weights
# ---------------------------------------------------------------------------

class TestMinVarWeights:

    def test_single_asset(self):
        w = _min_var_weights(np.array([[0.04]]))
        np.testing.assert_allclose(w, [1.0])

    def test_sum_to_one(self):
        rng = np.random.default_rng(3)
        A = rng.normal(size=(5, 5))
        cov = A @ A.T + np.eye(5)
        w = _min_var_weights(cov)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-8)

    def test_non_negative(self):
        rng = np.random.default_rng(11)
        A = rng.normal(size=(4, 4))
        cov = A @ A.T + np.eye(4)
        w = _min_var_weights(cov)
        assert np.all(w >= -1e-8)

    def test_known_two_asset(self):
        """For two uncorrelated assets, min-var = inverse variance weights."""
        cov = np.array([[1.0, 0.0], [0.0, 4.0]])  # sigmas 1 and 2
        w = _min_var_weights(cov)
        expected = np.array([4.0, 1.0]) / 5.0  # inv-var: [1/1, 1/4] / sum
        np.testing.assert_allclose(w, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# HERCOpt — instantiation errors
# ---------------------------------------------------------------------------

class TestHERCInstantiation:

    def test_no_args_raises(self):
        with pytest.raises(ValueError):
            HERCOpt()

    def test_returns_not_dataframe_raises(self):
        returns_np = get_returns().values
        with pytest.raises(TypeError):
            HERCOpt(returns=returns_np)

    def test_invalid_linkage_raises(self):
        returns = get_returns()
        herc = HERCOpt(returns=returns)
        with pytest.raises(ValueError):
            herc.optimize(linkage_method="invalid_method")

    def test_from_returns(self):
        returns = get_returns()
        herc = HERCOpt(returns=returns)
        assert herc.n_assets == len(returns.columns)

    def test_from_cov_matrix(self):
        S = CovarianceShrinkage(get_data()).ledoit_wolf()
        herc = HERCOpt(cov_matrix=S)
        assert herc.n_assets == S.shape[0]


# ---------------------------------------------------------------------------
# HERCOpt — portfolio weights
# ---------------------------------------------------------------------------

class TestHERCPortfolio:

    @pytest.fixture(autouse=True)
    def setup(self):
        returns = get_returns()
        self.herc = HERCOpt(returns=returns)
        self.weights = self.herc.optimize(linkage_method="ward")

    def test_weights_is_ordered_dict(self):
        assert isinstance(self.weights, (dict, collections.OrderedDict))

    def test_weights_sum_to_one(self):
        np.testing.assert_allclose(sum(self.weights.values()), 1.0, atol=1e-8)

    def test_weights_non_negative(self):
        assert all(v >= -1e-8 for v in self.weights.values())

    def test_all_assets_in_weights(self):
        returns = get_returns()
        assert set(self.weights.keys()) == set(returns.columns)

    def test_clusters_set_after_optimize(self):
        assert self.herc.clusters is not None

    def test_portfolio_performance_returns_tuple(self):
        perf = self.herc.portfolio_performance(risk_free_rate=0.02)
        assert len(perf) == 3

    def test_portfolio_performance_cov_only(self):
        S = CovarianceShrinkage(get_data()).ledoit_wolf()
        herc = HERCOpt(cov_matrix=S)
        herc.optimize()
        mu, vol, sharpe = herc.portfolio_performance()
        assert mu is None
        assert sharpe is None
        assert vol > 0

    def test_ward_and_single_give_different_weights(self):
        returns = get_returns()
        herc_ward = HERCOpt(returns=returns)
        herc_single = HERCOpt(returns=returns)
        w_ward = herc_ward.optimize(linkage_method="ward")
        w_single = herc_single.optimize(linkage_method="single")
        # Not identical
        diff = sum(abs(w_ward[k] - w_single[k]) for k in w_ward)
        assert diff > 1e-6


# ---------------------------------------------------------------------------
# HERCOpt vs HRPOpt — sanity comparison
# ---------------------------------------------------------------------------

class TestHERCvsHRP:

    def test_herc_differs_from_hrp(self):
        """HERC should produce different weights than HRP."""
        returns = get_returns()
        hrp = HRPOpt(returns=returns)
        herc = HERCOpt(returns=returns)
        w_hrp = hrp.optimize(linkage_method="ward")
        w_herc = herc.optimize(linkage_method="ward")
        tickers = list(w_hrp.keys())
        diff = sum(abs(w_hrp[t] - w_herc[t]) for t in tickers)
        assert diff > 1e-4, "HERC and HRP weights are unexpectedly identical"

    def test_herc_lower_or_equal_portfolio_variance(self):
        """HERC uses ERC cluster risk, so its portfolio variance should
        not be dramatically worse than HRP's."""
        returns = get_returns()
        cov = returns.cov().values
        tickers = list(returns.columns)

        hrp = HRPOpt(returns=returns)
        w_hrp_dict = hrp.optimize(linkage_method="ward")
        w_hrp = np.array([w_hrp_dict[t] for t in tickers])

        herc = HERCOpt(returns=returns)
        w_herc_dict = herc.optimize(linkage_method="ward")
        w_herc = np.array([w_herc_dict[t] for t in tickers])

        var_hrp = float(w_hrp @ cov @ w_hrp)
        var_herc = float(w_herc @ cov @ w_herc)
        # Both should be well below equal-weight variance
        w_eq = np.ones(len(tickers)) / len(tickers)
        var_eq = float(w_eq @ cov @ w_eq)
        assert var_hrp < var_eq * 1.5
        assert var_herc < var_eq * 1.5


# ---------------------------------------------------------------------------
# NCOpt — instantiation errors
# ---------------------------------------------------------------------------

class TestNCOInstantiation:

    def test_no_args_raises(self):
        with pytest.raises(ValueError):
            NCOpt()

    def test_returns_not_dataframe_raises(self):
        with pytest.raises(TypeError):
            NCOpt(returns=get_returns().values)

    def test_invalid_internal_opt_raises(self):
        nco = NCOpt(returns=get_returns())
        with pytest.raises(ValueError, match="internal_opt"):
            nco.optimize(internal_opt="bad_method")

    def test_invalid_meta_opt_raises(self):
        nco = NCOpt(returns=get_returns())
        with pytest.raises(ValueError, match="meta_opt"):
            nco.optimize(meta_opt="bad_method")

    def test_invalid_linkage_raises(self):
        nco = NCOpt(returns=get_returns())
        with pytest.raises(ValueError):
            nco.optimize(linkage_method="not_a_method")

    def test_n_clusters_exceeds_assets_raises(self):
        nco = NCOpt(returns=get_returns())
        with pytest.raises(ValueError, match="n_assets"):
            nco.optimize(n_clusters=9999)

    def test_n_clusters_less_than_2_raises(self):
        nco = NCOpt(returns=get_returns())
        with pytest.raises(ValueError, match="n_clusters"):
            nco.optimize(n_clusters=1)

    def test_from_cov_only(self):
        S = CovarianceShrinkage(get_data()).ledoit_wolf()
        nco = NCOpt(cov_matrix=S)
        assert nco.n_assets == S.shape[0]


# ---------------------------------------------------------------------------
# NCOpt — portfolio weights
# ---------------------------------------------------------------------------

class TestNCOPortfolio:

    @pytest.fixture(autouse=True)
    def setup(self):
        returns = get_returns()
        self.nco = NCOpt(returns=returns)
        self.weights = self.nco.optimize(n_clusters=4)

    def test_weights_type(self):
        assert isinstance(self.weights, (dict, collections.OrderedDict))

    def test_weights_sum_to_one(self):
        np.testing.assert_allclose(sum(self.weights.values()), 1.0, atol=1e-6)

    def test_weights_non_negative(self):
        assert all(v >= -1e-8 for v in self.weights.values())

    def test_all_tickers_present(self):
        returns = get_returns()
        assert set(self.weights.keys()) == set(returns.columns)

    def test_clusters_attribute(self):
        assert self.nco.clusters is not None
        assert len(self.nco.clusters) == 4
        all_assets = [t for tickers in self.nco.clusters.values() for t in tickers]
        returns = get_returns()
        assert set(all_assets) == set(returns.columns)

    def test_cluster_linkage_attribute(self):
        assert self.nco.cluster_linkage is not None

    def test_portfolio_performance(self):
        perf = self.nco.portfolio_performance()
        assert len(perf) == 3
        assert perf[1] > 0  # volatility always positive


# ---------------------------------------------------------------------------
# NCOpt — all method combinations
# ---------------------------------------------------------------------------

class TestNCOMethodCombinations:

    def _run(self, internal, meta, n_clusters=4):
        returns = get_returns()
        nco = NCOpt(returns=returns)
        w = nco.optimize(n_clusters=n_clusters, internal_opt=internal, meta_opt=meta)
        assert abs(sum(w.values()) - 1.0) < 1e-6
        assert all(v >= -1e-8 for v in w.values())
        return w

    def test_min_var_min_var(self):
        self._run("min_variance", "min_variance")

    def test_erc_erc(self):
        self._run("erc", "erc")

    def test_equal_equal(self):
        self._run("equal", "equal")

    def test_min_var_erc(self):
        self._run("min_variance", "erc")

    def test_erc_min_var(self):
        self._run("erc", "min_variance")

    def test_equal_min_var(self):
        self._run("equal", "min_variance")

    def test_different_n_clusters(self):
        returns = get_returns()
        for k in [2, 3, 5, 10]:
            nco = NCOpt(returns=returns)
            w = nco.optimize(n_clusters=k)
            np.testing.assert_allclose(sum(w.values()), 1.0, atol=1e-6)

    def test_cov_only(self):
        """NCO must work with only a covariance matrix (no returns)."""
        S = CovarianceShrinkage(get_data()).ledoit_wolf()
        nco = NCOpt(cov_matrix=S)
        w = nco.optimize(n_clusters=4)
        np.testing.assert_allclose(sum(w.values()), 1.0, atol=1e-6)

    def test_default_n_clusters(self):
        """Default n_clusters = floor(sqrt(n_assets)) should not raise."""
        returns = get_returns()
        nco = NCOpt(returns=returns)
        w = nco.optimize()  # n_clusters=None → auto
        np.testing.assert_allclose(sum(w.values()), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# NCOpt — performance is meaningfully different from HRP
# ---------------------------------------------------------------------------

class TestNCOvsHRP:

    def test_nco_differs_from_hrp(self):
        returns = get_returns()
        hrp = HRPOpt(returns=returns)
        nco = NCOpt(returns=returns)
        w_hrp = hrp.optimize(linkage_method="ward")
        w_nco = nco.optimize(n_clusters=4)
        tickers = list(w_hrp.keys())
        diff = sum(abs(w_hrp[t] - w_nco.get(t, 0)) for t in tickers)
        assert diff > 1e-4


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------

class TestImports:

    def test_herc_importable(self):
        from pypfopt import HERCOpt as H
        assert H is HERCOpt

    def test_nco_importable(self):
        from pypfopt import NCOpt as N
        assert N is NCOpt

    def test_both_in_all(self):
        import pypfopt
        assert "HERCOpt" in pypfopt.__all__
        assert "NCOpt" in pypfopt.__all__
