"""
The ``hierarchical_portfolio`` module seeks to implement one of the recent advances in
portfolio optimization – the application of hierarchical clustering models in allocation.

All of the hierarchical classes have a similar API to ``EfficientFrontier``, though since
many hierarchical models currently don't support different objectives, the actual allocation
happens with a call to `optimize()`.

Currently implemented:

- ``HRPOpt`` implements the Hierarchical Risk Parity (HRP) portfolio. Code reproduced with
  permission from Marcos Lopez de Prado (2016).
- ``HERCOpt`` implements the Hierarchical Equal Risk Contribution (HERC) portfolio
  (Raffinot 2018), which uses Equal Risk Contribution weights at each level of the
  hierarchy rather than the inverse-variance weights used by HRP.
- ``NCOpt`` implements the Nested Cluster Optimization (NCO) portfolio
  (Lopez de Prado 2019), which optimizes within clusters and then across clusters
  in a two-level nested procedure.
"""

import collections

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from scipy.optimize import minimize

from pypfopt.base import BaseOptimizer, portfolio_performance
from pypfopt.risk_models import cov_to_corr


class HRPOpt(BaseOptimizer):
    """
    A HRPOpt object (inheriting from BaseOptimizer) constructs a hierarchical
    risk parity portfolio.

    Instance variables:

    - Inputs

        - ``n_assets`` - int
        - ``tickers`` - str list
        - ``returns`` - pd.DataFrame

    - Output:

        - ``weights`` - np.ndarray
        - ``clusters`` - linkage matrix corresponding to clustered assets.

    Public methods:

    - ``optimize()`` calculates weights using HRP
    - ``portfolio_performance()`` calculates the expected return, volatility and Sharpe ratio for
      the optimized portfolio.
    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(self, returns=None, cov_matrix=None):
        """
        Parameters
        ----------
        returns : pd.DataFrame
            asset historical returns
        cov_matrix : pd.DataFrame
            covariance of asset returns

        Raises
        ------
        TypeError
            if ``returns`` is not a dataframe
        """
        if returns is None and cov_matrix is None:
            raise ValueError("Either returns or cov_matrix must be provided")

        if returns is not None and not isinstance(returns, pd.DataFrame):
            raise TypeError("returns are not a dataframe")

        self.returns = returns
        self.cov_matrix = cov_matrix
        self.clusters = None

        if returns is None:
            tickers = list(cov_matrix.columns)
        else:
            tickers = list(returns.columns)
        super().__init__(len(tickers), tickers)

    @staticmethod
    def _get_cluster_var(cov, cluster_items):
        """
        Compute the variance per cluster

        Parameters
        ----------
        cov : np.ndarray
            covariance matrix
        cluster_items : list
            tickers in the cluster

        Returns
        -------
        float
            the variance per cluster
        """
        # Compute variance per cluster
        cov_slice = cov.loc[cluster_items, cluster_items]
        weights = 1 / np.diag(cov_slice)  # Inverse variance weights
        weights /= weights.sum()
        return np.linalg.multi_dot((weights, cov_slice, weights))

    @staticmethod
    def _get_quasi_diag(link):
        """
        Sort clustered items by distance

        Parameters
        ----------
        link : np.ndarray
            linkage matrix after clustering

        Returns
        -------
        list
            sorted list of indices
        """
        return sch.to_tree(link, rd=False).pre_order()

    @staticmethod
    def _raw_hrp_allocation(cov, ordered_tickers):
        """
        Given the clusters, compute the portfolio that minimises risk by
        recursively traversing the hierarchical tree from the top.

        Parameters
        ----------
        cov : np.ndarray
            covariance matrix
        ordered_tickers : str list
            list of tickers ordered by distance

        Returns
        -------
        pd.Series
            raw portfolio weights
        """
        w = pd.Series(1.0, index=ordered_tickers)
        cluster_items = [ordered_tickers]  # initialize all items in one cluster

        while len(cluster_items) > 0:
            cluster_items = [
                i[j:k]
                for i in cluster_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]  # bi-section
            # For each pair, optimize locally.
            for i in range(0, len(cluster_items), 2):
                first_cluster = cluster_items[i]
                second_cluster = cluster_items[i + 1]
                # Form the inverse variance portfolio for this pair
                first_variance = HRPOpt._get_cluster_var(cov, first_cluster)
                second_variance = HRPOpt._get_cluster_var(cov, second_cluster)
                alpha = 1 - first_variance / (first_variance + second_variance)
                w[first_cluster] *= alpha  # weight 1
                w[second_cluster] *= 1 - alpha  # weight 2
        return w

    def optimize(self, linkage_method="single"):
        """
        Construct a hierarchical risk parity portfolio, using Scipy hierarchical clustering
        (see `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`_)

        Parameters
        ----------
        linkage_method : str
            which scipy linkage method to use

        Returns
        -------
        OrderedDict
            weights for the HRP portfolio
        """
        if linkage_method not in sch._LINKAGE_METHODS:
            raise ValueError("linkage_method must be one recognised by scipy")

        if self.returns is None:
            cov = self.cov_matrix
            corr = cov_to_corr(self.cov_matrix).round(6)
        else:
            corr, cov = self.returns.corr(), self.returns.cov()

        # Compute distance matrix, with ClusterWarning fix as
        # per https://stackoverflow.com/questions/18952587/

        # this can avoid some nasty floating point issues
        matrix = np.sqrt(np.clip((1.0 - corr) / 2.0, a_min=0.0, a_max=1.0))
        dist = ssd.squareform(matrix, checks=False)

        self.clusters = sch.linkage(dist, linkage_method)
        sort_ix = HRPOpt._get_quasi_diag(self.clusters)
        ordered_tickers = corr.index[sort_ix].tolist()
        hrp = HRPOpt._raw_hrp_allocation(cov, ordered_tickers)
        weights = collections.OrderedDict(hrp.sort_index())
        self.set_weights(weights)
        return weights

    def portfolio_performance(self, verbose=False, risk_free_rate=0.0, frequency=252):
        """
        After optimising, calculate (and optionally print) the performance of the optimal
        portfolio. Currently calculates expected return, volatility, and the Sharpe ratio
        assuming returns are daily

        Parameters
        ----------
        verbose : bool, optional
            whether performance should be printed, defaults to False
        risk_free_rate : float, optional
            risk-free rate of borrowing/lending, defaults to 0.0.
            The period of the risk-free rate should correspond to the
            frequency of expected returns.
        frequency : int, optional
            number of time periods in a year, defaults to 252 (the number
            of trading days in a year)

        Raises
        ------
        ValueError
            if weights have not been calculated yet

        Returns
        -------
        (float, float, float)
            expected return, volatility, Sharpe ratio.
        """
        if self.returns is None:
            cov = self.cov_matrix
            mu = None
        else:
            cov = self.returns.cov() * frequency
            mu = self.returns.mean() * frequency

        return portfolio_performance(self.weights, mu, cov, verbose, risk_free_rate)


def _erc_weights_ccd(cov: np.ndarray, tol: float = 1e-12, max_iter: int = 500) -> np.ndarray:
    """
    Equal Risk Contribution weights via cyclical coordinate descent.

    Finds w ≥ 0, sum(w)=1, such that w_i*(Σw)_i = w_j*(Σw)_j for all i, j —
    i.e., every asset contributes the same fraction to total portfolio variance.

    Uses the multiplicative update of Roncalli (2013):
        w_i ← w_i * sqrt(target_budget / marginal_risk_contribution_i)
    normalised at each step.

    Parameters
    ----------
    cov : np.ndarray
        (n, n) covariance matrix (must be PD).
    tol : float
        Convergence threshold on max change in weights.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    np.ndarray
        (n,) ERC weight vector summing to 1.
    """
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    w = np.ones(n) / n
    for _ in range(max_iter):
        Sigma_w = cov @ w
        port_var = float(w @ Sigma_w)
        rc = w * Sigma_w  # risk contributions (unnormalised)
        target = port_var / n  # equal budget
        # Multiplicative update: w_i ← w_i * sqrt(target / rc_i)
        w_new = w * np.sqrt(target / np.maximum(rc, 1e-30))
        w_new = np.maximum(w_new, 0.0)
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    return w


def _min_var_weights(cov: np.ndarray) -> np.ndarray:
    """
    Long-only minimum variance weights.

    Uses the analytic unconstrained solution (Σ⁻¹ 1 / 1ᵀ Σ⁻¹ 1) when all
    resulting weights are non-negative; otherwise falls back to SLSQP.

    Parameters
    ----------
    cov : np.ndarray
        (n, n) covariance matrix.

    Returns
    -------
    np.ndarray
        (n,) weight vector summing to 1, all ≥ 0.
    """
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    try:
        inv_cov = np.linalg.inv(cov)
        ones = np.ones(n)
        w = inv_cov @ ones / (ones @ inv_cov @ ones)
        if (w >= -1e-8).all():
            w = np.maximum(w, 0.0)
            return w / w.sum()
    except np.linalg.LinAlgError:
        pass

    # Long-only constrained via SLSQP
    res = minimize(
        lambda w_: float(w_ @ cov @ w_),
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w_: w_.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    w = np.maximum(res.x, 0.0)
    return w / w.sum()


class HERCOpt(BaseOptimizer):
    """
    A HERCOpt object constructs a Hierarchical Equal Risk Contribution (HERC)
    portfolio (Raffinot, 2018).

    HERC extends Hierarchical Risk Parity (HRP) by replacing the
    inverse-variance weighting with Equal Risk Contribution (ERC) at every
    level of the hierarchy.  The structure is otherwise identical to HRP:

    1. Compute the correlation-based distance matrix.
    2. Perform hierarchical clustering.
    3. Order assets by quasi-diagonal traversal.
    4. Recursively bisect, allocating between sub-clusters in proportion to
       the **inverse of each cluster's ERC-portfolio variance** (rather than
       the inverse-variance-portfolio variance used by HRP).

    Because ERC accounts for within-cluster correlations when computing the
    cluster's effective risk, HERC tends to produce more balanced risk
    allocations than HRP, especially when intra-cluster correlations are high.

    Instance variables:

    - Inputs

        - ``n_assets`` - int
        - ``tickers`` - str list
        - ``returns`` - pd.DataFrame or None
        - ``cov_matrix`` - pd.DataFrame or None

    - Output:

        - ``weights`` - OrderedDict
        - ``clusters`` - linkage matrix

    Examples
    --------
    .. code-block:: python

        from pypfopt import HERCOpt
        herc = HERCOpt(returns=returns_df)
        weights = herc.optimize()
        herc.portfolio_performance(verbose=True)

    References
    ----------
    Raffinot, T. (2018). Hierarchical clustering-based asset allocation.
    *Journal of Portfolio Management*, 44(2), 89-99.
    """

    def __init__(self, returns=None, cov_matrix=None):
        """
        Parameters
        ----------
        returns : pd.DataFrame, optional
            Asset historical returns (T × n).  Either ``returns`` or
            ``cov_matrix`` must be supplied.
        cov_matrix : pd.DataFrame, optional
            Covariance matrix of asset returns (n × n).

        Raises
        ------
        ValueError
            If neither ``returns`` nor ``cov_matrix`` is provided.
        TypeError
            If ``returns`` is not a DataFrame.
        """
        if returns is None and cov_matrix is None:
            raise ValueError("Either returns or cov_matrix must be provided")
        if returns is not None and not isinstance(returns, pd.DataFrame):
            raise TypeError("returns are not a dataframe")

        self.returns = returns
        self.cov_matrix = cov_matrix
        self.clusters = None

        tickers = list(cov_matrix.columns) if returns is None else list(returns.columns)
        super().__init__(len(tickers), tickers)

    @staticmethod
    def _get_cluster_var_erc(cov: pd.DataFrame, cluster_items: list) -> float:
        """
        Variance of the ERC portfolio built from *cluster_items*.

        Parameters
        ----------
        cov : pd.DataFrame
            Full covariance matrix.
        cluster_items : list
            Asset names in the cluster.

        Returns
        -------
        float
            Portfolio variance under ERC weights.
        """
        cov_slice = cov.loc[cluster_items, cluster_items].values
        w = _erc_weights_ccd(cov_slice)
        return float(w @ cov_slice @ w)

    @staticmethod
    def _raw_herc_allocation(cov: pd.DataFrame, ordered_tickers: list) -> pd.Series:
        """
        Recursive bisection allocation using ERC cluster variances.

        Identical in structure to HRP's ``_raw_hrp_allocation``, but calls
        ``_get_cluster_var_erc`` instead of ``_get_cluster_var`` so that the
        between-cluster split reflects ERC-weighted risk.

        Parameters
        ----------
        cov : pd.DataFrame
            Full covariance matrix.
        ordered_tickers : list
            Tickers ordered by quasi-diagonal traversal.

        Returns
        -------
        pd.Series
            Portfolio weights indexed by ticker.
        """
        w = pd.Series(1.0, index=ordered_tickers)
        cluster_items = [ordered_tickers]

        while len(cluster_items) > 0:
            cluster_items = [
                i[j:k]
                for i in cluster_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            for i in range(0, len(cluster_items), 2):
                first_cluster = cluster_items[i]
                second_cluster = cluster_items[i + 1]
                first_variance = HERCOpt._get_cluster_var_erc(cov, first_cluster)
                second_variance = HERCOpt._get_cluster_var_erc(cov, second_cluster)
                alpha = 1 - first_variance / (first_variance + second_variance)
                w[first_cluster] *= alpha
                w[second_cluster] *= 1 - alpha

        return w

    def optimize(self, linkage_method: str = "ward") -> collections.OrderedDict:
        """
        Construct the HERC portfolio.

        Parameters
        ----------
        linkage_method : str, optional (default ``"ward"``)
            Scipy linkage method; any value accepted by
            ``scipy.cluster.hierarchy.linkage``.  Ward's method is recommended
            for HERC as it minimises within-cluster variance at each merge.

        Returns
        -------
        OrderedDict
            Portfolio weights (asset → weight).

        Raises
        ------
        ValueError
            If ``linkage_method`` is not recognised by scipy.
        """
        if linkage_method not in sch._LINKAGE_METHODS:
            raise ValueError("linkage_method must be one recognised by scipy")

        if self.returns is None:
            cov = self.cov_matrix
            corr = cov_to_corr(self.cov_matrix).round(6)
        else:
            corr, cov = self.returns.corr(), self.returns.cov()

        matrix = np.sqrt(np.clip((1.0 - corr) / 2.0, a_min=0.0, a_max=1.0))
        dist = ssd.squareform(matrix, checks=False)

        self.clusters = sch.linkage(dist, linkage_method)
        sort_ix = HRPOpt._get_quasi_diag(self.clusters)
        ordered_tickers = corr.index[sort_ix].tolist()

        herc = HERCOpt._raw_herc_allocation(cov, ordered_tickers)
        weights = collections.OrderedDict(herc.sort_index())
        self.set_weights(weights)
        return weights

    def portfolio_performance(
        self, verbose: bool = False, risk_free_rate: float = 0.0, frequency: int = 252
    ):
        """
        Calculate (and optionally print) the performance of the HERC portfolio.

        Parameters
        ----------
        verbose : bool, optional (default False)
            Print the performance summary.
        risk_free_rate : float, optional (default 0.0)
            Risk-free rate for Sharpe ratio calculation.
        frequency : int, optional (default 252)
            Number of trading periods per year.

        Returns
        -------
        (float, float, float)
            Expected return, annual volatility, Sharpe ratio.
            Expected return and Sharpe ratio are ``None`` when only a
            covariance matrix was supplied.
        """
        if self.returns is None:
            cov = self.cov_matrix
            mu = None
        else:
            cov = self.returns.cov() * frequency
            mu = self.returns.mean() * frequency

        return portfolio_performance(self.weights, mu, cov, verbose, risk_free_rate)


class NCOpt(BaseOptimizer):
    """
    A NCOpt object constructs a Nested Cluster Optimization (NCO) portfolio
    (Lopez de Prado, 2019).

    NCO partitions the asset universe into clusters, solves a portfolio
    optimisation problem *within* each cluster, and then solves a second
    optimisation *across* the resulting cluster-portfolios.  This two-level
    structure allows a large and correlated asset universe to be decomposed
    into smaller, nearly independent subproblems.

    **Algorithm**

    1. Compute the correlation-based distance matrix and perform hierarchical
       clustering; cut the dendrogram to obtain ``n_clusters`` clusters.
    2. **Intra-cluster step**: for each cluster, compute an optimal portfolio
       over the cluster's assets (min-variance, ERC, or equal-weight).
    3. **Meta step**: treat each cluster as a single synthetic asset; compute
       the covariance of the cluster-portfolio returns and optimise over the
       resulting ``n_clusters`` assets.
    4. Combine: final weight for asset i in cluster c is
       ``w_intra[c][i] * w_meta[c]``.

    Instance variables:

    - Inputs

        - ``n_assets`` - int
        - ``tickers`` - str list
        - ``returns`` - pd.DataFrame or None
        - ``cov_matrix`` - pd.DataFrame or None

    - Output:

        - ``weights`` - OrderedDict
        - ``clusters`` - dict mapping cluster index → list of tickers
        - ``cluster_linkage`` - linkage matrix

    Examples
    --------
    .. code-block:: python

        from pypfopt import NCOpt
        nco = NCOpt(returns=returns_df)
        weights = nco.optimize(n_clusters=4, internal_opt="min_variance", meta_opt="erc")
        nco.portfolio_performance(verbose=True)

    References
    ----------
    Lopez de Prado, M. (2019). A Robust Estimator of the Efficient Frontier.
    *SSRN Working Paper*.  https://ssrn.com/abstract=3469961
    """

    _VALID_OPT = {"min_variance", "erc", "equal"}

    def __init__(self, returns=None, cov_matrix=None):
        """
        Parameters
        ----------
        returns : pd.DataFrame, optional
            Asset historical returns (T × n).
        cov_matrix : pd.DataFrame, optional
            Covariance matrix (n × n).

        Raises
        ------
        ValueError
            If neither ``returns`` nor ``cov_matrix`` is provided.
        TypeError
            If ``returns`` is not a DataFrame.
        """
        if returns is None and cov_matrix is None:
            raise ValueError("Either returns or cov_matrix must be provided")
        if returns is not None and not isinstance(returns, pd.DataFrame):
            raise TypeError("returns are not a dataframe")

        self.returns = returns
        self.cov_matrix = cov_matrix
        self.clusters = None
        self.cluster_linkage = None

        tickers = list(cov_matrix.columns) if returns is None else list(returns.columns)
        super().__init__(len(tickers), tickers)

    @staticmethod
    def _get_n_clusters(n_assets: int) -> int:
        """Default cluster count: floor(sqrt(n_assets)), minimum 2."""
        return max(2, int(np.floor(np.sqrt(n_assets))))

    @staticmethod
    def _assign_clusters(linkage_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Cut a hierarchical dendrogram to obtain *n_clusters* flat clusters.

        Returns an integer array of shape (n_assets,) with cluster labels
        starting at 0.
        """
        labels = sch.fcluster(linkage_matrix, n_clusters, criterion="maxclust")
        return labels - 1  # zero-indexed

    @staticmethod
    def _intra_cluster_weights(cov: pd.DataFrame, tickers: list, method: str) -> np.ndarray:
        """
        Solve the within-cluster portfolio problem.

        Parameters
        ----------
        cov : pd.DataFrame
            Full covariance matrix.
        tickers : list
            Assets in this cluster.
        method : str
            One of ``"min_variance"``, ``"erc"``, or ``"equal"``.

        Returns
        -------
        np.ndarray
            Weight vector of length len(tickers), summing to 1.
        """
        n = len(tickers)
        if n == 1:
            return np.array([1.0])

        cov_slice = cov.loc[tickers, tickers].values

        if method == "min_variance":
            return _min_var_weights(cov_slice)
        elif method == "erc":
            return _erc_weights_ccd(cov_slice)
        else:  # equal
            return np.ones(n) / n

    def optimize(
        self,
        n_clusters: int = None,
        internal_opt: str = "min_variance",
        meta_opt: str = "min_variance",
        linkage_method: str = "ward",
    ) -> collections.OrderedDict:
        """
        Construct the NCO portfolio.

        Parameters
        ----------
        n_clusters : int, optional
            Number of asset clusters.  Defaults to ``floor(sqrt(n_assets))``.
        internal_opt : str, optional (default ``"min_variance"``)
            Within-cluster optimisation objective.
            One of ``"min_variance"``, ``"erc"``, ``"equal"``.
        meta_opt : str, optional (default ``"min_variance"``)
            Across-cluster (meta) optimisation objective.
            One of ``"min_variance"``, ``"erc"``, ``"equal"``.
        linkage_method : str, optional (default ``"ward"``)
            Scipy hierarchical linkage method.

        Returns
        -------
        OrderedDict
            Portfolio weights (asset → weight).

        Raises
        ------
        ValueError
            If ``n_clusters`` exceeds the number of assets, or if an
            unrecognised optimisation method is provided.
        """
        if internal_opt not in self._VALID_OPT:
            raise ValueError("internal_opt must be one of %s" % self._VALID_OPT)
        if meta_opt not in self._VALID_OPT:
            raise ValueError("meta_opt must be one of %s" % self._VALID_OPT)
        if linkage_method not in sch._LINKAGE_METHODS:
            raise ValueError("linkage_method must be one recognised by scipy")

        n = self.n_assets
        if n_clusters is None:
            n_clusters = self._get_n_clusters(n)
        if n_clusters > n:
            raise ValueError("n_clusters (%d) cannot exceed n_assets (%d)" % (n_clusters, n))
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")

        # --- Clustering ---
        if self.returns is None:
            cov = self.cov_matrix
            corr = cov_to_corr(self.cov_matrix).round(6)
        else:
            corr, cov = self.returns.corr(), self.returns.cov()

        tickers = list(corr.columns)
        matrix = np.sqrt(np.clip((1.0 - corr) / 2.0, a_min=0.0, a_max=1.0))
        dist = ssd.squareform(matrix, checks=False)
        self.cluster_linkage = sch.linkage(dist, linkage_method)
        labels = NCOpt._assign_clusters(self.cluster_linkage, n_clusters)

        # Build cluster → ticker mapping
        self.clusters = collections.defaultdict(list)
        for i, label in enumerate(labels):
            self.clusters[int(label)].append(tickers[i])
        self.clusters = dict(self.clusters)

        # --- Intra-cluster step ---
        intra_weights = {}
        for c, ctickers in self.clusters.items():
            w_intra = NCOpt._intra_cluster_weights(cov, ctickers, method=internal_opt)
            intra_weights[c] = pd.Series(w_intra, index=ctickers)

        # --- Build meta-covariance ---
        # meta_cov[i,j] = w_i' Sigma w_j  (analytic from full cov)
        cluster_keys = sorted(self.clusters.keys())
        n_c = len(cluster_keys)
        meta_cov = np.zeros((n_c, n_c))
        for ci, c_i in enumerate(cluster_keys):
            for cj, c_j in enumerate(cluster_keys):
                w_i = intra_weights[c_i].reindex(tickers).fillna(0.0).values
                w_j = intra_weights[c_j].reindex(tickers).fillna(0.0).values
                meta_cov[ci, cj] = float(w_i @ cov.values @ w_j)

        # --- Meta optimisation ---
        meta_cov_df = pd.DataFrame(
            meta_cov,
            index=["c%d" % k for k in cluster_keys],
            columns=["c%d" % k for k in cluster_keys],
        )
        if meta_opt == "min_variance":
            w_meta = _min_var_weights(meta_cov)
        elif meta_opt == "erc":
            w_meta = _erc_weights_ccd(meta_cov)
        else:
            w_meta = np.ones(n_c) / n_c
        w_meta_series = pd.Series(w_meta, index=cluster_keys)

        # --- Combine ---
        final_weights = pd.Series(0.0, index=tickers)
        for ci, c in enumerate(cluster_keys):
            for ticker, w_intra_val in intra_weights[c].items():
                final_weights[ticker] = w_intra_val * float(w_meta_series[c])

        weights = collections.OrderedDict(final_weights.sort_index())
        self.set_weights(weights)
        return weights

    def portfolio_performance(
        self, verbose: bool = False, risk_free_rate: float = 0.0, frequency: int = 252
    ):
        """
        Calculate (and optionally print) the performance of the NCO portfolio.

        Parameters
        ----------
        verbose : bool, optional (default False)
            Print the performance summary.
        risk_free_rate : float, optional (default 0.0)
            Risk-free rate for Sharpe ratio calculation.
        frequency : int, optional (default 252)
            Number of trading periods per year.

        Returns
        -------
        (float, float, float)
            Expected return, annual volatility, Sharpe ratio.
        """
        if self.returns is None:
            cov = self.cov_matrix
            mu = None
        else:
            cov = self.returns.cov() * frequency
            mu = self.returns.mean() * frequency

        return portfolio_performance(self.weights, mu, cov, verbose, risk_free_rate)
