"""
The ``hierarchical_portfolio`` module seeks to implement one of the recent advances in
portfolio optimization – the application of hierarchical clustering models in allocation.

All of the hierarchical classes have a similar API to ``EfficientFrontier``, though since
many hierarchical models currently don't support different objectives, the actual allocation
happens with a call to `optimize()`.

Currently implemented:

- ``HRPOpt`` implements the Hierarchical Risk Parity (HRP) portfolio. Code reproduced with
  permission from Marcos Lopez de Prado (2016).
"""

import collections

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

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
    Equal Risk Contribution weights via Spinu (2013) cyclical coordinate descent.

    Finds w ≥ 0, sum(w)=1 such that every asset contributes the same fraction
    to total portfolio variance:  w_i*(Σw)_i = w_j*(Σw)_j  for all i, j.

    At each CCD step the exact one-dimensional sub-problem is solved:

        Σᵢᵢ·wᵢ² + (Σw − Σᵢᵢ·wᵢ)·wᵢ − 1/n = 0

    taking its positive root.  Weights are NOT normalised between coordinate
    updates; normalisation happens once per full pass, then at the end.
    This unconstrained formulation converges reliably for any PD covariance
    matrix, including those with negative off-diagonal entries.

    Parameters
    ----------
    cov : np.ndarray
        (n, n) covariance matrix (must be positive definite).
    tol : float
        Convergence threshold on max absolute change in weights.
    max_iter : int
        Maximum number of full passes over all coordinates.

    Returns
    -------
    np.ndarray
        (n,) ERC weight vector summing to 1.

    References
    ----------
    Spinu, F. (2013). An Algorithm for Computing Risk Parity Weights.
    SSRN working paper.
    """
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    b = 1.0 / n  # equal risk budget
    w = np.ones(n) / n
    for _ in range(max_iter):
        w_prev = w.copy()
        for i in range(n):
            a_ii = float(cov[i, i])
            cross = float(cov[i] @ w) - a_ii * w[i]
            disc = cross * cross + 4.0 * a_ii * b
            w[i] = (-cross + np.sqrt(max(disc, 0.0))) / (2.0 * a_ii)
        if np.max(np.abs(w - w_prev)) < tol:
            break

    w /= w.sum()
    return w


class ERCOpt(BaseOptimizer):
    """
    Equal Risk Contribution (ERC) / Risk Parity portfolio optimizer.

    Constructs weights w ≥ 0, sum(w)=1 such that each asset contributes
    an equal fraction of total portfolio variance:

    .. code-block:: text

        w_i · (Σw)_i / (w'Σw) = 1/n   for all i

    Equivalently, the marginal risk contributions  w_i·(Σw)_i  are all equal.
    This is also called the *Risk Parity* portfolio and satisfies

    .. code-block:: text

        w ∝ Σ⁻¹ 1   (inverse-variance) when Σ is diagonal.

    Unlike mean-variance optimization, ERC requires only a covariance estimate
    and has been shown to be more robust out-of-sample than max-Sharpe or
    min-variance portfolios (Maillard, Roncalli & Teiletche 2010).

    Instance variables:

    - Inputs

        - ``n_assets`` - int
        - ``tickers`` - str list
        - ``returns`` - pd.DataFrame  (if provided)
        - ``cov_matrix`` - pd.DataFrame  (if provided)

    - Output:

        - ``weights`` - np.ndarray

    Public methods:

    - ``optimize()`` calculates ERC weights
    - ``portfolio_performance()`` calculates expected return, volatility and Sharpe ratio
    - ``set_weights()`` creates self.weights from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros
    - ``save_weights_to_file()`` saves weights to csv, json, or txt

    Examples
    --------
    ::

        from pypfopt import ERCOpt

        erc = ERCOpt(returns=returns_df)
        weights = erc.optimize()
        erc.portfolio_performance(verbose=True)

        # From a covariance matrix directly
        erc = ERCOpt(cov_matrix=cov_df)
        weights = erc.optimize()

    References
    ----------
    Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of
    Equally Weighted Risk Contribution Portfolios. *Journal of Portfolio
    Management*, 36(4), 60-70.

    Spinu, F. (2013). An Algorithm for Computing Risk Parity Weights.
    SSRN working paper.
    """

    def __init__(self, returns=None, cov_matrix=None):
        """
        Parameters
        ----------
        returns : pd.DataFrame, optional
            Asset historical returns (T × n). Used to compute the sample
            covariance matrix if ``cov_matrix`` is not provided.
        cov_matrix : pd.DataFrame, optional
            Covariance matrix of asset returns (n × n). At least one of
            ``returns`` or ``cov_matrix`` must be supplied.

        Raises
        ------
        ValueError
            If neither ``returns`` nor ``cov_matrix`` is provided.
        TypeError
            If ``returns`` is not a pandas DataFrame.
        """
        if returns is None and cov_matrix is None:
            raise ValueError("Either returns or cov_matrix must be provided")
        if returns is not None and not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")

        self.returns = returns
        self.cov_matrix = cov_matrix

        tickers = list(cov_matrix.columns) if returns is None else list(returns.columns)
        super().__init__(len(tickers), tickers)

    def optimize(self, tol=1e-12, max_iter=500):
        """
        Compute the Equal Risk Contribution (Risk Parity) portfolio.

        Uses the Spinu (2013) cyclical coordinate descent: iteratively solves
        the exact one-dimensional sub-problem for each asset until the
        maximum weight change falls below ``tol``.

        Parameters
        ----------
        tol : float, optional
            Convergence tolerance, default 1e-12.
        max_iter : int, optional
            Maximum CCD iterations, default 500.

        Returns
        -------
        OrderedDict
            ``{ticker: weight}`` mapping, weights sum to 1 and all ≥ 0.
        """
        cov = (
            self.returns.cov()
            if self.cov_matrix is None
            else self.cov_matrix
        )
        cov_arr = np.asarray(cov)
        raw_w = _erc_weights_ccd(cov_arr, tol=tol, max_iter=max_iter)
        weights = collections.OrderedDict(zip(self.tickers, raw_w))
        self.set_weights(weights)
        return weights

    def portfolio_performance(self, verbose=False, risk_free_rate=0.0, frequency=252):
        """
        After optimising, calculate (and optionally print) the performance of
        the ERC portfolio.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print the performance, default False.
        risk_free_rate : float, optional
            Annualised risk-free rate, default 0.0.
        frequency : int, optional
            Number of periods per year, default 252 (trading days).

        Returns
        -------
        (float, float, float)
            Expected return, volatility, Sharpe ratio.

        Raises
        ------
        ValueError
            If ``optimize()`` has not been called yet.
        """
        if self.returns is None:
            cov = self.cov_matrix
            mu = None
        else:
            cov = self.returns.cov() * frequency
            mu = self.returns.mean() * frequency

        return portfolio_performance(self.weights, mu, cov, verbose, risk_free_rate)
