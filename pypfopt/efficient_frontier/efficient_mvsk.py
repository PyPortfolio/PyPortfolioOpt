"""
The ``efficient_mvsk`` submodule houses the EfficientMVSK class, which
generates optimal portfolios by jointly optimizing over the first four
moments (mean, variance, skewness, kurtosis) using Yau's Affine-Normal
Descent algorithm.

Requires the ``yand-mvsk`` package: ``pip install yand-mvsk``.
"""

import numpy as np
import pandas as pd

from pypfopt.base import BaseOptimizer


class EfficientMVSK(BaseOptimizer):
    """
    Optimize portfolios using the mean-variance-skewness-kurtosis (MVSK)
    framework via the YAND algorithm (Wang, Niu, Sheshmani, Yau, 2025).

    The objective minimized is::

        f(x) = -c1 * mean(x) + c2 * var(x) - c3 * skew(x) + c4 * kurt(x)

    where the preference coefficients ``c`` can be derived from a CRRA
    utility function or specified directly.

    This extends the classical mean-variance framework by penalizing
    negative skewness (crash risk) and excess kurtosis (tail risk).

    Instance variables:

    - Inputs:

        - ``n_assets`` - int
        - ``tickers`` - str list
        - ``returns`` - np.ndarray, shape (T, n)
        - ``gamma`` - float, CRRA risk aversion
        - ``c`` - np.ndarray, shape (4,), preference vector

    - Output: ``weights`` - np.ndarray

    Public methods:

    - ``min_mvsk()`` minimizes the MVSK objective
    - ``portfolio_performance()`` calculates return, volatility, Sharpe,
      skewness and excess kurtosis of the optimized portfolio.
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(
        self,
        returns,
        gamma=6.0,
        c=None,
        weight_bounds=(0, 1),
    ):
        """
        Parameters
        ----------
        returns : pd.DataFrame or np.array
            (historic) returns for all your assets (no NaNs).
            Rows are observations, columns are assets.
        gamma : float, optional
            CRRA risk-aversion parameter, defaults to 6.0.
            Higher values penalize variance, skewness and kurtosis more.
            Ignored if ``c`` is provided.
        c : array-like, shape (4,), optional
            Custom preference vector [c1, c2, c3, c4].
            If provided, overrides ``gamma``.
        weight_bounds : tuple (float, float), optional
            (min_weight, max_weight) for each asset, defaults to (0, 1).
            Only long-only constraints are fully supported; upper bounds
            are accepted for API compatibility but values < 1 will raise
            NotImplementedError.

        Raises
        ------
        ImportError
            if ``yand-mvsk`` is not installed
        NotImplementedError
            if per-asset upper weight bounds < 1 are requested
        """
        try:
            from yand_mvsk import crra_coefficients, check_convexity
        except ImportError:
            raise ImportError(
                "Please install yand-mvsk: pip install yand-mvsk"
            )

        if isinstance(returns, pd.DataFrame):
            tickers = list(returns.columns)
            returns = returns.values
        else:
            tickers = None

        self.returns = np.asarray(returns, dtype=float)
        n_assets = self.returns.shape[1]

        if tickers is None:
            tickers = list(range(n_assets))

        super().__init__(n_assets, tickers)

        self.gamma = gamma
        self.c = np.asarray(c, dtype=float) if c is not None else crra_coefficients(gamma)
        self._convex = check_convexity(self.c)
        self._result = None

        # Parse weight bounds
        if isinstance(weight_bounds, tuple) and len(weight_bounds) == 2:
            lb, ub = weight_bounds
            self._lower_bound = 0.0 if lb is None else float(lb)
            if ub is not None and ub < 1.0:
                raise NotImplementedError(
                    "Per-asset upper weight bounds < 1 are not yet supported "
                    "by the YAND-MVSK solver."
                )
        else:
            raise TypeError(
                "weight_bounds must be a (lower, upper) tuple. "
                "Per-asset bounds are not supported."
            )

    @classmethod
    def from_prices(cls, prices, gamma=6.0, **kwargs):
        """
        Create an EfficientMVSK instance from a price DataFrame or array.
        Automatically computes simple returns.

        Parameters
        ----------
        prices : pd.DataFrame or np.array
            historic asset prices
        gamma : float, optional
            CRRA risk-aversion parameter, defaults to 6.0
        **kwargs
            forwarded to the constructor

        Returns
        -------
        EfficientMVSK
        """
        if isinstance(prices, pd.DataFrame):
            returns = prices.pct_change().dropna()
        else:
            prices = np.asarray(prices, dtype=float)
            returns = prices[1:] / prices[:-1] - 1.0
        return cls(returns, gamma=gamma, **kwargs)

    def min_mvsk(self):
        """
        Minimize the MVSK objective function (maximize risk-adjusted utility
        across the first four moments).

        Returns
        -------
        OrderedDict
            asset weights for the MVSK-optimal portfolio
        """
        from yand_mvsk import yand_mvsk_solve

        tau = max(self._lower_bound, 1e-8)
        self._result = yand_mvsk_solve(self.returns, self.c, tau=tau)

        if not self._result.converged:
            import warnings
            warnings.warn(
                f"YAND-MVSK solver did not converge after {self._result.n_iter} "
                f"iterations (KKT residual: {self._result.kkt_residual:.2e}). "
                "Results may be suboptimal."
            )

        self.weights = self._result.x
        return self._make_output_weights()

    @property
    def convex(self):
        """Whether the preference coefficients satisfy the convexity condition."""
        return self._convex

    def portfolio_performance(self, verbose=False, risk_free_rate=0.0):
        """
        After optimizing, calculate (and optionally print) the performance
        of the optimal portfolio, including higher-moment statistics.

        Parameters
        ----------
        verbose : bool, optional
            whether performance should be printed, defaults to False
        risk_free_rate : float, optional
            risk-free rate of borrowing/lending, defaults to 0.0

        Raises
        ------
        ValueError
            if weights have not been calculated yet

        Returns
        -------
        (float, float, float, float, float)
            expected return, volatility, Sharpe ratio, skewness, excess kurtosis.
        """
        if self.weights is None:
            raise ValueError("Weights not yet computed")

        port_returns = self.returns @ self.weights
        mu = port_returns.mean() * 252
        vol = port_returns.std(ddof=1) * np.sqrt(252)
        sharpe = (mu - risk_free_rate) / vol if vol > 1e-10 else 0.0

        centered = port_returns - port_returns.mean()
        m2 = np.mean(centered ** 2)
        skew = np.mean(centered ** 3) / (m2 ** 1.5) if m2 > 1e-30 else 0.0
        kurt = np.mean(centered ** 4) / (m2 ** 2) - 3.0 if m2 > 1e-30 else 0.0

        if verbose:
            print("Expected annual return: {:.1f}%".format(100 * mu))
            print("Annual volatility: {:.1f}%".format(100 * vol))
            print("Sharpe Ratio: {:.2f}".format(sharpe))
            print("Skewness: {:.3f}".format(skew))
            print("Excess Kurtosis: {:.3f}".format(kurt))

        return mu, vol, sharpe, skew, kurt
