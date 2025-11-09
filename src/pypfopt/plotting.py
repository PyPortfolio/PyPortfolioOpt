"""
The ``plotting`` module houses all the functions to generate various plots.

Currently implemented:

  - ``plot_covariance`` - plot a correlation matrix
  - ``plot_dendrogram`` - plot the hierarchical clusters in a portfolio
  - ``plot_efficient_frontier`` – plot the efficient frontier from an EfficientFrontier or CLA object
  - ``plot_weights`` - bar chart of weights
"""

import warnings

import numpy as np
import scipy.cluster.hierarchy as sch

from . import CLA, EfficientFrontier, exceptions, risk_models

try:
    import matplotlib.pyplot as plt
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    raise ImportError("Please install matplotlib via pip or poetry")


def _get_plotly():
    """Helper function to import plotly only when needed"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        return go, make_subplots
    except (ModuleNotFoundError, ImportError):
        raise ImportError(
            "Please install plotly via pip or poetry to use interactive plots"
        )


def _plot_io(**kwargs):
    """
    Helper method to optionally save the figure to file.

    :param filename: name of the file to save to, defaults to None (doesn't save)
    :type filename: str, optional
    :param dpi: dpi of figure to save or plot, defaults to 300
    :type dpi: int (between 50-500)
    :param showfig: whether to plt.show() the figure, defaults to False
    :type showfig: bool, optional
    """
    filename = kwargs.get("filename", None)
    showfig = kwargs.get("showfig", False)
    dpi = kwargs.get("dpi", 300)

    plt.tight_layout()
    if filename:
        plt.savefig(fname=filename, dpi=dpi)
    if showfig:  # pragma: no cover
        plt.show()


def plot_covariance(cov_matrix, plot_correlation=False, show_tickers=True, **kwargs):
    """
    Generate a basic plot of the covariance (or correlation) matrix, given a
    covariance matrix.

    :param cov_matrix: covariance matrix
    :type cov_matrix: pd.DataFrame or np.ndarray
    :param plot_correlation: whether to plot the correlation matrix instead, defaults to False.
    :type plot_correlation: bool, optional
    :param show_tickers: whether to use tickers as labels (not recommended for large portfolios),
                        defaults to True
    :type show_tickers: bool, optional

    :return: matplotlib axis
    :rtype: matplotlib.axes object
    """
    if plot_correlation:
        matrix = risk_models.cov_to_corr(cov_matrix)
    else:
        matrix = cov_matrix
    fig, ax = plt.subplots()

    cax = ax.imshow(matrix)
    fig.colorbar(cax)

    if show_tickers:
        ax.set_xticks(np.arange(0, matrix.shape[0], 1))
        ax.set_xticklabels(matrix.index)
        ax.set_yticks(np.arange(0, matrix.shape[0], 1))
        ax.set_yticklabels(matrix.index)
        plt.xticks(rotation=90)

    _plot_io(**kwargs)

    return ax


def plot_dendrogram(hrp, ax=None, show_tickers=True, **kwargs):
    """
    Plot the clusters in the form of a dendrogram.

    :param hrp: HRPpt object that has already been optimized.
    :type hrp: object
    :param show_tickers: whether to use tickers as labels (not recommended for large portfolios),
                        defaults to True
    :type show_tickers: bool, optional
    :param filename: name of the file to save to, defaults to None (doesn't save)
    :type filename: str, optional
    :param showfig: whether to plt.show() the figure, defaults to False
    :type showfig: bool, optional
    :return: matplotlib axis
    :rtype: matplotlib.axes object
    """
    ax = ax or plt.gca()

    if hrp.clusters is None:
        warnings.warn(
            "hrp param has not been optimized.  Attempting optimization.",
            RuntimeWarning,
        )
        hrp.optimize()

    if show_tickers:
        sch.dendrogram(hrp.clusters, labels=hrp.tickers, ax=ax, orientation="top")
        ax.tick_params(axis="x", rotation=90)
        plt.tight_layout()
    else:
        sch.dendrogram(hrp.clusters, no_labels=True, ax=ax)

    _plot_io(**kwargs)

    return ax


def _plot_cla(cla, points, ax, show_assets, show_tickers, interactive):
    """
    Helper function to plot the efficient frontier from a CLA object
    """
    if interactive:
        go, _ = _get_plotly()

    if cla.weights is None:
        cla.max_sharpe()
    optimal_ret, optimal_risk, sharpe_max = cla.portfolio_performance()
    opt_weights = cla.weights
    if cla.frontier_values is None:
        cla.efficient_frontier(points=points)

    mus, sigmas, weights = cla.frontier_values

    if interactive:
        # Create the label
        hovertemplate = "Risk: %{x}<br>Return: %{y}<extra>"
        # Loop over each asset and append its information
        for i, ticker in enumerate(cla.tickers):
            hovertemplate += f"{ticker}: %{{customdata[{i}]:.4%}}<br>"
        hovertemplate += "</extra>"

        ax.add_trace(
            go.Scatter(
                x=sigmas,
                y=mus,
                mode="lines",
                line=dict(color="lightskyblue", width=2),
                name="Efficient frontier",
                customdata=weights,
                hovertemplate=hovertemplate,
            )
        )
        ax.add_trace(
            go.Scatter(
                x=[optimal_risk],
                y=[optimal_ret],
                customdata=[opt_weights, [sharpe_max]],
                mode="markers",
                name="Max Sharpe Portfolio",
                marker=dict(size=12, symbol="x", color="coral"),
                hovertemplate="Sharpe: %{{customdata[1]:.4}}<br>" + hovertemplate,
            )
        )
    else:
        ax.plot(sigmas, mus, label="Efficient frontier")
        ax.scatter(
            optimal_risk, optimal_ret, marker="x", s=100, color="r", label="optimal"
        )

    asset_mu = cla.expected_returns
    asset_sigma = np.sqrt(np.diag(cla.cov_matrix))
    if show_assets:
        if interactive:
            ax.add_trace(
                go.Scatter(
                    x=asset_sigma,
                    y=asset_mu,
                    mode="markers",
                    name="Assets",
                    marker=dict(size=10, symbol="star-diamond", color="silver"),
                    hovertemplate="Risk: %{x}<br>Return: %{y}<extra></extra>",
                )
            )
        else:
            ax.scatter(
                asset_sigma,
                asset_mu,
                s=30,
                color="k",
                label="assets",
            )
            if show_tickers:
                for i, label in enumerate(cla.tickers):
                    ax.annotate(label, (asset_sigma[i], asset_mu[i]))
    return ax


def _ef_default_returns_range(ef, points):
    """
    Helper function to generate a range of returns from the GMV returns to
    the maximum (constrained) returns
    """
    ef_minvol = ef.deepcopy()
    ef_maxret = ef.deepcopy()

    ef_minvol.min_volatility()
    min_ret = ef_minvol.portfolio_performance()[0]
    max_ret = ef_maxret._max_return()
    return np.linspace(min_ret, max_ret - 0.0001, points)


def _plot_ef(ef, ef_param, ef_param_range, ax, show_assets, show_tickers, interactive):
    """
    Helper function to plot the efficient frontier from an EfficientFrontier object
    """
    if interactive:
        go, _ = _get_plotly()

    mus, sigmas = [], []

    # Create a portfolio for each value of ef_param_range
    for param_value in ef_param_range:
        try:
            if ef_param == "utility":
                ef.max_quadratic_utility(param_value)
            elif ef_param == "risk":
                ef.efficient_risk(param_value)
            elif ef_param == "return":
                ef.efficient_return(param_value)
            else:
                raise NotImplementedError(
                    "ef_param should be one of {'utility', 'risk', 'return'}"
                )
        except exceptions.OptimizationError:
            continue
        except ValueError:
            warnings.warn(
                "Could not construct portfolio for parameter value {:.3f}".format(
                    param_value
                )
            )

        ret, sigma, _ = ef.portfolio_performance()
        mus.append(ret)
        sigmas.append(sigma)

    if interactive:
        ax.add_trace(
            go.Scatter(
                x=sigmas,
                y=mus,
                mode="lines",
                name="Efficient frontier",
                line=dict(width=2, color="lightskyblue"),
            )
        )
    else:
        ax.plot(sigmas, mus, label="Efficient frontier")

    asset_mu = ef.expected_returns
    asset_sigma = np.sqrt(np.diag(ef.cov_matrix))
    if show_assets:
        if interactive:
            ax.add_trace(
                go.Scatter(
                    x=asset_sigma,
                    y=asset_mu,
                    mode="markers",
                    marker=dict(size=10, symbol="star-diamond", color="silver"),
                )
            )
        else:
            ax.scatter(
                asset_sigma,
                asset_mu,
                s=30,
                color="k",
                label="assets",
            )
            if show_tickers:
                for i, label in enumerate(ef.tickers):
                    ax.annotate(label, (asset_sigma[i], asset_mu[i]))
    return ax


def plot_efficient_frontier(
    opt,
    ef_param="return",
    ef_param_range=None,
    points=100,
    ax=None,
    show_assets=True,
    show_tickers=False,
    interactive=False,
    **kwargs,
):
    """
    Plot the efficient frontier based on either a CLA or EfficientFrontier object.

    :param opt: an instantiated optimizer object BEFORE optimising an objective
    :type opt: EfficientFrontier or CLA
    :param ef_param: [EfficientFrontier] whether to use a range over utility, risk, or return.
                     Defaults to "return".
    :type ef_param: str, one of {"utility", "risk", "return"}.
    :param ef_param_range: the range of parameter values for ef_param.
                           If None, automatically compute a range from min->max return.
    :type ef_param_range: np.array or list (recommended to use np.arange or np.linspace)
    :param points: number of points to plot, defaults to 100. This is overridden if
                   an `ef_param_range` is provided explicitly.
    :type points: int, optional
    :param show_assets: whether we should plot the asset risks/returns also, defaults to True
    :type show_assets: bool, optional
    :param show_tickers: whether we should annotate each asset with its ticker, defaults to False
    :type show_tickers: bool, optional
    :param interactive: Switch rendering engine between Plotly and Matplotlib
    :type show_tickers: bool, optional
    :param filename: name of the file to save to, defaults to None (doesn't save)
    :type filename: str, optional
    :param showfig: whether to plt.show() the figure, defaults to False
    :type showfig: bool, optional
    :return: matplotlib axis
    :rtype: matplotlib.axes object
    """
    if interactive:
        go, _ = _get_plotly()
        ax = go.Figure()
    else:
        ax = ax or plt.gca()

    if isinstance(opt, CLA):
        ax = _plot_cla(
            opt,
            points,
            ax=ax,
            show_assets=show_assets,
            show_tickers=show_tickers,
            interactive=interactive,
        )
    elif isinstance(opt, EfficientFrontier):
        if ef_param_range is None:
            ef_param_range = _ef_default_returns_range(opt, points)

        ax = _plot_ef(
            opt,
            ef_param,
            ef_param_range,
            ax=ax,
            show_assets=show_assets,
            show_tickers=show_tickers,
            interactive=interactive,
        )
    else:
        raise NotImplementedError("Please pass EfficientFrontier or CLA object")

    if interactive:
        ax.update_layout(
            xaxis_title="Volatility",
            yaxis_title="Return",
        )
    else:
        ax.legend()
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Return")

        _plot_io(**kwargs)
    return ax


def plot_weights(weights, ax=None, **kwargs):
    """
    Plot the portfolio weights as a horizontal bar chart

    :param weights: the weights outputted by any PyPortfolioOpt optimizer
    :type weights: {ticker: weight} dict
    :param ax: ax to plot to, optional
    :type ax: matplotlib.axes
    :return: matplotlib axis
    :rtype: matplotlib.axes
    """
    ax = ax or plt.gca()

    desc = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in desc]
    vals = [i[1] for i in desc]

    y_pos = np.arange(len(labels))

    ax.barh(y_pos, vals)
    ax.set_xlabel("Weight")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    _plot_io(**kwargs)
    return ax
