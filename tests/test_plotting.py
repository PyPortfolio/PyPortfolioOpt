import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pypfopt import (
    CLA,
    EfficientFrontier,
    HRPOpt,
    expected_returns,
    plotting,
    risk_models,
)
from tests.utilities_for_tests import get_data, setup_efficient_frontier


def test_correlation_plot():
    plt.figure()
    df = get_data()
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    ax = plotting.plot_covariance(S, showfig=False)
    assert len(ax.findobj()) > 250
    plt.clf()
    ax = plotting.plot_covariance(S, plot_correlation=True, showfig=False)
    assert len(ax.findobj()) > 250
    plt.clf()
    ax = plotting.plot_covariance(S, show_tickers=False, showfig=False)
    assert len(ax.findobj()) > 130
    plt.clf()
    ax = plotting.plot_covariance(
        S, plot_correlation=True, show_tickers=False, showfig=False
    )
    assert len(ax.findobj()) > 130
    plt.clf()

    temp_folder = tempfile.TemporaryDirectory()
    temp_folder_path = temp_folder.name
    plot_filename = os.path.join(temp_folder_path, "plot.png")
    ax = plotting.plot_covariance(S, filename=plot_filename, showfig=False)
    assert len(ax.findobj()) > 250
    assert os.path.exists(plot_filename)
    assert os.path.getsize(plot_filename) > 0
    temp_folder.cleanup()
    plt.clf()
    plt.close()


def test_dendrogram_plot():
    plt.figure()
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    hrp = HRPOpt(returns)
    hrp.optimize()

    ax = plotting.plot_dendrogram(hrp, showfig=False)
    assert len(ax.findobj()) > 180
    assert isinstance(ax.findobj()[0], matplotlib.collections.LineCollection)
    plt.clf()

    ax = plotting.plot_dendrogram(hrp, show_tickers=False, showfig=False)
    assert len(ax.findobj()) > 60
    assert isinstance(ax.findobj()[0], matplotlib.collections.LineCollection)
    plt.clf()
    plt.close()

    # Test that passing an unoptimized HRPOpt works, but issues a warning as
    #  this should already have been optimized according to the API.
    hrp = HRPOpt(returns)
    with pytest.warns(RuntimeWarning) as w:
        ax = plotting.plot_dendrogram(hrp, show_tickers=False, showfig=False)
        assert len(w) <= 2  # the second is FutureWarning if exists
        assert (
            str(w[0].message)
            == "hrp param has not been optimized.  Attempting optimization."
        )
        assert len(ax.findobj()) > 60
        assert isinstance(ax.findobj()[0], matplotlib.collections.LineCollection)
    plt.clf()
    plt.close()


def test_cla_plot():
    plt.figure()
    df = get_data()
    rets = expected_returns.mean_historical_return(df)
    S = risk_models.exp_cov(df)
    cla = CLA(rets, S)

    ax = plotting.plot_efficient_frontier(cla, showfig=False)
    assert len(ax.findobj()) > 130
    plt.clf()

    ax = plotting.plot_efficient_frontier(cla, show_assets=False, showfig=False)
    assert len(ax.findobj()) > 150
    plt.clf()
    plt.close()


def test_cla_plot_ax():
    plt.figure()
    df = get_data()
    rets = expected_returns.mean_historical_return(df)
    S = risk_models.exp_cov(df)
    cla = CLA(rets, S)

    fig, ax = plt.subplots(figsize=(12, 10))
    plotting.plot_efficient_frontier(cla, ax=ax)
    assert len(ax.findobj()) > 130
    plt.close()
    plt.close()


def test_default_ef_plot():
    plt.figure()
    ef = setup_efficient_frontier()
    ax = plotting.plot_efficient_frontier(ef, show_assets=True)
    assert len(ax.findobj()) > 120
    plt.clf()

    # with constraints
    ef = setup_efficient_frontier()
    ef.add_constraint(lambda x: x <= 0.15)
    ef.add_constraint(lambda x: x[0] == 0.05)
    ax = plotting.plot_efficient_frontier(ef)
    assert len(ax.findobj()) > 120
    plt.clf()
    plt.close()


def test_default_ef_plot_labels():
    plt.figure()
    ef = setup_efficient_frontier()
    ax = plotting.plot_efficient_frontier(ef, show_assets=True, show_tickers=True)
    assert len(ax.findobj()) > 125
    plt.clf()


def test_ef_plot_utility():
    plt.figure()
    ef = setup_efficient_frontier()
    delta_range = np.arange(0.001, 50, 1)
    ax = plotting.plot_efficient_frontier(
        ef, ef_param="utility", ef_param_range=delta_range, showfig=False
    )
    assert len(ax.findobj()) > 120
    plt.clf()
    plt.close()


def test_ef_plot_errors():
    plt.figure()
    ef = setup_efficient_frontier()
    delta_range = np.arange(0.001, 50, 1)
    # Test invalid ef_param
    with pytest.raises(NotImplementedError):
        plotting.plot_efficient_frontier(
            ef, ef_param="blah", ef_param_range=delta_range, showfig=False
        )
    # Test invalid optimizer
    with pytest.raises(NotImplementedError):
        plotting.plot_efficient_frontier(
            None, ef_param_range=delta_range, showfig=False
        )
    plt.clf()
    plt.close()


def test_ef_plot_risk():
    plt.figure()
    ef = setup_efficient_frontier()
    ef.min_volatility()
    min_risk = ef.portfolio_performance()[1]

    ef = setup_efficient_frontier()
    risk_range = np.linspace(min_risk + 0.05, 0.5, 30)
    ax = plotting.plot_efficient_frontier(
        ef, ef_param="risk", ef_param_range=risk_range, showfig=False
    )
    assert len(ax.findobj()) > 120
    plt.clf()
    plt.close()


def test_ef_plot_return():
    plt.figure()
    ef = setup_efficient_frontier()
    # Internally _max_return() is used, so subtract epsilon
    max_ret = ef.expected_returns.max() - 0.0001
    return_range = np.linspace(0, max_ret, 30)
    ax = plotting.plot_efficient_frontier(
        ef, ef_param="return", ef_param_range=return_range, showfig=False
    )
    assert len(ax.findobj()) > 120
    plt.clf()
    plt.close()


def test_ef_plot_utility_short():
    plt.figure()
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(None, None)
    )
    delta_range = np.linspace(0.001, 20, 50)
    ax = plotting.plot_efficient_frontier(
        ef, ef_param="utility", ef_param_range=delta_range, showfig=False
    )
    assert len(ax.findobj()) > 150
    plt.clf()
    plt.close()


def test_constrained_ef_plot_utility():
    plt.figure()
    ef = setup_efficient_frontier()
    ef.add_constraint(lambda w: w[0] >= 0.2)
    ef.add_constraint(lambda w: w[2] == 0.15)
    ef.add_constraint(lambda w: w[3] + w[4] <= 0.10)

    delta_range = np.linspace(0.001, 20, 50)
    ax = plotting.plot_efficient_frontier(
        ef, ef_param="utility", ef_param_range=delta_range, showfig=False
    )
    assert len(ax.findobj()) > 120
    plt.clf()
    plt.close()


def test_constrained_ef_plot_risk():
    plt.figure()
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(None, None)
    )

    ef.add_constraint(lambda w: w[0] >= 0.2)
    ef.add_constraint(lambda w: w[2] == 0.15)
    ef.add_constraint(lambda w: w[3] + w[4] <= 0.10)

    # 100 portfolios with risks between 0.10 and 0.30
    risk_range = np.linspace(0.157, 0.40, 50)
    ax = plotting.plot_efficient_frontier(
        ef, ef_param="risk", ef_param_range=risk_range, show_assets=True, showfig=False
    )
    assert len(ax.findobj()) > 130
    plt.clf()
    plt.close()


def test_weight_plot():
    plt.figure()
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    hrp = HRPOpt(returns)
    w = hrp.optimize()

    ax = plotting.plot_weights(w, showfig=False)
    assert len(ax.findobj()) > 190
    plt.clf()
    plt.close()


def test_weight_plot_multi():
    ef = setup_efficient_frontier()
    w1 = ef.min_volatility()
    ef = setup_efficient_frontier()
    w2 = ef.max_sharpe()

    fig, (ax1, ax2) = plt.subplots(2)
    plotting.plot_weights(w1, ax1, showfig=False)
    plotting.plot_weights(w2, ax2, showfig=False)

    assert len(fig.axes) == 2
    assert len(fig.axes[0].findobj()) > 200
    assert len(fig.axes[1].findobj()) > 200
    plt.close()


def test_weight_plot_add_attribute():
    plt.figure()

    ef = setup_efficient_frontier()
    w = ef.min_volatility()
    ax = plotting.plot_weights(w)
    ax.set_title("Test")
    assert len(ax.findobj()) > 200
    plt.close()


def test_plotting_edge_case():
    # raised in issue #333
    mu = pd.Series([0.043389, 0.036194])
    S = pd.DataFrame([[0.000562, 0.002273], [0.002273, 0.027710]])
    ef = EfficientFrontier(mu, S)
    fig, ax = plt.subplots()

    with pytest.warns(UserWarning):
        plotting.plot_efficient_frontier(
            ef,
            ef_param="return",
            ef_param_range=np.linspace(0.036194, 0.043389, 10),
            ax=ax,
            show_assets=False,
        )


def test_plot_efficient_frontier():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    optimal_ret, optimal_risk, _ = ef.portfolio_performance(risk_free_rate=0.02)
