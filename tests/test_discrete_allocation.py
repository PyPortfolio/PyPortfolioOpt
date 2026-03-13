from cvxpy.error import SolverError
import numpy as np
import pandas as pd
import pytest

from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from tests.utilities_for_tests import get_data, setup_efficient_frontier


def test_get_latest_prices():
    df = get_data()
    latest_prices = get_latest_prices(df)
    assert len(latest_prices) == 20
    assert list(latest_prices.index) == list(df.columns)
    assert latest_prices.name == pd.Timestamp(2018, 4, 11)


def test_get_latest_prices_error():
    df = get_data()
    with pytest.raises(TypeError):
        get_latest_prices(df.values)


def test_remove_zero_positions():
    raw = {"MA": 14, "FB": 12, "XOM": 0, "PFE": 51, "BABA": 5, "GOOG": 0}

    da = DiscreteAllocation({}, pd.Series(dtype=float))
    assert da._remove_zero_positions(raw) == {"MA": 14, "FB": 12, "PFE": 51, "BABA": 5}


def test_greedy_portfolio_allocation():
    ef = setup_efficient_frontier()
    w = ef.max_sharpe(risk_free_rate=0.02)

    df = get_data()
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices, short_ratio=0.3)
    allocation, leftover = da.greedy_portfolio()

    assert allocation == {
        "MA": 20,
        "FB": 12,
        "PFE": 54,
        "BABA": 4,
        "AAPL": 4,
        "BBY": 2,
        "SBUX": 1,
        "GOOG": 1,
    }

    total = 0
    for ticker, num in allocation.items():
        total += num * latest_prices[ticker]
    np.testing.assert_almost_equal(total + leftover, 10000, decimal=4)

    # Cover the verbose parameter,
    allocation_verbose, leftover_verbose = da.greedy_portfolio(verbose=True)
    assert allocation_verbose == allocation
    assert leftover_verbose == leftover


def test_greedy_allocation_rmse_error():
    ef = setup_efficient_frontier()
    w = ef.max_sharpe(risk_free_rate=0.02)

    df = get_data()
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices)
    da.greedy_portfolio()

    np.testing.assert_almost_equal(
        da._allocation_rmse_error(verbose=True), 0.017086185150415774
    )


def test_greedy_portfolio_allocation_short():
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    w = ef.max_sharpe(risk_free_rate=0.02)

    df = get_data()
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices, short_ratio=0.3)
    allocation, leftover = da.greedy_portfolio()

    assert allocation == {
        "MA": 19,
        "PFE": 42,
        "FB": 7,
        "GOOG": 1,
        "BABA": 5,
        "AAPL": 4,
        "SBUX": 8,
        "BBY": 6,
        "XOM": 4,
        "WMT": 3,
        "BAC": -32,
        "AMD": -48,
        "SHLD": -132,
        "GM": -9,
        "RRC": -19,
        "GE": -14,
        "T": -5,
        "UAA": -8,
    }
    long_total = 0
    short_total = 0
    for ticker, num in allocation.items():
        if num > 0:
            long_total += num * latest_prices[ticker]
        else:
            short_total -= num * latest_prices[ticker]

    np.testing.assert_almost_equal(
        long_total + short_total + leftover, 13000, decimal=4
    )

    # Cover the verbose parameter,
    allocation_verbose, leftover_verbose = da.greedy_portfolio(verbose=True)
    assert allocation_verbose == allocation
    assert leftover_verbose == leftover


def test_greedy_allocation_rmse_error_short():
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    w = ef.max_sharpe(risk_free_rate=0.02)

    df = get_data()
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices, short_ratio=0.3)
    da.greedy_portfolio()

    np.testing.assert_almost_equal(
        da._allocation_rmse_error(verbose=True), 0.06063511265243106
    )


def test_greedy_portfolio_allocation_short_different_params():
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    w = ef.max_sharpe(risk_free_rate=0.02)

    df = get_data()
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(
        w, latest_prices, total_portfolio_value=50000, short_ratio=0.5
    )
    allocation, leftover = da.greedy_portfolio()

    assert allocation == {
        "MA": 96,
        "PFE": 211,
        "FB": 34,
        "GOOG": 4,
        "BABA": 22,
        "AAPL": 17,
        "SBUX": 38,
        "AMZN": 2,
        "BBY": 27,
        "XOM": 19,
        "WMT": 10,
        "BAC": -269,
        "AMD": -399,
        "SHLD": -1099,
        "GM": -78,
        "RRC": -154,
        "GE": -119,
        "T": -41,
        "UAA": -64,
    }

    long_total = 0
    short_total = 0
    for ticker, num in allocation.items():
        if num > 0:
            long_total += num * latest_prices[ticker]
        else:
            short_total -= num * latest_prices[ticker]

    np.testing.assert_almost_equal(long_total + short_total + leftover, 75000)


def test_greedy_portfolio_allocation_short_different_params_reinvest():
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    w = ef.max_sharpe(risk_free_rate=0.02)

    df = get_data()
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(
        w, latest_prices, total_portfolio_value=50000, short_ratio=0.5
    )
    allocation, leftover = da.greedy_portfolio(reinvest=True)

    assert allocation == {
        "MA": 145,
        "PFE": 317,
        "FB": 53,
        "GOOG": 6,
        "BABA": 34,
        "AAPL": 27,
        "SBUX": 58,
        "AMZN": 2,
        "BBY": 41,
        "XOM": 30,
        "WMT": 17,
        "JPM": 1,
        "BAC": -269,
        "AMD": -399,
        "SHLD": -1099,
        "GM": -78,
        "RRC": -154,
        "GE": -119,
        "T": -41,
        "UAA": -64,
    }

    long_total = 0
    short_total = 0
    for ticker, num in allocation.items():
        if num > 0:
            long_total += num * latest_prices[ticker]
        else:
            short_total -= num * latest_prices[ticker]

    np.testing.assert_almost_equal(long_total + short_total + leftover, 100000)


def test_lp_portfolio_allocation():
    ef = setup_efficient_frontier()
    w = ef.max_sharpe(risk_free_rate=0.02)

    df = get_data()
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices, short_ratio=0.3)
    allocation, leftover = da.lp_portfolio()

    # Gives different answers on different machines
    # assert allocation == {
    #     "AMD": 1,
    #     "GOOG": 1,
    #     "AAPL": 4,
    #     "FB": 12,
    #     "BABA": 4,
    #     "BBY": 1,
    #     "MA": 20,
    #     "PFE": 54,
    #     "SBUX": 1,
    # } or allocation == {
    #     "GOOG": 1,
    #     "AAPL": 4,
    #     "FB": 12,
    #     "BABA": 4,
    #     "BBY": 1,
    #     "MA": 20,
    #     "PFE": 54,
    #     "SBUX": 1,
    # }

    total = 0
    for ticker, num in allocation.items():
        total += num * latest_prices[ticker]
    np.testing.assert_almost_equal(total + leftover, 10000, decimal=4)

    # # Cover the verbose parameter,
    # allocation_verbose, leftover_verbose = da.lp_portfolio(verbose=True)
    # assert allocation_verbose == allocation
    # assert leftover_verbose == leftover


def test_lp_allocation_rmse_error():
    ef = setup_efficient_frontier()
    w = ef.max_sharpe(risk_free_rate=0.02)

    df = get_data()
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices, short_ratio=0.3)
    da.lp_portfolio()
    assert da._allocation_rmse_error(verbose=True) < 0.02


def test_lp_portfolio_allocation_short():
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    w = ef.max_sharpe(risk_free_rate=0.02)

    df = get_data()
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices, short_ratio=0.3)
    allocation, leftover = da.lp_portfolio()

    # assert allocation == {
    #     "GOOG": 1,
    #     "AAPL": 4,
    #     "FB": 7,
    #     "BABA": 5,
    #     "WMT": 3,
    #     "XOM": 4,
    #     "BBY": 6,
    #     "MA": 19,
    #     "PFE": 42,
    #     "SBUX": 8,
    #     "GE": -14,
    #     "AMD": -48,
    #     "BAC": -32,
    #     "GM": -9,
    #     "T": -5,
    #     "UAA": -8,
    #     "SHLD": -132,
    #     "RRC": -19,
    # }

    long_total = 0
    short_total = 0
    for ticker, num in allocation.items():
        if num > 0:
            long_total += num * latest_prices[ticker]
        else:
            short_total -= num * latest_prices[ticker]

    np.testing.assert_almost_equal(
        long_total + short_total + leftover, 13000, decimal=4
    )

    # # Cover the verbose parameter,
    # allocation_verbose, leftover_verbose = da.lp_portfolio(verbose=True)
    # assert allocation_verbose == allocation
    # assert leftover_verbose == leftover


def test_lp_portfolio_allocation_short_reinvest():
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    w = ef.max_sharpe(risk_free_rate=0.02)

    df = get_data()
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices, short_ratio=0.3)
    allocation, leftover = da.lp_portfolio(reinvest=True)

    # assert allocation == {
    #     "GOOG": 1,
    #     "AAPL": 5,
    #     "FB": 10,
    #     "BABA": 6,
    #     "WMT": 3,
    #     "XOM": 6,
    #     "BBY": 7,
    #     "MA": 26,
    #     "PFE": 55,
    #     "SBUX": 10,
    #     "JPM": 1,
    #     "GE": -14,
    #     "AMD": -48,
    #     "BAC": -32,
    #     "GM": -9,
    #     "T": -5,
    #     "UAA": -8,
    #     "SHLD": -132,
    #     "RRC": -19,
    # }

    long_total = 0
    short_total = 0
    for ticker, num in allocation.items():
        if num > 0:
            long_total += num * latest_prices[ticker]
        else:
            short_total -= num * latest_prices[ticker]

    np.testing.assert_almost_equal(
        long_total + short_total + leftover, 16000, decimal=5
    )


def test_lp_allocation_rmse_error_short():
    ef = setup_efficient_frontier(weight_bounds=(-1, 1))
    w = ef.max_sharpe(risk_free_rate=0.02)

    df = get_data()
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices, short_ratio=0.3)
    da.lp_portfolio()
    assert da._allocation_rmse_error(verbose=True) < 0.1


def test_lp_portfolio_allocation_different_params():
    ef = setup_efficient_frontier()
    w = ef.max_sharpe(risk_free_rate=0.02)

    df = get_data()
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(
        w, latest_prices, total_portfolio_value=80000, short_ratio=0.4
    )
    allocation, leftover = da.lp_portfolio()

    # assert allocation == {
    #     "GOOG": 3,
    #     "AAPL": 32,
    #     "FB": 99,
    #     "BABA": 34,
    #     "AMZN": 2,
    #     "BBY": 15,
    #     "MA": 164,
    #     "PFE": 438,
    #     "SBUX": 15,
    # }

    total = 0
    for ticker, num in allocation.items():
        total += num * latest_prices[ticker]
    np.testing.assert_almost_equal(total + leftover, 80000, decimal=4)


def test_rmse_decreases_with_value():
    # As total_portfolio_value increases, rmse should decrease.
    ef = setup_efficient_frontier()
    w = ef.max_sharpe(risk_free_rate=0.02)

    df = get_data()
    latest_prices = get_latest_prices(df)

    da1 = DiscreteAllocation(w, latest_prices, total_portfolio_value=10000)
    da1.greedy_portfolio()
    rmse1 = da1._allocation_rmse_error(verbose=True)

    da2 = DiscreteAllocation(w, latest_prices, total_portfolio_value=100000)
    da2.greedy_portfolio()
    rmse2 = da2._allocation_rmse_error(verbose=True)

    assert rmse2 < rmse1


def test_allocation_errors():
    ef = setup_efficient_frontier()
    w = ef.max_sharpe(risk_free_rate=0.02)

    df = get_data()
    latest_prices = get_latest_prices(df)

    assert DiscreteAllocation(w, latest_prices)
    with pytest.raises(TypeError):
        DiscreteAllocation(ef.weights, latest_prices)
    with pytest.raises(TypeError):
        DiscreteAllocation(w, latest_prices.values.tolist())
    with pytest.raises(ValueError):
        DiscreteAllocation(w, latest_prices, total_portfolio_value=0)
    with pytest.raises(ValueError):
        DiscreteAllocation(w, latest_prices, short_ratio=-0.4)
    with pytest.raises(SolverError):
        da = DiscreteAllocation(w, latest_prices)
        da.lp_portfolio(solver="ABCDEF")

    w2 = w.copy()
    w2["AAPL"] = np.nan
    with pytest.raises(ValueError):
        DiscreteAllocation(w2, latest_prices)
    latest_prices.iloc[0] = np.nan
    with pytest.raises(TypeError):
        DiscreteAllocation(w, latest_prices)
