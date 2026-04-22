from pypfopt.data import (
    load_example_market_caps,
    load_example_prices,
    load_spy_prices,
)


def test_load_example_prices():
    prices = load_example_prices()

    assert prices.index.name == "date"
    assert str(prices.index.dtype).startswith("datetime64")
    assert {"AAPL", "AMD", "AMZN", "META", "WMT", "XOM"}.issubset(prices.columns)


def test_load_spy_prices():
    prices = load_spy_prices()

    assert prices.index.name == "date"
    assert str(prices.index.dtype).startswith("datetime64")
    assert list(prices.columns) == ["SPY"]


def test_load_example_market_caps():
    market_caps = load_example_market_caps()

    assert set(market_caps) == {
        "AAPL",
        "AMZN",
        "BAC",
        "GOOG",
        "JPM",
        "MA",
        "META",
        "PFE",
        "SBUX",
        "WMT",
    }
    assert all(value > 0 for value in market_caps.values())
