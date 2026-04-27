from pypfopt.data import (
    load_cookbook_prices,
    load_example_market_caps,
    load_example_prices,
    load_spy_prices,
)

MEAN_VARIANCE_TICKERS = {
    "ACN",
    "AMZN",
    "COST",
    "DIS",
    "F",
    "GILD",
    "JPM",
    "KO",
    "LUV",
    "MA",
    "MSFT",
    "PFE",
    "TSLA",
    "UNH",
    "XOM",
}

ADVANCED_AND_HRP_TICKERS = {
    "AAPL",
    "AMD",
    "BAC",
    "BLK",
    "CVS",
    "DIS",
    "INTU",
    "JD",
    "MA",
    "NVDA",
    "PBI",
    "TGT",
    "TM",
    "UL",
    "WMT",
}

BLACK_LITTERMAN_TICKERS = {
    "AMZN",
    "BAC",
    "COST",
    "DIS",
    "DPZ",
    "KO",
    "MCD",
    "MSFT",
    "NAT",
    "SBUX",
}


def test_load_example_prices():
    prices = load_example_prices()

    assert prices.index.name == "date"
    assert str(prices.index.dtype).startswith("datetime64")
    assert {"AAPL", "AMD", "AMZN", "META", "WMT", "XOM"}.issubset(prices.columns)


def test_load_cookbook_prices():
    prices = load_cookbook_prices()
    expected_tickers = (
        MEAN_VARIANCE_TICKERS | ADVANCED_AND_HRP_TICKERS | BLACK_LITTERMAN_TICKERS
    )
    mean_variance_prices = (
        prices[sorted(MEAN_VARIANCE_TICKERS)].loc["1990":].dropna(how="all")
    )

    assert prices.index.name == "date"
    assert str(prices.index.dtype).startswith("datetime64")
    assert expected_tickers.issubset(prices.columns)
    assert mean_variance_prices.shape[1] == len(MEAN_VARIANCE_TICKERS)
    assert not mean_variance_prices.empty


def test_load_spy_prices():
    prices = load_spy_prices()

    assert prices.index.name == "date"
    assert str(prices.index.dtype).startswith("datetime64")
    assert list(prices.columns) == ["SPY"]


def test_load_example_market_caps():
    market_caps = load_example_market_caps()

    assert set(market_caps) == BLACK_LITTERMAN_TICKERS
    assert all(value > 0 for value in market_caps.values())
