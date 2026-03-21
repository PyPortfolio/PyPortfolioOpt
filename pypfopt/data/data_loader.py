from importlib import resources

import pandas as pd


def _load_raw_data(filename: str, **read_csv_kwargs):
    with resources.files(__package__).joinpath(filename).open("r") as f:
        return pd.read_csv(f, **read_csv_kwargs)


def load_stockdata(tickers: list = None, start: str = None, end: str = None):
    """
    Load example stock price data.

    This function loads a synthetic stock price dataset included with the
    package. The data can optionally be filtered by ticker symbols and
    date range.

    Parameters
    ----------
    tickers : list of str, optional
        List of ticker symbols to include. If ``None``, all available
        tickers are returned.
    start : str, optional
        Start date for filtering the dataset (inclusive). Should be
        interpretable by ``pandas.to_datetime``.
    end : str, optional
        End date for filtering the dataset (inclusive). Should be
        interpretable by ``pandas.to_datetime``.

    Returns
    -------
    pandas.DataFrame
        DataFrame of stock prices indexed by date. Columns correspond to
        ticker symbols and values represent price levels.

    Notes
    -----
    The dataset is bundled with the package and does not rely on external
    data sources. It is intended for examples and tutorials.
    """
    df = _load_raw_data("stock_prices.csv", parse_dates=["date"])

    if start is not None:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["date"] <= pd.to_datetime(end)]

    if tickers is not None:
        cols = ["date"] + tickers
        df = df[cols]

    return df.set_index("date")


def load_marketcaps(tickers: list = None):
    """
    Load bundled example market capitalisation data.

    This function loads synthetic market capitalisation values for the
    example assets included in the package.

    Parameters
    ----------
    tickers : list of str, optional
        List of ticker symbols to return. If ``None``, market caps for all
        available tickers are returned.

    Returns
    -------
    dict
        Dictionary mapping ticker symbols to market capitalisation values.

    Notes
    -----
    The values are synthetic and provided solely for use in examples
    demonstrating portfolio optimisation methods.
    """
    df = _load_raw_data("market_caps.csv")

    if tickers is not None:
        available = set(df["ticker"])
        invalid = set(tickers) - available
        if invalid:
            raise ValueError(f"Invalid tickers: {invalid}")

        df = df[df["ticker"].isin(tickers)]

    return dict(zip(df["ticker"], df["market_cap"]))


def available_tickers():
    """
    Return the list of available ticker symbols.

    Returns
    -------
    list of str
        Sorted list of ticker symbols present in the bundled example
        dataset.

    Notes
    -----
    These tickers correspond to the columns available in the example
    stock price dataset returned by :func:`load_stockdata`.
    """
    cols = [
        "AAPL",
        "ACN",
        "AMD",
        "AMZN",
        "BAC",
        "BLK",
        "COST",
        "CVS",
        "DIS",
        "DPZ",
        "F",
        "GILD",
        "INTU",
        "JD",
        "JPM",
        "KO",
        "LUV",
        "MA",
        "MCD",
        "MSFT",
        "NAT",
        "NVDA",
        "PBI",
        "PFE",
        "SBUX",
        "SPY",
        "TGT",
        "TM",
        "TSLA",
        "UL",
        "UNH",
        "WMT",
        "XOM",
    ]

    return cols
