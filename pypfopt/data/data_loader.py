import pandas as pd
from importlib import resources


def _load_raw_data(filename: str, **read_csv_kwargs):
    with resources.files(__package__).joinpath(filename).open("r") as f:
        return pd.read_csv(f, **read_csv_kwargs)


def load_stockdata(tickers: list = None, start: str = None, end: str = None):

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

    df = _load_raw_data("market_caps.csv")

    if tickers is not None:
        available = set(df["ticker"])
        invalid = set(tickers) - available
        if invalid:
            raise ValueError(f"Invalid tickers: {invalid}")

        df = df[df["ticker"].isin(tickers)]

    return dict(zip(df["ticker"], df["market_cap"]))


def available_tickers():

    df = _load_raw_data("stock_prices.csv", parse_dates=["date"])
    cols = [c for c in df.columns if c != "date"]
    cols.sort()

    return cols
