import json
from importlib.resources import files

import pandas as pd

__all__ = [
    "load_example_market_caps",
    "load_example_prices",
    "load_spy_prices",
]


def _load_csv(filename: str) -> pd.DataFrame:
    resource = files(__package__).joinpath(filename)
    with resource.open("rb") as handle:
        return pd.read_csv(handle, parse_dates=["date"], index_col="date")


def load_example_prices() -> pd.DataFrame:
    return _load_csv("stock_prices.csv")


def load_spy_prices() -> pd.DataFrame:
    return _load_csv("spy_prices.csv")


def load_example_market_caps() -> dict[str, int]:
    resource = files(__package__).joinpath("example_market_caps.json")
    with resource.open("r", encoding="utf-8") as handle:
        market_caps = json.load(handle)
    return {ticker: int(value) for ticker, value in market_caps.items()}
