import yfinance as yf

if __name__ == "__main__":
    tickers = [
        "AAPL",
        "AMD",
        "AMZN",
        "BABA",
        "BAC",
        "BBY",
        "GE",
        "GM",
        "GOOG",
        "JPM",
        "MA",
        "META",
        "PFE",
        "RRC",
        "SBUX",
        "T",
        "UAA",
        "WMT",
        "XOM",
        "MSFT",
        "KO",
        "COST",
        "LUV",
        "UNH",
        "ACN",
        "DIS",
        "GILD",
        "F",
        "TSLA",
        "BLK",
        "TM",
        "JD",
        "INTU",
        "UL",
        "CVS",
        "NVDA",
        "PBI",
        "TGT",
        "NAT",
        "DPZ",
        "MCD",
        "SPY",
        "FB",
        "IBM",
        "NFLX",
        "QCOM",
        "VZ",
    ]

    data = yf.download(tickers, start="1990-01-01")["Close"]
    prices = data.to_csv("data/prices.csv")

    for ticker in tickers:
        assert ticker in data.columns, f"Missing data for {ticker}"
