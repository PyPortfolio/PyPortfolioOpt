import numpy as np
import pandas as pd

from pypfopt import (
    CLA,
    BlackLittermanModel,
    EfficientFrontier,
    HRPOpt,
    black_litterman,
    expected_returns,
    plotting,
    risk_models,
)

# Reading in the data; preparing expected returns and a risk model
df = pd.read_csv("tests/resources/stock_prices.csv", parse_dates=True, index_col="date")
returns = df.pct_change().dropna()
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)


# Now try with a nonconvex objective from  Kolm et al (2014)
def deviation_risk_parity(w, cov_matrix):
    diff = w * np.dot(cov_matrix, w) - (w * np.dot(cov_matrix, w)).reshape(-1, 1)
    return (diff**2).sum().sum()


ef = EfficientFrontier(mu, S)
weights = ef.nonconvex_objective(deviation_risk_parity, ef.cov_matrix)
ef.portfolio_performance(verbose=True)

"""
Expected annual return: 22.9%
Annual volatility: 19.2%
Sharpe Ratio: 1.09
"""

# Black-Litterman
spy_prices = pd.read_csv(
    "tests/resources/spy_prices.csv", parse_dates=True, index_col=0, squeeze=True
)
delta = black_litterman.market_implied_risk_aversion(spy_prices)

mcaps = {
    "GOOG": 927e9,
    "AAPL": 1.19e12,
    "FB": 574e9,
    "BABA": 533e9,
    "AMZN": 867e9,
    "GE": 96e9,
    "AMD": 43e9,
    "WMT": 339e9,
    "BAC": 301e9,
    "GM": 51e9,
    "T": 61e9,
    "UAA": 78e9,
    "SHLD": 0,
    "XOM": 295e9,
    "RRC": 1e9,
    "BBY": 22e9,
    "MA": 288e9,
    "PFE": 212e9,
    "JPM": 422e9,
    "SBUX": 102e9,
}
prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

# 1. SBUX will drop by 20%
# 2. GOOG outperforms FB by 10%
# 3. BAC and JPM will outperform T and GE by 15%
views = np.array([-0.20, 0.10, 0.15]).reshape(-1, 1)
picking = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -0.5, 0, 0, 0.5, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
    ]
)
bl = BlackLittermanModel(S, Q=views, P=picking, pi=prior, tau=0.01)
rets = bl.bl_returns()
ef = EfficientFrontier(rets, S)
ef.max_sharpe()
print(ef.clean_weights())
ef.portfolio_performance(verbose=True)

"""
{'GOOG': 0.2015,
 'AAPL': 0.2368,
 'FB': 0.0,
 'BABA': 0.06098,
 'AMZN': 0.17148,
 'GE': 0.0,
 'AMD': 0.0,
 'WMT': 0.0,
 'BAC': 0.18545,
 'GM': 0.0,
 'T': 0.0,
 'UAA': 0.0,
 'SHLD': 0.0,
 'XOM': 0.0,
 'RRC': 0.0,
 'BBY': 0.0,
 'MA': 0.0,
 'PFE': 0.0,
 'JPM': 0.14379,
 'SBUX': 0.0}

Expected annual return: 15.3%
Annual volatility: 28.7%
Sharpe Ratio: 0.46
"""


# Hierarchical risk parity
hrp = HRPOpt(returns)
weights = hrp.optimize()
hrp.portfolio_performance(verbose=True)
print(weights)
plotting.plot_dendrogram(hrp)  # to plot dendrogram

"""
Expected annual return: 10.8%
Annual volatility: 13.2%
Sharpe Ratio: 0.66

{'AAPL': 0.022258941278778397,
 'AMD': 0.02229402179669211,
 'AMZN': 0.016086842079875,
 'BABA': 0.07963382071794091,
 'BAC': 0.014409222455552262,
 'BBY': 0.0340641943824504,
 'FB': 0.06272994714663534,
 'GE': 0.05519063444162849,
 'GM': 0.05557666024185722,
 'GOOG': 0.049560084289929286,
 'JPM': 0.017675709092515708,
 'MA': 0.03812737349732021,
 'PFE': 0.07786528342813454,
 'RRC': 0.03161528695094597,
 'SBUX': 0.039844436656239136,
 'SHLD': 0.027113184241298865,
 'T': 0.11138956508836476,
 'UAA': 0.02711590957075009,
 'WMT': 0.10569551148587905,
 'XOM': 0.11175337115721229}
"""


# Crticial Line Algorithm
cla = CLA(mu, S)
print(cla.max_sharpe())
cla.portfolio_performance(verbose=True)
plotting.plot_efficient_frontier(cla)  # to plot

"""
{'GOOG': 0.020889868669945022,
 'AAPL': 0.08867994115132602,
 'FB': 0.19417572932251745,
 'BABA': 0.10492386821217001,
 'AMZN': 0.0644908140418782,
 'GE': 0.0,
 'AMD': 0.0,
 'WMT': 0.0034898157701416382,
 'BAC': 0.0,
 'GM': 0.0,
 'T': 2.4138966206946562e-19,
 'UAA': 0.0,
 'SHLD': 0.0,
 'XOM': 0.0005100736411646903,
 'RRC': 0.0,
 'BBY': 0.05967818998203106,
 'MA': 0.23089949598834422,
 'PFE': 0.19125123325029705,
 'JPM': 0.0,
 'SBUX': 0.041010969970184656}

Expected annual return: 32.5%
Annual volatility: 21.3%
Sharpe Ratio: 1.43
"""
