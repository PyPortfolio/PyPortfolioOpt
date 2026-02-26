.. image:: ../media/logo_v1-grey.png
   :scale: 40 %
   :align: center
   :alt: PyPortfolioOpt

.. raw:: html

    <meta prefix="og: http://ogp.me/ns#" property="og:title" content="PyPortfolioOpt" />
    <meta prefix="og: http://ogp.me/ns#" property="og:description" content="Portfolio optimization in python" />
    <meta prefix="og: http://ogp.me/ns#" property="og:image" content="https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/media/logo_v1.png"/>

    <embed>
        <p align="center">
            <a href="https://www.python.org/">
                <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
                    alt="python"></a> &nbsp;
            <a href="https://pypi.org/project/PyPortfolioOpt/">
                <img src="https://img.shields.io/badge/pypi-v1.5.6-brightgreen.svg"
                    alt="python"></a> &nbsp;
            <a href="https://opensource.org/licenses/MIT">
                <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg"
                    alt="MIT license"></a> &nbsp;
            <a href="https://github.com/pyportfolio/pyportfolioopt/graphs/commit-activity">
                <img src="https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg"
                    alt="MIT license"></a> &nbsp;
            <a style="border-width:0" href="https://doi.org/10.21105/joss.03066">
                <img src="https://joss.theoj.org/papers/10.21105/joss.03066/status.svg"
                    alt="DOI badge" ></a> &nbsp;
        </p>
    </embed>


PyPortfolioOpt is a library that implements portfolio optimization methods, including
classical efficient frontier techniques and Black-Litterman allocation, as well as more
recent developments in the field like shrinkage and Hierarchical Risk Parity, along with
some novel experimental features like exponentially-weighted covariance matrices.

It is **extensive** yet easily
**extensible**, and can be useful for both the casual investor and the serious
practitioner. Whether you are a fundamentals-oriented investor who has identified a
handful of undervalued picks, or an algorithmic trader who has a basket of
strategies, PyPortfolioOpt can help you combine your alpha sources
in a risk-efficient way.


Installation
============

If you would like to play with PyPortfolioOpt interactively in your browser, you may launch Binder
`here <https://mybinder.org/v2/gh/pyportfolio/pyportfolioopt/main/?filepath=cookbook>`__. It takes a
while to set up, but it lets you try out the cookbook recipes without having to install anything.

Prior to installing PyPortfolioOpt, you need to install C++. On macOS, this means that you need
to install XCode Command Line Tools (see `here <https://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/>`__).

For Windows users, download Visual Studio `here <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`__,
with additional instructions `here <https://docs.google.com/presentation/d/0B4GsMXCRaSSIOWpYQkstajlYZ0tPVkNQSElmTWh1dXFaYkJr/edit?usp=sharing&ouid=117107708911390632479&resourcekey=0-HEezB2NFstz1GjKDkroJSQ&rtpof=true&sd=true>`__.

Installation can then be done via pip::

    pip install PyPortfolioOpt

(you may need to follow separate installation instructions for `cvxopt <https://cvxopt.org/install/index.html#>`__ and `cvxpy <https://www.cvxpy.org/install/>`__).

For the sake of best practice, it is good to do this with a dependency manager. I suggest you
set yourself up with `poetry <https://github.com/sdispater/poetry>`_, then within a new poetry project
run:

.. code-block:: text

    poetry add PyPortfolioOpt

The alternative is to clone/download the project, then in the project directory run

.. code-block:: text

    python setup.py install

Thanks to Thomas Schmelzer, PyPortfolioOpt now supports Docker (requires 
**make**, **docker**, **docker-compose**). Build your first container with 
``make build``; run tests with ``make test``. For more information, please read
`this guide <https://docker-curriculum.com/#introduction>`_.


.. note::
    If any of these methods don't work, please `raise an issue
    <https://github.com/pyportfolio/pyportfolioopt/issues>`_ with the 'packaging' label on GitHub



For developers
--------------

If you are planning on using PyPortfolioOpt as a starting template for significant
modifications, it probably makes sense to clone the repository and to just use the
source code

.. code-block:: text

    git clone https://github.com/pyportfolio/pyportfolioopt

Alternatively, if you still want the convenience of a global ``from pypfopt import x``,
you should try

.. code-block:: text

    pip install -e git+https://github.com/pyportfolio/pyportfolioopt.git


A Quick Example
===============

This section contains a quick look at what PyPortfolioOpt can do. For a guided tour,
please check out the :ref:`user-guide`. For even more examples, check out the Jupyter
notebooks in the `cookbook <https://github.com/pyportfolio/pyportfolioopt/tree/main/cookbook>`_.

If you already have expected returns ``mu`` and a risk model ``S`` for your set of
assets, generating an optimal portfolio is as easy as::

    from pypfopt.efficient_frontier import EfficientFrontier

    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()

However, if you would like to use PyPortfolioOpt's built-in methods for
calculating the expected returns and covariance matrix from historical data,
that's fine too::

    import pandas as pd
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns

    # Read in price data
    df = pd.read_csv("tests/resources/stock_prices.csv", parse_dates=True, index_col="date")

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    ef.portfolio_performance(verbose=True)

This outputs the following:

.. code-block:: text

   Expected annual return: 33.0%
   Annual volatility: 21.7%
   Sharpe Ratio: 1.43


Contents
========

.. toctree::
    :maxdepth: 2

    UserGuide
    ExpectedReturns
    RiskModels
    MeanVariance
    GeneralEfficientFrontier
    BlackLitterman
    OtherOptimizers
    Postprocessing
    Plotting

.. toctree::
    :maxdepth: 1
    :caption: Other information
    
    FAQ
    Roadmap
    Citing
    Contributing
    About

Project principles and design decisions
=======================================

- It should be easy to swap out individual components of the optimization process
  with the user's proprietary improvements.
- Usability is everything: it is better to be self-explanatory than consistent.
- There is no point in portfolio optimization unless it can be practically
  applied to real asset prices.
- Everything that has been implemented should be tested.
- Inline documentation is good: dedicated (separate) documentation is better.
  The two are not mutually exclusive.
- Formatting should never get in the way of good code: because of this,
  I have deferred **all** formatting decisions to `Black
  <https://github.com/ambv/black>`_.


Advantages over existing implementations
========================================

- Includes both classical methods (Markowitz 1952 and Black-Litterman), suggested best practices
  (e.g covariance shrinkage), along with many recent developments and novel
  features, like L2 regularisation, exponential covariance, hierarchical risk parity.
- Native support for pandas dataframes: easily input your daily prices data.
- Extensive practical tests, which use real-life data.
- Easy to combine with your proprietary strategies and models.
- Robust to missing data, and price-series of different lengths (e.g FB data
  only goes back to 2012 whereas AAPL data goes back to 1980).


Contributors
=============

This is a non-exhaustive unordered list of contributors. I am sincerely grateful for all
of your efforts!

- Philipp Schiele
- Carl Peasnell
- Felipe Schneider
- Dingyuan Wang
- Pat Newell
- Aditya Bhutra
- Thomas Schmelzer
- Rich Caputo
- Nicolas Knudde 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
