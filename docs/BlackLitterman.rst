.. _black-litterman:

##########################
Black-Litterman Allocation
##########################

The Black-Litterman (BL) model [1]_ takes a Bayesian approach to asset allocation.
Specifically, it combines a **prior** estimate of returns (for example, the market-implied
returns) with **views** on certain assets, to produce a **posterior** estimate of expected
returns. The advantages of this are:

- You can provide views on only a subset of assets and BL will meaningfully propagate it, 
  taking into account the covariance with other assets.
- You can provide *confidence* in your views.
- Using Black-Litterman posterior returns results in much more stable portfolios than
  using mean-historical return. 

Essentially, Black-Litterman treats the vector of expected returns itself as a quantity to
be estimated. The Black-Litterman formula is given below:

.. math:: 

    E(R) = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1}[(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q]

- :math:`E(R)` is a Nx1 vector of expected returns, where *N* is the number of assets.
- :math:`Q` is a Kx1 vector of views.
- :math:`P` is the KxN **picking matrix** which maps views to the universe of assets.
  Essentially, it tells the model which view corresponds to which asset(s).
- :math:`\Omega` is the KxK **uncertainty matrix** of views. 
- :math:`\Pi` is the Nx1 vector of prior expected returns. 
- :math:`\Sigma` is the NxN covariance matrix of asset returns (as always)
- :math:`\tau` is a scalar tuning constant. 

Though the formula appears to be quite unwieldy, it turns out that the formula simply represents
a weighted average between the prior estimate of returns and the views, where the weighting
is determined by the confidence in the views and the parameter :math:`\tau`. 

Similarly, we can calculate a posterior estimate of the covariance matrix:

.. math::

    \hat{\Sigma} = \Sigma + [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1}


Though the algorithm is relatively simple, BL proved to be a challenge from a software
engineering perspective because it's not quite clear how best to fit it into PyPortfolioOpt's
API. The full discussion can be found on a `Github issue thread <https://github.com/pyportfolio/pyportfolioopt/issues/48>`_,
but I ultimately decided that though BL is not technically an optimizer, it didn't make sense to
split up its methods into `expected_returns` or `risk_models`. I have thus made it an independent
module and owing to the comparatively extensive theory, have given it a dedicated documentation page.
I'd like to thank  `Felipe Schneider <https://github.com/schneiderfelipe>`_ for his multiple
contributions to the Black-Litterman implementation. A full example of its usage, including the acquisition
of market cap data for free, please refer to the `cookbook recipe <https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/cookbook/4-Black-Litterman-Allocation.ipynb>`_.

.. tip:: 

    Thomas Kirschenmann has built a neat interactive `Black-Litterman tool <https://github.com/thk3421-models/cardiel>`_
    on top of PyPortfolioOpt, which allows you to visualise BL outputs and compare optimization objectives.

Priors
======

You can think of the prior as the "default" estimate, in the absence of any information. 
Black and Litterman (1991) [2]_ provide the insight that a natural choice for this prior
is the market's estimate of the return, which is embedded into the market capitalisation
of the asset. 

Every asset in the market portfolio contributes a certain amount of risk to the portfolio.
Standard theory suggests that investors must be compensated for the risk that they take, so
we can attribute to each asset an expected compensation (i.e prior estimate of returns). This
is quantified by the market-implied risk premium, which is the market's excess return divided
by its variance: 

.. math::

    \delta = \frac{R-R_f}{\sigma^2}

To calculate the market-implied returns, we then use the following formula:

.. math::

    \Pi = \delta \Sigma w_{mkt}

Here, :math:`w_{mkt}` denotes the market-cap weights. This formula is calculating the total
amount of risk contributed by an asset and multiplying it with the market price of risk,
resulting in the market-implied returns vector :math:`\Pi`. We can use PyPortfolioOpt to calculate
this as follows::


    from pypfopt import black_litterman, risk_models

    """
    cov_matrix is a NxN sample covariance matrix
    mcaps is a dict of market caps
    market_prices is a series of S&P500 prices
    """
    delta = black_litterman.market_implied_risk_aversion(market_prices)
    prior = black_litterman.market_implied_prior_returns(mcaps, delta, cov_matrix)


There is nothing stopping you from using any prior you see fit (but it must have the same dimensionality as the universe).
If you think that the mean historical returns are a good prior, 
you could go with that. But a significant body of research shows that mean historical returns are a completely uninformative
prior. 

.. note::

    You don't technically have to provide a prior estimate to the Black-Litterman model. This is particularly useful
    if your views (and confidences) were generated by some proprietary model, in which case BL is essentially a clever way
    of mixing your views.


Views
=====

In the Black-Litterman model, users can either provide **absolute** or **relative** views. Absolute views are statements like:
"AAPL will return 10%" or "XOM will drop 40%". Relative views, on the other hand, are statements like "GOOG will outperform FB by 3%".

These views must be specified in the vector :math:`Q` and mapped to the asset universe via the picking matrix :math:`P`. A brief
example of this is shown below, though a comprehensive guide is given by `Idzorek <https://faculty.fuqua.duke.edu/~charvey/Teaching/BA453_2006/Idzorek_onBL.pdf>`_.
Let's say that our universe is defined by the ordered list: SBUX, GOOG, FB, AAPL, BAC, JPM, T, GE, MSFT, XOM. We want to represent
four views on these 10 assets, two absolute and two relative:

1. SBUX will drop 20% (absolute)
2. MSFT will rise by 5% (absolute)
3. GOOG outperforms FB by 10%
4. BAC and JPM will outperform T and GE by 15%

The corresponding views vector is formed by taking the numbers above and putting them into a column::

    Q = np.array([-0.20, 0.05, 0.10, 0.15]).reshape(-1, 1)

The picking matrix is more interesting. Remember that its role is to link the views (which mention 8 assets) to the universe of 10
assets. Arguably, this is the most important part of the model because it is what allows us to propagate our expectations (and
confidences in expectations) into the model::

    P = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.5, -0.5, -0.5, 0, 0],
        ]
    )

A brief explanation of the above:

- Each view has a corresponding row in the picking matrix (the order matters)
- Absolute views have a single 1 in the column corresponding to the ticker's order in the universe. 
- Relative views have a positive number in the nominally outperforming asset columns and a negative number
  in the nominally underperforming asset columns. The numbers in each row should sum up to 0.


PyPortfolioOpt provides a helper method for inputting absolute views as either a ``dict`` or ``pd.Series`` – 
if you have relative views, you must build your picking matrix manually:: 

    from pypfopt.black_litterman import BlackLittermanModel

    viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.15}
    bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict)


Confidence matrix and tau
=========================

The confidence matrix is a diagonal covariance matrix containing the variances of each view. One heuristic for calculating
:math:`\Omega` is to say that is proportional to the variance of the priors. This is reasonable - quantities that move
around a lot are harder to forecast! Hence PyPortfolioOpt does not require you to input a confidence matrix, and defaults to:

.. math::

    \Omega = \tau * P \Sigma P^T

Alternatively, we provide an implementation of Idzorek's method [1]_. This allows you to specify your view uncertainties as
percentage confidences. To use this, choose ``omega="idzorek"`` and pass a list of confidences (from 0 to 1) into the ``view_confidences``
parameter.

You are of course welcome to provide your own estimate. This is particularly applicable if your views are the output
of some statistical model, which may also provide the view uncertainty.

Another parameter that controls the relative weighting of the priors views is :math:`\tau`. There is a lot to be said about tuning
this parameter, with many contradictory rules of thumb. Indeed, there has been an entire paper written on it [3]_. We choose
the sensible default :math:`\tau = 0.05`.

.. note::

    If you use the default estimate of :math:`\Omega`, or ``omega="idzorek"``, it turns out that the value of :math:`\tau` does not matter. This
    is a consequence of the mathematics: the :math:`\tau` cancels in the matrix multiplications.


Output of the BL model
======================

The BL model outputs posterior estimates of the returns and covariance matrix. The default suggestion in the literature is to
then input these into an optimizer (see :ref:`efficient-frontier`). A quick alternative, which is quite useful for debugging, is
to calculate the weights implied by the returns vector [4]_. It is actually the reverse of the procedure we used to calculate the
returns implied by the market weights. 

.. math::

    w = (\delta \Sigma)^{-1} E(R)

In PyPortfolioOpt, this is available under ``BlackLittermanModel.bl_weights()``. Because the ``BlackLittermanModel`` class
inherits from ``BaseOptimizer``, this follows the same API as the ``EfficientFrontier`` objects::

    from pypfopt import black_litterman
    from pypfopt.black_litterman import BlackLittermanModel
    from pypfopt.efficient_frontier import EfficientFrontier

    viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.15}
    bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict)

    rets = bl.bl_returns()
    ef = EfficientFrontier(rets, cov_matrix)

    # OR use return-implied weights
    delta = black_litterman.market_implied_risk_aversion(market_prices)
    bl.bl_weights(delta)
    weights = bl.clean_weights()


Documentation reference
=======================

.. automodule:: pypfopt.black_litterman
    :members:
    :exclude-members: BlackLittermanModel

    .. autoclass:: BlackLittermanModel
        :members:

        .. automethod:: __init__

        .. caution::

            You **must** specify the covariance matrix and either absolute views or *both* Q and P, except in the special case
            where you provide exactly one view per asset, in which case P is inferred. 

References
==========

.. [1] Idzorek T. A step-by-step guide to the Black-Litterman model: Incorporating user-specified confidence levels. In: Forecasting Expected Returns in the Financial Markets. Elsevier Ltd; 2007. p. 17–38. 
.. [2] Black, F; Litterman, R. Combining investor views with market equilibrium. The Journal of Fixed Income, 1991.
.. [3] Walters, Jay, The Factor Tau in the Black-Litterman Model (October 9, 2013). Available at SSRN: https://ssrn.com/abstract=1701467 or http://dx.doi.org/10.2139/ssrn.1701467
.. [4] Walters J. The Black-Litterman Model in Detail (2014). SSRN Electron J.;(February 2007):1–65. 
