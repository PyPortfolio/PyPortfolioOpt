.. _efficient-frontier:

##########################
General Efficient Frontier
##########################

The mean-variance optimization methods described previously can be used whenever you have a vector
of expected returns and a covariance matrix. The objective and constraints will be some combination
of the portfolio return and portfolio volatility. 

However, you may want to construct the efficient frontier for an entirely different type of risk model
(one that doesn't depend on covariance matrices), or optimize an objective unrelated to portfolio
return (e.g tracking error). PyPortfolioOpt comes with several popular alternatives and provides support
for custom optimization problems.

Efficient Semivariance
======================

Instead of penalising volatility, mean-semivariance optimization seeks to only penalise
downside volatility, since upside volatility may be desirable. 

There are two approaches to the mean-semivariance optimization problem. The first is to use a
heuristic (i.e "quick and dirty") solution: pretending that the semicovariance matrix
(implemented in :py:mod:`risk_models`) is a typical covariance matrix and doing standard
mean-variance optimization. It can be shown that this *does not* yield a portfolio that
is efficient in mean-semivariance space (though it might be a good-enough approximation).

Fortunately, it is possible to write mean-semivariance optimization as a convex problem
(albeit one with many variables), that can be solved to give an "exact" solution.
For example, to maximise return for a target semivariance
:math:`s^*` (long-only), we would solve the following problem:

.. math::

    \begin{equation*}
    \begin{aligned}
    & \underset{w}{\text{maximise}} & & w^T \mu \\
    & \text{subject to} & & n^T n \leq s^*  \\
    &&& B w - p + n = 0 \\
    &&& w^T \mathbf{1} = 1 \\
    &&& n \geq 0 \\
    &&& p \geq 0. \\
    \end{aligned}
    \end{equation*}

Here, **B** is the :math:`T \times N` (scaled) matrix of excess returns:
``B = (returns - benchmark) / sqrt(T)``. Additional linear equality constraints and
convex inequality constraints can be added. 

PyPortfolioOpt allows users to optimize along the efficient semivariance frontier
via the :py:class:`EfficientSemivariance` class. :py:class:`EfficientSemivariance` inherits from
:py:class:`EfficientFrontier`, so it has the same utility methods 
(e.g :py:func:`add_constraint`, :py:func:`portfolio_performance`), but finds portfolios on the mean-semivariance
frontier. Note  that some of the parent methods, like :py:func:`max_sharpe` and :py:func:`min_volatility`
are not applicable to mean-semivariance portfolios, so calling them returns ``NotImplementedError``.

:py:class:`EfficientSemivariance` has a slightly different API to :py:class:`EfficientFrontier`. Instead of passing
in a covariance matrix, you should past in a dataframe of historical/simulated returns (this can be constructed
from your price dataframe using the helper method :py:func:`expected_returns.returns_from_prices`). Here
is a full example, in which we seek the portfolio that minimises the semivariance for a target
annual return of 20%::

    from pypfopt import expected_returns, EfficientSemivariance

    df = ... # your dataframe of prices
    mu = expected_returns.mean_historical_returns(df)
    historical_returns = expected_returns.returns_from_prices(df)

    es = EfficientSemivariance(mu, historical_returns)
    es.efficient_return(0.20)

    # We can use the same helper methods as before
    weights = es.clean_weights() 
    print(weights)
    es.portfolio_performance(verbose=True)

The ``portfolio_performance`` method outputs the expected portfolio return, semivariance,
and the Sortino ratio (like the Sharpe ratio, but for downside deviation).

Interested readers should refer to Estrada (2007) [1]_ for more details. I'd like to thank 
`Philipp Schiele <https://github.com/phschiele>`_ for authoring the bulk
of the efficient semivariance functionality and documentation (all errors are my own). The
implementation is based on Markowitz et al (2019) [2]_.

.. caution::

    Finding portfolios on the mean-semivariance frontier is computationally harder
    than standard mean-variance optimization: our implementation uses ``2T + N`` optimization variables, 
    meaning that for 50 assets and 3 years of data, there are about 1500 variables.  
    While :py:class:`EfficientSemivariance` allows for additional constraints/objectives in principle,
    you are much more likely to run into solver errors. I suggest that you keep :py:class:`EfficientSemivariance`
    problems small and minimally constrained.

.. autoclass:: pypfopt.efficient_frontier.EfficientSemivariance
    :members:
    :exclude-members: max_sharpe, min_volatility

Efficient CVaR
==============

The **conditional value-at-risk** (a.k.a **expected shortfall**) is a popular measure of tail risk. The CVaR can be
thought of as the average of losses that occur on "very bad days", where "very bad" is quantified by the parameter
:math:`\beta`.

For example, if we calculate the CVaR to be 10% for :math:`\beta = 0.95`, we can be 95% confident that the worst-case
average daily loss will be 10%. Put differently, the CVaR is the average of all losses so severe that they only occur
:math:`(1-\beta)\%` of the time. 

While CVaR is quite an intuitive concept, a lot of new notation is required to formulate it mathematically (see
the `wiki page <https://en.wikipedia.org/wiki/Expected_shortfall>`_ for more details). We will adopt the following
notation: 

- *w* for the vector of portfolio weights
- *r* for a vector of asset returns (daily), with probability distribution :math:`p(r)`.
- :math:`L(w, r) = - w^T r` for the loss of the portfolio
- :math:`\alpha` for the portfolio value-at-risk (VaR) with confidence :math:`\beta`.

The CVaR can then be written as:

.. math::
    CVaR(w, \beta) = \frac{1}{1-\beta} \int_{L(w, r) \geq \alpha (w)} L(w, r) p(r)dr.

This is a nasty expression to optimize because we are essentially integrating over VaR values. The key insight
of Rockafellar and Uryasev (2001) [3]_ is that we can can equivalently optimize the following convex function:

.. math:: 
    F_\beta (w, \alpha) = \alpha + \frac{1}{1-\beta} \int [-w^T r - \alpha]^+ p(r) dr,

where :math:`[x]^+ = \max(x, 0)`. The authors prove that minimising :math:`F_\beta(w, \alpha)` over all
:math:`w, \alpha` minimises the CVaR. Suppose we have a sample of *T* daily returns (these
can either be historical or simulated). The integral in the expression becomes a sum, so the CVaR optimization
problem reduces to a linear program:

.. math::

    \begin{equation*}
    \begin{aligned}
    & \underset{w, \alpha}{\text{minimise}} & & \alpha + \frac{1}{1-\beta} \frac 1 T \sum_{i=1}^T u_i \\
    & \text{subject to} & & u_i \geq 0  \\
    &&&  u_i \geq -w^T r_i - \alpha. \\
    \end{aligned}
    \end{equation*}

This formulation introduces a new variable for each datapoint (similar to Efficient Semivariance), so
you may run into performance issues for long returns dataframes. At the same time, you should aim to
provide a sample of data that is large enough to include tail events. 

I am grateful to `Nicolas Knudde <https://github.com/nknudde>`_ for the initial draft (all errors are my own).
The implementation is based on Rockafellar and Uryasev (2001) [3]_.


.. autoclass:: pypfopt.efficient_frontier.EfficientCVaR
    :members:
    :exclude-members: max_sharpe, min_volatility, max_quadratic_utility

EfficientCDaR
==============

The **conditional drawdown at risk** (CDaR) is a more exotic measure of tail risk. It tries to alleviate the problems
with Efficient Semivariance and Efficient CVaR in that it accounts for the timespan of material decreases in value. The CDaR can be
thought of as the average of losses that occur on "very bad periods", where "very bad" is quantified by the parameter
:math:`\beta`. The drawdown is defined as the difference in non-compounded return to the previous peak.

Put differently, the CDaR is the average of all drawdowns so severe that they only occur
:math:`(1-\beta)\%` of the time. When :math:`\beta = 1` CDaR is simply the maximum drawdown.

While drawdown is quite an intuitive concept, a lot of new notation is required to formulate it mathematically (see
`the wiki page <https://en.wikipedia.org/wiki/Drawdown_(economics)>`_ for more details). We will adopt the following
notation:

- *w* for the vector of portfolio weights
- *r* for a vector of cumulative asset returns (daily), with probability distribution :math:`p(r(t))`.
- :math:`D(w, r, t) = \max_{\tau<t}(w^T r(\tau))-w^T r(t)` for the drawdown of the portfolio
- :math:`\alpha` for the portfolio drawdown (DaR) with confidence :math:`\beta`.

The CDaR can then be written as:

.. math::
    CDaR(w, \beta) = \frac{1}{1-\beta} \int_{D(w, r, t) \geq \alpha (w)} D(w, r, t) p(r(t))dr(t).

This is a nasty expression to optimise because we are essentially integrating over VaR values. The key insight
of Chekhlov, Rockafellar and Uryasev (2005) [4]_ is that we can can equivalently optimise a convex function, which can be
transformed to a linear problem (in the same manner as for CVaR). 

.. autoclass:: pypfopt.efficient_frontier.EfficientCDaR
    :members:
    :exclude-members: max_sharpe, min_volatility, max_quadratic_utility

I am grateful to `Nicolas Knudde <https://github.com/nknudde>`_ for implementing this feature.


.. _custom-optimization:

Custom optimization problems
============================

We have seen previously that it is easy to add constraints to ``EfficientFrontier`` objects (and
by extension, other general efficient frontier objects like ``EfficientSemivariance``). However, what if you aren't interested
in anything related to ``max_sharpe()``, ``min_volatility()``, ``efficient_risk()`` etc and want to
set up a completely new problem to optimize for some custom objective?

For example, perhaps our objective is to construct a basket of assets that best replicates a
particular index, in other words, to minimise the **tracking error**. This does not fit within
a mean-variance optimization paradigm, but we can still implement it in PyPortfolioOpt::

    from pypfopt.base_optimizer import BaseConvexOptimizer
    from pypfopt.objective_functions import ex_post_tracking_error

    historic_rets = ... # dataframe of historic asset returns
    benchmark_rets = ... # pd.Series of historic benchmark returns (same index as historic)

    opt = BaseConvexOptimizer(
        n_assets=len(historic_returns.columns), 
        tickers=historic_returns.columns,
        weight_bounds=(0, 1)
    )
    opt.convex_objective(
        ex_post_tracking_error,
        historic_returns=historic_rets,
        benchmark_returns=benchmark_rets,
    ) 
    weights = opt.clean_weights()

The ``EfficientFrontier`` class inherits from ``BaseConvexOptimizer``. It may be more convenient
to call ``convex_objective`` from an ``EfficientFrontier`` instance than from ``BaseConvexOptimizer``,
particularly if your objective depends on the mean returns or covariance matrix. 

You can either optimize some generic ``convex_objective``
(which *must* be built using ``cvxpy`` atomic functions -- see `here <https://www.cvxpy.org/tutorial/functions/index.html>`_)
or a ``nonconvex_objective``, which uses ``scipy.optimize`` as the backend and thus has a completely
different API. For more examples, check out this `cookbook recipe
<https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/cookbook/3-Advanced-Mean-Variance-Optimization.ipynb>`_.

    .. class:: pypfopt.base_optimizer.BaseConvexOptimizer

        .. automethod:: convex_objective

        .. automethod:: nonconvex_objective



References
==========

.. [1] Estrada, J (2007). `Mean-Semivariance Optimization: A Heuristic Approach <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1028206>`_.
.. [2] Markowitz, H.; Starer, D.; Fram, H.; Gerber, S. (2019). `Avoiding the Downside <https://www.hudsonbaycapital.com/documents/FG/hudsonbay/research/599440_paper.pdf>`_.
.. [3] Rockafellar, R.; Uryasev, D. (2001). `Optimization of conditional value-at-risk <https://www.ise.ufl.edu/uryasev/files/2011/11/CVaR1_JOR.pdf>`_
.. [4] Chekhlov, A.; Rockafellar, R.; Uryasev, D. (2005). `Drawdown measure in portfolio optimization <http://www.math.columbia.edu/~chekhlov/IJTheoreticalAppliedFinance.8.1.2005.pdf>`_
