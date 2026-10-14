.. _post-processing:

#######################
Post-processing weights
#######################

After optimal weights have been generated, it is often necessary to do some
post-processing before they can be used practically. In particular, you are
likely using portfolio optimization techniques to generate a
**portfolio allocation** – a list of tickers and corresponding integer quantities
that you could go and purchase at a broker.

However, it is not trivial to convert the continuous weights (output by any of our
optimization methods) into an actionable allocation. For example, let us say that we
have $10,000 that we would like to allocate. If we multiply the weights by this total
portfolio value, the result will be dollar amounts of each asset. So if the optimal weight
for Apple is 0.15, we need $1500 worth of Apple stock. However, Apple shares come
in discrete units ($190 at the time of writing), so we will not be able to buy
exactly $1500 of stock. The best we can do is to buy the number of shares that
gets us closest to the desired dollar value.

PyPortfolioOpt offers two ways of solving this problem: one using a simple greedy
algorithm, the other using integer programming.

Greedy algorithm
================

The greedy allocator accepts integer-valued quotes as well as floating-point prices
in ``latest_prices``, without changing the input Series or its dtype.

``DiscreteAllocation.greedy_portfolio()`` proceeds in two 'rounds'.
In the first round, we buy as many shares as we can for each asset without going over
the desired weight. In the Apple example, :math:`1500/190 \approx 7.89`, so we buy 7
shares at a cost of $1330. After iterating through all of the assets, we will have a
lot of money left over (since we always rounded down).

In the second round, we calculate how far the current weights deviate from the
existing weights for each asset. We wanted Apple to form 15% of the portfolio
(with total value $10,000), but we only bought $1330 worth of Apple stock, so
there is a deviation of :math:`0.15 - 0.133`. Some assets will have a higher
deviation from the ideal, so we will purchase shares of these first. We then
repeat the process, always buying shares of the asset whose current weight is
furthest away from the ideal weight. Though this algorithm will not guarantee
the optimal solution, I have found that it allows us to generate discrete
allocations with very little money left over (e.g $12 left on a $10,000 portfolio).

That being said, we can see that on the test dataset (for a standard ``max_sharpe``
portfolio), the allocation method may deviate rather widely from the desired weights,
particularly for companies with a high share price (e.g AMZN).

.. code-block:: text
    
    Funds remaining: 12.15
    MA: allocated 0.242, desired 0.246
    FB: allocated 0.200, desired 0.199
    PFE: allocated 0.183, desired 0.184
    BABA: allocated 0.088, desired 0.096
    AAPL: allocated 0.086, desired 0.092
    AMZN: allocated 0.000, desired 0.072
    BBY: allocated 0.064, desired 0.061
    SBUX: allocated 0.036, desired 0.038
    GOOG: allocated 0.102, desired 0.013
    Allocation has RMSE: 0.038


Integer programming
===================

This method (credit to `Dingyuan Wang <https://github.com/gumblex>`_ for the first implementation)
treats the discrete allocation as an integer programming problem. In effect, the integer
programming approach searches the space of possible allocations to find the one that is
closest to our desired weights. We will use the following notation:

- :math:`T \in \mathbb{R}` is the total dollar value to be allocated
- :math:`p \in \mathbb{R}^n` is the array of latest prices
- :math:`w \in \mathbb{R}^n` is the set of target weights
- :math:`x \in \mathbb{Z}^n` is the integer allocation (i.e the result)
- :math:`r \in \mathbb{R}` is the remaining unallocated value, i.e :math:`r = T - x \cdot p`.

The optimization problem is then given by:

.. math::

    \begin{equation*}
    \begin{aligned}
    & \underset{x \in \mathbb{Z}^n}{\text{minimise}} & & r + \lVert wT - x \odot p \rVert_1  \\
    & \text{subject to} & & r + x \cdot p = T\\
    \end{aligned}
    \end{equation*}

This is straightforward to translate into ``cvxpy``.

.. info::

    Though ``lp_portfolio()`` produces allocations with a lower RMSE, some testing
    shows that it is between 100 and 1000 times slower than ``greedy_portfolio()``.
    This doesn't matter for small portfolios (it should still take less than a second),
    but the runtime for integer programs grows exponentially as the number of stocks, so
    for large portfolios you may have to use ``greedy_portfolio()``.

.. warning::

    PyPortfolioOpt uses ``ECOS_BB`` as a default solver for integer programming. ``ECOS_BB`` has known correctness issues (see `here <https://www.cvxpy.org/tutorial/advanced/index.html#mixed-integer-programs>`_ for a discussion).
    An alternative is to use ``GLPK_MI``, which comes packaged with ``cvxopt``.

Dealing with shorts
===================

As of v0.4, ``DiscreteAllocation`` automatically deals with shorts by finding separate discrete
allocations for the long-only and short-only portions. If your portfolio has shorts,
you should pass a short ratio. The default is 0.30, corresponding to a 130/30 long-short balance.
Practically, this means that you would go long $10,000 of some stocks, short $3000 of some other
stocks, then use the proceeds from the shorts to go long another $3000.
Thus the total value of the resulting portfolio would be $13,000. 

Documentation reference
========================

.. automodule:: pypfopt.discrete_allocation

    .. autoclass:: DiscreteAllocation
        :members:
        :private-members:

        .. automethod:: __init__


