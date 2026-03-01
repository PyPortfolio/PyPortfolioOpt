.. _other-optimizers:

################
Other Optimizers
################

Efficient frontier methods involve the direct optimization of an objective subject to constraints.
However, there are some portfolio optimization schemes that are completely different in character.
PyPortfolioOpt provides support for these alternatives, while still giving you access to the same
pre and post-processing API.

.. note::
    As of v0.4, these other optimizers now inherit from ``BaseOptimizer`` or
    ``BaseConvexOptimizer``, so you no longer have to implement pre-processing and
    post-processing methods on your own. You can thus easily swap out, say,
    ``EfficientFrontier`` for ``HRPOpt``.

Hierarchical Risk Parity (HRP)
==============================

Hierarchical Risk Parity is a novel portfolio optimization method developed by
Marcos Lopez de Prado [1]_. Though a detailed explanation can be found in the
linked paper, here is a rough overview of how HRP works:


1. From a universe of assets, form a distance matrix based on the correlation
   of the assets.
2. Using this distance matrix, cluster the assets into a tree via hierarchical
   clustering
3. Within each branch of the tree, form the minimum variance portfolio (normally
   between just two assets).
4. Iterate over each level, optimally combining the mini-portfolios at each node.


The advantages of this are that it does not require the inversion of the covariance
matrix as with traditional mean-variance optimization, and seems to produce diverse
portfolios that perform well out of sample.

.. image:: ../media/dendrogram.png
   :width: 80%
   :align: center
   :alt: cluster diagram


.. automodule:: pypfopt.hierarchical_portfolio

    .. autoclass:: HRPOpt
        :members:
        :exclude-members: plot_dendrogram


        .. automethod:: __init__

.. _cla:

The Critical Line Algorithm
===========================

This is a robust alternative to the quadratic solver used to find mean-variance optimal portfolios,
that is especially advantageous when we apply linear inequalities. Unlike generic convex optimization routines, 
the CLA is specially designed for portfolio optimization. It is guaranteed to converge after a certain
number of iterations, and can efficiently derive the entire efficient frontier.

.. image:: ../media/cla_plot.png
   :width: 80%
   :align: center
   :alt: the Efficient Frontier

.. tip:: 

    In general, unless you have specific requirements e.g you would like to efficiently compute the entire
    efficient frontier for plotting, I would go with the standard ``EfficientFrontier`` optimizer.

I am most grateful to Marcos López de Prado and David Bailey for providing the implementation [2]_.
Permission for its distribution has been received by email. It has been modified such that it has
the same API, though as of v0.5.0 we only support ``max_sharpe()`` and ``min_volatility()``.


.. automodule:: pypfopt.cla

    .. autoclass:: CLA
        :members:
        :exclude-members: plot_efficient_frontier

        .. automethod:: __init__


Implementing your own optimizer
===============================

Please note that this is quite different to implementing :ref:`custom-optimization`, because in
that case we are still using the same convex optimization structure. However, HRP and CLA optimization
have a fundamentally different optimization method. In general, these are much more difficult
to code up compared to custom objective functions.

To implement a custom optimizer that is compatible with the rest of PyPortfolioOpt, just
extend ``BaseOptimizer`` (or ``BaseConvexOptimizer`` if you want to use ``cvxpy``),
both of which can be found in ``pypfopt.base``. This gives you access to utility
methods like ``clean_weights()``, as well as making sure that any output is compatible
with ``portfolio_performance()`` and post-processing methods.

.. automodule:: pypfopt.base

    .. autoclass:: BaseOptimizer
        :members:

        .. automethod:: __init__

    .. autoclass:: BaseConvexOptimizer
        :members:
        :private-members:

        .. automethod:: __init__


References
==========

.. [1] López de Prado, M. (2016). `Building Diversified Portfolios that Outperform Out of Sample <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>`_. The Journal of Portfolio Management, 42(4), 59–69.
.. [2] Bailey and Loópez de Prado (2013). `An Open-Source Implementation of the Critical-Line Algorithm for Portfolio Optimization <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2197616>`_ 
