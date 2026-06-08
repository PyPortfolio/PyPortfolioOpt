.. _expected-returns:

################
Expected Returns
################

Mean-variance optimization requires knowledge of the expected returns. In practice,
these are rather difficult to know with any certainty. Thus the best we can do is to
come up with estimates, for example by extrapolating historical data or factor models. This is the
main flaw in mean-variance optimization – the optimization procedure is sound, and provides
strong mathematical guarantees, *given the correct inputs*. This is one of the reasons
why I have emphasised modularity: users should be able to come up with their own
superior models and feed them into the optimizer.

.. caution::

    Supplying expected returns can do more harm than good. If
    predicting stock returns were as easy as calculating the mean historical return,
    we'd all be rich! For most use-cases, I would suggest that you focus your efforts
    on choosing an appropriate risk model (see :ref:`risk-models`). 

    As of v0.5.0, you can use :ref:`black-litterman` to significantly improve the quality of
    your estimate of the expected returns.

.. automodule:: pypfopt.expected_returns

    .. note::

        For any of these methods, if you would prefer to pass returns (the default is prices),
        set the boolean flag ``returns_data=True``

    .. autofunction:: mean_historical_return

        This is probably the default textbook approach. It is intuitive and easily interpretable,
        however the estimates are subject to large uncertainty. This is a problem especially in the
        context of a mean-variance optimizer, which will maximise the erroneous inputs.


    .. autofunction:: ema_historical_return

        The exponential moving average is a simple improvement over the mean historical
        return; it gives more credence to recent returns and thus aims to increase the relevance
        of the estimates. This is parameterised by the ``span`` parameter, which gives users
        the ability to decide exactly how much more weight is given to recent data.
        Generally, I would err on the side of a higher span – in the limit, this tends towards
        the mean historical return. However, if you plan on rebalancing much more frequently,
        there is a case to be made for lowering the span in order to capture recent trends.

    .. autofunction:: capm_return

    .. autofunction:: returns_from_prices

    .. autofunction:: prices_from_returns
    
    .. autofunction:: ff_return

        Estimates expected returns using the Fama-French factor models.

        Set ``model="ff3"`` to use the three-factor specification with
        market (Mkt-RF), size (SMB), and value (HML) factors.

        Set ``model="ff5"`` to use the five-factor specification, which
        additionally includes profitability (RMW) and investment (CMA)
        factors.

        Factor data must be supplied via the ``factor_data`` argument and
        contain the required factor return columns indexed by date.


.. References
.. ==========
