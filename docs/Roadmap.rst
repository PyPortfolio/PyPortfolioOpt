.. _roadmap:

#####################
Roadmap and Changelog
#####################


Roadmap
=======

PyPortfolioOpt is now a "mature" package – it is stable and I don't intend to implement major new functionality (though I will endeavour to fix bugs).

1.5.0
=====

- Major redesign of the backend, thanks to `Philipp Schiele <https://github.com/phschiele>`_
  - Because we use ``cp.Parameter``, we can efficiently re-run optimisation problems with different constants (e.g risk targets) 
  - This leads to a significant improvement in plotting performance as we no longer have to repeatedly re-instantiate ``EfficientFrontier``.
- Several misc bug fixes (thanks to `Eric Armbruster <https://github.com/armbruer>`_ and `Ayoub Ennassiri <https://github.com/samatix>`_)

1.5.1
-----

Mucked up the versioning on the 1.5.0 launch. Sorry!

1.5.2
-----

Minor bug fixes

1.5.3
-----

- Reworked packaging: ``cvxpy`` is no longer a requirement as we default to ``ECOS_BB`` for discrete allocation. 
- Bumped minimum python version to ``3.8``. I would love to keep as many versions compatible (and I think most of the 
  functionality *should* still work with ``3.6, 3.7`` but the dependencies have gotten too tricky to manage).
- Changed to numpy pseudoinverse to allow for "cash" assets
- Ticker labels for efficient frontier plot

1.5.4
-----

- Fixed ``cvxpy`` deprecating deepcopy. Thanks to Philipp for the fix! 
- Several other tiny checks and bug fixes. Cheers to everyone for the PRs! 

1.5.5
-----

- `Tuan Tran <https://github.com/88d52bdba0366127fffca9dfa93895>`_ is now the primary maintainer for PyPortfolioOpt
- Wide range of bug fixes and code improvements.

1.5.6
-----

- Various bug fixes

1.4.0
=====

- Finally implemented CVaR optimization! This has been one of the most requested features. Many thanks
  to `Nicolas Knudde <https://github.com/nknudde>`_ for the initial draft. 
- Re-architected plotting so users can pass an ax, allowing for complex plots (see cookbook).
- Helper method to compute the max-return portfolio (thanks to `Philipp Schiele <https://github.com/phschiele>`_)
  for the suggestion). 
- Several bug fixes and test improvements (thanks to `Carl Peasnell <https://github.com/SeaPea1>`_).

1.4.1
-----

- 100% test coverage
- Reorganised docs; added FAQ page
- Reorganised module structure to make it more scalable
- Python 3.9 support, dockerfile versioning, misc packaging improvements (e.g cvxopt optional)

1.4.2
-----

- Implemented CDaR optimization – full credit to `Nicolas Knudde <https://github.com/nknudde>`_.
- Misc bug fixes


1.3.0
=====

- Significantly improved plotting functionality: can now plot constrained efficient frontier!
- Efficient semivariance portfolios (thanks to `Philipp Schiele <https://github.com/phschiele>`_)
- Improved functionality for portfolios with short positions (thanks to `Rich Caputo <https://github.com/arcaputo3>`_).
- Significant improvement in test coverage (thanks to `Carl Peasnell <https://github.com/SeaPea1>`_).
- Several bug fixes and usability improvements.
- Migrated from TravisCI to Github Actions.

1.3.1
-----

- Minor cleanup (forgotten commits from v1.3.0).


1.2.0
=====

- Added Idzorek's method for calculating the ``omega`` matrix given percentage confidences.
- Fixed max sharpe to allow for custom constraints
- Grouped sector constraints
- Improved error tracebacks
- Adding new cookbook for examples (in progress).
- Packaging: added bettter instructions for windows, added docker support.

1.2.1
-----

Fixed critical ordering bug in sector constraints

1.2.2
-----

Matplotlib now required dependency; support for pandas 1.0. 

1.2.3
-----

- Added support for changing solvers and verbose output
- Changed dict to OrderedDict to support python 3.5
- Improved packaging/dependencies: simplified requirements.txt, improved processes before pushing.

1.2.4
-----

- Fixed bug in Ledoit-Wolf shrinkage calculation.
- Fixed bug in plotting docs that caused them not to render. 

1.2.5
-----

- Fixed compounding in ``expected_returns`` (thanks to `Aditya Bhutra <https://github.com/bhutraaditya>`_).
- Improvements in advanced cvxpy API (thanks to `Pat Newell <https://github.com/pmn4>`_).
- Deprecating James-Stein
- Exposed ``linkage_method`` in HRP. 
- Added support for cvxpy 1.1.
- Added an error check for ``efficient_risk``. 
- Small improvements to docs.

1.2.6
-----

- Fixed order-dependence bug in Black-Litterman ``market_implied_prior_returns``
- Fixed inaccuracy in BL cookbook.
- Fixed bug in exponential covariance.

1.2.7
-----

- Fixed bug which required conservative risk targets for long/short portfolios.


1.1.0
=====

- Multiple additions and improvements to ``risk_models``:
    
  - Introduced a new API, in which the function ``risk_models.risk_matrix(method="...")`` allows
    all the different risk models to be called. This should make testing easier.
  - All methods now accept returns data instead of prices, if you set the flag ``returns_data=True``.
- Automatically fix non-positive semidefinite covariance matrices!

- Additions and improvements to ``expected_returns``:

  - Introduced a new API, in which the function ``expected_returns.return_model(method="...")`` allows
    all the different return models to be called. This should make testing easier.
  - Added option to 'properly' compound returns.
  - Added the CAPM return model.

- ``from pypfopt import plotting``: moved all plotting functionality into a new class and added
  new plots. All other plotting functions (scattered in different classes) have been retained,
  but are now deprecated.


1.0.0
=====

- Migrated backend from ``scipy`` to ``cvxpy`` and made significant breaking changes to the API

  - PyPortfolioOpt is now significantly more robust and numerically stable.
  - These changes will not affect basic users, who can still access features like ``max_sharpe()``.
  - However, additional objectives and constraints (including L2 regularisation) are now 
    explicitly added before optimising some 'primary' objective.

- Added basic plotting capabilities for the efficient frontier, hierarchical clusters, 
  and HRP dendrograms.
- Added a basic transaction cost objective.
- Made breaking changes to some modules and classes so that PyPortfolioOpt is easier to extend
  in future:
  
  - Replaced ``BaseScipyOptimizer`` with ``BaseConvexOptimizer``
  - ``hierarchical_risk_parity`` was replaced by ``hierarchical_portfolios`` to leave the door open for other hierarchical methods.
  - Sadly, removed CVaR optimization for the time being until I can properly fix it.

1.0.1
-----

Fixed minor issues in CLA: weight bound bug, ``efficient_frontier`` needed weights to be called, ``set_weights`` not needed.

1.0.2
-----

Fixed small but important bug where passing ``expected_returns=None`` fails. According to the docs, users
should be able to only pass covariance if they want to only optimize min volatility.


0.5.0
=====

- Black-Litterman model and docs.
- Custom bounds per asset
- Improved ``BaseOptimizer``, adding a method that writes weights
  to text and fixing a bug in ``set_weights``.
- Unconstrained quadratic utility optimization (analytic)
- Revamped docs, with information on types of attributes and
  more examples.

0.5.1
-----

Fixed an error with dot products by amending the pandas requirements.

0.5.2
-----

Made PuLP, sklearn, noisyopt optional dependencies to improve installation
experience.

0.5.3
-----

- Fixed an optimization bug in ``EfficientFrontier.efficient_risk``. An error is now
  thrown if optimization fails.
- Added a hidden API to change the scipy optimizer method. 

0.5.4
-----

- Improved the Black-Litterman linear algebra to avoid inverting the uncertainty matrix. 
  It is now possible to have 100% confidence in views.
- Clarified regarding the role of tau.
- Added a ``pipfile`` for ``pipenv`` users.
- Removed Value-at-risk from docs to discourage usage until it is properly fixed.

0.5.5
-----

Began migration to cvxpy by changing the discrete allocation backend from PuLP to cvxpy. 

0.4.0
=====

- Major improvements to ``discrete_allocation``. Added functionality to allocate shorts;
  modified the linear programming method suggested by `Dingyuan Wang <https://github.com/gumblex>`_;
  added postprocessing section to User Guide.
- Further refactoring and docs for ``HRPOpt``.
- Major documentation update, e.g to support custom optimizers

0.4.1
-----

- Added CLA back in after getting permission from Dr Marcos López de Prado
- Added more tests for different risk models.

0.4.2
-----

- Minor fix for ``clean_weights``
- Removed official support for python 3.4.
- Minor improvement to semicovariance, thanks to `Felipe Schneider <https://github.com/schneiderfelipe>`_.

0.4.3
-----

- Added ``prices_from_returns`` utility function and provided better docs for ``returns_from_prices``.
- Added ``cov_to_corr`` method to produce correlation matrices from covariance matrices.
- Fixed readme examples.



0.3.0
=====

- Merged an amazing PR from `Dingyuan Wang <https://github.com/gumblex>`_ that rearchitects
  the project to make it more self-consistent and extensible.
- New algorithm: ML de Prado's CLA
- New algorithms for converting continuous allocation to discrete (using linear
  programming).
- Merged a `PR <https://github.com/pyportfolio/pyportfolioopt/pull/22>`__ implementing Single Factor and
  Constant Correlation shrinkage.

0.3.1
-----

Merged `PR <https://github.com/pyportfolio/pyportfolioopt/pull/23>`__ from `TommyBark <https://github.com/TommyBark>`_ 
fixing a bug in the arguments of a call to ``portfolio_performance``.

0.3.3
-----

Migrated the project internally to use the ``poetry`` dependency manager. Will still keep ``setup.py`` and ``requirements.txt``, but ``poetry`` is now the recommended way to interact with PyPortfolioOpt.

0.3.4
-----

Refactored shrinkage models, including single factor and constant correlation.



0.2.0
=====

- Hierarchical Risk Parity optimization
- Semicovariance matrix
- Exponential covariance matrix
- CVaR optimization
- Better support for custom objective functions
- Multiple bug fixes (including minimum volatility vs minimum variance)
- Refactored so all optimizers inherit from a ``BaseOptimizer``.

0.2.1
-----

- Included python 3.7 in travis build
- Merged PR from `schneiderfelipe <https://github.com/schneiderfelipe>`_ to fix an error message.


0.1.0
=====

Initial release:

- Efficient frontier (max sharpe, min variance, target risk/return)
- L2 regularisation
- Discrete allocation
- Mean historical returns, exponential mean returns
- Sample covariance, sklearn wrappers.
- Tests
- Docs

0.1.1
-----

Minor bug fixes and documentation
