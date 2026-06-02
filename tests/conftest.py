"""Pytest configuration for PyPortfolioOpt tests.

Fixes compatibility between pytest-randomly and numpy's legacy
random seed API. See https://github.com/PyPortfolio/PyPortfolioOpt/issues/725
"""

import numpy as np

_original_np_random_seed = np.random.seed


def _safe_np_random_seed(seed=None):
      """Constrain seed to numpy's valid range [0, 2**32 - 1].

          pytest-randomly may generate seeds exceeding this range when combining
              the base seed with per-test offsets, causing ValueError in numpy's
                  legacy random API.
                      """
      if seed is not None and isinstance(seed, (int, np.integer)):
                seed = seed % (2**32)
            _original_np_random_seed(seed)


np.random.seed = _safe_np_random_seed
