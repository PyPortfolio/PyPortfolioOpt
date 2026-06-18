"""Pytest configuration for PyPortfolioOpt tests.

Fixes compatibility between pytest-randomly and numpy's legacy
random seed API. See https://github.com/PyPortfolio/PyPortfolioOpt/issues/725
"""

import numpy as np

_original_np_random_seed = np.random.seed
_UINT32_MODULUS = 2**32


def _safe_np_random_seed(seed=None):
    """Wrap oversized nonnegative integer seeds into NumPy's valid range."""
    if isinstance(seed, (int, np.integer)) and seed >= _UINT32_MODULUS:
        seed = seed % _UINT32_MODULUS
    _original_np_random_seed(seed)


np.random.seed = _safe_np_random_seed
