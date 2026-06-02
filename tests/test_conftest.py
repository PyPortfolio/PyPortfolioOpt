"""Tests for the conftest.py seed safety patch."""

import numpy as np
import pytest


class TestSafeNumpySeed:
      """Verify numpy.random.seed handles oversized seeds from pytest-randomly."""

    def test_seed_exceeding_uint32_max(self):
              """Seeds above 2**32 - 1 should not raise ValueError."""
              np.random.seed(2**32 + 12345)

    def test_seed_at_uint32_boundary(self):
              """Seed exactly at 2**32 should wrap to 0."""
              np.random.seed(2**32)

    def test_seed_zero(self):
              """Seed 0 should work as normal."""
              np.random.seed(0)

    def test_seed_max_valid(self):
              """Seed 2**32 - 1 should work as normal."""
              np.random.seed(2**32 - 1)

    def test_seed_none(self):
              """Seed None should work as normal (random seed)."""
              np.random.seed(None)

    def test_determinism_preserved(self):
              """Patched seed should still produce deterministic results."""
              np.random.seed(42)
              a = np.random.rand(5)
              np.random.seed(42)
              b = np.random.rand(5)
              np.testing.assert_array_equal(a, b)

    def test_large_seed_determinism(self):
              """Oversized seeds should produce deterministic results."""
              large_seed = 2**33 + 99
              np.random.seed(large_seed)
              a = np.random.rand(5)
              np.random.seed(large_seed)
              b = np.random.rand(5)
              np.testing.assert_array_equal(a, b)
      
