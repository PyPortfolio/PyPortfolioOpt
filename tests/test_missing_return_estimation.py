import numpy as np
import pandas as pd
import pytest

from pypfopt import risk_models


def estimate(data, method):
    model = risk_models.CovarianceShrinkage(data, returns_data=True, frequency=1)
    return (
        model.oracle_approximating() if method == "oas" else model.ledoit_wolf(method)
    )


@pytest.mark.parametrize(
    "method", ["constant_variance", "single_factor", "constant_correlation", "oas"]
)
def test_shrinkage_uses_complete_observations(method):
    rng = np.random.default_rng(52)
    returns = pd.DataFrame(rng.normal(size=(120, 3)) / 100, columns=list("ABC"))
    returns.loc[:79, "B"] = np.nan
    returns.loc[95, "C"] = np.nan
    original = returns.copy()
    with pytest.warns(UserWarning, match="81.*incomplete"):
        actual = estimate(returns, method)
    expected = estimate(returns.dropna(), method)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(returns, original)


@pytest.mark.parametrize(
    "method", ["constant_variance", "single_factor", "constant_correlation", "oas"]
)
@pytest.mark.parametrize("rows", [0, 1])
def test_shrinkage_requires_two_complete_observations(method, rows):
    returns = pd.DataFrame({"A": [0.1, 0.2, 0.3], "B": [np.nan, np.nan, np.nan]})
    if rows:
        returns.loc[2, "B"] = 0.1
    with pytest.raises(ValueError, match="two complete"):
        estimate(returns, method)


def test_semicovariance_uses_pairwise_observation_counts():
    returns = pd.DataFrame(
        {"A": [-0.1, 0.2, np.nan, -0.2], "B": [np.nan, -0.1, -0.1, 0.4]}
    )
    actual = risk_models.semicovariance(
        returns, returns_data=True, benchmark=0, frequency=1
    )
    expected = pd.DataFrame(
        [[1 / 60, 0], [0, 1 / 150]], index=returns.columns, columns=returns.columns
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_semicovariance_keeps_observed_zero_returns():
    returns = pd.DataFrame({"A": [-0.1, 0.0, -0.2], "B": [0.0, -0.1, 0.0]})
    actual = risk_models.semicovariance(
        returns, returns_data=True, benchmark=0, frequency=1
    )
    np.testing.assert_allclose(actual, [[1 / 60, 0], [0, 1 / 300]])


def test_semicovariance_rejects_pairs_without_joint_observations():
    returns = pd.DataFrame({"A": [0.1, np.nan], "B": [np.nan, 0.2]})
    with pytest.raises(ValueError, match="joint observation"):
        risk_models.semicovariance(returns, returns_data=True)


def test_semicovariance_uses_joint_counts_off_diagonal():
    returns = pd.DataFrame({"A": [-0.1, -0.1, np.nan], "B": [-0.1, np.nan, -0.1]})
    actual = risk_models.semicovariance(
        returns, returns_data=True, benchmark=0, frequency=1
    )
    np.testing.assert_allclose(actual, np.full((2, 2), 0.01))


@pytest.mark.parametrize("frequency", [1, 252])
@pytest.mark.parametrize(
    "fix_method, expected",
    [
        ("spectral", [[0.0075, 0.0075], [0.0075, 0.0075]]),
        ("diag", [[0.0105, 0.01], [0.01, 0.0105]]),
    ],
)
def test_semicovariance_repairs_indefinite_pairwise_estimate(
    fix_method, expected, frequency
):
    returns = pd.DataFrame({"A": [-0.1, 0.0, np.nan], "B": [-0.1, np.nan, 0.0]})
    original = returns.copy()
    # Each variance uses two observations; the cross-product uses one.
    # Raw matrix [[.005, .01], [.01, .005]] has eigenvalues -.005 and .015.
    # Spectral clipping gives .015/2 in every entry (diagonal increases 50%).
    # Diagonal repair instead adds 1.1 * .005 to both diagonal entries.
    with pytest.warns(UserWarning, match="non positive semidefinite") as caught:
        actual = risk_models.semicovariance(
            returns,
            returns_data=True,
            benchmark=0,
            frequency=frequency,
            fix_method=fix_method,
        )
    assert len(caught) == 1
    expected = pd.DataFrame(
        np.array(expected) * frequency, index=returns.columns, columns=returns.columns
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert np.linalg.eigvalsh(actual).min() >= -1e-12
    pd.testing.assert_frame_equal(returns, original)


def test_semicovariance_complete_data_retains_full_sample_formula():
    returns = pd.DataFrame({"A": [-0.1, 0.02, 0.0], "B": [0.03, -0.05, -0.02]})
    benchmark = 0.01
    downside = np.minimum(returns.to_numpy() - benchmark, 0)
    expected = downside.T @ downside / len(returns) * 252
    actual = risk_models.semicovariance(
        returns, returns_data=True, benchmark=benchmark, frequency=252
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
