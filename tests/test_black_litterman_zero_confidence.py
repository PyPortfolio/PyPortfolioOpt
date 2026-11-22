import numpy as np
import pandas as pd
import pytest

from pypfopt.black_litterman import BlackLittermanModel


@pytest.mark.parametrize("cov_first", [False, True])
@pytest.mark.parametrize("scale", [1.0, 1e9])
def test_idzorek_zero_confidence_is_no_information(cov_first, scale):
    sigma = scale * np.array([[2.0, 0.7], [0.7, 1.0]])
    labels = ["synthetic_b", "synthetic_a"]
    prior = np.array([0.01, -0.02])
    model = BlackLittermanModel(
        pd.DataFrame(sigma, index=labels, columns=labels), pi=prior,
        absolute_views={"synthetic_a": -1.0, "synthetic_b": 1.0},
        omega="idzorek", view_confidences=np.array([[0.0], [0.0]]), tau=0.1,
    )
    if cov_first:
        model.bl_cov()
    mean = model.bl_returns()
    covariance = model.bl_cov()
    np.testing.assert_array_equal(mean, prior)
    np.testing.assert_allclose(covariance, 1.1 * sigma, rtol=1e-13, atol=0)
    assert list(mean.index) == list(covariance.columns) == list(covariance.index) == labels
    assert model.Q.shape == (2, 1) and model.P.shape == model.omega.shape == (2, 2)
    np.testing.assert_array_equal(model.omega, np.eye(2) * 1e6)


@pytest.mark.parametrize("confidence", [0.25, 1.0])
@pytest.mark.parametrize("cov_first", [False, True])
def test_idzorek_mixed_matches_scalar_gaussian_conditioning(confidence, cov_first):
    sigma = 1e7 * np.array([[3.0, 0.5], [0.5, 1.0]])
    prior = np.array([0.2, -0.4])
    active_view = np.array([1.0, 1.0])
    ignored_view = np.array([1.0, -1.0])
    correct = np.array([0.7, 25.0])
    tau = 0.1
    model = BlackLittermanModel(
        sigma, pi=prior, Q=correct, P=np.array([active_view, ignored_view]),
        omega="idzorek", view_confidences=[confidence, 0.0], tau=tau,
    )
    if cov_first:
        model.bl_cov()
    p = sigma @ active_view
    v = active_view @ p
    gain = confidence / v
    expected = prior + gain * p * (correct[0] - active_view @ prior)
    predictive = sigma + tau * (sigma - gain * np.outer(p, p))
    np.testing.assert_allclose(model.bl_returns(), expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(model.bl_cov(), predictive, atol=1e-8, rtol=1e-14)
    system = model._A
    np.testing.assert_allclose(model.bl_returns(), expected, atol=1e-14, rtol=0)
    assert model._A is system


def test_custom_finite_noise_value_is_not_an_idzorek_marker():
    sigma = np.array([[1e7]])
    noise = np.array([[1e6]])
    model = BlackLittermanModel(
        sigma, pi=np.array([0.0]), Q=np.array([1.0]), P=np.array([[1.0]]),
        omega=noise, tau=0.1,
    )
    np.testing.assert_allclose(model.bl_returns(), [0.5], atol=0, rtol=0)
    np.testing.assert_allclose(model.bl_cov(), [[10.5e6]], atol=0, rtol=0)
    np.testing.assert_array_equal(model.omega, noise)


@pytest.mark.parametrize("replacement", [True, False])
def test_omega_explicitly_modified_before_solve_still_applies(replacement):
    model = BlackLittermanModel(
        np.array([[1e7]]), pi=np.array([0.0]), Q=np.array([1.0]),
        P=np.array([[1.0]]), omega="idzorek", view_confidences=[0], tau=0.1,
    )
    if replacement:
        model.omega = np.array([[2e6]])
    else:
        model.omega[0, 0] = 2e6
    np.testing.assert_allclose(model.bl_returns(), [1 / 3])
    np.testing.assert_allclose(model.bl_cov(), [[1e7 + 1e6 * 2 / 3]])


def test_original_confidence_input_not_retained():
    confidence = np.array([0.])
    model = BlackLittermanModel(
        np.array([[1e7]]), pi=np.array([0.0]), Q=np.array([1.0]),
        P=np.array([[1.0]]), omega="idzorek", view_confidences=confidence, tau=0.1,
    )
    confidence[0] = 0.5
    np.testing.assert_allclose(model.bl_returns(), [0.0])


def test_idzorek_zero_confidence_preserves_legacy_singular_weight_path():
    direction = np.array([2.0, 1.0])
    sigma = np.outer(direction, direction)
    prior = 0.2 * direction
    model = BlackLittermanModel(
        sigma, pi=prior, Q=prior.copy(), P=np.eye(2),
        omega="idzorek", view_confidences=[0.0, 0.0], tau=0.1, risk_aversion=2,
    )
    weights = model.bl_weights()
    # In the consistent rank-one system, the minimum-norm solution is
    # parallel to the range direction and its normalized sum is 1.
    np.testing.assert_allclose(list(weights.values()), [2 / 3, 1 / 3])
    np.testing.assert_allclose(model.posterior_rets, prior)
    assert model._A.shape == (2, 2)
