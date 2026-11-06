"""Tests for the minimal Python binding (`cytopk.simulate_twocmt_iv`)."""

import numpy as np
import pytest

import cytopk

TIMES = np.array([0.0, 0.5, 1.0, 2.0, 7.0, 14.0, 28.0])


def closed_form_bolus(theta, dose, t):
    """Two-compartment central concentration after an IV bolus at t=0 (bi-exponential)."""
    cl, v1, q, v2 = theta
    k10, k12, k21 = cl / v1, q / v1, q / v2
    s = k10 + k12 + k21
    a = 0.5 * (s + np.sqrt((k10 + k12 - k21) ** 2 + 4 * k12 * k21))
    b = k10 * k21 / a
    return dose / v1 * ((a - k21) / (a - b) * np.exp(-a * t) + (k21 - b) / (a - b) * np.exp(-b * t))


def sim(theta, **kw):
    return cytopk.simulate_twocmt_iv(
        np.asarray(theta, dtype=np.float64), TIMES, np.array([0.0]), np.array([100.0]), np.array([0.0]), **kw
    )


def test_shape_and_closed_form():
    rng = np.random.default_rng(0)
    theta = np.exp(np.log([0.2, 3.0, 0.5, 3.0]) + 0.5 * rng.standard_normal((50, 4)))
    y = sim(theta, rtol=1e-10, atol=1e-12)
    assert y.shape == (50, TIMES.size, 1) and y.dtype == np.float64
    ref = np.stack([closed_form_bolus(th, 100.0, TIMES) for th in theta])
    np.testing.assert_allclose(y[:, :, 0], ref, rtol=1e-7)


def test_thread_count_does_not_change_results():
    theta = np.exp(np.random.default_rng(1).normal(np.log([0.2, 3.0, 0.5, 3.0]), 0.5, (500, 4)))
    a, b, c = sim(theta, n_threads=1), sim(theta, n_threads=4), sim(theta)
    assert np.array_equal(a, b) and np.array_equal(a, c)


def test_non_contiguous_theta():
    theta = np.asfortranarray(np.tile([0.2, 3.0, 0.5, 3.0], (10, 1)))
    assert np.array_equal(sim(theta), sim(np.ascontiguousarray(theta)))


@pytest.mark.parametrize(
    "theta, match",
    [
        ([[0.2, 0.0, 0.5, 3.0]], "V1"),
        ([[0.2, 3.0, 0.5]], "expects 4 parameters"),
        ([[0.2, 3.0, np.nan, 3.0]], "Q"),
    ],
)
def test_invalid_parameters_raise_value_error(theta, match):
    with pytest.raises(ValueError, match=match):
        sim(theta)


def test_invalid_inputs_raise_value_error():
    th = np.array([[0.2, 3.0, 0.5, 3.0]])
    with pytest.raises(ValueError, match="equal length"):
        cytopk.simulate_twocmt_iv(th, TIMES, np.array([0.0]), np.array([1.0, 2.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="output times"):
        cytopk.simulate_twocmt_iv(th, TIMES[::-1].copy(), np.array([0.0]), np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="n_threads"):
        sim(th, n_threads=0)
    with pytest.raises(ValueError, match="maximum number of steps"):
        sim(th, max_steps=2)
