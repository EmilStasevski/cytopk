"""Tests for cytopk.simulate_ddm (drift-diffusion model)."""

import math

import numpy as np
import pytest

import cytopk


def test_shapes_dtypes_and_determinism():
    th = np.tile([1.0, 1.5, 0.4, 0.3], (2000, 1))
    rt, ch = cytopk.simulate_ddm(th, seed=1)
    assert rt.shape == ch.shape == (2000,) and rt.dtype == np.float64 and ch.dtype == np.int8
    assert set(np.unique(ch)) <= {0, 1}
    assert np.all(rt > 0.3)
    a = cytopk.simulate_ddm(th, seed=1, n_threads=1)
    b = cytopk.simulate_ddm(th, seed=1, n_threads=4)
    assert np.array_equal(a[1], ch) and np.array_equal(b[1], ch)
    assert np.array_equal(a[0].view(np.uint64), rt.view(np.uint64))
    assert not np.array_equal(cytopk.simulate_ddm(th, seed=2)[0], rt)


def test_choice_probability_close_to_corrected_analytic():
    v, a, z = 1.0, 1.5, 0.4
    n = 200_000
    _, ch = cytopk.simulate_ddm(np.tile([v, a, z, 0.3], (n, 1)), seed=5)
    shift = 0.5825971579390106 * math.sqrt(0.001)
    p = lambda a_, za: (1 - math.exp(-2 * v * za)) / (1 - math.exp(-2 * v * a_))  # noqa: E731
    pc, p0 = p(a + 2 * shift, z * a + shift), p(a, z * a)
    tol = 4 * math.sqrt(pc * (1 - pc) / n) + 0.1 * abs(p0 - pc)
    assert abs(np.mean(ch == 1) - pc) <= tol


def test_timeout_and_validation():
    rt, ch = cytopk.simulate_ddm(np.array([[0.0, 100.0, 0.5, 0.2]]), seed=0, max_t=0.01)
    assert ch[0] == -1 and math.isnan(rt[0])
    with pytest.raises(ValueError, match="z"):
        cytopk.simulate_ddm(np.array([[1.0, 1.0, 1.5, 0.2]]), seed=0)
    with pytest.raises(ValueError, match="expects 4 parameters"):
        cytopk.simulate_ddm(np.zeros((1, 3)), seed=0)
    with pytest.raises(ValueError, match="dt and max_t"):
        cytopk.simulate_ddm(np.array([[1.0, 1.0, 0.5, 0.2]]), seed=0, dt=-1.0)
