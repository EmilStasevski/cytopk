"""M8 cohort generator (examples/cohort): the FCS pipeline and the fast multinomial path agree."""

import sys
from pathlib import Path

import numpy as np
import pytest

import cytopk

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "cohort"))
cohort = pytest.importorskip("cohort")


def test_pipeline_counts_equal_numpy_counts(tmp_path):
    """Write -> read -> gate with cytopk equals gating the float32 values in NumPy (deterministic)."""
    rng = np.random.default_rng(5)
    for q in [0.05, 0.3]:
        path = tmp_path / f"q{q}.fcs"
        cohort.write_sample(path, np.random.default_rng(7), q, subject=0, time=1.0)
        y, _ = cohort.events(np.random.default_rng(7), q)
        x = cohort.to_linear(y).astype(np.float64)
        t = np.arcsinh(x / cohort.COFACTOR)
        cells = t[:, 0] >= 2.5
        cd3 = cells & (t[:, 1] >= 2.5)
        act = cd3 & (t[:, 2] >= 2.0)
        got = cohort.counts_via_fcs([path])[0]
        assert got.tolist() == [cells.sum(), cd3.sum(), act.sum()]
        f = cytopk.read_fcs(path)
        assert f.text["SYNTHETIC"] == "yes" and f.names == cohort.CHANNELS
    del rng


def test_event_model_matches_closed_form_gate_probabilities():
    """Mean gate counts of the event-level model equal N * gate_probs(q) (z < 5 over 200 samples)."""
    rng = np.random.default_rng(11)
    s = cytopk.GatingStrategy(cohort.STRATEGY)
    for q in [0.1, 0.25]:
        c = []
        for _ in range(200):
            y, _ = cohort.events(rng, q)
            r = s.apply(cohort.to_linear(y), names=cohort.CHANNELS)
            c.append([g["count"] for g in r["gates"]])
        c = np.array(c, dtype=np.float64)
        p = cohort.gate_probs(q)
        mean, se = c.mean(0), np.sqrt(cohort.N_EVENTS * p * (1 - p) / len(c))
        assert np.all(np.abs(mean - cohort.N_EVENTS * p) < 5 * se), (mean, cohort.N_EVENTS * p, se)
        fast = cohort.counts_fast(rng, np.full(2000, q)).astype(np.float64)
        assert np.all(np.abs(fast.mean(0) - cohort.N_EVENTS * p) < 5 * se / np.sqrt(10))
        np.testing.assert_allclose(fast.std(0), c.std(0), rtol=0.25)


def test_fast_simulator_shapes_and_finiteness():
    th, s = cohort.simulate_fast(np.random.default_rng(0), 64)
    assert th.shape == (64, 7) and s.shape == (64, 9, 2)
    assert np.isfinite(s).all()
    p = cohort.pd_fraction(th)
    assert np.all((p >= 0.1 - 1e-9) & (p < 1))
