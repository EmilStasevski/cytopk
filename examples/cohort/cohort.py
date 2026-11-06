"""Synthetic cytometry cohort for the M8 end-to-end example (illustrative toy model, no real data).

Per subject: log theta (7 PK/PD parameters of `twocmt_iv_idr`, the E1 prior of benchmarks/sbi_flow) -> the IDR
activated-cell fraction p(t) at the 9 E1 sampling times after one 200 mg 1-hour IV infusion. Per sample (one
subject, one time): the activated fraction among CD3+ cells is q ~ Beta(p phi, (1 - p) phi) (sample-to-sample
variation, as in E1), and N_EVENTS events are drawn from a mixture of axis-aligned Gaussians on the arcsinh
scale (cofactor COFACTOR) over three channels:

    population        weight            FSC-A        CD3          CD69        (arcsinh scale: mean, sd)
    debris            W_DEBRIS          (1.0, 0.5)   (0.5, 0.6)   (0.5, 0.6)
    CD3- cells        W_CD3NEG          (4.0, 0.4)   (1.0, 0.5)   (0.8, 0.5)
    CD3+ resting      W_CD3POS (1 - q)  (4.0, 0.4)   (4.0, 0.5)   (0.8, 0.5)
    CD3+ activated    W_CD3POS q        (4.2, 0.4)   (4.0, 0.5)   (3.2, 0.6)

The events are written as linear values (COFACTOR * sinh(y)) to float32 FCS 3.1 files with the cytopk writer,
read back with cytopk, and gated with STRATEGY (cells -> cd3 -> activated). The gate count summaries are the
observations for BayesFlow.

Because every gate is a rectangle and every component is an axis-aligned Gaussian, the probability that one
event falls into each gate is a closed-form sum of products of normal CDFs, so for a given q the gate counts
are exactly multinomial (`gate_probs`, `counts_fast`). Training uses that fast path; the SBC test set goes through
the real FCS write -> read -> gate pipeline (`counts_via_fcs`), so a mismatch between the two would show up as
miscalibration. `python/tests/test_cohort.py` also checks the two against each other directly.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "sbi_flow"))
from common import (DOSE_AMT, DOSE_DUR, DOSE_TIME, PARAM_NAMES, PHI, PRIOR_MEDIAN, PRIOR_SD,  # noqa: E402
                    TIMES, prior_batch)

COFACTOR = 150.0
N_EVENTS = 10_000
CHANNELS = ["FSC-A", "CD3", "CD69"]
W_DEBRIS, W_CD3NEG, W_CD3POS = 0.10, 0.30, 0.60
# (mean, sd) per channel, arcsinh scale; order: debris, CD3-, CD3+ resting, CD3+ activated
MEANS = np.array([[1.0, 0.5, 0.5], [4.0, 1.0, 0.8], [4.0, 4.0, 0.8], [4.2, 4.0, 3.2]])
SDS = np.array([[0.5, 0.6, 0.6], [0.4, 0.5, 0.5], [0.4, 0.5, 0.5], [0.4, 0.5, 0.6]])

STRATEGY = f"""
# M8 example gating: arcsinh with cofactor {COFACTOR:g} on every channel, then cells -> cd3 -> activated.
[transform]
cofactor = {COFACTOR}
arcsinh = ["FSC-A", "CD3", "CD69"]

[stats]
medians = ["CD3", "CD69"]

[[gate]]
name = "cells"
range = {{ "FSC-A" = [2.5, inf] }}

[[gate]]
name = "cd3"
parent = "cells"
range = {{ CD3 = [2.5, inf] }}

[[gate]]
name = "activated"
parent = "cd3"
range = {{ CD69 = [2.0, inf] }}
"""
# Gate rectangles on the arcsinh scale, per gate the (lo, hi) per channel; the nesting is cumulative.
GATES = {"cells": {0: (2.5, math.inf)}, "cd3": {0: (2.5, math.inf), 1: (2.5, math.inf)},
         "activated": {0: (2.5, math.inf), 1: (2.5, math.inf), 2: (2.0, math.inf)}}
GATE_NAMES = list(GATES)


def weights(q):
    """Mixture weights (..., 4) for activated fraction q (...)."""
    q = np.asarray(q, dtype=np.float64)
    return np.stack([np.full_like(q, W_DEBRIS), np.full_like(q, W_CD3NEG), W_CD3POS * (1 - q), W_CD3POS * q],
                    axis=-1)


def _ncdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _box_prob(k, box):
    p = 1.0
    for j, (lo, hi) in box.items():
        m, s = MEANS[k, j], SDS[k, j]
        p *= (_ncdf((hi - m) / s) if hi < math.inf else 1.0) - (_ncdf((lo - m) / s) if lo > -math.inf else 0.0)
    return p


# P(event of component k is in gate g), shape (4 components, 3 gates)
BOX = np.array([[_box_prob(k, GATES[g]) for g in GATE_NAMES] for k in range(4)])


def gate_probs(q):
    """P(one event is in each gate) for activated fraction q: (..., 3) in GATE_NAMES order (nested gates)."""
    return weights(q) @ BOX


def pd_fraction(log_theta, n_threads=None):
    """Noise-free IDR activated fraction p at TIMES, shape (B, 9)."""
    import cytopk
    y = cytopk.simulate("twocmt_iv_idr", np.exp(log_theta), TIMES, np.array([DOSE_TIME]), np.array([DOSE_AMT]),
                        np.array([DOSE_DUR]), rtol=10 ** -7.5, atol=10 ** -10.5, n_threads=n_threads)
    return y[:, :, 1]


def sample_q(rng, p):
    """Per-sample activated fraction among CD3+ cells, q ~ Beta(p phi, (1 - p) phi)."""
    return rng.beta(p * PHI, (1.0 - p) * PHI)


def counts_fast(rng, q, n_events=N_EVENTS):
    """Gate counts (..., 3) [cells, cd3, activated] drawn exactly as the event-level model would give them:
    a multinomial over the disjoint regions outside cells / cells not cd3 / cd3 not activated / activated."""
    g = gate_probs(q)
    regions = np.stack([1 - g[..., 0], g[..., 0] - g[..., 1], g[..., 1] - g[..., 2], g[..., 2]], axis=-1)
    regions = np.clip(regions, 0.0, 1.0)
    regions /= regions.sum(-1, keepdims=True)
    flat = regions.reshape(-1, 4)
    c = np.array([rng.multinomial(n_events, r) for r in flat]).reshape(*regions.shape)
    return np.stack([c[..., 1] + c[..., 2] + c[..., 3], c[..., 2] + c[..., 3], c[..., 3]], axis=-1)


def events(rng, q, n_events=N_EVENTS):
    """One sample's events on the arcsinh scale, (n_events, 3), shuffled; and the component labels."""
    lab = rng.choice(4, size=n_events, p=weights(q))
    y = MEANS[lab] + SDS[lab] * rng.standard_normal((n_events, 3))
    return y, lab


def to_linear(y):
    return (COFACTOR * np.sinh(y)).astype(np.float32)


def write_sample(path, rng, q, subject, time, n_events=N_EVENTS):
    import cytopk
    y, _ = events(rng, q, n_events)
    cytopk.write_fcs(path, to_linear(y), CHANNELS, long_names=["forward scatter", "CD3", "CD69"],
                     keywords={"$SRC": f"cytopk M8 synthetic subject {subject}", "SUBJECT": str(subject),
                               "TIME_DAYS": f"{time:g}", "SYNTHETIC": "yes"})


def counts_via_fcs(paths, n_threads=None):
    """Read and gate FCS files with cytopk; counts (len(paths), 3) [cells, cd3, activated]."""
    import cytopk
    s = cytopk.GatingStrategy(STRATEGY)
    out = []
    for r in cytopk.gate_fcs_many([str(p) for p in paths], s, n_threads=n_threads):
        if isinstance(r, Exception):
            raise r
        out.append([g["count"] for g in r["gates"]])
    return np.array(out, dtype=np.int64)


def summaries(counts):
    """Observation for BayesFlow from gate counts (..., 9 times, 3): per time [logit activated/cd3,
    log cd3/cells]; continuity-corrected, so empty gates stay finite. Shape (..., 9, 2)."""
    c = counts.astype(np.float64)
    a = (c[..., 2] + 0.5) / (c[..., 1] + 1.0)
    b = (c[..., 1] + 0.5) / (c[..., 0] + 1.0)
    return np.stack([np.log(a / (1 - a)), np.log(b)], axis=-1)


def simulate_fast(rng, n, n_threads=None):
    """Training batch: log theta (n, 7) and summaries (n, 9, 2) via the exact multinomial path."""
    log_theta = prior_batch(rng, n)
    q = sample_q(rng, pd_fraction(log_theta, n_threads))
    return log_theta, summaries(counts_fast(rng, q))


__all__ = ["CHANNELS", "COFACTOR", "GATE_NAMES", "N_EVENTS", "PARAM_NAMES", "PRIOR_MEDIAN", "PRIOR_SD",
           "STRATEGY", "TIMES", "counts_fast", "counts_via_fcs", "events", "gate_probs", "pd_fraction",
           "sample_q", "simulate_fast", "summaries", "write_sample"]
