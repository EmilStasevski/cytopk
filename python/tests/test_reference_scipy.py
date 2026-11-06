"""Nonlinear models vs an independent tight-tolerance SciPy reference.

The reference uses SciPy DOP853 (8th order) at rtol=1e-12, atol=1e-14. It integrates segment by segment between
every dose event and output time, so the reference involves no interpolation. cytopk runs at rtol=1e-10, atol=1e-12.
"""

import numpy as np
import pytest

scipy_integrate = pytest.importorskip("scipy.integrate")

import cytopk


def rhs_mm(y, th, rate):
    vmax, km, v1, q, v2 = th
    c1, c2 = y[0] / v1, y[1] / v2
    tr = q * (c1 - c2)
    return np.array([-vmax * c1 / (km + c1) - tr + rate[0], tr + rate[1]])


def rhs_idr(y, th, rate):
    cl, v1, q, v2, kout, emax, ec50 = th
    c1, c2 = y[0] / v1, y[1] / v2
    tr = q * (c1 - c2)
    return np.array([
        -cl * c1 - tr + rate[0],
        tr + rate[1],
        kout * (1 + emax * c1 / (ec50 + c1)) - kout * y[2] + rate[2],
    ])


MODELS = {
    "twocmt_mm_iv": (rhs_mm, lambda th: np.zeros(2), lambda y, th: [y[0] / th[2]]),
    "twocmt_iv_idr": (rhs_idr, lambda th: np.array([0.0, 0.0, 1.0]),
                      lambda y, th: [y[0] / th[1], y[2] / (y[2] + 9.0)]),
}


def reference(model, theta, times, doses):
    """doses: list of (time, amount, duration, state); duration 0 = bolus."""
    rhs, init, obs = MODELS[model]
    y = init(theta)
    events = {0.0} | {d[0] for d in doses} | {d[0] + d[2] for d in doses if d[2] > 0} | set(times)
    stops = sorted(t for t in events if t <= times[-1])
    out, t = {}, 0.0
    for s in stops:
        if s > t:
            rate = np.zeros(y.size)
            for t0, amt, dur, st in doses:
                if dur > 0 and t0 <= t < t0 + dur:
                    rate[st] += amt / dur
            sol = scipy_integrate.solve_ivp(lambda _t, yy: rhs(yy, theta, rate), (t, s), y,
                                            method="DOP853", rtol=1e-12, atol=1e-14)
            assert sol.success
            y = sol.y[:, -1]
            t = s
        for t0, amt, dur, st in doses:
            if dur == 0 and t0 == s:
                y = y.copy()
                y[st] += amt
        out[s] = obs(y, theta)
    return np.array([out[t] for t in times])


WEEKLY = [(7.0 * i, 800.0, 1 / 24, 0) for i in range(4)]
SCENARIOS = [
    ("twocmt_mm_iv", [30.0, 2.0, 3.0, 0.5, 3.0], [(0.0, 500.0, 0.0, 0)]),          # saturating bolus
    ("twocmt_mm_iv", [30.0, 2.0, 3.0, 0.5, 3.0], [(0.0, 10.0, 0.0, 0)]),           # near-linear regime
    ("twocmt_mm_iv", [60.0, 5.0, 4.0, 1.0, 6.0], WEEKLY),                          # repeated infusions
    ("twocmt_iv_idr", [0.2, 3.0, 0.5, 3.0, 0.3, 2.0, 5.0], [(0.0, 200.0, 1 / 24, 0)]),
    ("twocmt_iv_idr", [0.5, 3.0, 1.0, 5.0, 2.0, 5.0, 1.0], WEEKLY),               # fast kout
    ("twocmt_iv_idr", [0.2, 3.0, 0.5, 3.0, 0.3, 2.0, 5.0], [(0.0, 300.0, 0.0, 1), (3.0, 50.0, 0.5, 0)]),
]
TIMES = np.array([0.0, 0.01, 0.04, 0.5, 1, 2, 3, 3.5, 4, 7, 7.04, 10, 14, 14.04, 21, 21.04, 24, 28, 35.0])


@pytest.mark.parametrize("model, theta, doses", SCENARIOS)
def test_matches_tight_scipy_reference(model, theta, doses):
    theta = np.array(theta)
    ref = reference(model, theta, TIMES, doses)
    d = np.array(doses, dtype=float)
    y = cytopk.simulate(model, theta[None, :], TIMES, d[:, 0], d[:, 1], d[:, 2],
                        dose_state=d[:, 3].astype(np.int64), rtol=1e-10, atol=1e-12)[0]
    assert y.shape == ref.shape
    for k in range(ref.shape[1]):
        r, a = ref[:, k], y[:, k]
        mask = np.abs(r) > 1e-6 * np.abs(r).max()
        rel = np.max(np.abs(a - r)[mask] / np.abs(r[mask]))
        assert rel < 1e-7, f"{model} output {k}: max rel err {rel:.2e}"
