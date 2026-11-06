"""Shared model definition and simulator implementations for the SBI-flow experiments (E1, E2).

Model: `twocmt_iv_idr` (docs/models.md). One 200 mg 1-hour IV infusion at day 0, 9 sampling times.
- Prior: log-normal around the illustrative values in docs/models.md (median CL 0.2, V1 3, Q 0.5, V2 3, kout 0.3,
  Emax 2, EC50 5), sd 0.3 on the log scale. The inference target is log(theta), 7 values.
- Observations per time point: log C_obs = log C + 0.15 * eps (PK noise), and a gate count
  k ~ BetaBinomial(n = 10 000, p * phi, (1 - p) * phi) with phi = 100.
- The simulator output `obs` has shape (9, 3): [time, log C_obs, k / n].

The ODE part differs between implementations. The noise stage is the same NumPy code for all of them.
Tolerances are the values tuned to an achieved max relative error <= 1e-6 in benchmarks/bench_nonlinear.py
(Dormand-Prince rtol 10^-7.5; SciPy RK45 rtol 10^-6.25). Agreement on this design is re-checked by `agreement()`.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

import numpy as np

TIMES = np.array([0.04, 0.5, 1.0, 2.0, 4.0, 7.0, 14.0, 21.0, 28.0])
DOSE_TIME, DOSE_AMT, DOSE_DUR = 0.0, 200.0, 1.0 / 24.0
PRIOR_MEDIAN = np.array([0.2, 3.0, 0.5, 3.0, 0.3, 2.0, 5.0])
PRIOR_SD = 0.3
SIGMA_PK, N_EVENTS, PHI = 0.15, 10_000, 100.0
PARAM_NAMES = ["CL", "V1", "Q", "V2", "kout", "Emax", "EC50"]

RTOL_DP = 10 ** -7.5  # cytopk / Numba Dormand-Prince
RTOL_SCIPY = 10 ** -6.25  # SciPy RK45
ATOL_FACTOR = 1e-3  # atol = 1e-3 * rtol (mg)
P_CORE = 0  # i5-14400F: CPUs 0-11 are P-core threads


# ------------------------------------------------------------------ prior and noise (shared)
def prior_batch(rng, n):
    return np.log(PRIOR_MEDIAN) + PRIOR_SD * rng.standard_normal((n, 7))


def add_noise(rng, conc, p):
    """conc, p: (B, 9) noise-free outputs -> obs (B, 9, 3) float64."""
    log_c = np.log(conc) + SIGMA_PK * rng.standard_normal(conc.shape)
    q = rng.beta(p * PHI, (1.0 - p) * PHI)
    k = rng.binomial(N_EVENTS, q)
    t = np.broadcast_to(TIMES, conc.shape)
    return np.stack([t, log_c, k / N_EVENTS], axis=-1)


# ------------------------------------------------------------------ ODE implementations (noise-free)
def ode_cytopk(theta, n_threads=None, rtol=RTOL_DP):
    import cytopk
    y = cytopk.simulate("twocmt_iv_idr", theta, TIMES, np.array([DOSE_TIME]), np.array([DOSE_AMT]),
                        np.array([DOSE_DUR]), rtol=rtol, atol=rtol * ATOL_FACTOR, n_threads=n_threads)
    return y[:, :, 0], y[:, :, 1]


def _scipy_one(th, rtol=RTOL_SCIPY):
    from scipy.integrate import solve_ivp
    cl, v1, q, v2, kout, emax, ec50 = th

    def f(t, y, r):
        c1, c2 = y[0] / v1, y[1] / v2
        tr = q * (c1 - c2)
        return [-cl * c1 - tr + r, tr, kout * (1 + emax * c1 / (ec50 + c1)) - kout * y[2]]

    # Piecewise: the infusion [0, 1 h], then no input. t_eval uses SciPy's dense output.
    during = TIMES[TIMES <= DOSE_DUR]
    s1 = solve_ivp(f, (0.0, DOSE_DUR), [0.0, 0.0, 1.0], method="RK45", rtol=rtol, atol=rtol * ATOL_FACTOR,
                   t_eval=np.append(during, DOSE_DUR), args=(DOSE_AMT / DOSE_DUR,))
    s2 = solve_ivp(f, (DOSE_DUR, TIMES[-1]), s1.y[:, -1], method="RK45", rtol=rtol, atol=rtol * ATOL_FACTOR,
                   t_eval=TIMES[TIMES > DOSE_DUR], args=(0.0,))
    y = np.concatenate([s1.y[:, : during.size], s2.y], axis=1)
    return y[0] / v1, y[2] / (y[2] + 9.0)


def ode_scipy(theta):
    out = [_scipy_one(th) for th in theta]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


_NUMBA = {}


def numba_kernels():
    """Same Dormand-Prince design as cytopk-sim, compiled with nogil=True. Returns (serial, parallel)."""
    if _NUMBA:
        return _NUMBA["serial"], _NUMBA["parallel"]
    from numba import njit, prange

    @njit(inline="always", nogil=True)
    def rhs(y, th, r, dy):
        c1 = y[0] / th[1]
        c2 = y[1] / th[3]
        tr = th[2] * (c1 - c2)
        dy[0] = -th[0] * c1 - tr + r
        dy[1] = tr
        dy[2] = th[4] * (1.0 + th[5] * c1 / (th[6] + c1)) - th[4] * y[2]

    @njit(nogil=True)
    def subject(th, stops, rates, out_idx, rtol, atol, oc, op):
        ns = 3
        y = np.zeros(ns); y[2] = 1.0
        yn = np.zeros(ns); ys = np.zeros(ns)
        k1 = np.zeros(ns); k2 = np.zeros(ns); k3 = np.zeros(ns); k4 = np.zeros(ns)
        k5 = np.zeros(ns); k6 = np.zeros(ns); k7 = np.zeros(ns)
        t = 0.0; r = 0.0; h = 0.0
        for s in range(stops.size):
            t1 = stops[s]
            if t1 > t:
                if h <= 0.0:
                    rhs(y, th, r, k1)
                    d0 = 0.0; d1 = 0.0
                    for i in range(ns):
                        sc = atol + rtol * abs(y[i])
                        d0 += (y[i] / sc) ** 2; d1 += (k1[i] / sc) ** 2
                    d0 = np.sqrt(d0 / ns); d1 = np.sqrt(d1 / ns)
                    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
                    for i in range(ns):
                        ys[i] = y[i] + h0 * k1[i]
                    rhs(ys, th, r, k2)
                    d2 = 0.0
                    for i in range(ns):
                        sc = atol + rtol * abs(y[i])
                        d2 += ((k2[i] - k1[i]) / sc) ** 2
                    d2 = np.sqrt(d2 / ns) / h0
                    dm = max(d1, d2)
                    h1 = max(1e-6, h0 * 1e-3) if dm <= 1e-15 else (0.01 / dm) ** 0.2
                    h = min(100 * h0, h1)
                rhs(y, th, r, k1)
                last_rej = False
                while t < t1:
                    rem = t1 - t
                    hh = h
                    clamped = hh >= rem * 0.99
                    if clamped:
                        hh = rem
                    for i in range(ns):
                        ys[i] = y[i] + hh * (0.2 * k1[i])
                    rhs(ys, th, r, k2)
                    for i in range(ns):
                        ys[i] = y[i] + hh * (3 / 40 * k1[i] + 9 / 40 * k2[i])
                    rhs(ys, th, r, k3)
                    for i in range(ns):
                        ys[i] = y[i] + hh * (44 / 45 * k1[i] - 56 / 15 * k2[i] + 32 / 9 * k3[i])
                    rhs(ys, th, r, k4)
                    for i in range(ns):
                        ys[i] = y[i] + hh * (19372 / 6561 * k1[i] - 25360 / 2187 * k2[i]
                                             + 64448 / 6561 * k3[i] - 212 / 729 * k4[i])
                    rhs(ys, th, r, k5)
                    for i in range(ns):
                        ys[i] = y[i] + hh * (9017 / 3168 * k1[i] - 355 / 33 * k2[i] + 46732 / 5247 * k3[i]
                                             + 49 / 176 * k4[i] - 5103 / 18656 * k5[i])
                    rhs(ys, th, r, k6)
                    for i in range(ns):
                        yn[i] = y[i] + hh * (35 / 384 * k1[i] + 500 / 1113 * k3[i] + 125 / 192 * k4[i]
                                             - 2187 / 6784 * k5[i] + 11 / 84 * k6[i])
                    rhs(yn, th, r, k7)
                    err = 0.0
                    for i in range(ns):
                        e = hh * (71 / 57600 * k1[i] - 71 / 16695 * k3[i] + 71 / 1920 * k4[i]
                                  - 17253 / 339200 * k5[i] + 22 / 525 * k6[i] - 1 / 40 * k7[i])
                        sc = atol + rtol * max(abs(y[i]), abs(yn[i]))
                        err += (e / sc) ** 2
                    err = np.sqrt(err / ns)
                    fac = 10.0 if err == 0.0 else min(10.0, max(0.2, 0.9 * err ** -0.2))
                    if err <= 1.0:
                        for i in range(ns):
                            y[i] = yn[i]; k1[i] = k7[i]
                        t = t1 if clamped else t + hh
                        if last_rej:
                            fac = min(fac, 1.0)
                        hn = hh * fac
                        h = max(h, hn) if clamped else hn
                        last_rej = False
                    else:
                        h = hh * min(fac, 1.0)
                        last_rej = True
            t = t1
            r = rates[s]
            j = out_idx[s]
            if j >= 0:
                oc[j] = y[0] / th[1]
                op[j] = y[2] / (y[2] + 9.0)

    @njit(nogil=True)
    def serial(theta, stops, rates, out_idx, rtol, atol, oc, op):
        for n in range(theta.shape[0]):
            subject(theta[n], stops, rates, out_idx, rtol, atol, oc[n], op[n])

    @njit(nogil=True, parallel=True)
    def parallel(theta, stops, rates, out_idx, rtol, atol, oc, op):
        for n in prange(theta.shape[0]):
            subject(theta[n], stops, rates, out_idx, rtol, atol, oc[n], op[n])

    _NUMBA.update(serial=serial, parallel=parallel)
    return serial, parallel


def _plan():
    stops = np.unique(np.concatenate([[DOSE_TIME, DOSE_TIME + DOSE_DUR], TIMES]))
    rates = np.array([DOSE_AMT / DOSE_DUR if DOSE_TIME <= s < DOSE_TIME + DOSE_DUR else 0.0 for s in stops])
    out_idx = np.array([int(np.where(TIMES == s)[0][0]) if s in TIMES else -1 for s in stops], dtype=np.int64)
    return stops, rates, out_idx


PLAN = _plan()


def ode_numba(theta, parallel=True, rtol=RTOL_DP):
    serial, par = numba_kernels()
    fn = par if parallel else serial
    theta = np.ascontiguousarray(theta)
    oc = np.empty((theta.shape[0], TIMES.size))
    op = np.empty_like(oc)
    fn(theta, *PLAN, rtol, rtol * ATOL_FACTOR, oc, op)
    return oc, op


# ------------------------------------------------------------------ BayesFlow simulators
ODE = {"scipy_loop": ode_scipy, "numba_parallel": lambda th: ode_numba(th, True),
       "cytopk": lambda th: ode_cytopk(th, None)}


def batched_simulator(kind, seed=0, ode=None, stage_times=None):
    """(b) SciPy loop, (c) Numba, (d) cytopk: one Python call per batch (LambdaSimulator, is_batched=True).
    If `stage_times` is a list, (t_prior, t_ode, t_noise) is appended to it for each call."""
    import bayesflow as bf
    rng = np.random.default_rng(seed)
    ode = ode or ODE[kind]

    def sample(batch_shape):
        n = int(np.prod(batch_shape))
        t0 = time.perf_counter()
        log_theta = prior_batch(rng, n)
        t1 = time.perf_counter()
        conc, p = ode(np.exp(log_theta))
        t2 = time.perf_counter()
        obs = add_noise(rng, conc, p)
        if stage_times is not None:
            stage_times.append((t1 - t0, t2 - t1, time.perf_counter() - t2))
        return {"log_theta": log_theta, "obs": obs}

    return bf.simulators.LambdaSimulator(sample, is_batched=True)


def per_sample_simulator(seed=0):
    """(a) plain per-sample functions via bf.make_simulator. The ODE uses SciPy solve_ivp (RK45) per sample."""
    import bayesflow as bf
    rng = np.random.default_rng(seed)

    def prior():
        return {"log_theta": np.log(PRIOR_MEDIAN) + PRIOR_SD * rng.standard_normal(7)}

    def likelihood(log_theta):
        conc, p = _scipy_one(np.exp(log_theta))
        return {"obs": add_noise(rng, conc[None], p[None])[0]}

    return bf.make_simulator([prior, likelihood])


def agreement(n=200, seed=1):
    """Max relative error of each ODE implementation vs cytopk at rtol 1e-12 on this design."""
    rng = np.random.default_rng(seed)
    theta = np.exp(prior_batch(rng, n))
    ref_c, ref_p = ode_cytopk(theta, rtol=1e-12)
    out = {}
    for name, fn in {**ODE, "numba_serial": lambda th: ode_numba(th, False)}.items():
        c, p = fn(theta)
        out[name] = float(max(np.max(np.abs(c / ref_c - 1)), np.max(np.abs(p / ref_p - 1))))
    return out


# ------------------------------------------------------------------ metadata
def meta(extra=None):
    import importlib.metadata as md

    def cmd(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    vers = {}
    for p in ["bayesflow", "keras", "jax", "jaxlib", "jax-cuda13-plugin", "numba", "numpy", "scipy", "joblib", "cytopk"]:
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    cpu = next((l.split(":", 1)[1].strip() for l in (cmd("lscpu") or "").splitlines()
                if l.startswith("Model name")), platform.processor())
    gpu = cmd("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
    return {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": cmd("git", "rev-parse", "HEAD"), "git_dirty": bool(cmd("git", "status", "--porcelain")),
        "cpu": cpu, "logical_cpus": os.cpu_count(), "gpu": gpu, "os": platform.platform(),
        "python": sys.version.split()[0], "rustc": cmd("rustc", "--version"), "versions": vers,
        "loadavg_at_start": os.getloadavg(), **(extra or {}),
    }


def pin_to_p_core():
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {P_CORE})
