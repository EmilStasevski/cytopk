"""Go/no-go benchmark: cytopk (Rust) vs SciPy solve_ivp loop, NumPy closed form, Numba Dormand-Prince.

Model twocmt_iv, 4 weekly 1-hour infusions of 200 mg, 20 output times, rtol=1e-6, atol=1e-9 for every
adaptive solver. Accuracy is the max relative error against the closed form, over points where
C > 1e-6 * Cmax of that subject (same metric as the Rust tests).

    python benchmarks/quick_bench.py            # writes benchmarks/results/quick_bench.{json,csv,md}
    python benchmarks/quick_bench.py --smoke    # tiny sizes, no files written

This is a quick go/no-go check (few repeats), not the full M4 protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numba
import numpy as np
import scipy
from numba import njit, prange
from scipy.integrate import solve_ivp

import cytopk

RTOL, ATOL = 1e-6, 1e-9
DOSE_TIME = np.array([0.0, 7.0, 14.0, 21.0])
DOSE_AMT = np.full(4, 200.0)
DOSE_DUR = np.full(4, 1.0 / 24.0)
TIMES = np.array(
    [0.04, 0.5, 1, 2, 4, 7, 7.04, 8, 10, 14, 14.04, 15, 17, 21, 21.04, 22, 24, 28, 31, 35],
    dtype=np.float64,
)
assert TIMES.size == 20
ALL_THREADS = os.cpu_count() or 1
SCIPY_MAX_SUBJECTS = 2000  # the SciPy loop is timed on a subset and reported per simulation


def draw_theta(n: int, seed: int = 0) -> np.ndarray:
    """CL, V1, Q, V2 log-normal around the brief's illustrative values (sd 0.5 on log scale)."""
    rng = np.random.default_rng(seed)
    centre = np.log([0.2, 3.0, 0.5, 3.0])
    return np.exp(centre + 0.5 * rng.standard_normal((n, 4)))


# ---------------------------------------------------------------- NumPy closed form (vectorised over N)
def numpy_closed_form(theta: np.ndarray) -> np.ndarray:
    cl, v1, q, v2 = (theta[:, i : i + 1] for i in range(4))  # (N, 1)
    k10, k12, k21 = cl / v1, q / v1, q / v2
    s = k10 + k12 + k21
    alpha = 0.5 * (s + np.sqrt((k10 + k12 - k21) ** 2 + 4 * k12 * k21))
    beta = k10 * k21 / alpha
    c_a, c_b = (alpha - k21) / (alpha - beta), (k21 - beta) / (alpha - beta)
    out = np.zeros((theta.shape[0], TIMES.size))
    for td, amt, dur in zip(DOSE_TIME, DOSE_AMT, DOSE_DUR):
        lag = TIMES[None, :] - td  # (1, T)
        on = lag >= 0
        r = amt / dur
        lag_c = np.clip(lag, 0.0, None)
        during = lag_c <= dur
        for c, lam in ((c_a, alpha), (c_b, beta)):
            phi_s = -np.expm1(-lam * lag_c) / lam
            phi_d = -np.expm1(-lam * dur) / lam
            after = phi_d * np.exp(-lam * (lag_c - dur))
            out += np.where(on, r * c * np.where(during, phi_s, after), 0.0)
    return out / v1


# ---------------------------------------------------------------- integration plan shared by SciPy / Numba
def build_plan():
    """Stops = union of dose starts, infusion ends and output times (same as the Rust driver)."""
    ends = DOSE_TIME + DOSE_DUR
    stops = np.unique(np.concatenate([DOSE_TIME, ends, TIMES]))
    stops = stops[stops <= TIMES[-1]]
    rate = np.array(
        [sum(a / d for t0, a, d in zip(DOSE_TIME, DOSE_AMT, DOSE_DUR) if t0 <= s < t0 + d) for s in stops]
    )
    is_out = np.searchsorted(TIMES, stops)  # TIMES are unique here
    out_idx = np.where(np.isin(stops, TIMES), is_out, -1)
    return stops, rate, out_idx


STOPS, RATES, OUT_IDX = build_plan()


# ---------------------------------------------------------------- SciPy solve_ivp loop
def scipy_loop(theta: np.ndarray) -> np.ndarray:
    """Typical 'just call SciPy' code: integrate piecewise between dose events, t_eval for outputs."""
    seg_edges = np.unique(np.concatenate([[0.0], DOSE_TIME, DOSE_TIME + DOSE_DUR, [TIMES[-1]]]))
    out = np.empty((theta.shape[0], TIMES.size))
    for i, (cl, v1, q, v2) in enumerate(theta):
        y = np.zeros(2)
        for a, b in zip(seg_edges[:-1], seg_edges[1:]):
            r = sum(amt / d for t0, amt, d in zip(DOSE_TIME, DOSE_AMT, DOSE_DUR) if t0 <= a < t0 + d)

            def f(t, y, cl=cl, v1=v1, q=q, v2=v2, r=r):
                c1, c2 = y[0] / v1, y[1] / v2
                tr = q * (c1 - c2)
                return [-cl * c1 - tr + r, tr]

            mask = (TIMES >= a) & (TIMES <= b) if a == 0.0 else (TIMES > a) & (TIMES <= b)
            te = TIMES[mask]
            k = te.size
            if k == 0 or te[-1] != b:
                te = np.append(te, b)  # also evaluate at the segment end to carry the state over
            sol = solve_ivp(f, (a, b), y, method="RK45", rtol=RTOL, atol=ATOL, t_eval=te)
            out[i, mask] = sol.y[0, :k] / v1
            y = sol.y[:, -1]
    return out


# ---------------------------------------------------------------- Numba Dormand-Prince (same algorithm as Rust)
@njit(cache=False, fastmath=False)
def _rhs(y, cl, v1, q, v2, r, dy):
    c1 = y[0] / v1
    c2 = y[1] / v2
    tr = q * (c1 - c2)
    dy[0] = -cl * c1 - tr + r
    dy[1] = tr


@njit(cache=False)
def _subject(th, stops, rates, out_idx, rtol, atol, out_row):
    cl, v1, q, v2 = th[0], th[1], th[2], th[3]
    y = np.zeros(2)
    yn = np.zeros(2)
    ys = np.zeros(2)
    k1 = np.zeros(2); k2 = np.zeros(2); k3 = np.zeros(2); k4 = np.zeros(2)
    k5 = np.zeros(2); k6 = np.zeros(2); k7 = np.zeros(2)
    t = 0.0
    r = 0.0
    h = 0.0
    for s in range(stops.size):
        t1 = stops[s]
        if t1 > t:
            if h <= 0.0:
                # Hairer starting step, as in the Rust code
                _rhs(y, cl, v1, q, v2, r, k1)
                d0 = 0.0; d1 = 0.0
                for i in range(2):
                    sc = atol + rtol * abs(y[i])
                    d0 += (y[i] / sc) ** 2; d1 += (k1[i] / sc) ** 2
                d0 = np.sqrt(d0 / 2); d1 = np.sqrt(d1 / 2)
                h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
                for i in range(2):
                    ys[i] = y[i] + h0 * k1[i]
                _rhs(ys, cl, v1, q, v2, r, k2)
                d2 = 0.0
                for i in range(2):
                    sc = atol + rtol * abs(y[i])
                    d2 += ((k2[i] - k1[i]) / sc) ** 2
                d2 = np.sqrt(d2 / 2) / h0
                dm = max(d1, d2)
                h1 = max(1e-6, h0 * 1e-3) if dm <= 1e-15 else (0.01 / dm) ** 0.2
                h = min(100 * h0, h1)
            _rhs(y, cl, v1, q, v2, r, k1)
            last_rej = False
            while t < t1:
                rem = t1 - t
                hh = h
                clamped = hh >= rem * 0.99
                if clamped:
                    hh = rem
                for i in range(2):
                    ys[i] = y[i] + hh * (0.2 * k1[i])
                _rhs(ys, cl, v1, q, v2, r, k2)
                for i in range(2):
                    ys[i] = y[i] + hh * (3 / 40 * k1[i] + 9 / 40 * k2[i])
                _rhs(ys, cl, v1, q, v2, r, k3)
                for i in range(2):
                    ys[i] = y[i] + hh * (44 / 45 * k1[i] - 56 / 15 * k2[i] + 32 / 9 * k3[i])
                _rhs(ys, cl, v1, q, v2, r, k4)
                for i in range(2):
                    ys[i] = y[i] + hh * (19372 / 6561 * k1[i] - 25360 / 2187 * k2[i]
                                         + 64448 / 6561 * k3[i] - 212 / 729 * k4[i])
                _rhs(ys, cl, v1, q, v2, r, k5)
                for i in range(2):
                    ys[i] = y[i] + hh * (9017 / 3168 * k1[i] - 355 / 33 * k2[i] + 46732 / 5247 * k3[i]
                                         + 49 / 176 * k4[i] - 5103 / 18656 * k5[i])
                _rhs(ys, cl, v1, q, v2, r, k6)
                for i in range(2):
                    yn[i] = y[i] + hh * (35 / 384 * k1[i] + 500 / 1113 * k3[i] + 125 / 192 * k4[i]
                                         - 2187 / 6784 * k5[i] + 11 / 84 * k6[i])
                _rhs(yn, cl, v1, q, v2, r, k7)
                err = 0.0
                for i in range(2):
                    e = hh * (71 / 57600 * k1[i] - 71 / 16695 * k3[i] + 71 / 1920 * k4[i]
                              - 17253 / 339200 * k5[i] + 22 / 525 * k6[i] - 1 / 40 * k7[i])
                    sc = atol + rtol * max(abs(y[i]), abs(yn[i]))
                    err += (e / sc) ** 2
                err = np.sqrt(err / 2)
                fac = 10.0 if err == 0.0 else min(10.0, max(0.2, 0.9 * err ** -0.2))
                if err <= 1.0:
                    for i in range(2):
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
            out_row[j] = y[0] / v1


@njit(parallel=True, cache=False)
def numba_batch(theta, stops, rates, out_idx, rtol, atol, out):
    for n in prange(theta.shape[0]):
        _subject(theta[n], stops, rates, out_idx, rtol, atol, out[n])


def numba_run(theta):
    out = np.empty((theta.shape[0], TIMES.size))
    numba_batch(theta, STOPS, RATES, OUT_IDX, RTOL, ATOL, out)
    return out


# ---------------------------------------------------------------- Rust
def rust_run(theta, n_threads):
    y = cytopk.simulate_twocmt_iv(theta, TIMES, DOSE_TIME, DOSE_AMT, DOSE_DUR, rtol=RTOL, atol=ATOL,
                                  n_threads=n_threads)
    return y[:, :, 0]


# ---------------------------------------------------------------- harness
P_CORE = 0  # i5-14400F: CPUs 0-11 are P-core threads, 12-15 E-cores (lscpu -e); single-thread runs pin here
METHODS = {
    "numpy_closed_form": lambda th, thr: numpy_closed_form(th),
    "cytopk_rust_dopri5": lambda th, thr: rust_run(th, thr),
    "numba_dopri5": lambda th, thr: numba_run(th),
    "scipy_solve_ivp_rk45": lambda th, thr: scipy_loop(th),
}


def max_rel_err(y, ref):
    peak = ref.max(axis=1, keepdims=True)
    mask = ref > 1e-6 * peak
    return float(np.max(np.abs(y - ref)[mask] / ref[mask]))


def worker(method: str, threads: int, n: int, n_measured: int, repeats: int) -> dict:
    """Runs one configuration in a fresh process. Pins first, so that every thread created later
    (Rayon pool, Numba workers) inherits the affinity."""
    if threads == 1 and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {P_CORE})
    extra = {}
    if method == "numba_dopri5":
        numba.set_num_threads(threads)
        t0 = time.perf_counter()
        numba_run(draw_theta(2))
        extra["compile_s"] = time.perf_counter() - t0
    fn = METHODS[method]
    theta = draw_theta(n)[:n_measured]
    ref = numpy_closed_form(theta)
    fn(theta[: min(200, n_measured)], threads)  # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        y = fn(theta, threads)
        ts.append(time.perf_counter() - t0)
    per_sim = float(np.median(ts)) / n_measured
    return {
        "method": method, "threads": threads, "N": n, "N_measured": n_measured, "repeats": repeats,
        "time_s_median": per_sim * n, "time_s_min": min(ts) / n_measured * n,
        "time_s_max": max(ts) / n_measured * n, "sims_per_s": 1.0 / per_sim,
        "max_rel_err": max_rel_err(y, ref), "extrapolated": n_measured != n,
        "pinned_to_cpu": P_CORE if threads == 1 else None, **extra,
    }


def run_config(method, threads, n, n_measured, repeats):
    cmd = [sys.executable, __file__, "--worker", method, str(threads), str(n), str(n_measured), str(repeats)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    row = json.loads(res.stdout.strip().splitlines()[-1])
    print(f"{method:22s} thr={threads:>2} N={n:>7} time={row['time_s_median']:9.4f}s "
          f"sims/s={row['sims_per_s']:12.0f} err={row['max_rel_err']:.2e}"
          + (f" (measured on {n_measured})" if row["extrapolated"] else ""), flush=True)
    return row


def meta():
    def cmd(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    cpu = next((l.split(":", 1)[1].strip() for l in (cmd("lscpu") or "").splitlines()
                if l.startswith("Model name")), platform.processor())
    return {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": cmd("git", "rev-parse", "HEAD"),
        "git_dirty": bool(cmd("git", "status", "--porcelain")),
        "cpu": cpu, "logical_cpus": os.cpu_count(), "os": platform.platform(),
        "loadavg_at_start": os.getloadavg(),
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "numba": numba.__version__, "rustc": cmd("rustc", "--version"),
        "rtol": RTOL, "atol": ATOL, "n_times": int(TIMES.size),
        "doses": "4 x 200 mg, 1 h infusion, weekly (day 0, 7, 14, 21)", "model": "twocmt_iv",
        "theta": "log-normal, median (CL, V1, Q, V2) = (0.2, 3, 0.5, 3), sd 0.5, seed 0",
        "protocol": "one subprocess per row; single-thread rows pinned to one P-core; "
                    "warm-up call excluded; median of repeats",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker", nargs=5)
    args = ap.parse_args()
    if args.worker:
        method, thr, n, nm, rep = args.worker
        print(json.dumps(worker(method, int(thr), int(n), int(nm), int(rep))))
        return
    sizes = [200, 1000] if args.smoke else [10_000, 100_000]
    m = meta()
    rows = []
    for n in sizes:
        reps = 2 if args.smoke else (5 if n >= 100_000 else 9)
        rows.append(run_config("numpy_closed_form", 1, n, n, reps))
        for thr in (1, ALL_THREADS):
            rows.append(run_config("cytopk_rust_dopri5", thr, n, n, reps))
        for thr in (1, ALL_THREADS):
            rows.append(run_config("numba_dopri5", thr, n, n, reps))
        rows.append(run_config("scipy_solve_ivp_rk45", 1, n, min(n, SCIPY_MAX_SUBJECTS), 1))
    if args.smoke:
        return
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "quick_bench.json").write_text(json.dumps({"meta": m, "rows": rows}, indent=2) + "\n")
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(out / "quick_bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    lines = [
        "| method | threads | N | time (s, median) | sims/s | max rel. error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        note = " ¹" if r["extrapolated"] else ""
        err = "reference" if r["method"] == "numpy_closed_form" else f"{r['max_rel_err']:.1e}"
        lines.append(f"| {r['method']} | {r['threads']} | {r['N']:,} | {r['time_s_median']:.4g}{note} | "
                     f"{r['sims_per_s']:,.0f} | {err} |")
    comp = [r.get("compile_s") for r in rows if r["method"] == "numba_dopri5" and r.get("compile_s")]
    lines += [
        "",
        f"¹ SciPy timed on the first {SCIPY_MAX_SUBJECTS} subjects (1 run) and scaled linearly to N.",
        f"Numba JIT compile time (excluded from the table): {min(comp):.2f}–{max(comp):.2f} s per process.",
        f"rtol={RTOL}, atol={ATOL}; {m['cpu']}, {m['logical_cpus']} logical CPUs; "
        f"single-thread rows pinned to CPU {P_CORE}; git {str(m['git_sha'])[:7]}"
        + (" (dirty working tree)" if m["git_dirty"] else ""),
    ]
    (out / "quick_bench.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
