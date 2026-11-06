"""Nonlinear-model benchmark: cytopk (Rust) vs Numba, SciPy solve_ivp loop and JAX + diffrax (CPU).

Models: twocmt_mm_iv (Michaelis-Menten) and twocmt_iv_idr (indirect response). Scenario: 4 weekly 1-hour infusions
of 200 mg, 20 output times. There is no closed form, so the NumPy closed-form baseline does not apply.

**Matched achieved accuracy.** The error of every method is measured against a reference solution: cytopk at
rtol=1e-12, atol=1e-15, itself cross-checked against SciPy DOP853 at rtol=1e-12 on a subset (stored in the JSON).
Each method's rtol is tuned on a calibration subset to the loosest value on a 4-per-decade grid
(atol = 1e-3 * rtol) whose max relative error is <= TARGET_ERR. The table reports the error measured on the
timed run.

    python benchmarks/bench_nonlinear.py          # all models; writes benchmarks/results/bench_nonlinear.{json,md}
    python benchmarks/bench_nonlinear.py --smoke  # tiny sizes, nothing written

Protocol: one subprocess per row; single-thread rows are pinned to one P-core before any thread pool exists;
the warm-up/compile call is excluded and reported separately; median of repeats.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE / "scratch"  # git-ignored: reference solutions shared with worker processes
P_CORE = 0  # i5-14400F: CPUs 0-11 are P-core threads, 12-15 E-cores (lscpu -e)

TARGET_ERR = 1e-6
RTOL_GRID = [10 ** (-e / 4) for e in range(16, 49)]  # 1e-4 ... 1e-12
CALIB_N = {"cytopk_rust_dopri5": 2000, "numba_dopri5": 2000, "jax_diffrax_dopri5": 2000,
           "scipy_solve_ivp_rk45": 2000}  # SciPy: tuned on the same subjects it is timed on
SCIPY_MAX_SUBJECTS = 2000
SIZES = [10_000, 100_000]

DOSE_TIME = [0.0, 7.0, 14.0, 21.0]
DOSE_AMT = [200.0] * 4
DOSE_DUR = [1.0 / 24.0] * 4
TIMES = [0.04, 0.5, 1, 2, 4, 7, 7.04, 8, 10, 14, 14.04, 15, 17, 21, 21.04, 22, 24, 28, 31, 35]

MODELS = {
    # name: (median theta, n_states, initial state)
    "twocmt_mm_iv": ([3.0, 10.0, 3.0, 0.5, 3.0], 2, [0.0, 0.0]),            # Vmax, Km, V1, Q, V2
    "twocmt_iv_idr": ([0.2, 3.0, 0.5, 3.0, 0.3, 2.0, 5.0], 3, [0.0, 0.0, 1.0]),
}


def draw_theta(model, n, seed=0):
    import numpy as np
    med = np.log(MODELS[model][0])
    rng = np.random.default_rng(seed)
    return np.exp(med + 0.5 * rng.standard_normal((n, med.size)))


def segments():
    """Smooth pieces between dose events: [(a, b, infusion rate into state 0)]."""
    edges = sorted({0.0, *DOSE_TIME, *(t + d for t, d in zip(DOSE_TIME, DOSE_DUR)), TIMES[-1]})
    edges = [e for e in edges if e <= TIMES[-1]]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        r = sum(m / d for t, m, d in zip(DOSE_TIME, DOSE_AMT, DOSE_DUR) if t <= a < t + d)
        out.append((a, b, r))
    return out


def max_rel_err(y, ref):
    """Max relative error over (subject, output), counting points where |ref| > 1e-6 * max_t |ref|."""
    import numpy as np
    peak = np.abs(ref).max(axis=1, keepdims=True)
    mask = np.abs(ref) > 1e-6 * peak
    return float(np.max(np.abs(y - ref)[mask] / np.abs(ref)[mask]))


# ------------------------------------------------------------------ methods (imported lazily in workers)
def make_rust(model):
    import numpy as np
    import cytopk
    a = [np.array(x) for x in (TIMES, DOSE_TIME, DOSE_AMT, DOSE_DUR)]

    def run(theta, threads, rtol, atol):
        return cytopk.simulate(model, theta, *a, rtol=rtol, atol=atol, max_steps=1_000_000, n_threads=threads)
    return run


def make_numba(model):
    """Same Dormand-Prince design as cytopk-sim (controller, stops at every event and output time)."""
    import numba
    import numpy as np
    from numba import njit, prange
    ns = MODELS[model][1]
    y0_init = np.array(MODELS[model][2])

    if model == "twocmt_mm_iv":
        @njit(inline="always")
        def rhs(y, th, r, dy):
            c1 = y[0] / th[2]
            c2 = y[1] / th[4]
            tr = th[3] * (c1 - c2)
            dy[0] = -th[0] * c1 / (th[1] + c1) - tr + r
            dy[1] = tr

        @njit(inline="always")
        def obs(y, th, out):
            out[0] = y[0] / th[2]
    else:
        @njit(inline="always")
        def rhs(y, th, r, dy):
            c1 = y[0] / th[1]
            c2 = y[1] / th[3]
            tr = th[2] * (c1 - c2)
            dy[0] = -th[0] * c1 - tr + r
            dy[1] = tr
            dy[2] = th[4] * (1.0 + th[5] * c1 / (th[6] + c1)) - th[4] * y[2]

        @njit(inline="always")
        def obs(y, th, out):
            out[0] = y[0] / th[1]
            out[1] = y[2] / (y[2] + 9.0)

    @njit
    def subject(th, y0i, stops, rates, out_idx, rtol, atol, out):
        y = y0i.copy()
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
                obs(y, th, out[j])

    @njit(parallel=True)
    def batch(theta, y0i, stops, rates, out_idx, rtol, atol, out):
        for n in prange(theta.shape[0]):
            subject(theta[n], y0i, stops, rates, out_idx, rtol, atol, out[n])

    times = np.array(TIMES)
    stops = np.unique(np.concatenate([DOSE_TIME, np.add(DOSE_TIME, DOSE_DUR), times]))
    stops = stops[stops <= times[-1]]
    rates = np.array([sum(m / d for t, m, d in zip(DOSE_TIME, DOSE_AMT, DOSE_DUR) if t <= s < t + d)
                      for s in stops])
    out_idx = np.array([TIMES.index(s) if s in TIMES else -1 for s in stops.tolist()], dtype=np.int64)
    n_out = 1 if model == "twocmt_mm_iv" else 2

    def run(theta, threads, rtol, atol):
        numba.set_num_threads(threads)
        out = np.empty((theta.shape[0], times.size, n_out))
        batch(theta, y0_init, stops, rates, out_idx, rtol, atol, out)
        return out
    return run


def make_scipy(model):
    """The typical 'just call SciPy' code: piecewise between dose events, t_eval (dense output) for outputs."""
    import numpy as np
    from scipy.integrate import solve_ivp
    times = np.array(TIMES)
    segs = segments()
    y0 = np.array(MODELS[model][2])

    def run(theta, threads, rtol, atol):
        n_out = 1 if model == "twocmt_mm_iv" else 2
        out = np.empty((theta.shape[0], times.size, n_out))
        for i, th in enumerate(theta):
            if model == "twocmt_mm_iv":
                vmax, km, v1, q, v2 = th

                def f(t, y, r):
                    c1, c2 = y[0] / v1, y[1] / v2
                    tr = q * (c1 - c2)
                    return [-vmax * c1 / (km + c1) - tr + r, tr]
            else:
                cl, v1, q, v2, kout, emax, ec50 = th

                def f(t, y, r):
                    c1, c2 = y[0] / v1, y[1] / v2
                    tr = q * (c1 - c2)
                    return [-cl * c1 - tr + r, tr, kout * (1 + emax * c1 / (ec50 + c1)) - kout * y[2]]
            y = y0.copy()
            for a, b, r in segs:
                mask = (times >= a) & (times <= b) if a == 0.0 else (times > a) & (times <= b)
                te = times[mask]
                k = te.size
                if k == 0 or te[-1] != b:
                    te = np.append(te, b)
                sol = solve_ivp(f, (a, b), y, method="RK45", rtol=rtol, atol=atol, t_eval=te, args=(r,))
                v1_ = th[2] if model == "twocmt_mm_iv" else th[1]
                out[i, mask, 0] = sol.y[0, :k] / v1_
                if n_out == 2:
                    out[i, mask, 1] = sol.y[2, :k] / (sol.y[2, :k] + 9.0)
                y = sol.y[:, -1]
        return out
    return run


def make_jax(model):
    import jax
    jax.config.update("jax_enable_x64", True)
    import diffrax
    import jax.numpy as jnp
    import numpy as np

    if model == "twocmt_mm_iv":
        def f(t, y, args):
            th, r = args
            c1, c2 = y[0] / th[2], y[1] / th[4]
            tr = th[3] * (c1 - c2)
            return jnp.stack([-th[0] * c1 / (th[1] + c1) - tr + r, tr])

        def obs(ys, th):
            return (ys[:, 0] / th[2])[:, None]
    else:
        def f(t, y, args):
            th, r = args
            c1, c2 = y[0] / th[1], y[1] / th[3]
            tr = th[2] * (c1 - c2)
            return jnp.stack([-th[0] * c1 - tr + r, tr, th[4] * (1 + th[5] * c1 / (th[6] + c1)) - th[4] * y[2]])

        def obs(ys, th):
            return jnp.stack([ys[:, 0] / th[1], ys[:, 2] / (ys[:, 2] + 9.0)], axis=1)

    term = diffrax.ODETerm(f)
    times = np.array(TIMES)
    plan = []
    for a, b, r in segments():
        mask = (times >= a) & (times <= b) if a == 0.0 else (times > a) & (times <= b)
        te = times[mask]
        k = te.size
        save = np.append(te, b) if (k == 0 or te[-1] != b) else te
        plan.append((a, b, r, k, jnp.array(save)))
    y0 = jnp.array(MODELS[model][2])

    @jax.jit
    def solve(theta, rtol, atol):
        ctrl = diffrax.PIDController(rtol=rtol, atol=atol)

        def one(th):
            y = y0
            outs = []
            for a, b, r, k, save in plan:
                sol = diffrax.diffeqsolve(term, diffrax.Dopri5(), a, b, None, y, args=(th, r),
                                          saveat=diffrax.SaveAt(ts=save), stepsize_controller=ctrl,
                                          max_steps=100_000)
                if k:
                    outs.append(sol.ys[:k])
                y = sol.ys[-1]
            return obs(jnp.concatenate(outs), th)
        return jax.vmap(one)(theta)

    def run(theta, threads, rtol, atol):
        return np.asarray(solve(jnp.asarray(theta), rtol, atol).block_until_ready())
    return run


FACTORIES = {"cytopk_rust_dopri5": make_rust, "numba_dopri5": make_numba,
             "scipy_solve_ivp_rk45": make_scipy, "jax_diffrax_dopri5": make_jax}


# ------------------------------------------------------------------ worker / orchestration
def pin(threads):
    if threads == 1 and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {P_CORE})


def worker_calibrate(method, model, ref_path):
    import numpy as np
    pin(1)
    ref = np.load(ref_path)
    n = min(CALIB_N[method], ref.shape[0])
    theta, ref = draw_theta(model, n), ref[:n]
    run = FACTORIES[method](model)
    t0 = time.perf_counter()
    run(theta[:2], 1, 1e-6, 1e-9)
    first = time.perf_counter() - t0
    chosen = None
    for rtol in RTOL_GRID:
        err = max_rel_err(run(theta, 1, rtol, rtol * 1e-3), ref)
        if err <= TARGET_ERR:
            chosen = (rtol, err)
            break
    return {"method": method, "model": model, "calib_n": n, "rtol": chosen and chosen[0],
            "atol": chosen and chosen[0] * 1e-3, "calib_err": chosen and chosen[1], "first_call_s": first}


def worker_time(method, model, threads, n, n_measured, repeats, rtol, ref_path):
    import numpy as np
    pin(threads)
    run = FACTORIES[method](model)
    theta = draw_theta(model, n)[:n_measured]
    ref = np.load(ref_path)[:n_measured]
    atol = rtol * 1e-3
    t0 = time.perf_counter()
    # Warm-up at the full shape (JAX recompiles per shape). The SciPy loop is warmed up on a few subjects only.
    run(theta if method != "scipy_solve_ivp_rk45" else theta[:5], threads, rtol, atol)
    warm = time.perf_counter() - t0
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        y = run(theta, threads, rtol, atol)
        ts.append(time.perf_counter() - t0)
    per_sim = float(np.median(ts)) / n_measured
    return {
        "method": method, "model": model, "threads": threads, "N": n, "N_measured": n_measured,
        "repeats": repeats, "rtol": rtol, "atol": atol,
        "time_s_median": per_sim * n, "time_s_min": min(ts) / n_measured * n,
        "time_s_max": max(ts) / n_measured * n, "sims_per_s": 1.0 / per_sim,
        "max_rel_err": max_rel_err(y, ref), "extrapolated": n_measured != n,
        "warmup_call_s": warm, "pinned_to_cpu": P_CORE if threads == 1 else None,
    }


def sub(*args):
    res = subprocess.run([sys.executable, __file__, "--worker", json.dumps(args)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"worker {args} failed:\n{res.stderr[-3000:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def build_reference(model, n):
    """Tight cytopk solution (all threads), cross-checked against SciPy DOP853 on the first 50 subjects."""
    import numpy as np
    from scipy.integrate import solve_ivp
    import cytopk
    theta = draw_theta(model, n)
    a = [np.array(x) for x in (TIMES, DOSE_TIME, DOSE_AMT, DOSE_DUR)]
    ref = cytopk.simulate(model, theta, *a, rtol=1e-12, atol=1e-15, max_steps=10_000_000)
    # Independent check: SciPy DOP853 at tight tolerance, stopping at every event and output time.
    stops = sorted({*TIMES, *DOSE_TIME, *(t + d for t, d in zip(DOSE_TIME, DOSE_DUR))})
    stops = [s for s in stops if s <= TIMES[-1]]
    chk = np.empty((50,) + ref.shape[1:])
    for i, th in enumerate(theta[:50]):
        y = np.array(MODELS[model][2])
        t = 0.0
        for s in stops:
            if s > t:
                r = sum(m / d for t0, m, d in zip(DOSE_TIME, DOSE_AMT, DOSE_DUR) if t0 <= t < t0 + d)
                if model == "twocmt_mm_iv":
                    def f(_t, y, th=th, r=r):
                        c1, c2 = y[0] / th[2], y[1] / th[4]
                        tr = th[3] * (c1 - c2)
                        return [-th[0] * c1 / (th[1] + c1) - tr + r, tr]
                else:
                    def f(_t, y, th=th, r=r):
                        c1, c2 = y[0] / th[1], y[1] / th[3]
                        tr = th[2] * (c1 - c2)
                        return [-th[0] * c1 - tr + r, tr, th[4] * (1 + th[5] * c1 / (th[6] + c1)) - th[4] * y[2]]
                y = solve_ivp(f, (t, s), y, method="DOP853", rtol=1e-12, atol=1e-14).y[:, -1]
                t = s
            if s in TIMES:
                j = TIMES.index(s)
                if model == "twocmt_mm_iv":
                    chk[i, j] = [y[0] / th[2]]
                else:
                    chk[i, j] = [y[0] / th[1], y[2] / (y[2] + 9.0)]
    return ref, max_rel_err(ref[:50], chk)


def meta():
    import numpy as np, scipy, numba
    def cmd(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    try:
        import jax, diffrax
        jv = {"jax": jax.__version__, "diffrax": diffrax.__version__}
    except ImportError:
        jv = {"jax": None, "diffrax": None}
    cpu = next((l.split(":", 1)[1].strip() for l in (cmd("lscpu") or "").splitlines()
                if l.startswith("Model name")), platform.processor())
    return {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": cmd("git", "rev-parse", "HEAD"), "git_dirty": bool(cmd("git", "status", "--porcelain")),
        "cpu": cpu, "logical_cpus": os.cpu_count(), "os": platform.platform(),
        "loadavg_at_start": os.getloadavg(), "python": sys.version.split()[0],
        "numpy": np.__version__, "scipy": scipy.__version__, "numba": numba.__version__, **jv,
        "rustc": cmd("rustc", "--version"),
        "target_max_rel_err": TARGET_ERR, "rtol_grid": "10^(-k/4), k=16..48; atol = 1e-3 * rtol",
        "scenario": "4 x 200 mg 1-h infusions on days 0, 7, 14, 21; 20 output times to day 35",
        "theta": {m: f"log-normal, median {v[0]}, sd 0.5, seed 0" for m, v in MODELS.items()},
        "protocol": "one subprocess per row; single-thread rows pinned to one P-core; warm-up call excluded "
                    "(reported as warmup_call_s); median of repeats; errors vs a tight reference",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--worker")
    args = ap.parse_args()
    if args.worker:
        a = json.loads(args.worker)
        fn = worker_calibrate if a[0] == "calibrate" else worker_time
        print(json.dumps(fn(*a[1:])))
        return
    import numpy as np
    sizes = [300, 1000] if args.smoke else SIZES
    SCRATCH.mkdir(exist_ok=True)
    m = meta()
    have_jax = m["jax"] is not None
    methods = ["cytopk_rust_dopri5", "numba_dopri5", "scipy_solve_ivp_rk45"] + (["jax_diffrax_dopri5"] * have_jax)
    rows, calib, refs = [], [], {}
    for model in args.models:
        ref, ref_check = build_reference(model, max(sizes))
        ref_path = SCRATCH / f"ref_{model}.npy"
        np.save(ref_path, ref)
        refs[model] = {"method": "cytopk rtol=1e-12 atol=1e-15",
                       "max_rel_diff_vs_scipy_dop853_first50": ref_check}
        print(f"[{model}] reference vs SciPy DOP853 (50 subjects): {ref_check:.1e}", flush=True)
        for method in methods:
            c = sub("calibrate", method, model, str(ref_path))
            calib.append(c)
            print(f"[{model}] calibrate {method:22s} rtol={c['rtol']} err={c['calib_err']}", flush=True)
            if c["rtol"] is None:
                continue
            for n in sizes:
                reps = 2 if args.smoke else (5 if n >= 100_000 else 7)
                if method == "scipy_solve_ivp_rk45":
                    configs = [(1, min(n, SCIPY_MAX_SUBJECTS), 1)]
                else:
                    configs = [(1, n, reps), (os.cpu_count(), n, reps)]
                for thr, nm, rp in configs:
                    r = sub("time", method, model, thr, n, nm, rp, c["rtol"], str(ref_path))
                    rows.append(r)
                    print(f"[{model}] {method:22s} thr={thr:>2} N={n:>7} {r['time_s_median']:9.4f}s "
                          f"{r['sims_per_s']:12.0f}/s err={r['max_rel_err']:.2e}", flush=True)
    if args.smoke:
        return
    out = HERE / "results"
    (out / "bench_nonlinear.json").write_text(
        json.dumps({"meta": m, "reference": refs, "calibration": calib, "rows": rows}, indent=2) + "\n")
    lines = []
    for model in args.models:
        lines += [f"**{model}**", "",
                  "| method | threads | N | time (s, median) | sims/s | tuned rtol | achieved max rel. error |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in (r for r in rows if r["model"] == model):
            note = " ¹" if r["extrapolated"] else ""
            lines.append(f"| {r['method']} | {r['threads']} | {r['N']:,} | {r['time_s_median']:.4g}{note} | "
                         f"{r['sims_per_s']:,.0f} | {r['rtol']:.1e} | {r['max_rel_err']:.1e} |")
        lines.append("")
    lines += [f"¹ SciPy timed on the first {SCIPY_MAX_SUBJECTS} subjects (1 run) and scaled linearly to N.",
              f"Each method tuned to achieved max rel. error ≤ {TARGET_ERR:g} on a calibration subset; "
              "warm-up/JIT compile excluded (see `warmup_call_s` in the JSON).",
              f"{m['cpu']}, {m['logical_cpus']} logical CPUs; single-thread rows pinned to CPU {P_CORE}; "
              f"git {str(m['git_sha'])[:7]}" + (" (dirty working tree)" if m["git_dirty"] else "")]
    (out / "bench_nonlinear.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
