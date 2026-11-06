"""E3: drift-diffusion model (Euler-Maruyama, s = 1, dt = 0.001, max 10 s) in four implementations.

RNG and normal sampler per implementation (the model is RNG-bound, so this matters):
  python_pure      random.Random(seed): Mersenne Twister MT19937; random.gauss.
  numpy_vectorized np.random.default_rng(seed): PCG64; Generator.standard_normal (ziggurat). Loops over time
                   steps, vectorised over the still-active trials (compacted every step).
  numba_*          xoshiro256++ per trial, seeded from (seed, i) by the same SplitMix64 mixing as cytopk;
                   Marsaglia polar method. The same algorithm as cytopk, so the draws should be bit-identical.
  cytopk_*         cytopk.simulate_ddm (Rust): xoshiro256++ per trial from (seed, i); Marsaglia polar method.

Correctness gate (runs first, and the script fails before writing anything if a check fails):
  * Per implementation: |p_hat - P_corr| <= 4 SE + 0.1 |P - P_corr|. P is the continuous-time analytic
    probability (1 - exp(-2 v z a)) / (1 - exp(-2 v a)). P_corr applies the Broadie-Glasserman-Kou
    discrete-monitoring correction (boundaries shifted outward by 0.5826 sqrt(dt)). The 10 % term allows for
    the correction being first-order only. Rule fixed before measuring; also in crates/cytopk-sim/tests/ddm.rs.
  * Each implementation vs cytopk: two-proportion |z| < 4 for choice, and a two-sample KS test of RT with p > 1e-3.
  * No timeouts.

    python e3_ddm.py        # -> results/e3_ddm.json
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

DT, MAX_T = 0.001, 10.0
CHECK_THETA = (1.0, 1.5, 0.4, 0.3)  # v, a, z, t0
CHECK_N = {"python_pure": 200_000, "numpy_vectorized": 1_000_000, "numba_serial": 1_000_000,
           "cytopk_default": 1_000_000}
SIZES = [1_000, 10_000, 100_000, 1_000_000]
MAX_MEASURED = {"python_pure": 10_000}  # larger N are extrapolated linearly (and labelled so)
BGK_BETA = 0.5825971579390106


# ------------------------------------------------------------------ implementations
def ddm_python(theta, seed):
    rng = random.Random(seed)
    gauss, sd, max_n = rng.gauss, math.sqrt(DT), round(MAX_T / DT)
    rt = np.empty(len(theta))
    ch = np.empty(len(theta), dtype=np.int8)
    for i, (v, a, z, t0) in enumerate(theta.tolist()):
        x, n, drift, c = z * a, 0, v * DT, -1
        while n < max_n:
            n += 1
            x += drift + sd * gauss(0.0, 1.0)
            if x >= a:
                c = 1
                break
            if x <= 0.0:
                c = 0
                break
        rt[i] = n * DT + t0 if c >= 0 else np.nan
        ch[i] = c
    return rt, ch


def ddm_numpy(theta, seed):
    rng = np.random.default_rng(seed)
    n_tot = theta.shape[0]
    v, a, z, t0 = (theta[:, k].copy() for k in range(4))
    x = z * a
    idx = np.arange(n_tot)
    rt = np.full(n_tot, np.nan)
    ch = np.full(n_tot, -1, dtype=np.int8)
    drift, sd = v * DT, math.sqrt(DT)
    step = 0
    while idx.size and step < round(MAX_T / DT):
        step += 1
        x += drift + sd * rng.standard_normal(idx.size)
        up, lo = x >= a, x <= 0.0
        done = up | lo
        if done.any():
            d = idx[done]
            rt[d] = step * DT + t0[done]
            ch[d] = up[done].astype(np.int8)
            keep = ~done
            idx, x, a, drift, t0 = idx[keep], x[keep], a[keep], drift[keep], t0[keep]
    return rt, ch


_NB = {}


def numba_fns():
    if _NB:
        return _NB
    from numba import njit, prange, uint64

    GOLD, C1, C2 = uint64(0x9E3779B97F4A7C15), uint64(0xBF58476D1CE4E5B9), uint64(0x94D049BB133111EB)
    MIXK = uint64(0xA0761D6478BD642F)

    @njit(inline="always")
    def splitmix(st):
        st = st + GOLD
        z = st
        z = (z ^ (z >> uint64(30))) * C1
        z = (z ^ (z >> uint64(27))) * C2
        return st, z ^ (z >> uint64(31))

    @njit(inline="always")
    def rotl(x, k):
        return (x << uint64(k)) | (x >> uint64(64 - k))

    @njit(inline="always")
    def next_u64(s):
        result = rotl(s[0] + s[3], 23) + s[0]
        t = s[1] << uint64(17)
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl(s[3], 45)
        return result

    @njit(inline="always")
    def next_f64(s):
        return float(next_u64(s) >> uint64(11)) * (1.0 / 9007199254740992.0)

    @njit(nogil=True)
    def trial(seed, i, v, a, z, t0, dt, max_n, s):
        sa, ka = splitmix(uint64(seed))
        sb, kb = splitmix(uint64(i) ^ MIXK)
        key = ka ^ kb
        for j in range(4):
            key, s[j] = splitmix(key)
        has_spare = False
        spare = 0.0
        x = z * a
        drift = v * dt
        sd = math.sqrt(dt)
        n = 0
        while n < max_n:
            n += 1
            if has_spare:
                g = spare
                has_spare = False
            else:
                while True:
                    u = 2.0 * next_f64(s) - 1.0
                    w = 2.0 * next_f64(s) - 1.0
                    q = u * u + w * w
                    if q < 1.0 and q > 0.0:
                        break
                m = math.sqrt(-2.0 * math.log(q) / q)
                spare = w * m
                has_spare = True
                g = u * m
            x += drift + sd * g
            if x >= a:
                return n * dt + t0, 1
            if x <= 0.0:
                return n * dt + t0, 0
        return np.nan, -1

    @njit(nogil=True)
    def serial(theta, seed, dt, max_n, rt, ch):
        s = np.zeros(4, dtype=np.uint64)
        for i in range(theta.shape[0]):
            rt[i], ch[i] = trial(seed, i, theta[i, 0], theta[i, 1], theta[i, 2], theta[i, 3], dt, max_n, s)

    @njit(nogil=True, parallel=True)
    def parallel(theta, seed, dt, max_n, rt, ch):
        for i in prange(theta.shape[0]):
            s = np.zeros(4, dtype=np.uint64)
            rt[i], ch[i] = trial(seed, i, theta[i, 0], theta[i, 1], theta[i, 2], theta[i, 3], dt, max_n, s)

    _NB.update(serial=serial, parallel=parallel)
    return _NB


def ddm_numba(theta, seed, parallel):
    fn = numba_fns()["parallel" if parallel else "serial"]
    theta = np.ascontiguousarray(theta)
    rt = np.empty(theta.shape[0])
    ch = np.empty(theta.shape[0], dtype=np.int8)
    fn(theta, seed, DT, round(MAX_T / DT), rt, ch)
    return rt, ch


def ddm_cytopk(theta, seed, n_threads):
    import cytopk
    return cytopk.simulate_ddm(theta, seed, dt=DT, max_t=MAX_T, n_threads=n_threads)


IMPLS = {  # name: (pinned single thread?, function)
    "python_pure": (True, ddm_python),
    "numpy_vectorized": (True, ddm_numpy),
    "numba_serial": (True, lambda th, s: ddm_numba(th, s, False)),
    "numba_parallel": (False, lambda th, s: ddm_numba(th, s, True)),
    "cytopk_1thread": (True, lambda th, s: ddm_cytopk(th, s, 1)),
    "cytopk_default": (False, lambda th, s: ddm_cytopk(th, s, None)),
}


def prior_theta(n, seed=0):
    r = np.random.default_rng(seed)
    return np.column_stack([r.uniform(-2, 2, n), r.uniform(0.8, 2.0, n), r.uniform(0.3, 0.7, n),
                            r.uniform(0.1, 0.5, n)])


# ------------------------------------------------------------------ correctness gate
def p_upper(v, a, za):
    return (1 - math.exp(-2 * v * za)) / (1 - math.exp(-2 * v * a))


def correctness():
    from scipy.stats import ks_2samp
    v, a, z, t0 = CHECK_THETA
    shift = BGK_BETA * math.sqrt(DT)
    P, Pc = p_upper(v, a, z * a), p_upper(v, a + 2 * shift, z * a + shift)
    tol_extra = 0.1 * abs(P - Pc)
    out = {"theta": CHECK_THETA, "P_continuous": P, "P_corrected": Pc, "rule":
           "|p_hat - P_corr| <= 4 SE + 0.1|P - P_corr|; vs cytopk: |z| < 4 and KS p > 1e-3", "impls": {}}
    res = {}
    for name, n in CHECK_N.items():
        th = np.tile(CHECK_THETA, (n, 1))
        res[name] = IMPLS[name][1](th, 2024)
    ok = True
    rt_ref, ch_ref = res["cytopk_default"]
    for name, (rt, ch) in res.items():
        n = ch.size
        p_hat = float(np.mean(ch == 1))
        se = math.sqrt(Pc * (1 - Pc) / n)
        tol = 4 * se + tol_extra
        timeouts = int(np.sum(ch == -1))
        row = {"n": n, "p_hat": p_hat, "se": se, "tolerance": tol, "abs_dev_from_P_corr": abs(p_hat - Pc),
               "abs_dev_from_P": abs(p_hat - P), "timeouts": timeouts, "pass_analytic": abs(p_hat - Pc) <= tol}
        if name != "cytopk_default":
            p2 = float(np.mean(ch_ref == 1))
            pool = (p_hat * n + p2 * ch_ref.size) / (n + ch_ref.size)
            zstat = (p_hat - p2) / math.sqrt(pool * (1 - pool) * (1 / n + 1 / ch_ref.size))
            ks = ks_2samp(rt, rt_ref)
            row.update({"z_vs_cytopk": zstat, "ks_stat_vs_cytopk": float(ks.statistic),
                        "ks_p_vs_cytopk": float(ks.pvalue), "pass_vs_cytopk": abs(zstat) < 4 and ks.pvalue > 1e-3})
            if name == "numba_serial":
                row["bit_identical_to_cytopk"] = bool(np.array_equal(ch, ch_ref) and
                                                      np.array_equal(rt.view(np.uint64), rt_ref.view(np.uint64)))
        ok &= row["pass_analytic"] and row.get("pass_vs_cytopk", True) and timeouts == 0
        out["impls"][name] = row
        print(f"[check] {name:18s} n={n:8d} p_hat={p_hat:.5f} (P_corr {Pc:.5f}, tol {tol:.5f})"
              + (f" z={row['z_vs_cytopk']:+.2f} KS p={row['ks_p_vs_cytopk']:.3f}" if "z_vs_cytopk" in row else "")
              + (f" bit-identical={row['bit_identical_to_cytopk']}" if "bit_identical_to_cytopk" in row else ""),
              flush=True)
    out["all_pass"] = bool(ok)
    return out


# ------------------------------------------------------------------ timing workers
def worker(name):
    pinned, fn = IMPLS[name]
    if pinned:
        common.pin_to_p_core()
    t0 = time.perf_counter()
    fn(prior_theta(10), 1)
    cold = time.perf_counter() - t0
    rows = []
    for n in SIZES:
        n_meas = min(n, MAX_MEASURED.get(name, n))
        th = prior_theta(n_meas)
        fn(th[: min(1000, n_meas)], 3)  # warm-up
        ts, total = [], 0.0
        while (total < 3.0 or len(ts) < 3) and len(ts) < 50:
            t0 = time.perf_counter()
            fn(th, 7 + len(ts))
            dt = time.perf_counter() - t0
            ts.append(dt)
            total += dt
            if total > 60:
                break
        med = float(np.median(ts))
        rows.append({"impl": name, "N": n, "N_measured": n_meas, "repeats": len(ts),
                     "time_s_median": med * n / n_meas, "sims_per_s": n_meas / med, "extrapolated": n_meas != n,
                     "pinned": pinned})
    return {"impl": name, "cold_first_call_s": cold, "rows": rows}


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        print(json.dumps(worker(sys.argv[2])))
        return
    chk = correctness()
    if not chk["all_pass"]:
        print(json.dumps(chk, indent=2))
        sys.exit("E3 correctness gate FAILED: implementations disagree; no results written")
    res = []
    for name in IMPLS:
        p = subprocess.run([sys.executable, __file__, "--worker", name], capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(p.stderr[-3000:])
        r = json.loads(p.stdout.strip().splitlines()[-1])
        res.append(r)
        print(f"{name:18s} cold {r['cold_first_call_s']:.2f}s  " + "  ".join(
            f"N={x['N']}:{x['sims_per_s']:.0f}/s" for x in r["rows"]), flush=True)
    rng_info = {
        "python_pure": "MT19937 (random.Random), random.gauss",
        "numpy_vectorized": "PCG64 (np.random.default_rng), ziggurat standard_normal",
        "numba_serial": "xoshiro256++ per trial from (seed, i) via SplitMix64; Marsaglia polar",
        "numba_parallel": "same as numba_serial",
        "cytopk_1thread": "xoshiro256++ per trial from (seed, i) via SplitMix64; Marsaglia polar (Rust)",
        "cytopk_default": "same as cytopk_1thread",
    }
    out = HERE / "results" / "e3_ddm.json"
    out.write_text(json.dumps({"meta": common.meta({"experiment": "E3 DDM", "dt": DT, "max_t": MAX_T,
                                                    "prior": "v~U(-2,2), a~U(0.8,2), z~U(0.3,0.7), t0~U(0.1,0.5)",
                                                    "rng": rng_info}),
                               "correctness": chk, "results": res}, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
