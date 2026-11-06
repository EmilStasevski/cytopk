"""E2b: does a multi-threaded simulator call slow down the single-threaded NumPy code that follows it?

Observed while profiling E2: in a BayesFlow LambdaSimulator, the NumPy noise stage (log-normal + beta-binomial)
ran several times slower right after a 16-thread ODE call than on its own. This script measures it. The noise
stage is timed alone and right after each ODE call. The main Python thread is either free (default) or pinned
to one P-core *after* the thread pools exist, so the pool threads keep all CPUs.

    python e2_interference.py   # -> results/e2_interference.json
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

E_CORES = set(range(12, 16))  # i5-14400F (lscpu -e: MAXMHZ 3500)


def cpu_now():
    with open(f"/proc/self/task/{threading.get_native_id()}/stat") as f:
        return int(f.read().split()[38])


def measure(n, before, rng, reps=41):
    th = np.exp(common.prior_batch(rng, n))
    c, p = common.ode_cytopk(th)
    ts, on_e = [], 0
    for i in range(reps):
        if before is not None:
            before(th)
        cpu = cpu_now()
        t0 = time.perf_counter()
        common.add_noise(rng, c, p)
        dt = time.perf_counter() - t0
        if i:  # first repetition is warm-up
            ts.append(dt)
            on_e += cpu in E_CORES
    return {"noise_ms_median": float(np.median(ts) * 1e3), "noise_ms_p90": float(np.quantile(ts, 0.9) * 1e3),
            "main_thread_on_e_core_fraction": on_e / (reps - 1)}


def main():
    rng = np.random.default_rng(0)
    warm = np.exp(common.prior_batch(rng, 64))
    common.ode_cytopk(warm)
    common.ode_numba(warm, True)  # create both thread pools before any pinning
    conds = {
        "noise_alone": None,
        "after_cytopk_16_threads": lambda th: common.ode_cytopk(th),
        "after_numba_parallel_16_threads": lambda th: common.ode_numba(th, True),
        "after_cytopk_1_thread": lambda th: common.ode_cytopk(th, n_threads=1),
    }
    rows = []
    for main_pinned in (False, True):
        if main_pinned:
            os.sched_setaffinity(0, {common.P_CORE})  # pid 0 = this thread only
        for n in (512, 2048, 16384):
            for name, fn in conds.items():
                r = {"main_thread_pinned": main_pinned, "batch": n, "condition": name, **measure(n, fn, rng)}
                rows.append(r)
                print(f"pinned={main_pinned!s:5} n={n:6d} {name:32s} {r['noise_ms_median']:7.3f} ms "
                      f"(E-core {r['main_thread_on_e_core_fraction']:.2f})", flush=True)
    out = HERE / "results" / "e2_interference.json"
    out.write_text(json.dumps({"meta": common.meta({"experiment": "E2b noise-stage interference"}),
                               "rows": rows}, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
