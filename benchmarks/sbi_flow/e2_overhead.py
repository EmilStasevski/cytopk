"""E2: batch-size and overhead sweep of one simulator call.

Implementations (twocmt_iv_idr, noise-free ODE call unless marked):
  cytopk_1thread     cytopk.simulate(..., n_threads=1)       pinned to one P-core
  cytopk_default     cytopk.simulate(..., n_threads=None)    global Rayon pool (16 threads)
  numba_serial       @njit(nogil=True) loop                  pinned to one P-core
  numba_parallel     @njit(nogil=True, parallel=True) prange (16 threads)
  bf_cytopk          BayesFlow LambdaSimulator.sample: prior + cytopk + NumPy noise (16 threads)
  bf_numba           BayesFlow LambdaSimulator.sample: prior + Numba parallel + NumPy noise (16 threads)

Per batch size: 3 warm-up calls, then calls are repeated until >= 0.5 s and >= 5 calls have run (at most 5000).
The median latency is reported.

    python e2_overhead.py --tag before     # -> results/e2_overhead_before.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

SIZES = [1, 8, 32, 128, 512, 2048, 16384, 131072]
IMPLS = {
    "cytopk_1thread": (True, lambda th: common.ode_cytopk(th, n_threads=1)),
    "cytopk_default": (False, lambda th: common.ode_cytopk(th, n_threads=None)),
    "numba_serial": (True, lambda th: common.ode_numba(th, parallel=False)),
    "numba_parallel": (False, lambda th: common.ode_numba(th, parallel=True)),
    "bf_cytopk": (False, None),
    "bf_numba": (False, None),
}


def worker(name):
    pinned, fn = IMPLS[name]
    if pinned:
        common.pin_to_p_core()
    if name.startswith("bf_"):
        sim = common.batched_simulator("cytopk" if name == "bf_cytopk" else "numba_parallel")
        call = lambda n, th: sim.sample((n,))  # noqa: E731
    else:
        call = lambda n, th: fn(th)  # noqa: E731
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    call(2, np.exp(common.prior_batch(rng, 2)))
    cold = time.perf_counter() - t0
    rows = []
    for n in SIZES:
        th = np.exp(common.prior_batch(rng, n))
        for _ in range(3):
            call(n, th)
        ts, total = [], 0.0
        while (total < 0.5 or len(ts) < 5) and len(ts) < 5000:
            t0 = time.perf_counter()
            call(n, th)
            dt = time.perf_counter() - t0
            ts.append(dt)
            total += dt
        med = float(np.median(ts))
        rows.append({"impl": name, "batch": n, "calls": len(ts), "latency_s_median": med,
                     "latency_s_min": float(min(ts)), "sims_per_s": n / med, "pinned": pinned})
    return {"impl": name, "cold_first_call_s": cold, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker)))
        return
    env = {**os.environ, "KERAS_BACKEND": "jax", "JAX_PLATFORMS": "cpu"}
    res = []
    for name in IMPLS:
        p = subprocess.run([sys.executable, __file__, "--worker", name], capture_output=True, text=True, env=env)
        if p.returncode:
            raise RuntimeError(p.stderr[-3000:])
        r = json.loads(p.stdout.strip().splitlines()[-1])
        res.append(r)
        print(f"{name:16s} cold {r['cold_first_call_s']:.3f}s  " + "  ".join(
            f"{x['batch']}:{x['latency_s_median'] * 1e6:.0f}us" for x in r["rows"]), flush=True)
    out = HERE / "results" / f"e2_overhead_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"meta": common.meta({"experiment": "E2", "tag": args.tag}),
                               "results": res}, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
