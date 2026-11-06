"""E1: share of BayesFlow training time spent in simulation, per simulator implementation.

    t_sim  = wall time of simulator.sample((B,)) plus the workflow adapter, which converts to float32. This is
             exactly what OnlineDataset does per batch.
    t_step = one training step from fit_offline on pre-simulated data with the same batch size and networks.
             It is timed between on_train_batch_begin and on_train_batch_end, with a sync on the loss.
    share  = t_sim / (t_sim + t_step)

Simulators (twocmt_iv_idr, see common.py): (a) per-sample functions via bf.make_simulator (SciPy solve_ivp per
sample), (b) batched SciPy solve_ivp loop, (c) batched Numba (parallel, nogil), (d) batched cytopk, and
(e) (a) with SequentialSimulator.sample_parallel (joblib, n_jobs=-1).
Networks: BayesFlow defaults, bf.networks.CouplingFlow() and bf.networks.TimeSeriesNetwork(), via BasicWorkflow.

    python e1_share.py sim             # t_sim and offline-dataset build times (CPU)  -> results/e1_sim.json
    python e1_share.py step --device cpu|gpu   # t_step                               -> results/e1_step_<dev>.json
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

BATCHES = [32, 128, 512, 2048]
IMPLS = ["per_sample_make_simulator", "scipy_loop_batched", "numba_parallel_batched", "cytopk_batched",
         "per_sample_sample_parallel"]
SLOW = {"per_sample_make_simulator", "scipy_loop_batched", "per_sample_sample_parallel"}
STEP_COUNT, STEP_WARMUP = 60, 10


def make(impl):
    if impl == "per_sample_make_simulator":
        s = common.per_sample_simulator()
        return s.sample
    if impl == "per_sample_sample_parallel":
        s = common.per_sample_simulator()
        return lambda shape: s.sample_parallel(shape, n_jobs=-1)
    kind = {"scipy_loop_batched": "scipy_loop", "numba_parallel_batched": "numba_parallel",
            "cytopk_batched": "cytopk"}[impl]
    return common.batched_simulator(kind).sample


def adapter():
    import bayesflow as bf
    wf = bf.BasicWorkflow(inference_variables=["log_theta"], summary_variables=["obs"])
    return wf.adapter


def sim_worker(impl):
    sample = make(impl)
    adapt = adapter()
    t0 = time.perf_counter()
    adapt(sample((32,)))
    cold = time.perf_counter() - t0
    rows = []
    for b in BATCHES:
        adapt(sample((b,)))  # warm-up at this size
        ts, ts_adapt, total = [], [], 0.0
        budget, min_calls = (15.0, 3) if impl in SLOW else (2.0, 30)
        while (total < budget or len(ts) < min_calls) and len(ts) < 500:
            t0 = time.perf_counter()
            d = sample((b,))
            t1 = time.perf_counter()
            out = adapt(d)
            t2 = time.perf_counter()
            assert out["summary_variables"].dtype == np.float32
            ts.append(t2 - t0)
            ts_adapt.append(t2 - t1)
            total += t2 - t0
            if impl in SLOW and len(ts) >= min_calls and total > budget:
                break
        rows.append({"impl": impl, "batch": b, "calls": len(ts), "t_sim_s_median": float(np.median(ts)),
                     "t_sim_s_p10": float(np.quantile(ts, 0.1)), "t_sim_s_p90": float(np.quantile(ts, 0.9)),
                     "t_adapter_s_median": float(np.median(ts_adapt))})
    # Offline dataset build (simulator.sample only; the adapter runs per batch during training).
    build = {}
    for n in (100_000, 1_000_000):
        n_meas = min(n, 2000) if impl in SLOW else n
        t0 = time.perf_counter()
        sample((n_meas,))
        dt = time.perf_counter() - t0
        build[str(n)] = {"seconds": dt * n / n_meas, "n_measured": n_meas, "extrapolated": n_meas != n}
    return {"impl": impl, "cold_first_call_s": cold, "rows": rows, "offline_build": build}


class StepTimer:
    def __init__(self):
        import keras

        class CB(keras.callbacks.Callback):
            def __init__(cb):
                super().__init__()
                cb.t = []

            def on_train_batch_begin(cb, batch, logs=None):
                cb.t0 = time.perf_counter()

            def on_train_batch_end(cb, batch, logs=None):
                float(logs["loss"])  # forces the asynchronous JAX computation to finish
                cb.t.append(time.perf_counter() - cb.t0)
        self.cb = CB()


def step_worker(device):
    import bayesflow as bf
    import jax
    rows = []
    sim = common.batched_simulator("cytopk", seed=123)
    for b in BATCHES:
        data = sim.sample((b * STEP_COUNT,))
        wf = bf.BasicWorkflow(inference_network=bf.networks.CouplingFlow(),
                              summary_network=bf.networks.TimeSeriesNetwork(),
                              inference_variables=["log_theta"], summary_variables=["obs"])
        timer = StepTimer()
        wf.fit_offline(data, epochs=1, batch_size=b, callbacks=[timer.cb], verbose=0)
        t = np.array(timer.cb.t)
        steady = t[STEP_WARMUP:]
        rows.append({"batch": b, "steps": int(t.size), "first_step_s": float(t[0]),
                     "t_step_s_median": float(np.median(steady)), "t_step_s_p10": float(np.quantile(steady, 0.1)),
                     "t_step_s_p90": float(np.quantile(steady, 0.9))})
        print(f"[{device}] B={b:5d} first {t[0]:.2f}s  median {np.median(steady) * 1e3:.2f} ms", flush=True)
    return {"device": device, "jax_devices": [str(d) for d in jax.devices()], "rows": rows,
            "networks": "bf.networks.CouplingFlow() + bf.networks.TimeSeriesNetwork() (defaults), BasicWorkflow"
                        " defaults (lr 5e-4, standardize inference_variables)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sim", "step"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--worker")
    args = ap.parse_args()
    if args.worker:
        print(json.dumps(sim_worker(args.worker)))
        return
    out_dir = HERE / "results"
    if args.mode == "step":
        res = step_worker(args.device)
        res["meta"] = common.meta({"experiment": "E1 t_step", "device": args.device})
        (out_dir / f"e1_step_{args.device}.json").write_text(json.dumps(res, indent=2) + "\n")
        return
    env = {**os.environ, "KERAS_BACKEND": "jax", "JAX_PLATFORMS": "cpu"}
    res = []
    for impl in IMPLS:
        p = subprocess.run([sys.executable, __file__, "sim", "--worker", impl], capture_output=True, text=True,
                           env=env)
        if p.returncode:
            raise RuntimeError(p.stderr[-3000:])
        r = json.loads(p.stdout.strip().splitlines()[-1])
        res.append(r)
        print(f"{impl:28s} cold {r['cold_first_call_s']:.2f}s  " + "  ".join(
            f"B{x['batch']}:{x['t_sim_s_median'] * 1e3:.2f}ms" for x in r["rows"]) + "  build 1e6: "
            f"{r['offline_build']['1000000']['seconds']:.1f}s", flush=True)
    (out_dir / "e1_sim.json").write_text(json.dumps(
        {"meta": common.meta({"experiment": "E1 t_sim", "agreement_max_rel_err_vs_tight": common.agreement()}),
         "results": res}, indent=2) + "\n")


if __name__ == "__main__":
    main()
