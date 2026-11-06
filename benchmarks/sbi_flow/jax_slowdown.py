"""Bisect the in-process simulator slowdown seen in E1 (ODE stage 5.3-5.8x slower than in E2's clean process).

E1 timed the ODE call inside a BayesFlow/JAX process, with prior, noise and adapter work between calls. E2 timed
it back to back in a clean process. Both differences were present at once, so this script separates them.
Every configuration runs in its own subprocess.

Part A/B (`--worker ode`): one noise-free ODE call at B = 2048 (cytopk 16-thread pool or Numba parallel, 16
threads), median of ROUNDS calls after WARM calls. Factors:
  --jax      none | import (import jax) | init (+ jax.devices()) | jit (+ one jitted call)
             | bf (import bayesflow, build the BasicWorkflow adapter and call it, as in E1)
  --pattern  b2b          ODE calls back to back (as in E2)
             noise        the shared NumPy noise stage on the main thread between calls (as in E1)
             sleep_<ms>   time.sleep between calls
             spin_<ms>    busy-wait on the main thread between calls
             interleave   cytopk and Numba calls alternate, nothing else
             e1           the full E1 loop (4 batched simulators, prior + ODE + noise + adapter, round-robin)
  --pin main pins the main thread to CPU 0 after every pool exists. Environment variants (JAX_PLATFORMS,
  OMP_NUM_THREADS, XLA_FLAGS, OMP_WAIT_POLICY, NUMBA_THREADING_LAYER) are set by the parent and recorded.
Part D: noise / sleep / spin gaps with the main thread pinned to CPU 0 or not.
Part C (`--worker freq`): one thread pinned to CPU 0, no pool, cytopk n_threads=1. Back-to-back calls are
compared with sleep and spin gaps of the same length, at several batch sizes. scaling_cur_freq of CPU 0 is read
after every call.

Recorded: os.sched_getaffinity, threads in the process, threadpoolctl.threadpool_info(), the Numba threading
layer, cpufreq governor / EPP / driver, and mean scaling_cur_freq per CPU (sampled by a separate process).

    python jax_slowdown.py --parts ABC                                   # -> results/jax_slowdown.json
    python jax_slowdown.py --parts D --out results/jax_slowdown_pin.json
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

B, ROUNDS, WARM = 2048, 200, 20
CPUFREQ = Path("/sys/devices/system/cpu")


def cpufreq_static():
    def rd(p):
        try:
            return p.read_text().strip()
        except OSError:
            return None
    c0 = CPUFREQ / "cpu0" / "cpufreq"
    idle = []
    for s in sorted((CPUFREQ / "cpu0" / "cpuidle").glob("state*")):
        idle.append({"name": rd(s / "name"), "exit_latency_us": rd(s / "latency"), "disabled": rd(s / "disable")})
    return {"governor": rd(c0 / "scaling_governor"), "epp": rd(c0 / "energy_performance_preference"),
            "driver": rd(c0 / "scaling_driver"), "no_turbo": rd(CPUFREQ / "intel_pstate" / "no_turbo"),
            "cpuinfo_min_khz": rd(c0 / "cpuinfo_min_freq"), "cpuinfo_max_khz": rd(c0 / "cpuinfo_max_freq"),
            "cpuidle_driver": rd(CPUFREQ / "cpuidle" / "current_driver"), "cpu0_idle_states": idle}


def freq_sampler(stop_file, out_file):
    """Own process: mean scaling_cur_freq (MHz) per CPU, sampled every 50 ms until stop_file exists."""
    cpus = sorted(int(p.name[3:]) for p in CPUFREQ.glob("cpu[0-9]*"))
    acc = {c: [] for c in cpus}
    while not os.path.exists(stop_file):
        for c in cpus:
            try:
                acc[c].append(int((CPUFREQ / f"cpu{c}" / "cpufreq" / "scaling_cur_freq").read_text()) / 1e3)
            except OSError:
                pass
        time.sleep(0.05)
    Path(out_file).write_text(json.dumps({c: float(np.mean(v)) for c, v in acc.items() if v}))


def setup_jax(level):
    if level == "none":
        return None
    if level == "bf":
        import bayesflow as bf
        wf = bf.BasicWorkflow(inference_variables=["log_theta"], summary_variables=["obs"])
        wf.adapter(common.batched_simulator("cytopk").sample((32,)))
        return wf.adapter
    import jax
    if level in ("init", "jit"):
        jax.devices()
    if level == "jit":
        import jax.numpy as jnp
        jax.jit(lambda x: jnp.sin(x).sum())(jnp.ones(1000)).block_until_ready()
    return None


def gap(pattern):
    if pattern.startswith("sleep_"):
        time.sleep(float(pattern[6:]) * 1e-3)
    elif pattern.startswith("spin_"):
        t1 = time.perf_counter()
        while time.perf_counter() - t1 < float(pattern[5:]) * 1e-3:
            pass


def ode_worker(args):
    import numba
    adapter = setup_jax(args.jax)
    rng = np.random.default_rng(0)
    theta = np.exp(common.prior_batch(rng, B))

    def cy(th, thr=None):
        return common.ode_cytopk(th, n_threads=thr)

    def nb(th, thr=16):
        numba.set_num_threads(thr)
        return common.ode_numba(th, parallel=True)
    for thr in (None, 16, 8):  # create every pool (cytopk caches one per thread count) before any pinning
        cy(theta, thr)
    nb(theta, 8), nb(theta, 16)
    if args.pin == "main":
        os.sched_setaffinity(0, {common.P_CORE})
    times, gaps = [], []
    if args.pattern == "e1":
        if adapter is None:
            raise SystemExit("pattern e1 needs --jax bf")
        stages, sims = {}, {}
        for name, (kind, thr) in {"cytopk16": ("cytopk", 16), "numba16": ("numba", 16),
                                  "cytopk8": ("cytopk", 8), "numba8": ("numba", 8)}.items():
            stages[name] = []
            ode = (lambda thr: lambda th: cy(th, thr))(thr) if kind == "cytopk" else \
                (lambda thr: lambda th: nb(th, thr))(thr)
            sims[name] = common.batched_simulator(kind, ode=ode, stage_times=stages[name]).sample
        names = list(sims)
        for r in range(WARM + ROUNDS):
            for name in names[r % 4:] + names[: r % 4]:
                adapter(sims[name]((B,)))
        times = [s[1] for s in stages["cytopk16" if args.impl == "cytopk" else "numba16"][WARM:]]
    else:
        main, other = (cy, nb) if args.impl == "cytopk" else (nb, cy)
        for r in range(WARM + ROUNDS):
            t0 = time.perf_counter()
            conc, p = main(theta)
            dt = time.perf_counter() - t0
            if r >= WARM:
                times.append(dt)
            if args.pattern == "noise":
                t1 = time.perf_counter()
                common.add_noise(rng, conc, p)
                gaps.append(time.perf_counter() - t1)
            elif args.pattern == "interleave":
                other(theta)
            else:
                gap(args.pattern)
    import threadpoolctl
    t = np.array(times)
    return {"ode_ms_median": float(np.median(t) * 1e3), "ode_ms_p10": float(np.quantile(t, 0.1) * 1e3),
            "ode_ms_p90": float(np.quantile(t, 0.9) * 1e3), "calls": int(t.size),
            "gap_ms_median": float(np.median(gaps) * 1e3) if gaps else None,
            "affinity": sorted(os.sched_getaffinity(0)), "threads_in_process": len(os.listdir("/proc/self/task")),
            "threadpools": [{k: d.get(k) for k in ("user_api", "internal_api", "num_threads", "prefix")}
                            for d in threadpoolctl.threadpool_info()],
            "numba_threading_layer": numba.threading_layer(), "jax_loaded": "jax" in sys.modules}


def freq_worker(args):
    """Single thread on CPU 0, no pool: call time and CPU 0 scaling_cur_freq for b2b / sleep / spin gaps."""
    os.sched_setaffinity(0, {0})
    th = np.exp(common.prior_batch(np.random.default_rng(0), args.batch))
    fr = open(CPUFREQ / "cpu0" / "cpufreq" / "scaling_cur_freq")
    for _ in range(WARM):
        common.ode_cytopk(th, n_threads=1)
    ts, fs = [], []
    for _ in range(ROUNDS):
        gap(args.pattern)
        t0 = time.perf_counter()
        common.ode_cytopk(th, n_threads=1)
        ts.append(time.perf_counter() - t0)
        fr.seek(0)
        fs.append(int(fr.read()) / 1e3)
    return {"call_ms_median": float(np.median(ts) * 1e3), "call_ms_p10": float(np.quantile(ts, 0.1) * 1e3),
            "call_ms_p90": float(np.quantile(ts, 0.9) * 1e3), "cpu0_freq_mhz_median": float(np.median(fs)),
            "cpu0_freq_mhz_p10": float(np.quantile(fs, 0.1)), "cpu0_freq_mhz_p90": float(np.quantile(fs, 0.9))}


# ------------------------------------------------------------------ configurations
BASE_ENV = {"KERAS_BACKEND": "jax", "JAX_PLATFORMS": "cpu"}
ODE_CONFIGS = []  # (part, label, impl, jax, pattern, pin, extra env)
for impl in ("cytopk", "numba"):
    for jx in ("none", "import", "init", "jit", "bf"):
        ODE_CONFIGS.append(("A", f"b2b, jax={jx}", impl, jx, "b2b", "none", {}))
    ODE_CONFIGS.append(("A", "b2b, jax=init, JAX_PLATFORMS=cuda,cpu", impl, "init", "b2b", "none",
                        {"JAX_PLATFORMS": "cuda,cpu"}))
    for pat in ("noise", "interleave"):
        for jx in ("none", "bf"):
            ODE_CONFIGS.append(("B", f"{pat}, jax={jx}", impl, jx, pat, "none", {}))
    for pat in ("sleep_0.05", "sleep_0.2", "sleep_1", "sleep_3", "spin_3"):
        ODE_CONFIGS.append(("B", f"{pat}, jax=none", impl, "none", pat, "none", {}))
    ODE_CONFIGS.append(("B", "e1, jax=bf (E1 replica)", impl, "bf", "e1", "none", {}))
    for lab, pin, env in [("main thread pinned to CPU 0", "main", {}),
                          ("OMP_NUM_THREADS=1", "none", {"OMP_NUM_THREADS": "1"}),
                          ("XLA_FLAGS single-threaded Eigen", "none",
                           {"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"}),
                          ("OMP_WAIT_POLICY=passive", "none", {"OMP_WAIT_POLICY": "passive"}),
                          ("OMP_WAIT_POLICY=active", "none", {"OMP_WAIT_POLICY": "active"}),
                          ("NUMBA_THREADING_LAYER=workqueue", "none", {"NUMBA_THREADING_LAYER": "workqueue"})]:
        ODE_CONFIGS.append(("B", f"e1, jax=bf, {lab}", impl, "bf", "e1", pin, env))
# Part D: does a pinned, busy main thread keep the pool fast? (same patterns, main thread pinned or not)
for impl in ("cytopk", "numba"):
    for pat in ("noise", "sleep_3", "spin_3"):
        for pin in ("none", "main"):
            ODE_CONFIGS.append(("D", f"{pat}, jax=none, pin={pin}", impl, "none", pat, pin, {}))
FREQ_CONFIGS = [(b, pat) for b in (16, 128, 512, 2048) for pat in ("b2b", "sleep_3", "spin_3")]


def run_sub(argv, env):
    p = subprocess.run([sys.executable, __file__, *argv], capture_output=True, text=True, env=env)
    if p.returncode:
        raise RuntimeError(" ".join(argv) + "\n" + p.stderr[-3000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["ode", "freq"])
    ap.add_argument("--impl", default="cytopk")
    ap.add_argument("--jax", default="none")
    ap.add_argument("--pattern", default="b2b")
    ap.add_argument("--pin", default="none")
    ap.add_argument("--batch", type=int, default=B)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--parts", default="ABCD", help="subset of parts to run, e.g. D")
    ap.add_argument("--sampler", nargs=2)
    ap.add_argument("--out", default=str(HERE / "results" / "jax_slowdown.json"))
    a = ap.parse_args()
    if a.sampler:
        return freq_sampler(*a.sampler)
    if a.worker:
        print(json.dumps(ode_worker(a) if a.worker == "ode" else freq_worker(a)))
        return
    scratch = HERE.parent / "scratch"
    scratch.mkdir(exist_ok=True)
    ode_rows, freq_rows = [], []
    for rep in range(a.repeats):  # whole passes, so drift over time shows up as a difference between repeats
        for part, label, impl, jx, pat, pin, extra in ODE_CONFIGS:
            if part not in a.parts:
                continue
            stop, fout = scratch / "freq.stop", scratch / "freq.json"
            stop.unlink(missing_ok=True)
            sp = subprocess.Popen([sys.executable, __file__, "--sampler", str(stop), str(fout)])
            r = run_sub(["--worker", "ode", "--impl", impl, "--jax", jx, "--pattern", pat, "--pin", pin],
                        {**os.environ, **BASE_ENV, **extra})
            stop.touch()
            sp.wait()
            fr = json.loads(fout.read_text())
            r.update(part=part, label=label, repeat=rep, impl=impl, jax=jx, pattern=pat, pin=pin, env=extra,
                     mean_freq_mhz_p_cores=float(np.mean([v for c, v in fr.items() if int(c) < 12])),
                     mean_freq_mhz_e_cores=float(np.mean([v for c, v in fr.items() if int(c) >= 12])))
            ode_rows.append(r)
            print(f"{impl:6s} {label:52s} rep{rep}  ode {r['ode_ms_median']:6.2f} ms (p10 {r['ode_ms_p10']:.2f} "
                  f"p90 {r['ode_ms_p90']:.2f})  thr {r['threads_in_process']:3d}", flush=True)
        for b, pat in FREQ_CONFIGS if "C" in a.parts else []:
            r = run_sub(["--worker", "freq", "--batch", str(b), "--pattern", pat], {**os.environ, **BASE_ENV})
            r.update(batch=b, pattern=pat, repeat=rep)
            freq_rows.append(r)
            print(f"freq B={b:5d} {pat:8s} rep{rep}  call {r['call_ms_median']:7.3f} ms  cpu0 "
                  f"{r['cpu0_freq_mhz_median']:.0f} MHz", flush=True)
    Path(a.out).write_text(json.dumps({
        "meta": common.meta({"experiment": "JAX slowdown bisection", "B": B, "rounds": ROUNDS, "warm": WARM,
                             "cpufreq": cpufreq_static()}),
        "ode_rows": ode_rows, "freq_rows": freq_rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
