"""M8 end-to-end example: simulate a cohort -> FCS files -> read and gate with cytopk -> BayesFlow -> SBC.

Steps (all seeds fixed):
 1. Training set: N_TRAIN subjects via the exact multinomial gate-count path (cohort.simulate_fast).
 2. Test set A ("fcs"): N_TEST subjects whose 9 samples each are written as FCS files (cytopk writer, 10^4
    events x 3 channels), read back and gated by cytopk (gate_fcs_many). Files go to data/m8/ (git-ignored).
 3. Test set B ("fast"): N_TEST further subjects via the fast path, for comparison.
 4. BayesFlow 2.x BasicWorkflow (CouplingFlow + TimeSeriesNetwork, library defaults), fit_offline.
 5. SBC on A and B: N_POST posterior draws per subject; ranks, ECDF and histogram plots, calibration error,
    contraction, prior-normalised RMSE (BayesFlow defaults).

    KERAS_BACKEND=jax .venv-sbi/bin/python examples/cohort/run_m8.py
      -> examples/cohort/results/m8.json and examples/cohort/figures/*.png
    ... run_m8.py --quick   # small smoke run into benchmarks/scratch/m8_quick
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import cohort  # noqa: E402

N_TRAIN, N_VAL, N_TEST, N_POST = 40_000, 1_000, 300, 1_000
EPOCHS, BATCH = 60, 256
DATA = ROOT / "data" / "m8"
OUT, FIG = HERE / "results", HERE / "figures"


def obs_array(s):
    """(n, 9, 2) summaries -> (n, 9, 3) with the sampling time (days, log1p) as the first feature."""
    t = np.broadcast_to(np.log1p(cohort.TIMES)[None, :, None], (*s.shape[:2], 1))
    return np.concatenate([t, s], axis=-1).astype(np.float32)


def fcs_test_set(rng, n):
    DATA.mkdir(parents=True, exist_ok=True)
    log_theta = cohort.prior_batch(rng, n)
    p = cohort.pd_fraction(log_theta)
    q = cohort.sample_q(rng, p)
    paths = []
    t0 = time.perf_counter()
    for i in range(n):
        for j, tj in enumerate(cohort.TIMES):
            f = DATA / f"subject{i:04d}_t{j}.fcs"
            cohort.write_sample(f, rng, q[i, j], subject=i, time=tj)
            paths.append(f)
    t_write = time.perf_counter() - t0
    t0 = time.perf_counter()
    counts = cohort.counts_via_fcs(paths).reshape(n, len(cohort.TIMES), 3)
    t_gate = time.perf_counter() - t0
    size = sum(f.stat().st_size for f in paths)
    return log_theta, q, counts, {"files": len(paths), "bytes": size, "write_s": t_write,
                                  "read_and_gate_s": t_gate}


def sbc(wf, log_theta, obs, name):
    import bayesflow as bf
    import matplotlib
    matplotlib.use("Agg")
    from scipy import stats
    post = wf.sample(conditions={"obs": obs}, num_samples=N_POST)["log_theta"]  # (n, N_POST, 7)
    names = cohort.PARAM_NAMES
    ranks = (post < log_theta[:, None, :]).sum(1)  # 0..N_POST
    bins = 20
    hist = np.stack([np.histogram(ranks[:, k], bins=bins, range=(0, N_POST + 1))[0] for k in range(7)])
    chi2_p = [float(stats.chisquare(h).pvalue) for h in hist]
    ks_p = [float(stats.kstest((ranks[:, k] + 0.5) / (N_POST + 1), "uniform").pvalue) for k in range(7)]
    ce = bf.diagnostics.metrics.calibration_error(post, log_theta, variable_names=names)
    pc = bf.diagnostics.metrics.posterior_contraction(post, log_theta, variable_names=names)
    rmse = bf.diagnostics.metrics.root_mean_squared_error(post, log_theta, variable_names=names)
    FIG.mkdir(exist_ok=True)
    for kind, fn in [("ecdf", bf.diagnostics.plots.calibration_ecdf),
                     ("hist", bf.diagnostics.plots.calibration_histogram),
                     ("recovery", bf.diagnostics.plots.recovery)]:
        # calibration_histogram in BayesFlow 2.1.0 indexes past the last parameter when 7 parameters leave an
        # empty cell in its default grid; one row of 7 avoids that.
        kw = {"difference": True} if kind == "ecdf" else {"num_row": 1, "num_col": 7} if kind == "hist" else {}
        fig = fn(post, log_theta, variable_names=names, **kw)
        fig.savefig(FIG / f"sbc_{kind}_{name}.png", dpi=90, bbox_inches="tight")
    return {"n_subjects": int(log_theta.shape[0]), "n_posterior_draws": N_POST, "rank_bins": bins,
            "rank_chi2_pvalue": dict(zip(names, chi2_p)), "rank_ks_pvalue": dict(zip(names, ks_p)),
            "calibration_error": dict(zip(names, map(float, ce["values"]))),
            "posterior_contraction": dict(zip(names, map(float, pc["values"]))),
            "rmse_normalised": dict(zip(names, map(float, rmse["values"]))),
            "posterior_sd_mean": dict(zip(names, map(float, post.std(1).mean(0)))),
            "prior_sd": cohort.PRIOR_SD}


def main():
    global N_TRAIN, N_VAL, N_TEST, N_POST, EPOCHS, DATA, OUT, FIG
    if "--quick" in sys.argv:  # smoke test of the whole pipeline; writes to benchmarks/scratch, not results/
        N_TRAIN, N_VAL, N_TEST, N_POST, EPOCHS = 2_000, 200, 20, 100, 2
        DATA = ROOT / "data" / "m8_quick"
        OUT = FIG = ROOT / "benchmarks" / "scratch" / "m8_quick"
        OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "benchmarks" / "sbi_flow"))
    import bayesflow as bf
    import common
    import jax
    rng = np.random.default_rng(20261106)
    t0 = time.perf_counter()
    th_tr, s_tr = cohort.simulate_fast(rng, N_TRAIN)
    th_va, s_va = cohort.simulate_fast(rng, N_VAL)
    t_sim = time.perf_counter() - t0
    th_a, q_a, c_a, fcs_info = fcs_test_set(np.random.default_rng(1), N_TEST)
    th_b, s_b = cohort.simulate_fast(np.random.default_rng(2), N_TEST)
    s_a = cohort.summaries(c_a)
    # Same subjects, fast path given the same q: the summaries should agree in distribution (no common noise).
    s_a_fast = cohort.summaries(cohort.counts_fast(np.random.default_rng(3), q_a))
    wf = bf.BasicWorkflow(inference_network=bf.networks.CouplingFlow(),
                          summary_network=bf.networks.TimeSeriesNetwork(),
                          inference_variables=["log_theta"], summary_variables=["obs"])
    t0 = time.perf_counter()
    hist = wf.fit_offline({"log_theta": th_tr.astype(np.float32), "obs": obs_array(s_tr)}, epochs=EPOCHS,
                          batch_size=BATCH,
                          validation_data={"log_theta": th_va.astype(np.float32), "obs": obs_array(s_va)},
                          verbose=0)
    t_fit = time.perf_counter() - t0
    res = {"fcs": sbc(wf, th_a, obs_array(s_a), "fcs"), "fast": sbc(wf, th_b, obs_array(s_b), "fast")}
    import matplotlib.pyplot as plt
    fig = bf.diagnostics.plots.loss(hist)
    fig.savefig(FIG / "loss.png", dpi=90, bbox_inches="tight")
    plt.close("all")
    d = s_a - s_a_fast
    out = {
        "meta": common.meta({"experiment": "M8 end-to-end cohort", "jax_devices": [str(x) for x in jax.devices()],
                             "keras_backend": os.environ.get("KERAS_BACKEND")}),
        "design": {"n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST, "epochs": EPOCHS, "batch_size": BATCH,
                   "n_events": cohort.N_EVENTS, "cofactor": cohort.COFACTOR, "times_days": cohort.TIMES.tolist(),
                   "networks": "bf.networks.CouplingFlow() + bf.networks.TimeSeriesNetwork() (defaults), "
                               "BasicWorkflow defaults", "params": cohort.PARAM_NAMES,
                   "gates": cohort.GATE_NAMES, "strategy_toml": cohort.STRATEGY},
        "timing_s": {"simulate_train_and_val_fast": t_sim, "fit": t_fit, **{f"fcs_{k}": v for k, v in fcs_info.items()
                                                                            if k.endswith("_s")}},
        "fcs_test_set": {k: v for k, v in fcs_info.items() if not k.endswith("_s")},
        "fcs_vs_fast_same_q": {"mean_diff": d.reshape(-1, 2).mean(0).tolist(),
                               "sd_diff": d.reshape(-1, 2).std(0).tolist(),
                               "features": ["logit activated/cd3", "log cd3/cells"]},
        "loss_final": {"train": float(hist.history["loss"][-1]), "val": float(hist.history["val_loss"][-1])},
        "sbc": res,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "m8.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("meta", "design")}, indent=1))


if __name__ == "__main__":
    main()
