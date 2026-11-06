"""Print the markdown tables of benchmarks/sbi_flow/README.md from the committed results/*.json.

    python make_tables.py > /tmp/tables.md
"""

import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
B = [32, 128, 512, 2048]


def load(n):
    return json.loads((R / n).read_text())


def ms(x):
    return f"{x * 1e3:.3g}" if x * 1e3 < 100 else f"{x * 1e3:,.0f}"


def e1():
    sim = load("e1_sim.json")
    step = {d: {r["batch"]: r for r in load(f"e1_step_{d}.json")["rows"]} for d in ("cpu", "gpu")}
    rows = {}
    for r in sim["fast_interleaved"]["rows"]:
        rows.setdefault(r["impl"], {})[r["batch"]] = r
    for res in sim["results"]:
        for r in res["rows"]:
            rows.setdefault(r["impl"], {})[r["batch"]] = r
    names = [("per_sample_make_simulator", "(a) per-sample `make_simulator` (solve_ivp)"),
             ("scipy_loop_batched", "(b) batched SciPy solve_ivp loop"),
             ("per_sample_sample_parallel", "(e) (a) + `sample_parallel` (joblib, 16 jobs)"),
             ("numba_parallel_batched", "(c) Numba parallel, nogil (16 threads)"),
             ("cytopk_batched", "(d) cytopk (16 threads)"),
             ("numba_parallel_8threads", "(c) Numba parallel (8 threads)"),
             ("cytopk_8threads", "(d) cytopk (8 threads)")]
    out = ["**t_sim** (ms, median; simulator.sample + adapter → float32; CPU):", "",
           "| simulator | " + " | ".join(f"B={b}" for b in B) + " |", "|---|" + "---:|" * len(B)]
    for k, lab in names:
        out.append(f"| {lab} | " + " | ".join(ms(rows[k][b]["t_sim_s_median"]) for b in B) + " |")
    out += ["", "**t_step** (ms, median of steps 11–60; first step = XLA compile, in parentheses):", "",
            "| device | " + " | ".join(f"B={b}" for b in B) + " |", "|---|" + "---:|" * len(B)]
    for d in ("cpu", "gpu"):
        out.append(f"| {d.upper()} | " + " | ".join(
            f"{ms(step[d][b]['t_step_s_median'])} ({step[d][b]['first_step_s']:.1f} s)" for b in B) + " |")
    for d in ("cpu", "gpu"):
        out += ["", f"**Simulation share** t_sim / (t_sim + t_step), training on **{d.upper()}**:", "",
                "| simulator | " + " | ".join(f"B={b}" for b in B) + " |", "|---|" + "---:|" * len(B)]
        for k, lab in names:
            out.append(f"| {lab} | " + " | ".join(
                f"{100 * rows[k][b]['t_sim_s_median'] / (rows[k][b]['t_sim_s_median'] + step[d][b]['t_step_s_median']):.1f} %"
                for b in B) + " |")
    out += ["", "**Where cytopk's t_sim goes** (16 threads, ms, medians):", "",
            "| stage | " + " | ".join(f"B={b}" for b in B) + " |", "|---|" + "---:|" * len(B)]
    for key, lab in [("t_prior_s_median", "prior (NumPy)"), ("t_ode_s_median", "ODE (Rust)"),
                     ("t_noise_s_median", "noise (NumPy log-normal + beta-binomial)"),
                     ("t_adapter_s_median", "BayesFlow adapter")]:
        out.append(f"| {lab} | " + " | ".join(ms(rows["cytopk_batched"][b][key]) for b in B) + " |")
    out.append("| ODE, Numba 16 threads, same rounds | " + " | ".join(
        ms(rows["numba_parallel_batched"][b]["t_ode_s_median"]) for b in B) + " |")
    out += ["", "**cytopk vs Numba** (same process, interleaved): ratio Numba / cytopk:", "",
            "| quantity | " + " | ".join(f"B={b}" for b in B) + " |", "|---|" + "---:|" * len(B)]
    for a, c, lab in [("numba_parallel_batched", "cytopk_batched", "t_sim, 16 threads"),
                      ("numba_parallel_8threads", "cytopk_8threads", "t_sim, 8 threads")]:
        out.append(f"| {lab} | " + " | ".join(
            f"{rows[a][b]['t_sim_s_median'] / rows[c][b]['t_sim_s_median']:.2f}×" for b in B) + " |")
    out.append("| ODE stage only, 16 threads | " + " | ".join(
        f"{rows['numba_parallel_batched'][b]['t_ode_s_median'] / rows['cytopk_batched'][b]['t_ode_s_median']:.2f}×"
        for b in B) + " |")
    build = {**sim["fast_interleaved"]["offline_build"], **{r["impl"]: r["offline_build"] for r in sim["results"]}}
    cold = {**sim["fast_interleaved"]["cold_first_call_s"], **{r["impl"]: r["cold_first_call_s"] for r in sim["results"]}}
    out += ["", "**Offline dataset build** (simulator.sample, seconds) and cold start (first call incl. JIT / joblib "
            "pool start, seconds):", "", "| simulator | 10^5 | 10^6 | cold start |", "|---|---:|---:|---:|"]
    for k, lab in names[:5]:
        f = lambda n: f"{build[k][n]['seconds']:,.1f}" + (" ¹" if build[k][n]["extrapolated"] else "")  # noqa: E731
        out.append(f"| {lab} | {f('100000')} | {f('1000000')} | {cold[k]:.3f} |")
    out.append("")
    out.append("¹ Measured on 2000 simulations and scaled linearly.")
    ag = sim["meta"]["agreement_max_rel_err_vs_tight"]
    out.append("")
    out.append("Agreement of the noise-free ODE outputs with cytopk at rtol 1e-12 (200 draws from the prior, max relative "
               "error): " + ", ".join(f"{k} {v:.1e}" for k, v in ag.items()) + ".")
    return "\n".join(out)


def e2():
    res = {}
    for tag in ("before", "after"):
        for r in load(f"e2_overhead_{tag}.json")["results"]:
            res[(tag, r["impl"])] = {x["batch"]: x for x in r["rows"]}
    sizes = sorted(res[("after", "cytopk_default")])
    out = ["| impl | " + " | ".join(f"{b:,}" for b in sizes) + " |", "|---|" + "---:|" * len(sizes)]

    def lat(x):
        us = x["latency_s_median"] * 1e6
        return f"{us:,.0f}" if us >= 100 else f"{us:.1f}"
    for tag, impl, lab in [("before", "cytopk_1thread", "cytopk 1 thread — **before fix**"),
                           ("after", "cytopk_1thread", "cytopk 1 thread"),
                           ("before", "cytopk_default", "cytopk 16 threads — **before fix**"),
                           ("after", "cytopk_default", "cytopk 16 threads"),
                           ("after", "numba_serial", "Numba serial"), ("after", "numba_parallel", "Numba parallel"),
                           ("after", "bf_cytopk", "BayesFlow LambdaSimulator + cytopk"),
                           ("after", "bf_numba", "BayesFlow LambdaSimulator + Numba")]:
        out.append(f"| {lab} | " + " | ".join(lat(res[(tag, impl)][b]) for b in sizes) + " |")
    out += ["", "Throughput at the largest batch (131,072), sims/s: " + ", ".join(
        f"{impl} {res[('after', impl)][131072]['sims_per_s']:,.0f}"
        for impl in ("cytopk_1thread", "numba_serial", "cytopk_default", "numba_parallel")) + "."]
    inter = load("e2_interference.json")["rows"]
    out += ["", "**E2b noise-stage interference** (NumPy noise stage, ms, median of 40; main thread free vs pinned "
            "to one P-core after the pools exist):", "",
            "| condition | B=512 free | B=512 pinned | B=2048 free | B=2048 pinned | B=16384 free | B=16384 pinned |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    conds = ["noise_alone", "after_cytopk_16_threads", "after_numba_parallel_16_threads", "after_cytopk_1_thread"]
    for c in conds:
        cells = []
        for n in (512, 2048, 16384):
            for pinned in (False, True):
                r = next(x for x in inter if x["condition"] == c and x["batch"] == n and x["main_thread_pinned"] == pinned)
                cells.append(f"{r['noise_ms_median']:.2f}")
        out.append(f"| {c.replace('_', ' ')} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def e3():
    d = load("e3_ddm.json")
    res = {r["impl"]: {x["N"]: x for x in r["rows"]} for r in d["results"]}
    cold = {r["impl"]: r["cold_first_call_s"] for r in d["results"]}
    sizes = [1000, 10_000, 100_000, 1_000_000]
    lab = {"python_pure": "pure Python", "numpy_vectorized": "NumPy vectorised", "numba_serial": "Numba serial",
           "cytopk_1thread": "cytopk 1 thread", "numba_parallel": "Numba parallel (16)",
           "cytopk_default": "cytopk (16 threads)"}
    out = ["| implementation | threads | " + " | ".join(f"N={n:,}" for n in sizes) + " | cold start (s) |",
           "|---|---|" + "---:|" * len(sizes) + "---:|"]
    for k in lab:
        cells = []
        for n in sizes:
            x = res[k][n]
            cells.append(f"{x['sims_per_s']:,.0f}" + (" ¹" if x["extrapolated"] else ""))
        thr = "1 (pinned)" if res[k][sizes[0]]["pinned"] else "16"
        out.append(f"| {lab[k]} | {thr} | " + " | ".join(cells) + f" | {cold[k]:.2f} |")
    out += ["", "¹ pure Python measured up to N = 10,000; the per-simulation rate is constant, so larger N reuse it "
            "(time scaled linearly)."]
    c = d["correctness"]
    out += ["", f"**Correctness gate** (θ = v {c['theta'][0]}, a {c['theta'][1]}, z {c['theta'][2]}, t0 "
            f"{c['theta'][3]}): P (continuous) = {c['P_continuous']:.5f}, P_corr (discrete-monitoring correction) = "
            f"{c['P_corrected']:.5f}.", "",
            "| implementation | n | p̂ | abs(p̂ − P_corr) | tolerance | abs(p̂ − P) | z vs cytopk | KS p vs cytopk |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in c["impls"].items():
        z = f"{r['z_vs_cytopk']:+.2f}" if "z_vs_cytopk" in r else "—"
        ks = f"{r['ks_p_vs_cytopk']:.3f}" if "ks_p_vs_cytopk" in r else "—"
        extra = " (bit-identical to cytopk)" if r.get("bit_identical_to_cytopk") else ""
        out.append(f"| {lab.get(k, k)}{extra} | {r['n']:,} | {r['p_hat']:.5f} | {r['abs_dev_from_P_corr']:.5f} | "
                   f"{r['tolerance']:.5f} | {r['abs_dev_from_P']:.5f} | {z} | {ks} |")
    out += ["", f"All checks pass: **{c['all_pass']}**."]
    return "\n".join(out)


def meta_line():
    m = load("e1_sim.json")["meta"]
    v = m["versions"]
    return (f"{m['cpu']} ({m['logical_cpus']} logical CPUs: 6 P-cores with HT + 4 E-cores), GPU {m['gpu']}; "
            f"{m['os']}; Python {m['python']}; BayesFlow {v['bayesflow']}, Keras {v['keras']}, JAX/jaxlib "
            f"{v['jax']} (+ jax-cuda13-plugin {v['jax-cuda13-plugin']}), Numba {v['numba']}, NumPy {v['numpy']}, "
            f"SciPy {v['scipy']}, joblib {v['joblib']}; {m['rustc']}.")


if __name__ == "__main__":
    print("## META\n" + meta_line() + "\n\n## E1\n" + e1() + "\n\n## E2\n" + e2() + "\n\n## E3\n" + e3())
