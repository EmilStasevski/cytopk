"""Figures for benchmarks/sbi_flow/README.md, drawn only from the committed results/*.json.

Colour follows the entity in every figure (validated with the dataviz palette validator, light mode):
cytopk blue, Numba orange, SciPy/NumPy aqua, plain Python yellow, joblib magenta. Three of these are below 3:1
contrast on the surface, so every line is also direct-labelled and has its own marker shape.

    python plots.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parent / "results"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
    "lines.markersize": 6,
})


def load(name):
    return json.loads((R / name).read_text())


def place_labels(ax, items, min_gap_px=13):
    """Direct end labels that never overlap: sort by height, push apart to a minimum pixel gap, draw a leader line
    when a label had to move."""
    fig = ax.figure
    fig.canvas.draw()
    tr = ax.transData
    pts = sorted(((tr.transform((x, y))[1], x, y, t) for x, y, t in items), key=lambda r: r[0])
    placed = []
    for ypx, x, y, t in pts:
        target = ypx if not placed else max(ypx, placed[-1][0] + min_gap_px)
        placed.append((target, ypx, x, y, t))
    shift = max(0.0, placed[-1][0] - (ax.bbox.y1 - 4)) if placed else 0.0  # keep inside the axes
    for target, ypx, x, y, t in placed:
        dy_pt = (target - shift - ypx) * 72.0 / fig.dpi
        ax.annotate(t, (x, y), xytext=(8, dy_pt), textcoords="offset points", va="center", fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=3)
                    if abs(dy_pt) > 2 else None)


def fig_e1():
    d = load("e1_sim.json")
    sim = {r["impl"]: {x["batch"]: x["t_sim_s_median"] for x in r["rows"]} for r in d["results"]}
    for x in d["fast_interleaved"]["rows"]:
        sim.setdefault(x["impl"], {})[x["batch"]] = x["t_sim_s_median"]
    style = {
        "cytopk_batched": (BLUE, "o", "cytopk (Rust)"),
        "numba_parallel_batched": (ORANGE, "s", "Numba parallel"),
        "scipy_loop_batched": (AQUA, "^", "SciPy loop"),
        "per_sample_make_simulator": (YELLOW, "D", "per-sample make_simulator"),
        "per_sample_sample_parallel": (MAGENTA, "v", "per-sample + joblib"),
    }
    devices = [d for d in ("cpu", "gpu") if (R / f"e1_step_{d}.json").exists()]
    fig, axes = plt.subplots(1, len(devices), figsize=(5.6 * len(devices), 4.2), sharey=True, squeeze=False)
    for ax, dev in zip(axes[0], devices):
        step = {x["batch"]: x["t_step_s_median"] for x in load(f"e1_step_{dev}.json")["rows"]}
        ends = []
        for impl, (c, m, lab) in style.items():
            bs = sorted(sim[impl])
            sh = [100 * sim[impl][b] / (sim[impl][b] + step[b]) for b in bs]
            ax.plot(bs, sh, color=c, marker=m, label=lab)
            ends.append((bs[-1], sh[-1], lab))
        for thr in (5, 20):
            ax.axhline(thr, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
            ax.annotate(f"{thr} % rule", (32, thr), xytext=(2, 3), textcoords="offset points", fontsize=8,
                        color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 128, 512, 2048], ["32", "128", "512", "2048"])
        ax.set_xlim(24, 2048 * 9)
        ax.set_ylim(0, 100)
        ax.set_xlabel("batch size")
        ax.set_title(f"training step on {dev.upper()}", loc="left", fontsize=10, color=INK2)
        place_labels(ax, ends)
    axes[0][0].set_ylabel("simulation share of (sim + train step), %")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle("E1: share of BayesFlow training time spent simulating (twocmt_iv_idr)", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(R / "e1_share.png", dpi=150)


def fig_e2():
    after = {r["impl"]: r["rows"] for r in load("e2_overhead_after.json")["results"]}
    before = {r["impl"]: r["rows"] for r in load("e2_overhead_before.json")["results"]}
    style = {
        "cytopk_default": (BLUE, "o", "-", "cytopk, 16 threads"),
        "cytopk_1thread": (BLUE, "o", ":", "cytopk, 1 thread"),
        "numba_parallel": (ORANGE, "s", "-", "Numba parallel"),
        "numba_serial": (ORANGE, "s", ":", "Numba serial"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ends = []
    for impl, (c, m, ls, lab) in style.items():
        xs = [x["batch"] for x in after[impl]]
        ys = [x["latency_s_median"] * 1e6 / x["batch"] for x in after[impl]]
        ax.plot(xs, ys, color=c, marker=m, linestyle=ls, label=lab)
        ends.append((xs[-1], ys[-1], lab))
    xb = [x["batch"] for x in before["cytopk_1thread"]]
    yb = [x["latency_s_median"] * 1e6 / x["batch"] for x in before["cytopk_1thread"]]
    ax.plot(xb, yb, color=INK2, marker="x", linestyle=(0, (2, 2)), linewidth=1.2, label="cytopk, 1 thread, before fix")
    ax.annotate("cytopk 1 thread\nbefore fix", (xb[0], yb[0]), xytext=(8, 4), textcoords="offset points",
                fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, 131072 * 40)
    ax.set_xlabel("batch size (simulations per call)")
    ax.set_ylabel("µs per simulation (median call latency / batch)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("E2: per-simulation cost vs batch size (twocmt_iv_idr, raw ODE call)", loc="left", fontsize=11)
    fig.tight_layout()
    place_labels(ax, ends)
    fig.savefig(R / "e2_overhead.png", dpi=150)


def fig_e3():
    res = {r["impl"]: r["rows"] for r in load("e3_ddm.json")["results"]}
    panels = {
        "single thread (pinned to one P-core)": {
            "cytopk_1thread": (BLUE, "o", "cytopk (Rust)"), "numba_serial": (ORANGE, "s", "Numba"),
            "numpy_vectorized": (AQUA, "^", "NumPy vectorised"), "python_pure": (YELLOW, "D", "pure Python")},
        "16 threads": {"cytopk_default": (BLUE, "o", "cytopk (Rust)"), "numba_parallel": (ORANGE, "s", "Numba")},
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, st) in zip(axes, panels.items()):
        ends = []
        for impl, (c, m, lab) in st.items():
            rows = res[impl]
            meas = [x for x in rows if not x["extrapolated"]]
            ax.plot([x["N"] for x in meas], [x["sims_per_s"] for x in meas], color=c, marker=m, label=lab)
            ends.append((meas[-1]["N"], meas[-1]["sims_per_s"], lab))
        ax._ends = ends
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(700, 1e6 * 12)
        ax.set_xlabel("simulations per call (N)")
        ax.set_title(title, loc="left", fontsize=10, color=INK2)
    axes[0].set_ylabel("simulations per second")
    fig.suptitle("E3: drift-diffusion model, Euler–Maruyama dt = 1 ms (measured points only)", x=0.01, ha="left",
                 fontsize=11)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ax in axes:
        place_labels(ax, ax._ends)
    fig.savefig(R / "e3_ddm.png", dpi=150)


if __name__ == "__main__":
    fig_e1()
    fig_e2()
    if (R / "e3_ddm.json").exists():
        fig_e3()
