"""Markdown tables for examples/cohort/README.md from results/m8.json (never typed by hand)."""

import json
from pathlib import Path

d = json.loads((Path(__file__).resolve().parent / "results" / "m8.json").read_text())
P = d["design"]["params"]
METRICS = [("posterior_contraction", "contraction", 2), ("rmse_normalised", "NRMSE (1 = no better than prior)", 2),
           ("calibration_error", "calibration error", 3), ("rank_chi2_pvalue", f"rank χ² p ({d['sbc']['fcs']['rank_bins']} bins)", 2),
           ("rank_ks_pvalue", "rank KS p", 2)]
for s, title in [("fcs", "Test set through FCS files"), ("fast", "Test set from the fast path")]:
    r = d["sbc"][s]
    print(f"**{title}** ({r['n_subjects']} subjects, {r['n_posterior_draws']} posterior draws each):\n")
    print("| metric | " + " | ".join(P) + " |")
    print("|---|" + "---:|" * len(P))
    for k, name, nd in METRICS:
        print(f"| {name} | " + " | ".join(f"{r[k][p]:.{nd}f}" for p in P) + " |")
    print()
t, f = d["timing_s"], d["fcs_test_set"]
print(f"Training: {d['design']['n_train']:,} subjects, {d['design']['epochs']} epochs, batch {d['design']['batch_size']}, "
      f"fit {t['fit']:.0f} s on {d['meta']['jax_devices'][0]}; final loss train {d['loss_final']['train']:.2f}, "
      f"validation {d['loss_final']['val']:.2f}.")
print(f"FCS test set: {f['files']:,} files, {f['bytes'] / 1e6:,.0f} MB; writing {t['fcs_write_s']:.1f} s; "
      f"reading and gating with gate_fcs_many {t['fcs_read_and_gate_s']:.2f} s.")
m, sd = d["fcs_vs_fast_same_q"]["mean_diff"], d["fcs_vs_fast_same_q"]["sd_diff"]
print(f"FCS vs fast path, same q (FCS minus fast): mean difference {m[0]:+.4f} (logit activated/cd3), {m[1]:+.4f} "
      f"(log cd3/cells); sd of the differences {sd[0]:.3f} and {sd[1]:.3f}.")
low = [(s, k, p, d["sbc"][s][k][p]) for s in ("fcs", "fast") for k in ("rank_chi2_pvalue", "rank_ks_pvalue")
       for p in P if d["sbc"][s][k][p] < 0.05]
print(f"Rank p-values below 0.05: {len(low)} of {2 * 2 * len(P)}: " +
      ", ".join(f"{s} {k.split('_')[1]} {p} {v:.3f}" for s, k, p, v in low) + ".")
