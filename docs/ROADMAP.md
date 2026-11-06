# Roadmap

**Done**
- M0: workspace, CI (fmt, clippy, tests), Apache-2.0 licence, decision log.
- M1: `cytopk-sim` with IV bolus/infusion dosing, `onecmt_iv` and `twocmt_iv`, Dormand–Prince 5(4) + RK4, deterministic parallel batches, closed-form oracles and tests.
- Go/no-go: minimal PyO3 binding and a linear-model benchmark (`benchmarks/results/quick_bench.json`).

**Next**
- M2: nonlinear PK (`twocmt_mm_iv`) and indirect-response PD (`twocmt_iv_idr`), with a benchmark on the nonlinear model.
- M3: full Python API (`cytopk.simulate`, schedules, seeded noise models), type stubs, wheels.
- M4: full simulation benchmark protocol (matched achieved accuracy, JAX/diffrax, scaling curves).
- M5–M7: `cytopk-fcs` FCS 3.0/3.1 reader/writer, transforms, gating, CLI, FCS benchmarks.
- M8: end-to-end example: simulate → write FCS → read and gate → amortized inference with BayesFlow, SBC checks.
- M9: polish and v0.1.0 tag. Nothing is published to crates.io/PyPI without an explicit decision.
