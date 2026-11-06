# Roadmap

**Done**
- M0: workspace, CI (fmt, clippy, Rust tests, pytest), Apache-2.0 licence, decision log.
- M1: `cytopk-sim` with IV bolus/infusion dosing, `onecmt_iv` and `twocmt_iv`, Dormand–Prince 5(4) + RK4, deterministic parallel batches, closed-form oracles.
- M2: `twocmt_mm_iv` (Michaelis–Menten) and `twocmt_iv_idr` (indirect-response PD), tested against SciPy references, conservation laws and exact limits; exposed as `cytopk.simulate(model, ...)`.
- Benchmarks: linear go/no-go (`quick_bench.json`) and nonlinear accuracy-matched benchmark (`bench_nonlinear.json`).

**Next**
- M3: full Python API (schedule objects, seeded noise models: log-normal PK, beta-binomial gate counts), type stubs, wheels.
- M4: remaining benchmark work (scaling curves over N and threads, peak memory, R tools if wanted, GPU for JAX).
- M5–M7: `cytopk-fcs` FCS 3.0/3.1 reader/writer, transforms, gating, CLI, FCS benchmarks.
- M8: end-to-end example: simulate → write FCS → read and gate → amortized inference with BayesFlow, SBC checks.
- M9: polish and v0.1.0 tag. Nothing is published to crates.io/PyPI without an explicit decision.
