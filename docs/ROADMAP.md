# Roadmap

**Done**
- M0: workspace, CI (fmt, clippy, Rust tests, pytest), Apache-2.0 licence, decision log.
- M1: IV bolus/infusion dosing, `onecmt_iv`, `twocmt_iv`, Dormand–Prince 5(4) + RK4, deterministic parallel batches, closed-form oracles.
- M2: `twocmt_mm_iv` and `twocmt_iv_idr`, tested against SciPy references, conservation laws and exact limits; `cytopk.simulate(model, ...)`.
- DDM simulator (`cytopk.simulate_ddm`) with per-trial xoshiro256++ streams.
- Benchmarks: linear go/no-go, nonlinear accuracy-matched, and the SBI-flow study (`benchmarks/sbi_flow/`). Result there: with a
  compiled simulator, Rust ≈ Numba; the simulator matters only with GPU training, and then mostly through NumPy noise + per-call overhead.

**Next**
- M5–M7: `cytopk-fcs` FCS 3.0/3.1 reader/writer, transforms, gating, CLI, FCS benchmarks (recommended next).
- M3 (small): noise models in the compiled batch (log-normal PK, beta-binomial gates); full Python API, stubs, wheels.
- M8: end-to-end example: simulate → write FCS → read and gate → amortized inference with BayesFlow, SBC checks.
- M9: polish and v0.1.0 tag. Nothing is published to crates.io/PyPI without an explicit decision.
