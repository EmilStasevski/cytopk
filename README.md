# cytopk

Deterministic intravenous PK/PD simulators for simulation-based inference, and a robust FCS 3.0/3.1 flow-cytometry
reader and writer. Written in Rust, with Python bindings.

> **Research prototype.** The models are toy models with illustrative parameter values. They are **not validated for
> clinical or regulatory use.** The repository contains no real patient data.

## Status

| milestone | status |
|---|---|
| M0 scaffolding (workspace, CI, licence) | done |
| M1 PK core: `onecmt_iv`, `twocmt_iv`, IV bolus/infusion dosing, Dormand–Prince 5(4) and RK4, parallel batches | done |
| Minimal Python binding (`cytopk.simulate`, `cytopk.models`) and a linear-model go/no-go benchmark | done |
| M2 nonlinear PK (`twocmt_mm_iv`) and indirect-response PD (`twocmt_iv_idr`), exposed in Python, nonlinear benchmark | done |
| Drift-diffusion simulator (`cytopk.simulate_ddm`) and an SBI-flow study with BayesFlow ([`benchmarks/sbi_flow/`](benchmarks/sbi_flow/README.md)) | done |
| M5 FCS reader (`cytopk-fcs`): strict/lenient modes, F/D/I data, `$NEXTDATA`, mmap, Parquet, Python binding, robustness and speed benchmarks ([`benchmarks/fcs/`](benchmarks/fcs/README.md)) | done |
| M3 full Python API; M6 FCS transforms, compensation and gating; end-to-end amortized inference example | planned |

See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Crates and packages

| name | kind | purpose |
|---|---|---|
| `cytopk-sim` | Rust crate | models, dosing schedules, solvers, batch simulation |
| `cytopk-fcs` | Rust crate | FCS reader/writer; feature `arrow` adds Arrow/Parquet output |
| `cytopk-py` | Rust crate (cdylib) | PyO3 bindings, built into the Python package `cytopk` |
| `cytopk` | Python package (`python/`) | `import cytopk` |

Nothing is published to crates.io or PyPI. Build from source.

## Build and test

```sh
# Rust
cargo test --workspace

# Python extension (needs a virtualenv with maturin, numpy, pytest)
python -m venv .venv && .venv/bin/pip install maturin numpy pytest
cd python && VIRTUAL_ENV=$PWD/../.venv ../.venv/bin/maturin develop --release && cd ..
.venv/bin/python -m pytest python/tests
```

## Quickstart

Rust:

```rust
use cytopk_sim::{Dose, Schedule, SolverOpts, TwoCmtIv, simulate_one};

// 200 mg as a 1-hour infusion; theta = [CL (L/day), V1 (L), Q (L/day), V2 (L)]
let sched = Schedule::new(vec![Dose::infusion(0.0, 200.0, 1.0 / 24.0, 0)])?;
let sol = simulate_one(&TwoCmtIv, &[0.2, 3.0, 0.5, 3.0], &sched, &[0.5, 1.0, 7.0, 28.0], &SolverOpts::default())?;
println!("{:?}", sol.outputs); // central concentration, mg/L
```

Python:

```python
import numpy as np, cytopk

theta = np.tile([0.2, 3.0, 0.5, 3.0], (10_000, 1))      # N x [CL, V1, Q, V2]
times = np.array([0.5, 1.0, 7.0, 28.0])                  # days
conc = cytopk.simulate("twocmt_iv", theta, times,
                       dose_time=np.array([0.0]), dose_amount=np.array([200.0]),
                       dose_duration=np.array([1 / 24]))   # 0 = bolus
conc.shape   # (10000, 4, 1)

cytopk.models()["twocmt_iv_idr"]   # {'params': ['CL', 'V1', 'Q', 'V2', 'kout', 'Emax', 'EC50'], ..., 'outputs': ['C', 'p']}
```

FCS (Python):

```python
f = cytopk.read_fcs("sample.fcs")          # lenient (default); strict=True rejects any deviation
f.data                                     # (events, channels) float32 for $DATATYPE F, float64 for D and I
f.names, f.text["$CYT"], f.spillover       # $PnN names, keywords, spillover matrix (or None)
f.warnings                                 # what lenient mode repaired, e.g. [{"kind": "EmptyValue", ...}]
f.to_parquet("sample.parquet")
files = cytopk.read_fcs_many(paths)        # parallel; failed files come back as FcsReadError objects
cytopk.write_fcs("out.fcs", data, ["FSC-A", "SSC-A"])
```

Units are days, mg, L and mg/L. The output at time `t` includes any bolus given at `t`. Results are bit-identical for any
thread count.

## Validation

The linear models are tested against closed-form solutions on 1000 random parameter sets and schedules per model. The nonlinear
models are tested against a tight-tolerance SciPy DOP853 reference, exact limits and conservation laws. Further tests
check mass balance, `AUC = dose/CL`, dosing conventions, property-based invariants, and bit-identical batches across thread counts.
Details and the error figures are in [`docs/models.md`](docs/models.md). The design rationale is in [`docs/decisions.md`](docs/decisions.md).

## Benchmarks

The main benchmark uses the **nonlinear** models, which have no closed form. Script:
[`benchmarks/bench_nonlinear.py`](benchmarks/bench_nonlinear.py). Results:
[`benchmarks/results/bench_nonlinear.json`](benchmarks/results/bench_nonlinear.json). Every number below comes from that file.

The scenario is 4 weekly 1-hour infusions of 200 mg with 20 output times, and per-subject parameters drawn log-normal.
Tools are compared at **matched achieved accuracy**, not equal rtol. Each tool's rtol was tuned on 2000 subjects to the loosest value
with max relative error ≤ 1e-6 against a tight reference solution. The reference agrees with SciPy DOP853 at rtol 1e-12 to
≤ 1.8e-12. The table reports the error measured in the timed run. At N = 100 000 it is slightly above the target
(1.2–1.4e-6), for every tool. The Numba baseline is a line-by-line port of the same Dormand–Prince design. JAX + diffrax runs on CPU
with float64, `jit` + `vmap`.

`twocmt_mm_iv` (Michaelis–Menten elimination):

| method | threads | N | time (s) | sims/s | achieved max rel. error |
|---|---:|---:|---:|---:|---:|
| cytopk (Rust) | 1 | 100,000 | 1.112 | 89,931 | 1.2e-06 |
| cytopk (Rust) | 16 | 100,000 | 0.08997 | 1,111,511 | 1.2e-06 |
| Numba, same algorithm | 1 | 100,000 | 1.126 | 88,832 | 1.2e-06 |
| Numba, same algorithm | 16 | 100,000 | 0.09749 | 1,025,777 | 1.2e-06 |
| JAX + diffrax Dopri5 | 1 | 100,000 | 5.139 | 19,458 | 1.4e-06 |
| JAX + diffrax Dopri5 | 16 | 100,000 | 19.32 | 5,177 | 1.4e-06 |
| SciPy `solve_ivp` RK45, Python loop | 1 | 100,000 | 211.8 ¹ | 472 | 6.9e-07 |

`twocmt_iv_idr` (indirect-response PD):

| method | threads | N | time (s) | sims/s | achieved max rel. error |
|---|---:|---:|---:|---:|---:|
| cytopk (Rust) | 1 | 100,000 | 1.638 | 61,051 | 1.3e-06 |
| cytopk (Rust) | 16 | 100,000 | 0.1329 | 752,626 | 1.3e-06 |
| Numba, same algorithm | 1 | 100,000 | 1.463 | 68,353 | 1.3e-06 |
| Numba, same algorithm | 16 | 100,000 | 0.162 | 617,345 | 1.3e-06 |
| JAX + diffrax Dopri5 | 1 | 100,000 | 16.16 | 6,190 | 1.2e-06 |
| JAX + diffrax Dopri5 | 16 | 100,000 | 47.45 | 2,107 | 1.2e-06 |
| SciPy `solve_ivp` RK45, Python loop | 1 | 100,000 | 250.5 ¹ | 399 | 8.5e-07 |

¹ Timed on 2000 subjects and scaled linearly. The N = 10 000 rows and all tuning details are in
[`bench_nonlinear.md`](benchmarks/results/bench_nonlinear.md) and the JSON. Times exclude JIT warm-up: at N = 100 000 the warm-up call took
2.3–2.8 s for Numba and 9.6–20.7 s for JAX. cytopk needs none. Hardware: Intel Core i5-14400F (6 P-cores + 4 E-cores,
16 threads), Linux, Python 3.13, NumPy 2.5.4, SciPy 1.18.1, Numba 0.68.1, JAX 0.12.0.1, diffrax 0.7.2, Rust 1.99.1.
Single-thread rows are pinned to one P-core.

**Where Rust is worth it, and where it isn't.** Against Numba running the same algorithm, the gap is small.
- **Michaelis–Menten model:** the two are equal on one thread (ratio 1.01), and Rust is 1.08× faster on 16 threads.
- **Indirect-response model:** Numba is *faster* on one thread (1.12× at N = 100 000, 1.07× at N = 10 000), and Rust is 1.22× faster on 16 threads.

So the speed alone is not a strong reason to choose Rust over a well-written Numba kernel. The practical differences are elsewhere:
- **No JIT warm-up.** Numba and JAX need a warm-up call; cytopk needs none.
- **Determinism is guaranteed and tested.** Output is bit-identical for any thread count. A per-subject Numba kernel can have this property too, but cytopk guarantees it in its API and tests it.
- **No compiler at runtime.** cytopk is a compiled library, whereas Numba needs LLVM (llvmlite) at runtime.
- **One core in Rust.** The simulators and the FCS reader share one Rust workspace and one Python package.

The SciPy `solve_ivp` loop is a weak baseline: it pays Python overhead on every right-hand-side call. On one thread cytopk is
153–200× faster than it, which mainly shows that looping over SciPy is a poor fit for simulation-based inference. JAX + diffrax on CPU was slower than both compiled loops here, and slower with 16
threads than with one (`vmap` steps all subjects in lockstep). This is an untuned CPU setup, and GPU was not tested. For linear models,
none of this matters much. A vectorised **NumPy closed form is faster than single-threaded Rust** for `twocmt_iv`
(2.5–2.6× in the go/no-go benchmark,
[`benchmarks/results/quick_bench.json`](benchmarks/results/quick_bench.json)). A closed form does not apply to the nonlinear
models above.

## Does the simulator speed matter in a BayesFlow workflow?

[`benchmarks/sbi_flow/`](benchmarks/sbi_flow/README.md) measures this for `twocmt_iv_idr` with BayesFlow 2.1. The decision rules
were committed before measuring. Measured on this machine:
- **The big step is Python-level → compiled.** A SciPy-loop or per-sample simulator takes 51–82 % of training time with CPU
  training, and 94–99.5 % with GPU training. A compiled batched simulator, Numba or cytopk, takes 3.0–6.5 % with CPU training.
- **Rust vs Numba is not a speed argument.** cytopk's simulation time is 1.05–1.17× lower than Numba's, well short of the 1.5× threshold. On a stochastic
  drift-diffusion model, single-thread throughput is 0.98–1.00× of Numba's, with bit-identical output.
- **With GPU training the simulator matters, but not through the ODE solve.** It takes 36–62 % of training time (16 threads), and the ODE stage is
  only 16–40 % of cytopk's simulation time. The rest is the NumPy noise stage, BayesFlow's adapter and the prior.

On these measurements, cytopk's case is no JIT warm-up (first call 0.002 s vs 1.69 s for Numba), tested
determinism, and no compiler at runtime. It is not raw speed.

The in-process slowdown seen there (an ODE call 5.3× slower inside the BayesFlow process) is **not caused by JAX**.
Idle CPU cores clock down between calls under this machine's power-saving defaults. A busy main thread pinned to
one core avoids it ([details](benchmarks/sbi_flow/README.md#in-process-slowdown-cause)).

## FCS reader

`cytopk-fcs` reads FCS 3.0/3.1 (2.0 and 3.2 in lenient mode), with `$DATATYPE` F/D/I (any whole-byte integer width),
either byte order, `$NEXTDATA` chains, supplemental TEXT and `$SPILLOVER`.
- **Strict mode** rejects anything the specification forbids.
- **Lenient mode** (the default) repairs the deviations catalogued by Bras & van der Velden (2020): empty values,
  delimiter errors, wrong `$TOT`/`$ENDDATA`, HEADER/TEXT offset disagreements, malformed `$SPILLOVER`, and more. It
  reports each repair as a warning. The full list is in [`docs/fcs_notes.md`](docs/fcs_notes.md).
- **No panics:** mutation proptests run in CI, and a cargo-fuzz run is recorded in
  [`benchmarks/results/fcs_fuzz.json`](benchmarks/results/fcs_fuzz.json).

Measured in [`benchmarks/fcs/`](benchmarks/fcs/README.md) against flowio, FlowKit, fcsparser and flowCore:
- **Real public files:** cytopk returns data for 28 of 30. fcsparser (under NumPy 1.x) also reads 28, flowio's
  lenient mode 25 and flowCore 21. The two files cytopk rejects fail in every reader.
- **Agreement:** where readers disagree on values, the three cases traced so far are errors in the other readers.
- **Speed:** on large float32 files cytopk is 15.8–25.8× faster than the fastest other reader, but mostly through a
  16-thread decode. With one thread the advantage is 1.93–4.01×.
- **Where others win:** fcsparser is faster on a 32-bit integer file (11.3 vs 19.6 ms).
- **Memory:** peak memory is about the same as fcsparser's and lower than flowio, FlowKit and flowCore.

## Licence

[Apache License 2.0](LICENSE). Copyright 2026 Emil Stasevski.

## Citation

See [`CITATION.cff`](CITATION.cff).
