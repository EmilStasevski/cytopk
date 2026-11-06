# cytopk

Fast, deterministic intravenous PK/PD simulators for simulation-based inference, written in Rust with Python bindings.
FCS flow-cytometry I/O is planned.

> **Research prototype.** The models are toy models with illustrative parameter values. They are **not validated for
> clinical or regulatory use.** The repository contains no real patient data.

## Status

| milestone | status |
|---|---|
| M0 scaffolding (workspace, CI, licence) | done |
| M1 PK core: `onecmt_iv`, `twocmt_iv`, IV bolus/infusion dosing, Dormand–Prince 5(4) and RK4, parallel batches | done |
| Minimal Python binding (`cytopk.simulate_twocmt_iv`) and a linear-model go/no-go benchmark | done |
| M2 nonlinear PK (`twocmt_mm_iv`) and indirect-response PD (`twocmt_iv_idr`) | planned |
| M3+ full Python API, FCS reader/writer, end-to-end amortized inference example | planned |

See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Crates and packages

| name | kind | purpose |
|---|---|---|
| `cytopk-sim` | Rust crate | models, dosing schedules, solvers, batch simulation |
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
conc = cytopk.simulate_twocmt_iv(theta, times,
                                 dose_time=np.array([0.0]), dose_amount=np.array([200.0]),
                                 dose_duration=np.array([1 / 24]))   # 0 = bolus
conc.shape   # (10000, 4, 1)
```

Units are days, mg, L and mg/L. The output at time `t` includes any bolus given at `t`. Results are bit-identical for any
thread count.

## Validation

The solver is tested against closed-form solutions on 1000 random parameter sets and schedules per model. Further tests
check mass balance, `AUC = dose/CL`, dosing conventions, property-based invariants, and bit-identical batches across thread counts.
Details and the error figures are in [`docs/models.md`](docs/models.md). The design rationale is in [`docs/decisions.md`](docs/decisions.md).

## Benchmarks

There is so far only a go/no-go benchmark on the *linear* two-compartment model. Script: [`benchmarks/quick_bench.py`](benchmarks/quick_bench.py).
Results: [`benchmarks/results/quick_bench.json`](benchmarks/results/quick_bench.json), with the table in
[`quick_bench.md`](benchmarks/results/quick_bench.md). A benchmark on the nonlinear models will follow with M2.

## Licence

[Apache License 2.0](LICENSE). Copyright 2026 Emil Stasevski.

## Citation

See [`CITATION.cff`](CITATION.cff).
