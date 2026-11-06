# cytopk

FCS flow-cytometry I/O and fast intravenous PK/PD simulators for simulation-based inference.
The core is in Rust and the Python bindings are planned.

> **Status: early development (pre-v0.1).** This is a research prototype. The models are toy models with
> illustrative parameter values. They are **not validated for clinical or regulatory use.** The repo
> contains no real patient data.

## Components

| crate | status | purpose |
|---|---|---|
| `cytopk-sim` | in progress | IV bolus/infusion compartmental PK (and later PD) ODE simulator, batch-parallel |
| `cytopk-fcs` | planned | FCS 3.0/3.1 reader/writer, transforms, gating |
| `cytopk-py` | planned | PyO3 bindings (`import cytopk`) |
| `cytopk-cli` | planned | command-line interface |

## Build and test

```sh
cargo test --workspace
```

See `docs/models.md` for the model equations and `docs/decisions.md` for the design log.

## Licence

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.
