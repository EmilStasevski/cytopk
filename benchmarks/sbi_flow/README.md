# Where does a Rust simulator matter in a BayesFlow workflow?

Status: **in progress.** The decision rules below were committed *before* any measurement was taken.

## Decision rules (fixed in advance)

1. Rust is **"critical for speed"** in an SBI flow only if both of these hold:
   - with a reasonable baseline implementation, simulation is **at least ~20 % of training time** at realistic batch sizes, **and**
   - cytopk beats the best baseline (Numba, nogil/parallel) by **at least 1.5×** at those batch sizes.
2. If cytopk is only **competitive with Numba**, the case for Rust is not speed. It is determinism, no JIT warm-up,
   no compiler at runtime, no Python dependency, and robust file I/O.
3. If simulation is **under ~5 % of training time**, simulator speed does not matter for this workflow on this hardware.

"Realistic batch sizes" means 32–512, the range of typical BayesFlow training batch sizes. 2048 is reported as an upper end.

## Experiments

- **E1** Simulation share of BayesFlow training (`twocmt_iv_idr`, PK noise + beta-binomial gate counts).
- **E2** Batch-size and overhead sweep of the simulator call (tests the Python binding, not only the solver).
- **E3** A stochastic simulator where Python loops hurt: drift-diffusion model (Euler–Maruyama).
- **E4** (stretch) Writing 10^7 simulations as shards for an on-disk dataset.

Results, tables and verdicts are added below once measured. Every number comes from a committed `results/*.json`.
