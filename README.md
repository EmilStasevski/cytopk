# cytopk

**High-performance Rust-powered pharmacokinetic/pharmacodynamic (PK/PD) simulation and flow cytometry toolkit for Python.**

`cytopk` is a research prototype designed around two computationally intensive tasks:

1. **Deterministic PK/PD simulation** – efficient intravenous pharmacokinetic simulations for simulation-based inference (SBI).
2. **FCS parsing and processing** – reliable reading and processing of FCS 3.0/3.1 flow cytometry files.

By implementing its computational core in Rust and exposing it through Python bindings, `cytopk` combines high-performance multithreading and memory efficiency with the flexibility of the Python data science ecosystem.

> [!WARNING]
> **Research prototype:** The implemented models are illustrative toy models with example parameter values. They have not been validated for clinical or regulatory applications. The repository contains no real patient data.

## Performance Benchmarks

Benchmarks were conducted on a modern desktop CPU (Intel Core i5-14400F, 16 threads). Results demonstrate the benefits of Rust-based parallel execution compared with established Python alternatives.

| Task | `cytopk` | Best alternative | Speedup |
|:---|---:|---:|---:|
| Read a 1.6 GB FCS file | 64.3 ms | 1.66 s (`flowio`) | **25.8×** |
| Read 500 small FCS files | 0.06 s | 0.23 s (`flowio`) | **4.1×** |
| Peak memory (1.6 GB FCS file) | 3,052 MB | 4,578 MB (`flowio`) | **1.5× less** |
| Nonlinear PK ODE (100,000 subjects) | 1.11M simulations/s | 5,177 simulations/s (JAX + `diffrax`) | **215×** |
| First batched call (no JIT) | 0.0016 s | 1.69 s (Numba) | **~1,050×** |

### Benchmark Limitations

The performance advantages depend on the workload and execution configuration.

- **Numba remains highly competitive:** A line-by-line implementation of the same algorithm in Numba achieves comparable single-threaded ODE performance. Numba is approximately 1.12× faster for the indirect-response model.
- **NumPy excels at linear PK:** A closed-form NumPy implementation outperforms single-threaded `cytopk` by approximately 2.5×.
- **Parallelism drives much of the FCS performance:** On a single thread, the FCS parsing advantage decreases from approximately 25× to 2–4×.

These results suggest that the main advantage of `cytopk` is not simply raw single-threaded execution speed, but the combination of parallel execution, deterministic computation, and low startup overhead.

## Why Rust Instead of Numba?

If single-threaded performance can be comparable, why implement the computational core in Rust?

| Feature | `cytopk` (Rust) | Numba | JAX |
|:---|:---|:---|:---|
| First-call compilation | None | Required | Required |
| JIT warm-up overhead | None | ~2 s | Up to ~20 s |
| Deterministic results across thread counts | Yes | Not guaranteed by default | Configuration-dependent |
| LLVM runtime dependency | No | Yes (`llvmlite`) | Backend-dependent |
| FCS parsing integration | Native | Not a primary use case | Not a primary use case |

The main design advantages are:

1. **No JIT warm-up:** Compiled Rust code executes immediately without runtime compilation.
2. **Deterministic simulations:** Results are bit-identical across different thread counts, supporting reproducible simulation-based inference.
3. **No runtime compiler:** No need to rely on LLVM or `llvmlite` during execution.
4. **Unified architecture:** Numerical simulation and FCS parsing are implemented within one cohesive Rust workspace.

## FCS Parsing and Processing

Flow cytometry files can contain inconsistencies, malformed metadata, and deviations from the official specification. `cytopk` addresses these challenges through two parsing modes:

- **Strict mode:** Rejects files that violate the FCS specification.
- **Lenient mode:** Attempts safe recovery from common formatting and metadata issues, including empty values, delimiter errors, inconsistent offsets, and malformed spillover matrices.

### Compatibility Testing

The parser was evaluated against 30 challenging, publicly available real-world FCS files.

- Successfully extracted data from **28 out of 30 files**.
- Matched or outperformed `fcsparser`, `flowio`, and `flowCore` in the tested compatibility scenarios.
- Identified potential silent data-corruption issues in alternative parsers, including precision loss when converting 32-bit integers to `float32`.

The project also documents these findings and prepares upstream issue reports to improve the reliability of the broader FCS parsing ecosystem.

## Python API

The project is organized into several Rust crates:

- `cytopk-sim` – PK/PD simulation engine.
- `cytopk-fcs` – FCS parsing and data processing.
- `cytopk-gate` – flow cytometry gating functionality.

The components are exposed through a unified Python package using PyO3 bindings.

### 1. PK/PD Simulation

Example: simulating a two-compartment intravenous PK model for 10,000 subjects.

```python
import numpy as np
import cytopk

# N subjects x [CL, V1, Q, V2]
theta = np.tile(
    [0.2, 3.0, 0.5, 3.0],
    (10_000, 1)
)

# Observation times (days)
times = np.array([0.5, 1.0, 7.0, 28.0])

# Simulate a 200 mg infusion over one hour
conc = cytopk.simulate(
    "twocmt_iv",
    theta,
    times,
    dose_time=np.array([0.0]),
    dose_amount=np.array([200.0]),
    dose_duration=np.array([1 / 24])
)

print(conc.shape)  # (10000, 4, 1)
```

### 2. Reading FCS Files

```python
import cytopk

# Read an FCS file using lenient parsing
f = cytopk.read_fcs("sample.fcs")

# Inspect parsing warnings
print(f.warnings)

# Export to Parquet
f.to_parquet("sample.parquet")

# Parallel processing of multiple files
files = cytopk.read_fcs_many(paths)
```

### 3. Gating Strategies

`cytopk` also supports applying gating strategies directly to FCS files.

```python
import cytopk

# Load a gating strategy
strategy = cytopk.GatingStrategy.from_file(
    "strategy.toml"
)

# Read and gate multiple files in parallel
results = cytopk.gate_fcs_many(
    paths,
    strategy
)
```

The gating pipeline supports transformations such as arcsinh and rectangular gating, with processing performed in Rust.

## Validation and BayesFlow Integration

The simulation engine is validated against independent numerical references to ensure numerical correctness.

Validation procedures include:

- **Linear PK models:** Comparison against analytical closed-form solutions across thousands of randomly generated parameter configurations.
- **Nonlinear PK models:** Comparison against a high-accuracy SciPy DOP853 numerical reference.
- **Pharmacokinetic invariants:** Tests covering mass balance and exact dosing invariants.

### End-to-End Simulation-Based Inference

The library was also evaluated in a complete simulation-based inference workflow involving:

1. Simulating a virtual cohort.
2. Generating and writing 2,700 FCS files.
3. Reading and applying gating strategies to the generated files.
4. Performing inference using BayesFlow.

The complete data-preparation stage takes approximately **0.22 seconds**.

In CPU-based BayesFlow training benchmarks, the traditional Python/SciPy simulation pipeline accounts for approximately 51–82% of total training time.

Replacing the simulator with `cytopk` reduces this bottleneck to approximately 3–6.5% of total training time.

This substantially reduces simulation overhead and allows computational resources to be allocated more effectively to the inference stage.

## Project Goals

`cytopk` explores how systems programming and high-performance computing can improve computational workflows in pharmacometrics and flow cytometry.

Its main objectives are:

- Efficient, deterministic PK/PD simulations for large virtual populations.
- Reliable FCS 3.0/3.1 parsing with strict and lenient validation.
- Parallel processing of large collections of flow cytometry files.
- Integration with Python-based scientific computing and SBI frameworks.
- Reproducible numerical results without runtime JIT compilation.

**The broader goal is to provide a fast, memory-efficient, and reproducible computational foundation for research workflows combining pharmacokinetic simulation, flow cytometry, and simulation-based inference.**
