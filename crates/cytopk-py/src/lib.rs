//! Minimal PyO3 bindings: only `simulate_twocmt_iv`, used by the go/no-go benchmark.
//! The full Python API comes in M3 (see docs/ROADMAP.md).

use cytopk_sim::{Dose, Schedule, SimError, SolverOpts, TwoCmtIv, simulate_batch};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::ThreadPool;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Thread pools are cached per thread count. Building a pool per call costs about a millisecond and
/// showed up in the first benchmark run.
fn pool(n: usize) -> PyResult<Arc<ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();
    let mut map = POOLS
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| PyRuntimeError::new_err("thread-pool cache poisoned"))?;
    if let Some(p) = map.get(&n) {
        return Ok(Arc::clone(p));
    }
    let p = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    );
    map.insert(n, Arc::clone(&p));
    Ok(p)
}

fn to_py(e: SimError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Simulate `twocmt_iv` for every row of `theta` (shape N x 4: CL, V1, Q, V2).
///
/// Doses go into the central compartment. `dose_duration[i] == 0` means bolus, otherwise infusion.
/// Returns central concentrations with shape (N, T, 1). `n_threads=None` uses the global Rayon pool.
#[pyfunction]
#[pyo3(signature = (theta, times, dose_time, dose_amount, dose_duration, rtol=1e-6, atol=1e-9, max_steps=100_000, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn simulate_twocmt_iv<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    dose_time: PyReadonlyArray1<'py, f64>,
    dose_amount: PyReadonlyArray1<'py, f64>,
    dose_duration: PyReadonlyArray1<'py, f64>,
    rtol: f64,
    atol: f64,
    max_steps: usize,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let (dt, da, dd) = (
        dose_time.as_array(),
        dose_amount.as_array(),
        dose_duration.as_array(),
    );
    if dt.len() != da.len() || dt.len() != dd.len() {
        return Err(PyValueError::new_err("dose arrays must have equal length"));
    }
    let doses = dt
        .iter()
        .zip(da.iter())
        .zip(dd.iter())
        .map(|((&t, &a), &d)| {
            if d == 0.0 {
                Dose::bolus(t, a, 0)
            } else {
                Dose::infusion(t, a, d, 0)
            }
        })
        .collect();
    let sched = Schedule::new(doses).map_err(to_py)?;
    let times = times.as_array().to_vec();
    let opts = SolverOpts {
        max_steps,
        ..SolverOpts::dopri5(rtol, atol)
    };
    let theta = theta.as_array();
    let pool = match n_threads {
        Some(0) => return Err(PyValueError::new_err("n_threads must be >= 1")),
        Some(n) => Some(pool(n)?),
        None => None,
    };
    let out = py
        .detach(|| {
            let run = || simulate_batch(&TwoCmtIv, theta, &sched, &times, &opts);
            match &pool {
                Some(p) => p.install(run),
                None => run(),
            }
        })
        .map_err(to_py)?;
    Ok(out.into_pyarray(py))
}

/// Native extension module.
#[pymodule]
fn _cytopk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_twocmt_iv, m)?)?;
    Ok(())
}
