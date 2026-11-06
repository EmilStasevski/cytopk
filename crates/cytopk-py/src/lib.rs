//! PyO3 bindings: `simulate` for every built-in model, and `models()` to list them.
//! The fuller Python API (schedules, noise models, type stubs) follows in M3 (see docs/ROADMAP.md).

use cytopk_sim::{
    Dose, Model, OneCmtIv, Schedule, SimError, SolverOpts, TwoCmtIv, TwoCmtIvIdr, TwoCmtMmIv,
    simulate_batch,
};
use ndarray::{Array3, ArrayView2};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::ThreadPool;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

fn to_py(e: SimError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

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

const MODELS: [&dyn Model; 4] = [&OneCmtIv, &TwoCmtIv, &TwoCmtMmIv, &TwoCmtIvIdr];

fn run_model(
    model: &str,
    theta: ArrayView2<'_, f64>,
    sched: &Schedule,
    times: &[f64],
    opts: &SolverOpts,
) -> Option<Result<Array3<f64>, SimError>> {
    // Static dispatch: each arm gets its own monomorphised solver.
    Some(match model {
        "onecmt_iv" => simulate_batch(&OneCmtIv, theta, sched, times, opts),
        "twocmt_iv" => simulate_batch(&TwoCmtIv, theta, sched, times, opts),
        "twocmt_mm_iv" => simulate_batch(&TwoCmtMmIv, theta, sched, times, opts),
        "twocmt_iv_idr" => simulate_batch(&TwoCmtIvIdr, theta, sched, times, opts),
        _ => return None,
    })
}

/// Simulate `model` for every row of `theta` (shape N x n_params) and return outputs (N, T, n_outputs).
///
/// Doses: `dose_duration[i] == 0` means bolus, otherwise a zero-order infusion. `dose_state` (default: all 0,
/// the central compartment) selects the receiving state. `n_threads=None` uses the global Rayon pool.
/// Results are bit-identical for any `n_threads`. Errors (bad shapes, parameters, times, step budget) raise
/// `ValueError`.
#[pyfunction]
#[pyo3(signature = (model, theta, times, dose_time, dose_amount, dose_duration, dose_state=None,
                    rtol=1e-6, atol=1e-9, max_steps=100_000, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn simulate<'py>(
    py: Python<'py>,
    model: &str,
    theta: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    dose_time: PyReadonlyArray1<'py, f64>,
    dose_amount: PyReadonlyArray1<'py, f64>,
    dose_duration: PyReadonlyArray1<'py, f64>,
    dose_state: Option<PyReadonlyArray1<'py, i64>>,
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
    let states: Vec<i64> = match &dose_state {
        Some(s) => s.as_array().to_vec(),
        None => vec![0; dt.len()],
    };
    if dt.len() != da.len() || dt.len() != dd.len() || dt.len() != states.len() {
        return Err(PyValueError::new_err("dose arrays must have equal length"));
    }
    let mut doses = Vec::with_capacity(dt.len());
    for i in 0..dt.len() {
        let state = usize::try_from(states[i])
            .map_err(|_| PyValueError::new_err("dose_state must be >= 0"))?;
        doses.push(if dd[i] == 0.0 {
            Dose::bolus(dt[i], da[i], state)
        } else {
            Dose::infusion(dt[i], da[i], dd[i], state)
        });
    }
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
    let out = py.detach(|| {
        let run = || run_model(model, theta, &sched, &times, &opts);
        match &pool {
            Some(p) => p.install(run),
            None => run(),
        }
    });
    match out {
        None => Err(PyValueError::new_err(format!(
            "unknown model `{model}`; available: {}",
            MODELS.map(|m| m.name()).join(", ")
        ))),
        Some(r) => Ok(r.map_err(to_py)?.into_pyarray(py)),
    }
}

/// Built-in models: `{name: {"params": [...], "states": [...], "outputs": [...]}}`.
#[pyfunction]
fn models(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    for m in MODELS {
        let info = PyDict::new(py);
        info.set_item("params", m.param_names().to_vec())?;
        info.set_item("states", m.state_names().to_vec())?;
        info.set_item("outputs", m.output_names().to_vec())?;
        d.set_item(m.name(), info)?;
    }
    Ok(d)
}

/// Native extension module.
#[pymodule]
fn _cytopk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(models, m)?)?;
    Ok(())
}
