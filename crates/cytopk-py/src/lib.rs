//! PyO3 bindings: `simulate` for every built-in model, and `models()` to list them.
//! The fuller Python API (schedules, noise models, type stubs) follows in M3 (see docs/ROADMAP.md).
//! FCS reading and writing: see `fcs.rs`.

mod fcs;
mod gate;

use cytopk_sim::ddm::{DdmOpts, simulate_ddm as ddm_batch};
use cytopk_sim::{
    Dose, MIN_PARALLEL_BATCH, Model, OneCmtIv, Parallelism, Schedule, SimError, SolverOpts,
    TwoCmtIv, TwoCmtIvIdr, TwoCmtMmIv, simulate_batch_with,
};
use ndarray::{Array3, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
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
    par: Parallelism,
) -> Option<Result<Array3<f64>, SimError>> {
    // Static dispatch: each arm gets its own monomorphised solver.
    Some(match model {
        "onecmt_iv" => simulate_batch_with(&OneCmtIv, theta, sched, times, opts, par),
        "twocmt_iv" => simulate_batch_with(&TwoCmtIv, theta, sched, times, opts, par),
        "twocmt_mm_iv" => simulate_batch_with(&TwoCmtMmIv, theta, sched, times, opts, par),
        "twocmt_iv_idr" => simulate_batch_with(&TwoCmtIvIdr, theta, sched, times, opts, par),
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
    // n_threads=1 and batches too small to gain from threads run on the calling thread: handing a call to a
    // pool and waking its worker costs about one simulation (docs/decisions.md, SBI-flow E2).
    let serial = n_threads == Some(1) || theta.nrows() < MIN_PARALLEL_BATCH;
    let pool = match n_threads {
        Some(0) => return Err(PyValueError::new_err("n_threads must be >= 1")),
        Some(n) if !serial => Some(pool(n)?),
        _ => None,
    };
    let par = if serial {
        Parallelism::Serial
    } else {
        Parallelism::Auto
    };
    let out = py.detach(|| {
        let run = || run_model(model, theta, &sched, &times, &opts, par);
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

/// Drift-diffusion model, one trial per row of `theta` (N x 4: v, a, z, t0). Returns `(rt, choice)`:
/// float64 reaction times (s; NaN on timeout) and int8 choices (1 upper, 0 lower, -1 timeout).
/// Simulation `i` uses its own xoshiro256++ stream from `(seed, i)`. Results are bit-identical for any `n_threads`.
#[pyfunction]
#[pyo3(signature = (theta, seed, dt=0.001, max_t=10.0, n_threads=None))]
#[allow(clippy::type_complexity)]
fn simulate_ddm<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<'py, f64>,
    seed: u64,
    dt: f64,
    max_t: f64,
    n_threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i8>>)> {
    let theta = theta.as_array();
    let serial = n_threads == Some(1) || theta.nrows() < MIN_PARALLEL_BATCH;
    let pool = match n_threads {
        Some(0) => return Err(PyValueError::new_err("n_threads must be >= 1")),
        Some(n) if !serial => Some(pool(n)?),
        _ => None,
    };
    let par = if serial {
        Parallelism::Serial
    } else {
        Parallelism::Auto
    };
    let opts = DdmOpts { dt, max_t };
    let (rt, ch) = py
        .detach(|| {
            let run = || ddm_batch(theta, seed, &opts, par);
            match &pool {
                Some(p) => p.install(run),
                None => run(),
            }
        })
        .map_err(to_py)?;
    Ok((rt.into_pyarray(py), ch.into_pyarray(py)))
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
    m.add_function(wrap_pyfunction!(simulate_ddm, m)?)?;
    fcs::register(m)?;
    gate::register(m)?;
    Ok(())
}
