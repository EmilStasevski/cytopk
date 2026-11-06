//! Single-subject and batch simulation drivers.

use ndarray::{Array2, Array3, ArrayView2};
use rayon::prelude::*;

use crate::solver::{MAX_STATES, SolverStats, Workspace, integrate_segment};
use crate::{Model, Schedule, SimError, SolverOpts};

/// The result of [`simulate_one`].
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// The requested output times.
    pub times: Vec<f64>,
    /// States at each output time, shape `T × n_states`.
    pub states: Array2<f64>,
    /// Observations at each output time, shape `T × n_outputs`.
    pub outputs: Array2<f64>,
    /// Solver counters.
    pub stats: SolverStats,
}

/// Points where integration stops: dose events and output times, merged. Built once per batch.
#[derive(Debug)]
struct Plan {
    stops: Vec<Stop>,
    /// Boluses `(state, amount)`, grouped by stop.
    boluses: Vec<(usize, f64)>,
    /// Infusion rates valid on `[stop_k, stop_{k+1})`, flattened as `n_stops × n_states`.
    rates: Vec<f64>,
}

#[derive(Debug)]
struct Stop {
    t: f64,
    boluses: std::ops::Range<usize>,
    /// Range of output indices recorded at this stop (after events are applied).
    outputs: std::ops::Range<usize>,
}

fn validate_times(times: &[f64]) -> Result<(), SimError> {
    let mut prev = 0.0;
    for (index, &value) in times.iter().enumerate() {
        if !value.is_finite() || value < prev {
            return Err(SimError::InvalidTimes { index, value });
        }
        prev = value;
    }
    Ok(())
}

impl Plan {
    fn new(sched: &Schedule, times: &[f64], n_states: usize) -> Self {
        let t_last = times.last().copied().unwrap_or(f64::NEG_INFINITY);
        let mut ts = sched.event_times();
        ts.extend_from_slice(times);
        ts.sort_by(f64::total_cmp);
        ts.dedup();
        // Events after the last output time cannot affect any output.
        ts.retain(|&t| t <= t_last);

        let mut stops = Vec::with_capacity(ts.len());
        let mut boluses = Vec::new();
        let mut rates = vec![0.0; ts.len() * n_states];
        let mut di = 0; // cursor into doses, which are sorted by time
        let mut oi = 0; // cursor into output times, which are sorted
        let doses = sched.doses();
        for (k, &t) in ts.iter().enumerate() {
            let b0 = boluses.len();
            while di < doses.len() && doses[di].time <= t {
                let d = &doses[di];
                if d.duration.is_none() && d.time == t {
                    boluses.push((d.state, d.amount));
                }
                di += 1;
            }
            let o0 = oi;
            while oi < times.len() && times[oi] == t {
                oi += 1;
            }
            sched.infusion_rates_at(t, &mut rates[k * n_states..(k + 1) * n_states]);
            stops.push(Stop {
                t,
                boluses: b0..boluses.len(),
                outputs: o0..oi,
            });
        }
        Self {
            stops,
            boluses,
            rates,
        }
    }
}

fn check_inputs<M: Model + ?Sized>(
    m: &M,
    sched: &Schedule,
    times: &[f64],
    opts: &SolverOpts,
) -> Result<(), SimError> {
    opts.validate()?;
    validate_times(times)?;
    if m.n_states() > MAX_STATES {
        return Err(SimError::TooManyStates {
            model: m.name(),
            n_states: m.n_states(),
            max: MAX_STATES,
        });
    }
    sched.check_states(m.name(), m.n_states())
}

/// Runs one subject. `states_out` (if given) has `T × n_states` entries and `obs_out` has `T × n_outputs`,
/// both row-major.
#[allow(clippy::too_many_arguments)]
fn run<M: Model + ?Sized>(
    m: &M,
    theta: &[f64],
    plan: &Plan,
    opts: &SolverOpts,
    mut states_out: Option<&mut [f64]>,
    obs_out: &mut [f64],
) -> Result<SolverStats, SimError> {
    let ns = m.n_states();
    let no = m.n_outputs();
    let mut stats = SolverStats::default();
    // Stage buffers and state live on this stack frame: this lets LLVM keep them apart
    // (no aliasing), which was worth ~35 % in the go/no-go profiling (docs/decisions.md).
    let mut ws = Workspace::new();
    let mut ybuf = [0.0; MAX_STATES];
    let y = &mut ybuf[..ns];
    m.init(theta, y);
    let mut t = 0.0;
    let zeros = vec![0.0; ns];
    let mut input: &[f64] = &zeros; // no infusion is running before the first stop
    for (k, stop) in plan.stops.iter().enumerate() {
        integrate_segment(m, theta, input, t, stop.t, y, &mut ws, opts, &mut stats)?;
        t = stop.t;
        for &(state, amount) in &plan.boluses[stop.boluses.clone()] {
            y[state] += amount;
        }
        input = &plan.rates[k * ns..(k + 1) * ns];
        for j in stop.outputs.clone() {
            m.observe(y, theta, &mut obs_out[j * no..(j + 1) * no]);
            if let Some(s) = states_out.as_deref_mut() {
                s[j * ns..(j + 1) * ns].copy_from_slice(y);
            }
        }
    }
    Ok(stats)
}

/// Simulates one subject with parameters `theta` and returns states and outputs at `times`.
///
/// `times` must be finite, `>= 0` and non-decreasing. Repeated times are allowed. The simulation
/// starts at `t = 0` from [`Model::init`]. An output at time `t` includes every bolus given at `t`.
pub fn simulate_one<M: Model + ?Sized>(
    m: &M,
    theta: &[f64],
    sched: &Schedule,
    times: &[f64],
    opts: &SolverOpts,
) -> Result<Solution, SimError> {
    check_inputs(m, sched, times, opts)?;
    m.check_params(theta)?;
    let plan = Plan::new(sched, times, m.n_states());
    let (nt, ns, no) = (times.len(), m.n_states(), m.n_outputs());
    let mut states = vec![0.0; nt * ns];
    let mut outputs = vec![0.0; nt * no];
    let stats = run(m, theta, &plan, opts, Some(&mut states), &mut outputs)?;
    let shape_err = |_| SimError::InvalidOptions("internal shape error");
    Ok(Solution {
        times: times.to_vec(),
        states: Array2::from_shape_vec((nt, ns), states).map_err(shape_err)?,
        outputs: Array2::from_shape_vec((nt, no), outputs).map_err(shape_err)?,
        stats,
    })
}

/// How [`simulate_batch_with`] distributes simulations over threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parallelism {
    /// Run every simulation on the calling thread. No thread pool is involved, so no cross-thread hand-off.
    Serial,
    /// Use the current Rayon pool, except for batches smaller than [`MIN_PARALLEL_BATCH`], which run serially.
    Auto,
}

/// Batches smaller than this run on the calling thread under [`Parallelism::Auto`]. Handing a call to the Rayon
/// pool and waking its workers costs about one simulation's worth of time (measured in the SBI-flow E2
/// benchmark, docs/decisions.md), so tiny batches do not gain from threads.
pub const MIN_PARALLEL_BATCH: usize = 4;

/// Simulates every row of `thetas` (shape `N × n_params`) with [`Parallelism::Auto`].
///
/// Returns observations with shape `N × T × n_outputs`. The results are bit-identical for any number
/// of threads and either parallelism mode, because each simulation is independent and deterministic.
/// If several rows fail, the error for the lowest row index is returned, wrapped in [`SimError::InBatch`].
pub fn simulate_batch<M: Model + ?Sized>(
    m: &M,
    thetas: ArrayView2<'_, f64>,
    sched: &Schedule,
    times: &[f64],
    opts: &SolverOpts,
) -> Result<Array3<f64>, SimError> {
    simulate_batch_with(m, thetas, sched, times, opts, Parallelism::Auto)
}

/// Like [`simulate_batch`], with explicit control over threading.
pub fn simulate_batch_with<M: Model + ?Sized>(
    m: &M,
    thetas: ArrayView2<'_, f64>,
    sched: &Schedule,
    times: &[f64],
    opts: &SolverOpts,
    parallelism: Parallelism,
) -> Result<Array3<f64>, SimError> {
    check_inputs(m, sched, times, opts)?;
    let n = thetas.nrows();
    if thetas.ncols() != m.n_params() {
        return Err(SimError::ParamCount {
            model: m.name(),
            expected: m.n_params(),
            got: thetas.ncols(),
        });
    }
    for (index, row) in thetas.outer_iter().enumerate() {
        let theta: Vec<f64> = row.to_vec();
        m.check_params(&theta).map_err(|e| SimError::InBatch {
            index,
            source: Box::new(e),
        })?;
    }
    let (nt, ns, no) = (times.len(), m.n_states(), m.n_outputs());
    let mut out = Array3::<f64>::zeros((n, nt, no));
    let chunk = nt * no;
    if n == 0 || chunk == 0 {
        return Ok(out);
    }
    let plan = Plan::new(sched, times, ns);
    let flat = out
        .as_slice_mut()
        .ok_or(SimError::InvalidOptions("internal layout error"))?;
    let one = |theta: &mut Vec<f64>, (i, obs): (usize, &mut [f64])| {
        for (dst, &src) in theta.iter_mut().zip(thetas.row(i)) {
            *dst = src;
        }
        run(m, theta, &plan, opts, None, obs)
            .map(|_| ())
            .map_err(|e| SimError::InBatch {
                index: i,
                source: Box::new(e),
            })
    };
    if parallelism == Parallelism::Serial || n < MIN_PARALLEL_BATCH {
        let mut theta = vec![0.0; m.n_params()];
        // Sequential order, so the first error is the lowest index, as in the parallel path.
        for item in flat.chunks_mut(chunk).enumerate() {
            one(&mut theta, item)?;
        }
        return Ok(out);
    }
    let results: Vec<Result<(), SimError>> = flat
        .par_chunks_mut(chunk)
        .enumerate()
        .map_init(|| vec![0.0; m.n_params()], one)
        .collect();
    results.into_iter().find(Result::is_err).unwrap_or(Ok(()))?;
    Ok(out)
}
