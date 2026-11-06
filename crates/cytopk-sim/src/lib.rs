//! `cytopk-sim`: compartmental PK/PD simulators for simulation-based inference.
//!
//! *Research prototype*: toy models with illustrative parameter values. Not validated for
//! clinical or regulatory use.
//!
//! Units: time in days, amounts in mg, volumes in L, concentrations in mg/L.
//!
//! ```
//! use cytopk_sim::{Dose, Schedule, SolverOpts, TwoCmtIv, simulate_one};
//!
//! let sched = Schedule::new(vec![Dose::infusion(0.0, 200.0, 1.0 / 24.0, 0)])?;
//! let times = [0.5, 1.0, 7.0, 28.0];
//! let sol = simulate_one(&TwoCmtIv, &[0.2, 3.0, 0.5, 3.0], &sched, &times, &SolverOpts::default())?;
//! assert_eq!(sol.outputs.shape(), &[4, 1]);
//! # Ok::<(), cytopk_sim::SimError>(())
//! ```

pub mod analytic;
pub mod ddm;
mod dosing;
mod error;
mod model;
mod models;
mod simulate;
mod solver;

pub use dosing::{Dose, Schedule};
pub use error::SimError;
pub use model::{Constraint, Model};
pub use models::{IDR_R_REF, OneCmtIv, TwoCmtIv, TwoCmtIvIdr, TwoCmtMmIv};
pub use simulate::{
    MIN_PARALLEL_BATCH, Parallelism, Solution, simulate_batch, simulate_batch_with, simulate_one,
};
pub use solver::{MAX_STATES, Method, SolverOpts, SolverStats};
