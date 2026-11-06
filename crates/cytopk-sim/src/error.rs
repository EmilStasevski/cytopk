//! Error type for simulation.

/// Errors produced while validating inputs or integrating a model.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SimError {
    /// A dose has a non-finite or out-of-range field.
    #[error("invalid dose at index {index}: {reason}")]
    InvalidDose {
        /// Index of the dose in the list passed by the caller.
        index: usize,
        /// Human-readable reason.
        reason: &'static str,
    },
    /// A dose targets a state the model does not have.
    #[error("dose targets state {state}, but model `{model}` has only {n_states} states")]
    DoseStateOutOfRange {
        /// Model name.
        model: &'static str,
        /// Requested state index.
        state: usize,
        /// Number of model states.
        n_states: usize,
    },
    /// The model has more states than the solver's fixed buffer capacity.
    #[error("model `{model}` has {n_states} states; the solver supports at most {max}")]
    TooManyStates {
        /// Model name.
        model: &'static str,
        /// Number of model states.
        n_states: usize,
        /// Capacity ([`crate::MAX_STATES`]).
        max: usize,
    },
    /// A closed-form solution is not available for this dosing.
    #[error("closed form not available: {0}")]
    AnalyticUnsupported(&'static str),
    /// Wrong number of parameters.
    #[error("model `{model}` expects {expected} parameters, got {got}")]
    ParamCount {
        /// Model name.
        model: &'static str,
        /// Expected number of parameters.
        expected: usize,
        /// Number of parameters supplied.
        got: usize,
    },
    /// A parameter value violates its constraint.
    #[error("parameter `{name}` = {value} is invalid: must be {constraint}")]
    InvalidParam {
        /// Parameter name.
        name: &'static str,
        /// Supplied value.
        value: f64,
        /// Description of the constraint.
        constraint: &'static str,
    },
    /// Output times are not finite, non-negative and non-decreasing.
    #[error(
        "output times must be finite, >= 0 and non-decreasing (violated at index {index}, value {value})"
    )]
    InvalidTimes {
        /// Index of the offending time.
        index: usize,
        /// Offending value.
        value: f64,
    },
    /// Solver options are invalid.
    #[error("invalid solver options: {0}")]
    InvalidOptions(&'static str),
    /// The adaptive step size became too small to make progress.
    #[error("step size underflow at t = {t}")]
    StepSizeUnderflow {
        /// Time at which the solver gave up.
        t: f64,
    },
    /// The step budget was exhausted.
    #[error("maximum number of steps ({max_steps}) exceeded at t = {t}")]
    MaxStepsExceeded {
        /// Step budget.
        max_steps: usize,
        /// Time at which the solver gave up.
        t: f64,
    },
    /// The state became NaN or infinite.
    #[error("non-finite state encountered at t = {t}")]
    NonFinite {
        /// Time at which this was detected.
        t: f64,
    },
    /// An error in one simulation of a batch.
    #[error("simulation {index}: {source}")]
    InBatch {
        /// Row index in the parameter matrix.
        index: usize,
        /// Underlying error.
        source: Box<SimError>,
    },
}
