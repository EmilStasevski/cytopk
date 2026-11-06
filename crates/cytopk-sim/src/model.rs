//! The [`Model`] trait implemented by every simulator model.

use crate::SimError;

/// Constraint on a single model parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Constraint {
    /// Finite and `> 0`.
    Positive,
    /// Finite and `>= 0`.
    NonNegative,
}

impl Constraint {
    fn holds(self, x: f64) -> bool {
        x.is_finite()
            && match self {
                Self::Positive => x > 0.0,
                Self::NonNegative => x >= 0.0,
            }
    }

    fn describe(self) -> &'static str {
        match self {
            Self::Positive => "finite and > 0",
            Self::NonNegative => "finite and >= 0",
        }
    }
}

/// An ODE model `dy/dt = f(t, y, theta) + input`, where `input` holds the infusion rates per state.
///
/// Implementations must be pure. The same inputs must give bit-identical outputs, because batch
/// determinism depends on it.
pub trait Model: Send + Sync {
    /// Stable identifier, e.g. `"twocmt_iv"`.
    fn name(&self) -> &'static str;
    /// Names of the states (e.g. compartment amounts).
    fn state_names(&self) -> &'static [&'static str];
    /// Names of the parameters, in the order expected in `theta`.
    fn param_names(&self) -> &'static [&'static str];
    /// One constraint per parameter, in the same order as [`Model::param_names`].
    fn param_constraints(&self) -> &'static [Constraint];
    /// Names of the observed outputs.
    fn output_names(&self) -> &'static [&'static str];

    /// Number of states.
    fn n_states(&self) -> usize {
        self.state_names().len()
    }
    /// Number of parameters.
    fn n_params(&self) -> usize {
        self.param_names().len()
    }
    /// Number of outputs.
    fn n_outputs(&self) -> usize {
        self.output_names().len()
    }

    /// Checks the parameter count and that every parameter satisfies its constraint.
    fn check_params(&self, theta: &[f64]) -> Result<(), SimError> {
        if theta.len() != self.n_params() {
            return Err(SimError::ParamCount {
                model: self.name(),
                expected: self.n_params(),
                got: theta.len(),
            });
        }
        for ((&x, &name), &c) in theta
            .iter()
            .zip(self.param_names())
            .zip(self.param_constraints())
        {
            if !c.holds(x) {
                return Err(SimError::InvalidParam {
                    name,
                    value: x,
                    constraint: c.describe(),
                });
            }
        }
        Ok(())
    }

    /// Baseline state at `t = 0`, before any dose. The default is all zeros (a drug-naive subject).
    fn init(&self, _theta: &[f64], y0: &mut [f64]) {
        y0.fill(0.0);
    }

    /// Right-hand side: writes `dy/dt` into `dy`. `input[i]` is the infusion rate into state `i`.
    fn rhs(&self, t: f64, y: &[f64], theta: &[f64], input: &[f64], dy: &mut [f64]);

    /// Observation function: writes the outputs for state `y` into `out`.
    fn observe(&self, y: &[f64], theta: &[f64], out: &mut [f64]);
}
