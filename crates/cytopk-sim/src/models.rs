//! Built-in models. Units: time in days, amounts in mg, volumes in L, clearances in L/day,
//! concentrations in mg/L. The equations are in `docs/models.md`.

use crate::model::{Constraint, Model};

/// One-compartment model with linear elimination, IV dosing.
///
/// States `[A1]`, parameters `[CL, V]`, output `[C = A1 / V]`.
/// `dA1/dt = -CL/V * A1 + input_0`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OneCmtIv;

impl Model for OneCmtIv {
    fn name(&self) -> &'static str {
        "onecmt_iv"
    }
    fn state_names(&self) -> &'static [&'static str] {
        &["A1"]
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["CL", "V"]
    }
    fn param_constraints(&self) -> &'static [Constraint] {
        &[Constraint::NonNegative, Constraint::Positive]
    }
    fn output_names(&self) -> &'static [&'static str] {
        &["C"]
    }
    #[inline]
    fn rhs(&self, _t: f64, y: &[f64], theta: &[f64], input: &[f64], dy: &mut [f64]) {
        let (cl, v) = (theta[0], theta[1]);
        dy[0] = -cl / v * y[0] + input[0];
    }
    #[inline]
    fn observe(&self, y: &[f64], theta: &[f64], out: &mut [f64]) {
        out[0] = y[0] / theta[1];
    }
}

/// Two-compartment model with linear elimination from the central compartment, IV dosing.
///
/// States `[A1 (central), A2 (peripheral)]`, parameters `[CL, V1, Q, V2]`, output `[C = A1 / V1]`.
/// ```text
/// dA1/dt = -(CL + Q)/V1 * A1 + Q/V2 * A2 + input_0
/// dA2/dt =        Q/V1 * A1 - Q/V2 * A2 + input_1
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct TwoCmtIv;

impl Model for TwoCmtIv {
    fn name(&self) -> &'static str {
        "twocmt_iv"
    }
    fn state_names(&self) -> &'static [&'static str] {
        &["A1", "A2"]
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["CL", "V1", "Q", "V2"]
    }
    fn param_constraints(&self) -> &'static [Constraint] {
        &[
            Constraint::NonNegative,
            Constraint::Positive,
            Constraint::NonNegative,
            Constraint::Positive,
        ]
    }
    fn output_names(&self) -> &'static [&'static str] {
        &["C"]
    }
    #[inline]
    fn rhs(&self, _t: f64, y: &[f64], theta: &[f64], input: &[f64], dy: &mut [f64]) {
        let (cl, v1, q, v2) = (theta[0], theta[1], theta[2], theta[3]);
        let c1 = y[0] / v1;
        let c2 = y[1] / v2;
        let transfer = q * (c1 - c2);
        dy[0] = -cl * c1 - transfer + input[0];
        dy[1] = transfer + input[1];
    }
    #[inline]
    fn observe(&self, y: &[f64], theta: &[f64], out: &mut [f64]) {
        out[0] = y[0] / theta[1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimError;

    #[test]
    fn metadata_and_param_checks() {
        assert_eq!(OneCmtIv.name(), "onecmt_iv");
        assert_eq!(
            (
                OneCmtIv.n_states(),
                OneCmtIv.n_params(),
                OneCmtIv.n_outputs()
            ),
            (1, 2, 1)
        );
        assert_eq!(
            (
                TwoCmtIv.n_states(),
                TwoCmtIv.n_params(),
                TwoCmtIv.n_outputs()
            ),
            (2, 4, 1)
        );
        assert!(OneCmtIv.check_params(&[0.0, 3.0]).is_ok());
        assert!(matches!(
            OneCmtIv.check_params(&[0.2]),
            Err(SimError::ParamCount {
                expected: 2,
                got: 1,
                ..
            })
        ));
        assert!(matches!(
            OneCmtIv.check_params(&[0.2, 0.0]),
            Err(SimError::InvalidParam { name: "V", .. })
        ));
        assert!(matches!(
            TwoCmtIv.check_params(&[-0.1, 3.0, 0.5, 3.0]),
            Err(SimError::InvalidParam { name: "CL", .. })
        ));
        assert!(TwoCmtIv.check_params(&[0.2, 3.0, f64::NAN, 3.0]).is_err());
    }

    #[test]
    fn rhs_and_observe() {
        let mut dy = [0.0; 2];
        TwoCmtIv.rhs(
            0.0,
            &[6.0, 3.0],
            &[0.3, 3.0, 0.5, 3.0],
            &[1.0, 0.0],
            &mut dy,
        );
        // c1 = 2, c2 = 1, transfer = 0.5
        assert_eq!(dy, [-0.6 - 0.5 + 1.0, 0.5]);
        let mut out = [0.0];
        TwoCmtIv.observe(&[6.0, 3.0], &[0.3, 3.0, 0.5, 3.0], &mut out);
        assert_eq!(out, [2.0]);
        let mut dy1 = [0.0];
        OneCmtIv.rhs(0.0, &[10.0], &[1.0, 5.0], &[0.0], &mut dy1);
        assert_eq!(dy1, [-2.0]);
        let mut y0 = [9.0, 9.0];
        TwoCmtIv.init(&[0.3, 3.0, 0.5, 3.0], &mut y0);
        assert_eq!(y0, [0.0, 0.0]);
    }
}
