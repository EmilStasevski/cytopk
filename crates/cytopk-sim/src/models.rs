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

/// Two-compartment model with saturable (Michaelis–Menten) elimination from the central compartment.
///
/// States `[A1, A2]`, parameters `[Vmax (mg/day), Km (mg/L), V1, Q, V2]`, output `[C = A1 / V1]`.
/// ```text
/// dA1/dt = -Vmax * C1 / (Km + C1) - Q * (C1 - C2) + input_0
/// dA2/dt =                          Q * (C1 - C2) + input_1
/// ```
/// For `C1 << Km` this reduces to linear elimination with `CL = Vmax / Km`.
#[derive(Debug, Clone, Copy, Default)]
pub struct TwoCmtMmIv;

impl Model for TwoCmtMmIv {
    fn name(&self) -> &'static str {
        "twocmt_mm_iv"
    }
    fn state_names(&self) -> &'static [&'static str] {
        &["A1", "A2"]
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["Vmax", "Km", "V1", "Q", "V2"]
    }
    fn param_constraints(&self) -> &'static [Constraint] {
        &[
            Constraint::NonNegative,
            Constraint::Positive,
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
        let (vmax, km, v1, q, v2) = (theta[0], theta[1], theta[2], theta[3], theta[4]);
        let c1 = y[0] / v1;
        let c2 = y[1] / v2;
        let transfer = q * (c1 - c2);
        dy[0] = -vmax * c1 / (km + c1) - transfer + input[0];
        dy[1] = transfer + input[1];
    }
    #[inline]
    fn observe(&self, y: &[f64], theta: &[f64], out: &mut [f64]) {
        out[0] = y[0] / theta[2];
    }
}

/// Reference level for the activated-cell fraction: `p = R / (R + R_REF)`. With `R(0) = 1` this
/// gives a 10 % baseline. This is a toy choice (see `docs/models.md`).
pub const IDR_R_REF: f64 = 9.0;

/// [`TwoCmtIv`] PK driving an indirect-response PD state `R` (stimulated production of activated cells).
///
/// States `[A1, A2, R]`, parameters `[CL, V1, Q, V2, kout (1/day), Emax (-), EC50 (mg/L)]`,
/// outputs `[C = A1 / V1, p = R / (R + 9)]`. `R(0) = 1` (normalised baseline).
/// ```text
/// dR/dt = kout * (1 + Emax * C / (EC50 + C)) - kout * R
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct TwoCmtIvIdr;

impl Model for TwoCmtIvIdr {
    fn name(&self) -> &'static str {
        "twocmt_iv_idr"
    }
    fn state_names(&self) -> &'static [&'static str] {
        &["A1", "A2", "R"]
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["CL", "V1", "Q", "V2", "kout", "Emax", "EC50"]
    }
    fn param_constraints(&self) -> &'static [Constraint] {
        &[
            Constraint::NonNegative,
            Constraint::Positive,
            Constraint::NonNegative,
            Constraint::Positive,
            Constraint::Positive,
            Constraint::NonNegative,
            Constraint::Positive,
        ]
    }
    fn output_names(&self) -> &'static [&'static str] {
        &["C", "p"]
    }
    fn init(&self, _theta: &[f64], y0: &mut [f64]) {
        y0[0] = 0.0;
        y0[1] = 0.0;
        y0[2] = 1.0;
    }
    #[inline]
    fn rhs(&self, t: f64, y: &[f64], theta: &[f64], input: &[f64], dy: &mut [f64]) {
        TwoCmtIv.rhs(t, &y[..2], &theta[..4], &input[..2], &mut dy[..2]);
        let (kout, emax, ec50) = (theta[4], theta[5], theta[6]);
        let c = y[0] / theta[1];
        dy[2] = kout * (1.0 + emax * c / (ec50 + c)) - kout * y[2] + input[2];
    }
    #[inline]
    fn observe(&self, y: &[f64], theta: &[f64], out: &mut [f64]) {
        out[0] = y[0] / theta[1];
        out[1] = y[2] / (y[2] + IDR_R_REF);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimError;

    #[test]
    fn nonlinear_models_rhs_init_observe() {
        let mut dy = [0.0; 2];
        // c1 = 2, c2 = 1: elimination = 10 * 2 / (2 + 2) = 5, transfer = 0.5
        TwoCmtMmIv.rhs(
            0.0,
            &[6.0, 3.0],
            &[10.0, 2.0, 3.0, 0.5, 3.0],
            &[1.0, 0.0],
            &mut dy,
        );
        assert_eq!(dy, [-5.0 - 0.5 + 1.0, 0.5]);
        let mut out = [0.0];
        TwoCmtMmIv.observe(&[6.0, 3.0], &[10.0, 2.0, 3.0, 0.5, 3.0], &mut out);
        assert_eq!(out, [2.0]);
        assert!(
            TwoCmtMmIv
                .check_params(&[10.0, 0.0, 3.0, 0.5, 3.0])
                .is_err()
        );

        let th = [0.3, 3.0, 0.5, 3.0, 0.3, 2.0, 5.0];
        let mut y0 = [9.0; 3];
        TwoCmtIvIdr.init(&th, &mut y0);
        assert_eq!(y0, [0.0, 0.0, 1.0]);
        let mut dy = [0.0; 3];
        TwoCmtIvIdr.rhs(0.0, &y0, &th, &[0.0; 3], &mut dy);
        assert_eq!(dy, [0.0, 0.0, 0.0], "baseline is a steady state");
        // C = 15/3 = 5 = EC50 -> stimulation 1 + 2 * 0.5 = 2; dR = 0.3 * 2 - 0.3 * 1
        TwoCmtIvIdr.rhs(0.0, &[15.0, 0.0, 1.0], &th, &[0.0; 3], &mut dy);
        assert!((dy[2] - 0.3).abs() < 1e-15);
        let mut out = [0.0; 2];
        TwoCmtIvIdr.observe(&[15.0, 0.0, 1.0], &th, &mut out);
        assert_eq!(out, [5.0, 0.1]);
        assert_eq!(TwoCmtIvIdr.n_outputs(), 2);
        assert!(
            TwoCmtIvIdr
                .check_params(&[0.3, 3.0, 0.5, 3.0, 0.0, 2.0, 5.0])
                .is_err()
        );
    }

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
