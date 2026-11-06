//! Closed-form concentrations for the linear models, used as test oracles (and later as the
//! counterpart of the NumPy baseline).
//!
//! For a linear model with central-compartment dosing, the central amount after a unit bolus is a
//! sum of exponentials `sum_i c_i exp(-lambda_i s)`. Bolus and zero-order infusion responses follow
//! from this impulse response by superposition. The derivation is in `docs/models.md`.

use crate::{Model, OneCmtIv, Schedule, SimError, TwoCmtIv};

/// Unit impulse response of the central amount: up to two exponentials.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Impulse {
    c: [f64; 2],
    lambda: [f64; 2],
    n: usize,
}

/// `(1 - exp(-lambda x)) / lambda`, with the limit `x` for `lambda = 0`.
#[inline]
fn phi(lambda: f64, x: f64) -> f64 {
    if lambda == 0.0 {
        x
    } else {
        -(-lambda * x).exp_m1() / lambda
    }
}

impl Impulse {
    fn onecmt(cl: f64, v: f64) -> Self {
        Self {
            c: [1.0, 0.0],
            lambda: [cl / v, 0.0],
            n: 1,
        }
    }

    fn twocmt(cl: f64, v1: f64, q: f64, v2: f64) -> Self {
        let (k10, k12, k21) = (cl / v1, q / v1, q / v2);
        if q == 0.0 {
            // No distribution: the central compartment behaves like a 1-cmt model.
            return Self {
                c: [1.0, 0.0],
                lambda: [k10, 0.0],
                n: 1,
            };
        }
        // alpha, beta are the roots of s^2 - (k10 + k12 + k21) s + k10 k21 = 0. The discriminant is
        // (k10 + k12 - k21)^2 + 4 k12 k21 > 0 for q > 0, so alpha > beta.
        let sum = k10 + k12 + k21;
        let disc = (k10 + k12 - k21).powi(2) + 4.0 * k12 * k21;
        let alpha = 0.5 * (sum + disc.sqrt());
        let beta = k10 * k21 / alpha; // product of roots / alpha, which avoids cancellation
        let d = alpha - beta;
        Self {
            c: [(alpha - k21) / d, (k21 - beta) / d],
            lambda: [alpha, beta],
            n: 2,
        }
    }

    /// Central amount at `times` for `sched`, by superposition.
    fn amounts(&self, sched: &Schedule, times: &[f64]) -> Result<Vec<f64>, SimError> {
        if sched.doses().iter().any(|d| d.state != 0) {
            return Err(SimError::AnalyticUnsupported(
                "only doses into the central compartment (state 0)",
            ));
        }
        let out = times
            .iter()
            .map(|&t| {
                let mut a = 0.0;
                for d in sched.doses().iter().filter(|d| d.time <= t) {
                    let s = t - d.time;
                    for i in 0..self.n {
                        let (c, l) = (self.c[i], self.lambda[i]);
                        a += match d.duration {
                            None => d.amount * c * (-l * s).exp(),
                            Some(dur) => {
                                let r = d.amount / dur;
                                if s <= dur {
                                    r * c * phi(l, s)
                                } else {
                                    r * c * phi(l, dur) * (-l * (s - dur)).exp()
                                }
                            }
                        };
                    }
                }
                a
            })
            .collect();
        Ok(out)
    }
}

fn check(m: &dyn Model, theta: &[f64]) -> Result<(), SimError> {
    m.check_params(theta)
}

/// Closed-form concentration for [`OneCmtIv`] with `theta = [CL, V]`.
///
/// Dosing convention as in [`crate::simulate_one`]: the output at `t` includes boluses at `t`.
pub fn onecmt_iv_conc(
    theta: &[f64],
    sched: &Schedule,
    times: &[f64],
) -> Result<Vec<f64>, SimError> {
    check(&OneCmtIv, theta)?;
    let (cl, v) = (theta[0], theta[1]);
    let a = Impulse::onecmt(cl, v).amounts(sched, times)?;
    Ok(a.into_iter().map(|x| x / v).collect())
}

/// Closed-form central concentration for [`TwoCmtIv`] with `theta = [CL, V1, Q, V2]`.
pub fn twocmt_iv_conc(
    theta: &[f64],
    sched: &Schedule,
    times: &[f64],
) -> Result<Vec<f64>, SimError> {
    check(&TwoCmtIv, theta)?;
    let (cl, v1, q, v2) = (theta[0], theta[1], theta[2], theta[3]);
    let a = Impulse::twocmt(cl, v1, q, v2).amounts(sched, times)?;
    Ok(a.into_iter().map(|x| x / v1).collect())
}

/// Macro-constants `(alpha, beta)` of the two-compartment model, with `alpha >= beta >= 0`.
pub fn twocmt_macro_rates(theta: &[f64]) -> Result<(f64, f64), SimError> {
    check(&TwoCmtIv, theta)?;
    let imp = Impulse::twocmt(theta[0], theta[1], theta[2], theta[3]);
    Ok(if imp.n == 2 {
        (imp.lambda[0], imp.lambda[1])
    } else {
        (imp.lambda[0], imp.lambda[0])
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::Dose;

    #[test]
    fn onecmt_bolus_and_infusion_closed_forms() {
        let (cl, v) = (0.5, 2.0);
        let k = cl / v;
        let s = Schedule::new(vec![Dose::bolus(0.0, 100.0, 0)]).unwrap();
        let c = onecmt_iv_conc(&[cl, v], &s, &[0.0, 1.0, 4.0]).unwrap();
        for (ci, t) in c.iter().zip([0.0, 1.0, 4.0]) {
            assert!((ci - 50.0 * (-k * t).exp()).abs() < 1e-12);
        }
        let s = Schedule::new(vec![Dose::infusion(0.0, 100.0, 2.0, 0)]).unwrap();
        let c = onecmt_iv_conc(&[cl, v], &s, &[1.0, 2.0, 3.0]).unwrap();
        let r0 = 50.0;
        let c_end = r0 / cl * (1.0 - (-k * 2.0).exp());
        assert!((c[0] - r0 / cl * (1.0 - (-k).exp())).abs() < 1e-12);
        assert!((c[1] - c_end).abs() < 1e-12);
        assert!((c[2] - c_end * (-k).exp()).abs() < 1e-12);
    }

    #[test]
    fn steady_state_accumulation_ratio() {
        // 1-cmt bolus every tau: trough before the (n+1)-th dose -> C1(tau) / (1 - e^{-k tau}).
        let (cl, v, tau) = (0.3, 3.0, 7.0);
        let k = cl / v;
        let s = Schedule::repeated(Dose::bolus(0.0, 100.0, 0), tau, 60).unwrap();
        let t = 60.0 * tau - 1e-9;
        let c = onecmt_iv_conc(&[cl, v], &s, &[t]).unwrap()[0];
        let expected = 100.0 / v * (-k * tau).exp() / (1.0 - (-k * tau).exp());
        assert!((c / expected - 1.0).abs() < 1e-6, "{c} vs {expected}");
    }

    #[test]
    fn twocmt_reduces_to_onecmt() {
        let s = Schedule::new(vec![Dose::infusion(0.0, 100.0, 0.5, 0)]).unwrap();
        let ts = [0.25, 1.0, 5.0];
        let a = twocmt_iv_conc(&[0.2, 3.0, 0.0, 3.0], &s, &ts).unwrap();
        let b = onecmt_iv_conc(&[0.2, 3.0], &s, &ts).unwrap();
        assert_eq!(a, b);
        // No elimination and no distribution: the amount is conserved.
        let c = twocmt_iv_conc(&[0.0, 3.0, 0.0, 3.0], &s, &[10.0]).unwrap();
        assert!((c[0] - 100.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn twocmt_initial_value_and_macro_rates() {
        let theta = [0.2, 3.0, 0.5, 3.0];
        let s = Schedule::new(vec![Dose::bolus(0.0, 90.0, 0)]).unwrap();
        let c0 = twocmt_iv_conc(&theta, &s, &[0.0]).unwrap()[0];
        assert!((c0 - 30.0).abs() < 1e-12);
        let (a, b) = twocmt_macro_rates(&theta).unwrap();
        let (k10, k12, k21) = (0.2 / 3.0, 0.5 / 3.0, 0.5 / 3.0);
        assert!((a + b - (k10 + k12 + k21)).abs() < 1e-14);
        assert!((a * b - k10 * k21).abs() < 1e-14);
        assert!(a > b && b > 0.0);
    }

    #[test]
    fn rejects_peripheral_doses_and_bad_params() {
        let s = Schedule::new(vec![Dose::bolus(0.0, 1.0, 1)]).unwrap();
        assert!(matches!(
            twocmt_iv_conc(&[0.2, 3.0, 0.5, 3.0], &s, &[1.0]),
            Err(SimError::AnalyticUnsupported(_))
        ));
        assert!(onecmt_iv_conc(&[0.2, 0.0], &Schedule::empty(), &[1.0]).is_err());
    }
}
