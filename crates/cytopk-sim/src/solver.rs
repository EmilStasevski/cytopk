//! Explicit Runge–Kutta integrators: adaptive Dormand–Prince 5(4) and fixed-step classical RK4.
//!
//! The integrators work on one *smooth segment* `[t0, t1]` at a time. The caller splits the time
//! axis at every discontinuity (dose, infusion end) and every requested output time, so no step
//! ever crosses a discontinuity, and outputs are taken at step ends without interpolation.

use crate::{Model, SimError};

/// Integration method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Method {
    /// Adaptive Dormand–Prince 5(4) (the method behind MATLAB `ode45` and SciPy `RK45`).
    Dopri5,
    /// Classical 4th-order Runge–Kutta with fixed step size `step` (days). Each segment is split into
    /// `ceil(len / step)` equal steps. Meant for reference and debugging.
    Rk4 {
        /// Maximum step size.
        step: f64,
    },
}

/// Solver options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverOpts {
    /// Integration method.
    pub method: Method,
    /// Relative tolerance (Dopri5 only).
    pub rtol: f64,
    /// Absolute tolerance (Dopri5 only), in state units (mg).
    pub atol: f64,
    /// Maximum number of step attempts (accepted plus rejected) per simulation.
    pub max_steps: usize,
    /// Initial step size. `None` uses the Hairer–Nørsett–Wanner starting-step heuristic.
    pub h_init: Option<f64>,
    /// Upper bound on the step size. `None` means unbounded.
    pub h_max: Option<f64>,
}

impl Default for SolverOpts {
    fn default() -> Self {
        Self {
            method: Method::Dopri5,
            rtol: 1e-6,
            atol: 1e-9,
            max_steps: 100_000,
            h_init: None,
            h_max: None,
        }
    }
}

impl SolverOpts {
    /// Dormand–Prince with the given tolerances and default values for everything else.
    pub fn dopri5(rtol: f64, atol: f64) -> Self {
        Self {
            rtol,
            atol,
            ..Self::default()
        }
    }

    /// Fixed-step RK4 with step size `step`.
    pub fn rk4(step: f64) -> Self {
        Self {
            method: Method::Rk4 { step },
            ..Self::default()
        }
    }

    /// Checks that all options are finite and in range.
    pub fn validate(&self) -> Result<(), SimError> {
        let pos = |x: f64| x.is_finite() && x > 0.0;
        if let Method::Rk4 { step } = self.method
            && !pos(step)
        {
            return Err(SimError::InvalidOptions("RK4 step must be finite and > 0"));
        }
        if !pos(self.rtol) {
            return Err(SimError::InvalidOptions("rtol must be finite and > 0"));
        }
        if !(self.atol.is_finite() && self.atol >= 0.0) {
            return Err(SimError::InvalidOptions("atol must be finite and >= 0"));
        }
        if self.max_steps == 0 {
            return Err(SimError::InvalidOptions("max_steps must be > 0"));
        }
        if self.h_init.is_some_and(|h| !pos(h)) {
            return Err(SimError::InvalidOptions("h_init must be finite and > 0"));
        }
        if self.h_max.is_some_and(|h| !pos(h)) {
            return Err(SimError::InvalidOptions("h_max must be finite and > 0"));
        }
        Ok(())
    }
}

/// Counters for one simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SolverStats {
    /// Accepted steps.
    pub n_accepted: usize,
    /// Rejected steps (Dopri5 only).
    pub n_rejected: usize,
    /// Right-hand-side evaluations.
    pub n_rhs: usize,
}

/// Largest number of states a model may have. The solver keeps its stage buffers in fixed-size
/// stack arrays of this capacity (see `docs/decisions.md`, "stack buffers"). The capacity has no
/// measurable cost, and it lets LLVM prove the buffers don't alias.
pub const MAX_STATES: usize = 32;

/// Stage buffers for one simulation. Create it as a local (on the stack), not on the heap.
/// Only the first `n_states` entries of each buffer are used.
#[derive(Debug, Clone)]
pub(crate) struct Workspace {
    k: [[f64; MAX_STATES]; 7],
    y_stage: [f64; MAX_STATES],
    y_new: [f64; MAX_STATES],
    /// Current step size proposal carried across segments (`0.0` = not yet chosen).
    pub(crate) h: f64,
}

impl Workspace {
    pub(crate) fn new() -> Self {
        Self {
            k: [[0.0; MAX_STATES]; 7],
            y_stage: [0.0; MAX_STATES],
            y_new: [0.0; MAX_STATES],
            h: 0.0,
        }
    }
}

// Dormand–Prince 5(4) coefficients (Hairer, Nørsett & Wanner, Solving ODEs I, Table 5.2).
const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;
const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;
const B1: f64 = 35.0 / 384.0;
const B3: f64 = 500.0 / 1113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2187.0 / 6784.0;
const B6: f64 = 11.0 / 84.0;
// Error weights e = b(5th order) - b*(4th order).
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

const SAFETY: f64 = 0.9;
const FAC_MIN: f64 = 0.2;
const FAC_MAX: f64 = 10.0;

/// Integrates `y` from `t0` to `t1` in place, using the input rates `input`, which are constant on the segment.
#[allow(clippy::too_many_arguments)]
pub(crate) fn integrate_segment<M: Model + ?Sized>(
    m: &M,
    theta: &[f64],
    input: &[f64],
    t0: f64,
    t1: f64,
    y: &mut [f64],
    ws: &mut Workspace,
    opts: &SolverOpts,
    stats: &mut SolverStats,
) -> Result<(), SimError> {
    if t1 <= t0 {
        return Ok(());
    }
    match opts.method {
        Method::Dopri5 => dopri5(m, theta, input, t0, t1, y, ws, opts, stats),
        Method::Rk4 { step } => rk4(m, theta, input, t0, t1, y, ws, step, opts, stats),
    }
}

#[inline]
fn err_norm(err: impl Iterator<Item = (f64, f64, f64)>, rtol: f64, atol: f64, n: usize) -> f64 {
    // RMS norm of e_i / (atol + rtol * max(|y_i|, |y_new_i|)), as in Hairer et al.
    let s: f64 = err
        .map(|(e, y0, y1)| {
            let sc = atol + rtol * y0.abs().max(y1.abs());
            let r = e / sc;
            r * r
        })
        .sum();
    (s / n as f64).sqrt()
}

#[allow(clippy::too_many_arguments)]
fn initial_step<M: Model + ?Sized>(
    m: &M,
    theta: &[f64],
    input: &[f64],
    t0: f64,
    y: &[f64],
    ws: &mut Workspace,
    opts: &SolverOpts,
    stats: &mut SolverStats,
) -> f64 {
    // Hairer, Nørsett & Wanner, Solving ODEs I, Sec. II.4 ("starting step size"), order p = 5.
    let n = y.len();
    let [k1, k2, ..] = &mut ws.k;
    let (k1, k2, y_stage) = (&mut k1[..n], &mut k2[..n], &mut ws.y_stage[..n]);
    m.rhs(t0, y, theta, input, k1);
    stats.n_rhs += 1;
    let sc = |yi: f64| opts.atol + opts.rtol * yi.abs();
    let rms =
        |it: &mut dyn Iterator<Item = f64>| (it.map(|x| x * x).sum::<f64>() / n as f64).sqrt();
    let d0 = rms(&mut y.iter().map(|&yi| yi / sc(yi)));
    let d1 = rms(&mut y.iter().zip(k1.iter()).map(|(&yi, &fi)| fi / sc(yi)));
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };
    for ((ys, &yi), &fi) in y_stage.iter_mut().zip(y).zip(k1.iter()) {
        *ys = yi + h0 * fi;
    }
    m.rhs(t0 + h0, y_stage, theta, input, k2);
    stats.n_rhs += 1;
    let d2 = rms(&mut y
        .iter()
        .zip(k1.iter().zip(k2.iter()))
        .map(|(&yi, (&f0, &f1))| (f1 - f0) / sc(yi)))
        / h0;
    let dm = d1.max(d2);
    let h1 = if dm <= 1e-15 {
        (h0 * 1e-3).max(1e-6)
    } else {
        (0.01 / dm).powf(1.0 / 5.0)
    };
    (100.0 * h0).min(h1)
}

#[allow(clippy::too_many_arguments)]
fn dopri5<M: Model + ?Sized>(
    m: &M,
    theta: &[f64],
    input: &[f64],
    t0: f64,
    t1: f64,
    y: &mut [f64],
    ws: &mut Workspace,
    opts: &SolverOpts,
    stats: &mut SolverStats,
) -> Result<(), SimError> {
    let n = y.len();
    if ws.h <= 0.0 {
        ws.h = opts
            .h_init
            .unwrap_or_else(|| initial_step(m, theta, input, t0, y, ws, opts, stats));
    }
    let h_max = opts.h_max.unwrap_or(f64::INFINITY);
    let mut t = t0;
    // The first stage is recomputed at every segment start because the RHS may have jumped.
    m.rhs(t, y, theta, input, &mut ws.k[0][..n]);
    stats.n_rhs += 1;
    let mut last_rejected = false;

    while t < t1 {
        if stats.n_accepted + stats.n_rejected >= opts.max_steps {
            return Err(SimError::MaxStepsExceeded {
                max_steps: opts.max_steps,
                t,
            });
        }
        let remaining = t1 - t;
        let mut h = ws.h.min(h_max);
        // Land exactly on t1, and avoid leaving a sliver shorter than 1 % of a step.
        let clamped = h >= remaining * 0.99;
        if clamped {
            h = remaining;
        }
        if h <= 16.0 * f64::EPSILON * t.abs().max(1.0) && !clamped {
            return Err(SimError::StepSizeUnderflow { t });
        }

        let Workspace {
            k, y_stage, y_new, ..
        } = ws;
        let [k1, k2, k3, k4, k5, k6, k7] = k;
        let (y_stage, y_new) = (&mut y_stage[..n], &mut y_new[..n]);
        macro_rules! stage {
            ($dst:expr, $c:expr, $a0:expr => $k0:expr $(, $a:expr => $kk:expr)*) => {{
                for i in 0..n {
                    y_stage[i] = y[i] + h * ($a0 * $k0[i] $(+ $a * $kk[i])*);
                }
                m.rhs(t + $c * h, y_stage, theta, input, &mut $dst[..n]);
            }};
        }
        stage!(k2, C2, A21 => k1);
        stage!(k3, C3, A31 => k1, A32 => k2);
        stage!(k4, C4, A41 => k1, A42 => k2, A43 => k3);
        stage!(k5, C5, A51 => k1, A52 => k2, A53 => k3, A54 => k4);
        stage!(k6, 1.0, A61 => k1, A62 => k2, A63 => k3, A64 => k4, A65 => k5);
        for i in 0..n {
            y_new[i] = y[i] + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i]);
        }
        let t_new = if clamped { t1 } else { t + h };
        m.rhs(t_new, y_new, theta, input, &mut k7[..n]);
        stats.n_rhs += 6;

        let err = err_norm(
            (0..n).map(|i| {
                let e = h
                    * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
                (e, y[i], y_new[i])
            }),
            opts.rtol,
            opts.atol,
            n,
        );
        if !err.is_finite() {
            // Treat a non-finite error as a hard rejection and shrink strongly.
            stats.n_rejected += 1;
            ws.h = h * FAC_MIN;
            last_rejected = true;
            continue;
        }
        let fac = if err == 0.0 {
            FAC_MAX
        } else {
            (SAFETY * err.powf(-0.2)).clamp(FAC_MIN, FAC_MAX)
        };
        if err <= 1.0 {
            stats.n_accepted += 1;
            y.copy_from_slice(y_new);
            std::mem::swap(k1, k7); // FSAL
            t = t_new;
            let fac = if last_rejected { fac.min(1.0) } else { fac };
            let h_next = h * fac;
            // A step clamped to the segment end says little about the natural step size,
            // so the larger proposal is kept for the next segment.
            ws.h = if clamped { ws.h.max(h_next) } else { h_next };
            last_rejected = false;
        } else {
            stats.n_rejected += 1;
            ws.h = h * fac.min(1.0);
            last_rejected = true;
        }
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(SimError::NonFinite { t: t1 });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rk4<M: Model + ?Sized>(
    m: &M,
    theta: &[f64],
    input: &[f64],
    t0: f64,
    t1: f64,
    y: &mut [f64],
    ws: &mut Workspace,
    step: f64,
    opts: &SolverOpts,
    stats: &mut SolverStats,
) -> Result<(), SimError> {
    let n = y.len();
    let n_steps_f = ((t1 - t0) / step).ceil().max(1.0);
    if n_steps_f > (opts.max_steps - stats.n_accepted.min(opts.max_steps)) as f64 {
        return Err(SimError::MaxStepsExceeded {
            max_steps: opts.max_steps,
            t: t0,
        });
    }
    let n_steps = n_steps_f as usize;
    let h = (t1 - t0) / n_steps as f64;
    let [k1, k2, k3, k4, ..] = &mut ws.k;
    let (k1, k2, k3, k4) = (&mut k1[..n], &mut k2[..n], &mut k3[..n], &mut k4[..n]);
    let y_stage = &mut ws.y_stage[..n];
    for j in 0..n_steps {
        let t = t0 + h * j as f64;
        m.rhs(t, y, theta, input, k1);
        for i in 0..n {
            y_stage[i] = y[i] + 0.5 * h * k1[i];
        }
        m.rhs(t + 0.5 * h, y_stage, theta, input, k2);
        for i in 0..n {
            y_stage[i] = y[i] + 0.5 * h * k2[i];
        }
        m.rhs(t + 0.5 * h, y_stage, theta, input, k3);
        for i in 0..n {
            y_stage[i] = y[i] + h * k3[i];
        }
        m.rhs(t + h, y_stage, theta, input, k4);
        for i in 0..n {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }
    stats.n_accepted += n_steps;
    stats.n_rhs += 4 * n_steps;
    if y.iter().any(|v| !v.is_finite()) {
        return Err(SimError::NonFinite { t: t1 });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn opts_validation() {
        assert!(SolverOpts::default().validate().is_ok());
        assert!(SolverOpts::dopri5(1e-8, 1e-12).validate().is_ok());
        assert!(SolverOpts::rk4(0.01).validate().is_ok());
        for bad in [
            SolverOpts::dopri5(0.0, 1e-9),
            SolverOpts::dopri5(f64::NAN, 1e-9),
            SolverOpts::dopri5(1e-6, -1.0),
            SolverOpts::rk4(0.0),
            SolverOpts {
                max_steps: 0,
                ..Default::default()
            },
            SolverOpts {
                h_init: Some(-1.0),
                ..Default::default()
            },
            SolverOpts {
                h_max: Some(f64::INFINITY),
                ..Default::default()
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }

    /// Butcher-tableau sanity: each row of A sums to its c, and b and b* each sum to 1.
    #[test]
    fn dopri_coefficients_consistent() {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-14;
        assert!(close(A21, C2));
        assert!(close(A31 + A32, C3));
        assert!(close(A41 + A42 + A43, C4));
        assert!(close(A51 + A52 + A53 + A54, C5));
        assert!(close(A61 + A62 + A63 + A64 + A65, 1.0));
        assert!(close(B1 + B3 + B4 + B5 + B6, 1.0));
        assert!(close(E1 + E3 + E4 + E5 + E6 + E7, 0.0));
    }
}
