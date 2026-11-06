#![allow(dead_code, clippy::unwrap_used)]

use cytopk_sim::{Dose, Schedule};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

pub fn log_uniform(r: &mut ChaCha8Rng, lo: f64, hi: f64) -> f64 {
    (r.random_range(lo.ln()..hi.ln())).exp()
}

/// Random 1-cmt parameters `[CL, V]`, log-uniform around the brief's illustrative values (wide).
pub fn onecmt_theta(r: &mut ChaCha8Rng) -> [f64; 2] {
    [log_uniform(r, 0.05, 2.0), log_uniform(r, 1.0, 20.0)]
}

/// Random 2-cmt parameters `[CL, V1, Q, V2]`.
pub fn twocmt_theta(r: &mut ChaCha8Rng) -> [f64; 4] {
    [
        log_uniform(r, 0.05, 2.0),
        log_uniform(r, 1.0, 20.0),
        log_uniform(r, 0.05, 5.0),
        log_uniform(r, 1.0, 20.0),
    ]
}

/// 1–4 doses into the central compartment at weekly slots, each a bolus or an infusion of 1 h to 1 day.
pub fn random_schedule(r: &mut ChaCha8Rng) -> Schedule {
    let n = r.random_range(1..=4);
    let doses = (0..n)
        .map(|i| {
            let time = 7.0 * i as f64 + if i > 0 { r.random_range(0.0..2.0) } else { 0.0 };
            let amount = log_uniform(r, 10.0, 1000.0);
            if r.random_bool(0.5) {
                Dose::bolus(time, amount, 0)
            } else {
                Dose::infusion(time, amount, r.random_range(1.0 / 24.0..1.0), 0)
            }
        })
        .collect();
    Schedule::new(doses).unwrap()
}

/// Sampling grid from the brief, extended to 35 days, plus every dose start and end time
/// (outputs exactly at discontinuities).
pub fn times_for(s: &Schedule) -> Vec<f64> {
    let mut t = vec![
        0.0, 0.04, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0, 14.0, 21.0, 28.0, 35.0,
    ];
    for d in s.doses() {
        t.push(d.time);
        t.push(d.end_time());
    }
    t.sort_by(f64::total_cmp);
    t
}

/// Max relative error over points where the exact value is above `floor_frac * max(exact)`, and
/// the max absolute error (scaled by `max(exact)`) over the remaining points.
pub fn rel_err(num: &[f64], exact: &[f64], floor_frac: f64) -> (f64, f64) {
    let peak = exact.iter().cloned().fold(0.0, f64::max);
    let mut rel: f64 = 0.0;
    let mut small: f64 = 0.0;
    for (&a, &e) in num.iter().zip(exact) {
        if e > floor_frac * peak {
            rel = rel.max((a - e).abs() / e.abs());
        } else if peak > 0.0 {
            small = small.max((a - e).abs() / peak);
        }
    }
    (rel, small)
}
