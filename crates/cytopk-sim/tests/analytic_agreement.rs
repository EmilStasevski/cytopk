//! Numerical solution vs closed form, on 1000 random parameter sets and schedules
//! per model. Run with `--nocapture` to see the observed maximum errors.
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use cytopk_sim::analytic::{onecmt_iv_conc, twocmt_iv_conc, twocmt_macro_rates};
use cytopk_sim::{Dose, Model, OneCmtIv, Schedule, SolverOpts, TwoCmtIv, simulate_one};

const N_SETS: usize = 1000;
/// Relative error is measured where C > FLOOR * Cmax. Below that, the error is measured
/// relative to Cmax (see docs/decisions.md).
const FLOOR: f64 = 1e-6;
const TOL: f64 = 1e-6;

fn check_model<M: Model>(
    m: &M,
    seed: u64,
    draw: impl Fn(&mut rand_chacha::ChaCha8Rng) -> Vec<f64>,
    exact: impl Fn(&[f64], &Schedule, &[f64]) -> Vec<f64>,
) {
    let opts = SolverOpts::dopri5(1e-8, 1e-12);
    let mut r = rng(seed);
    let (mut worst_rel, mut worst_small) = (0.0f64, 0.0f64);
    let mut steps = 0;
    for _ in 0..N_SETS {
        let theta = draw(&mut r);
        let sched = random_schedule(&mut r);
        let times = times_for(&sched);
        let sol = simulate_one(m, &theta, &sched, &times, &opts).unwrap();
        let num = sol.outputs.column(0).to_vec();
        let ex = exact(&theta, &sched, &times);
        let (rel, small) = rel_err(&num, &ex, FLOOR);
        assert!(
            rel <= TOL && small <= TOL,
            "{}: theta={theta:?} sched={sched:?} rel={rel:e} small={small:e}",
            m.name()
        );
        worst_rel = worst_rel.max(rel);
        worst_small = worst_small.max(small);
        steps += sol.stats.n_accepted;
    }
    println!(
        "{}: {N_SETS} sets, max rel err {worst_rel:.2e}, max small-value err/Cmax {worst_small:.2e}, mean accepted steps {:.0}",
        m.name(),
        steps as f64 / N_SETS as f64
    );
}

#[test]
fn onecmt_matches_closed_form() {
    check_model(
        &OneCmtIv,
        1,
        |r| onecmt_theta(r).to_vec(),
        |th, s, t| onecmt_iv_conc(th, s, t).unwrap(),
    );
}

#[test]
fn twocmt_matches_closed_form() {
    check_model(
        &TwoCmtIv,
        2,
        |r| twocmt_theta(r).to_vec(),
        |th, s, t| twocmt_iv_conc(th, s, t).unwrap(),
    );
}

/// The closed form itself is checked independently against a very fine fixed-step RK4 solution,
/// so that the two methods do not share a hidden mistake.
#[test]
fn closed_form_matches_fine_rk4() {
    let mut r = rng(3);
    for _ in 0..20 {
        let theta = twocmt_theta(&mut r);
        let sched = random_schedule(&mut r);
        let times = times_for(&sched);
        let sol = simulate_one(
            &TwoCmtIv,
            &theta,
            &sched,
            &times,
            &SolverOpts {
                max_steps: 10_000_000,
                ..SolverOpts::rk4(1e-3)
            },
        )
        .unwrap();
        let ex = twocmt_iv_conc(&theta, &sched, &times).unwrap();
        let (rel, small) = rel_err(&sol.outputs.column(0).to_vec(), &ex, FLOOR);
        assert!(rel < 1e-7 && small < 1e-7, "rel={rel:e} small={small:e}");
    }
}

/// RK4 has global order 4: halving the step should cut the error by about 16.
#[test]
fn rk4_converges_with_order_four() {
    let theta = [0.4, 3.0, 2.0, 5.0];
    let sched = Schedule::new(vec![Dose::bolus(0.0, 100.0, 0)]).unwrap();
    let times = [0.5, 1.0, 3.0];
    let ex = twocmt_iv_conc(&theta, &sched, &times).unwrap();
    let err = |h: f64| {
        let s = simulate_one(&TwoCmtIv, &theta, &sched, &times, &SolverOpts::rk4(h)).unwrap();
        s.outputs
            .column(0)
            .iter()
            .zip(&ex)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0, f64::max)
    };
    let ratio = err(0.05) / err(0.025);
    assert!((13.0..19.0).contains(&ratio), "ratio = {ratio}");
}

/// Tighter tolerances give smaller errors (a sanity check of the step-size controller).
#[test]
fn dopri_error_decreases_with_tolerance() {
    let theta = [0.2, 3.0, 0.5, 3.0];
    let (_, beta) = twocmt_macro_rates(&theta).unwrap();
    let sched = Schedule::repeated(Dose::infusion(0.0, 200.0, 1.0 / 24.0, 0), 7.0, 4).unwrap();
    let times: Vec<f64> = (1..=56).map(|i| i as f64 * 0.5).collect();
    let ex = twocmt_iv_conc(&theta, &sched, &times).unwrap();
    let mut prev = f64::INFINITY;
    for rtol in [1e-4, 1e-6, 1e-8, 1e-10] {
        let s = simulate_one(
            &TwoCmtIv,
            &theta,
            &sched,
            &times,
            &SolverOpts::dopri5(rtol, rtol * 1e-4),
        )
        .unwrap();
        let (rel, _) = rel_err(&s.outputs.column(0).to_vec(), &ex, FLOOR);
        assert!(rel < prev, "rtol={rtol}: {rel:e} !< {prev:e}");
        assert!(rel < 50.0 * rtol, "rtol={rtol}: rel={rel:e}");
        prev = rel;
    }
    assert!(beta > 0.0);
}
