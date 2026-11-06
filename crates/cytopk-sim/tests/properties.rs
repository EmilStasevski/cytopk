//! Brief section 5.3, tests 2–7 (the parts that apply to the linear IV models of M1).
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use cytopk_sim::analytic::{onecmt_iv_conc, twocmt_macro_rates};
use cytopk_sim::{
    Constraint, Dose, Model, OneCmtIv, Schedule, SimError, SolverOpts, TwoCmtIv, simulate_batch,
    simulate_one,
};
use ndarray::Array2;
use proptest::prelude::*;
use rand::RngExt;

/// `TwoCmtIv` plus two auxiliary states: AUC of the central concentration and the cumulative eliminated amount.
struct TwoCmtAux;

impl Model for TwoCmtAux {
    fn name(&self) -> &'static str {
        "twocmt_iv_aux"
    }
    fn state_names(&self) -> &'static [&'static str] {
        &["A1", "A2", "AUC", "ELIM"]
    }
    fn param_names(&self) -> &'static [&'static str] {
        TwoCmtIv.param_names()
    }
    fn param_constraints(&self) -> &'static [Constraint] {
        TwoCmtIv.param_constraints()
    }
    fn output_names(&self) -> &'static [&'static str] {
        &["C"]
    }
    fn rhs(&self, t: f64, y: &[f64], th: &[f64], input: &[f64], dy: &mut [f64]) {
        TwoCmtIv.rhs(t, &y[..2], th, &input[..2], &mut dy[..2]);
        let c = y[0] / th[1];
        dy[2] = c;
        dy[3] = th[0] * c;
    }
    fn observe(&self, y: &[f64], th: &[f64], out: &mut [f64]) {
        out[0] = y[0] / th[1];
    }
}

const TIGHT: SolverOpts = SolverOpts {
    method: cytopk_sim::Method::Dopri5,
    rtol: 1e-10,
    atol: 1e-12,
    max_steps: 1_000_000,
    h_init: None,
    h_max: None,
};

#[test]
fn mass_balance_with_and_without_elimination() {
    let mut r = rng(10);
    for _ in 0..200 {
        let mut theta = twocmt_theta(&mut r);
        if r.random_bool(0.5) {
            theta[0] = 0.0; // CL = 0: A1 + A2 must equal the dose given so far
        }
        let sched = random_schedule(&mut r);
        let times = times_for(&sched);
        let sol = simulate_one(&TwoCmtAux, &theta, &sched, &times, &TIGHT).unwrap();
        for (j, &t) in times.iter().enumerate() {
            let given: f64 = sched
                .doses()
                .iter()
                .map(|d| match d.duration {
                    None if d.time <= t => d.amount,
                    Some(dur) if d.time <= t => d.amount * ((t - d.time) / dur).min(1.0),
                    _ => 0.0,
                })
                .sum();
            let s = sol.states.row(j);
            let balance = given - s[3] - s[0] - s[1];
            assert!(
                balance.abs() <= 1e-8 * sched.total_amount(),
                "t={t} balance={balance:e}"
            );
            if theta[0] == 0.0 {
                assert_eq!(s[3], 0.0);
            }
        }
    }
}

#[test]
fn auc_to_infinity_equals_dose_over_cl() {
    let mut r = rng(11);
    for _ in 0..100 {
        let theta = twocmt_theta(&mut r);
        let (_, beta) = twocmt_macro_rates(&theta).unwrap();
        let sched = random_schedule(&mut r);
        // 60 terminal half-lives after the last dose leaves a fraction e^{-60 ln2} ~ 1e-18 unobserved.
        let t_end = sched.doses().last().unwrap().end_time() + 60.0 * std::f64::consts::LN_2 / beta;
        let sol = simulate_one(&TwoCmtAux, &theta, &sched, &[t_end], &TIGHT).unwrap();
        let auc = sol.states[[0, 2]];
        let expected = sched.total_amount() / theta[0];
        assert!(
            (auc / expected - 1.0).abs() < 1e-8,
            "auc={auc} expected={expected}"
        );
    }
}

#[test]
fn amounts_nonnegative_and_onecmt_bolus_decays_monotonically() {
    let opts = SolverOpts::dopri5(1e-6, 1e-9);
    let mut r = rng(12);
    let times: Vec<f64> = (0..=400).map(|i| i as f64 * 0.1).collect();
    for _ in 0..200 {
        let theta = twocmt_theta(&mut r);
        let sched = random_schedule(&mut r);
        let sol = simulate_one(&TwoCmtIv, &theta, &sched, &times, &opts).unwrap();
        assert!(sol.states.iter().all(|&a| a >= 0.0));

        let th1 = onecmt_theta(&mut r);
        let bolus = Schedule::new(vec![Dose::bolus(0.0, 100.0, 0)]).unwrap();
        let c = simulate_one(&OneCmtIv, &th1, &bolus, &times, &opts)
            .unwrap()
            .outputs;
        assert!(
            c.column(0)
                .windows(2)
                .into_iter()
                .all(|w| w[1] <= w[0] && w[1] >= 0.0)
        );
    }
}

#[test]
fn output_at_dose_time_includes_the_bolus() {
    let theta = [0.3, 3.0];
    let sched = Schedule::new(vec![Dose::bolus(0.0, 30.0, 0), Dose::bolus(7.0, 30.0, 0)]).unwrap();
    let times = [0.0, 7.0 - 1e-9, 7.0, 7.0, 7.0 + 1e-9];
    let sol = simulate_one(
        &OneCmtIv,
        &theta,
        &sched,
        &times,
        &SolverOpts::dopri5(1e-10, 1e-12),
    )
    .unwrap();
    let c = sol.outputs.column(0);
    assert!(
        (c[0] - 10.0).abs() < 1e-12,
        "first output includes the dose at t=0"
    );
    let jump = c[2] - c[1];
    assert!(
        (jump - 10.0).abs() < 1e-6,
        "jump at t=7 is the bolus: {jump}"
    );
    assert_eq!(c[2], c[3], "repeated output times give identical values");
    let ex = onecmt_iv_conc(&theta, &sched, &times).unwrap();
    for (a, e) in c.iter().zip(&ex) {
        assert!((a / e - 1.0).abs() < 1e-8);
    }
}

#[test]
fn doses_after_last_output_and_empty_inputs() {
    let sched = Schedule::new(vec![Dose::bolus(0.0, 10.0, 0), Dose::bolus(100.0, 1e6, 0)]).unwrap();
    let a = simulate_one(
        &OneCmtIv,
        &[0.3, 3.0],
        &sched,
        &[1.0],
        &SolverOpts::default(),
    )
    .unwrap();
    let only_first = Schedule::new(vec![Dose::bolus(0.0, 10.0, 0)]).unwrap();
    let b = simulate_one(
        &OneCmtIv,
        &[0.3, 3.0],
        &only_first,
        &[1.0],
        &SolverOpts::default(),
    )
    .unwrap();
    assert_eq!(a.outputs, b.outputs);
    let none = simulate_one(
        &OneCmtIv,
        &[0.3, 3.0],
        &Schedule::empty(),
        &[0.0, 5.0],
        &SolverOpts::default(),
    )
    .unwrap();
    assert!(none.outputs.iter().all(|&c| c == 0.0));
    let no_times =
        simulate_one(&OneCmtIv, &[0.3, 3.0], &sched, &[], &SolverOpts::default()).unwrap();
    assert_eq!(no_times.outputs.shape(), &[0, 1]);
    let empty_batch = simulate_batch(
        &TwoCmtIv,
        Array2::zeros((0, 4)).view(),
        &sched,
        &[1.0],
        &SolverOpts::default(),
    )
    .unwrap();
    assert_eq!(empty_batch.shape(), &[0, 1, 1]);
}

#[test]
fn input_validation_errors() {
    let s = Schedule::new(vec![Dose::bolus(0.0, 10.0, 0)]).unwrap();
    let o = SolverOpts::default();
    fn e<T: std::fmt::Debug>(r: Result<T, SimError>) -> SimError {
        r.unwrap_err()
    }
    assert!(matches!(
        e(simulate_one(&OneCmtIv, &[0.3], &s, &[1.0], &o)),
        SimError::ParamCount { .. }
    ));
    assert!(matches!(
        e(simulate_one(&OneCmtIv, &[0.3, -3.0], &s, &[1.0], &o)),
        SimError::InvalidParam { name: "V", .. }
    ));
    assert!(matches!(
        e(simulate_one(&OneCmtIv, &[0.3, 3.0], &s, &[2.0, 1.0], &o)),
        SimError::InvalidTimes { index: 1, .. }
    ));
    assert!(matches!(
        e(simulate_one(&OneCmtIv, &[0.3, 3.0], &s, &[-1.0], &o)),
        SimError::InvalidTimes { index: 0, .. }
    ));
    assert!(matches!(
        e(simulate_one(&OneCmtIv, &[0.3, 3.0], &s, &[f64::NAN], &o)),
        SimError::InvalidTimes { .. }
    ));
    let periph = Schedule::new(vec![Dose::bolus(0.0, 10.0, 1)]).unwrap();
    assert!(matches!(
        e(simulate_one(&OneCmtIv, &[0.3, 3.0], &periph, &[1.0], &o)),
        SimError::DoseStateOutOfRange { state: 1, .. }
    ));
    assert!(simulate_one(&TwoCmtIv, &[0.3, 3.0, 0.5, 3.0], &periph, &[1.0], &o).is_ok());
    let tiny_budget = SolverOpts {
        max_steps: 3,
        ..SolverOpts::dopri5(1e-10, 1e-12)
    };
    assert!(matches!(
        e(simulate_one(
            &OneCmtIv,
            &[0.3, 3.0],
            &s,
            &[30.0],
            &tiny_budget
        )),
        SimError::MaxStepsExceeded { .. }
    ));
    assert!(matches!(
        e(simulate_one(
            &OneCmtIv,
            &[0.3, 3.0],
            &s,
            &[1.0],
            &SolverOpts::rk4(-1.0)
        )),
        SimError::InvalidOptions(_)
    ));

    let mut thetas = Array2::from_elem((5, 2), 1.0);
    thetas[[3, 1]] = 0.0;
    thetas[[4, 1]] = 0.0;
    match e(simulate_batch(&OneCmtIv, thetas.view(), &s, &[1.0], &o)) {
        SimError::InBatch { index, source } => {
            assert_eq!(index, 3, "the lowest failing row is reported");
            assert!(matches!(*source, SimError::InvalidParam { .. }));
        }
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        e(simulate_batch(
            &OneCmtIv,
            Array2::zeros((2, 3)).view(),
            &s,
            &[1.0],
            &o
        )),
        SimError::ParamCount { got: 3, .. }
    ));
}

fn arb_dose() -> impl Strategy<Value = Dose> {
    (0.0..30.0f64, 1.0..500.0f64, prop::option::of(0.01..2.0f64)).prop_map(
        |(time, amount, duration)| Dose {
            time,
            amount,
            state: 0,
            duration,
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// The order of the dose list does not matter.
    #[test]
    fn unsorted_doses_give_same_schedule(mut doses in prop::collection::vec(arb_dose(), 0..8), seed in any::<u64>()) {
        let sorted = Schedule::new(doses.clone()).unwrap();
        prop_assert!(sorted.doses().windows(2).all(|w| w[0].time <= w[1].time));
        // Shuffle deterministically.
        let n = doses.len();
        for i in (1..n).rev() {
            let j = (seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64) % (i as u64 + 1)) as usize;
            doses.swap(i, j);
        }
        let shuffled = Schedule::new(doses).unwrap();
        let times: Vec<f64> = (0..=40).map(|i| i as f64).collect();
        let o = SolverOpts::dopri5(1e-8, 1e-12);
        let a = simulate_one(&TwoCmtIv, &[0.3, 3.0, 0.5, 4.0], &sorted, &times, &o).unwrap();
        let b = simulate_one(&TwoCmtIv, &[0.3, 3.0, 0.5, 4.0], &shuffled, &times, &o).unwrap();
        for (x, y) in a.outputs.iter().zip(b.outputs.iter()) {
            prop_assert!((x - y).abs() <= 1e-12 * x.abs().max(1.0));
        }
    }

    /// Invalid dose fields are rejected, never panic.
    #[test]
    fn invalid_doses_rejected(time in prop::num::f64::ANY, amount in prop::num::f64::ANY, dur in prop::option::of(prop::num::f64::ANY)) {
        let d = Dose { time, amount, state: 0, duration: dur };
        let valid = time.is_finite() && time >= 0.0 && amount.is_finite() && amount >= 0.0
            && dur.is_none_or(|x| x.is_finite() && x > 0.0 && (time + x).is_finite());
        prop_assert_eq!(Schedule::new(vec![d]).is_ok(), valid);
    }

    /// An infusion of vanishing duration approaches the bolus: the difference after the infusion is O(duration).
    #[test]
    fn short_infusion_approaches_bolus(theta in (0.05..2.0f64, 1.0..20.0f64, 0.05..5.0f64, 1.0..20.0f64), dur_exp in 3..8i32) {
        let theta = [theta.0, theta.1, theta.2, theta.3];
        let dur = 10f64.powi(-dur_exp);
        let times = [0.5, 1.0, 7.0];
        let o = SolverOpts::dopri5(1e-10, 1e-14);
        let inf = simulate_one(&TwoCmtIv, &theta, &Schedule::new(vec![Dose::infusion(0.0, 100.0, dur, 0)]).unwrap(), &times, &o).unwrap();
        let bol = simulate_one(&TwoCmtIv, &theta, &Schedule::new(vec![Dose::bolus(0.0, 100.0, 0)]).unwrap(), &times, &o).unwrap();
        // |d/dt C| <= (CL+Q)/V1 * C0, so a shift of at most dur gives a relative difference <= rate * dur (+ tolerance).
        let k = (theta[0] + theta[2]) / theta[1] + theta[2] / theta[3];
        for (a, b) in inf.outputs.iter().zip(bol.outputs.iter()) {
            prop_assert!((a - b).abs() <= (k * dur + 1e-8) * b.abs(), "a={} b={} dur={}", a, b, dur);
        }
    }

    /// Linear models: scaling all doses by s scales the amounts by s (up to solver tolerance).
    #[test]
    fn linear_in_dose(theta in (0.05..2.0f64, 1.0..20.0f64, 0.05..5.0f64, 1.0..20.0f64), scale in 0.01..100.0f64, doses in prop::collection::vec(arb_dose(), 1..4)) {
        let theta = [theta.0, theta.1, theta.2, theta.3];
        let s1 = Schedule::new(doses.clone()).unwrap();
        let s2 = Schedule::new(doses.into_iter().map(|d| Dose { amount: d.amount * scale, ..d }).collect()).unwrap();
        let times: Vec<f64> = (1..=35).map(|i| i as f64).collect();
        let o = SolverOpts::dopri5(1e-10, 1e-14);
        let a = simulate_one(&TwoCmtIv, &theta, &s1, &times, &o).unwrap();
        let b = simulate_one(&TwoCmtIv, &theta, &s2, &times, &o).unwrap();
        let peak = a.states.iter().cloned().fold(0.0, f64::max);
        for (x, y) in a.states.iter().zip(b.states.iter()) {
            prop_assert!((x * scale - y).abs() <= 1e-7 * peak * scale, "x*s={} y={}", x * scale, y);
        }
    }
}

/// Models larger than the solver's stack-buffer capacity are rejected with a clear error.
#[test]
fn too_many_states_is_an_error() {
    struct Big;
    static NAMES: [&str; cytopk_sim::MAX_STATES + 1] = ["x"; cytopk_sim::MAX_STATES + 1];
    impl Model for Big {
        fn name(&self) -> &'static str {
            "big"
        }
        fn state_names(&self) -> &'static [&'static str] {
            &NAMES
        }
        fn param_names(&self) -> &'static [&'static str] {
            &[]
        }
        fn param_constraints(&self) -> &'static [Constraint] {
            &[]
        }
        fn output_names(&self) -> &'static [&'static str] {
            &["y"]
        }
        fn rhs(&self, _: f64, _: &[f64], _: &[f64], _: &[f64], dy: &mut [f64]) {
            dy.fill(0.0);
        }
        fn observe(&self, y: &[f64], _: &[f64], out: &mut [f64]) {
            out[0] = y[0];
        }
    }
    let r = simulate_one(
        &Big,
        &[],
        &Schedule::empty(),
        &[1.0],
        &SolverOpts::default(),
    );
    assert!(matches!(
        r,
        Err(SimError::TooManyStates {
            n_states: 33,
            max: 32,
            ..
        })
    ));
}
