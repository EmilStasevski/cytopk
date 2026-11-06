//! Tests for the nonlinear M2 models: Michaelis–Menten PK (`twocmt_mm_iv`) and indirect-response PD
//! (`twocmt_iv_idr`). These models have no general closed form, so the tests use conservation laws, exact limits,
//! invariants, comparison properties and determinism. Tight-tolerance SciPy references live in
//! `python/tests/test_reference_scipy.py`.
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use cytopk_sim::analytic::twocmt_iv_conc;
use cytopk_sim::{
    Constraint, Dose, Method, Model, Schedule, SolverOpts, TwoCmtIv, TwoCmtIvIdr, TwoCmtMmIv,
    simulate_batch, simulate_one,
};
use ndarray::Array2;
use proptest::prelude::*;

const TIGHT: SolverOpts = SolverOpts {
    method: Method::Dopri5,
    rtol: 1e-10,
    atol: 1e-12,
    max_steps: 1_000_000,
    h_init: None,
    h_max: None,
};

fn mm_theta(r: &mut rand_chacha::ChaCha8Rng) -> [f64; 5] {
    [
        log_uniform(r, 5.0, 200.0), // Vmax, mg/day
        log_uniform(r, 0.5, 50.0),  // Km, mg/L
        log_uniform(r, 1.0, 20.0),
        log_uniform(r, 0.05, 5.0),
        log_uniform(r, 1.0, 20.0),
    ]
}

fn idr_theta(r: &mut rand_chacha::ChaCha8Rng) -> [f64; 7] {
    let pk = twocmt_theta(r);
    [
        pk[0],
        pk[1],
        pk[2],
        pk[3],
        log_uniform(r, 0.05, 3.0), // kout
        log_uniform(r, 0.1, 10.0), // Emax
        log_uniform(r, 0.5, 50.0), // EC50
    ]
}

/// Amount given up to time t (bolus included at its time; infusions pro rata).
fn given(s: &Schedule, t: f64) -> f64 {
    s.doses()
        .iter()
        .map(|d| match d.duration {
            None if d.time <= t => d.amount,
            Some(dur) if d.time <= t => d.amount * ((t - d.time) / dur).min(1.0),
            _ => 0.0,
        })
        .sum()
}

/// `TwoCmtMmIv` plus the cumulative eliminated amount as an auxiliary state.
struct MmWithElim;
impl Model for MmWithElim {
    fn name(&self) -> &'static str {
        "twocmt_mm_iv_elim"
    }
    fn state_names(&self) -> &'static [&'static str] {
        &["A1", "A2", "ELIM"]
    }
    fn param_names(&self) -> &'static [&'static str] {
        TwoCmtMmIv.param_names()
    }
    fn param_constraints(&self) -> &'static [Constraint] {
        TwoCmtMmIv.param_constraints()
    }
    fn output_names(&self) -> &'static [&'static str] {
        &["C"]
    }
    fn rhs(&self, t: f64, y: &[f64], th: &[f64], input: &[f64], dy: &mut [f64]) {
        TwoCmtMmIv.rhs(t, &y[..2], th, &input[..2], &mut dy[..2]);
        let c = y[0] / th[2];
        dy[2] = th[0] * c / (th[1] + c);
    }
    fn observe(&self, y: &[f64], th: &[f64], out: &mut [f64]) {
        out[0] = y[0] / th[2];
    }
}

#[test]
fn mm_mass_balance_and_no_elimination_limit() {
    let mut r = rng(30);
    for k in 0..200 {
        let mut theta = mm_theta(&mut r);
        if k % 4 == 0 {
            theta[0] = 0.0; // Vmax = 0: nothing is eliminated
        }
        let sched = random_schedule(&mut r);
        let times = times_for(&sched);
        let sol = simulate_one(&MmWithElim, &theta, &sched, &times, &TIGHT).unwrap();
        for (j, &t) in times.iter().enumerate() {
            let s = sol.states.row(j);
            let balance = given(&sched, t) - s[0] - s[1] - s[2];
            assert!(
                balance.abs() <= 1e-8 * sched.total_amount(),
                "t={t} balance={balance:e}"
            );
            // Explicit RK does not preserve positivity exactly. Once the drug is gone, amounts may dip
            // below zero by noise below the absolute tolerance (observed: -1.2e-14 mg with atol = 1e-12).
            assert!(s.iter().all(|&a| a >= -TIGHT.atol), "t={t} s={s:?}");
            if theta[0] == 0.0 {
                assert_eq!(s[2], 0.0);
            }
        }
    }
}

/// Km >> C: Michaelis–Menten elimination is linear with CL = Vmax / Km. Compare against the 2-cmt closed form.
#[test]
fn mm_linear_limit_matches_closed_form() {
    let mut r = rng(31);
    for _ in 0..100 {
        let pk = twocmt_theta(&mut r);
        let km = 1e9;
        let theta = [pk[0] * km, km, pk[1], pk[2], pk[3]];
        let sched = random_schedule(&mut r);
        let times = times_for(&sched);
        let sol = simulate_one(&TwoCmtMmIv, &theta, &sched, &times, &TIGHT).unwrap();
        let ex = twocmt_iv_conc(&pk, &sched, &times).unwrap();
        // C <= ~1000 mg / 1 L, so C / Km <= 1e-6 relative deviation from linearity.
        let (rel, small) = rel_err(&sol.outputs.column(0).to_vec(), &ex, 1e-6);
        assert!(rel < 2e-6 && small < 1e-8, "rel={rel:e} small={small:e}");
    }
}

/// Km -> 0 with Q = 0: zero-order elimination, so A1(t) = D - Vmax * t for a bolus while A1 > 0.
#[test]
fn mm_zero_order_limit() {
    let (vmax, d, v1) = (40.0, 200.0, 3.0);
    let theta = [vmax, 1e-9, v1, 0.0, 3.0];
    let sched = Schedule::new(vec![Dose::bolus(0.0, d, 0)]).unwrap();
    let times = [0.5, 1.0, 2.0, 4.0]; // the drug runs out at t = D / Vmax = 5
    let sol = simulate_one(&TwoCmtMmIv, &theta, &sched, &times, &TIGHT).unwrap();
    for (j, &t) in times.iter().enumerate() {
        let a1 = sol.states[[j, 0]];
        assert!(
            (a1 - (d - vmax * t)).abs() < 1e-6,
            "t={t}: {a1} vs {}",
            d - vmax * t
        );
    }
}

#[test]
fn idr_pk_part_matches_twocmt_and_r_is_bounded() {
    let mut r = rng(32);
    let opts = SolverOpts::dopri5(1e-10, 1e-12);
    for _ in 0..200 {
        let theta = idr_theta(&mut r);
        let sched = random_schedule(&mut r);
        let times = times_for(&sched);
        let sol = simulate_one(&TwoCmtIvIdr, &theta, &sched, &times, &opts).unwrap();
        let ex = twocmt_iv_conc(&theta[..4], &sched, &times).unwrap();
        let (rel, _) = rel_err(&sol.outputs.column(0).to_vec(), &ex, 1e-6);
        assert!(rel < 1e-7, "PK part rel err {rel:e}");
        let emax = theta[5];
        for j in 0..times.len() {
            let rr = sol.states[[j, 2]];
            // The stimulation lies in [1, 1 + Emax] and R(0) = 1, so R stays in [1, 1 + Emax].
            assert!(
                rr >= 1.0 - 1e-9 && rr <= 1.0 + emax + 1e-9,
                "R={rr} Emax={emax}"
            );
            let p = sol.outputs[[j, 1]];
            assert!((p - rr / (rr + 9.0)).abs() < 1e-15);
            assert!((0.1 - 1e-12..1.0).contains(&p));
            assert!(sol.states[[j, 0]] >= 0.0 && sol.states[[j, 1]] >= 0.0);
        }
    }
}

#[test]
fn idr_without_drug_effect_stays_at_baseline() {
    let theta = [0.2, 3.0, 0.5, 3.0, 0.3, 0.0, 5.0]; // Emax = 0
    let sched = Schedule::repeated(Dose::infusion(0.0, 800.0, 1.0 / 24.0, 0), 7.0, 4).unwrap();
    let sol = simulate_one(&TwoCmtIvIdr, &theta, &sched, &[1.0, 10.0, 28.0], &TIGHT).unwrap();
    assert!(sol.states.column(2).iter().all(|&r| r == 1.0));
    assert!(sol.outputs.column(1).iter().all(|&p| p == 0.1));
    // Without any dose, the baseline also holds when Emax > 0.
    let theta = [0.2, 3.0, 0.5, 3.0, 0.3, 2.0, 5.0];
    let none = simulate_one(
        &TwoCmtIvIdr,
        &theta,
        &Schedule::empty(),
        &[0.0, 28.0],
        &TIGHT,
    )
    .unwrap();
    assert!(none.outputs.column(1).iter().all(|&p| p == 0.1));
}

/// Response saturates in dose: a higher dose never lowers R, and R approaches 1 + Emax for a huge dose.
#[test]
fn idr_response_monotone_in_dose_and_saturates() {
    let theta = [0.2, 3.0, 0.5, 3.0, 0.3, 2.0, 5.0];
    let times: Vec<f64> = (1..=28).map(f64::from).collect();
    let mut prev: Option<Vec<f64>> = None;
    for dose in [10.0, 50.0, 200.0, 800.0, 1e6] {
        let s = Schedule::new(vec![Dose::infusion(0.0, dose, 1.0 / 24.0, 0)]).unwrap();
        let r = simulate_one(&TwoCmtIvIdr, &theta, &s, &times, &TIGHT)
            .unwrap()
            .states
            .column(2)
            .to_vec();
        if let Some(p) = &prev {
            assert!(
                r.iter().zip(p).all(|(a, b)| a >= b),
                "R not monotone in dose"
            );
        }
        prev = Some(r);
    }
    let r_max = prev.unwrap().into_iter().fold(0.0, f64::max);
    assert!(r_max > 2.9 && r_max <= 3.0, "R max {r_max}");
}

#[test]
fn nonlinear_batches_deterministic_across_threads() {
    let mut r = rng(33);
    let n = 1000;
    let mut mm = Array2::zeros((n, 5));
    for mut row in mm.outer_iter_mut() {
        row.assign(&ndarray::arr1(&mm_theta(&mut r)));
    }
    let mut idr = Array2::zeros((n, 7));
    for mut row in idr.outer_iter_mut() {
        row.assign(&ndarray::arr1(&idr_theta(&mut r)));
    }
    let sched = Schedule::repeated(Dose::infusion(0.0, 200.0, 1.0 / 24.0, 0), 7.0, 4).unwrap();
    let times = [0.04, 0.5, 1.0, 2.0, 4.0, 7.0, 14.0, 21.0, 28.0];
    let opts = SolverOpts::dopri5(1e-6, 1e-9);
    let all = std::thread::available_parallelism().map_or(4, |n| n.get());
    let pool = |k: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(k)
            .build()
            .unwrap()
    };
    let bits = |a: &ndarray::Array3<f64>| a.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let ref_mm =
        pool(1).install(|| simulate_batch(&TwoCmtMmIv, mm.view(), &sched, &times, &opts).unwrap());
    let ref_idr = pool(1)
        .install(|| simulate_batch(&TwoCmtIvIdr, idr.view(), &sched, &times, &opts).unwrap());
    assert_eq!(ref_idr.shape(), &[n, 9, 2]);
    for k in [2, all] {
        let a = pool(k)
            .install(|| simulate_batch(&TwoCmtMmIv, mm.view(), &sched, &times, &opts).unwrap());
        let b = pool(k)
            .install(|| simulate_batch(&TwoCmtIvIdr, idr.view(), &sched, &times, &opts).unwrap());
        assert_eq!(bits(&a), bits(&ref_mm), "mm differs with {k} threads");
        assert_eq!(bits(&b), bits(&ref_idr), "idr differs with {k} threads");
    }
}

#[test]
fn mm_auc_more_than_proportional_to_dose() {
    // Saturation: doubling a dose that saturates elimination more than doubles the exposure.
    let theta = [20.0, 5.0, 3.0, 0.5, 3.0];
    let auc = |dose: f64| {
        let s = Schedule::new(vec![Dose::bolus(0.0, dose, 0)]).unwrap();
        let times: Vec<f64> = (0..=4000).map(|i| i as f64 * 0.025).collect();
        let c = simulate_one(&TwoCmtMmIv, &theta, &s, &times, &TIGHT)
            .unwrap()
            .outputs;
        c.column(0)
            .windows(2)
            .into_iter()
            .map(|w| 0.0125 * (w[0] + w[1]))
            .sum::<f64>()
    };
    let (a1, a2) = (auc(100.0), auc(200.0));
    assert!(a2 / a1 > 2.2, "AUC ratio {}", a2 / a1);
    // Linear model: exactly proportional (to solver tolerance).
    let lin = |dose: f64| {
        let s = Schedule::new(vec![Dose::bolus(0.0, dose, 0)]).unwrap();
        simulate_one(&TwoCmtIv, &[1.0, 3.0, 0.5, 3.0], &s, &[1.0, 5.0], &TIGHT)
            .unwrap()
            .outputs
    };
    let (l1, l2) = (lin(100.0), lin(200.0));
    for (a, b) in l1.iter().zip(l2.iter()) {
        assert!((b / a - 2.0).abs() < 1e-8);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Comparison principle for the cooperative MM system: for k >= 1, C(k * dose) >= k * C(dose) at all
    /// times, because elimination per unit amount falls as concentration rises.
    #[test]
    fn mm_scaled_dose_dominates_scaled_response(seed in any::<u64>(), k in 1.5..20.0f64) {
        let mut r = rng(seed);
        let theta = mm_theta(&mut r);
        let sched = random_schedule(&mut r);
        let scaled = Schedule::new(sched.doses().iter().map(|d| Dose { amount: d.amount * k, ..*d }).collect()).unwrap();
        let times: Vec<f64> = (1..=35).map(f64::from).collect();
        let a = simulate_one(&TwoCmtMmIv, &theta, &sched, &times, &TIGHT).unwrap().outputs;
        let b = simulate_one(&TwoCmtMmIv, &theta, &scaled, &times, &TIGHT).unwrap().outputs;
        // Only above the solver's noise floor: near zero both values are dominated by the absolute tolerance
        // (regression case: 4e-13 vs 2e-12 mg/L at t = 35 d).
        let floor = 1e-8 * b.iter().cloned().fold(0.0, f64::max);
        for (x, y) in a.iter().zip(b.iter()) {
            if k * x > floor {
                prop_assert!(*y >= k * x * (1.0 - 1e-7), "C(kD)={} < k*C(D)={}", y, k * x);
            }
        }
        // And strictly more than proportional somewhere, unless elimination never saturates.
        let strict = a.iter().zip(b.iter()).any(|(x, y)| *y > k * x * (1.0 + 1e-6));
        let peak_c = a.iter().cloned().fold(0.0, f64::max) * k;
        prop_assert!(strict || peak_c < 1e-3 * theta[1], "no saturation effect seen");
    }

    /// Invalid parameters for the new models are rejected and never panic.
    #[test]
    fn nonlinear_param_validation(v in prop::collection::vec(prop::num::f64::ANY, 7)) {
        let s = Schedule::new(vec![Dose::bolus(0.0, 10.0, 0)]).unwrap();
        let o = SolverOpts { max_steps: 2000, ..SolverOpts::default() };
        let _ = simulate_one(&TwoCmtMmIv, &v[..5], &s, &[1.0], &o);
        let _ = simulate_one(&TwoCmtIvIdr, &v, &s, &[1.0], &o);
        let ok_mm = TwoCmtMmIv.check_params(&v[..5]).is_ok();
        prop_assert_eq!(ok_mm, v[0] >= 0.0 && v[1] > 0.0 && v[2] > 0.0 && v[3] >= 0.0 && v[4] > 0.0
            && v[..5].iter().all(|x| x.is_finite()));
    }
}
