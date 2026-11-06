//! Batch results are bit-identical for any thread count, across repeated runs, and match `simulate_one`.
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use cytopk_sim::{Dose, Schedule, SolverOpts, TwoCmtIv, simulate_batch, simulate_one};
use ndarray::{Array2, ShapeBuilder};

fn thetas(n: usize) -> Array2<f64> {
    let mut r = rng(20);
    let mut a = Array2::zeros((n, 4));
    for mut row in a.outer_iter_mut() {
        row.assign(&ndarray::arr1(&twocmt_theta(&mut r)));
    }
    a
}

#[test]
fn bit_identical_across_thread_counts_and_runs() {
    let th = thetas(2000);
    let sched = Schedule::repeated(Dose::infusion(0.0, 200.0, 1.0 / 24.0, 0), 7.0, 4).unwrap();
    let times = [0.04, 0.5, 1.0, 2.0, 4.0, 7.0, 14.0, 21.0, 28.0];
    let opts = SolverOpts::dopri5(1e-6, 1e-9);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| simulate_batch(&TwoCmtIv, th.view(), &sched, &times, &opts).unwrap())
    };
    let all = std::thread::available_parallelism().map_or(4, |n| n.get());
    let reference = run(1);
    assert_eq!(reference.shape(), &[2000, 9, 1]);
    for threads in [1, 2, all] {
        let out = run(threads);
        // Compare bit patterns, not values: this also catches -0.0 vs 0.0 and NaN payloads.
        let same = out
            .iter()
            .zip(reference.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "results differ with {threads} threads");
    }
    // The batch path equals the single-subject path, row by row.
    for i in [0, 999, 1999] {
        let one = simulate_one(
            &TwoCmtIv,
            th.row(i).as_slice().unwrap(),
            &sched,
            &times,
            &opts,
        )
        .unwrap();
        assert_eq!(
            one.outputs.column(0),
            reference.slice(ndarray::s![i, .., 0])
        );
    }
}

#[test]
fn non_contiguous_theta_views_work() {
    // A Fortran-order (column-major) parameter matrix must give the same results.
    let th = thetas(50);
    let mut f = Array2::zeros((50, 4).f());
    f.assign(&th);
    let sched = Schedule::new(vec![Dose::bolus(0.0, 100.0, 0)]).unwrap();
    let opts = SolverOpts::default();
    let a = simulate_batch(&TwoCmtIv, th.view(), &sched, &[1.0, 2.0], &opts).unwrap();
    let b = simulate_batch(&TwoCmtIv, f.view(), &sched, &[1.0, 2.0], &opts).unwrap();
    assert_eq!(a, b);
}

#[test]
fn serial_path_matches_parallel_path_and_errors_identically() {
    use cytopk_sim::{MIN_PARALLEL_BATCH, Parallelism, simulate_batch_with};
    let sched = Schedule::repeated(Dose::infusion(0.0, 200.0, 1.0 / 24.0, 0), 7.0, 4).unwrap();
    let times = [0.5, 1.0, 7.0, 14.0, 28.0];
    let opts = SolverOpts::dopri5(1e-6, 1e-9);
    for n in [1, MIN_PARALLEL_BATCH - 1, MIN_PARALLEL_BATCH, 500] {
        let th = thetas(n);
        let s = simulate_batch_with(
            &TwoCmtIv,
            th.view(),
            &sched,
            &times,
            &opts,
            Parallelism::Serial,
        )
        .unwrap();
        let p = simulate_batch_with(
            &TwoCmtIv,
            th.view(),
            &sched,
            &times,
            &opts,
            Parallelism::Auto,
        )
        .unwrap();
        let bits = |a: &ndarray::Array3<f64>| a.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&s), bits(&p), "n = {n}");
    }
    // A solver failure in several rows reports the lowest index in both modes.
    let mut th = thetas(50);
    th.row_mut(7)
        .assign(&ndarray::arr1(&[1e6, 1e-3, 1e6, 1e-3]));
    th.row_mut(30)
        .assign(&ndarray::arr1(&[1e6, 1e-3, 1e6, 1e-3]));
    let tiny = SolverOpts {
        max_steps: 5000,
        ..opts
    };
    for par in [Parallelism::Serial, Parallelism::Auto] {
        match simulate_batch_with(&TwoCmtIv, th.view(), &sched, &times, &tiny, par) {
            Err(cytopk_sim::SimError::InBatch { index, .. }) => assert_eq!(index, 7, "{par:?}"),
            other => panic!("{par:?}: {other:?}"),
        }
    }
}
