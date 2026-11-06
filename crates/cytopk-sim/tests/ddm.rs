//! Drift-diffusion simulator: analytic choice probability, determinism across threads, input validation.
#![allow(clippy::unwrap_used)]

use cytopk_sim::Parallelism;
use cytopk_sim::ddm::{DdmOpts, simulate_ddm};
use ndarray::Array2;

/// Continuous-time probability of hitting the upper boundary first: `(1 - e^{-2 v z a}) / (1 - e^{-2 v a})`.
fn p_upper(v: f64, a: f64, za: f64) -> f64 {
    (1.0 - (-2.0 * v * za).exp()) / (1.0 - (-2.0 * v * a).exp())
}

/// Broadie–Glasserman–Kou continuity correction for discrete monitoring: each boundary is moved outward by
/// beta * sqrt(dt), with beta = -zeta(1/2) / sqrt(2 pi) ≈ 0.5826.
fn p_upper_corrected(v: f64, a: f64, z: f64, dt: f64) -> f64 {
    let shift = 0.582_597_157_939_010_6 * dt.sqrt();
    p_upper(v, a + 2.0 * shift, z * a + shift)
}

/// Acceptance rule, fixed before looking at results: |p_hat - P_corr| <= 4 SE + 0.1 |P - P_corr|. The 10 % term
/// allows for the correction being exact only to first order for two barriers.
#[test]
fn choice_probability_matches_corrected_analytic_value() {
    let opts = DdmOpts::default();
    let n = 400_000;
    for &(v, a, z) in &[(1.0, 1.5, 0.4), (-0.7, 2.0, 0.6), (2.0, 1.0, 0.5)] {
        let theta = Array2::from_shape_fn((n, 4), |(_, j)| [v, a, z, 0.3][j]);
        let (rt, ch) = simulate_ddm(theta.view(), 42, &opts, Parallelism::Auto).unwrap();
        assert!(ch.iter().all(|&c| c != -1), "no timeouts expected");
        assert!(rt.iter().all(|&t| t > 0.3));
        let p_hat = ch.iter().filter(|&&c| c == 1).count() as f64 / n as f64;
        let (p, pc) = (p_upper(v, a, z * a), p_upper_corrected(v, a, z, opts.dt));
        let se = (pc * (1.0 - pc) / n as f64).sqrt();
        let tol = 4.0 * se + 0.1 * (p - pc).abs();
        println!("v={v} a={a} z={z}: p_hat={p_hat:.5} P={p:.5} P_corr={pc:.5} tol={tol:.5}");
        assert!(
            (p_hat - pc).abs() <= tol,
            "p_hat={p_hat} P_corr={pc} tol={tol}"
        );
    }
}

#[test]
fn bit_identical_across_threads_and_modes() {
    let n = 5000;
    let theta = Array2::from_shape_fn((n, 4), |(i, j)| {
        let f = i as f64 / n as f64;
        [-2.0 + 4.0 * f, 0.8 + 1.2 * f, 0.3 + 0.4 * f, 0.1 + 0.4 * f][j]
    });
    let opts = DdmOpts::default();
    let reference = simulate_ddm(theta.view(), 9, &opts, Parallelism::Serial).unwrap();
    let all = std::thread::available_parallelism().map_or(4, |n| n.get());
    for k in [1, 2, all] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(k)
            .build()
            .unwrap();
        let r = pool.install(|| simulate_ddm(theta.view(), 9, &opts, Parallelism::Auto).unwrap());
        assert_eq!(r.1, reference.1);
        assert!(
            r.0.iter()
                .zip(&reference.0)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
    let other_seed = simulate_ddm(theta.view(), 10, &opts, Parallelism::Auto).unwrap();
    assert_ne!(other_seed.0, reference.0);
    // A prefix of the batch gives the same values as the full batch (stream depends only on (seed, index)).
    let head = simulate_ddm(
        theta.slice(ndarray::s![..100, ..]),
        9,
        &opts,
        Parallelism::Auto,
    )
    .unwrap();
    assert_eq!(head.1[..], reference.1[..100]);
}

#[test]
fn ddm_input_validation() {
    let ok = Array2::from_shape_vec((1, 4), vec![1.0, 1.0, 0.5, 0.3]).unwrap();
    let opts = DdmOpts::default();
    assert!(simulate_ddm(ok.view(), 0, &opts, Parallelism::Auto).is_ok());
    for bad in [
        [f64::NAN, 1.0, 0.5, 0.3],
        [1.0, 0.0, 0.5, 0.3],
        [1.0, 1.0, 0.0, 0.3],
        [1.0, 1.0, 1.0, 0.3],
        [1.0, 1.0, 0.5, -0.1],
    ] {
        let th = Array2::from_shape_vec((1, 4), bad.to_vec()).unwrap();
        assert!(
            simulate_ddm(th.view(), 0, &opts, Parallelism::Auto).is_err(),
            "{bad:?}"
        );
    }
    assert!(simulate_ddm(Array2::zeros((1, 3)).view(), 0, &opts, Parallelism::Auto).is_err());
    assert!(
        simulate_ddm(
            ok.view(),
            0,
            &DdmOpts {
                dt: 0.0,
                max_t: 10.0
            },
            Parallelism::Auto
        )
        .is_err()
    );
    let empty = simulate_ddm(Array2::zeros((0, 4)).view(), 0, &opts, Parallelism::Auto).unwrap();
    assert!(empty.0.is_empty());
}
