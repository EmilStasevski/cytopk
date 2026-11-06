//! Profiling aid for the go/no-go benchmark scenario (twocmt_iv, 4 weekly 1 h infusions, 20 outputs).
//! `taskset -c 0 cargo run --release -p cytopk-sim --example step_stats`
#![allow(clippy::unwrap_used)]
use cytopk_sim::{Dose, Schedule, SolverOpts, TwoCmtIv, simulate_batch, simulate_one};
use ndarray::Array2;

const TIMES: [f64; 20] = [
    0.04, 0.5, 1., 2., 4., 7., 7.04, 8., 10., 14., 14.04, 15., 17., 21., 21.04, 22., 24., 28., 31.,
    35.,
];

fn us_per_sim(times: &[f64], sched: &Schedule, opts: &SolverOpts) -> f64 {
    let n = 20_000;
    let thetas = Array2::from_shape_fn((n, 4), |(i, j)| [0.2 + 1e-6 * i as f64, 3.0, 0.5, 3.0][j]);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    (0..7)
        .map(|_| {
            let t0 = std::time::Instant::now();
            std::hint::black_box(
                pool.install(|| {
                    simulate_batch(&TwoCmtIv, thetas.view(), sched, times, opts).unwrap()
                }),
            );
            t0.elapsed().as_secs_f64() / n as f64 * 1e6
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let sched = Schedule::repeated(Dose::infusion(0.0, 200.0, 1.0 / 24.0, 0), 7.0, 4).unwrap();
    let opts = SolverOpts::dopri5(1e-6, 1e-9);
    let theta = [0.2, 3.0, 0.5, 3.0];
    for (label, times) in [
        ("20 output times", &TIMES[..]),
        ("final time only", &[35.0][..]),
    ] {
        let s = simulate_one(&TwoCmtIv, &theta, &sched, times, &opts).unwrap();
        println!(
            "{label}: {:?}, batch 1 thread: {:.2} us/sim",
            s.stats,
            us_per_sim(times, &sched, &opts)
        );
    }
    let s = simulate_one(&TwoCmtIv, &theta, &Schedule::empty(), &[35.0], &opts).unwrap();
    println!(
        "no doses, final time only: {:?}, {:.2} us/sim",
        s.stats,
        us_per_sim(&[35.0], &Schedule::empty(), &opts)
    );
}
