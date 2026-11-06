//! Drift-diffusion model (DDM) simulator: Euler–Maruyama with deterministic per-simulation RNG streams.
//!
//! Model (diffusion coefficient s = 1): `dX = v dt + dW`, `X(0) = z·a`, absorbing boundaries at 0 and `a`.
//! On each step `X += v·dt + √dt·N(0, 1)`. The first step with `X ≥ a` gives choice 1 (upper boundary); the
//! first with `X ≤ 0` gives choice 0. The reaction time is `n·dt + t0` after `n` steps. If no boundary is hit
//! within `max_t`, the result is choice −1 with RT `NaN`.
//!
//! **RNG.** Every simulation `i` gets its own xoshiro256++ generator (Blackman & Vigna, 2018). It is seeded
//! from `(seed, i)` by a SplitMix64-based mixing function ([`Xoshiro256pp::stream`]), so results depend only on
//! `(seed, i)` and never on thread scheduling. Uniform doubles use the top 53 bits. Normal deviates use the
//! Marsaglia polar method, keeping the second deviate of each pair. These choices are deliberately simple, so
//! they can be reproduced exactly in other languages (the Numba baseline does this).

use ndarray::ArrayView2;
use rayon::prelude::*;

use crate::{MIN_PARALLEL_BATCH, Parallelism, SimError};

/// SplitMix64 step (Steele, Lea & Flood 2014): advances `state` and returns a well-mixed output.
#[inline]
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256++ pseudo-random generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    /// Generator with an explicit state (must not be all zero).
    pub fn from_state(s: [u64; 4]) -> Self {
        Self { s }
    }

    /// Independent stream for simulation `index` under `seed`. The key is
    /// `splitmix64(seed) ^ splitmix64(index ^ 0xA076_1D64_78BD_642F)`, and the four state words are the next four
    /// SplitMix64 outputs from that key, as in the reference seeding recommendation for xoshiro.
    pub fn stream(seed: u64, index: u64) -> Self {
        let mut a = seed;
        let mut b = index ^ 0xA076_1D64_78BD_642F;
        let mut key = splitmix64(&mut a) ^ splitmix64(&mut b);
        let s = [
            splitmix64(&mut key),
            splitmix64(&mut key),
            splitmix64(&mut key),
            splitmix64(&mut key),
        ];
        Self { s }
    }

    /// Next 64 random bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform double in `[0, 1)` from the top 53 bits.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Standard normal deviates by the Marsaglia polar method, caching the second deviate of each pair.
#[derive(Debug, Clone)]
pub struct PolarNormal {
    rng: Xoshiro256pp,
    spare: Option<f64>,
}

impl PolarNormal {
    /// Wraps a generator.
    pub fn new(rng: Xoshiro256pp) -> Self {
        Self { rng, spare: None }
    }

    /// Next N(0, 1) deviate.
    #[inline]
    pub fn sample(&mut self) -> f64 {
        if let Some(x) = self.spare.take() {
            return x;
        }
        loop {
            let u = 2.0 * self.rng.next_f64() - 1.0;
            let v = 2.0 * self.rng.next_f64() - 1.0;
            let s = u * u + v * v;
            if s < 1.0 && s > 0.0 {
                let m = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * m);
                return u * m;
            }
        }
    }
}

/// DDM discretisation settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DdmOpts {
    /// Time step (s).
    pub dt: f64,
    /// Maximum decision time (s) before the trial counts as a timeout.
    pub max_t: f64,
}

impl Default for DdmOpts {
    fn default() -> Self {
        Self {
            dt: 0.001,
            max_t: 10.0,
        }
    }
}

/// One simulated trial: `(rt, choice)` with choice 1 = upper, 0 = lower, −1 = timeout (rt NaN).
#[inline]
pub fn ddm_trial(
    v: f64,
    a: f64,
    z: f64,
    t0: f64,
    opts: &DdmOpts,
    normal: &mut PolarNormal,
) -> (f64, i8) {
    let max_n = (opts.max_t / opts.dt).round() as u64;
    let drift = v * opts.dt;
    let sd = opts.dt.sqrt();
    let mut x = z * a;
    let mut n: u64 = 0;
    while n < max_n {
        n += 1;
        x += drift + sd * normal.sample();
        if x >= a {
            return (n as f64 * opts.dt + t0, 1);
        }
        if x <= 0.0 {
            return (n as f64 * opts.dt + t0, 0);
        }
    }
    (f64::NAN, -1)
}

fn check_ddm(theta: ArrayView2<'_, f64>, opts: &DdmOpts) -> Result<(), SimError> {
    if !(opts.dt.is_finite() && opts.dt > 0.0 && opts.max_t.is_finite() && opts.max_t > 0.0) {
        return Err(SimError::InvalidOptions(
            "dt and max_t must be finite and > 0",
        ));
    }
    if opts.max_t / opts.dt > 1e12 {
        return Err(SimError::InvalidOptions("max_t / dt is too large"));
    }
    if theta.ncols() != 4 {
        return Err(SimError::ParamCount {
            model: "ddm",
            expected: 4,
            got: theta.ncols(),
        });
    }
    for (index, r) in theta.outer_iter().enumerate() {
        let bad = |name, value, constraint| {
            Err(SimError::InBatch {
                index,
                source: Box::new(SimError::InvalidParam {
                    name,
                    value,
                    constraint,
                }),
            })
        };
        if !r[0].is_finite() {
            return bad("v", r[0], "finite");
        }
        if !(r[1].is_finite() && r[1] > 0.0) {
            return bad("a", r[1], "finite and > 0");
        }
        if !(r[2] > 0.0 && r[2] < 1.0) {
            return bad("z", r[2], "in (0, 1)");
        }
        if !(r[3].is_finite() && r[3] >= 0.0) {
            return bad("t0", r[3], "finite and >= 0");
        }
    }
    Ok(())
}

/// Simulates one trial per row of `theta` (shape `N × 4`: `v, a, z, t0`). Returns `(rt, choice)`, each of length N.
/// Simulation `i` uses the stream `Xoshiro256pp::stream(seed, i)`, so results are bit-identical for any thread
/// count and either [`Parallelism`] mode.
pub fn simulate_ddm(
    theta: ArrayView2<'_, f64>,
    seed: u64,
    opts: &DdmOpts,
    parallelism: Parallelism,
) -> Result<(Vec<f64>, Vec<i8>), SimError> {
    check_ddm(theta, opts)?;
    let n = theta.nrows();
    let one = |i: usize| {
        let r = theta.row(i);
        let mut normal = PolarNormal::new(Xoshiro256pp::stream(seed, i as u64));
        ddm_trial(r[0], r[1], r[2], r[3], opts, &mut normal)
    };
    let pairs: Vec<(f64, i8)> = if parallelism == Parallelism::Serial || n < MIN_PARALLEL_BATCH {
        (0..n).map(one).collect()
    } else {
        (0..n).into_par_iter().map(one).collect()
    };
    Ok(pairs.into_iter().unzip())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xoshiro_reference_first_output() {
        // For state {1, 2, 3, 4}: rotl(1 + 4, 23) + 1 = 41943041 (reference implementation).
        let mut r = Xoshiro256pp::from_state([1, 2, 3, 4]);
        assert_eq!(r.next_u64(), 41_943_041);
    }

    #[test]
    fn streams_differ_and_are_reproducible() {
        let a: Vec<u64> = (0..4)
            .map(|_| 0)
            .scan(Xoshiro256pp::stream(7, 0), |r, _| Some(r.next_u64()))
            .collect();
        let b: Vec<u64> = (0..4)
            .map(|_| 0)
            .scan(Xoshiro256pp::stream(7, 0), |r, _| Some(r.next_u64()))
            .collect();
        let c: Vec<u64> = (0..4)
            .map(|_| 0)
            .scan(Xoshiro256pp::stream(7, 1), |r, _| Some(r.next_u64()))
            .collect();
        let d: Vec<u64> = (0..4)
            .map(|_| 0)
            .scan(Xoshiro256pp::stream(8, 0), |r, _| Some(r.next_u64()))
            .collect();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test]
    fn uniform_and_polar_moments() {
        let mut u = Xoshiro256pp::stream(1, 0);
        let n = 400_000;
        let xs: Vec<f64> = (0..n).map(|_| u.next_f64()).collect();
        assert!(xs.iter().all(|&x| (0.0..1.0).contains(&x)));
        let mean = xs.iter().sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 4.0 * (1.0 / 12.0f64 / n as f64).sqrt());
        let mut g = PolarNormal::new(Xoshiro256pp::stream(2, 0));
        let zs: Vec<f64> = (0..n).map(|_| g.sample()).collect();
        let m = zs.iter().sum::<f64>() / n as f64;
        let var = zs.iter().map(|z| (z - m).powi(2)).sum::<f64>() / n as f64;
        let kurt = zs.iter().map(|z| (z - m).powi(4)).sum::<f64>() / n as f64 / var.powi(2);
        // SE(mean) = 1/sqrt(n), SE(var) ~ sqrt(2/n), SE(kurtosis) ~ sqrt(24/n)
        assert!(m.abs() < 4.0 / (n as f64).sqrt(), "mean {m}");
        assert!(
            (var - 1.0).abs() < 4.0 * (2.0 / n as f64).sqrt(),
            "var {var}"
        );
        assert!(
            (kurt - 3.0).abs() < 4.0 * (24.0 / n as f64).sqrt(),
            "kurtosis {kurt}"
        );
    }

    #[test]
    fn trial_hits_a_boundary_with_strong_drift() {
        let mut g = PolarNormal::new(Xoshiro256pp::stream(3, 0));
        let (rt, c) = ddm_trial(50.0, 1.0, 0.5, 0.2, &DdmOpts::default(), &mut g);
        assert_eq!(c, 1);
        assert!(rt > 0.2 && rt < 0.3);
        // Timeout: a tiny max_t with a wide boundary.
        let opts = DdmOpts {
            dt: 0.001,
            max_t: 0.002,
        };
        let (rt, c) = ddm_trial(0.0, 100.0, 0.5, 0.2, &opts, &mut g);
        assert_eq!(c, -1);
        assert!(rt.is_nan());
    }
}
