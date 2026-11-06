//! Line-by-line Rust port of the Numba kernel in benchmarks/quick_bench.py (fixed 2 states, no traits).
//! Used only to bisect the performance gap between cytopk-sim and Numba.
#![allow(clippy::all, missing_docs)]

#[inline(always)]
fn rhs(y: &[f64; 2], cl: f64, v1: f64, q: f64, v2: f64, r: f64, dy: &mut [f64; 2]) {
    let c1 = y[0] / v1;
    let c2 = y[1] / v2;
    let tr = q * (c1 - c2);
    dy[0] = -cl * c1 - tr + r;
    dy[1] = tr;
}

fn subject(
    th: &[f64],
    stops: &[f64],
    rates: &[f64],
    out_idx: &[i64],
    rtol: f64,
    atol: f64,
    out: &mut [f64],
) {
    let (cl, v1, q, v2) = (th[0], th[1], th[2], th[3]);
    let mut y = [0.0; 2];
    let (mut yn, mut ys) = ([0.0; 2], [0.0; 2]);
    let (mut k1, mut k2, mut k3, mut k4, mut k5, mut k6, mut k7) = (
        [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2],
    );
    let (mut t, mut r, mut h) = (0.0f64, 0.0f64, 0.0f64);
    for s in 0..stops.len() {
        let t1 = stops[s];
        if t1 > t {
            if h <= 0.0 {
                rhs(&y, cl, v1, q, v2, r, &mut k1);
                let (mut d0, mut d1) = (0.0, 0.0);
                for i in 0..2 {
                    let sc = atol + rtol * y[i].abs();
                    d0 += (y[i] / sc).powi(2);
                    d1 += (k1[i] / sc).powi(2);
                }
                let (d0, d1) = ((d0 / 2.0f64).sqrt(), (d1 / 2.0f64).sqrt());
                let h0 = if d0 < 1e-5 || d1 < 1e-5 {
                    1e-6
                } else {
                    0.01 * d0 / d1
                };
                for i in 0..2 {
                    ys[i] = y[i] + h0 * k1[i];
                }
                rhs(&ys, cl, v1, q, v2, r, &mut k2);
                let mut d2 = 0.0;
                for i in 0..2 {
                    let sc = atol + rtol * y[i].abs();
                    d2 += ((k2[i] - k1[i]) / sc).powi(2);
                }
                let d2 = (d2 / 2.0f64).sqrt() / h0;
                let dm = d1.max(d2);
                let h1 = if dm <= 1e-15 {
                    (h0 * 1e-3).max(1e-6)
                } else {
                    (0.01 / dm).powf(0.2)
                };
                h = (100.0 * h0).min(h1);
            }
            rhs(&y, cl, v1, q, v2, r, &mut k1);
            let mut last_rej = false;
            while t < t1 {
                let rem = t1 - t;
                let mut hh = h;
                let clamped = hh >= rem * 0.99;
                if clamped {
                    hh = rem;
                }
                for i in 0..2 {
                    ys[i] = y[i] + hh * (0.2 * k1[i]);
                }
                rhs(&ys, cl, v1, q, v2, r, &mut k2);
                for i in 0..2 {
                    ys[i] = y[i] + hh * (3. / 40. * k1[i] + 9. / 40. * k2[i]);
                }
                rhs(&ys, cl, v1, q, v2, r, &mut k3);
                for i in 0..2 {
                    ys[i] = y[i] + hh * (44. / 45. * k1[i] - 56. / 15. * k2[i] + 32. / 9. * k3[i]);
                }
                rhs(&ys, cl, v1, q, v2, r, &mut k4);
                for i in 0..2 {
                    ys[i] = y[i]
                        + hh * (19372. / 6561. * k1[i] - 25360. / 2187. * k2[i]
                            + 64448. / 6561. * k3[i]
                            - 212. / 729. * k4[i]);
                }
                rhs(&ys, cl, v1, q, v2, r, &mut k5);
                for i in 0..2 {
                    ys[i] = y[i]
                        + hh * (9017. / 3168. * k1[i] - 355. / 33. * k2[i]
                            + 46732. / 5247. * k3[i]
                            + 49. / 176. * k4[i]
                            - 5103. / 18656. * k5[i]);
                }
                rhs(&ys, cl, v1, q, v2, r, &mut k6);
                for i in 0..2 {
                    yn[i] = y[i]
                        + hh * (35. / 384. * k1[i] + 500. / 1113. * k3[i] + 125. / 192. * k4[i]
                            - 2187. / 6784. * k5[i]
                            + 11. / 84. * k6[i]);
                }
                rhs(&yn, cl, v1, q, v2, r, &mut k7);
                let mut err = 0.0;
                for i in 0..2 {
                    let e = hh
                        * (71. / 57600. * k1[i] - 71. / 16695. * k3[i] + 71. / 1920. * k4[i]
                            - 17253. / 339200. * k5[i]
                            + 22. / 525. * k6[i]
                            - 1. / 40. * k7[i]);
                    let sc = atol + rtol * y[i].abs().max(yn[i].abs());
                    err += (e / sc).powi(2);
                }
                let err = (err / 2.0f64).sqrt();
                let mut fac = if err == 0.0 {
                    10.0
                } else {
                    (0.9 * err.powf(-0.2)).max(0.2).min(10.0)
                };
                if err <= 1.0 {
                    y = yn;
                    k1 = k7;
                    t = if clamped { t1 } else { t + hh };
                    if last_rej {
                        fac = fac.min(1.0);
                    }
                    let hn = hh * fac;
                    h = if clamped { h.max(hn) } else { hn };
                    last_rej = false;
                } else {
                    h = hh * fac.min(1.0);
                    last_rej = true;
                }
            }
        }
        t = t1;
        r = rates[s];
        let j = out_idx[s];
        if j >= 0 {
            out[j as usize] = y[0] / v1;
        }
    }
}

fn main() {
    let times = [
        0.04, 0.5, 1., 2., 4., 7., 7.04, 8., 10., 14., 14.04, 15., 17., 21., 21.04, 22., 24., 28.,
        31., 35.,
    ];
    let (dt, dd, amt) = ([0.0, 7.0, 14.0, 21.0], 1.0 / 24.0, 200.0);
    let mut stops: Vec<f64> = times.to_vec();
    for &d in &dt {
        stops.push(d);
        stops.push(d + dd);
    }
    stops.sort_by(f64::total_cmp);
    stops.dedup();
    let rates: Vec<f64> = stops
        .iter()
        .map(|&s| dt.iter().filter(|&&d| d <= s && s < d + dd).count() as f64 * amt / dd)
        .collect();
    let out_idx: Vec<i64> = stops
        .iter()
        .map(|s| times.iter().position(|t| t == s).map_or(-1, |p| p as i64))
        .collect();
    let n = 20_000;
    let thetas: Vec<[f64; 4]> = (0..n)
        .map(|i| [0.2 + 1e-6 * i as f64, 3.0, 0.5, 3.0])
        .collect();
    let mut out = vec![0.0; n * 20];
    let best = (0..7)
        .map(|_| {
            let t0 = std::time::Instant::now();
            for (th, o) in thetas.iter().zip(out.chunks_mut(20)) {
                subject(th, &stops, &rates, &out_idx, 1e-6, 1e-9, o);
            }
            std::hint::black_box(&out);
            t0.elapsed().as_secs_f64() / n as f64 * 1e6
        })
        .fold(f64::INFINITY, f64::min);
    println!(
        "numba-port (plain Rust, fixed arrays): {best:.2} us/sim, out[0][..3] = {:?}",
        &out[..3]
    );
}
