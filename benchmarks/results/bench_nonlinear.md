**twocmt_mm_iv**

| method | threads | N | time (s, median) | sims/s | tuned rtol | achieved max rel. error |
|---|---:|---:|---:|---:|---:|---:|
| cytopk_rust_dopri5 | 1 | 10,000 | 0.1117 | 89,561 | 3.2e-08 | 8.1e-07 |
| cytopk_rust_dopri5 | 16 | 10,000 | 0.009459 | 1,057,202 | 3.2e-08 | 8.1e-07 |
| cytopk_rust_dopri5 | 1 | 100,000 | 1.112 | 89,931 | 3.2e-08 | 1.2e-06 |
| cytopk_rust_dopri5 | 16 | 100,000 | 0.08997 | 1,111,511 | 3.2e-08 | 1.2e-06 |
| numba_dopri5 | 1 | 10,000 | 0.1118 | 89,421 | 3.2e-08 | 8.1e-07 |
| numba_dopri5 | 16 | 10,000 | 0.0101 | 990,070 | 3.2e-08 | 8.1e-07 |
| numba_dopri5 | 1 | 100,000 | 1.126 | 88,832 | 3.2e-08 | 1.2e-06 |
| numba_dopri5 | 16 | 100,000 | 0.09749 | 1,025,777 | 3.2e-08 | 1.2e-06 |
| scipy_solve_ivp_rk45 | 1 | 10,000 | 22.28 ¹ | 449 | 5.6e-07 | 6.9e-07 |
| scipy_solve_ivp_rk45 | 1 | 100,000 | 211.8 ¹ | 472 | 5.6e-07 | 6.9e-07 |
| jax_diffrax_dopri5 | 1 | 10,000 | 0.2577 | 38,808 | 3.2e-07 | 9.4e-07 |
| jax_diffrax_dopri5 | 16 | 10,000 | 1.456 | 6,869 | 3.2e-07 | 9.4e-07 |
| jax_diffrax_dopri5 | 1 | 100,000 | 5.139 | 19,458 | 3.2e-07 | 1.4e-06 |
| jax_diffrax_dopri5 | 16 | 100,000 | 19.32 | 5,177 | 3.2e-07 | 1.4e-06 |

**twocmt_iv_idr**

| method | threads | N | time (s, median) | sims/s | tuned rtol | achieved max rel. error |
|---|---:|---:|---:|---:|---:|---:|
| cytopk_rust_dopri5 | 1 | 10,000 | 0.1614 | 61,952 | 3.2e-08 | 7.0e-07 |
| cytopk_rust_dopri5 | 16 | 10,000 | 0.01332 | 750,741 | 3.2e-08 | 7.0e-07 |
| cytopk_rust_dopri5 | 1 | 100,000 | 1.638 | 61,051 | 3.2e-08 | 1.3e-06 |
| cytopk_rust_dopri5 | 16 | 100,000 | 0.1329 | 752,626 | 3.2e-08 | 1.3e-06 |
| numba_dopri5 | 1 | 10,000 | 0.1504 | 66,503 | 3.2e-08 | 7.0e-07 |
| numba_dopri5 | 16 | 10,000 | 0.0146 | 684,937 | 3.2e-08 | 7.0e-07 |
| numba_dopri5 | 1 | 100,000 | 1.463 | 68,353 | 3.2e-08 | 1.3e-06 |
| numba_dopri5 | 16 | 100,000 | 0.162 | 617,345 | 3.2e-08 | 1.3e-06 |
| scipy_solve_ivp_rk45 | 1 | 10,000 | 24.78 ¹ | 404 | 5.6e-07 | 8.5e-07 |
| scipy_solve_ivp_rk45 | 1 | 100,000 | 250.5 ¹ | 399 | 5.6e-07 | 8.5e-07 |
| jax_diffrax_dopri5 | 1 | 10,000 | 0.7309 | 13,682 | 3.2e-08 | 9.0e-07 |
| jax_diffrax_dopri5 | 16 | 10,000 | 3.731 | 2,681 | 3.2e-08 | 9.0e-07 |
| jax_diffrax_dopri5 | 1 | 100,000 | 16.16 | 6,190 | 3.2e-08 | 1.2e-06 |
| jax_diffrax_dopri5 | 16 | 100,000 | 47.45 | 2,107 | 3.2e-08 | 1.2e-06 |

¹ SciPy timed on the first 2000 subjects (1 run) and scaled linearly to N.
Each method tuned to achieved max rel. error ≤ 1e-06 on a calibration subset; warm-up/JIT compile excluded (see `warmup_call_s` in the JSON).
Intel(R) Core(TM) i5-14400F, 16 logical CPUs; single-thread rows pinned to CPU 0; git 263bd78
