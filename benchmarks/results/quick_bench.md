| method | threads | N | time (s, median) | sims/s | max rel. error |
|---|---:|---:|---:|---:|---:|
| numpy_closed_form | 1 | 10,000 | 0.02476 | 403,928 | reference |
| cytopk_rust_dopri5 | 1 | 10,000 | 0.06136 | 162,977 | 1.4e-06 |
| cytopk_rust_dopri5 | 16 | 10,000 | 0.009323 | 1,072,613 | 1.4e-06 |
| numba_dopri5 | 1 | 10,000 | 0.06765 | 147,830 | 1.4e-06 |
| numba_dopri5 | 16 | 10,000 | 0.02214 | 451,690 | 1.4e-06 |
| scipy_solve_ivp_rk45 | 1 | 10,000 | 21.16 ¹ | 472 | 1.0e-06 |
| numpy_closed_form | 1 | 100,000 | 0.2385 | 419,325 | reference |
| cytopk_rust_dopri5 | 1 | 100,000 | 0.6201 | 161,260 | 1.8e-06 |
| cytopk_rust_dopri5 | 16 | 100,000 | 0.05427 | 1,842,772 | 1.8e-06 |
| numba_dopri5 | 1 | 100,000 | 0.6739 | 148,394 | 1.8e-06 |
| numba_dopri5 | 16 | 100,000 | 0.06638 | 1,506,591 | 1.8e-06 |
| scipy_solve_ivp_rk45 | 1 | 100,000 | 239.4 ¹ | 418 | 1.0e-06 |

¹ SciPy timed on the first 2000 subjects (1 run) and scaled linearly to N.
Numba JIT compile time (excluded from the table): 0.89–0.99 s per process.
rtol=1e-06, atol=1e-09; Intel(R) Core(TM) i5-14400F, 16 logical CPUs; single-thread rows pinned to CPU 0; git 772d522
