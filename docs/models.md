# Models

> Toy models with illustrative parameter values for a research prototype. Not validated for clinical or regulatory use.

## Units

| quantity | unit |
|---|---|
| time | day |
| amount | mg |
| volume (V, V1, V2) | L |
| clearance (CL, Q) | L/day |
| concentration | mg/L |
| infusion rate | mg/day |

## Implemented models (M1)

| id | states | parameters (constraint) | outputs |
|---|---|---|---|
| `onecmt_iv` | A1 | CL (≥ 0), V (> 0) | C = A1/V |
| `twocmt_iv` | A1 (central), A2 (peripheral) | CL (≥ 0), V1 (> 0), Q (≥ 0), V2 (> 0) | C = A1/V1 |

The milestone M2 models (`twocmt_mm_iv`, `twocmt_iv_idr`) will be documented here when they are implemented.

### `onecmt_iv`

```
dA1/dt = −(CL/V)·A1 + u1(t)
```

### `twocmt_iv`

```
dA1/dt = −CL·C1 − Q·(C1 − C2) + u1(t),    C1 = A1/V1
dA2/dt =          Q·(C1 − C2) + u2(t),    C2 = A2/V2
```

`u_i(t)` is the total infusion rate into state `i` (see Dosing). Every state starts at 0 at `t = 0` (a drug-naive subject).

## Dosing

- `Dose { time, amount, state, duration }`. If `duration` is `None` the dose is an **IV bolus**. Otherwise it is a **zero-order infusion**
  with rate `amount/duration`, active on the half-open interval `[time, time + duration)`.
- `state` selects the compartment that receives the dose (0 = central). IV doses normally go to state 0. Other states are allowed,
  which is how absorption compartments can be added later without changing the dosing code.
- Doses must have `time ≥ 0`, `amount ≥ 0`, and a finite `duration > 0` for infusions. The list does not need to be sorted, because `Schedule::new`
  sorts it. Overlapping infusions add up.
- **Output convention:** the output at time `t` includes every bolus given at exactly `t` (the state is right-continuous).
  Repeated output times are allowed and give identical rows.
- The simulation always starts at `t = 0`. Doses after the last output time are ignored, because they cannot affect any output.

## Solver

The time axis is split at every dose time, every infusion end, and every requested output time. Each piece is
integrated separately, so **no step crosses a discontinuity**. Outputs are read at piece ends, with no interpolation:
the output error is the solver's own controlled error, not interpolation error.

- `Dopri5` (default) is the adaptive Dormand–Prince 5(4) pair (Hairer, Nørsett & Wanner, *Solving ODEs I*, §II.4/II.5),
  with FSAL inside a piece and an RMS error norm with weights `atol + rtol·max(|y|, |y_new|)`. Step-size control is
  `h_new = h·clamp(0.9·err^(-1/5), 0.2, 10)`, and the step may not grow right after a rejection.
  The first step uses the Hairer starting-step heuristic, and later pieces reuse the last step-size proposal.
- `Rk4 { step }` is classical RK4. Each piece is split into `ceil(len/step)` equal steps. Use it for reference and debugging.
- Defaults: `rtol = 1e-6`, `atol = 1e-9` (mg), `max_steps = 100 000` attempts per simulation.

**Stiffness.** Both methods are explicit. For the parameter ranges used here, the eigenvalues are of order
0.01–10 per day over horizons of weeks, so the problems are not stiff and explicit RK is the right tool. A model with
very fast and very slow processes (e.g. kout ≫ the PK rates, or receptor binding kinetics in TMDD models) would
need an implicit method (BDF/Radau). That is out of scope for v0, and `max_steps` turns such a case into an error
instead of an endless run.

## Batch simulation and determinism

`simulate_batch` runs each row of an `N × P` parameter matrix on the Rayon pool and writes into a preallocated
`N × T × n_outputs` array. Each simulation is independent, uses only its own scratch buffers, and performs the
same floating-point operations in the same order, so **results are bit-identical for any thread count**
(tested with 1, 2 and all threads).

## Closed forms (test oracles)

For linear models with doses into the central compartment, the central amount after a unit bolus at time 0 is
the impulse response `g(s) = Σ_i c_i e^{−λ_i s}`. By superposition:

- bolus `D` at `t_d`: `A1(t) += D·g(t − t_d)` for `t ≥ t_d`;
- infusion with rate `R` over `[t_d, t_d + T]`: with `s = t − t_d` and `φ(λ, x) = (1 − e^{−λx})/λ` (`= x` when `λ = 0`),
  `A1 += R Σ_i c_i φ(λ_i, s)` for `s ≤ T` and `A1 += R Σ_i c_i φ(λ_i, T) e^{−λ_i (s − T)}` for `s > T`.

`φ` is evaluated with `expm1` to avoid cancellation for small `λx`.

**One compartment:** `c = [1]`, `λ = [k]` with `k = CL/V`. This gives `C(t) = (D/V) e^{−kt}` for a bolus, and
`C(t) = R/CL·(1 − e^{−kt})` during an infusion.

**Two compartments.** With micro-constants `k10 = CL/V1`, `k12 = Q/V1`, `k21 = Q/V2`, the system is
`dA/dt = K·A` with `K = [[−(k10 + k12), k21], [k12, −k21]]`. The eigenvalues of `K` are `−α` and `−β`, where `α, β` are the roots of
`s² − (k10 + k12 + k21)s + k10·k21 = 0`:

```
α = ½[(k10 + k12 + k21) + √((k10 + k12 − k21)² + 4·k12·k21)],    β = k10·k21 / α
```

Computing β from the product of the roots avoids the cancellation in `sum − √disc`. The discriminant equals
`(k10 + k12 − k21)² + 4·k12·k21`, which is > 0 whenever `Q > 0`, so `α > β`. For the central amount after a unit bolus,
`A1(0) = 1` and `A1'(0) = −(k10 + k12)`. Solving `c_α + c_β = 1`, `−α c_α − β c_β = −(k10 + k12)`, and using `α + β = k10 + k12 + k21`, gives

```
c_α = (α − k21)/(α − β),    c_β = (k21 − β)/(α − β)
```

For `Q = 0` the central compartment is decoupled and reduces to the 1-compartment case with `k = k10`.

The multiple-dose check uses the 1-cmt accumulation ratio: the trough after many boluses every `τ` approaches
`C_1(τ)/(1 − e^{−kτ})`.

## Validation summary (M1)

| check | result (see `cargo test -p cytopk-sim -- --nocapture`) |
|---|---|
| Dopri5 vs closed form, 1000 random parameter sets/schedules per model (1–4 doses, bolus or 1 h–1 day infusion, outputs at discontinuities), `rtol = 1e-8`, `atol = 1e-12` | max relative error 3.9e-8 (1-cmt), 2.3e-8 (2-cmt); limit 1e-6 |
| closed form vs fine fixed-step RK4 (h = 1e-3 d), an independent check of the oracle | rel. error < 1e-7 |
| RK4 global order | error ratio for h → h/2 in [13, 19] |
| Dopri5 error falls monotonically with rtol in 1e-4 … 1e-10 | pass |
| mass balance `dose given − eliminated − A1 − A2 = 0` (auxiliary state); `CL = 0` ⇒ nothing eliminated | ≤ 1e-8 × total dose |
| `AUC₀^∞ = dose/CL` (auxiliary AUC state) | rel. error < 1e-8 |
| non-negative amounts; monotone decay after a 1-cmt bolus | pass |
| output at a dose time includes the bolus; repeated output times are identical | pass |
| property tests: dose order irrelevant; invalid doses rejected (never panic); infusion → bolus as duration → 0; linear in dose | pass (128 cases each) |
| bit-identical batch output with 1, 2 and all threads; batch path = single-subject path; Fortran-order θ | pass |

The exact errors depend on the random draws (seeded, so reproducible). The limits in the table are the ones asserted by the tests.

## Illustrative prior ranges (for M2/M8, not used yet)

Values typical of a monoclonal-antibody-like drug, for demonstration only: CL ≈ 0.2 L/day, V1 ≈ 3 L, Q ≈ 0.5 L/day,
V2 ≈ 3 L, kout ≈ 0.3 1/day, Emax ≈ 2, EC50 ≈ 5 mg/L; doses 50/200/800 mg as 1-hour infusions. The M1 tests sample
wider log-uniform ranges: CL 0.05–2, V/V1/V2 1–20, Q 0.05–5.
