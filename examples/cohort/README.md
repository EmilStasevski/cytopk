# M8: synthetic cohort, end to end (first working version)

simulate PK/PD → write FCS → read and gate with cytopk → train BayesFlow on gate counts → simulation-based
calibration (SBC).

**Toy model, synthetic data only.** Everything here is illustrative: the `twocmt_iv_idr` parameters, the
cell populations and the gates. No real patient or instrument data is used, and the files carry the keyword
`SYNTHETIC/yes`.

## Generative model (`cohort.py`)

1. **Subjects.** `log θ` holds the 7 parameters of `twocmt_iv_idr` (CL, V1, Q, V2, kout, Emax, EC50). The prior
   is the E1 one: log-normal around the docs/models.md values, sd 0.3.
2. **PK/PD.** One 200 mg, 1-hour IV infusion. Samples are taken at the 9 E1 times (0.04 to 28 days). The IDR
   state gives the activated-cell fraction `p(t) = R/(R+9)`.
3. **Samples.** The activated fraction of CD3+ cells in a sample is `q ~ Beta(p·100, (1−p)·100)`. Each sample
   has 10^4 events from a 4-component Gaussian mixture on the arcsinh scale (cofactor 150): debris, CD3−,
   CD3+ resting and CD3+ activated, over FSC-A, CD3 and CD69.
4. **FCS.** Events are written as linear float32 values (`150·sinh(y)`) with `cytopk.write_fcs`.
5. **Gating.** `cytopk.gate_fcs_many` with the TOML strategy `cohort.STRATEGY`: cells → cd3 → activated.
6. **Summaries per time:** logit(activated / cd3) and log(cd3 / cells), plus the log sampling time. The input
   is (9, 3).

### Training path versus test path

Every gate is a rectangle and every mixture component is an axis-aligned Gaussian. For a given `q`, the
probability that an event lands in each gate is therefore a closed-form sum of normal-CDF products, and the
gate counts are exactly multinomial (`counts_fast`).

- **Training** uses this fast path: 40,000 subjects in under a second.
- **Test set A** goes through the real files: 300 subjects × 9 samples = 2,700 FCS files written, read and
  gated. If the fast path did not match the pipeline, SBC on set A would show it.
- **Test set B** uses the fast path, for comparison.
- **Direct checks** (`python/tests/test_cohort.py`):
  - write → read → gate gives exactly the counts NumPy computes on the stored float32 values;
  - the event-level mean counts match the closed-form gate probabilities;
  - the spread of the fast-path counts matches the event-level spread.

### Network

BayesFlow 2.1.0 `BasicWorkflow` with `CouplingFlow()` and `TimeSeriesNetwork()`, library defaults,
`fit_offline`. JAX backend on the GPU.

## Results

Source: `results/m8.json`, rendered by `make_tables.py`. Seeds are fixed.

**Test set through FCS files** (300 subjects, 1000 posterior draws each):

| metric | CL | V1 | Q | V2 | kout | Emax | EC50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| contraction | 0.03 | 0.00 | 0.02 | 0.08 | 0.22 | 0.71 | 0.00 |
| NRMSE (1 = no better than prior) | 0.98 | 1.01 | 0.98 | 0.97 | 0.88 | 0.53 | 1.01 |
| calibration error | 0.006 | 0.025 | 0.011 | 0.018 | 0.016 | 0.010 | 0.018 |
| rank χ² p (20 bins) | 0.31 | 0.11 | 0.20 | 0.04 | 0.11 | 0.99 | 0.39 |
| rank KS p | 0.69 | 0.34 | 0.51 | 0.48 | 0.76 | 0.99 | 0.19 |

**Test set from the fast path** (300 subjects, 1000 posterior draws each):

| metric | CL | V1 | Q | V2 | kout | Emax | EC50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| contraction | 0.05 | 0.00 | 0.12 | 0.05 | 0.24 | 0.66 | 0.04 |
| NRMSE (1 = no better than prior) | 0.99 | 1.00 | 0.95 | 0.98 | 0.90 | 0.57 | 0.99 |
| calibration error | 0.016 | 0.003 | 0.018 | 0.015 | 0.016 | 0.013 | 0.009 |
| rank χ² p (20 bins) | 0.43 | 0.71 | 0.24 | 0.23 | 0.01 | 0.42 | 0.78 |
| rank KS p | 0.88 | 0.58 | 0.05 | 0.29 | 0.54 | 0.70 | 0.37 |

Training: 40,000 subjects, 60 epochs, batch 256, fit 94 s on cuda:0; final loss train 9.05, validation 9.01.
FCS test set: 2,700 files, 325 MB; writing 2.7 s; reading and gating with gate_fcs_many 0.22 s.
FCS vs fast path, same q (FCS minus fast): mean difference -0.0007 (logit activated/cd3), +0.0004 (log cd3/cells); sd of the differences 0.050 and 0.011.
Rank p-values below 0.05: 3 of 28: fcs chi2 V2 0.041, fast chi2 kout 0.010, fast ks Q 0.048.

### Plots (`figures/`)

- **SBC rank ECDF difference with 95% bands:** `sbc_ecdf_fcs.png`, `sbc_ecdf_fast.png`.
- **Rank histograms:** `sbc_hist_*.png`.
- **Recovery** (posterior median vs truth): `sbc_recovery_*.png`.
- **Training loss:** `loss.png`.

## Reading the results

- **Calibration.** In both test sets, every rank ECDF stays inside the 95% band (see the plots), and every
  calibration error is ≤ 0.025. Three of the 28 rank tests have p < 0.05. Under perfect calibration about
  1.4 of 28 would, so this is not evidence of miscalibration, but it is also not proof of calibration at
  300 subjects.
- **Information.** The gate counts are informative about **Emax** (contraction about 0.7, NRMSE about 0.55)
  and somewhat about **kout** (about 0.2). They carry almost nothing about CL, V1, Q, V2 or EC50: the posteriors
  stay close to the prior (contraction ≤ 0.12).
  - The obvious reasons, **not tested here**, are that there are no PK observations and that a single dose
    level is used. A follow-up should vary the design (PK samples, several dose levels) and measure the
    contraction change.
  - This is a property of the cohort design, not of the FCS/gating pipeline.
- **FCS pipeline cost.** Writing 2,700 FCS files (325 MB) took 2.7 s. Reading and gating all of them in Rust took
  0.22 s, a small fraction of the 94 s training.

## Not done in this first version

- Inference is per subject, with no population (hierarchical) model.
- There are no PK observations, no dose variation, and no instrument effects (compensation, spillover,
  batch shifts).
- The gates are fixed. They are not learned or tuned per sample.
- There is a single training run with no seed replicates, and no comparison with a likelihood-based fit.

## Reproduce

```sh
.venv-sbi/bin/python examples/cohort/run_m8.py      # FCS files go to data/m8/ (git-ignored); GPU used if JAX sees one
python examples/cohort/make_tables.py
```
