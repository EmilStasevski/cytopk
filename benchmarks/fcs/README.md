# FCS reader benchmarks: robustness, speed, memory

cytopk-fcs (Rust, `cytopk.read_fcs`) compared with flowio 1.4.0, FlowKit 1.3.0, fcsparser 0.2.4
and flowCore 2.24.0 (R). Every number below comes from `benchmarks/results/fcs_robustness.json`,
`fcs_speed.json` or `fcs_fuzz.json`, rendered by `make_tables.py`.

Machine: Intel(R) Core(TM) i5-14400F, 16 logical CPUs; Linux-6.17.0-1023-oem-x86_64-with-glibc2.39; Python 3.13.13 (NumPy 2.5.4, pandas
3.1.1); R version 4.6.2 (2026-10-29); rustc 1.99.1 (58cf8f9c3 2026-10-13). The fcsparser NumPy-1 column ran in Python/NumPy/pandas/fcsparser
3.12.3 1.26.4 2.2.3 0.2.4.

## Summary

- **Robustness on real files.** cytopk lenient returned data for 28 of the 30 real public files. The other two
  fail in every reader: a 10-byte file, and a HEADER+TEXT file with no DATA. fcsparser under NumPy 1.x also reads
  28/30, flowio's lenient options 25/30, and flowCore 21/30.
  - On every real file that two readers both returned, the values agree exactly with cytopk, with three
    exceptions, all in the other readers:
    - fcsparser takes the HEADER DATA offset that points into TEXT (`data_start_offset_discrepancy_example`).
    - fcsparser's default float32 rounds 32-bit integers above 2^24 (`M0_WM278_S1`).
    - flowCore mis-decodes 24-bit integers (`Cytek_xP5`; cytopk agrees with fcsparser there).
- **Robustness on artificial files.** On the corpus, cytopk is correct on 7/7 conformant and 23/23
  non-conformant files. **This favours cytopk by construction:** I wrote both the deviations and their ground
  truth, using the same rules the reader implements (`docs/fcs_notes.md`). Treat it as evidence that each
  documented repair works, not as an independent comparison. The other readers' columns show which of these
  deviations they handle.
- **Strict mode** reads 6/30 real files. Most real files break the specification somewhere: FCS 2.0, `$TOT`
  padded with spaces, bytes after the last TEXT delimiter. That matches Bras & van der Velden (2020).
- **Speed, large files** (warm page cache): cytopk is 15.8–25.8× faster than the fastest other reader from
  10^6 events up. **Most of that is parallelism:** cytopk decodes large DATA segments on 16 threads and the
  other readers use one. With one thread (`cytopk_1thread`) the advantage is 1.93–4.01×.
- **Where others win:** fcsparser is faster on `M0_WM278_S1` (11.3 vs 19.6 ms). That file holds 32-bit integer
  data; cytopk decodes integers per event into float64, and the benchmark then converts to float32. On
  `fcs1_cleaned.lmd` cytopk is only 1.25× ahead.
- **Memory:** the extra peak RSS of one read is about 2× the float32 matrix for cytopk (the matrix plus the
  mapped file pages) and the same for fcsparser. flowio and FlowKit use about 3× and flowCore about 4× (it
  builds a double matrix).
- **Many small files:** 500 files take 0.06 s with `read_fcs_many` (16 threads), against 0.23 s for a flowio loop,
  0.68 s for fcsparser and 1.30 s for FlowKit. joblib with 16 processes is slower than a plain loop for every
  reader at this size (process start-up and pickling).
- **Fuzzing:** cargo-fuzz ran 378,136 inputs in 601 s (seeded with the corpus and the real
  files) with 0 crashes. The proptest mutation tests in CI found one real panic during
  development: a TEXT end clamp past the file end, fixed before the first commit.

## Robustness

Each (reader, file) runs in its own process with a 60 s timeout, so a crash or hang affects only that cell.

- **Readers.** Each reader runs with its defaults. `_lenient` columns turn on the reader's own leniency switches
  (flowio/FlowKit: `ignore_offset_error`, `ignore_offset_discrepancy`). fcsparser names channels by `$PnN`, like
  the others.
- **FlowKit on real files** is compared only on linear, gain-1, non-time channels. Its `raw` events already
  apply `$TIMESTEP`, gain and `$PnE`.
- **`tot_too_small`:** truth follows the specification (the first `$TOT` events). flowio and FlowKit return
  every event in DATA instead. That is a policy difference rather than corruption, but the event count does
  not match `$TOT`.
- **Not run:** the Bras & van der Velden TEST-FCS set needs a CAPTCHA download
  (`scripts/fetch_test_data.py --bras-archive`), and was not part of this run.

**Conformant files** (7): ✓ correct data and names · err = raised an error · WRONG = returned data that differ from the truth · names = data right, names wrong

| case | cytopk | cytopk_strict | flowio | flowio_lenient | flowkit | flowkit_lenient | fcsparser | fcsparser_numpy1 | flowcore |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| clean_f32_le | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| clean_f64_be | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | err | ✓ | ✓ |
| clean_int_mixed_widths_be | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | err | ✓ | err |
| header_data_offsets_zero | ✓ | ✓ | err | ✓ | err | ✓ | ✓ | ✓ | ✓ |
| delimiter_pipe_escaped_values | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | err |
| lowercase_keywords | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | err | err | err |
| int16_bits_above_PnR | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **correct** | **7/7** | **7/7** | **6/7** | **7/7** | **6/7** | **7/7** | **4/7** | **6/7** | **4/7** |

**Non-conformant files** (23): ✓ correct data and names · err = raised an error · WRONG = returned data that differ from the truth · names = data right, names wrong

| case | cytopk | cytopk_strict | flowio | flowio_lenient | flowkit | flowkit_lenient | fcsparser | fcsparser_numpy1 | flowcore |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| empty_value | ✓ | err | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| consecutive_empty_values | ✓ | err | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| missing_final_delimiter | ✓ | err | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| doubled_final_delimiter | ✓ | err | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | err |
| trailing_bytes_in_text | ✓ | err | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | err |
| keyword_with_spaces | ✓ | err | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| duplicate_keyword | ✓ | err | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| latin1_text | ✓ | err | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| missing_PnE | ✓ | err | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| missing_NEXTDATA | ✓ | err | ✓ | ✓ | ✓ | ✓ | err | err | ✓ |
| version_fcs2_0 | ✓ | err | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| enddata_off_by_one | ✓ | err | err | ✓ | err | ✓ | ✓ | ✓ | ✓ |
| header_keyword_data_mismatch | ✓ | err | err | ✓ | err | ✓ | err | err | err |
| tot_too_large | ✓ | err | ✓ | ✓ | ✓ | ✓ | err | err | err |
| tot_too_small | ✓ | err | WRONG | WRONG | WRONG | WRONG | ✓ | ✓ | err |
| truncated_data | ✓ | err | err | err | err | err | err | err | err |
| header_offsets_only | ✓ | err | err | err | err | err | ✓ | ✓ | ✓ |
| spillover_wrong_count | ✓ | err | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | err |
| spillover_unknown_channel | ✓ | err | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| spillover_non_numeric | ✓ | err | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | err |
| spillover_truncated | ✓ | err | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | err |
| spillover_empty | ✓ | err | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | err |
| nextdata_beyond_eof | ✓ | err | err | err | err | err | ✓ | ✓ | err |
| **correct** | **23/23** | **0/23** | **17/23** | **19/23** | **17/23** | **19/23** | **19/23** | **19/23** | **12/23** |

**Real public files** (30): ok = returned data · = = same names and data as cytopk · ≠ = differs from cytopk · err = error

| file | source | cytopk | cytopk_strict | flowio | flowio_lenient | flowkit | flowkit_lenient | fcsparser | fcsparser_numpy1 | flowcore |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 100715.fcs | FlowIO | ok | err | = | = | = | = | err | = | = |
| 3FITC_4PE_004.fcs | FlowIO | ok | err | = | = | = | = | = | = | = |
| B01 KC-A-W---91-US.fcs | FlowIO | ok | err | = | = | = | = | err | = | = |
| G11.fcs | FlowIO | ok | err | = | = | = | = | = | = | err |
| coulter.lmd | FlowIO | ok | err | err | err | err | err | = | = | = |
| data1.fcs | FlowIO | ok | err | = | = | = | = | err | = | = |
| data_start_offset_discrepancy_example.fcs | FlowIO | ok | err | err | = | err | err | ≠ | ≠ | err |
| data_stop_offset_discrepancy_example.fcs | FlowIO | ok | err | err | = | err | err | = | = | err |
| variable_int_example.fcs | FlowIO | ok | err | = | = | err | err | = | = | err |
| M0_WM278_S1.fcs | FlowIO | ok | err | = | = | = | = | ≠ | ≠ | = |
| 109567.fcs | FlowKit | ok | err | = | = | = | = | err | = | = |
| 101_DEN084Y5_15_E01_008_clean.fcs | FlowKit | ok | = | = | = | = | = | = | = | = |
| index_sorted_example.fcs | FlowKit | ok | err | = | = | = | = | err | = | = |
| data_set_simple_line_100.fcs | FlowKit | ok | = | = | = | = | = | = | = | = |
| test_comp_example.fcs | FlowKit | ok | err | err | = | err | = | err | = | = |
| test_data_2d_01.fcs | FlowKit | ok | = | = | = | = | = | = | = | = |
| Cytek_xP5.fcs | fcsparser | ok | = | err | err | err | err | err | = | ≠ |
| Sample_Well_A02.fcs | fcsparser | ok | err | = | = | = | = | err | = | = |
| facs_diva_test.fcs | fcsparser | ok | err | = | = | = | = | err | = | = |
| FCS_3.0_Fortessa_PBS_Specimen_001_A1_A01.fcs | fcsparser | ok | err | = | = | = | = | err | = | = |
| Guava Muse.fcs | fcsparser | ok | = | err | err | err | err | = | = | = |
| HTS_BD_LSR_II_Mixed_Specimen_001_D6_D06.fcs | fcsparser | ok | err | = | = | = | = | err | = | = |
| EY_2013-07-19_PBS_FCS_2.0_Custom_Without_Add_Wel | fcsparser | ok | err | = | = | = | = | = | = | = |
| FCS3.0_Custom_Compatible.fcs | fcsparser | ok | = | = | = | = | = | = | = | = |
| EY_2013-07-19_PBS_FCS_3.1_Well_A1.001.fcs | fcsparser | ok | err | err | = | err | = | = | = | err |
| SG_2014-09-26_Duplicate_Names.fcs | fcsparser | ok | err | err | = | err | = | = | = | err |
| corrupted.fcs | fcsparser | err | err | err | err | err | err | err | err | err |
| cyflow_cube_8.fcs | fcsparser | ok | err | = | = | = | = | = | = | err |
| sample_header.fcs | fcsparser | err | err | err | err | err | err | err | err | err |
| fcs1_cleaned.lmd | fcsparser | ok | err | = | = | = | = | = | = | = |
| **returned data** | | **28/30** | **6/30** | **20/30** | **25/30** | **19/30** | **22/30** | **17/30** | **28/30** | **21/30** |

## Speed and memory

- **Timing:** time from path to a float32 NumPy matrix (events × parameters), median of 5 reads after one
  warm-up read in the same process. The page cache is warm, so this measures parsing and conversion, not
  the disk.
- **flowCore:** timed inside R (`read.FCS` + `exprs()` as double), which is not the same output type.
- **Memory:** extra peak RSS = max RSS (GNU time) of a process that imports the reader and reads once, minus one
  that only imports it.
- **CPU clock:** this machine lowers the clock of idle cores (`benchmarks/sbi_flow`, slowdown section). It
  applies to every reader equally, and reads of 10^6+ events last long enough for the clock to ramp up.

**Time to a float32 NumPy matrix** (median of 5 warm-cache reads; flowCore: to an R double matrix):

| file | MB | cytopk | cytopk_1thread | flowio | flowkit | fcsparser | flowcore | fastest other / cytopk | fastest other / cytopk_1thread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_100000x20.fcs | 8.0 | 0.7 ms | 2.7 ms | 7.4 ms | 8.7 ms | 6.0 ms | 34.0 ms | 8.38× | 2.22× |
| gen_100000x40.fcs | 16.0 | 0.8 ms | 3.0 ms | 14.1 ms | 17.0 ms | 12.1 ms | 111.0 ms | 15.55× | 4.01× |
| gen_1000000x20.fcs | 80.0 | 3.7 ms | 30.5 ms | 83.4 ms | 130.5 ms | 75.3 ms | 443.0 ms | 20.31× | 2.47× |
| gen_1000000x40.fcs | 160.0 | 10.3 ms | 60.3 ms | 162.6 ms | 168.5 ms | 183.5 ms | 976.0 ms | 15.80× | 2.70× |
| gen_10000000x20.fcs | 800.0 | 32.0 ms | 382.2 ms | 998.0 ms | 1.30 s | 737.3 ms | 3.79 s | 23.01× | 1.93× |
| gen_10000000x40.fcs | 1,600.0 | 64.3 ms | 656.1 ms | 1.66 s | 1.78 s | 1.70 s | 7.33 s | 25.80× | 2.53× |
| 100715.fcs | 4.2 | 0.4 ms | 0.4 ms | 4.4 ms | 4.4 ms | err | 27.0 ms | 9.75× | 10.47× |
| 3FITC_4PE_004.fcs | 0.8 | 0.8 ms | 0.9 ms | 33.8 ms | 81.0 ms | 1.9 ms | 37.0 ms | 2.32× | 2.25× |
| B01 KC-A-W---91-US.fcs | 0.1 | 0.3 ms | 0.4 ms | 3.3 ms | 3.4 ms | err | 5.0 ms | 12.13× | 9.31× |
| G11.fcs | 0.3 | 0.4 ms | 0.1 ms | 0.6 ms | 1.6 ms | 0.9 ms | err | 1.76× | 4.98× |
| coulter.lmd | 0.9 | 0.4 ms | 0.7 ms | err | err | 0.9 ms | 15.0 ms | 2.01× | 1.27× |
| data1.fcs | 0.2 | 0.3 ms | 0.5 ms | 13.6 ms | 8.2 ms | err | 12.0 ms | 24.28× | 15.51× |
| data_start_offset_discrepancy_example.fcs | 0.0 | 0.4 ms | 0.2 ms | err | err | 0.9 ms | err | 2.44× | 4.35× |
| data_stop_offset_discrepancy_example.fcs | 0.0 | 0.2 ms | 0.2 ms | err | err | 0.9 ms | err | 4.81× | 4.77× |
| variable_int_example.fcs | 0.0 | 0.2 ms | 0.2 ms | 0.4 ms | err | 0.9 ms | err | 2.51× | 2.13× |
| M0_WM278_S1.fcs | 13.5 | 19.6 ms | 20.1 ms | 362.0 ms | 371.7 ms | 11.3 ms | 330.0 ms | 0.58× | 0.56× |
| 109567.fcs | 10.3 | 0.7 ms | 2.8 ms | 9.5 ms | 11.9 ms | err | 51.0 ms | 13.57× | 3.39× |
| 101_DEN084Y5_15_E01_008_clean.fcs | 17.4 | 0.9 ms | 4.6 ms | 8.1 ms | 13.3 ms | 11.2 ms | 82.0 ms | 9.15× | 1.77× |
| index_sorted_example.fcs | 0.0 | 0.1 ms | 0.1 ms | 0.3 ms | 0.7 ms | err | 3.0 ms | 3.18× | 2.18× |
| data_set_simple_line_100.fcs | 0.0 | 0.0 ms | 0.0 ms | 0.1 ms | 0.6 ms | 0.6 ms | 2.0 ms | 5.50× | 3.80× |
| test_comp_example.fcs | 0.8 | 0.2 ms | 0.2 ms | err | err | err | 6.0 ms | 36.11× | 27.66× |
| test_data_2d_01.fcs | 0.1 | 0.0 ms | 0.0 ms | 0.1 ms | 0.6 ms | 0.6 ms | 3.0 ms | 7.88× | 4.37× |
| Cytek_xP5.fcs | 1.0 | 0.5 ms | 0.9 ms | err | err | err | 259.0 ms | 548.42× | 295.12× |
| Sample_Well_A02.fcs | 0.6 | 1.1 ms | 1.3 ms | 18.4 ms | 22.0 ms | err | 28.0 ms | 16.10× | 14.07× |
| facs_diva_test.fcs | 4.0 | 0.5 ms | 0.6 ms | 5.1 ms | 4.9 ms | err | 23.0 ms | 10.45× | 8.38× |
| FCS_3.0_Fortessa_PBS_Specimen_001_A1_A01.fcs | 0.5 | 0.1 ms | 0.2 ms | 0.8 ms | 0.9 ms | err | 4.0 ms | 7.56× | 5.38× |
| Guava Muse.fcs | 8.5 | 0.1 ms | 0.1 ms | err | err | 0.7 ms | 3.0 ms | 7.63× | 5.34× |
| HTS_BD_LSR_II_Mixed_Specimen_001_D6_D06.fcs | 0.7 | 0.2 ms | 0.2 ms | 1.0 ms | 1.1 ms | err | 5.0 ms | 5.41× | 5.96× |
| EY_2013-07-19_PBS_FCS_2.0_Custom_Without_Add | 0.6 | 0.2 ms | 0.1 ms | 1.0 ms | 0.9 ms | 1.0 ms | 5.0 ms | 4.12× | 6.53× |
| FCS3.0_Custom_Compatible.fcs | 0.6 | 0.2 ms | 0.1 ms | 1.0 ms | 0.8 ms | 0.9 ms | 5.0 ms | 3.35× | 5.82× |
| EY_2013-07-19_PBS_FCS_3.1_Well_A1.001.fcs | 0.9 | 0.2 ms | 0.3 ms | err | err | 0.9 ms | err | 3.94× | 2.88× |
| SG_2014-09-26_Duplicate_Names.fcs | 0.3 | 0.1 ms | 0.1 ms | err | err | 0.8 ms | err | 7.56× | 7.76× |
| corrupted.fcs | 0.0 | err | err | err | err | err | err | — | — |
| cyflow_cube_8.fcs | 0.1 | 0.4 ms | 0.5 ms | 1.0 ms | 1.0 ms | 0.9 ms | err | 2.68× | 2.07× |
| sample_header.fcs | 0.0 | err | err | err | err | err | err | — | — |
| fcs1_cleaned.lmd | 0.7 | 1.5 ms | 1.5 ms | 24.8 ms | 26.6 ms | 1.9 ms | 38.0 ms | 1.25× | 1.25× |

**Extra peak RSS of one read** (MB; max RSS of import + read minus import only):

| file | cytopk | cytopk_1thread | flowio | flowkit | fcsparser | flowcore |
|---|---:|---:|---:|---:|---:|---:|
| gen_100000x20.fcs | 16 | 15 | 23 | 29 | 20 | 40 |
| gen_100000x40.fcs | 31 | 31 | 46 | 52 | 35 | 101 |
| gen_1000000x20.fcs | 153 | 153 | 229 | 242 | 158 | 467 |
| gen_1000000x40.fcs | 305 | 306 | 458 | 471 | 311 | 620 |
| gen_10000000x20.fcs | 1,527 | 1,526 | 2,289 | 2,370 | 1,532 | 3,061 |
| gen_10000000x40.fcs | 3,052 | 3,052 | 4,578 | 4,659 | 3,057 | 6,113 |

**500 small files** (10^4 events × 20 each), seconds (median of 3):

| mode | seconds |
|---|---:|
| cytopk_many | 0.06 |
| cytopk | 0.15 |
| cytopk_1thread | 0.13 |
| flowio | 0.23 |
| flowkit | 1.30 |
| fcsparser | 0.68 |
| cytopk_joblib | 0.98 |
| flowio_joblib | 1.02 |
| flowkit_joblib | 2.12 |
| fcsparser_joblib | 1.45 |

## Reproduce

```sh
uv venv .venv-fcs --python 3.13 && uv pip install --python .venv-fcs/bin/python -r benchmarks/fcs/requirements.lock
(cd python && VIRTUAL_ENV=$PWD/../.venv-fcs ../.venv-fcs/bin/maturin develop --release)
uv venv .venv-fcs-np1 --python 3.12 && uv pip install --python .venv-fcs-np1/bin/python -r benchmarks/fcs/requirements-numpy1.lock
# flowCore (optional): BiocManager::install("flowCore") into a user library, then export R_LIBS_USER=<that library>
.venv-fcs/bin/python scripts/fetch_test_data.py
.venv-fcs/bin/python benchmarks/fcs/corpus.py
.venv-fcs/bin/python benchmarks/fcs/bench_robustness.py
.venv-fcs/bin/python benchmarks/fcs/bench_speed.py
.venv-fcs/bin/python benchmarks/fcs/make_tables.py
```
