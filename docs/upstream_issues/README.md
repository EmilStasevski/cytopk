# Upstream issue drafts (not posted)

These are silent wrong-data cases in other FCS readers, found by `benchmarks/fcs`. A reader returned values
without an error, and the values are wrong. **Nothing here has been posted.** Each file is a draft for the
maintainers to review.

| Draft | Reader | Case | Found by |
|---|---|---|---|
| [fcsparser-header-data-offset.md](fcsparser-header-data-offset.md) | fcsparser 0.2.4 | HEADER DATA offset that points into TEXT is used | robustness benchmark |
| [fcsparser-float32-int32.md](fcsparser-float32-int32.md) | fcsparser 0.2.4 | default `dtype='float32'` rounds 32-bit integers above 2^24 | robustness benchmark |
| [flowcore-int24-big-endian.md](flowcore-int24-big-endian.md) | flowCore 2.24.0 | big-endian 24-bit integers read as little-endian | robustness benchmark |
| [fcsparser-int24-little-endian.md](fcsparser-int24-little-endian.md) | fcsparser 0.2.4 | little-endian 24-bit integers mis-decoded | the little-endian control file written for the flowCore case |

## Reproducing

```sh
python scripts/fetch_test_data.py                  # public files, pinned by commit and SHA-256
R_LIBS_USER=<flowCore library> .venv-fcs/bin/python docs/upstream_issues/repro.py
```

`repro.py` writes the small generated files in `files/` and records every value quoted in the drafts in
`repro_results.json`, along with the reader versions and the git SHA. It runs fcsparser in a NumPy 1.26
environment (`.venv-fcs-np1`), because fcsparser 0.2.4 fails on NumPy 2 (`ndarray.newbyteorder` was removed).
That failure is loud and is not one of these cases.

## How the ground truth was established

- **Generated files:** the values the script wrote, together with the hex dump of the DATA bytes.
- **Public files:** the bytes decoded by hand per FCS 3.1 (Spidlen et al. 2010):
  - §3.2 for HEADER/TEXT/DATA offsets;
  - `$BYTEORD`, `$PnB`, and `$PnR` for the integer bit mask.
- **Independent agreement:** at least one other reader gives the same values as cytopk in every case (named in
  each draft).

Versions tested: fcsparser 0.2.4 (Python 3.12.3, NumPy 1.26.4, pandas 2.2.3), flowCore 2.24.0 (R 4.6.2),
flowio 1.4.0 and cytopk 0.1.0.dev0. They are recorded in `repro_results.json` → `meta.versions_readers`.
