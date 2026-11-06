# Draft: flowCore reads big-endian 24-bit integer DATA as little-endian

*Target: github.com/RGLab/flowCore. Draft only, not posted. All values below are from
`repro_results.json` → `int24`.*

**Versions.** flowCore 2.24.0, R version 4.6.2.

## Summary

For `$DATATYPE I`, `$PnB 24` and `$BYTEORD 4,3,2,1`, `read.FCS` returns values whose three bytes are reversed,
then masked by `$PnR`. It raises no error or warning. The same values written little-endian
(`$BYTEORD 1,2,3,4`) are read correctly, so the problem is specific to big-endian 24-bit words.

## Minimal files

`files/int24_bigendian.fcs` and `files/int24_littleendian.fcs` (562 bytes each, generated) are FCS 3.0,
4 parameters × 24 bit, `$PnR 16777216`, 3 events. They are identical except for `$BYTEORD` and the byte order
of DATA.

```r
library(flowCore)
exprs(read.FCS("int24_bigendian.fcs", transformation = FALSE, truncate_max_range = FALSE))
```

| | event 0 | event 1 | event 2 |
|---|---|---|---|
| written (truth) | 1, 255, 256, 65535 | 65536, 1000000, 8388608, 16777215 | 0, 2, 4, 8 |
| big-endian, flowCore | **65536, 16711680**, 256, **16776960** | **1, 4211215, 128**, 16777215 | 0, **131072, 262144, 524288** |
| little-endian, flowCore | 1, 255, 256, 65535 | 65536, 1000000, 8388608, 16777215 | 0, 2, 4, 8 |

Big-endian DATA bytes for event 0: `00 00 01 | 00 00 ff | 00 01 00 | 00 ff ff`. flowCore's 65536 is
`0x010000`, the bytes `00 00 01` read little-endian. Its 16711680 is `0xff0000`, the bytes `00 00 ff` read
little-endian.

cytopk returns the written values for both files. fcsparser 0.2.4 also returns them for the big-endian file
(it has its own problem with the little-endian one; see `fcsparser-int24-little-endian.md`).

## Public file

fcsparser test file `fcsparser/tests/data/FlowCytometers/Cytek_xP5/Cytek_xP5.fcs` (fcsparser commit da70aaa,
SHA-256 `dd6aaa9a034c40ab5f3a2ce1ea4c8eab7d00a053426917e07a09f8d8d76ca51a`). It is FCS 3.0, `$DATATYPE I`,
8 × 24 bit, `$BYTEORD 4,3,2,1`, with `$PnR` 30000 for TIME and 1024 for the others.

- **First event bytes:** `00 00 00 00 01 1e 00 00 a4 00 00 9a 00 00 36 00 01 d6 00 03 ff 00 03 02`.
- **Decoded by hand** (big-endian, then the `$PnR` mask): 0, 286, 164, 154, 54, 470, 1023, 770. cytopk and
  fcsparser (NumPy 1 environment) both return this.
- **flowCore returns:** 0, 256, 0, 0, 0, 256, 768, 768. That is each word read little-endian, then masked
  with 1023. For example, `00 01 1e` gives `0x1e0100`, and `0x1e0100 & 0x3ff` = 256.

## Ground truth

FCS 3.1: `$BYTEORD 4,3,2,1` means big-endian (most significant byte first) for every integer width. `$PnB 24`
words are 3 bytes, and the values are masked to the smallest all-ones mask covering `$PnR` − 1.

Independent agreement: cytopk and fcsparser on the big-endian files; cytopk and flowCore itself on the
little-endian file.
