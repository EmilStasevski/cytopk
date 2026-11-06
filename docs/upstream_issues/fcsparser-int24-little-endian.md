# Draft: fcsparser mis-decodes little-endian 24-bit integer DATA

*Target: github.com/eyurtsev/fcsparser. Draft only, not posted. All values below are from
`repro_results.json` → `int24.generated_file_little_endian`.*

**Versions.** fcsparser 0.2.4, Python 3.12.3, NumPy 1.26.4, pandas 2.2.3.

**How it was found.** This case was not in the benchmarks. It came up in the little-endian control file
written for the flowCore 24-bit case, so no public file with this layout was tested.

## Summary

For `$DATATYPE I`, `$PnB 24` and `$BYTEORD 1,2,3,4`, fcsparser returns wrong values without an error or
warning. The same values written big-endian are read correctly.

## Minimal file

`files/int24_littleendian.fcs` (562 bytes, generated) is FCS 3.0, 4 parameters × 24 bit, `$PnR 16777216`,
`$BYTEORD 1,2,3,4`, 3 events. Its DATA bytes for event 0 are `01 00 00 | ff 00 00 | 00 01 00 | ff ff 00`.

```python
import fcsparser
meta, df = fcsparser.parse("int24_littleendian.fcs", reformat_meta=True, channel_naming="$PnN")
df.to_numpy()
```

| | event 0 | event 1 | event 2 |
|---|---|---|---|
| written (truth) | 1, 255, 256, 65535 | 65536, 1000000, 8388608, 16777215 | 0, 2, 4, 8 |
| fcsparser 0.2.4 | **256, 65280, 65536, 16776960** | **0, 4341760, 0, 16776960** | 0, **512, 1024, 2048** |

Most values are the truth × 256, truncated to 24 bits (1 → 256, 255 → 65280, 2 → 512). This looks like the
3-byte word is padded to 4 bytes on the wrong side before the little-endian conversion.

cytopk and flowCore 2.24.0 return the written values. flowio 1.4.0 raises an error (`cannot reshape array of
size 1 into shape (4)`).

## Ground truth

FCS 3.1: `$BYTEORD 1,2,3,4` means little-endian for every integer width. Independent agreement: cytopk and
flowCore.
