# Draft: fcsparser uses the HEADER DATA offset even when it points inside TEXT

*Target: github.com/eyurtsev/fcsparser. Draft only, not posted. All values below are from
`repro_results.json` → `offset`.*

**Versions.** fcsparser 0.2.4, Python 3.12.3, NumPy 1.26.4, pandas 2.2.3.

## Summary

When the HEADER's DATA start disagrees with `$BEGINDATA` and points inside the TEXT segment, `fcsparser.parse`
reads the DATA from the HEADER offset. It raises no error or warning, and the "events" it returns are TEXT
bytes.

## Minimal file

`files/header_data_offset_inside_text.fcs` (538 bytes, generated) is FCS 3.0, `$DATATYPE I`, 4 parameters ×
16 bit, `$BYTEORD 4,3,2,1`, 3 events.

- **HEADER:** `FCS3.0         256     495     474     497       0       0`, so TEXT is 256–495 and DATA is
  474–497.
- **TEXT:** `$BEGINDATA/514/`, `$ENDDATA/537/`.
- **DATA:** 514–537 = 24 bytes = `$TOT` 3 × 8 bytes. The bytes there are
  `00 01 00 02 00 03 00 04 03 e8 07 d0 0b b8 ff ff 00 07 00 08 00 09 00 0a`.

```python
import fcsparser
meta, df = fcsparser.parse("header_data_offset_inside_text.fcs", reformat_meta=True, channel_naming="$PnN")
df.to_numpy()
```

**Expected** (the values written):

```
[[   1,    2,    3,     4],
 [1000, 2000, 3000, 65535],
 [   7,    8,    9,    10]]
```

**Actual** (fcsparser 0.2.4), with no warning:

```
[[17473, 21569, 12085, 12596],
 [12068, 17742, 17476, 16724],
 [16687, 13619, 14127,  8224]]
```

These are ASCII bytes from TEXT. For example, 17473 = `0x4441` = `"DA"`, from `$BEGINDATA`.

## Ground truth

FCS 3.1 §3.2: the DATA segment cannot overlap TEXT, and the HEADER offsets must equal `$BEGINDATA` and
`$ENDDATA` when DATA fits below 99,999,999. Here the HEADER range starts inside TEXT (474 < 495), while the
keyword range is outside TEXT and has exactly `$TOT` × bytes per event (24) bytes. So the keyword range is
the right one.

Independent agreement on the public file: flowio 1.4.0 with `ignore_offset_discrepancy=True` and cytopk return
the same values. flowio (default) and flowCore 2.24.0 both refuse the file and name the discrepancy. Either
behaviour is fine; silent use of the HEADER offset is not.

## Public file

FlowIO test file `data/fcs_files/data_start_offset_discrepancy_example.fcs` (FlowIO commit 83d28a2, SHA-256
`414fdccae26a65996324261ab028fb1c762e448313ebcf6207894b4dc43eef47`).

- **HEADER:** TEXT 74–6080, DATA 5555–6188.
- **TEXT:** `$BEGINDATA 00006081`, `$ENDDATA 000000006188`, `$TOT 2`, 26 parameters (54 bytes per event, so 108
  bytes = 6081–6188).
- **What fcsparser returns:** the first row starts `12332, 12332, 12332, 12332`. That is `0x302C` = `"0,"`, the
  TEXT bytes `,0,0,0,0` at offset 5555.
- **What the others return:** cytopk and lenient flowio both start `49135, 61373, 48575, 49135`.
- **Caveat:** the DATA bytes in this public test file are themselves `ef bf bd …` (UTF-8 replacement
  characters). The file appears to have been text-processed at some point, so its values are not meaningful.
  The generated file above is the cleaner reproducer.

## Suggested behaviour

When the HEADER and keyword offsets disagree, either raise an error (as flowio and flowCore do), or prefer the
range that lies outside TEXT and matches `$TOT` × bytes per event, and warn.
