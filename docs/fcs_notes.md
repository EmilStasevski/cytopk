# FCS notes (cytopk-fcs)

Authority: FCS 3.1 specification (Spidlen et al. 2010, Cytometry A 77:97). The real-world deviations below come
from Bras & van der Velden 2020 ("Robust FCS Parsing: Exploring 211,359 Public Files", Cytometry A 97:1180).
`crates/cytopk-fcs/tests/quirks.rs` has one test per row: strict mode must reject the file, and lenient mode
must read the right data and report the warning kind shown.

## Modes

- **Strict** (`ReadOptions::strict()`, `read_fcs(path, strict=True)`): anything the specification forbids is
  an `FcsError`, and `warnings` is always empty.
- **Lenient** (the default) repairs the problems below and records each repair as an `FcsWarning`, with a
  kind, message, byte offset and keyword. It never prints.
- **Both modes error** on things that cannot be repaired: no HEADER, an unknown version, ASCII data
  (`$DATATYPE A`), non-list modes, byte orders other than `1,2,3,4` / `4,3,2,1`, integer widths other than
  widths that are not whole bytes (8–64 bits), no locatable DATA, and a first dataset that cannot be read.

## Lenient repairs

| Problem | Lenient behaviour | Warning kind |
|---|---|---|
| Version `FCS2.0` or `FCS3.2` | read with the 3.1 rules | Header |
| HEADER filler not spaces, field left-justified or non-numeric | parse what is there; non-numeric = 0 | Header |
| Empty value `/$KEY//$NEXT/...` | a doubled delimiter right after a *keyword* = empty value | EmptyValue |
| Missing closing delimiter | last value kept | Delimiter |
| Doubled closing delimiter `.../v//` | extra delimiter ignored | Delimiter |
| Bytes after the last delimiter (padding, DATA bytes) | ignored | Delimiter |
| Spaces around a keyword | trimmed | Delimiter |
| Duplicate keyword | first value kept | DuplicateKeyword |
| TEXT not UTF-8 | decoded as Latin-1 | Encoding |
| TEXT end beyond the file | clipped (TEXT is never clipped where the HEADER says DATA starts: that start is itself a known error) | TextOffsets |
| HEADER and `$BEGINDATA/$ENDDATA` disagree | first plausible range (inside the file, outside TEXT, ≥ `$TOT` events), keywords before HEADER | DataOffsets |
| DATA end beyond the file (truncated file) | clipped; complete events read | DataOffsets |
| DATA length ≠ `$TOT` × bytes/event (off-by-one `$ENDDATA`, wrong `$TOT`) | min(`$TOT`, complete events present) | DataSize |
| `$TOT` missing | events = DATA bytes / bytes per event | MissingKeyword |
| `$PnN` missing/empty | name `P<n>` | MissingKeyword |
| `$PnE` missing (common in FCS 2.0) | `0,0` (linear) | MissingKeyword |
| `$PnR` missing/non-numeric | range 0 (no masking) | MissingKeyword / BadValue |
| `$PnB` ≠ 32 for F (≠ 64 for D) | the type's width | BadValue |
| Numeric keyword with spaces or `1000.0` | parsed | BadValue |
| Malformed `$SPILLOVER`/`$SPILL`/`$COMP` (count, size, names not in `$PnN`, non-numbers, empty) | matrix ignored, the file is still read | Spillover |
| Invalid supplemental TEXT | ignored | SupplementalText |
| Invalid ANALYSIS offsets | ignored | Analysis |
| `$NEXTDATA` invalid / next dataset unreadable | earlier datasets returned | NextData |

### Empty values versus escaped delimiters

The specification forbids empty values, so in a conforming file a doubled delimiter is always an escape. Real
files contain empty values, and FlowCore and FlowJo both mistake them for escapes (Bras & van der Velden 2020).
In lenient mode, a doubled delimiter is an empty value only when it **directly follows a keyword**, which is
where a missing value leaves one. Elsewhere it stays an escape, so a date like `01//02//2020` (delimiter `/`)
still reads as `01/02/2020`.

One legal case this misreads: a value that *starts* with the delimiter. `/KEY///x/` is spec-legal
(`KEY` = `/x`) but reads as an empty `KEY` followed by keyword `/x`. To avoid writing such files, the cytopk
writer rejects values that start with the delimiter. In strict mode, a keyword that contains the delimiter is
an error, which catches the empty-value case.

## Data

- List mode only, event-major (`events × parameters` row-major, the file layout).
- `F` → float32 and `D` → float64, bit-exact.
- `I` → float64 (any whole-byte width, 8–64 bits; mixed widths allowed), exact up to 53 bits. Each value is masked with the smallest all-ones mask covering `$PnR - 1`;
  when `$PnR` is missing, zero, or larger than the word, every bit of the word is kept.
- Values are returned **as stored**. `$PnE` (log amplification) and `$PnG` (gain) are reported per channel but
  not applied. Applying `$PnE` belongs to the transforms milestone (M6). It is untested on real data with
  `$PnE ≠ 0,0` and is marked as such there.
- **Allocation guard:** the number of events decoded is min(`$TOT`, bytes present / bytes per event), so a file
  claiming 10^12 events allocates only what its bytes can fill. `$PAR` above the number of keywords (or
  100,000) is an error.

## Spillover convention

`$SPILLOVER`: `n, name_1..name_n, s_11, s_12, ..., s_nn`, row-major. Row i, column j is the fraction of the
signal of fluorochrome i (primary channel name_i) that appears in channel name_j, so the diagonal is 1.
Compensation is `x_comp = x · S⁻¹`, with `x` the event row restricted to the named channels in matrix order.
cytopk-fcs parses the matrix and does not invert it. Validating compensation against FlowKit or flowCore on a
real compensated file is part of M6.

## Writer

FCS 3.1 (or 3.0), list mode, F (default), D or I (per-channel 8/16/32/64 bits), either byte order, any
ASCII 1–126 delimiter, and several datasets chained with `$NEXTDATA`.
- **Layout:** each dataset is HEADER, TEXT, DATA. Offsets are relative to the dataset start.
- **Large DATA:** when DATA ends beyond byte 99,999,999, the HEADER DATA fields are 0 and only
  `$BEGINDATA`/`$ENDDATA` locate it.
- **No CRC** is written (the specification allows its absence).
- **Rejected inputs:** empty keywords or values, values that start with the delimiter, keywords the writer
  sets itself, and non-integer values for I.

## Memory mapping

`read_path` and `read_fcs` map the file read-only. If another process truncates the file while it is mapped,
the reader can fault (SIGBUS). Every mmap-based reader has this risk. Files that cannot be mapped, and empty
files, are read into memory instead.
