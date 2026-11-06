"""Artificial FCS corpus: conformant files plus one file per known real-world deviation, with ground truth.

The Bras & van der Velden (2020) test files need a manual, CAPTCHA-protected download (scripts/fetch_test_data.py),
so this corpus reproduces the deviation categories that paper reports. Each case starts from a file written by
the cytopk writer (`cytopk.write_fcs`). The deviation is then injected at the byte level, after re-laying the
file out so that DATA starts at byte 8192 and TEXT edits cannot move it.

Ground truth per case: the channel names and data matrix the file is meant to hold. Where the specification
decides an ambiguous case, truth follows the specification (for example a $TOT smaller than the DATA segment:
the first $TOT events). `kind` is "conformant" or "deviation".

    python corpus.py            # -> data/fcs_corpus/*.fcs + truth.npz + cases.json (git-ignored)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "fcs_corpus"
DATA_AT = 8192
N, P = 1000, 8
NAMES = ["FSC-A", "SSC-A", "FL1-A", "FL2-A", "FL3-A", "FL4-A", "FL5-A", "Time"]


def written(data, **kw) -> bytes:
    import cytopk
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.fcs"
        cytopk.write_fcs(p, data, kw.pop("names", NAMES), **kw)
        return p.read_bytes()


def split(b: bytes):
    """(version, delimiter, [(k, v)], data bytes) of a single-dataset file from the cytopk writer."""
    t0, t1 = int(b[10:18]), int(b[18:26])
    text = b[t0:t1 + 1]
    d = text[:1]
    body = text[1:-1].replace(d + d, b"\0ESC\0")
    toks = [t.replace(b"\0ESC\0", d) for t in body.split(d)]
    pairs = list(zip(toks[0::2], toks[1::2]))
    kv = {k: v for k, v in pairs}
    db, de = int(kv[b"$BEGINDATA"]), int(kv[b"$ENDDATA"])
    return b[:6], d, pairs, (b[db:de + 1] if de else b"")


class Case:
    def __init__(self, base: bytes):
        self.version, self.delim, self.pairs, self.data = split(base)
        self.header_data = None  # override (begin, end)
        self.text_edit = None    # callable(bytes) -> bytes, applied to the encoded TEXT
        self.tail = b""          # appended after DATA

    def set(self, k: bytes, v: bytes):
        for i, (a, _) in enumerate(self.pairs):
            if a.upper() == k.upper():  # keywords are case-insensitive; keep the file's spelling
                self.pairs[i] = (a, v)
                return
        self.pairs.append((k, v))

    def drop(self, k: bytes):
        self.pairs = [p for p in self.pairs if p[0] != k]

    def text(self) -> bytes:
        d = self.delim
        t = d + b"".join(x.replace(d, d + d) + d for kv in self.pairs for x in kv)
        return self.text_edit(t) if self.text_edit else t

    def build(self) -> bytes:
        n = len(self.data)
        self.set(b"$BEGINDATA", str(DATA_AT if n else 0).encode())
        self.set(b"$ENDDATA", str(DATA_AT + n - 1 if n else 0).encode())
        t = self.text()
        assert 58 + len(t) <= DATA_AT
        hd = self.header_data or ((DATA_AT, DATA_AT + n - 1) if n else (0, 0))
        h = self.version + b"    " + b"".join(f"{v:>8}".encode() for v in (58, 57 + len(t), *hd, 0, 0))
        return (h + t).ljust(DATA_AT, b" ") + self.data + self.tail


def sub(old: bytes, new: bytes):
    def f(t):
        assert old in t, old
        return t.replace(old, new, 1)
    return f


def cases():
    rng = np.random.default_rng(2020)
    f32 = (rng.standard_normal((N, P)) * 1000).astype(np.float32)
    ints = rng.integers(0, 1024, (N, P)).astype(np.float64)
    out = []

    def add(name, kind, b, truth, names=NAMES, note=""):
        out.append({"name": name, "kind": kind, "bytes": b, "truth": truth, "names": names, "note": note})

    base = written(f32)
    # ---- conformant
    add("clean_f32_le", "conformant", base, f32)
    add("clean_f64_be", "conformant", written(f32.astype(np.float64), datatype="D", byteorder="big"), f32)
    add("clean_int_mixed_widths_be", "conformant",
        written(ints, datatype="I", byteorder="big", bits=[16, 32, 16, 16, 32, 16, 16, 32]), ints)
    c = Case(base); c.header_data = (0, 0)
    add("header_data_offsets_zero", "conformant", c.build(), f32, note="offsets only in $BEGINDATA/$ENDDATA")
    add("delimiter_pipe_escaped_values", "conformant",
        written(f32, delimiter="|", keywords={"$COM": "a|b||c", "$DATE": "01|02|2020"}), f32)
    c = Case(base); c.pairs = [(k.lower(), v) for k, v in c.pairs]
    add("lowercase_keywords", "conformant", c.build(), f32, note="keywords are case-insensitive")
    # 16-bit ints with values above $PnR: the spec masks them with $PnR - 1 (here 1023).
    raw = rng.integers(0, 65536, (N, P)).astype(np.float64)
    c = Case(written(raw, datatype="I", bits=[16] * P))
    for j in range(P):
        c.set(f"$P{j + 1}R".encode(), b"1024")
    add("int16_bits_above_PnR", "conformant", c.build(), raw % 1024,
        note="spec: mask with $PnR - 1; readers that do not mask return the raw 16-bit values")

    # ---- deviations: TEXT
    c = Case(base); c.set(b"$CYT", b"X"); c.text_edit = sub(b"/$CYT/X/", b"/$CYT//")
    add("empty_value", "deviation", c.build(), f32)
    c = Case(base); c.set(b"$CYT", b"X"); c.set(b"$OP", b"X"); c.set(b"$SRC", b"X")
    c.text_edit = sub(b"/$CYT/X/$OP/X/$SRC/X/", b"/$CYT//$OP//$SRC//")
    add("consecutive_empty_values", "deviation", c.build(), f32)
    c = Case(base); c.text_edit = lambda t: t[:-1]
    add("missing_final_delimiter", "deviation", c.build(), f32)
    c = Case(base); c.text_edit = lambda t: t + b"/"
    add("doubled_final_delimiter", "deviation", c.build(), f32)
    c = Case(base); c.text_edit = lambda t: t + b"    \0\0\0\x01\x02"
    add("trailing_bytes_in_text", "deviation", c.build(), f32)
    c = Case(base); c.pairs.insert(0, (b" $COM ", b"spaces"))
    add("keyword_with_spaces", "deviation", c.build(), f32)
    c = Case(base); c.set(b"$COM", b"first"); c.pairs.append((b"$COM", b"second"))
    add("duplicate_keyword", "deviation", c.build(), f32)
    c = Case(base); c.set(b"$COM", "café µ".encode("latin-1"))
    add("latin1_text", "deviation", c.build(), f32)
    c = Case(base); c.drop(b"$P1E"); c.drop(b"$P2E")
    add("missing_PnE", "deviation", c.build(), f32)
    c = Case(base); c.drop(b"$NEXTDATA")
    add("missing_NEXTDATA", "deviation", c.build(), f32)
    c = Case(base); c.version = b"FCS2.0"
    add("version_fcs2_0", "deviation", c.build(), f32, note="FCS 2.0 header on 3.1 content")
    # ---- deviations: DATA offsets / sizes
    c = Case(base); c.data = c.data + b"\0"
    add("enddata_off_by_one", "deviation", c.build(), f32)
    c = Case(base); n = len(c.data); c.header_data = (DATA_AT + 4, DATA_AT + n + 3)
    add("header_keyword_data_mismatch", "deviation", c.build(), f32,
        note="HEADER DATA range shifted by 4 bytes; $BEGINDATA/$ENDDATA correct")
    c = Case(base); c.set(b"$TOT", str(N + 500).encode())
    add("tot_too_large", "deviation", c.build(), f32)
    c = Case(base); c.set(b"$TOT", str(N - 200).encode())
    add("tot_too_small", "deviation", c.build(), f32[: N - 200], note="truth: the first $TOT events")
    c = Case(base); b = c.build()
    add("truncated_data", "deviation", b[: len(b) - 1000], f32[: N - 32], note="last 1000 bytes cut")
    add("header_offsets_only", "deviation", _drop_data_keywords(Case(base)), f32,
        note="$BEGINDATA/$ENDDATA absent (required in 3.x), HEADER offsets valid")
    # ---- deviations: $SPILLOVER
    for name, val in [("spillover_wrong_count", b"3,FL1-A,FL2-A,1,0.1,0.2,1"),
                      ("spillover_unknown_channel", b"2,FL1-A,NOPE,1,0.1,0.2,1"),
                      ("spillover_non_numeric", b"2,FL1-A,FL2-A,1,x,0.2,1"),
                      ("spillover_truncated", b"2,FL1-A,FL2-A,1,0.1")]:
        c = Case(base); c.set(b"$SPILLOVER", val)
        add(name, "deviation", c.build(), f32)
    c = Case(base); c.set(b"$SPILLOVER", b"X"); c.text_edit = sub(b"/$SPILLOVER/X/", b"/$SPILLOVER//")
    add("spillover_empty", "deviation", c.build(), f32)
    # ---- deviations: $NEXTDATA
    c = Case(base); c.set(b"$NEXTDATA", b"999999999")
    add("nextdata_beyond_eof", "deviation", c.build(), f32)
    return out


def _drop_data_keywords(c: Case) -> bytes:
    n = len(c.data)
    c.drop(b"$BEGINDATA"); c.drop(b"$ENDDATA")
    t = c.text()
    h = c.version + b"    " + b"".join(f"{v:>8}".encode() for v in (58, 57 + len(t), DATA_AT, DATA_AT + n - 1, 0, 0))
    return (h + t).ljust(DATA_AT, b" ") + c.data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, truth = [], {}
    for c in cases():
        (OUT / f"{c['name']}.fcs").write_bytes(c["bytes"])
        truth[c["name"]] = c["truth"]
        meta.append({k: c[k] for k in ("name", "kind", "names", "note")} | {"events": int(c["truth"].shape[0])})
    np.savez_compressed(OUT / "truth.npz", **truth)
    (OUT / "cases.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(meta)} cases -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
