"""Reproducers for three silent-wrong-data cases found by benchmarks/fcs (drafts only; nothing is posted).

For each case: a public file (from scripts/fetch_test_data.py, pinned by SHA-256) and a generated equivalent
written here, the ground truth, and what each reader returns. Ground truth for generated files is the values we
wrote; for the public files it is the bytes decoded by hand per the FCS 3.1 spec (the hex dump is recorded),
plus agreement between independent readers.

    python docs/upstream_issues/repro.py      # -> docs/upstream_issues/repro_results.json
Readers: fcsparser runs in .venv-fcs-np1 (fcsparser 0.2.4 fails on NumPy 2 with `newbyteorder`), flowCore via
Rscript (R_LIBS_USER must point at the flowCore library), flowio in the current environment.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NP1 = ROOT / ".venv-fcs-np1" / "bin" / "python"
GEN = HERE / "files"  # small generated files, committed so the drafts can attach them

FCSPARSER = """
import json, sys, warnings, numpy as np, fcsparser
warnings.simplefilter("ignore")
kw = json.loads(sys.argv[2])
pick = kw.pop("pick", [])
meta, df = fcsparser.parse(sys.argv[1], reformat_meta=True, channel_naming="$PnN", **kw)
a = df.to_numpy()
print(json.dumps({"dtype": str(a.dtype), "shape": list(a.shape), "rows": a[:4].astype(np.float64).tolist(),
                  "picked": [float(a[i, j]) for i, j in pick]}))
"""
FLOWCORE = """
suppressPackageStartupMessages(library(flowCore))
a <- commandArgs(trailingOnly = TRUE)
ff <- suppressWarnings(read.FCS(a[1], transformation = FALSE, truncate_max_range = FALSE))
m <- exprs(ff)
cat(paste(dim(m), collapse = " "), "\\n")
for (i in seq_len(min(4, nrow(m)))) cat(sprintf("%.17g", m[i, ]), "\\n")
"""


def fcsparser_rows(path, **kw):
    p = subprocess.run([str(NP1), "-c", FCSPARSER, str(path), json.dumps(kw)], capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": p.stderr.strip().splitlines()[-1]}
    return json.loads(p.stdout)


def flowcore_rows(path):
    p = subprocess.run(["Rscript", "-e", FLOWCORE, str(path)], capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": p.stderr.strip()[-300:]}
    lines = p.stdout.strip().splitlines()
    return {"shape": [int(x) for x in lines[0].split()], "rows": [[float(x) for x in l.split()] for l in lines[1:]]}


def flowio_rows(path, **kw):
    import flowio
    try:
        f = flowio.FlowData(str(path), **kw)
        a = np.asarray(f.as_array(preprocess=False), dtype=np.float64)
        return {"shape": list(a.shape), "rows": a[:4].tolist()}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def cytopk_rows(path):
    import cytopk
    f = cytopk.read_fcs(path)
    a = np.asarray(f.data, dtype=np.float64)
    return {"shape": list(a.shape), "rows": a[:4].tolist(), "warnings": [w["kind"] for w in f.warnings]}


def raw_fcs(text_pairs, data, version=b"FCS3.0", header_data=None):
    """Minimal FCS writer for cases the cytopk writer refuses to produce (24-bit integers, wrong HEADER)."""
    def build(begin_data):
        kv = dict(text_pairs)
        kv["$BEGINDATA"] = str(begin_data)
        kv["$ENDDATA"] = str(begin_data + len(data) - 1)
        return ("/" + "".join(f"{k}/{v}/" for k, v in kv.items())).encode()
    begin = 256
    text = build(4096)
    begin_data = begin + len(text) + 16
    text = build(begin_data)
    end_text = begin + len(text) - 1
    hd = header_data or (begin_data, begin_data + len(data) - 1)
    head = version + b"    " + b"".join(f"{x:>8d}".encode() for x in (begin, end_text, hd[0], hd[1], 0, 0))
    out = head.ljust(begin, b" ") + text
    return out.ljust(begin_data, b" ") + data


def case_offset():
    """fcsparser uses the HEADER DATA offset even when it points inside TEXT and disagrees with $BEGINDATA."""
    pub = ROOT / "data" / "fcs" / "FlowIO" / "data_start_offset_discrepancy_example.fcs"
    b = pub.read_bytes()
    head = b[:58].decode()
    import cytopk
    t = cytopk.read_fcs(pub).text
    vals = np.array([[1, 2, 3, 4], [1000, 2000, 3000, 65535], [7, 8, 9, 10]], dtype=">u2")
    pairs = [("$BYTEORD", "4,3,2,1"), ("$DATATYPE", "I"), ("$MODE", "L"), ("$NEXTDATA", "0"), ("$PAR", "4"),
             ("$TOT", "3")] + [p for j in range(1, 5) for p in ((f"$P{j}B", "16"), (f"$P{j}E", "0,0"),
                                                                 (f"$P{j}N", f"CH{j}"), (f"$P{j}R", "65536"))]
    good = raw_fcs(pairs, vals.tobytes())
    begin_data = int(good[26:34])
    bad = raw_fcs(pairs, vals.tobytes(), header_data=(begin_data - 40, begin_data - 40 + vals.nbytes - 1))
    GEN.mkdir(parents=True, exist_ok=True)
    gen = GEN / "header_data_offset_inside_text.fcs"
    gen.write_bytes(bad)
    return {
        "public_file": {"path": str(pub.relative_to(ROOT)), "header": head,
                        "header_text": [int(head[10:18]), int(head[18:26])],
                        "header_data": [int(head[26:34]), int(head[34:42])],
                        "BEGINDATA": t["$BEGINDATA"], "ENDDATA": t["$ENDDATA"], "TOT": t["$TOT"],
                        "PAR": t["$PAR"], "bits": [int(t[f"$P{j}B"]) for j in range(1, int(t["$PAR"]) + 1)],
                        "cytopk": cytopk_rows(pub), "flowio_lenient": flowio_rows(pub, ignore_offset_discrepancy=True),
                        "fcsparser": fcsparser_rows(pub)},
        "generated_file": {"path": str(gen.relative_to(ROOT)), "header": bad[:58].decode(),
                           "true_rows": vals.astype(np.float64).tolist(),
                           "true_data_bytes_hex": vals.tobytes().hex(" "),
                           "cytopk": cytopk_rows(gen), "fcsparser": fcsparser_rows(gen),
                           "flowio": flowio_rows(gen), "flowcore": flowcore_rows(gen)},
    }


def case_float32():
    """fcsparser's default dtype float32 rounds 32-bit integer values above 2^24."""
    import cytopk
    pub = ROOT / "data" / "fcs" / "FlowIO" / "M0_WM278_S1.fcs"
    f = cytopk.read_fcs(pub)
    a = np.asarray(f.data, np.float64)
    big = np.argwhere(a > 2 ** 24)
    i, j = (int(x) for x in big[0])
    begin = int(f.text["$BEGINDATA"])
    bpe = 4 * a.shape[1]
    off = begin + i * bpe + 4 * j
    raw = pub.read_bytes()[off:off + 4]
    fp = fcsparser_rows(pub, pick=[[i, j]])
    fp64 = fcsparser_rows(pub, dtype="float64", pick=[[i, j]])
    vals = np.array([[0, 16777216, 16777217, 123456789], [4294967295, 16777219, 33554433, 1]], dtype=np.float64)
    GEN.mkdir(parents=True, exist_ok=True)
    gen = GEN / "int32_above_2pow24.fcs"
    cytopk.write_fcs(gen, vals, ["A", "B", "C", "D"], datatype="I", bits=[32] * 4, ranges=[2 ** 32] * 4)
    return {
        "public_file": {"path": str(pub.relative_to(ROOT)), "datatype": f.datatype, "byteorder": f.byte_order,
                        "bits": sorted({c["bits"] for c in f.channels}), "n_events": f.n_events,
                        "n_values_above_2pow24": int(big.shape[0]),
                        "first_value_above_2pow24": {"event": i, "param": j, "name": f.names[j],
                                                     "byte_offset": off, "bytes_hex": raw.hex(" "),
                                                     "decoded_by_hand": struct.unpack("<I" if f.byte_order == "little"
                                                                                      else ">I", raw)[0],
                                                     "cytopk": a[i, j],
                                                     "fcsparser_default": fp["picked"][0],
                                                     "fcsparser_float64": fp64["picked"][0]},
                        "fcsparser_default_dtype": fp.get("dtype"), "fcsparser_float64_dtype": fp64.get("dtype"),
                        "rows_0_3_param": j,
                        "cytopk_rows_param": a[:4, j].tolist(),
                        "fcsparser_default_rows_param": [r[j] for r in fp["rows"]],
                        "fcsparser_float64_rows_param": [r[j] for r in fp64["rows"]]},
        "generated_file": {"path": str(gen.relative_to(ROOT)), "true_rows": vals.tolist(),
                           "cytopk": cytopk_rows(gen), "fcsparser_default": fcsparser_rows(gen),
                           "fcsparser_float64": fcsparser_rows(gen, dtype="float64"),
                           "flowio": flowio_rows(gen), "flowcore": flowcore_rows(gen)},
    }


def case_int24():
    """flowCore misreads 24-bit integer DATA ($PnB 24)."""
    pub = ROOT / "data" / "fcs" / "fcsparser" / "Cytek_xP5.fcs"
    import cytopk
    f = cytopk.read_fcs(pub)
    begin = int(f.text["$BEGINDATA"])
    raw = pub.read_bytes()[begin:begin + 24]
    by_hand = [int.from_bytes(raw[3 * k:3 * k + 3], "big") for k in range(8)]
    masks = [(1 << (int(c["range"]) - 1).bit_length()) - 1 for c in f.channels]
    vals = [[1, 255, 256, 65535], [65536, 1000000, 8388608, 16777215], [0, 2, 4, 8]]
    data = b"".join(int(v).to_bytes(3, "big") for r in vals for v in r)
    pairs = [("$BYTEORD", "4,3,2,1"), ("$DATATYPE", "I"), ("$MODE", "L"), ("$NEXTDATA", "0"), ("$PAR", "4"),
             ("$TOT", "3")] + [p for j in range(1, 5) for p in ((f"$P{j}B", "24"), (f"$P{j}E", "0,0"),
                                                                 (f"$P{j}N", f"CH{j}"), (f"$P{j}R", "16777216"))]
    GEN.mkdir(parents=True, exist_ok=True)
    gen = GEN / "int24_bigendian.fcs"
    gen.write_bytes(raw_fcs(pairs, data))
    # Control: the same values as little-endian 24-bit words ($BYTEORD 1,2,3,4).
    data_le = b"".join(int(v).to_bytes(3, "little") for r in vals for v in r)
    gen_le = GEN / "int24_littleendian.fcs"
    gen_le.write_bytes(raw_fcs([("$BYTEORD", "1,2,3,4")] + pairs[1:], data_le))
    return {
        "public_file": {"path": str(pub.relative_to(ROOT)), "version": f.version, "datatype": f.datatype,
                        "BYTEORD": f.text["$BYTEORD"], "bits": [c["bits"] for c in f.channels],
                        "PnR": [c["range"] for c in f.channels], "first_event_bytes_hex": raw.hex(" "),
                        "first_event_decoded_by_hand": [v & m for v, m in zip(by_hand, masks)],
                        "cytopk": cytopk_rows(pub), "fcsparser": fcsparser_rows(pub),
                        "flowcore": flowcore_rows(pub), "flowio": flowio_rows(pub)},
        "generated_file": {"path": str(gen.relative_to(ROOT)), "true_rows": vals, "data_bytes_hex": data.hex(" "),
                           "cytopk": cytopk_rows(gen), "fcsparser": fcsparser_rows(gen),
                           "flowcore": flowcore_rows(gen), "flowio": flowio_rows(gen)},
        "generated_file_little_endian": {"path": str(gen_le.relative_to(ROOT)), "true_rows": vals,
                                         "BYTEORD": "1,2,3,4", "data_bytes_hex": data_le.hex(" "),
                                         "cytopk": cytopk_rows(gen_le), "fcsparser": fcsparser_rows(gen_le),
                                         "flowcore": flowcore_rows(gen_le), "flowio": flowio_rows(gen_le)},
    }


def versions():
    import importlib.metadata as md
    v = {p: md.version(p) for p in ["cytopk", "flowio", "numpy"]}
    v["fcsparser_env"] = subprocess.run([str(NP1), "-c", "import sys, numpy, pandas, importlib.metadata as m; "
                                         "print(sys.version.split()[0], numpy.__version__, pandas.__version__, "
                                         "m.version('fcsparser'))"], capture_output=True, text=True).stdout.strip()
    v["flowCore"] = subprocess.run(["Rscript", "-e", "cat(as.character(packageVersion('flowCore')))"],
                                   capture_output=True, text=True).stdout.strip()
    v["R"] = subprocess.run(["Rscript", "-e", "cat(R.version.string)"], capture_output=True, text=True).stdout.strip()
    return v


def main():
    sys.path.insert(0, str(ROOT / "benchmarks" / "sbi_flow"))
    import common
    out = {"meta": common.meta({"experiment": "upstream reproducers", "versions_readers": versions()}),
           "offset": case_offset(), "float32": case_float32(), "int24": case_int24()}
    (HERE / "repro_results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "meta"}, indent=1)[:6000])


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
