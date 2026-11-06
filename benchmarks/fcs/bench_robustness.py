"""FCS robustness: how often does each reader return the right data on conformant and non-conformant files?

Inputs: the artificial corpus (corpus.py, with ground truth) and the real public files from
scripts/fetch_test_data.py. No ground truth exists for the real files, so for them we record whether each
reader returns data and whether it agrees with the others.

Outcome per (reader, file):
  correct      names and data equal the ground truth (data exactly; integers after the spec's $PnR mask)
  wrong_data   the reader returned, but data (shape or values) differ from the truth (silent corruption)
  wrong_names  data right, names differ
  error        the reader raised an exception
  timeout      over TIMEOUT seconds
  crash        the process died (segfault, abort)
Each (reader, file) runs in its own subprocess, so a crash or hang cannot affect the others.

    python bench_robustness.py      # -> benchmarks/results/fcs_robustness.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import readers  # noqa: E402

CORPUS = ROOT / "data" / "fcs_corpus"
MANIFEST = ROOT / "data" / "fcs" / "manifest.json"
TIMEOUT = 60
# fcsparser 0.2.4 predates NumPy 2 (it calls ndarray.newbyteorder). It is also run in a NumPy 1.26 environment
# so that its result reflects the reader, not the NumPy version: .venv-fcs-np1 (Python 3.12, numpy<2, pandas<2.3).
PYTHON = {"fcsparser_numpy1": ROOT / ".venv-fcs-np1" / "bin" / "python"}


def python_for(reader):
    return str(PYTHON.get(reader, sys.executable))


def worker(reader, path, out):
    try:
        names, data = readers.READERS[reader](path)
        a = np.asarray(data, dtype=np.float64)
        np.save(out, a)
        print(json.dumps({"ok": True, "names": [str(n) for n in names]}))
    except Exception as e:  # noqa: BLE001 - any failure is an outcome
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}))


def run_python(reader, path, tmp):
    out = Path(tmp) / "a.npy"
    out.unlink(missing_ok=True)
    try:
        p = subprocess.run([python_for(reader), __file__, "--worker", reader, str(path), str(out)],
                           capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}, None
    if p.returncode != 0:
        return {"status": "crash", "error": p.stderr[-300:]}, None
    r = json.loads(p.stdout.strip().splitlines()[-1])
    if not r["ok"]:
        return {"status": "error", "error": r["error"]}, None
    return {"status": "ok", "names": r["names"]}, np.load(out)


def run_flowcore(paths, tmp):
    out = Path(tmp) / "fc"
    out.mkdir(exist_ok=True)
    res = []
    for i, path in enumerate(paths):  # one Rscript per file so crashes and hangs stay isolated
        d = out / str(i)
        d.mkdir(exist_ok=True)
        try:
            p = subprocess.run(["Rscript", str(HERE / "flowcore_reader.R"), "read", str(d), str(path)],
                               capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            res.append(({"status": "timeout"}, None))
            continue
        js = d / "0.json"
        if p.returncode != 0 or not js.exists():
            res.append(({"status": "crash", "error": p.stderr[-300:]}, None))
            continue
        r = json.loads(js.read_text())
        if not r["ok"]:
            res.append(({"status": "error", "error": r["error"][:300]}, None))
            continue
        a = np.fromfile(d / "0.bin", dtype="<f8").reshape(r["n"], r["p"]) if r["n"] * r["p"] else \
            np.zeros((r["n"], r["p"]))
        res.append(({"status": "ok", "names": r["names"]}, a))
    return res


def flowcore_available():
    try:
        return subprocess.run(["Rscript", "-e", "quit(status = !requireNamespace('flowCore', quietly = TRUE))"],
                              capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def classify(res, data, truth, names):
    if res["status"] != "ok":
        return res["status"]
    if data.shape != truth.shape or not np.array_equal(data, truth, equal_nan=True):
        return "wrong_data"
    if res["names"] != names:
        return "wrong_names"
    return "correct"


def available_readers():
    out = []
    for r, mod in readers.IMPORT.items():
        py = python_for(r)
        if Path(py).exists() and subprocess.run([py, "-c", f"import {mod}"], capture_output=True).returncode == 0:
            out.append(r)
    return out


def versions():
    import importlib.metadata as md
    v = {}
    for p in ["cytopk", "flowio", "flowkit", "fcsparser", "numpy", "pandas"]:
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    py1 = PYTHON["fcsparser_numpy1"]
    if py1.exists():
        v["numpy1_env"] = subprocess.run(
            [str(py1), "-c", "import sys, numpy, pandas, importlib.metadata as m; print(sys.version.split()[0], "
             "numpy.__version__, pandas.__version__, m.version('fcsparser'))"],
            capture_output=True, text=True).stdout.strip()
    try:
        v["flowCore"] = subprocess.run(["Rscript", "-e", "cat(as.character(packageVersion('flowCore')))"],
                                       capture_output=True, text=True, timeout=120).stdout.strip() or None
        v["R"] = subprocess.run(["Rscript", "-e", "cat(R.version.string)"], capture_output=True, text=True,
                                timeout=60).stdout.strip()
    except OSError:
        pass
    return v


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return worker(*sys.argv[2:5])
    sys.path.insert(0, str(ROOT / "benchmarks" / "sbi_flow"))
    import common  # metadata helper shared with the SBI-flow benchmarks
    rlist = available_readers()
    fc = flowcore_available()
    cases = json.loads((CORPUS / "cases.json").read_text())
    truth = np.load(CORPUS / "truth.npz")
    real = [ROOT / f["path"] for f in json.loads(MANIFEST.read_text())["files"]] if MANIFEST.exists() else []
    rows, real_rows = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for c in cases:
            path = CORPUS / f"{c['name']}.fcs"
            for r in rlist:
                res, data = run_python(r, path, tmp)
                rows.append({"case": c["name"], "kind": c["kind"], "reader": r,
                             "outcome": classify(res, data, truth[c["name"]], c["names"]),
                             "error": res.get("error")})
                print(f"{c['name']:32s} {r:16s} {rows[-1]['outcome']}", flush=True)
        if fc:
            for c, (res, data) in zip(cases, run_flowcore([CORPUS / f"{c['name']}.fcs" for c in cases], tmp)):
                rows.append({"case": c["name"], "kind": c["kind"], "reader": "flowcore",
                             "outcome": classify(res, data, truth[c["name"]], c["names"]),
                             "error": res.get("error")})
                print(f"{c['name']:32s} {'flowcore':16s} {rows[-1]['outcome']}", flush=True)
        for path in real:
            got = {}
            for r in rlist:
                got[r] = run_python(r, path, tmp)
            if fc:
                got["flowcore"] = run_flowcore([path], tmp)[0]
            ref_res, ref = got.get("cytopk", ({"status": "error"}, None))
            for r, (res, data) in got.items():
                agree = None
                if res["status"] == "ok" and ref_res["status"] == "ok" and r != "cytopk":
                    agree = bool(data.shape == ref.shape and np.array_equal(data, ref, equal_nan=True)
                                 and res["names"] == ref_res["names"])
                real_rows.append({"file": path.name, "source": path.parent.name, "reader": r,
                                  "status": res["status"], "agrees_with_cytopk": agree,
                                  "shape": list(data.shape) if data is not None else None,
                                  "error": res.get("error")})
                print(f"{path.name[:40]:40s} {r:16s} {res['status']:8s} agree={agree}", flush=True)
    out = ROOT / "benchmarks" / "results" / "fcs_robustness.json"
    out.write_text(json.dumps({"meta": common.meta({"experiment": "FCS robustness", "versions_fcs": versions(),
                                                    "readers": rlist + (["flowcore"] if fc else []),
                                                    "timeout_s": TIMEOUT}),
                               "corpus": rows, "real_files": real_rows}, indent=2) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
