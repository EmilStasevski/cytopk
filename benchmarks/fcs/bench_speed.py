"""FCS read speed and peak memory: cytopk vs flowio, FlowKit, fcsparser (and flowCore when installed).

(a) Single files written by the cytopk writer (float32, little-endian): 10^5, 10^6, 10^7 events x 20 and 40
    parameters, plus the real public files from scripts/fetch_test_data.py.
    Timed: path -> float32 NumPy matrix (events x parameters), including any conversion the reader needs.
    Each (reader, file) runs REPEATS times in one fresh process after one untimed warm-up read, so the page
    cache is warm and the numbers measure parsing and conversion, not the disk. The median is reported.
    Peak memory: maximum RSS of a process that imports the reader and reads the file once, minus that of a
    process that only imports it (GNU time -v). This is the extra memory the read costs.
(b) Many small files: N_SMALL files of 10^4 events x 20 parameters. cytopk.read_fcs_many (Rayon, 16 threads)
    and a plain loop for every reader, plus joblib (16 processes) for the Python readers. That is what users do
    for parallel reads.

(c) --cold: the same timings with the file evicted from the page cache before every timed read
    (posix_fadvise DONTNEED, see pagecache.py; residency is measured with mincore and recorded). No root here,
    so /proc/sys/vm/drop_caches is not used and the drive's own cache cannot be cleared. Peak memory is not
    re-measured (it does not depend on the cache). For many small files, all files are evicted before each run.

    python bench_speed.py          # -> benchmarks/results/fcs_speed.json
    python bench_speed.py --cold   # -> benchmarks/results/fcs_speed_cold.json
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import pagecache  # noqa: E402
import readers  # noqa: E402

GEN = ROOT / "data" / "fcs_bench"
SIZES = [(10**5, 20), (10**5, 40), (10**6, 20), (10**6, 40), (10**7, 20), (10**7, 40)]
REPEATS = 5
N_SMALL = 500
SPEED_READERS = ["cytopk", "cytopk_1thread", "flowio", "flowkit", "fcsparser"]
# cytopk decodes large DATA segments on the Rayon pool; cytopk_1thread (RAYON_NUM_THREADS=1) separates that
# parallelism from per-byte efficiency. The other readers are single-threaded.
ENV = {"cytopk_1thread": {"RAYON_NUM_THREADS": "1"}, "cytopk_many_1thread": {"RAYON_NUM_THREADS": "1"}}
ALIAS = {"cytopk_1thread": "cytopk"}
COLD = os.environ.get("FCS_COLD") == "1"  # set by --cold, inherited by the worker processes


def generate():
    import cytopk
    GEN.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    files = []
    for n, p in SIZES:
        f = GEN / f"gen_{n}x{p}.fcs"
        if not f.exists():
            a = np.empty((n, p), np.float32)
            for i in range(0, n, 10**6):  # in chunks to bound memory of the generator itself
                a[i:i + 10**6] = rng.standard_normal((min(10**6, n - i), p), dtype=np.float32) * 1000
            cytopk.write_fcs(f, a, [f"CH{j}" for j in range(p)])
        files.append(f)
    small = GEN / "small"
    small.mkdir(exist_ok=True)
    smalls = []
    for k in range(N_SMALL):
        f = small / f"s{k:04d}.fcs"
        if not f.exists():
            cytopk.write_fcs(f, rng.standard_normal((10**4, 20), dtype=np.float32), [f"CH{j}" for j in range(20)])
        smalls.append(f)
    return files, smalls


def to_f32(reader, path):
    names, data = readers.READERS[ALIAS.get(reader, reader)](path)
    a = np.asarray(data, dtype=np.float32)
    return a if a.flags.c_contiguous else np.ascontiguousarray(a)


def worker_time(reader, path):
    to_f32(reader, path)  # warm-up (page cache, lazy imports)
    ts, res = [], []
    for _ in range(REPEATS):
        if COLD:
            for _ in range(3):  # a read-ahead still in flight can repopulate a few pages; retry the eviction
                pagecache.drop(path)
                r = pagecache.resident(path)
                if r == 0.0:
                    break
            res.append(r)
        t0 = time.perf_counter()
        a = to_f32(reader, path)
        ts.append(time.perf_counter() - t0)
        del a
    print(json.dumps({"times": ts, "resident_before": res}))


def worker_mem(reader, path):
    import importlib
    importlib.import_module(readers.IMPORT[ALIAS.get(reader, reader)])
    if path != "-":
        to_f32(reader, path)


def worker_many(mode, listfile):
    paths = Path(listfile).read_text().splitlines()
    if COLD:
        for p in paths:
            pagecache.drop(p)
    t0 = time.perf_counter()
    if mode in ("cytopk_many", "cytopk_many_1thread"):
        import cytopk
        out = [np.asarray(f.data, np.float32) for f in cytopk.read_fcs_many(paths)]
    elif mode.endswith("_joblib"):
        from joblib import Parallel, delayed
        r = mode[: -len("_joblib")]
        out = Parallel(n_jobs=16)(delayed(to_f32)(r, p) for p in paths)
    else:
        out = [to_f32(mode, p) for p in paths]
    dt = time.perf_counter() - t0
    assert len(out) == len(paths)
    print(json.dumps({"seconds": dt}))


def peak_rss_kb(argv, env=None):
    p = subprocess.run(["/usr/bin/time", "-v", *argv], capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    if p.returncode != 0:
        return None, p.stderr[-300:]
    for line in p.stderr.splitlines():
        if "Maximum resident set size" in line:
            return int(line.split(":")[1]), None
    return None, "no RSS line"


def py(*a):
    return [sys.executable, __file__, *a]


def flowcore_cmds(path):
    t = ["Rscript", str(HERE / "flowcore_reader.R")]
    timing = t + ["time", str(REPEATS + 1), str(path)]
    if COLD:
        timing.append(shlex.join([sys.executable, str(HERE / "pagecache.py"), "drop", str(path)]))
    return timing, t + ["time", "1", str(path)], t + ["import"]


def storage():
    """Block device that holds the benchmark files (model and rotational flag from sysfs)."""
    try:
        src = subprocess.run(["findmnt", "-n", "-o", "SOURCE", "--target", str(GEN)], capture_output=True,
                             text=True).stdout.strip()
        dev = Path(src).name
        blk = next((d for d in Path("/sys/block").iterdir() if dev.startswith(d.name)), None)
        if blk is None:
            return {"device": dev}
        model = (blk / "device" / "model").read_text().strip() if (blk / "device" / "model").exists() else None
        return {"device": blk.name, "model": model, "rotational": (blk / "queue" / "rotational").read_text().strip()}
    except (OSError, StopIteration):
        return None


def main():
    global COLD
    if sys.argv[1:] == ["--cold"]:
        COLD = True
        os.environ["FCS_COLD"] = "1"
    elif len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        mode = sys.argv[1]
        if mode == "--time":
            return worker_time(sys.argv[2], sys.argv[3])
        if mode == "--mem":
            return worker_mem(sys.argv[2], sys.argv[3])
        if mode == "--many":
            return worker_many(sys.argv[2], sys.argv[3])
    sys.path.insert(0, str(ROOT / "benchmarks" / "sbi_flow"))
    import common
    from bench_robustness import available_readers, flowcore_available, versions
    files, smalls = generate()
    manifest = ROOT / "data" / "fcs" / "manifest.json"
    real = [ROOT / f["path"] for f in json.loads(manifest.read_text())["files"]] if manifest.exists() else []
    avail = available_readers()
    have = [r for r in SPEED_READERS if ALIAS.get(r, r) in avail]
    fc = flowcore_available()
    rows = []
    for path in files + real:
        size = path.stat().st_size
        for r in have + (["flowcore"] if fc else []):
            row = {"file": path.name, "bytes": size, "reader": r, "generated": path in files}
            if r == "flowcore":
                tcmd, mcmd, base = flowcore_cmds(path)
                p = subprocess.run(tcmd, capture_output=True, text=True)
                if p.returncode == 0:
                    ts = [float(x) for x in p.stdout.split()][1:]  # first call = warm-up
                    row["seconds_median"] = float(np.median(ts))
                    row["note"] = "flowCore: read.FCS + exprs() as double; R matrix, not NumPy float32"
                else:
                    row["error"] = p.stderr[-300:]
            else:
                p = subprocess.run(py("--time", r, str(path)), capture_output=True, text=True,
                                   env={**os.environ, **ENV.get(r, {})})
                if p.returncode == 0:
                    w = json.loads(p.stdout.strip().splitlines()[-1])
                    row["seconds_median"] = float(np.median(w["times"]))
                    if COLD:
                        row["seconds_all"] = w["times"]
                        row["resident_before_max"] = max(w["resident_before"])
                else:
                    row["error"] = p.stderr.strip().splitlines()[-1][:300] if p.stderr.strip() else "failed"
                mcmd, base = py("--mem", r, str(path)), py("--mem", r, "-")
            if "seconds_median" in row and not COLD:
                peak, err = peak_rss_kb(mcmd, ENV.get(r))
                b, _ = peak_rss_kb(base, ENV.get(r))
                if peak is not None and b is not None:
                    row["peak_rss_extra_mb"] = (peak - b) / 1024
                    row["peak_rss_mb"] = peak / 1024
            rows.append(row)
            print(f"{path.name[:42]:42s} {r:10s} {row.get('seconds_median', float('nan')):9.4f} s  "
                  f"+{row.get('peak_rss_extra_mb', float('nan')):8.1f} MB  {row.get('error', '')[:60]}", flush=True)
    listfile = GEN / "small_list.txt"
    listfile.write_text("\n".join(str(p) for p in smalls))
    many = []
    modes = ["cytopk_many", "cytopk_many_1thread"] + have + [f"{r}_joblib" for r in have if r not in ALIAS]
    for m in modes:
        ts = []
        for _ in range(3):
            p = subprocess.run(py("--many", m, str(listfile)), capture_output=True, text=True,
                               env={**os.environ, **ENV.get(m, {})})
            if p.returncode != 0:
                ts = None
                err = p.stderr.strip().splitlines()[-1][:300] if p.stderr.strip() else "failed"
                break
            ts.append(json.loads(p.stdout.strip().splitlines()[-1])["seconds"])
        many.append({"mode": m, "files": len(smalls), "seconds_median": float(np.median(ts)) if ts else None,
                     "seconds_all": ts, **({} if ts else {"error": err})})
        print(f"many {m:22s} {many[-1]['seconds_median']}", flush=True)
    out = ROOT / "benchmarks" / "results" / ("fcs_speed_cold.json" if COLD else "fcs_speed.json")
    cache = ({"page_cache": "evicted before every timed read (posix_fadvise DONTNEED, no root); drive cache "
                            "not cleared", "storage": storage()} if COLD else {"page_cache": "warm"})
    out.write_text(json.dumps({"meta": common.meta({"experiment": "FCS read speed and memory", **cache,
                                                    "versions_fcs": versions(), "repeats": REPEATS,
                                                    "n_small": N_SMALL, "cpu_threads": os.cpu_count()}),
                               "single": rows, "many": many}, indent=2) + "\n")


if __name__ == "__main__":
    main()
