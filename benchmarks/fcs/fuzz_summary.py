"""Record a cargo-fuzz run of the FCS reader as results JSON.

    cd fuzz && cargo +nightly fuzz run read_fcs -- -max_total_time=600 -max_len=400000 -rss_limit_mb=4096 \
        > ../benchmarks/scratch/fuzz.log 2>&1
    python benchmarks/fcs/fuzz_summary.py benchmarks/scratch/fuzz.log     # -> benchmarks/results/fcs_fuzz.json

Seed corpus: every file of data/fcs_corpus/ plus the real files from scripts/fetch_test_data.py under 300 kB.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "sbi_flow"))
import common  # noqa: E402


def main():
    log = Path(sys.argv[1]).read_text(errors="replace")
    done = re.findall(r"Done (\d+) runs in (\d+) second", log)
    final = re.findall(r"#(\d+)\s+DONE\s+cov: (\d+) ft: (\d+) corp: (\d+)/", log)
    crashes = len(re.findall(r"(panicked at|deadly signal|ERROR: libFuzzer|SUMMARY: )", log))
    artifacts = sorted(p.name for p in (ROOT / "fuzz" / "artifacts" / "read_fcs").glob("*")) \
        if (ROOT / "fuzz" / "artifacts" / "read_fcs").exists() else []
    nightly = subprocess.run(["rustc", "+nightly", "--version"], capture_output=True, text=True).stdout.strip()
    out = {
        "meta": common.meta({"experiment": "cargo-fuzz read_fcs", "rustc_nightly": nightly,
                             "command": "cargo +nightly fuzz run read_fcs -- -max_total_time=600 "
                                        "-max_len=400000 -rss_limit_mb=4096"}),
        "runs": int(done[-1][0]) if done else None,
        "seconds": int(done[-1][1]) if done else None,
        "coverage_edges": int(final[-1][1]) if final else None,
        "features": int(final[-1][2]) if final else None,
        "corpus_entries": int(final[-1][3]) if final else None,
        "crash_lines_in_log": crashes,
        "artifacts": artifacts,
    }
    (ROOT / "benchmarks" / "results" / "fcs_fuzz.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "meta"}))


if __name__ == "__main__":
    main()
