"""Markdown tables for benchmarks/fcs/README.md from the committed results JSON (never typed by hand).

    python make_tables.py > /tmp/fcs_tables.md
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
ORDER = ["cytopk", "cytopk_strict", "flowio", "flowio_lenient", "flowkit", "flowkit_lenient", "fcsparser",
         "fcsparser_numpy1", "flowcore"]


def robustness():
    d = json.loads((R / "fcs_robustness.json").read_text())
    readers = [r for r in ORDER if r in d["meta"]["readers"]]
    out = []
    by = defaultdict(dict)
    for x in d["corpus"]:
        by[x["case"]][x["reader"]] = x["outcome"]
    kinds = {x["case"]: x["kind"] for x in d["corpus"]}
    short = {"correct": "✓", "wrong_data": "WRONG", "wrong_names": "names", "error": "err", "timeout": "hang",
             "crash": "CRASH"}
    for kind, title in [("conformant", "Conformant files"), ("deviation", "Non-conformant files")]:
        cases = [c for c in by if kinds[c] == kind]
        out += [f"**{title}** ({len(cases)}): ✓ correct data and names · err = raised an error · WRONG = returned "
                "data that differ from the truth · names = data right, names wrong", "",
                "| case | " + " | ".join(readers) + " |", "|---|" + ":---:|" * len(readers)]
        for c in cases:
            out.append(f"| {c} | " + " | ".join(short.get(by[c].get(r, ""), "—") for r in readers) + " |")
        out.append("| **correct** | " + " | ".join(
            f"**{sum(by[c].get(r) == 'correct' for c in cases)}/{len(cases)}**" for r in readers) + " |")
        out.append("")
    if d["real_files"]:
        rf = defaultdict(dict)
        for x in d["real_files"]:
            rf[x["file"]][x["reader"]] = x
        rr = [r for r in readers if any(r in v for v in rf.values())]
        out += [f"**Real public files** ({len(rf)}): ok = returned data · = = same names and data as cytopk · "
                "≠ = differs from cytopk · err = error", "",
                "| file | source | " + " | ".join(rr) + " |", "|---|---|" + ":---:|" * len(rr)]
        for f, v in rf.items():
            cells = []
            for r in rr:
                x = v.get(r)
                if x is None:
                    cells.append("—")
                elif x["status"] != "ok":
                    cells.append("err")
                elif x["agrees_with_cytopk"] is None:
                    cells.append("ok")
                else:
                    cells.append("=" if x["agrees_with_cytopk"] else "≠")
            src = next(iter(v.values()))["source"]
            out.append(f"| {f[:48]} | {src} | " + " | ".join(cells) + " |")
        cnt = {r: Counter(v[r]["status"] for v in rf.values() if r in v) for r in rr}
        out.append("| **returned data** | | " + " | ".join(f"**{cnt[r]['ok']}/{len(rf)}**" for r in rr) + " |")
    return "\n".join(out)


def speed():
    d = json.loads((R / "fcs_speed.json").read_text())
    rows = d["single"]
    readers = [r for r in ORDER if any(x["reader"] == r for x in rows)]
    files = list(dict.fromkeys(x["file"] for x in rows))
    get = {(x["file"], x["reader"]): x for x in rows}

    def t(x):
        if x is None or "seconds_median" not in x:
            return "err"
        s = x["seconds_median"]
        return f"{s * 1e3:.1f} ms" if s < 1 else f"{s:.2f} s"

    def m(x):
        if x is None or x.get("peak_rss_extra_mb") is None:
            return "—"
        return f"{x['peak_rss_extra_mb']:,.0f}"
    out = ["**Time to a float32 NumPy matrix** (median of 5 warm-cache reads; flowCore: to an R double matrix):",
           "", "| file | MB | " + " | ".join(readers) + " | fastest other / cytopk |",
           "|---|---:|" + "---:|" * len(readers) + "---:|"]
    for f in files:
        x0 = get.get((f, "cytopk"))
        others = [get[(f, r)]["seconds_median"] for r in readers
                  if r != "cytopk" and "seconds_median" in get.get((f, r), {})]
        ratio = (f"{min(others) / x0['seconds_median']:.2f}×"
                 if others and x0 and "seconds_median" in x0 else "—")
        out.append(f"| {f[:44]} | {next(x['bytes'] for x in rows if x['file'] == f) / 1e6:,.1f} | "
                   + " | ".join(t(get.get((f, r))) for r in readers) + f" | {ratio} |")
    out += ["", "**Extra peak RSS of one read** (MB; max RSS of import + read minus import only):", "",
            "| file | " + " | ".join(readers) + " |", "|---|" + "---:|" * len(readers)]
    for f in files:
        if next(x["generated"] for x in rows if x["file"] == f):
            out.append(f"| {f} | " + " | ".join(m(get.get((f, r))) for r in readers) + " |")
    many = d["many"]
    out += ["", f"**{many[0]['files']} small files** (10^4 events × 20 each), seconds (median of 3):", "",
            "| mode | seconds |", "|---|---:|"]
    for x in many:
        out.append(f"| {x['mode']} | {x['seconds_median']:.2f} |" if x["seconds_median"] is not None
                   else f"| {x['mode']} | err |")
    return "\n".join(out)


if __name__ == "__main__":
    print("## Robustness\n" + robustness() + "\n\n## Speed\n" + speed())
