"""Fetch real public FCS files for the cross-reader tests and benchmarks into the git-ignored data/ folder.

Nothing fetched here is committed. Every file is pinned to a commit and checked against its SHA-256.
data/fcs/manifest.json records source URL, licence and SHA-256 for each file.

Sources (the repositories ship these files as test data under the repository licence):
  whitews/FlowIO     BSD-3-Clause   https://github.com/whitews/FlowIO
  whitews/FlowKit    BSD-3-Clause   https://github.com/whitews/FlowKit
  eyurtsev/fcsparser MIT            https://github.com/eyurtsev/fcsparser

Optional: the Bras & van der Velden (2020) test set, "Robust FCS Parsing: Exploring 211,359 Public Files",
Cytometry A 97:1180, doi:10.1002/cyto.a.24187. It is public on FlowRepository (FR-FCM-Z2LG,
http://flowrepository.org/id/FR-FCM-Z2LG). FlowRepository's terms place no restrictions on use or
redistribution but note possible third-party rights. The article is CC BY-NC-ND 4.0. A download needs a CAPTCHA
(or a FlowRepository API client ID, which this project does not have), so it cannot be scripted. Download
`artificial_fcs.7z` (218 MB) by hand from the experiment page, then:

    python scripts/fetch_test_data.py --bras-archive ~/Downloads/artificial_fcs.7z

That checks the MD5 published in the article's Supporting Information (Sup. 25) and extracts the archive to
data/bras2020/ with 7z.

    python scripts/fetch_test_data.py            # GitHub files only (about 60 MB)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

REPOS = {
    "whitews/FlowIO": ("83d28a22d42235c10d17afb017250ee208afed95", "BSD-3-Clause"),
    "whitews/FlowKit": ("f2159043b6a56e527d4baacf97490caf8354618e", "BSD-3-Clause"),
    "eyurtsev/fcsparser": ("da70aaa7ec92ff3bd9ce00aec4eea7c77ee8c096", "MIT"),
}

# (repo, path, sha256). A None sha256 means not pinned yet: it is printed on first fetch and then written here.
FILES = [
    ("whitews/FlowIO", "data/fcs_files/100715.fcs", "733b0f6d959527116860f4c78f0908db33fc71388c7717737b340c100b9f46e7"),
    ("whitews/FlowIO", "data/fcs_files/3FITC_4PE_004.fcs", "240e76eba7e04c7e9116c8c3ad8388ccb2faf58105d21b831d122764a3621e53"),
    ("whitews/FlowIO", "data/fcs_files/B01 KC-A-W---91-US.fcs", "33736fb7a40a4f571837c231d342df5fd4b20b5dd70439b74b3610f94b6c2a36"),
    ("whitews/FlowIO", "data/fcs_files/G11.fcs", "40ed9f6d4293a87077cab07a34c68b2497d4d69988adcb64a5f718098f48b56f"),
    ("whitews/FlowIO", "data/fcs_files/coulter.lmd", "3182e8522907b09e3d2ef4e50983502b86db52ea55aaf8528a10d6b408e3e46a"),
    ("whitews/FlowIO", "data/fcs_files/data1.fcs", "cba7812f5bbce700786d22d97cf9858c1301878aa840c64eadcd263a3a0a5508"),
    ("whitews/FlowIO", "data/fcs_files/data_start_offset_discrepancy_example.fcs", "414fdccae26a65996324261ab028fb1c762e448313ebcf6207894b4dc43eef47"),
    ("whitews/FlowIO", "data/fcs_files/data_stop_offset_discrepancy_example.fcs", "49956b68a5131af1bba423a26fdaabd9fdf109b51d5f421a42971c09c920cc9d"),
    ("whitews/FlowIO", "data/fcs_files/variable_int_example.fcs", "88d2933d52686b061f36d8596e0826c9d2e75eb3d654d570f4ad3f1153f9beb2"),
    ("whitews/FlowIO", "data/fcs_files/zero_gain/M0_WM278_S1.fcs", "f9e528334b63fe2f879f06164159a0e1ef5daa0c6e4e253f03c92ad030d6822f"),
    ("whitews/FlowKit", "data/109567.fcs", "6011d295c4718f79ae9f768790cdea9fa85445452b963d2cf7715b9f2d52a353"),
    ("whitews/FlowKit", "data/8_color_data_set/fcs_files/101_DEN084Y5_15_E01_008_clean.fcs", "55e2e1231e6a2b310856ef0722b59bb938277c446fe392f53dc007983ecead6d"),
    ("whitews/FlowKit", "data/index_sorted/index_sorted_example.fcs", "0e562897a8b995861463fe388af88d57c85276725d3f1d67f5bbb9c0e7016cf4"),
    ("whitews/FlowKit", "data/simple_line_example/data_set_simple_line_100.fcs", "b620d2e81afafe87bca9c4c43141eda3840220614581b735cd7acd8591143595"),
    ("whitews/FlowKit", "data/test_comp_example.fcs", "fb33bf60546d40fa7e892c67f51681f8af59d3be818a212035c649437f00a105"),
    ("whitews/FlowKit", "data/test_data_2d_01.fcs", "9dbc58f110b6928df55b5720d5e176975918f587213960fef62b1930f3a52113"),
    ("eyurtsev/fcsparser", "fcsparser/tests/data/FlowCytometers/Cytek_xP5/Cytek_xP5.fcs", "dd6aaa9a034c40ab5f3a2ce1ea4c8eab7d00a053426917e07a09f8d8d76ca51a"),
    ("eyurtsev/fcsparser", "fcsparser/tests/data/FlowCytometers/FACSCaliburHTS/Sample_Well_A02.fcs", "bf3fb0140c63ca583a01ec5d6af46df1d00f2e2cd6151f5533ab13a7d7f34e27"),
    ("eyurtsev/fcsparser", "fcsparser/tests/data/FlowCytometers/FACS_Diva/facs_diva_test.fcs", "8d0a72d1d219c880d9120bac0b7501afa23a08b4ecd2b074803e73eaa411802a"),
    ("eyurtsev/fcsparser",
     "fcsparser/tests/data/FlowCytometers/Fortessa/FCS_3.0_Fortessa_PBS_Specimen_001_A1_A01.fcs", "fa9011c86e8ad043ab623656646f329aea907e9655e20f94ade97eea4b9dc177"),
    ("eyurtsev/fcsparser", "fcsparser/tests/data/FlowCytometers/GuavaMuse/Guava Muse.fcs", "f9b0ca844ecee33c935a6c5e6a7d81c5afe155ce40f0ed663bc51f0d9c98f0e9"),
    ("eyurtsev/fcsparser",
     "fcsparser/tests/data/FlowCytometers/HTS_BD_LSR-II/HTS_BD_LSR_II_Mixed_Specimen_001_D6_D06.fcs", "47ecbe42cc442449aa2739731c2d32d8dbcca58fbaa135cf30583cca234f9277"),
    ("eyurtsev/fcsparser", "fcsparser/tests/data/FlowCytometers/MiltenyiBiotec/FCS2.0/"
     "EY_2013-07-19_PBS_FCS_2.0_Custom_Without_Add_Well_A1.001.fcs", "28c442794bf920c72cb7c84cdca0e972d5cbace49727ddedde9d717c8b049128"),
    ("eyurtsev/fcsparser",
     "fcsparser/tests/data/FlowCytometers/MiltenyiBiotec/FCS3.0/FCS3.0_Custom_Compatible.fcs", "9a56d8825ea533356443c3b849bef82c6027ca3b4eea9e52a4b74e2be25219ff"),
    ("eyurtsev/fcsparser", "fcsparser/tests/data/FlowCytometers/MiltenyiBiotec/FCS3.1/"
     "EY_2013-07-19_PBS_FCS_3.1_Well_A1.001.fcs", "8cfedc08f387a53cd492c98b5150c3a85ad704499aec07190de5bb249897598e"),
    ("eyurtsev/fcsparser",
     "fcsparser/tests/data/FlowCytometers/MiltenyiBiotec/FCS3.1/SG_2014-09-26_Duplicate_Names.fcs", "7ee7e214e54161ebea46aea860ef6c050dfdb1e093cd1d97684fbfd4bcb87cc4"),
    ("eyurtsev/fcsparser", "fcsparser/tests/data/FlowCytometers/corrupted/corrupted.fcs", "20f17232f1485cb999fb594671ab502d37ae44f88fc7eaeb5b3ffa6288c6c96d"),
    ("eyurtsev/fcsparser", "fcsparser/tests/data/FlowCytometers/cyflow_cube_8/cyflow_cube_8.fcs", "08b8ef7fc49d34c56543551cc4c16bbf50c85af49d89b3abbe40d6f6a5d6a8d9"),
    ("eyurtsev/fcsparser", "fcsparser/tests/data/FlowCytometers/cytek-nl-2000/sample_header.fcs", "1961e20bab436832ab1fad6f3563993d27181b263d8cc5d54274173b628e1fc3"),
    ("eyurtsev/fcsparser", "fcsparser/tests/data/FlowCytometers/fake_bitmask_error/fcs1_cleaned.lmd", "6250ffa9c5aca9da3c362297ed481a403594d109ce04f4e5e6504b9a70ed4493"),
]

BRAS_MD5 = "7208BE57B6E08FE13210BC7D0C69D209"  # artificial_fcs.7z, Supporting Information Sup. 25
BRAS_SOURCE = "http://flowrepository.org/id/FR-FCM-Z2LG (attachment artificial_fcs.7z)"
BRAS_LICENCE = ("FlowRepository terms of use: no restrictions on use or redistribution by ISAC; third-party rights "
                "may apply. Article: CC BY-NC-ND 4.0. Not redistributed by cytopk.")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_name(repo: str, path: str) -> Path:
    return DATA / "fcs" / repo.split("/")[1] / Path(path).name


def fetch_github(manifest: dict) -> int:
    bad = 0
    for repo, path, want in FILES:
        commit, licence = REPOS[repo]
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{urllib.request.quote(path)}"
        dest = local_name(repo, path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            tmp = dest.with_suffix(dest.suffix + ".part")
            req = urllib.request.Request(url, headers={"User-Agent": "cytopk-fetch-test-data"})
            with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f)
            tmp.rename(dest)
        got = sha256(dest)
        ok = want is None or got == want
        if not ok:
            bad += 1
            print(f"SHA-256 MISMATCH {dest.relative_to(ROOT)}: got {got}, want {want}", file=sys.stderr)
        elif want is None:
            print(f"unpinned {dest.relative_to(ROOT)}: {got}")
        manifest["files"].append({"path": str(dest.relative_to(ROOT)), "source": url, "repository": repo,
                                  "commit": commit, "licence": licence, "sha256": got,
                                  "bytes": dest.stat().st_size})
    return bad


def fetch_bras(archive: Path, manifest: dict) -> int:
    md5 = hashlib.md5()
    with archive.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    if md5.hexdigest().upper() != BRAS_MD5:
        print(f"MD5 mismatch for {archive}: {md5.hexdigest().upper()} != {BRAS_MD5}", file=sys.stderr)
        return 1
    out = DATA / "bras2020"
    out.mkdir(parents=True, exist_ok=True)
    seven = shutil.which("7z") or shutil.which("7za")
    if seven is None:
        print("7z not found; install p7zip to extract", file=sys.stderr)
        return 1
    subprocess.run([seven, "x", "-y", f"-o{out}", str(archive)], check=True, stdout=subprocess.DEVNULL)
    manifest["bras2020"] = {"archive_sha256": sha256(archive), "archive_md5": BRAS_MD5, "source": BRAS_SOURCE,
                            "licence": BRAS_LICENCE, "extracted_to": str(out.relative_to(ROOT)),
                            "reference": "Bras AE, van der Velden VHJ. Cytometry A 2020;97:1180-1186. "
                                         "doi:10.1002/cyto.a.24187"}
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bras-archive", type=Path, help="path to a hand-downloaded artificial_fcs.7z")
    a = ap.parse_args()
    (DATA / "fcs").mkdir(parents=True, exist_ok=True)
    mpath = DATA / "fcs" / "manifest.json"
    manifest = {"fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": []}
    if mpath.exists():
        manifest["bras2020"] = json.loads(mpath.read_text()).get("bras2020")
    bad = fetch_github(manifest)
    if a.bras_archive:
        bad += fetch_bras(a.bras_archive.expanduser(), manifest)
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(manifest['files'])} files, manifest at {mpath.relative_to(ROOT)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
