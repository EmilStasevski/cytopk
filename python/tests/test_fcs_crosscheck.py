"""Cross-check cytopk's FCS reader against flowio, FlowKit and fcsparser on real public files.

The files are fetched by scripts/fetch_test_data.py into data/fcs/ (git-ignored). The test skips without them,
and skips a reader that is not installed. FlowKit reads files through flowio, so the two are not independent.

Compared per file: $PnN names, event count, and the data matrix, exactly. Reader-specific adjustments:
- flowio: `as_array(preprocess=False)`, i.e. values as stored, like cytopk.
- FlowKit: `get_events(source="raw")` already applies $TIMESTEP, gain and $PnE log scaling, so only linear
  ($PnE 0,0), gain-1, non-time channels are compared.
- fcsparser: channel names from $PnN, and dtype float64 (its default float32 rounds integers above 2^24). A reader that fails on a file is recorded and
skipped: this test checks agreement, not robustness (benchmarks/fcs measures robustness).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import cytopk

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "fcs" / "manifest.json"
FILES = [ROOT / f["path"] for f in json.loads(MANIFEST.read_text())["files"]] if MANIFEST.exists() else []

pytestmark = pytest.mark.skipif(not FILES, reason="run scripts/fetch_test_data.py first")


def ref_flowio(p):
    import flowio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = flowio.FlowData(str(p), ignore_offset_error=True)
    names = list(f.pnn_labels)
    return names, np.asarray(f.as_array(preprocess=False), dtype=np.float64)


def ref_flowkit(p):
    import flowkit as fk
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = fk.Sample(str(p), ignore_offset_error=True)
    return list(s.pnn_labels), np.asarray(s.get_events(source="raw"), dtype=np.float64)


def ref_fcsparser(p):
    import fcsparser
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta, df = fcsparser.parse(str(p), reformat_meta=True, channel_naming="$PnN", dtype="float64")
    return list(meta["_channel_names_"]), df.to_numpy(dtype=np.float64)


READERS = {"flowio": ref_flowio, "flowkit": ref_flowkit, "fcsparser": ref_fcsparser}

# Reference-reader errors, verified by hand. The test asserts that the disagreement is still there (strict xfail),
# so a fixed upstream reader shows up as XPASS.
KNOWN = {
    ("data_start_offset_discrepancy_example.fcs", "fcsparser"):
        "fcsparser uses the HEADER DATA start (5555), which lies inside TEXT (74-6080); the $BEGINDATA keyword "
        "(6081) is right. flowio with ignore_offset_discrepancy agrees with cytopk.",
}


@pytest.mark.parametrize("reader", list(READERS))
@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_agrees_with_reader(path, reader, request):
    pytest.importorskip({"flowkit": "flowkit"}.get(reader, reader))
    if (path.name, reader) in KNOWN:
        request.applymarker(pytest.mark.xfail(reason=KNOWN[(path.name, reader)], strict=True))
    try:
        mine = cytopk.read_fcs(path)
    except cytopk.FcsReadError as e:
        try:
            READERS[reader](path)
        except Exception:
            pytest.skip(f"both fail: cytopk {e}")
        pytest.fail(f"cytopk fails where {reader} succeeds: {e}")
    try:
        names, ref = READERS[reader](path)
    except Exception as e:  # the reference reader cannot read this file
        pytest.skip(f"{reader} fails: {type(e).__name__}: {e}")
    assert mine.names == names
    data = np.asarray(mine.data, dtype=np.float64)
    assert data.shape == ref.shape
    if reader == "flowkit":
        keep = [j for j, c in enumerate(mine.channels)
                if c["log"] == (0.0, 0.0) and c["gain"] in (None, 1.0) and "time" not in c["name"].lower()]
        data, ref = data[:, keep], ref[:, keep]
    np.testing.assert_array_equal(data, ref)


@pytest.mark.parametrize("name", ["data_start_offset_discrepancy_example.fcs",
                                  "data_stop_offset_discrepancy_example.fcs"])
def test_offset_discrepancy_files_match_flowio_lenient(name):
    flowio = pytest.importorskip("flowio")
    p = next((f for f in FILES if f.name == name), None)
    if p is None:
        pytest.skip("file not fetched")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = flowio.FlowData(str(p), ignore_offset_discrepancy=True)
    f = cytopk.read_fcs(p)
    np.testing.assert_array_equal(np.asarray(f.data, np.float64), ref.as_array(preprocess=False))
    assert any(w["kind"] == "DataOffsets" for w in f.warnings)
    with pytest.raises(cytopk.FcsReadError):
        cytopk.read_fcs(p, strict=True)
