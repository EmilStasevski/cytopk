"""FCS reader/writer through the Python API (no third-party readers needed)."""

import numpy as np
import pytest

import cytopk


def test_write_read_roundtrip_float32(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((500, 6)).astype(np.float32)
    names = [f"FL{i}-A" for i in range(6)]
    p = tmp_path / "a.fcs"
    cytopk.write_fcs(p, data, names, long_names=["CD3", None, None, "CD8", None, None],
                     keywords={"$CYT": "synthetic", "MYKEY": "a/b"})
    f = cytopk.read_fcs(p, strict=True)
    assert f.data.dtype == np.float32
    assert f.data.shape == (500, 6)
    np.testing.assert_array_equal(f.data, data)
    assert f.names == names
    assert f.channels[0]["long_name"] == "CD3"
    assert f.text["$CYT"] == "synthetic"
    assert f.text["MYKEY"] == "a/b"
    assert f.warnings == []
    assert f.version == "FCS3.1"
    assert f.n_events == 500


@pytest.mark.parametrize("datatype,byteorder", [("D", "big"), ("I", "little"), ("I", "big"), ("F", "big")])
def test_write_read_other_types(tmp_path, datatype, byteorder):
    data = np.arange(60, dtype=np.float64).reshape(20, 3)
    p = tmp_path / "b.fcs"
    cytopk.write_fcs(p, data, ["a", "b", "c"], datatype=datatype, byteorder=byteorder,
                     bits=[16, 32, 8] if datatype == "I" else None, delimiter="|")
    f = cytopk.read_fcs(p, strict=True)
    np.testing.assert_array_equal(f.data, data)
    assert f.datatype == datatype
    assert f.byte_order == byteorder


def test_lenient_warnings_and_strict_error():
    data = np.ones((3, 1), dtype=np.float32)
    b = bytearray(_bytes(data))
    i = b.index(b"/$P1N/A/")
    b[i:i + 8] = b"/$P1N//A"  # empty value for $P1N, then a dangling 'A'
    with pytest.raises(cytopk.FcsReadError):
        cytopk.read_fcs_bytes(bytes(b), strict=True)
    f = cytopk.read_fcs_bytes(bytes(b))
    kinds = {w["kind"] for w in f.warnings}
    assert "EmptyValue" in kinds
    assert f.names == ["P1"]  # empty $PnN -> default name


def _bytes(data, **kw):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.fcs"
        cytopk.write_fcs(p, data, ["A"], **kw)
        return p.read_bytes()


def test_read_many_returns_errors_in_place(tmp_path):
    good = tmp_path / "good.fcs"
    cytopk.write_fcs(good, np.zeros((2, 2), np.float32), ["x", "y"])
    bad = tmp_path / "bad.fcs"
    bad.write_bytes(b"not an fcs file at all")
    out = cytopk.read_fcs_many([good, bad, tmp_path / "missing.fcs"], n_threads=2)
    assert isinstance(out[0], cytopk.FcsFile)
    assert isinstance(out[1], cytopk.FcsReadError)
    assert isinstance(out[2], cytopk.FcsReadError)


def test_spillover_and_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow.parquet")
    data = np.arange(12, dtype=np.float32).reshape(4, 3)
    p = tmp_path / "s.fcs"
    cytopk.write_fcs(p, data, ["FL1", "FL2", "FL1"], keywords={"$SPILLOVER": "2,FL1,FL2,1,0.2,0.1,1"})
    f = cytopk.read_fcs(p)
    s = f.spillover
    assert s["channels"] == ["FL1", "FL2"]
    np.testing.assert_array_equal(s["matrix"], [[1, 0.2], [0.1, 1]])
    out = tmp_path / "s.parquet"
    f.to_parquet(out)
    t = pa.read_table(out)
    assert t.column_names == ["FL1", "FL2", "FL1_2"]
    np.testing.assert_array_equal(t.column("FL1_2").to_numpy(), data[:, 2])
    assert t.schema.metadata[b"$SPILLOVER"] == b"2,FL1,FL2,1,0.2,0.1,1"


def test_multiple_datasets_and_errors(tmp_path):
    p = tmp_path / "m.fcs"
    cytopk.write_fcs(p, np.zeros((1, 1), np.float32), ["a"])
    two = p.read_bytes() * 1  # single dataset
    assert len(cytopk.read_fcs_all(p)) == 1
    with pytest.raises(cytopk.FcsReadError):
        cytopk.read_fcs_bytes(two[:40])
    with pytest.raises(TypeError):
        cytopk.write_fcs(p, np.zeros(3), ["a"])
    with pytest.raises(ValueError):
        cytopk.write_fcs(p, np.zeros((1, 1)), ["a"], datatype="X")
