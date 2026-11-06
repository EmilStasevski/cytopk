"""Python API of the gating layer: arcsinh, GatingStrategy (TOML), masks, gate_fcs_many."""

import numpy as np
import pytest

import cytopk

STRATEGY = """
[transform]
cofactor = 5.0
arcsinh = ["CD3"]
[transform.cofactors]
CD69 = 150.0
[stats]
medians = ["CD3", "CD69"]
[[gate]]
name = "cd3"
range = { CD3 = [1.0, inf] }
[[gate]]
name = "activated"
parent = "cd3"
range = { CD69 = [1.0, inf] }
"""


def data():
    rng = np.random.default_rng(0)
    return np.column_stack([rng.normal(10, 8, 5000), rng.normal(150, 200, 5000), rng.normal(0, 1, 5000)])


def naive(d):
    cd3 = np.arcsinh(d[:, 0] / 5.0)
    cd69 = np.arcsinh(d[:, 1] / 150.0)
    g1 = cd3 >= 1.0
    g2 = g1 & (cd69 >= 1.0)
    return cd3, cd69, g1, g2


def test_arcsinh_matches_numpy():
    x = np.linspace(-1e5, 1e5, 1001)
    np.testing.assert_allclose(cytopk.arcsinh(x, 150.0), np.arcsinh(x / 150.0), rtol=1e-15, atol=0)
    np.testing.assert_array_equal(cytopk.arcsinh(np.array([[0.0, 5.0]]), 5.0), np.arcsinh([[0.0, 1.0]]))
    with pytest.raises(ValueError):
        cytopk.arcsinh(x, 0.0)


def test_strategy_properties():
    s = cytopk.GatingStrategy(STRATEGY)
    assert s.gate_names == ["cd3", "activated"]
    assert s.gates[1] == {"name": "activated", "parent": "cd3", "bounds": {"CD69": (1.0, float("inf"))}}
    assert dict(s.cofactors) == {"CD3": 5.0, "CD69": 150.0}
    assert s.medians == ["CD3", "CD69"] and s.channels == ["CD3", "CD69"]
    assert "activated" in repr(s)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_apply_matches_numpy(dtype):
    d = data().astype(dtype)
    s = cytopk.GatingStrategy(STRATEGY)
    r = s.apply(d, names=["CD3", "CD69", "X"])
    cd3, cd69, g1, g2 = naive(d.astype(np.float64))
    assert r["n_events"] == 5000
    a, b = r["gates"]
    assert (a["name"], a["parent"], b["parent"]) == ("cd3", None, "cd3")
    assert a["count"] == g1.sum() and b["count"] == g2.sum()
    assert a["fraction_of_total"] == pytest.approx(g1.mean(), rel=1e-15)
    assert b["fraction_of_parent"] == pytest.approx(g2.sum() / g1.sum(), rel=1e-15)
    assert b["medians"]["CD3"] == pytest.approx(np.median(cd3[g2]), rel=1e-14)
    assert b["medians"]["CD69"] == pytest.approx(np.median(cd69[g2]), rel=1e-14)
    m = s.masks(d, names=["CD3", "CD69", "X"])
    np.testing.assert_array_equal(m["cd3"], g1)
    np.testing.assert_array_equal(m["activated"], g2)


def test_fcs_file_and_many(tmp_path):
    d = data().astype(np.float32)
    s = cytopk.GatingStrategy(STRATEGY)
    paths = []
    for k in range(3):
        p = tmp_path / f"s{k}.fcs"
        cytopk.write_fcs(p, d[k * 1000:(k + 2) * 1000], ["CD3", "CD69", "X"])
        paths.append(str(p))
    (tmp_path / "bad.fcs").write_bytes(b"not an fcs file")
    many = cytopk.gate_fcs_many(paths + [str(tmp_path / "bad.fcs")], s, n_threads=2)
    for p, r in zip(paths, many):
        assert r == s.apply(cytopk.read_fcs(p))
    assert isinstance(many[-1], cytopk.FcsReadError)
    f = cytopk.read_fcs(paths[0])
    assert s.apply(f) == s.apply(np.asarray(f.data), names=f.names)


def test_strategy_errors(tmp_path):
    with pytest.raises(ValueError, match="parent"):
        cytopk.GatingStrategy("[[gate]]\nname='b'\nparent='a'\nrange={X=[0.0,1.0]}")
    with pytest.raises(ValueError, match="TOML"):
        cytopk.GatingStrategy("[[gate]]\nname='a'\npolygon=[1]")
    s = cytopk.GatingStrategy(STRATEGY)
    with pytest.raises(ValueError, match="CD69"):
        s.apply(np.zeros((3, 2)), names=["CD3", "X"])
    with pytest.raises(TypeError):
        s.apply(np.zeros((3, 2)))
    with pytest.raises(TypeError):
        s.apply(np.zeros((3, 2), dtype=np.int64), names=["CD3", "CD69"])
    p = tmp_path / "s.toml"
    p.write_text(STRATEGY)
    assert cytopk.GatingStrategy.from_file(p).gate_names == ["cd3", "activated"]
    r = cytopk.gate_fcs_many([], s)
    assert r == []
