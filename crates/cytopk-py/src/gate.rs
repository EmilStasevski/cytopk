//! Gating bindings: `arcsinh`, the `GatingStrategy` class (TOML rectangular gates) and `gate_fcs_many`.

use crate::fcs::PyFcsFile;
use cytopk_fcs::{Data, FcsError, ReadOptions, Source};
use cytopk_gate::{GateError, Strategy, Summary};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

fn to_py(e: GateError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `asinh(x / cofactor)`, element-wise, as float64.
#[pyfunction]
#[pyo3(signature = (x, cofactor=cytopk_gate::DEFAULT_COFACTOR))]
fn arcsinh<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
    cofactor: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    if !(cofactor.is_finite() && cofactor > 0.0) {
        return Err(PyValueError::new_err("cofactor must be finite and > 0"));
    }
    Ok(x.as_array()
        .mapv(|v| cytopk_gate::arcsinh(v, cofactor))
        .into_pyarray(py))
}

/// A gating strategy parsed from TOML: arcsinh cofactors, rectangular gates (with parents) and the channels
/// whose medians are reported. See `docs/gating.md` for the schema.
#[pyclass(name = "GatingStrategy", module = "cytopk", frozen)]
pub struct PyStrategy {
    inner: Strategy,
}

fn summary_dict<'py>(py: Python<'py>, s: &Summary) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n_events", s.n_events)?;
    let gates = PyList::empty(py);
    for g in &s.gates {
        let gd = PyDict::new(py);
        gd.set_item("name", &g.name)?;
        gd.set_item("parent", &g.parent)?;
        gd.set_item("count", g.count)?;
        gd.set_item("fraction_of_parent", g.fraction_of_parent)?;
        gd.set_item("fraction_of_total", g.fraction_of_total)?;
        let m = PyDict::new(py);
        for (ch, v) in &g.medians {
            m.set_item(ch, v)?;
        }
        gd.set_item("medians", m)?;
        gates.append(gd)?;
    }
    d.set_item("gates", gates)?;
    Ok(d)
}

impl PyStrategy {
    /// Run `f` on (names, data) taken from an FcsFile, or from a 2-D float32/float64 array plus `names`.
    fn with_data<R>(
        &self,
        data: &Bound<'_, PyAny>,
        names: Option<Vec<String>>,
        f: impl Fn(&Strategy, &[String], DataView<'_>) -> Result<R, GateError>,
    ) -> PyResult<R> {
        if let Ok(file) = data.cast::<PyFcsFile>() {
            if names.is_some() {
                return Err(PyTypeError::new_err(
                    "names must not be given with an FcsFile",
                ));
            }
            let file = file.get();
            let names = file.names();
            let arr = file.data_bound(data.py());
            return self.with_data(&arr, Some(names), f);
        }
        let names =
            names.ok_or_else(|| PyTypeError::new_err("names are required with an array"))?;
        if let Ok(a) = data.extract::<PyReadonlyArray2<'_, f32>>() {
            f(&self.inner, &names, DataView::F32(a.as_array())).map_err(to_py)
        } else if let Ok(a) = data.extract::<PyReadonlyArray2<'_, f64>>() {
            f(&self.inner, &names, DataView::F64(a.as_array())).map_err(to_py)
        } else {
            Err(PyTypeError::new_err(
                "data must be an FcsFile or a 2-D float32/float64 array",
            ))
        }
    }
}

enum DataView<'a> {
    F32(ndarray::ArrayView2<'a, f32>),
    F64(ndarray::ArrayView2<'a, f64>),
}

fn apply(s: &Strategy, names: &[String], d: DataView<'_>) -> Result<Summary, GateError> {
    match d {
        DataView::F32(a) => s.apply(names, a),
        DataView::F64(a) => s.apply(names, a),
    }
}

#[pymethods]
impl PyStrategy {
    /// Parse a strategy from a TOML string.
    #[new]
    fn new(toml: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Strategy::from_toml(toml).map_err(to_py)?,
        })
    }

    /// Parse a strategy from a TOML file.
    #[staticmethod]
    fn from_file(path: std::path::PathBuf) -> PyResult<Self> {
        let s = std::fs::read_to_string(&path)
            .map_err(|e| PyValueError::new_err(format!("{}: {e}", path.display())))?;
        Self::new(&s)
    }

    /// Gate names in definition order.
    #[getter]
    fn gate_names(&self) -> Vec<String> {
        self.inner.gates.iter().map(|g| g.name.clone()).collect()
    }

    /// Gates as dicts: name, parent, bounds {channel: (lo, hi)}.
    #[getter]
    fn gates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for g in &self.inner.gates {
            let d = PyDict::new(py);
            d.set_item("name", &g.name)?;
            d.set_item("parent", g.parent.map(|p| self.inner.gates[p].name.clone()))?;
            let b = PyDict::new(py);
            for (ch, lo, hi) in &g.bounds {
                b.set_item(ch, (*lo, *hi))?;
            }
            d.set_item("bounds", b)?;
            out.append(d)?;
        }
        Ok(out)
    }

    /// {channel: cofactor} for the arcsinh-transformed channels.
    #[getter]
    fn cofactors(&self) -> Vec<(String, f64)> {
        self.inner.cofactors.clone()
    }

    /// Channels whose medians are reported.
    #[getter]
    fn medians(&self) -> Vec<String> {
        self.inner.medians.clone()
    }

    /// Every channel the strategy reads.
    #[getter]
    fn channels(&self) -> Vec<String> {
        self.inner.channels()
    }

    /// Apply to one sample: an FcsFile, or a 2-D array with `names`. Returns
    /// {"n_events", "gates": [{"name", "parent", "count", "fraction_of_parent", "fraction_of_total",
    /// "medians": {channel: value}}]}. Medians are on the transformed scale.
    #[pyo3(signature = (data, names=None))]
    fn apply<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        names: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let s = self.with_data(data, names, apply)?;
        summary_dict(py, &s)
    }

    /// {gate name: boolean mask over events}.
    #[pyo3(signature = (data, names=None))]
    fn masks<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        names: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let m = self.with_data(data, names, |s, n, d| match d {
            DataView::F32(a) => s.masks(n, a),
            DataView::F64(a) => s.masks(n, a),
        })?;
        let out = PyDict::new(py);
        for (g, mask) in self.inner.gates.iter().zip(m) {
            out.set_item(&g.name, mask.into_pyarray(py))?;
        }
        Ok(out)
    }

    fn __repr__(&self) -> String {
        format!("GatingStrategy(gates={:?})", self.gate_names())
    }
}

/// Read and gate many FCS files in parallel (Rayon), without materialising the events in Python.
/// Returns one summary dict per path (as `GatingStrategy.apply`), or the exception for that path.
#[pyfunction]
#[pyo3(signature = (paths, strategy, strict=false, n_threads=None))]
fn gate_fcs_many<'py>(
    py: Python<'py>,
    paths: Vec<std::path::PathBuf>,
    strategy: &PyStrategy,
    strict: bool,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyList>> {
    enum E {
        Read(FcsError),
        Gate(GateError),
    }
    let o = if strict {
        ReadOptions::strict()
    } else {
        ReadOptions::default()
    };
    let s = &strategy.inner;
    let work = || -> Vec<Result<Summary, E>> {
        paths
            .par_iter()
            .map(|p| {
                let src = Source::open(p).map_err(E::Read)?;
                let f = cytopk_fcs::read(src.bytes(), &o).map_err(E::Read)?;
                let names: Vec<String> = f.channels.iter().map(|c| c.name.clone()).collect();
                match &f.data {
                    Data::F32(a) => s.apply(&names, a.view()),
                    Data::F64(a) => s.apply(&names, a.view()),
                }
                .map_err(E::Gate)
            })
            .collect()
    };
    let res = match n_threads {
        None => py.detach(work),
        Some(0) => return Err(PyValueError::new_err("n_threads must be >= 1")),
        Some(n) => {
            let pool = crate::pool(n)?;
            py.detach(|| pool.install(work))
        }
    };
    let out = PyList::empty(py);
    for r in res {
        match r {
            Ok(s) => out.append(summary_dict(py, &s)?)?,
            Err(E::Read(e)) => {
                out.append(crate::fcs::FcsReadError::new_err(e.to_string()).into_value(py))?
            }
            Err(E::Gate(e)) => out.append(to_py(e).into_value(py))?,
        }
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStrategy>()?;
    m.add_function(wrap_pyfunction!(arcsinh, m)?)?;
    m.add_function(wrap_pyfunction!(gate_fcs_many, m)?)?;
    Ok(())
}
