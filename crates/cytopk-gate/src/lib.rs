//! Thin gating layer for cytopk (M6, first part): the arcsinh transform with a configurable cofactor,
//! rectangular gates defined in TOML, and per-gate event counts, fractions and medians.
//!
//! Out of scope here (later milestones): compensation, `$PnE` log decoding, polygon/ellipse gates, logicle.
//!
//! # Strategy file
//!
//! ```toml
//! [transform]
//! cofactor = 5.0                 # default arcsinh cofactor
//! arcsinh = ["CD3", "CD69"]      # channels transformed as asinh(x / cofactor); others stay linear
//! [transform.cofactors]          # optional per-channel cofactors (these channels are transformed too)
//! CD69 = 150.0
//!
//! [stats]
//! medians = ["CD3", "CD69"]      # medians reported for every gate
//!
//! [[gate]]
//! name = "cd3"
//! range = { CD3 = [1.5, inf] }   # lo <= x <= hi on the transformed scale; -inf / inf for open ends
//!
//! [[gate]]
//! name = "activated"
//! parent = "cd3"                 # events must also be in the parent; parents are defined earlier
//! range = { CD69 = [2.0, inf] }
//! ```
//!
//! Gates and medians use the transformed values. A NaN value is never inside a gate and is left out of medians.

use std::collections::{BTreeMap, HashMap, HashSet};

use ndarray::ArrayView2;
use rayon::prelude::*;
use serde::Deserialize;

/// Errors from parsing a strategy or applying it to data.
#[derive(Debug, thiserror::Error)]
pub enum GateError {
    /// The TOML did not parse or does not match the schema.
    #[error("strategy TOML: {0}")]
    Toml(String),
    /// The strategy parsed but is inconsistent (duplicate gate, unknown parent, bad bound or cofactor).
    #[error("strategy: {0}")]
    Invalid(String),
    /// A channel named in the strategy is not among the data's channel names.
    #[error("channel {0:?} is not in the data (channels: {1})")]
    MissingChannel(String, String),
    /// Data shape and channel names disagree.
    #[error("data has {0} columns but {1} channel names")]
    Shape(usize, usize),
}

/// `asinh(x / cofactor)`, the usual variance-stabilising transform for cytometry intensities.
#[inline]
pub fn arcsinh(x: f64, cofactor: f64) -> f64 {
    (x / cofactor).asinh()
}

/// Inverse of [`arcsinh`]: `cofactor * sinh(y)`.
#[inline]
pub fn arcsinh_inv(y: f64, cofactor: f64) -> f64 {
    cofactor * y.sinh()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStrategy {
    #[serde(default)]
    transform: RawTransform,
    #[serde(default)]
    stats: RawStats,
    #[serde(default)]
    gate: Vec<RawGate>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTransform {
    cofactor: Option<f64>,
    #[serde(default)]
    arcsinh: Vec<String>,
    #[serde(default)]
    cofactors: BTreeMap<String, f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStats {
    #[serde(default)]
    medians: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGate {
    name: String,
    parent: Option<String>,
    range: BTreeMap<String, [f64; 2]>,
}

/// Default arcsinh cofactor when `[transform] cofactor` is not given (a common choice for mass cytometry;
/// fluorescence data usually needs a larger one, so set it explicitly).
pub const DEFAULT_COFACTOR: f64 = 5.0;

/// One rectangular gate: every `(channel, lo, hi)` must hold, `lo <= x <= hi`, plus the parent gate if any.
#[derive(Debug, Clone, PartialEq)]
pub struct Gate {
    /// Unique gate name.
    pub name: String,
    /// Index of the parent gate in [`Strategy::gates`] (always an earlier gate), or `None` for all events.
    pub parent: Option<usize>,
    /// `(channel, lo, hi)` bounds on the transformed scale.
    pub bounds: Vec<(String, f64, f64)>,
}

/// A parsed gating strategy. Build it with [`Strategy::from_toml`].
#[derive(Debug, Clone, PartialEq)]
pub struct Strategy {
    /// Channel -> arcsinh cofactor, for the transformed channels (sorted by name).
    pub cofactors: Vec<(String, f64)>,
    /// Gates in definition order (parents first).
    pub gates: Vec<Gate>,
    /// Channels whose medians are reported for every gate.
    pub medians: Vec<String>,
}

/// Counts and medians of one gate in one sample.
#[derive(Debug, Clone, PartialEq)]
pub struct GateStats {
    /// Gate name.
    pub name: String,
    /// Parent gate name (`None`: all events).
    pub parent: Option<String>,
    /// Events in the gate.
    pub count: u64,
    /// `count / parent count` (NaN when the parent is empty).
    pub fraction_of_parent: f64,
    /// `count / all events` (NaN when the sample is empty).
    pub fraction_of_total: f64,
    /// `(channel, median)` on the transformed scale, in [`Strategy::medians`] order; NaN when no event of the
    /// gate has a non-NaN value. Even counts give the mean of the two middle values.
    pub medians: Vec<(String, f64)>,
}

/// Result of applying a strategy to one sample.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// All events in the sample.
    pub n_events: u64,
    /// One entry per gate, in strategy order.
    pub gates: Vec<GateStats>,
}

impl Strategy {
    /// Parse and validate a strategy (see the crate docs for the schema).
    pub fn from_toml(s: &str) -> Result<Self, GateError> {
        let raw: RawStrategy = toml::from_str(s).map_err(|e| GateError::Toml(e.to_string()))?;
        let default = raw.transform.cofactor.unwrap_or(DEFAULT_COFACTOR);
        let check_cof = |c: f64, what: &str| {
            if c.is_finite() && c > 0.0 {
                Ok(c)
            } else {
                Err(GateError::Invalid(format!(
                    "{what}: cofactor must be finite and > 0, got {c}"
                )))
            }
        };
        check_cof(default, "[transform] cofactor")?;
        let mut cof: BTreeMap<String, f64> = BTreeMap::new();
        for ch in &raw.transform.arcsinh {
            if cof.insert(ch.clone(), default).is_some() {
                return Err(GateError::Invalid(format!(
                    "channel {ch:?} listed twice in [transform] arcsinh"
                )));
            }
        }
        for (ch, &c) in &raw.transform.cofactors {
            cof.insert(
                ch.clone(),
                check_cof(c, &format!("[transform.cofactors] {ch}"))?,
            );
        }
        let mut index: HashMap<String, usize> = HashMap::new();
        let mut gates = Vec::with_capacity(raw.gate.len());
        for g in raw.gate {
            if g.name.is_empty() {
                return Err(GateError::Invalid("gate with an empty name".into()));
            }
            if index.contains_key(&g.name) {
                return Err(GateError::Invalid(format!(
                    "gate {:?} defined twice",
                    g.name
                )));
            }
            let parent = match &g.parent {
                None => None,
                Some(p) => Some(*index.get(p).ok_or_else(|| {
                    GateError::Invalid(format!(
                        "gate {:?}: parent {p:?} is not defined before it",
                        g.name
                    ))
                })?),
            };
            if g.range.is_empty() {
                return Err(GateError::Invalid(format!(
                    "gate {:?} has no range",
                    g.name
                )));
            }
            let mut bounds = Vec::with_capacity(g.range.len());
            for (ch, [lo, hi]) in g.range {
                if lo.is_nan() || hi.is_nan() || lo > hi {
                    return Err(GateError::Invalid(format!(
                        "gate {:?}, channel {ch:?}: need lo <= hi, got [{lo}, {hi}]",
                        g.name
                    )));
                }
                bounds.push((ch, lo, hi));
            }
            index.insert(g.name.clone(), gates.len());
            gates.push(Gate {
                name: g.name,
                parent,
                bounds,
            });
        }
        let mut seen = HashSet::new();
        for m in &raw.stats.medians {
            if !seen.insert(m) {
                return Err(GateError::Invalid(format!(
                    "channel {m:?} listed twice in [stats] medians"
                )));
            }
        }
        Ok(Self {
            cofactors: cof.into_iter().collect(),
            gates,
            medians: raw.stats.medians,
        })
    }

    /// Every channel the strategy reads (gate bounds and medians), sorted and unique.
    pub fn channels(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .gates
            .iter()
            .flat_map(|g| g.bounds.iter().map(|b| b.0.clone()))
            .chain(self.medians.iter().cloned())
            .collect();
        v.sort();
        v.dedup();
        v
    }

    fn cofactor(&self, ch: &str) -> Option<f64> {
        self.cofactors
            .iter()
            .find(|(c, _)| c == ch)
            .map(|&(_, f)| f)
    }

    /// Transformed columns of the channels the strategy uses.
    fn columns<T: Copy + Into<f64> + Sync>(
        &self,
        names: &[String],
        data: ArrayView2<'_, T>,
    ) -> Result<HashMap<String, Vec<f64>>, GateError> {
        if data.ncols() != names.len() {
            return Err(GateError::Shape(data.ncols(), names.len()));
        }
        let mut out = HashMap::new();
        for ch in self.channels() {
            let j = names
                .iter()
                .position(|n| *n == ch)
                .ok_or_else(|| GateError::MissingChannel(ch.clone(), names.join(", ")))?;
            let col = data.column(j);
            let v: Vec<f64> = match self.cofactor(&ch) {
                Some(c) => col.iter().map(|&x| arcsinh(x.into(), c)).collect(),
                None => col.iter().map(|&x| x.into()).collect(),
            };
            out.insert(ch, v);
        }
        Ok(out)
    }

    /// One boolean mask per gate (events x 1, in strategy order).
    pub fn masks<T: Copy + Into<f64> + Sync>(
        &self,
        names: &[String],
        data: ArrayView2<'_, T>,
    ) -> Result<Vec<Vec<bool>>, GateError> {
        let cols = self.columns(names, data)?;
        Ok(self.masks_from(&cols, data.nrows()))
    }

    fn masks_from(&self, cols: &HashMap<String, Vec<f64>>, n: usize) -> Vec<Vec<bool>> {
        let mut masks: Vec<Vec<bool>> = Vec::with_capacity(self.gates.len());
        for g in &self.gates {
            let mut m = match g.parent {
                Some(p) => masks[p].clone(),
                None => vec![true; n],
            };
            for (ch, lo, hi) in &g.bounds {
                let c = &cols[ch];
                for (mi, &x) in m.iter_mut().zip(c) {
                    *mi &= *lo <= x && x <= *hi;
                }
            }
            masks.push(m);
        }
        masks
    }

    /// Apply the strategy to one sample: `data` is events x channels, `names` its channel names ($PnN).
    pub fn apply<T: Copy + Into<f64> + Sync>(
        &self,
        names: &[String],
        data: ArrayView2<'_, T>,
    ) -> Result<Summary, GateError> {
        let n = data.nrows();
        let cols = self.columns(names, data)?;
        let masks = self.masks_from(&cols, n);
        let counts: Vec<u64> = masks
            .iter()
            .map(|m| m.iter().filter(|&&b| b).count() as u64)
            .collect();
        let total = n as u64;
        let gates = self
            .gates
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let parent_count = g.parent.map_or(total, |p| counts[p]);
                let medians = self
                    .medians
                    .par_iter()
                    .map(|ch| {
                        let mut v: Vec<f64> = cols[ch]
                            .iter()
                            .zip(&masks[i])
                            .filter(|(x, m)| **m && !x.is_nan())
                            .map(|(&x, _)| x)
                            .collect();
                        (ch.clone(), median(&mut v))
                    })
                    .collect();
                GateStats {
                    name: g.name.clone(),
                    parent: g.parent.map(|p| self.gates[p].name.clone()),
                    count: counts[i],
                    fraction_of_parent: ratio(counts[i], parent_count),
                    fraction_of_total: ratio(counts[i], total),
                    medians,
                }
            })
            .collect();
        Ok(Summary {
            n_events: total,
            gates,
        })
    }
}

fn ratio(a: u64, b: u64) -> f64 {
    if b == 0 {
        f64::NAN
    } else {
        a as f64 / b as f64
    }
}

/// Exact median (mean of the two middle values for even lengths); NaN for an empty slice. Reorders `v`.
pub fn median(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let (_, &mut hi, _) = v.select_nth_unstable_by(n / 2, f64::total_cmp);
    if n % 2 == 1 {
        return hi;
    }
    let lo = v[..n / 2].iter().copied().fold(f64::NEG_INFINITY, f64::max);
    0.5 * (lo + hi)
}
