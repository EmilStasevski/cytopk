//! `$SPILLOVER` (FCS 3.1), and the non-standard `$SPILL` and `$COMP` variants.
//!
//! `$SPILLOVER` / `$SPILL` format: `n,name_1,...,name_n,s_11,s_12,...,s_nn`, with the matrix row-major.
//! Row i, column j is the fraction of the signal of fluorochrome i (primarily detected in channel name_i) that
//! appears in channel name_j, so the diagonal is 1. Compensation of an event row `x` (restricted to the named
//! channels, in this order) is `x_comp = x * S^-1`. This crate only parses the matrix and does not invert it.
//! `$COMP` (FCS 2.0/3.0) is read only when it has the same `n,names,values` layout. The older form, without
//! names, is reported as malformed.

use crate::error::{Diag, FcsError, WarningKind};
use ndarray::Array2;

/// A spillover matrix and the channels it applies to.
#[derive(Debug, Clone, PartialEq)]
pub struct Spillover {
    /// Keyword it was read from (`$SPILLOVER`, `$SPILL` or `$COMP`).
    pub keyword: String,
    /// Channel names ($PnN) in matrix order.
    pub channels: Vec<String>,
    /// n x n row-major matrix (see the module docs for the convention).
    pub matrix: Array2<f64>,
}

/// Parse the first spillover keyword present. Malformed values are a warning in lenient mode, an error in
/// strict mode, and never stop a lenient read.
pub(crate) fn find_spillover(
    get: impl Fn(&str) -> Option<String>,
    channel_names: &[String],
    d: &mut Diag,
) -> Result<Option<Spillover>, FcsError> {
    for key in ["$SPILLOVER", "$SPILL", "SPILL", "$COMP"] {
        let Some(v) = get(key) else { continue };
        match parse(&v, channel_names) {
            Ok(s) => {
                return Ok(Some(Spillover {
                    keyword: key.to_owned(),
                    ..s
                }));
            }
            Err(msg) => {
                d.problem(
                    WarningKind::Spillover,
                    format!("malformed spillover matrix ignored: {msg}"),
                    None,
                    Some(key),
                    |m| FcsError::Keyword {
                        keyword: key.into(),
                        msg: m,
                    },
                )?;
            }
        }
    }
    Ok(None)
}

fn parse(v: &str, channel_names: &[String]) -> Result<Spillover, String> {
    let parts: Vec<&str> = v.split(',').map(str::trim).collect();
    let n: usize = parts
        .first()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("first field {:?} is not a count", parts.first()))?;
    if n == 0 {
        return Err("n = 0".into());
    }
    let need = n
        .checked_mul(n)
        .and_then(|nn| nn.checked_add(n + 1))
        .ok_or("n too large")?;
    if parts.len() != need {
        return Err(format!(
            "{} fields, expected 1 + n + n^2 = {need} for n = {n}",
            parts.len()
        ));
    }
    let channels: Vec<String> = parts[1..=n].iter().map(|s| (*s).to_owned()).collect();
    if let Some(bad) = channels.iter().find(|c| c.parse::<f64>().is_ok()) {
        return Err(format!(
            "channel name {bad:?} is a number (matrix without names?)"
        ));
    }
    if let Some(missing) = channels.iter().find(|c| !channel_names.contains(c)) {
        return Err(format!("channel {missing:?} is not a $PnN of this file"));
    }
    let mut vals = Vec::with_capacity(n * n);
    for s in &parts[n + 1..] {
        let x: f64 = s
            .parse()
            .map_err(|_| format!("value {s:?} is not a number"))?;
        if !x.is_finite() {
            return Err(format!("value {s:?} is not finite"));
        }
        vals.push(x);
    }
    let matrix = Array2::from_shape_vec((n, n), vals).map_err(|e| e.to_string())?;
    Ok(Spillover {
        keyword: String::new(),
        channels,
        matrix,
    })
}
