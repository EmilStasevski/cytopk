//! FCS writer: list mode, `$DATATYPE` F (default), D or I, either byte order, any delimiter, one or more
//! datasets chained with `$NEXTDATA`. The output follows FCS 3.1 (or 3.0 when asked), so the strict reader
//! accepts it.
//!
//! Layout per dataset: HEADER (58 bytes), TEXT, DATA. Offsets are relative to the dataset start. When DATA
//! ends beyond byte 99,999,999, the HEADER DATA fields are 0 and only `$BEGINDATA`/`$ENDDATA` locate it, as
//! the specification requires. No CRC is written.

use crate::error::FcsError;
use crate::header::{HEADER_LEN, Version};
use crate::{ByteOrder, DataType};
use ndarray::ArrayView2;

/// Borrowed event matrix, events x parameters.
#[derive(Debug, Clone, Copy)]
pub enum DataRef<'a> {
    /// f32 values.
    F32(ArrayView2<'a, f32>),
    /// f64 values.
    F64(ArrayView2<'a, f64>),
}

impl DataRef<'_> {
    fn dim(&self) -> (usize, usize) {
        match self {
            DataRef::F32(a) => a.dim(),
            DataRef::F64(a) => a.dim(),
        }
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        match self {
            DataRef::F32(a) => f64::from(a[[i, j]]),
            DataRef::F64(a) => a[[i, j]],
        }
    }
}

/// One output parameter.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct WriteChannel {
    /// `$PnN` (required, non-empty).
    pub name: String,
    /// `$PnS`.
    pub long_name: Option<String>,
    /// `$PnR`. Default: 262144 for F/D, 2^bits for I.
    pub range: Option<f64>,
    /// `$PnB` for `$DATATYPE` I (8, 16, 32 or 64; default 32). Ignored for F and D.
    pub bits: Option<u32>,
}

impl WriteChannel {
    /// Channel with a name and defaults for everything else.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }
}

/// One dataset to write.
#[derive(Debug, Clone, Copy)]
pub struct DatasetSpec<'a> {
    /// Events x parameters.
    pub data: DataRef<'a>,
    /// One entry per column of `data`.
    pub channels: &'a [WriteChannel],
    /// Extra keyword/value pairs, written after the required ones. They may not repeat a keyword the writer
    /// sets itself.
    pub keywords: &'a [(String, String)],
}

/// Writer options.
#[derive(Debug, Clone, Copy)]
pub struct WriteOptions {
    /// FCS3.0 or FCS3.1 (default).
    pub version: Version,
    /// Output data type (default F).
    pub datatype: DataType,
    /// Default little-endian.
    pub byte_order: ByteOrder,
    /// TEXT delimiter (default `/`). Must be ASCII 1-126.
    pub delimiter: u8,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            version: Version::Fcs3_1,
            datatype: DataType::F,
            byte_order: ByteOrder::Little,
            delimiter: b'/',
        }
    }
}

/// Write one dataset.
pub fn write_fcs<'a>(
    data: DataRef<'a>,
    channels: &'a [WriteChannel],
    keywords: &'a [(String, String)],
    opts: &WriteOptions,
) -> Result<Vec<u8>, FcsError> {
    write_datasets(
        &[DatasetSpec {
            data,
            channels,
            keywords,
        }],
        opts,
    )
}

/// Write several datasets chained with `$NEXTDATA`.
pub fn write_datasets(sets: &[DatasetSpec<'_>], opts: &WriteOptions) -> Result<Vec<u8>, FcsError> {
    let werr = |m: String| FcsError::Write(m);
    if !matches!(opts.version, Version::Fcs3_0 | Version::Fcs3_1) {
        return Err(werr("only FCS3.0 and FCS3.1 can be written".into()));
    }
    if !(1..=126).contains(&opts.delimiter) {
        return Err(werr("delimiter must be ASCII 1-126".into()));
    }
    if sets.is_empty() {
        return Err(werr("no datasets".into()));
    }
    let mut out = Vec::new();
    for (k, s) in sets.iter().enumerate() {
        let last = k + 1 == sets.len();
        out.extend_from_slice(&dataset(s, opts, last)?);
    }
    Ok(out)
}

fn dataset(s: &DatasetSpec<'_>, o: &WriteOptions, last: bool) -> Result<Vec<u8>, FcsError> {
    let werr = |m: String| FcsError::Write(m);
    let (n, p) = s.data.dim();
    if s.channels.len() != p {
        return Err(werr(format!(
            "{} channels for {p} data columns",
            s.channels.len()
        )));
    }
    if p == 0 {
        return Err(werr("at least one parameter is required".into()));
    }
    let bits: Vec<u32> = s
        .channels
        .iter()
        .map(|c| match o.datatype {
            DataType::F => Ok(32),
            DataType::D => Ok(64),
            DataType::I => match c.bits.unwrap_or(32) {
                b @ (8 | 16 | 32 | 64) => Ok(b),
                b => Err(werr(format!("{}: {b} bits (use 8, 16, 32 or 64)", c.name))),
            },
        })
        .collect::<Result<_, _>>()?;
    let data = encode(s, o, &bits)?;

    let mut fixed: Vec<(String, String)> = vec![
        (
            "$BYTEORD".into(),
            match o.byte_order {
                ByteOrder::Little => "1,2,3,4".into(),
                ByteOrder::Big => "4,3,2,1".into(),
            },
        ),
        (
            "$DATATYPE".into(),
            match o.datatype {
                DataType::F => "F".into(),
                DataType::D => "D".into(),
                DataType::I => "I".into(),
            },
        ),
        ("$MODE".into(), "L".into()),
        ("$PAR".into(), p.to_string()),
        ("$TOT".into(), n.to_string()),
    ];
    for (j, c) in s.channels.iter().enumerate() {
        let q = j + 1;
        if c.name.is_empty() {
            return Err(werr(format!("channel {q} has an empty name")));
        }
        let range = c.range.unwrap_or(match o.datatype {
            DataType::I => 2f64.powi(bits[j] as i32),
            _ => 262_144.0,
        });
        fixed.push((format!("$P{q}B"), bits[j].to_string()));
        fixed.push((format!("$P{q}E"), "0,0".into()));
        fixed.push((format!("$P{q}N"), c.name.clone()));
        fixed.push((format!("$P{q}R"), fmt_num(range)));
        if let Some(l) = &c.long_name {
            fixed.push((format!("$P{q}S"), l.clone()));
        }
    }
    let reserved = |k: &str| {
        let k = k.trim().to_ascii_uppercase();
        fixed.iter().any(|(f, _)| *f == k)
            || matches!(
                k.as_str(),
                "$BEGINDATA"
                    | "$ENDDATA"
                    | "$BEGINANALYSIS"
                    | "$ENDANALYSIS"
                    | "$BEGINSTEXT"
                    | "$ENDSTEXT"
                    | "$NEXTDATA"
            )
    };
    for (k, _) in s.keywords {
        if reserved(k) {
            return Err(werr(format!("keyword {k} is set by the writer")));
        }
    }

    // Offsets appear inside TEXT, so iterate until the TEXT length stops changing.
    let data_len = data.len() as u64;
    let (mut bd, mut ed, mut nx) = (0u64, 0u64, 0u64);
    for _ in 0..8 {
        let text = text_segment(&fixed, s.keywords, (bd, ed, nx), o.delimiter)?;
        let text_end = HEADER_LEN as u64 + text.len() as u64 - 1;
        let (nbd, ned) = if data_len == 0 {
            (0, 0)
        } else {
            (text_end + 1, text_end + data_len)
        };
        let nnx = if last { 0 } else { text_end + 1 + data_len };
        if (nbd, ned, nnx) == (bd, ed, nx) {
            if text_end > 99_999_999 {
                return Err(werr("TEXT longer than the HEADER can address".into()));
            }
            let (hb, he) = if ned > 99_999_999 { (0, 0) } else { (nbd, ned) };
            let mut out = Vec::with_capacity(HEADER_LEN + text.len() + data.len());
            out.extend_from_slice(o.version.as_str().as_bytes());
            out.extend_from_slice(b"    ");
            for v in [HEADER_LEN as u64, text_end, hb, he, 0, 0] {
                out.extend_from_slice(format!("{v:>8}").as_bytes());
            }
            out.extend_from_slice(&text);
            out.extend_from_slice(&data);
            return Ok(out);
        }
        (bd, ed, nx) = (nbd, ned, nnx);
    }
    Err(werr("TEXT offsets did not converge".into()))
}

fn fmt_num(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

fn text_segment(
    fixed: &[(String, String)],
    extra: &[(String, String)],
    (bd, ed, nx): (u64, u64, u64),
    delim: u8,
) -> Result<Vec<u8>, FcsError> {
    let offsets = [
        ("$BEGINANALYSIS", 0),
        ("$BEGINDATA", bd),
        ("$BEGINSTEXT", 0),
        ("$ENDANALYSIS", 0),
        ("$ENDDATA", ed),
        ("$ENDSTEXT", 0),
        ("$NEXTDATA", nx),
    ];
    let mut t = vec![delim];
    let mut push = |s: &str, is_key: bool| -> Result<(), FcsError> {
        if s.is_empty() {
            return Err(FcsError::Write("empty keyword or value".into()));
        }
        let b = s.as_bytes();
        if is_key && b.contains(&delim) {
            return Err(FcsError::Write(format!(
                "keyword {s:?} contains the delimiter"
            )));
        }
        if !is_key && b[0] == delim {
            // Legal, but lenient readers (this one included) read `/KEY///x/` as an empty value.
            return Err(FcsError::Write(format!(
                "value {s:?} starts with the delimiter; choose another delimiter"
            )));
        }
        for &c in b {
            t.push(c);
            if c == delim {
                t.push(delim);
            }
        }
        t.push(delim);
        Ok(())
    };
    for (k, v) in offsets {
        push(k, true)?;
        push(&v.to_string(), false)?;
    }
    for (k, v) in fixed.iter().chain(extra) {
        push(k, true)?;
        push(v, false)?;
    }
    Ok(t)
}

fn encode(s: &DatasetSpec<'_>, o: &WriteOptions, bits: &[u32]) -> Result<Vec<u8>, FcsError> {
    let (n, p) = s.data.dim();
    let le = o.byte_order == ByteOrder::Little;
    let bytes_per_event: usize = bits.iter().map(|b| (*b / 8) as usize).sum();
    let mut out = Vec::with_capacity(n * bytes_per_event);
    for i in 0..n {
        for (j, &b) in bits.iter().enumerate().take(p) {
            match o.datatype {
                DataType::F => {
                    let x = match s.data {
                        DataRef::F32(a) => a[[i, j]],
                        DataRef::F64(a) => a[[i, j]] as f32,
                    };
                    out.extend_from_slice(&if le { x.to_le_bytes() } else { x.to_be_bytes() });
                }
                DataType::D => {
                    let x = s.data.get(i, j);
                    out.extend_from_slice(&if le { x.to_le_bytes() } else { x.to_be_bytes() });
                }
                DataType::I => {
                    let x = s.data.get(i, j);
                    let max = if b == 64 {
                        u64::MAX as f64
                    } else {
                        (1u64 << b) as f64
                    };
                    if !(x.is_finite() && x >= 0.0 && x.fract() == 0.0 && x < max) {
                        return Err(FcsError::Write(format!(
                            "event {i}, {}: {x} is not an unsigned {b}-bit integer",
                            s.channels[j].name
                        )));
                    }
                    let v = x as u64;
                    let w = (b / 8) as usize;
                    let all = v.to_le_bytes();
                    if le {
                        out.extend_from_slice(&all[..w]);
                    } else {
                        out.extend(all[..w].iter().rev());
                    }
                }
            }
        }
    }
    Ok(out)
}
