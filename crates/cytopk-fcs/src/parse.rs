//! One dataset: HEADER -> TEXT -> keywords -> channels -> DATA -> $NEXTDATA.

use crate::error::{Diag, FcsError, WarningKind};
use crate::header::{HEADER_LEN, Header, Segment, Version, parse_header, trim_ascii};
use crate::spillover::find_spillover;
use crate::text::{RawPair, parse_text};
use crate::{ByteOrder, Channel, Data, DataType, FcsFile, ReadOptions};
use ndarray::Array2;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Above this many values the DATA decode runs on the Rayon pool.
const PAR_DECODE_VALUES: usize = 1 << 20;
/// Upper bound on $PAR. Every parameter needs at least four keywords, so a larger value means a corrupt file.
const MAX_PAR: usize = 100_000;

pub(crate) struct Parsed {
    pub file: FcsFile,
    /// Absolute offset of the next dataset, if $NEXTDATA points to one.
    pub next: Option<u64>,
}

pub(crate) fn parse_dataset(
    file: &[u8],
    base: u64,
    opts: &ReadOptions,
) -> Result<Parsed, FcsError> {
    let mut d = Diag::new(opts.strict);
    let len = file.len() as u64;
    let header = parse_header(file, base, &mut d)?;

    // ---- TEXT
    let text_seg = text_bounds(&header, base, len, &mut d)?;
    let raw = parse_text(
        &file[text_seg.0 as usize..=text_seg.1 as usize],
        text_seg.0,
        &mut d,
    )?;
    let mut kw = Keywords::build(&raw, &mut d)?;
    supplemental_text(file, base, len, &mut kw, &mut d)?;

    // ---- required keywords
    let mode = kw.get("$MODE");
    match mode.as_deref().map(str::trim) {
        Some("L") | Some("l") => {}
        None => d.problem(
            WarningKind::MissingKeyword,
            "$MODE missing; list mode assumed",
            None,
            Some("$MODE"),
            |m| key_err("$MODE", m),
        )?,
        Some(other) => {
            return Err(FcsError::Unsupported(format!(
                "$MODE {other:?} (only list mode L is supported)"
            )));
        }
    }
    let n_par = kw
        .required_uint("$PAR", &mut d)?
        .ok_or_else(|| key_err("$PAR", "missing"))?;
    let n_par = usize::try_from(n_par).unwrap_or(usize::MAX);
    if n_par > MAX_PAR || n_par > raw.len() {
        return Err(key_err(
            "$PAR",
            format!("{n_par} parameters but only {} keywords", raw.len()),
        ));
    }
    let datatype = match kw.get("$DATATYPE").as_deref().map(str::trim) {
        Some("F") | Some("f") => DataType::F,
        Some("D") | Some("d") => DataType::D,
        Some("I") | Some("i") => DataType::I,
        Some("A") | Some("a") => {
            return Err(FcsError::Unsupported("$DATATYPE A (ASCII)".into()));
        }
        Some(o) => return Err(key_err("$DATATYPE", format!("unknown value {o:?}"))),
        None => return Err(key_err("$DATATYPE", "missing")),
    };
    let byte_order = parse_byteorder(kw.get("$BYTEORD"))?;

    // ---- channels
    let mut channels = Vec::with_capacity(n_par);
    for p in 1..=n_par {
        channels.push(channel(&kw, p, datatype, &mut d)?);
    }
    let bytes_per_event: u64 = channels.iter().map(|c| u64::from(c.bits / 8)).sum();

    // ---- DATA
    let tot = kw.required_uint("$TOT", &mut d)?;
    if tot.is_none() {
        d.problem(
            WarningKind::MissingKeyword,
            "$TOT missing; event count taken from the DATA length",
            None,
            Some("$TOT"),
            |m| key_err("$TOT", m),
        )?;
    }
    let (data_abs, n_events) = data_bounds(
        &header,
        &kw,
        base,
        len,
        tot,
        bytes_per_event,
        (text_seg.0 - base, text_seg.1 - base),
        &mut d,
    )?;
    let data = decode(
        file, data_abs, n_events, &channels, datatype, byte_order, &mut d,
    )?;

    // ---- ANALYSIS
    let analysis = analysis(file, &header, &kw, base, len, &mut d)?;

    // ---- spillover
    let names: Vec<String> = channels.iter().map(|c| c.name.clone()).collect();
    let spillover = find_spillover(|k| kw.get(k), &names, &mut d)?;

    // ---- $NEXTDATA
    let next = next_dataset(&kw, base, len, &mut d)?;

    Ok(Parsed {
        file: FcsFile {
            offset: base,
            header,
            text: kw.map,
            raw_text: raw,
            channels,
            datatype,
            byte_order,
            data,
            analysis,
            spillover,
            warnings: d.warnings,
        },
        next,
    })
}

fn key_err(k: &str, m: impl Into<String>) -> FcsError {
    FcsError::Keyword {
        keyword: k.into(),
        msg: m.into(),
    }
}

/// Absolute inclusive TEXT range.
fn text_bounds(h: &Header, base: u64, len: u64, d: &mut Diag) -> Result<(u64, u64), FcsError> {
    let herr = |m: String| FcsError::Header {
        field: "TEXT offsets",
        offset: base + 10,
        msg: m,
    };
    let begin = base + h.text.begin;
    let mut end = base + h.text.end;
    if h.text.begin < HEADER_LEN as u64 || h.text.end < h.text.begin || begin >= len {
        return Err(herr(format!(
            "TEXT [{}, {}] invalid for a file of {len} bytes",
            h.text.begin, h.text.end
        )));
    }
    if end >= len {
        d.problem(
            WarningKind::TextOffsets,
            format!("TEXT end {end} beyond the file end; clipped"),
            Some(base + 18),
            None,
            herr,
        )?;
        end = len - 1;
    }
    // TEXT is never clipped on the HEADER's word: a wrong HEADER DATA start inside TEXT is a known quirk, and
    // clipping there would drop the $BEGINDATA/$ENDDATA keywords that locate the real DATA.
    Ok((begin, end.min(len - 1)))
}

/// Normalised keyword map (upper-case keys; the first of duplicate keys wins in lenient mode).
struct Keywords {
    map: BTreeMap<String, String>,
}

impl Keywords {
    fn build(raw: &[RawPair], d: &mut Diag) -> Result<Self, FcsError> {
        let mut map = BTreeMap::new();
        for p in raw {
            let trimmed = p.key.trim();
            if trimmed.len() != p.key.len() {
                d.problem(
                    WarningKind::Delimiter,
                    format!("keyword {:?} has surrounding spaces; trimmed", p.key),
                    None,
                    Some(trimmed),
                    |m| key_err(trimmed, m),
                )?;
            }
            let k = trimmed.to_ascii_uppercase();
            if map.contains_key(&k) {
                d.problem(
                    WarningKind::DuplicateKeyword,
                    "keyword appears more than once; first value kept",
                    None,
                    Some(&k),
                    |m| key_err(&k, m),
                )?;
                continue;
            }
            map.insert(k, p.value.clone());
        }
        Ok(Self { map })
    }

    fn get(&self, k: &str) -> Option<String> {
        self.map.get(k).cloned()
    }

    fn get_ref(&self, k: &str) -> Option<&str> {
        self.map.get(k).map(String::as_str)
    }

    /// Unsigned integer value. `Ok(None)` if absent; a malformed value is an error.
    fn required_uint(&self, k: &str, d: &mut Diag) -> Result<Option<u64>, FcsError> {
        let Some(v) = self.get_ref(k) else {
            return Ok(None);
        };
        parse_uint(v, k, d).map(Some)
    }
}

/// Unsigned integer; lenient mode also accepts surrounding spaces and an integral float such as "1000.0".
fn parse_uint(v: &str, k: &str, d: &mut Diag) -> Result<u64, FcsError> {
    if let Ok(x) = v.parse::<u64>() {
        return Ok(x);
    }
    let t = v.trim();
    let lenient = t.parse::<u64>().ok().or_else(|| {
        t.parse::<f64>()
            .ok()
            .filter(|x| x.is_finite() && *x >= 0.0 && x.fract() == 0.0 && *x < 1.8e19)
            .map(|x| x as u64)
    });
    match lenient {
        Some(x) => {
            d.problem(
                WarningKind::BadValue,
                format!("{v:?} is not a plain unsigned integer; read as {x}"),
                None,
                Some(k),
                |m| key_err(k, m),
            )?;
            Ok(x)
        }
        None => Err(key_err(k, format!("{v:?} is not an unsigned integer"))),
    }
}

fn parse_byteorder(v: Option<String>) -> Result<ByteOrder, FcsError> {
    let Some(v) = v else {
        return Err(key_err("$BYTEORD", "missing"));
    };
    let s: String = v.chars().filter(|c| !c.is_whitespace()).collect();
    match s.as_str() {
        "1,2,3,4" | "1,2" | "1,2,3,4,5,6,7,8" => Ok(ByteOrder::Little),
        "4,3,2,1" | "2,1" | "8,7,6,5,4,3,2,1" => Ok(ByteOrder::Big),
        _ => Err(FcsError::Unsupported(format!(
            "$BYTEORD {v:?} (only 1,2,3,4 and 4,3,2,1)"
        ))),
    }
}

fn channel(kw: &Keywords, p: usize, dt: DataType, d: &mut Diag) -> Result<Channel, FcsError> {
    let key = |s: &str| format!("$P{p}{s}");
    let kb = key("B");
    let bits_raw = kw
        .get_ref(&kb)
        .ok_or_else(|| key_err(&kb, "missing"))?
        .trim()
        .to_owned();
    let bits: u32 = match bits_raw.parse() {
        Ok(b) => b,
        Err(_) if bits_raw == "*" => {
            return Err(FcsError::Unsupported(format!(
                "{kb} = * (free-format ASCII data)"
            )));
        }
        Err(_) => return Err(key_err(&kb, format!("{bits_raw:?} is not a bit width"))),
    };
    let want = match dt {
        DataType::F => Some(32),
        DataType::D => Some(64),
        DataType::I => None,
    };
    let bits = match want {
        Some(w) if bits != w => {
            d.problem(
                WarningKind::BadValue,
                format!("{bits} bits for $DATATYPE {dt:?}; {w} used"),
                None,
                Some(&kb),
                |m| key_err(&kb, m),
            )?;
            w
        }
        Some(w) => w,
        None => {
            if bits == 0 || bits > 64 || bits % 8 != 0 {
                return Err(FcsError::Unsupported(format!(
                    "{kb} = {bits} (integer widths must be whole bytes, 8 to 64 bits)"
                )));
            }
            bits
        }
    };
    let kn = key("N");
    let name = match kw.get(&kn) {
        Some(n) if !n.is_empty() => n,
        _ => {
            d.problem(
                WarningKind::MissingKeyword,
                format!("{kn} missing; named P{p}"),
                None,
                Some(&kn),
                |m| key_err(&kn, m),
            )?;
            format!("P{p}")
        }
    };
    let kr = key("R");
    let range = match kw.get_ref(&kr).map(str::trim) {
        Some(r) => match r.parse::<f64>() {
            Ok(x) if x.is_finite() && x >= 0.0 => x,
            _ => {
                d.problem(
                    WarningKind::BadValue,
                    format!("{kr} = {r:?} is not a number; ignored"),
                    None,
                    Some(&kr),
                    |m| key_err(&kr, m),
                )?;
                0.0
            }
        },
        None => {
            d.problem(
                WarningKind::MissingKeyword,
                format!("{kr} missing"),
                None,
                Some(&kr),
                |m| key_err(&kr, m),
            )?;
            0.0
        }
    };
    let ke = key("E");
    let log = match kw.get_ref(&ke) {
        Some(e) => match parse_pair(e) {
            Some(x) => x,
            None => {
                d.problem(
                    WarningKind::BadValue,
                    format!("{ke} = {e:?} is not \"f1,f2\"; 0,0 used"),
                    None,
                    Some(&ke),
                    |m| key_err(&ke, m),
                )?;
                (0.0, 0.0)
            }
        },
        None => {
            // $PnE is required from FCS 3.0 on; FCS 2.0 files often omit it (linear).
            d.problem(
                WarningKind::MissingKeyword,
                format!("{ke} missing; linear (0,0) assumed"),
                None,
                Some(&ke),
                |m| key_err(&ke, m),
            )?;
            (0.0, 0.0)
        }
    };
    let gain = kw
        .get_ref(&key("G"))
        .and_then(|g| g.trim().parse::<f64>().ok());
    Ok(Channel {
        name,
        long_name: kw.get(&key("S")).filter(|s| !s.is_empty()),
        bits,
        range,
        log,
        gain,
    })
}

fn parse_pair(s: &str) -> Option<(f64, f64)> {
    let (a, b) = s.split_once(',')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

/// Returns the absolute start of DATA and the number of events to decode.
#[allow(clippy::too_many_arguments)] // internal; the arguments are the parsed pieces it reconciles
fn data_bounds(
    h: &Header,
    kw: &Keywords,
    base: u64,
    len: u64,
    tot: Option<u64>,
    bpe: u64,
    text: (u64, u64),
    d: &mut Diag,
) -> Result<(u64, u64), FcsError> {
    let derr = |m: String| FcsError::Data(m);
    let mut quiet = Diag::new(false);
    let kb = kw.required_uint("$BEGINDATA", &mut quiet).ok().flatten();
    let ke = kw.required_uint("$ENDDATA", &mut quiet).ok().flatten();
    let from_kw = match (kb, ke) {
        (Some(b), Some(e)) => Some(Segment { begin: b, end: e }),
        _ => None,
    };
    if from_kw.is_none() && h.version != Version::Fcs2_0 {
        d.problem(
            WarningKind::MissingKeyword,
            "$BEGINDATA/$ENDDATA missing or malformed; HEADER offsets used",
            None,
            Some("$BEGINDATA"),
            derr,
        )?;
    }
    let expected = tot.map(|t| t.saturating_mul(bpe));
    // A segment is plausible if it lies after the HEADER, inside the file, and fits the expected size.
    // Plausible: after the HEADER, inside the file, outside TEXT, and large enough for $TOT events.
    let fits = |s: &Segment| {
        s.begin >= HEADER_LEN as u64
            && s.end >= s.begin
            && base.saturating_add(s.end) < len
            && (s.begin > text.1 || s.end < text.0)
            && expected.is_none_or(|e| s.end - s.begin + 1 >= e)
    };
    let seg = match (h.data.is_zero(), from_kw) {
        (true, Some(k)) => k,
        (true, None) => {
            if tot == Some(0) || bpe == 0 {
                return Ok((base, 0));
            }
            return Err(derr("no DATA offsets in HEADER or TEXT".into()));
        }
        (false, None) => h.data,
        (false, Some(k)) if k == h.data || k.is_zero() => h.data,
        (false, Some(k)) => {
            // Prefer the TEXT keywords (the specification's authority for 3.x and FlowIO's default), then
            // the HEADER, among segments that are plausible; otherwise the keywords if they lie inside the file.
            let inside = |s: &Segment| {
                s.begin >= HEADER_LEN as u64 && s.end >= s.begin && base.saturating_add(s.end) < len
            };
            let pick = if fits(&k) {
                k
            } else if fits(&h.data) {
                h.data
            } else if inside(&k) || !inside(&h.data) {
                k
            } else {
                h.data
            };
            d.problem(
                WarningKind::DataOffsets,
                format!(
                    "HEADER DATA [{}, {}] and $BEGINDATA/$ENDDATA [{}, {}] disagree; using [{}, {}]",
                    h.data.begin, h.data.end, k.begin, k.end, pick.begin, pick.end
                ),
                Some(base + 26),
                Some("$BEGINDATA"),
                derr,
            )?;
            pick
        }
    };
    if tot == Some(0) || bpe == 0 {
        return Ok((base.saturating_add(seg.begin).min(len), 0));
    }
    if seg.begin < HEADER_LEN as u64 || seg.end < seg.begin {
        return Err(derr(format!(
            "DATA [{}, {}] is not a valid range",
            seg.begin, seg.end
        )));
    }
    let begin = base.saturating_add(seg.begin);
    if begin >= len {
        return Err(derr(format!(
            "DATA starts at byte {begin}, beyond the file end ({len} bytes)"
        )));
    }
    let mut end = base.saturating_add(seg.end);
    if end >= len {
        d.problem(
            WarningKind::DataOffsets,
            format!("DATA end {end} is beyond the file end ({len} bytes); clipped"),
            Some(begin),
            None,
            derr,
        )?;
        end = len - 1;
    }
    let have = end - begin + 1;
    let fit = have / bpe;
    let n = match tot {
        Some(t) => {
            let e = t.saturating_mul(bpe);
            if have != e {
                let msg = if have > e {
                    format!(
                        "DATA has {have} bytes, $TOT x bytes/event = {e}; first {t} events read ({} extra bytes)",
                        have - e
                    )
                } else {
                    format!(
                        "DATA has {have} bytes, $TOT x bytes/event = {e}; {fit} complete events read"
                    )
                };
                d.problem(WarningKind::DataSize, msg, Some(begin), Some("$TOT"), derr)?;
            }
            t.min(fit)
        }
        None => {
            if have % bpe != 0 {
                d.problem(
                    WarningKind::DataSize,
                    format!("DATA has {have} bytes, not a multiple of {bpe} bytes/event"),
                    Some(begin),
                    None,
                    derr,
                )?;
            }
            fit
        }
    };
    Ok((begin, n))
}

/// Mask for integer channel values: the smallest all-ones mask covering $PnR - 1 (FCS 3.1, section 3.3.12),
/// or every bit of the word when $PnR is missing, zero, or larger than the word.
fn int_mask(bits: u32, range: f64) -> u64 {
    let full = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    if range.is_nan() || range < 1.0 {
        return full;
    }
    let r = range.ceil();
    if r >= full as f64 {
        return full;
    }
    (r as u64)
        .checked_next_power_of_two()
        .map_or(full, |m| (m - 1) & full)
}

fn decode(
    file: &[u8],
    begin: u64,
    n_events: u64,
    ch: &[Channel],
    dt: DataType,
    bo: ByteOrder,
    _d: &mut Diag,
) -> Result<Data, FcsError> {
    let n_par = ch.len();
    let n = usize::try_from(n_events).map_err(|_| FcsError::Data("too many events".into()))?;
    let bpe: usize = ch.iter().map(|c| (c.bits / 8) as usize).sum();
    let start = begin as usize;
    let bytes = &file[start..start + n * bpe]; // in bounds: n <= available bytes / bpe
    let le = bo == ByteOrder::Little;
    match dt {
        DataType::F => {
            let conv = |c: &[u8]| {
                let a = [c[0], c[1], c[2], c[3]];
                if le {
                    f32::from_le_bytes(a)
                } else {
                    f32::from_be_bytes(a)
                }
            };
            let v: Vec<f32> = if n * n_par >= PAR_DECODE_VALUES {
                bytes.par_chunks_exact(4).map(conv).collect()
            } else {
                bytes.chunks_exact(4).map(conv).collect()
            };
            Ok(Data::F32(shape(v, n, n_par)?))
        }
        DataType::D => {
            let conv = |c: &[u8]| {
                let mut a = [0u8; 8];
                a.copy_from_slice(c);
                if le {
                    f64::from_le_bytes(a)
                } else {
                    f64::from_be_bytes(a)
                }
            };
            let v: Vec<f64> = if n * n_par >= PAR_DECODE_VALUES {
                bytes.par_chunks_exact(8).map(conv).collect()
            } else {
                bytes.chunks_exact(8).map(conv).collect()
            };
            Ok(Data::F64(shape(v, n, n_par)?))
        }
        DataType::I => {
            let widths: Vec<usize> = ch.iter().map(|c| (c.bits / 8) as usize).collect();
            let masks: Vec<u64> = ch.iter().map(|c| int_mask(c.bits, c.range)).collect();
            let row = |ev: &[u8], out: &mut [f64]| {
                let mut o = 0;
                for (j, (&w, &m)) in widths.iter().zip(&masks).enumerate() {
                    let b = &ev[o..o + w];
                    let mut x = 0u64;
                    if le {
                        for &byte in b.iter().rev() {
                            x = (x << 8) | u64::from(byte);
                        }
                    } else {
                        for &byte in b {
                            x = (x << 8) | u64::from(byte);
                        }
                    }
                    out[j] = (x & m) as f64;
                    o += w;
                }
            };
            let mut v = vec![0f64; n * n_par];
            if n_par > 0 {
                if n * n_par >= PAR_DECODE_VALUES {
                    v.par_chunks_exact_mut(n_par)
                        .zip(bytes.par_chunks_exact(bpe))
                        .for_each(|(out, ev)| row(ev, out));
                } else {
                    for (out, ev) in v.chunks_exact_mut(n_par).zip(bytes.chunks_exact(bpe)) {
                        row(ev, out);
                    }
                }
            }
            Ok(Data::F64(shape(v, n, n_par)?))
        }
    }
}

fn shape<T>(v: Vec<T>, n: usize, p: usize) -> Result<Array2<T>, FcsError> {
    Array2::from_shape_vec((n, p), v).map_err(|e| FcsError::Data(e.to_string()))
}

fn analysis(
    file: &[u8],
    h: &Header,
    kw: &Keywords,
    base: u64,
    len: u64,
    d: &mut Diag,
) -> Result<Option<Vec<u8>>, FcsError> {
    let mut quiet = Diag::new(false);
    let seg = if !h.analysis.is_zero() {
        h.analysis
    } else {
        match (
            kw.required_uint("$BEGINANALYSIS", &mut quiet)
                .ok()
                .flatten(),
            kw.required_uint("$ENDANALYSIS", &mut quiet).ok().flatten(),
        ) {
            (Some(b), Some(e)) => Segment { begin: b, end: e },
            _ => return Ok(None),
        }
    };
    if seg.is_zero() {
        return Ok(None);
    }
    if seg.begin < HEADER_LEN as u64 || seg.end < seg.begin || base.saturating_add(seg.end) >= len {
        d.problem(
            WarningKind::Analysis,
            format!("ANALYSIS [{}, {}] invalid; ignored", seg.begin, seg.end),
            Some(base + 42),
            None,
            |m| FcsError::Header {
                field: "ANALYSIS offsets",
                offset: base + 42,
                msg: m,
            },
        )?;
        return Ok(None);
    }
    Ok(Some(
        file[(base + seg.begin) as usize..=(base + seg.end) as usize].to_vec(),
    ))
}

fn supplemental_text(
    file: &[u8],
    base: u64,
    len: u64,
    kw: &mut Keywords,
    d: &mut Diag,
) -> Result<(), FcsError> {
    let mut quiet = Diag::new(false);
    let (Some(b), Some(e)) = (
        kw.required_uint("$BEGINSTEXT", &mut quiet).ok().flatten(),
        kw.required_uint("$ENDSTEXT", &mut quiet).ok().flatten(),
    ) else {
        return Ok(());
    };
    if b == 0 && e == 0 {
        return Ok(());
    }
    let serr = |m: String| FcsError::Text {
        offset: base + b,
        msg: m,
    };
    if b < HEADER_LEN as u64 || e < b || base.saturating_add(e) >= len {
        return d.problem(
            WarningKind::SupplementalText,
            format!("supplemental TEXT [{b}, {e}] invalid; ignored"),
            Some(base + b),
            Some("$BEGINSTEXT"),
            serr,
        );
    }
    let mut sd = Diag::new(d.strict);
    let pairs = match parse_text(
        &file[(base + b) as usize..=(base + e) as usize],
        base + b,
        &mut sd,
    ) {
        Ok(p) => p,
        Err(err) => {
            return d.problem(
                WarningKind::SupplementalText,
                format!("supplemental TEXT unreadable ({err}); ignored"),
                Some(base + b),
                Some("$BEGINSTEXT"),
                serr,
            );
        }
    };
    d.warnings.append(&mut sd.warnings);
    for p in pairs {
        let k = p.key.trim().to_ascii_uppercase();
        match kw.map.entry(k) {
            std::collections::btree_map::Entry::Vacant(e) => {
                e.insert(p.value);
            }
            std::collections::btree_map::Entry::Occupied(e) => {
                let k = e.key().clone();
                d.problem(
                    WarningKind::DuplicateKeyword,
                    "keyword in both TEXT and supplemental TEXT; primary kept",
                    None,
                    Some(&k),
                    |m| key_err(&k, m),
                )?;
            }
        }
    }
    Ok(())
}

fn next_dataset(kw: &Keywords, base: u64, len: u64, d: &mut Diag) -> Result<Option<u64>, FcsError> {
    let raw = kw.get_ref("$NEXTDATA").map(trim_str);
    let off = match raw {
        None => return Ok(None), // FCS 2.0 often lacks it; nothing to follow
        Some(v) => match v.parse::<u64>() {
            Ok(x) => x,
            Err(_) => {
                d.problem(
                    WarningKind::NextData,
                    format!("$NEXTDATA {v:?} is not a number; no further datasets read"),
                    None,
                    Some("$NEXTDATA"),
                    |m| key_err("$NEXTDATA", m),
                )?;
                return Ok(None);
            }
        },
    };
    if off == 0 {
        return Ok(None);
    }
    let abs = base.saturating_add(off);
    if abs.saturating_add(HEADER_LEN as u64) > len {
        d.problem(
            WarningKind::NextData,
            format!("$NEXTDATA {off} points beyond the file end; ignored"),
            None,
            Some("$NEXTDATA"),
            |m| key_err("$NEXTDATA", m),
        )?;
        return Ok(None);
    }
    Ok(Some(abs))
}

fn trim_str(s: &str) -> &str {
    // `trim_ascii` on bytes keeps this consistent with the HEADER parser.
    std::str::from_utf8(trim_ascii(s.as_bytes())).unwrap_or(s)
}
