//! The 58-byte HEADER: version string, four spaces, six right-justified 8-byte ASCII offsets.

use crate::error::{Diag, FcsError, WarningKind};

/// FCS version from the HEADER.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Version {
    /// FCS 2.0 (lenient mode only).
    Fcs2_0,
    /// FCS 3.0.
    Fcs3_0,
    /// FCS 3.1.
    Fcs3_1,
    /// FCS 3.2 (lenient mode only; read with the 3.1 rules).
    Fcs3_2,
}

impl Version {
    /// The six-byte version string.
    pub fn as_str(self) -> &'static str {
        match self {
            Version::Fcs2_0 => "FCS2.0",
            Version::Fcs3_0 => "FCS3.0",
            Version::Fcs3_1 => "FCS3.1",
            Version::Fcs3_2 => "FCS3.2",
        }
    }
}

/// Inclusive byte range `[begin, end]` relative to the start of the dataset. `begin == end == 0` means absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Segment {
    /// First byte.
    pub begin: u64,
    /// Last byte (inclusive).
    pub end: u64,
}

impl Segment {
    /// Both offsets are zero.
    pub fn is_zero(&self) -> bool {
        self.begin == 0 && self.end == 0
    }
}

/// Parsed HEADER of one dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct Header {
    /// Version string.
    pub version: Version,
    /// TEXT segment as given in the HEADER.
    pub text: Segment,
    /// DATA segment as given in the HEADER (may be 0,0 when the offsets are only in TEXT).
    pub data: Segment,
    /// ANALYSIS segment as given in the HEADER.
    pub analysis: Segment,
}

pub(crate) const HEADER_LEN: usize = 58;

pub(crate) fn parse_header(file: &[u8], base: u64, d: &mut Diag) -> Result<Header, FcsError> {
    let have = (file.len() as u64).saturating_sub(base);
    if have < HEADER_LEN as u64 {
        return Err(FcsError::TooShort { offset: base, have });
    }
    let b = base as usize; // fits: base < file.len()
    let h = &file[b..b + HEADER_LEN];
    let version = match &h[0..6] {
        b"FCS3.0" => Version::Fcs3_0,
        b"FCS3.1" => Version::Fcs3_1,
        b"FCS2.0" | b"FCS3.2" => {
            let v = if &h[0..6] == b"FCS2.0" {
                Version::Fcs2_0
            } else {
                Version::Fcs3_2
            };
            d.problem(
                WarningKind::Header,
                format!("{} is read with the FCS 3.1 rules", v.as_str()),
                Some(base),
                None,
                |_| FcsError::Version {
                    found: v.as_str().into(),
                    offset: base,
                },
            )?;
            v
        }
        other => {
            return Err(FcsError::Version {
                found: String::from_utf8_lossy(other).into_owned(),
                offset: base,
            });
        }
    };
    if h[6..10] != *b"    " {
        d.problem(
            WarningKind::Header,
            "bytes 6-9 of the HEADER are not spaces",
            Some(base + 6),
            None,
            |m| FcsError::Header {
                field: "filler",
                offset: base + 6,
                msg: m,
            },
        )?;
    }
    const NAMES: [&str; 6] = [
        "TEXT begin",
        "TEXT end",
        "DATA begin",
        "DATA end",
        "ANALYSIS begin",
        "ANALYSIS end",
    ];
    let mut v = [0u64; 6];
    for (i, name) in NAMES.iter().enumerate() {
        let off = 10 + 8 * i;
        v[i] = parse_field(&h[off..off + 8], name, base + off as u64, d)?;
    }
    Ok(Header {
        version,
        text: Segment {
            begin: v[0],
            end: v[1],
        },
        data: Segment {
            begin: v[2],
            end: v[3],
        },
        analysis: Segment {
            begin: v[4],
            end: v[5],
        },
    })
}

/// Right-justified, space-padded unsigned integer. All spaces means 0 (common for an absent ANALYSIS).
fn parse_field(
    raw: &[u8],
    field: &'static str,
    offset: u64,
    d: &mut Diag,
) -> Result<u64, FcsError> {
    let s = trim_ascii(raw);
    if s.is_empty() {
        return Ok(0);
    }
    let trailing = raw.last().is_some_and(|c| *c == b' ');
    if s.iter().all(u8::is_ascii_digit) && !trailing {
        return Ok(s.iter().fold(0u64, |a, c| a * 10 + u64::from(c - b'0')));
    }
    if s.iter().all(u8::is_ascii_digit) {
        d.problem(
            WarningKind::Header,
            format!("{field} is left-justified"),
            Some(offset),
            None,
            |m| FcsError::Header {
                field,
                offset,
                msg: m,
            },
        )?;
        return Ok(s.iter().fold(0u64, |a, c| a * 10 + u64::from(c - b'0')));
    }
    d.problem(
        WarningKind::Header,
        format!(
            "{field} is not a number ({:?}); treated as 0",
            String::from_utf8_lossy(raw)
        ),
        Some(offset),
        None,
        |m| FcsError::Header {
            field,
            offset,
            msg: m,
        },
    )?;
    Ok(0)
}

pub(crate) fn trim_ascii(mut s: &[u8]) -> &[u8] {
    while let [first, rest @ ..] = s {
        if first.is_ascii_whitespace() {
            s = rest;
        } else {
            break;
        }
    }
    while let [rest @ .., last] = s {
        if last.is_ascii_whitespace() {
            s = rest;
        } else {
            break;
        }
    }
    s
}
