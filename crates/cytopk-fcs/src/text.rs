//! TEXT segment tokeniser.
//!
//! The first byte is the delimiter. Keywords and values alternate, each followed by the delimiter. A delimiter
//! inside a keyword or value is written twice. The specification forbids empty values, so in a conforming file
//! a doubled delimiter is always an escape.
//!
//! Real files break this in a few known ways (Bras & van der Velden 2020, Cytometry A 97:1180). Lenient mode
//! repairs them, and strict mode reports them as errors:
//! - **Empty values.** `/$KEY//$NEXT/v/` means `$KEY` = "" in practice, but a literal reading of the spec makes
//!   the keyword `$KEY/$NEXT`. Lenient rule: a doubled delimiter that directly follows a *keyword* ends the
//!   keyword and marks an empty value. Doubled delimiters inside *values* stay escapes, so a value such as
//!   `01//02//2020` (with `/` as delimiter) still reads as `01/02/2020`.
//!   Strict mode follows the literal reading and then rejects a keyword that contains the delimiter.
//! - **Missing closing delimiter.** The last value runs to the end of the segment. It is kept, with a warning.
//! - **Doubled closing delimiter** (`.../v//`). The extra delimiter is ignored, with a warning.
//! - **Trailing bytes after the last delimiter** (padding, or bytes of DATA when the TEXT end offset is too
//!   large). They are ignored, with a warning, if they would form a keyword with no value.
//! - **An odd number of tokens.** The dangling keyword is dropped, with a warning.
//! - **Not UTF-8.** Decoded as Latin-1, with a warning.
//! - **Spaces around keywords.** Trimmed, with a warning.

use crate::error::{Diag, FcsError, WarningKind};

/// One keyword/value pair as stored in the file (keyword case and surrounding spaces preserved).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawPair {
    /// Keyword as written.
    pub key: String,
    /// Value as written (escapes removed).
    pub value: String,
}

/// Tokenise a TEXT (or supplemental TEXT) segment. `seg_offset` is the absolute offset of `seg[0]`.
pub(crate) fn parse_text(
    seg: &[u8],
    seg_offset: u64,
    d: &mut Diag,
) -> Result<Vec<RawPair>, FcsError> {
    let Some(&delim) = seg.first() else {
        return Err(FcsError::Text {
            offset: seg_offset,
            msg: "empty TEXT segment".into(),
        });
    };
    if d.strict && !(1..=126).contains(&delim) {
        return Err(FcsError::Text {
            offset: seg_offset,
            msg: format!("delimiter byte {delim:#04x} is not ASCII 1-126"),
        });
    }
    let n = seg.len();
    let mut tokens: Vec<Vec<u8>> = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    let mut i = 1;
    let mut closed = true; // the last token was terminated by a delimiter
    while i < n {
        let b = seg[i];
        if b != delim {
            cur.push(b);
            closed = false;
            i += 1;
            continue;
        }
        let doubled = i + 1 < n && seg[i + 1] == delim;
        if !doubled {
            tokens.push(std::mem::take(&mut cur));
            closed = true;
            i += 1;
            continue;
        }
        let in_keyword = tokens.len() % 2 == 0;
        let at_end = i + 2 >= n;
        if !d.strict && in_keyword && !cur.is_empty() {
            // `/$KEY//...`: keyword, then an empty value.
            let key = String::from_utf8_lossy(&cur).into_owned();
            d.warn(
                WarningKind::EmptyValue,
                "empty keyword value (doubled delimiter after a keyword)",
                Some(seg_offset + i as u64),
                Some(&key.trim().to_ascii_uppercase()),
            );
            tokens.push(std::mem::take(&mut cur));
            tokens.push(Vec::new());
            closed = true;
            i += 2;
            continue;
        }
        if !d.strict && at_end && !in_keyword {
            // `.../value//` at the very end: a closing delimiter plus a spare one.
            d.warn(
                WarningKind::Delimiter,
                "doubled closing delimiter",
                Some(seg_offset + i as u64),
                None,
            );
            tokens.push(std::mem::take(&mut cur));
            closed = true;
            i += 2;
            continue;
        }
        cur.push(delim); // escaped delimiter
        closed = false;
        i += 2;
    }
    if !closed {
        let off = Some(seg_offset + n as u64);
        if tokens.len() % 2 == 1 {
            d.problem(
                WarningKind::Delimiter,
                "TEXT does not end with the delimiter; last value kept",
                off,
                None,
                |m| FcsError::Text {
                    offset: seg_offset + n as u64,
                    msg: m,
                },
            )?;
            tokens.push(std::mem::take(&mut cur));
        } else {
            d.problem(
                WarningKind::Delimiter,
                format!(
                    "{} trailing bytes after the last delimiter ignored",
                    cur.len()
                ),
                off,
                None,
                |m| FcsError::Text {
                    offset: seg_offset + n as u64,
                    msg: m,
                },
            )?;
        }
    }
    if tokens.len() % 2 == 1 {
        let k = tokens.pop().unwrap_or_default();
        d.problem(
            WarningKind::Delimiter,
            format!(
                "keyword {:?} has no value; dropped",
                String::from_utf8_lossy(&k)
            ),
            Some(seg_offset + n as u64),
            None,
            |m| FcsError::Text {
                offset: seg_offset + n as u64,
                msg: m,
            },
        )?;
    }
    let mut latin1 = false;
    let mut out = Vec::with_capacity(tokens.len() / 2);
    let mut it = tokens.into_iter();
    while let (Some(k), Some(v)) = (it.next(), it.next()) {
        let key = decode(k, &mut latin1);
        let value = decode(v, &mut latin1);
        if d.strict && key.as_bytes().contains(&delim) {
            return Err(FcsError::Text {
                offset: seg_offset,
                msg: format!("keyword {key:?} contains the delimiter (an empty value before it?)"),
            });
        }
        if key.trim().is_empty() {
            d.problem(
                WarningKind::Delimiter,
                "empty keyword; pair dropped",
                Some(seg_offset),
                None,
                |m| FcsError::Text {
                    offset: seg_offset,
                    msg: m,
                },
            )?;
            continue;
        }
        if value.is_empty() && d.strict {
            return Err(FcsError::Keyword {
                keyword: key.to_ascii_uppercase(),
                msg: "empty value".into(),
            });
        }
        out.push(RawPair { key, value });
    }
    if latin1 {
        d.problem(
            WarningKind::Encoding,
            "TEXT is not valid UTF-8; decoded as Latin-1",
            Some(seg_offset),
            None,
            |m| FcsError::Text {
                offset: seg_offset,
                msg: m,
            },
        )?;
    }
    Ok(out)
}

fn decode(bytes: Vec<u8>, latin1: &mut bool) -> String {
    match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => {
            *latin1 = true;
            e.into_bytes().iter().map(|&b| char::from(b)).collect()
        }
    }
}
