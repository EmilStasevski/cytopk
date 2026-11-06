//! Test helpers: hand-built FCS files whose TEXT can be edited freely.
//!
//! DATA always starts at byte `DATA_AT`, so editing TEXT never moves it.
#![allow(dead_code, clippy::unwrap_used)]

pub const DATA_AT: u64 = 4096;

/// A hand-built single-dataset file.
#[derive(Clone)]
pub struct Raw {
    pub version: &'static str,
    pub delim: u8,
    /// Keyword/value pairs, written in this order and escaped (a delimiter in a value is doubled).
    pub pairs: Vec<(String, String)>,
    pub data: Vec<u8>,
    /// HEADER DATA fields; default `(DATA_AT, DATA_AT + len - 1)`.
    pub header_data: Option<(u64, u64)>,
    /// Bytes appended to the TEXT segment after the last delimiter (still inside the HEADER's TEXT range).
    pub text_tail: Vec<u8>,
}

/// Standard list-mode F32 little-endian file: `n` events, channels named `names`, values `i*10 + j`.
pub fn f32_file(n: usize, names: &[&str]) -> Raw {
    let p = names.len();
    let mut data = Vec::new();
    for i in 0..n {
        for j in 0..p {
            data.extend_from_slice(&((i * 10 + j) as f32).to_le_bytes());
        }
    }
    let mut pairs = vec![
        kv("$BYTEORD", "1,2,3,4"),
        kv("$DATATYPE", "F"),
        kv("$MODE", "L"),
        kv("$NEXTDATA", "0"),
        kv("$PAR", &p.to_string()),
        kv("$TOT", &n.to_string()),
    ];
    for (j, name) in names.iter().enumerate() {
        let q = j + 1;
        pairs.push(kv(&format!("$P{q}B"), "32"));
        pairs.push(kv(&format!("$P{q}E"), "0,0"));
        pairs.push(kv(&format!("$P{q}N"), name));
        pairs.push(kv(&format!("$P{q}R"), "262144"));
    }
    let len = data.len() as u64;
    let mut r = Raw {
        version: "FCS3.1",
        delim: b'/',
        pairs,
        data,
        header_data: None,
        text_tail: Vec::new(),
    };
    r.set("$BEGINDATA", &DATA_AT.to_string());
    r.set("$ENDDATA", &(DATA_AT + len.max(1) - 1).to_string());
    r
}

pub fn kv(k: &str, v: &str) -> (String, String) {
    (k.to_owned(), v.to_owned())
}

impl Raw {
    pub fn set(&mut self, k: &str, v: &str) {
        if let Some(p) = self.pairs.iter_mut().find(|p| p.0 == k) {
            p.1 = v.to_owned();
        } else {
            self.pairs.push(kv(k, v));
        }
    }

    pub fn remove(&mut self, k: &str) {
        self.pairs.retain(|p| p.0 != k);
    }

    /// TEXT bytes exactly as they will be written.
    pub fn text(&self) -> Vec<u8> {
        let d = self.delim;
        let mut t = vec![d];
        for (k, v) in &self.pairs {
            for part in [k, v] {
                for &c in part.as_bytes() {
                    t.push(c);
                    if c == d {
                        t.push(d);
                    }
                }
                t.push(d);
            }
        }
        t.extend_from_slice(&self.text_tail);
        t
    }

    /// Build with the TEXT produced by `text()`.
    pub fn build(&self) -> Vec<u8> {
        self.build_with_text(&self.text())
    }

    /// Build with arbitrary TEXT bytes (for quirks `text()` cannot express).
    pub fn build_with_text(&self, text: &[u8]) -> Vec<u8> {
        assert!(
            58 + text.len() as u64 <= DATA_AT,
            "TEXT too long for the test layout"
        );
        let len = self.data.len() as u64;
        let (db, de) = self
            .header_data
            .unwrap_or((DATA_AT, DATA_AT + len.max(1) - 1));
        let mut f = Vec::new();
        f.extend_from_slice(self.version.as_bytes());
        f.extend_from_slice(b"    ");
        for v in [58, 58 + text.len() as u64 - 1, db, de, 0, 0] {
            f.extend_from_slice(format!("{v:>8}").as_bytes());
        }
        f.extend_from_slice(text);
        f.resize(DATA_AT as usize, b' ');
        f.extend_from_slice(&self.data);
        f
    }
}

/// Replace the first occurrence of `from` with `to` in `hay` (panics if absent: a broken test).
pub fn replace(hay: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
    let i = hay
        .windows(from.len())
        .position(|w| w == from)
        .unwrap_or_else(|| panic!("{:?} not found", String::from_utf8_lossy(from)));
    let mut v = hay[..i].to_vec();
    v.extend_from_slice(to);
    v.extend_from_slice(&hay[i + from.len()..]);
    v
}
