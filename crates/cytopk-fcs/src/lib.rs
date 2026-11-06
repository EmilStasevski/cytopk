//! Robust reader and small writer for FCS 3.0/3.1 flow-cytometry files.
//!
//! - **Reads** list-mode data with `$DATATYPE` F (f32), D (f64) or I (unsigned integers of any whole-byte width up to 64 bits,
//!   mixed widths allowed, masked by `$PnR`), little- or big-endian, multiple datasets (`$NEXTDATA`),
//!   supplemental TEXT, ANALYSIS and `$SPILLOVER`. FCS 2.0 and 3.2 are read in lenient mode with a warning.
//! - **Two modes.** [`ReadOptions::strict`] rejects anything the specification forbids. Lenient mode (the
//!   default) repairs the known real-world problems and records each repair as a [`FcsWarning`]. The repairs
//!   are listed in the docs of [`text`] and `docs/fcs_notes.md`.
//! - **Never panics** on any input (tested with mutation proptests and cargo-fuzz), and never allocates more
//!   than the file can back: the event count is capped by the bytes actually present.
//! - Files are memory-mapped ([`read_path`]); [`read_many`] reads many files in parallel.
//! - With the `arrow` feature: [`FcsFile::to_record_batch`] and [`FcsFile::write_parquet`].

#[cfg(feature = "arrow")]
pub mod arrow_io;
pub mod error;
pub mod header;
mod parse;
pub mod spillover;
pub mod text;
pub mod write;

pub use error::{FcsError, FcsWarning, WarningKind};
pub use header::{Header, Segment, Version};
pub use spillover::Spillover;
pub use text::RawPair;
pub use write::{DataRef, DatasetSpec, WriteChannel, WriteOptions, write_datasets, write_fcs};

use ndarray::Array2;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;

/// `$DATATYPE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// IEEE 754 single precision.
    F,
    /// IEEE 754 double precision.
    D,
    /// Unsigned integers with per-parameter widths `$PnB`.
    I,
}

/// `$BYTEORD`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteOrder {
    /// `1,2,3,4`.
    Little,
    /// `4,3,2,1`.
    Big,
}

/// One parameter (channel).
#[derive(Debug, Clone, PartialEq)]
pub struct Channel {
    /// `$PnN`, short name.
    pub name: String,
    /// `$PnS`, long name / stain, if present and non-empty.
    pub long_name: Option<String>,
    /// `$PnB`, bits per value.
    pub bits: u32,
    /// `$PnR`, range (0 if missing or unparsable).
    pub range: f64,
    /// `$PnE` as `(decades, offset)`; `(0, 0)` is linear. Values are returned as stored, not converted.
    pub log: (f64, f64),
    /// `$PnG`, gain, if present.
    pub gain: Option<f64>,
}

/// Event data, events x parameters, row-major (the file layout).
#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    /// `$DATATYPE` F.
    F32(Array2<f32>),
    /// `$DATATYPE` D, or I (integers are converted to f64 exactly; widths up to 53 bits are exact).
    F64(Array2<f64>),
}

impl Data {
    /// (events, parameters).
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Data::F32(a) => a.dim(),
            Data::F64(a) => a.dim(),
        }
    }

    /// Copy to f64.
    pub fn to_f64(&self) -> Array2<f64> {
        match self {
            Data::F32(a) => a.mapv(f64::from),
            Data::F64(a) => a.clone(),
        }
    }
}

/// One dataset of an FCS file.
#[derive(Debug, Clone, PartialEq)]
pub struct FcsFile {
    /// Absolute byte offset of this dataset in the file (0 for the first).
    pub offset: u64,
    /// The HEADER as written.
    pub header: Header,
    /// Keywords, upper-case, `$` kept; for duplicates the first value. Includes supplemental TEXT.
    pub text: BTreeMap<String, String>,
    /// Keyword/value pairs exactly as in the primary TEXT, in file order.
    pub raw_text: Vec<RawPair>,
    /// Parameters in `$P1..$Pn` order.
    pub channels: Vec<Channel>,
    /// `$DATATYPE`.
    pub datatype: DataType,
    /// `$BYTEORD`.
    pub byte_order: ByteOrder,
    /// Events x parameters.
    pub data: Data,
    /// Raw ANALYSIS segment, if present and valid.
    pub analysis: Option<Vec<u8>>,
    /// Spillover matrix, if present and well formed.
    pub spillover: Option<Spillover>,
    /// What lenient mode repaired or ignored (empty for a clean file, and always empty in strict mode).
    pub warnings: Vec<FcsWarning>,
}

impl FcsFile {
    /// Number of events.
    pub fn n_events(&self) -> usize {
        self.data.shape().0
    }

    /// `$PnN` names.
    pub fn names(&self) -> Vec<&str> {
        self.channels.iter().map(|c| c.name.as_str()).collect()
    }

    /// Keyword value (case-insensitive; give it with or without the `$`).
    pub fn keyword(&self, key: &str) -> Option<&str> {
        let k = key.to_ascii_uppercase();
        self.text
            .get(&k)
            .or_else(|| self.text.get(&format!("${k}")))
            .map(String::as_str)
    }
}

/// Reading options.
#[derive(Debug, Clone, Default)]
pub struct ReadOptions {
    /// Reject anything the specification forbids instead of repairing it.
    pub strict: bool,
}

impl ReadOptions {
    /// Strict mode.
    pub fn strict() -> Self {
        Self { strict: true }
    }

    /// Lenient mode (the default).
    pub fn lenient() -> Self {
        Self { strict: false }
    }
}

/// Upper bound on datasets per file (guards against $NEXTDATA cycles in corrupt files).
const MAX_DATASETS: usize = 10_000;

/// Read every dataset. In lenient mode an error in a later dataset ends the list and becomes a warning on
/// the first dataset; the first dataset itself must be readable.
pub fn read_all(bytes: &[u8], opts: &ReadOptions) -> Result<Vec<FcsFile>, FcsError> {
    let mut out: Vec<FcsFile> = Vec::new();
    let mut base = 0u64;
    loop {
        let idx = out.len();
        match parse::parse_dataset(bytes, base, opts) {
            Ok(p) => {
                out.push(p.file);
                match p.next {
                    Some(n) if n > base && out.len() < MAX_DATASETS => base = n,
                    Some(n) => {
                        let msg = format!(
                            "$NEXTDATA of dataset {idx} points to byte {n}, not after the dataset start {base}"
                        );
                        if opts.strict {
                            return Err(FcsError::InDataset {
                                index: idx,
                                offset: base,
                                source: Box::new(FcsError::Keyword {
                                    keyword: "$NEXTDATA".into(),
                                    msg,
                                }),
                            });
                        }
                        push_first(&mut out, WarningKind::NextData, msg, Some(base));
                        break;
                    }
                    None => break,
                }
            }
            Err(e) if idx == 0 => return Err(e),
            Err(e) => {
                if opts.strict {
                    return Err(FcsError::InDataset {
                        index: idx,
                        offset: base,
                        source: Box::new(e),
                    });
                }
                push_first(
                    &mut out,
                    WarningKind::NextData,
                    format!(
                        "dataset {idx} at byte {base} unreadable ({e}); later datasets ignored"
                    ),
                    Some(base),
                );
                break;
            }
        }
    }
    Ok(out)
}

fn push_first(out: &mut [FcsFile], kind: WarningKind, message: String, offset: Option<u64>) {
    if let Some(f) = out.first_mut() {
        f.warnings.push(FcsWarning {
            kind,
            message,
            offset,
            keyword: Some("$NEXTDATA".into()),
        });
    }
}

/// Read the first dataset only (later datasets are not parsed).
pub fn read(bytes: &[u8], opts: &ReadOptions) -> Result<FcsFile, FcsError> {
    parse::parse_dataset(bytes, 0, opts).map(|p| p.file)
}

/// A file opened for reading: memory-mapped where possible, otherwise read into memory.
pub struct Source {
    inner: SourceInner,
}

enum SourceInner {
    Mmap(memmap2::Mmap),
    Bytes(Vec<u8>),
}

impl Source {
    /// Open `path`. Empty files and files that cannot be mapped are read with `std::fs::read`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, FcsError> {
        let path = path.as_ref();
        let io = |e: std::io::Error| FcsError::Io(format!("{}: {e}", path.display()));
        let f = std::fs::File::open(path).map_err(io)?;
        let len = f.metadata().map_err(io)?.len();
        if len > 0 {
            // SAFETY: the map is read-only and private to this process. If another process truncates the
            // file while it is mapped, reads may fault (SIGBUS); that is the accepted risk of every mmap
            // reader and is documented in docs/fcs_notes.md.
            #[allow(unsafe_code)]
            let m = unsafe { memmap2::Mmap::map(&f) };
            if let Ok(m) = m {
                return Ok(Self {
                    inner: SourceInner::Mmap(m),
                });
            }
        }
        Ok(Self {
            inner: SourceInner::Bytes(std::fs::read(path).map_err(io)?),
        })
    }

    /// The file contents.
    pub fn bytes(&self) -> &[u8] {
        match &self.inner {
            SourceInner::Mmap(m) => m,
            SourceInner::Bytes(b) => b,
        }
    }
}

/// Memory-map `path` and read every dataset.
pub fn read_path(path: impl AsRef<Path>, opts: &ReadOptions) -> Result<Vec<FcsFile>, FcsError> {
    let src = Source::open(path)?;
    read_all(src.bytes(), opts)
}

/// Read many files in parallel on the current Rayon pool. The result order matches `paths`.
pub fn read_many<P: AsRef<Path> + Sync>(
    paths: &[P],
    opts: &ReadOptions,
) -> Vec<Result<Vec<FcsFile>, FcsError>> {
    paths.par_iter().map(|p| read_path(p, opts)).collect()
}
