//! Errors (reading stops) and warnings (lenient mode repaired or ignored something and kept going).

use std::fmt;
use thiserror::Error;

/// A reason the file could not be read. Messages carry byte offsets and keyword names where they exist.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum FcsError {
    /// The file could not be opened or read.
    #[error("I/O error: {0}")]
    Io(String),
    /// Fewer than the 58 bytes a HEADER needs.
    #[error("file too short for an FCS HEADER: {have} bytes at offset {offset}, need 58")]
    TooShort {
        /// Absolute offset of the dataset.
        offset: u64,
        /// Bytes available from that offset.
        have: u64,
    },
    /// The first six bytes are not a supported version string.
    #[error(
        "not a supported FCS version: {found:?} at offset {offset} (supported: FCS3.0, FCS3.1; lenient also FCS2.0, FCS3.2)"
    )]
    Version {
        /// What was found (lossy text).
        found: String,
        /// Absolute offset of the dataset.
        offset: u64,
    },
    /// A HEADER field is malformed or points outside the file.
    #[error("HEADER {field} (byte {offset}): {msg}")]
    Header {
        /// Field name, e.g. "TEXT begin".
        field: &'static str,
        /// Absolute byte offset of the field.
        offset: u64,
        /// What is wrong.
        msg: String,
    },
    /// The TEXT segment cannot be tokenised.
    #[error("TEXT segment (byte {offset}): {msg}")]
    Text {
        /// Absolute byte offset where the problem was found.
        offset: u64,
        /// What is wrong.
        msg: String,
    },
    /// A required keyword is missing or has an unusable value.
    #[error("keyword {keyword}: {msg}")]
    Keyword {
        /// Normalised (upper-case) keyword.
        keyword: String,
        /// What is wrong.
        msg: String,
    },
    /// The DATA segment cannot be located or decoded.
    #[error("DATA segment: {0}")]
    Data(String),
    /// Valid FCS, but a feature this reader does not implement (e.g. ASCII data, histogram mode).
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// Error in dataset `index` (counting from 0) of a multi-dataset file.
    #[error("dataset {index} at byte {offset}: {source}")]
    InDataset {
        /// Dataset index.
        index: usize,
        /// Absolute offset of the dataset.
        offset: u64,
        /// The underlying error.
        source: Box<FcsError>,
    },
    /// The writer was given unusable input.
    #[error("write: {0}")]
    Write(String),
}

/// What a lenient-mode warning is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum WarningKind {
    /// Version string FCS2.0/FCS3.2, non-space filler, or a non-numeric HEADER field.
    Header,
    /// Missing, doubled or unexpected delimiters in TEXT; trailing bytes after the last delimiter.
    Delimiter,
    /// Empty keyword value (forbidden by the specification).
    EmptyValue,
    /// The same keyword more than once (the first value is kept).
    DuplicateKeyword,
    /// Text that is not UTF-8; decoded as Latin-1.
    Encoding,
    /// TEXT end beyond the file end.
    TextOffsets,
    /// HEADER and $BEGINDATA/$ENDDATA disagree, or DATA lies partly outside the file.
    DataOffsets,
    /// DATA length differs from $TOT x bytes per event.
    DataSize,
    /// A required keyword is missing and a default was used.
    MissingKeyword,
    /// A keyword value could not be parsed and was ignored or replaced.
    BadValue,
    /// $NEXTDATA is invalid or the next dataset could not be read.
    NextData,
    /// $SPILLOVER / $SPILL / $COMP is malformed and was ignored.
    Spillover,
    /// Supplemental TEXT segment invalid and ignored.
    SupplementalText,
    /// ANALYSIS segment offsets invalid; segment ignored.
    Analysis,
}

/// Something lenient mode repaired or ignored. Strict mode turns the same condition into an [`FcsError`].
#[derive(Debug, Clone, PartialEq)]
pub struct FcsWarning {
    /// Category.
    pub kind: WarningKind,
    /// Human-readable description.
    pub message: String,
    /// Absolute byte offset, if the problem has one.
    pub offset: Option<u64>,
    /// Normalised keyword, if the problem concerns one.
    pub keyword: Option<String>,
}

impl fmt::Display for FcsWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)?;
        if let Some(k) = &self.keyword {
            write!(f, " [{k}]")?;
        }
        if let Some(o) = self.offset {
            write!(f, " (byte {o})")?;
        }
        Ok(())
    }
}

/// Collects warnings in lenient mode, or turns the first problem into an error in strict mode.
pub(crate) struct Diag {
    pub strict: bool,
    pub warnings: Vec<FcsWarning>,
}

impl Diag {
    pub fn new(strict: bool) -> Self {
        Self {
            strict,
            warnings: Vec::new(),
        }
    }

    /// Record `kind`; in strict mode return `err()` instead.
    pub fn problem(
        &mut self,
        kind: WarningKind,
        message: impl Into<String>,
        offset: Option<u64>,
        keyword: Option<&str>,
        err: impl FnOnce(String) -> FcsError,
    ) -> Result<(), FcsError> {
        let message = message.into();
        if self.strict {
            return Err(err(message));
        }
        self.warn(kind, message, offset, keyword);
        Ok(())
    }

    /// Record a warning in either mode (for things the specification allows but are worth reporting).
    pub fn warn(
        &mut self,
        kind: WarningKind,
        message: impl Into<String>,
        offset: Option<u64>,
        keyword: Option<&str>,
    ) {
        self.warnings.push(FcsWarning {
            kind,
            message: message.into(),
            offset,
            keyword: keyword.map(str::to_owned),
        });
    }
}
