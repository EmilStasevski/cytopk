//! Known real-world deviations from the specification: strict mode must reject each one, and lenient mode
//! must read the file, return the right data, and report the matching warning kind.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;
use common::{DATA_AT, f32_file, kv, replace};
use cytopk_fcs::{ByteOrder, Data, DataType, FcsError, ReadOptions, WarningKind, read, read_all};

fn strict() -> ReadOptions {
    ReadOptions::strict()
}
fn lenient() -> ReadOptions {
    ReadOptions::lenient()
}

fn kinds(f: &cytopk_fcs::FcsFile) -> Vec<WarningKind> {
    f.warnings.iter().map(|w| w.kind).collect()
}

fn f32_data(f: &cytopk_fcs::FcsFile) -> Vec<f32> {
    match &f.data {
        Data::F32(a) => a.iter().copied().collect(),
        Data::F64(_) => panic!("expected f32"),
    }
}

fn expected(n: usize, p: usize) -> Vec<f32> {
    (0..n)
        .flat_map(|i| (0..p).map(move |j| (i * 10 + j) as f32))
        .collect()
}

#[test]
fn clean_file_reads_in_both_modes_without_warnings() {
    let b = f32_file(5, &["FSC-A", "SSC-A", "CD3"]).build();
    for o in [strict(), lenient()] {
        let f = read(&b, &o).unwrap();
        assert!(f.warnings.is_empty(), "{:?}", f.warnings);
        assert_eq!(f.names(), ["FSC-A", "SSC-A", "CD3"]);
        assert_eq!(f32_data(&f), expected(5, 3));
        assert_eq!(f.datatype, DataType::F);
        assert_eq!(f.byte_order, ByteOrder::Little);
    }
}

#[test]
fn empty_keyword_value() {
    let mut r = f32_file(4, &["A", "B"]);
    r.pairs.insert(0, kv("$P1S", "PLACEHOLDER"));
    let t = replace(&r.text(), b"/$P1S/PLACEHOLDER/", b"/$P1S//");
    let b = r.build_with_text(&t);
    assert!(read(&b, &strict()).is_err());
    let f = read(&b, &lenient()).unwrap();
    assert_eq!(f.keyword("$P1S"), Some(""));
    assert_eq!(f.channels[0].long_name, None);
    assert!(kinds(&f).contains(&WarningKind::EmptyValue));
    assert_eq!(f32_data(&f), expected(4, 2));
}

#[test]
fn several_consecutive_empty_values() {
    let mut r = f32_file(2, &["A"]);
    r.pairs.insert(0, kv("$FIL", "X"));
    r.pairs.insert(0, kv("$CYT", "X"));
    let t = replace(&r.text(), b"/$CYT/X/$FIL/X/", b"/$CYT//$FIL//");
    let f = read(&r.build_with_text(&t), &lenient()).unwrap();
    assert_eq!(f.keyword("$CYT"), Some(""));
    assert_eq!(f.keyword("$FIL"), Some(""));
    assert_eq!(f32_data(&f), expected(2, 1));
}

#[test]
fn escaped_delimiter_in_value_is_kept_in_both_modes() {
    let mut r = f32_file(2, &["A"]);
    r.pairs.push(kv("$DATE", "01/02/2020"));
    r.pairs.push(kv("$FIL", "a/b//c"));
    let b = r.build();
    for o in [strict(), lenient()] {
        let f = read(&b, &o).unwrap();
        assert_eq!(f.keyword("$DATE"), Some("01/02/2020"));
        assert_eq!(f.keyword("$FIL"), Some("a/b//c"));
        assert!(f.warnings.is_empty());
    }
}

#[test]
fn unusual_delimiters() {
    for d in [b'|', b'\\', 0x0c, b'*', b' '] {
        let mut r = f32_file(3, &["A", "B"]);
        r.delim = d;
        r.pairs.push(kv("$COM", &format!("x{}y", d as char)));
        let f = read(&r.build(), &strict()).unwrap();
        assert_eq!(f.keyword("$COM"), Some(format!("x{}y", d as char).as_str()));
        assert_eq!(f32_data(&f), expected(3, 2));
    }
}

#[test]
fn missing_final_delimiter() {
    let r = f32_file(3, &["A"]);
    let mut t = r.text();
    t.pop();
    let b = r.build_with_text(&t);
    assert!(read(&b, &strict()).is_err());
    let f = read(&b, &lenient()).unwrap();
    assert!(kinds(&f).contains(&WarningKind::Delimiter));
    assert_eq!(f32_data(&f), expected(3, 1));
}

#[test]
fn doubled_final_delimiter() {
    let r = f32_file(3, &["A"]);
    let mut t = r.text();
    t.push(b'/');
    let b = r.build_with_text(&t);
    assert!(read(&b, &strict()).is_err());
    let f = read(&b, &lenient()).unwrap();
    assert!(kinds(&f).contains(&WarningKind::Delimiter));
    assert_eq!(
        f.keyword("$ENDDATA").map(str::to_owned),
        Some((DATA_AT + 11).to_string())
    );
}

#[test]
fn trailing_bytes_after_last_delimiter() {
    let mut r = f32_file(3, &["A"]);
    r.text_tail = b"   \0\0junk".to_vec();
    let b = r.build();
    assert!(read(&b, &strict()).is_err());
    let f = read(&b, &lenient()).unwrap();
    assert!(kinds(&f).contains(&WarningKind::Delimiter));
    assert_eq!(f32_data(&f), expected(3, 1));
}

#[test]
fn keywords_are_case_insensitive_and_trimmed() {
    let mut r = f32_file(2, &["A"]);
    r.pairs
        .iter_mut()
        .for_each(|p| p.0 = p.0.to_ascii_lowercase());
    r.pairs.push(kv(" $COM ", "hello"));
    let b = r.build();
    assert!(read(&b, &strict()).is_err()); // spaces around a keyword
    let f = read(&b, &lenient()).unwrap();
    assert_eq!(f.keyword("$com"), Some("hello"));
    assert_eq!(f32_data(&f), expected(2, 1));
    assert_eq!(f.raw_text[0].key, "$byteord");
}

#[test]
fn duplicate_keyword_keeps_first() {
    let mut r = f32_file(2, &["A"]);
    r.pairs.push(kv("$COM", "first"));
    r.pairs.push(kv("$COM", "second"));
    let b = r.build();
    assert!(read(&b, &strict()).is_err());
    let f = read(&b, &lenient()).unwrap();
    assert_eq!(f.keyword("$COM"), Some("first"));
    assert!(kinds(&f).contains(&WarningKind::DuplicateKeyword));
}

#[test]
fn latin1_text() {
    let r = f32_file(2, &["A"]);
    let mut t = r.text();
    t.extend_from_slice(b"$COM/caf\xe9/");
    let b = r.build_with_text(&t);
    assert!(read(&b, &strict()).is_err());
    let f = read(&b, &lenient()).unwrap();
    assert_eq!(f.keyword("$COM"), Some("café"));
    assert!(kinds(&f).contains(&WarningKind::Encoding));
}

#[test]
fn header_data_offsets_zero_uses_keywords() {
    let mut r = f32_file(6, &["A", "B"]);
    r.header_data = Some((0, 0));
    for o in [strict(), lenient()] {
        let f = read(&r.build(), &o).unwrap();
        assert_eq!(f32_data(&f), expected(6, 2));
        assert!(f.warnings.is_empty());
    }
}

#[test]
fn off_by_one_enddata() {
    let mut r = f32_file(6, &["A", "B"]);
    let len = 6 * 2 * 4;
    r.data.push(0xAB); // one byte more than $TOT x 8
    r.set("$ENDDATA", &(DATA_AT + len).to_string());
    r.header_data = Some((DATA_AT, DATA_AT + len));
    let b = r.build();
    assert!(matches!(read(&b, &strict()), Err(FcsError::Data(_))));
    let f = read(&b, &lenient()).unwrap();
    assert!(kinds(&f).contains(&WarningKind::DataSize));
    assert_eq!(f32_data(&f), expected(6, 2));
}

#[test]
fn header_and_keyword_data_offsets_disagree() {
    let mut r = f32_file(6, &["A", "B"]);
    r.header_data = Some((DATA_AT + 8, DATA_AT + 8 + 47)); // wrong: shifted by one event
    // HEADER range still fits in the file and has the right size: keyword offsets must win only if HEADER
    // is implausible, so make the HEADER range run past the file end.
    r.header_data = Some((DATA_AT + 8, DATA_AT + 8 + 47 + 1000));
    let b = r.build();
    assert!(read(&b, &strict()).is_err());
    let f = read(&b, &lenient()).unwrap();
    assert!(kinds(&f).contains(&WarningKind::DataOffsets));
    assert_eq!(f32_data(&f), expected(6, 2));
}

#[test]
fn tot_too_large_reads_available_events() {
    let mut r = f32_file(6, &["A", "B"]);
    r.set("$TOT", "10");
    let b = r.build();
    assert!(read(&b, &strict()).is_err());
    let f = read(&b, &lenient()).unwrap();
    assert_eq!(f.n_events(), 6);
    assert!(kinds(&f).contains(&WarningKind::DataSize));
}

#[test]
fn tot_too_small_trusts_tot() {
    let mut r = f32_file(6, &["A", "B"]);
    r.set("$TOT", "4");
    let b = r.build();
    assert!(read(&b, &strict()).is_err());
    let f = read(&b, &lenient()).unwrap();
    assert_eq!(f32_data(&f), expected(4, 2));
}

#[test]
fn truncated_data() {
    let r = f32_file(6, &["A", "B"]);
    let mut b = r.build();
    b.truncate(b.len() - 13); // lose 1.6 events
    assert!(read(&b, &strict()).is_err());
    let f = read(&b, &lenient()).unwrap();
    assert_eq!(f32_data(&f), expected(4, 2));
    assert!(kinds(&f).contains(&WarningKind::DataOffsets));
}

#[test]
fn truncated_text() {
    let r = f32_file(3, &["A", "B"]);
    let b = r.build();
    let cut = b.len().min(58 + 40);
    let t = &b[..cut];
    for o in [strict(), lenient()] {
        assert!(read(t, &o).is_err(), "a file cut inside TEXT has no DATA");
    }
}

#[test]
fn missing_pnn_and_pne() {
    let mut r = f32_file(2, &["A", "B"]);
    r.remove("$P2N");
    r.remove("$P1E");
    let b = r.build();
    assert!(read(&b, &strict()).is_err());
    let f = read(&b, &lenient()).unwrap();
    assert_eq!(f.names(), ["A", "P2"]);
    assert_eq!(f.channels[0].log, (0.0, 0.0));
    assert!(kinds(&f).contains(&WarningKind::MissingKeyword));
}

#[test]
fn fcs2_and_fcs32_versions() {
    for v in ["FCS2.0", "FCS3.2"] {
        let mut r = f32_file(2, &["A"]);
        r.version = v;
        let b = r.build();
        assert!(matches!(read(&b, &strict()), Err(FcsError::Version { .. })));
        let f = read(&b, &lenient()).unwrap();
        assert_eq!(f32_data(&f), expected(2, 1));
        assert!(kinds(&f).contains(&WarningKind::Header));
    }
    let mut r = f32_file(2, &["A"]);
    r.version = "FCS4.0";
    assert!(matches!(
        read(&r.build(), &lenient()),
        Err(FcsError::Version { .. })
    ));
}

#[test]
fn spillover_good_and_malformed() {
    let mut r = f32_file(2, &["FL1", "FL2", "FSC"]);
    r.pairs.push(kv("$SPILLOVER", "2,FL1,FL2,1,0.1,0.05,1"));
    let f = read(&r.build(), &strict()).unwrap();
    let s = f.spillover.as_ref().unwrap();
    assert_eq!(s.channels, ["FL1", "FL2"]);
    assert_eq!(s.matrix[[0, 1]], 0.1);
    assert_eq!(s.matrix[[1, 0]], 0.05);

    for bad in [
        "2,FL1,FL2,1,0.1,0.05", // too few values
        "3,FL1,FL2,1,0,0,1",    // count disagrees
        "2,FL1,NOPE,1,0,0,1",   // unknown channel
        "2,FL1,FL2,1,x,0,1",    // non-numeric value
        "2,1,0,0,1,0,0,1,0",    // no names (old $COMP style)
        "",                     // empty
    ] {
        let mut r = f32_file(2, &["FL1", "FL2", "FSC"]);
        r.pairs
            .push(kv("$SPILLOVER", if bad.is_empty() { "PLACE" } else { bad }));
        let mut b = r.build();
        if bad.is_empty() {
            b = r.build_with_text(&replace(&r.text(), b"/$SPILLOVER/PLACE/", b"/$SPILLOVER//"));
        }
        assert!(read(&b, &strict()).is_err(), "{bad}");
        let f = read(&b, &lenient()).unwrap();
        assert!(f.spillover.is_none());
        assert_eq!(f32_data(&f), expected(2, 3), "{bad}");
        if !bad.is_empty() {
            assert!(kinds(&f).contains(&WarningKind::Spillover), "{bad}");
        }
    }
}

#[test]
fn integer_data_mixed_widths_big_endian_and_masking() {
    // $P1B 16 with $P1R 1024 (mask 0x3FF), $P2B 32 with $P2R 2^32, $P3B 8.
    let mut r = f32_file(0, &["A", "B", "C"]);
    r.set("$DATATYPE", "I");
    r.set("$BYTEORD", "4,3,2,1");
    r.set("$P1B", "16");
    r.set("$P1R", "1024");
    r.set("$P2R", "4294967296");
    r.set("$P3B", "8");
    r.set("$P3R", "256");
    let rows: [(u16, u32, u8); 3] = [(5, 70_000, 7), (0xFFFF, u32::MAX, 255), (1023, 0, 0)];
    let mut data = Vec::new();
    for (a, b, c) in rows {
        data.extend_from_slice(&a.to_be_bytes());
        data.extend_from_slice(&b.to_be_bytes());
        data.push(c);
    }
    r.data = data;
    r.set("$TOT", "3");
    r.set("$ENDDATA", &(DATA_AT + 21 - 1).to_string());
    for o in [strict(), lenient()] {
        let f = read(&r.build(), &o).unwrap();
        let Data::F64(a) = &f.data else { panic!() };
        assert_eq!(a.row(0).to_vec(), [5.0, 70_000.0, 7.0]);
        assert_eq!(a.row(1).to_vec(), [1023.0, u32::MAX as f64, 255.0]); // 0xFFFF masked by 0x3FF
        assert_eq!(a.row(2).to_vec(), [1023.0, 0.0, 0.0]);
        assert_eq!(f.byte_order, ByteOrder::Big);
    }
}

#[test]
fn unsupported_features_are_errors_in_both_modes() {
    for (k, v) in [("$DATATYPE", "A"), ("$MODE", "H"), ("$BYTEORD", "3,4,1,2")] {
        let mut r = f32_file(2, &["A"]);
        r.set(k, v);
        for o in [strict(), lenient()] {
            assert!(
                matches!(read(&r.build(), &o), Err(FcsError::Unsupported(_))),
                "{k}={v}"
            );
        }
    }
}

#[test]
fn absurd_par_and_tot_do_not_allocate() {
    let mut r = f32_file(2, &["A"]);
    r.set("$PAR", "4000000000");
    assert!(read(&r.build(), &lenient()).is_err());
    let mut r = f32_file(2, &["A"]);
    r.set("$TOT", "1000000000000");
    let f = read(&r.build(), &lenient()).unwrap();
    assert_eq!(f.n_events(), 2);
}

#[test]
fn nextdata_chain_and_bad_nextdata() {
    use cytopk_fcs::{DataRef, DatasetSpec, WriteChannel, WriteOptions, write_datasets};
    use ndarray::Array2;
    let a = Array2::from_shape_fn((3, 2), |(i, j)| (i + j) as f32);
    let b = Array2::from_shape_fn((5, 1), |(i, _)| i as f32 * 2.0);
    let ca = [WriteChannel::named("X"), WriteChannel::named("Y")];
    let cb = [WriteChannel::named("Z")];
    let bytes = write_datasets(
        &[
            DatasetSpec {
                data: DataRef::F32(a.view()),
                channels: &ca,
                keywords: &[],
            },
            DatasetSpec {
                data: DataRef::F32(b.view()),
                channels: &cb,
                keywords: &[],
            },
        ],
        &WriteOptions::default(),
    )
    .unwrap();
    let all = read_all(&bytes, &strict()).unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[1].names(), ["Z"]);
    assert_eq!(all[1].data, Data::F32(b.clone()));
    assert!(all[1].offset > 0);

    // $NEXTDATA pointing past the end: strict error, lenient keeps dataset 0 with a warning.
    let first_len = all[1].offset as usize;
    let mut broken = bytes[..first_len].to_vec();
    broken.extend_from_slice(
        b"garbage that is not a header at all, long enough to be read as one....",
    );
    assert!(read_all(&broken, &strict()).is_err());
    let l = read_all(&broken, &lenient()).unwrap();
    assert_eq!(l.len(), 1);
    assert!(kinds(&l[0]).contains(&WarningKind::NextData));
}

#[test]
fn header_data_start_inside_text_uses_keyword_offsets() {
    // FlowIO's data_start_offset_discrepancy_example.fcs: the HEADER DATA start points into TEXT. Clipping
    // TEXT there would drop $BEGINDATA/$ENDDATA; the keyword offsets are the right ones.
    let mut r = f32_file(6, &["A", "B"]);
    r.pairs.push(kv("$COM", &"x".repeat(200)));
    let text_len = r.text().len() as u64;
    r.header_data = Some((58 + text_len - 100, DATA_AT + 47));
    let b = r.build();
    assert!(read(&b, &strict()).is_err());
    let f = read(&b, &lenient()).unwrap();
    assert_eq!(f32_data(&f), expected(6, 2));
    assert_eq!(f.keyword("$COM").map(str::len), Some(200));
    assert!(kinds(&f).contains(&WarningKind::DataOffsets));
}

#[test]
fn integer_24_bit_channels() {
    let mut r = f32_file(0, &["A", "B"]);
    r.set("$DATATYPE", "I");
    r.set("$P1B", "24");
    r.set("$P1R", "16777216");
    r.set("$P2B", "16");
    r.set("$P2R", "65536");
    let mut data = Vec::new();
    for (a, b) in [(0x00AB_CDEFu32, 7u16), (16_777_215, 65_535)] {
        data.extend_from_slice(&a.to_le_bytes()[..3]);
        data.extend_from_slice(&b.to_le_bytes());
    }
    r.data = data;
    r.set("$TOT", "2");
    r.set("$ENDDATA", &(DATA_AT + 10 - 1).to_string());
    let f = read(&r.build(), &strict()).unwrap();
    let Data::F64(a) = &f.data else { panic!() };
    assert_eq!(a.row(0).to_vec(), [f64::from(0x00AB_CDEF), 7.0]);
    assert_eq!(a.row(1).to_vec(), [16_777_215.0, 65_535.0]);
}

#[test]
fn missing_nextdata_is_strict_error_lenient_warning() {
    let mut r = f32_file(2, &["A"]);
    r.remove("$NEXTDATA");
    let b = r.build();
    assert!(read(&b, &strict()).is_err());
    let f = read(&b, &lenient()).unwrap();
    assert!(kinds(&f).contains(&WarningKind::MissingKeyword));
    assert_eq!(f32_data(&f), expected(2, 1));
}
