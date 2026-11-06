//! write -> read round trips: data bit-identical, keywords identical, strict reader accepts the writer's
//! output with no warnings.
#![allow(clippy::unwrap_used)]

use cytopk_fcs::{
    ByteOrder, Data, DataRef, DataType, DatasetSpec, ReadOptions, Version, WriteChannel,
    WriteOptions, read, read_all, write_datasets, write_fcs,
};
use ndarray::Array2;
use proptest::prelude::*;

fn value_strategy() -> impl Strategy<Value = String> {
    // Printable ASCII and a few non-ASCII characters; never empty; never starting with the delimiter.
    proptest::string::string_regex("[A-Za-z0-9 _.,:;=+()|/\\\\é\u{3bc}-]{1,24}")
        .unwrap()
        .prop_filter("non-empty, not starting with a delimiter", |s| {
            !s.is_empty() && !s.starts_with(['/', '|', '\\'])
        })
}

fn opts_strategy() -> impl Strategy<Value = WriteOptions> {
    (
        prop_oneof![Just(Version::Fcs3_0), Just(Version::Fcs3_1)],
        prop_oneof![Just(DataType::F), Just(DataType::D), Just(DataType::I)],
        prop_oneof![Just(ByteOrder::Little), Just(ByteOrder::Big)],
        prop_oneof![
            Just(b'/'),
            Just(b'|'),
            Just(b'\\'),
            Just(0x0cu8),
            Just(b'*')
        ],
    )
        .prop_map(|(version, datatype, byte_order, delimiter)| WriteOptions {
            version,
            datatype,
            byte_order,
            delimiter,
        })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

    #[test]
    fn roundtrip_random_files(
        p in 1usize..=40,
        n in 0usize..=600,
        opts in opts_strategy(),
        seed in any::<u64>(),
        kws in proptest::collection::vec(("[A-Z][A-Z0-9_]{0,10}", value_strategy()), 0..8),
        bits_choice in proptest::collection::vec(prop_oneof![Just(8u32), Just(16), Just(32), Just(64)], 40),
    ) {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let channels: Vec<WriteChannel> = (0..p).map(|j| WriteChannel {
            name: format!("CH{j}-A"),
            long_name: (j % 3 == 0).then(|| format!("marker {j}")),
            range: None,
            bits: Some(bits_choice[j]),
        }).collect();
        // Extra keywords: unique, not reserved (prefixing with X avoids $-keywords the writer sets).
        let mut extra: Vec<(String, String)> = Vec::new();
        for (k, v) in kws {
            let k = format!("X{k}");
            if !extra.iter().any(|(e, _)| e.eq_ignore_ascii_case(&k)) {
                extra.push((k, v));
            }
        }
        let d = opts.delimiter as char;
        prop_assume!(extra.iter().all(|(_, v)| !v.starts_with(d)));

        let bytes;
        let expect: Data;
        match opts.datatype {
            DataType::F => {
                // Arbitrary bit patterns, NaNs included: the round trip must preserve every bit.
                let a = Array2::from_shape_fn((n, p), |_| f32::from_bits(rng.random::<u32>()));
                bytes = write_fcs(DataRef::F32(a.view()), &channels, &extra, &opts).unwrap();
                expect = Data::F32(a);
            }
            DataType::D => {
                let a = Array2::from_shape_fn((n, p), |_| f64::from_bits(rng.random::<u64>()));
                bytes = write_fcs(DataRef::F64(a.view()), &channels, &extra, &opts).unwrap();
                expect = Data::F64(a);
            }
            DataType::I => {
                let a = Array2::from_shape_fn((n, p), |(_, j)| {
                    let b = bits_choice[j].min(53); // f64 is exact up to 2^53
                    (rng.random::<u64>() >> (64 - b)) as f64
                });
                bytes = write_fcs(DataRef::F64(a.view()), &channels, &extra, &opts).unwrap();
                expect = Data::F64(a);
            }
        }
        let f = read(&bytes, &ReadOptions::strict()).unwrap();
        prop_assert!(f.warnings.is_empty());
        prop_assert_eq!(f.n_events(), n);
        prop_assert_eq!(f.channels.len(), p);
        match (&f.data, &expect) {
            (Data::F32(a), Data::F32(b)) => prop_assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())),
            (Data::F64(a), Data::F64(b)) => prop_assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())),
            _ => prop_assert!(false, "data type changed"),
        }
        for (j, c) in channels.iter().enumerate() {
            prop_assert_eq!(&f.channels[j].name, &c.name);
            prop_assert_eq!(&f.channels[j].long_name, &c.long_name);
        }
        for (k, v) in &extra {
            prop_assert_eq!(f.keyword(k), Some(v.as_str()));
        }
        prop_assert_eq!(f.header.version, opts.version);
        // Lenient mode must agree with strict mode on a conforming file.
        let l = read(&bytes, &ReadOptions::lenient()).unwrap();
        prop_assert!(l.warnings.is_empty());
        prop_assert_eq!(&l.text, &f.text);
    }

    #[test]
    fn roundtrip_multiple_datasets(sizes in proptest::collection::vec((1usize..6, 0usize..50), 1..5)) {
        let arrays: Vec<Array2<f32>> = sizes.iter().enumerate()
            .map(|(k, &(p, n))| Array2::from_shape_fn((n, p), |(i, j)| (k * 1000 + i * 10 + j) as f32))
            .collect();
        let chans: Vec<Vec<WriteChannel>> = sizes.iter()
            .map(|&(p, _)| (0..p).map(|j| WriteChannel::named(format!("P{j}"))).collect())
            .collect();
        let specs: Vec<DatasetSpec<'_>> = arrays.iter().zip(&chans)
            .map(|(a, c)| DatasetSpec { data: DataRef::F32(a.view()), channels: c, keywords: &[] })
            .collect();
        let bytes = write_datasets(&specs, &WriteOptions::default()).unwrap();
        let all = read_all(&bytes, &ReadOptions::strict()).unwrap();
        prop_assert_eq!(all.len(), arrays.len());
        for (f, a) in all.iter().zip(&arrays) {
            prop_assert_eq!(&f.data, &Data::F32(a.clone()));
        }
    }
}

#[test]
fn large_event_counts_roundtrip() {
    // 10^5 events x 40 channels (the brief's upper end for the property test, run once).
    let (n, p) = (100_000, 40);
    let a = Array2::from_shape_fn((n, p), |(i, j)| (i as f32) * 0.5 + j as f32);
    let ch: Vec<_> = (0..p)
        .map(|j| WriteChannel::named(format!("C{j}")))
        .collect();
    let b = write_fcs(DataRef::F32(a.view()), &ch, &[], &WriteOptions::default()).unwrap();
    let f = read(&b, &ReadOptions::strict()).unwrap();
    assert_eq!(f.data, Data::F32(a));
}

#[test]
fn data_beyond_header_limit_uses_keyword_offsets() {
    // > 99,999,999 bytes of DATA: the HEADER DATA fields must be 0 and $BEGINDATA/$ENDDATA used.
    let (n, p) = (3_200_000, 8); // 102.4 MB
    let a = Array2::from_shape_fn((n, p), |(i, j)| (i ^ j) as f32);
    let ch: Vec<_> = (0..p)
        .map(|j| WriteChannel::named(format!("C{j}")))
        .collect();
    let b = write_fcs(DataRef::F32(a.view()), &ch, &[], &WriteOptions::default()).unwrap();
    assert_eq!(&b[26..42], b"       0       0");
    let f = read(&b, &ReadOptions::strict()).unwrap();
    assert!(f.header.data.is_zero());
    assert_eq!(f.data, Data::F32(a));
}

#[test]
fn writer_rejects_bad_input() {
    let a = Array2::<f64>::from_elem((2, 1), 1.5);
    let ch = [WriteChannel::named("A")];
    let o = |dt| WriteOptions {
        datatype: dt,
        ..WriteOptions::default()
    };
    assert!(write_fcs(DataRef::F64(a.view()), &ch, &[], &o(DataType::I)).is_err()); // not an integer
    assert!(write_fcs(DataRef::F64(a.view()), &[], &[], &o(DataType::F)).is_err()); // channel count
    let kw = [("$TOT".to_owned(), "5".to_owned())];
    assert!(write_fcs(DataRef::F64(a.view()), &ch, &kw, &o(DataType::F)).is_err()); // reserved
    let kw = [("$COM".to_owned(), String::new())];
    assert!(write_fcs(DataRef::F64(a.view()), &ch, &kw, &o(DataType::F)).is_err()); // empty value
    let kw = [("$COM".to_owned(), "/x".to_owned())];
    assert!(write_fcs(DataRef::F64(a.view()), &ch, &kw, &o(DataType::F)).is_err()); // leading delimiter
}
