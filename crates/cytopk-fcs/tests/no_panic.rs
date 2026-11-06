//! Adversarial inputs: random byte flips, insertions, deletions, truncations, overwritten HEADER fields and
//! keyword values, plus fully random bytes. Reading must return Ok or Err, never panic, in both modes.
//! The cargo-fuzz target in `fuzz/` runs the same entry point for much longer.
#![allow(clippy::unwrap_used)]

use cytopk_fcs::{
    DataRef, DataType, DatasetSpec, ReadOptions, WriteChannel, WriteOptions, read_all,
    write_datasets,
};
use ndarray::Array2;
use proptest::prelude::*;

fn seeds() -> Vec<Vec<u8>> {
    let a = Array2::from_shape_fn((7, 3), |(i, j)| (i * 3 + j) as f32);
    let ch = [
        WriteChannel {
            name: "FL1".into(),
            long_name: Some("CD3".into()),
            range: None,
            bits: Some(16),
        },
        WriteChannel {
            name: "FL2".into(),
            long_name: None,
            range: Some(1024.0),
            bits: Some(32),
        },
        WriteChannel {
            name: "FSC".into(),
            long_name: None,
            range: None,
            bits: Some(8),
        },
    ];
    let kw = vec![
        ("$SPILLOVER".to_owned(), "2,FL1,FL2,1,0.1,0.2,1".to_owned()),
        ("$DATE".to_owned(), "01/02/2020".to_owned()),
    ];
    let mut out = Vec::new();
    for dt in [DataType::F, DataType::D, DataType::I] {
        let spec = DatasetSpec {
            data: DataRef::F32(a.view()),
            channels: &ch,
            keywords: &kw,
        };
        out.push(
            write_datasets(
                &[spec, spec],
                &WriteOptions {
                    datatype: dt,
                    ..WriteOptions::default()
                },
            )
            .unwrap(),
        );
    }
    out
}

fn both_modes(b: &[u8]) {
    let _ = read_all(b, &ReadOptions::strict());
    let _ = read_all(b, &ReadOptions::lenient());
}

#[derive(Debug, Clone)]
enum Edit {
    Flip(usize, u8),
    Insert(usize, Vec<u8>),
    Delete(usize, usize),
    Truncate(usize),
    Digits(usize, u64),
}

fn edit() -> impl Strategy<Value = Edit> {
    prop_oneof![
        (any::<usize>(), any::<u8>()).prop_map(|(i, b)| Edit::Flip(i, b)),
        (
            any::<usize>(),
            proptest::collection::vec(
                prop_oneof![Just(b'/'), Just(b'$'), Just(b'0'), Just(b' '), any::<u8>()],
                1..6
            )
        )
            .prop_map(|(i, v)| Edit::Insert(i, v)),
        (any::<usize>(), 1usize..16).prop_map(|(i, n)| Edit::Delete(i, n)),
        any::<usize>().prop_map(Edit::Truncate),
        // Overwrite a run of bytes with a decimal number: hits HEADER offsets and numeric keywords.
        (
            any::<usize>(),
            prop_oneof![
                Just(0u64),
                Just(1),
                Just(u64::MAX),
                Just(99_999_999),
                any::<u64>()
            ]
        )
            .prop_map(|(i, v)| Edit::Digits(i, v)),
    ]
}

fn apply(b: &mut Vec<u8>, e: &Edit) {
    if b.is_empty() {
        return;
    }
    match e {
        Edit::Flip(i, x) => {
            let k = i % b.len();
            b[k] ^= x | 1;
        }
        Edit::Insert(i, v) => {
            let k = i % (b.len() + 1);
            b.splice(k..k, v.iter().copied());
        }
        Edit::Delete(i, n) => {
            let k = i % b.len();
            let end = (k + n).min(b.len());
            b.drain(k..end);
        }
        Edit::Truncate(i) => b.truncate(i % (b.len() + 1)),
        Edit::Digits(i, v) => {
            let s = v.to_string();
            let k = i % b.len();
            for (o, c) in s.bytes().enumerate() {
                if k + o < b.len() {
                    b[k + o] = c;
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 4000, ..ProptestConfig::default() })]

    #[test]
    fn mutated_files_never_panic(which in 0usize..3, edits in proptest::collection::vec(edit(), 1..6)) {
        let mut b = seeds()[which].clone();
        for e in &edits {
            apply(&mut b, e);
        }
        both_modes(&b);
    }

    #[test]
    fn random_bytes_never_panic(mut b in proptest::collection::vec(any::<u8>(), 0..400), v in 0usize..3) {
        if b.len() >= 6 {
            b[..6].copy_from_slice([b"FCS3.1", b"FCS3.0", b"FCS2.0"][v]);
        }
        both_modes(&b);
    }
}

#[test]
fn every_truncation_of_a_seed_never_panics() {
    for s in seeds() {
        for n in 0..=s.len() {
            both_modes(&s[..n]);
        }
    }
}

#[test]
fn every_single_byte_overwrite_in_header_and_text_never_panics() {
    for s in seeds() {
        let text_end = 600.min(s.len());
        for i in 0..text_end {
            for x in [0u8, b' ', b'/', b'9', 0xff] {
                let mut b = s.clone();
                b[i] = x;
                both_modes(&b);
            }
        }
    }
}
