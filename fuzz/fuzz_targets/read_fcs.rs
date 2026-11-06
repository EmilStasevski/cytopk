//! Any byte string: reading in strict and lenient mode returns Ok or Err and never panics.
#![no_main]
use cytopk_fcs::{ReadOptions, read_all};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = read_all(data, &ReadOptions::strict());
    let _ = read_all(data, &ReadOptions::lenient());
});
