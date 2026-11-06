//! cytopk-gate: arcsinh, strategy parsing and validation, and gate statistics against naive evaluation.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use cytopk_gate::{GateError, Strategy, arcsinh, arcsinh_inv, median};
use ndarray::{Array2, array};
use proptest::prelude::*;

const EXAMPLE: &str = r#"
[transform]
cofactor = 5.0
arcsinh = ["CD3"]
[transform.cofactors]
CD69 = 150.0

[stats]
medians = ["CD3", "CD69"]

[[gate]]
name = "cd3"
range = { CD3 = [1.0, inf] }

[[gate]]
name = "activated"
parent = "cd3"
range = { CD69 = [1.0, inf] }
"#;

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn arcsinh_values_and_inverse() {
    assert_eq!(arcsinh(0.0, 5.0), 0.0);
    assert!((arcsinh(5.0, 5.0) - 1.0f64.asinh()).abs() < 1e-15);
    assert!((arcsinh(-150.0, 150.0) + 1.0f64.asinh()).abs() < 1e-15);
    for &x in &[-1e6, -3.0, 0.5, 42.0, 1e9] {
        for &c in &[1.0, 5.0, 150.0] {
            let y = arcsinh(x, c);
            assert!((arcsinh_inv(y, c) - x).abs() <= 1e-9 * x.abs().max(1.0));
        }
    }
}

#[test]
fn parses_example() {
    let s = Strategy::from_toml(EXAMPLE).unwrap();
    assert_eq!(
        s.cofactors,
        vec![("CD3".into(), 5.0), ("CD69".into(), 150.0)]
    );
    assert_eq!(s.gates.len(), 2);
    assert_eq!(s.gates[1].parent, Some(0));
    assert_eq!(s.gates[0].bounds, vec![("CD3".into(), 1.0, f64::INFINITY)]);
    assert_eq!(s.channels(), names(&["CD3", "CD69"]));
}

#[test]
fn hand_computed_counts_fractions_medians() {
    let s = Strategy::from_toml(EXAMPLE).unwrap();
    // CD3 transformed with cofactor 5: asinh(x/5) >= 1  <=>  x >= 5 sinh(1) = 5.876
    // CD69 with cofactor 150: >= 1  <=>  x >= 150 sinh(1) = 176.28
    let d = array![
        [0.0, 1000.0, 7.0],  // not cd3
        [10.0, 100.0, 7.0],  // cd3, not activated
        [20.0, 200.0, 7.0],  // cd3, activated
        [30.0, 300.0, 7.0],  // cd3, activated
        [5.876, 999.0, 7.0], // just below 5 sinh(1) = 5.8760059...: not cd3
    ];
    let out = s.apply(&names(&["CD3", "CD69", "X"]), d.view()).unwrap();
    assert_eq!(out.n_events, 5);
    let g = &out.gates;
    assert_eq!((g[0].count, g[1].count), (3, 2));
    assert_eq!(g[1].parent.as_deref(), Some("cd3"));
    assert!((g[0].fraction_of_total - 0.6).abs() < 1e-15);
    assert!((g[1].fraction_of_parent - 2.0 / 3.0).abs() < 1e-15);
    assert!((g[1].fraction_of_total - 0.4).abs() < 1e-15);
    // medians on the transformed scale; cd3 = rows 1..=3, activated = rows 2, 3 (even: mean of the two)
    assert_eq!(g[0].medians[0], ("CD3".to_string(), arcsinh(20.0, 5.0)));
    assert_eq!(g[0].medians[1], ("CD69".to_string(), arcsinh(200.0, 150.0)));
    let m = 0.5 * (arcsinh(20.0, 5.0) + arcsinh(30.0, 5.0));
    assert!((g[1].medians[0].1 - m).abs() < 1e-15);
}

#[test]
fn bounds_inclusive_nan_excluded_empty_parent() {
    let s = Strategy::from_toml(
        "[stats]\nmedians=['A']\n[[gate]]\nname='g'\nrange={A=[1.0,2.0]}\n[[gate]]\nname='h'\nparent='g'\nrange={A=[5.0,6.0]}\n[[gate]]\nname='k'\nparent='h'\nrange={A=[5.0,6.0]}",
    )
    .unwrap();
    let d = array![[1.0], [2.0], [f64::NAN], [0.999], [2.001]];
    let out = s.apply(&names(&["A"]), d.view()).unwrap();
    assert_eq!(out.gates[0].count, 2);
    assert_eq!(out.gates[0].medians[0].1, 1.5);
    assert_eq!(out.gates[1].count, 0);
    assert!(out.gates[1].medians[0].1.is_nan());
    assert!(out.gates[2].fraction_of_parent.is_nan()); // parent empty
    let empty = Array2::<f64>::zeros((0, 1));
    let out = s.apply(&names(&["A"]), empty.view()).unwrap();
    assert_eq!(out.n_events, 0);
    assert!(out.gates[0].fraction_of_total.is_nan());
}

#[test]
fn f32_and_f64_agree() {
    let s = Strategy::from_toml(EXAMPLE).unwrap();
    let d64 = array![[10.0, 100.0], [20.0, 200.5], [30.25, 300.0], [1.0, 1.0]];
    let d32 = d64.mapv(|x| x as f32);
    let n = names(&["CD3", "CD69"]);
    assert_eq!(
        s.apply(&n, d64.view()).unwrap(),
        s.apply(&n, d32.view()).unwrap()
    );
}

#[test]
fn invalid_strategies_are_rejected() {
    let bad = [
        ("[[gate]]\nname='a'\nrange={X=[2.0,1.0]}", "lo <= hi"),
        ("[[gate]]\nname='a'\nrange={X=[nan,1.0]}", "lo <= hi"),
        ("[[gate]]\nname='a'\nrange={}", "no range"),
        ("[[gate]]\nname=''\nrange={X=[0.0,1.0]}", "empty name"),
        (
            "[[gate]]\nname='a'\nrange={X=[0.0,1.0]}\n[[gate]]\nname='a'\nrange={X=[0.0,1.0]}",
            "twice",
        ),
        (
            "[[gate]]\nname='b'\nparent='a'\nrange={X=[0.0,1.0]}\n[[gate]]\nname='a'\nrange={X=[0.0,1.0]}",
            "not defined before",
        ),
        ("[transform]\ncofactor=0.0", "cofactor"),
        ("[transform.cofactors]\nX=-1.0", "cofactor"),
        ("[transform]\narcsinh=['X','X']", "twice"),
        ("[stats]\nmedians=['X','X']", "twice"),
    ];
    for (t, msg) in bad {
        match Strategy::from_toml(t) {
            Err(GateError::Invalid(m)) => assert!(m.contains(msg), "{t}: {m}"),
            other => panic!("{t}: expected Invalid, got {other:?}"),
        }
    }
    for t in [
        "[[gate]]\nname='a'\nrange={X=[0.0]}",
        "[gates]\nx=1",
        "[[gate]]\nname='a'\nrange={X=[0,1]}\npolygon=1",
        "nonsense =",
    ] {
        assert!(
            matches!(Strategy::from_toml(t), Err(GateError::Toml(_))),
            "{t}"
        );
    }
}

#[test]
fn apply_errors() {
    let s = Strategy::from_toml(EXAMPLE).unwrap();
    let d = Array2::<f64>::zeros((3, 2));
    assert!(
        matches!(s.apply(&names(&["CD3", "X"]), d.view()), Err(GateError::MissingChannel(c, _)) if c == "CD69")
    );
    assert!(matches!(
        s.apply(&names(&["CD3"]), d.view()),
        Err(GateError::Shape(2, 1))
    ));
}

fn naive_median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

proptest! {
    #[test]
    fn median_matches_sort(v in prop::collection::vec(-1e6f64..1e6, 0..200)) {
        let m = median(&mut v.clone());
        let r = naive_median(v);
        prop_assert!(m == r || (m.is_nan() && r.is_nan()));
    }

    /// Counts, fractions and medians equal a naive per-event evaluation (random data, cofactors, nested gates).
    #[test]
    fn apply_matches_naive(
        rows in prop::collection::vec(prop::collection::vec(-500f64..500.0, 3), 0..300),
        cof in 0.5f64..200.0,
        lo in prop::collection::vec(-3f64..3.0, 3),
        width in prop::collection::vec(0f64..4.0, 3),
    ) {
        let t = format!(
            "[transform]\ncofactor={cof}\narcsinh=['A','B']\n[stats]\nmedians=['A','C']\n\
             [[gate]]\nname='g1'\nrange={{A=[{},{}]}}\n\
             [[gate]]\nname='g2'\nparent='g1'\nrange={{B=[{},{}], C=[{},{}]}}",
            lo[0], lo[0] + width[0], lo[1], lo[1] + width[1], 100.0 * lo[2], 100.0 * (lo[2] + width[2]));
        let s = Strategy::from_toml(&t).unwrap();
        let n = rows.len();
        let d = Array2::from_shape_vec((n, 3), rows.concat()).unwrap();
        let out = s.apply(&names(&["A", "B", "C"]), d.view()).unwrap();
        let tr: Vec<[f64; 3]> = rows.iter().map(|r| [arcsinh(r[0], cof), arcsinh(r[1], cof), r[2]]).collect();
        let in1 = |r: &[f64; 3]| lo[0] <= r[0] && r[0] <= lo[0] + width[0];
        let in2 = |r: &[f64; 3]| in1(r) && lo[1] <= r[1] && r[1] <= lo[1] + width[1]
            && 100.0 * lo[2] <= r[2] && r[2] <= 100.0 * (lo[2] + width[2]);
        let g1: Vec<_> = tr.iter().filter(|r| in1(r)).collect();
        let g2: Vec<_> = tr.iter().filter(|r| in2(r)).collect();
        prop_assert_eq!(out.gates[0].count as usize, g1.len());
        prop_assert_eq!(out.gates[1].count as usize, g2.len());
        let med = |g: &Vec<&[f64; 3]>, j: usize| naive_median(g.iter().map(|r| r[j]).collect());
        for (gi, g) in [(0, &g1), (1, &g2)] {
            let (a, c) = (med(g, 0), med(g, 2));
            let got = &out.gates[gi].medians;
            prop_assert!(got[0].1 == a || (a.is_nan() && got[0].1.is_nan()));
            prop_assert!(got[1].1 == c || (c.is_nan() && got[1].1.is_nan()));
        }
        if !g1.is_empty() {
            prop_assert_eq!(out.gates[1].fraction_of_parent, g2.len() as f64 / g1.len() as f64);
        }
    }
}
