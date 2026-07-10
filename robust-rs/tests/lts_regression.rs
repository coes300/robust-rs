//! LTS on `starsCYG`: the retained-subset analogue of the MM weight-vector
//! check. A correct FAST-LTS must trim exactly the outliers, its `h`-subset
//! must exclude the four giant stars (cases 11, 20, 30, 34) and recover a
//! positive main-sequence slope, with the breakdown point its coverage implies.

use ndarray::Array2;
use robust_rs::prelude::*;

/// `starsCYG` with an intercept column prepended (`47 × 2`).
fn stars_with_intercept() -> (Array2<f64>, ndarray::Array1<f64>) {
    let (x_raw, y) = robust_rs::datasets::stars_cyg();
    let n = x_raw.nrows();
    let mut x = Array2::ones((n, 2));
    x.column_mut(1).assign(&x_raw.column(0));
    (x, y)
}

#[test]
fn lts_trims_the_giants_and_recovers_positive_slope() {
    let (x, y) = stars_with_intercept();
    let lts = Lts::default().fit(&x, &y).unwrap();

    assert!(
        lts.coefficients()[1] > 0.0,
        "LTS slope should be positive, got {}",
        lts.coefficients()[1]
    );

    // The four giant stars must be among the trimmed observations.
    for case in [11usize, 20, 30, 34] {
        assert!(
            !lts.subset().contains(&(case - 1)),
            "giant star case {case} must be trimmed, but was retained"
        );
    }
    assert_eq!(lts.subset().len(), lts.coverage, "subset size must equal h");
}

#[test]
fn lts_breakdown_matches_coverage() {
    let (x, y) = stars_with_intercept();
    let lts = Lts::default().fit(&x, &y).unwrap(); // n = 47, h = 25
    let n = 47.0;
    let h = lts.coverage as f64;
    assert!((lts.breakdown_point() - (n - h + 1.0) / n).abs() < 1e-12);
    assert!(
        lts.breakdown_point() > 0.45,
        "max-breakdown LTS should be ≈0.49, got {}",
        lts.breakdown_point()
    );
}

#[test]
fn lts_higher_coverage_lowers_breakdown() {
    let (x, y) = stars_with_intercept();
    let max_bp = Lts::default().fit(&x, &y).unwrap();
    let higher_cov = Lts::default().coverage(0.9).fit(&x, &y).unwrap();
    assert!(higher_cov.coverage > max_bp.coverage);
    assert!(higher_cov.breakdown_point() < max_bp.breakdown_point());
}

#[test]
fn lts_is_reproducible_from_the_seed() {
    let (x, y) = stars_with_intercept();
    let a = Lts::default().seed(7).fit(&x, &y).unwrap();
    let b = Lts::default().seed(7).fit(&x, &y).unwrap();
    assert_eq!(a.coefficients, b.coefficients, "same seed ⇒ identical fit");
    assert_eq!(a.subset, b.subset, "same seed ⇒ identical retained subset");
}
