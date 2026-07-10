//! Theil–Sen and Hodges–Lehmann: the median-of-pairwise estimators.

use robust_rs::error::RobustError;
use robust_rs::prelude::*;

#[test]
fn theil_sen_recovers_the_line_and_resists_a_vertical_outlier() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let clean = [3.0, 5.0, 7.0, 9.0, 11.0]; // y = 2x + 1
    let fit = theil_sen(&x, &clean).unwrap();
    assert!((fit.slope() - 2.0).abs() < 1e-12);
    assert!((fit.intercept() - 1.0).abs() < 1e-12);

    // One gross vertical outlier must not move the fit (≈29% breakdown).
    let mut bad = clean;
    bad[2] = 100.0;
    let fit2 = theil_sen(&x, &bad).unwrap();
    assert!(
        (fit2.slope() - 2.0).abs() < 1e-12,
        "TS slope should resist the outlier, got {}",
        fit2.slope()
    );
    assert!((fit2.intercept() - 1.0).abs() < 1e-12);
    assert!((fit2.predict(6.0) - 13.0).abs() < 1e-12);
}

#[test]
fn theil_sen_skips_x_ties_and_flags_an_all_tied_design() {
    // A repeated x is fine as long as some pair has distinct x (the tie is skipped).
    let x = [1.0, 1.0, 2.0, 3.0];
    let y = [1.0, 5.0, 3.0, 5.0];
    assert!(theil_sen(&x, &y).is_ok());

    // Every x identical ⇒ no defined slope ⇒ singular design.
    let flat_x = [2.0, 2.0, 2.0];
    let yy = [1.0, 2.0, 3.0];
    assert_eq!(
        theil_sen(&flat_x, &yy).unwrap_err(),
        RobustError::SingularDesign
    );
}

#[test]
fn hodges_lehmann_is_the_median_of_walsh_averages() {
    // Walsh averages of {1,2,3} (i ≤ j, self-pairs included):
    // {1, 1.5, 2, 2, 2.5, 3} ⇒ median 2.
    let hl = hodges_lehmann(&[1.0, 2.0, 3.0]).unwrap();
    assert!((hl.estimate() - 2.0).abs() < 1e-12);
}

#[test]
fn hodges_lehmann_is_translation_equivariant() {
    let data = [2.0, 3.0, 5.0, 7.0, 11.0];
    let base = hodges_lehmann(&data).unwrap().estimate();
    let shifted: Vec<f64> = data.iter().map(|x| x + 10.0).collect();
    let shifted_est = hodges_lehmann(&shifted).unwrap().estimate();
    assert!((shifted_est - (base + 10.0)).abs() < 1e-12);
}

#[test]
fn hodges_lehmann_reports_known_theory_constants() {
    let hl = hodges_lehmann(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    // Wilcoxon ARE 3/π ≈ 0.955, reported directly (not via a ρ).
    assert!((hl.gaussian_efficiency() - 3.0 / std::f64::consts::PI).abs() < 1e-15);
    assert!((hl.breakdown_point() - (1.0 - std::f64::consts::FRAC_1_SQRT_2)).abs() < 1e-15);
}
