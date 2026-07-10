//! v0.2 exit gate: MM-regression on Hertzsprung–Russell `starsCYG` recovers the
//! main-sequence positive slope that OLS gets wrong, rejects the four giant
//! stars (cases 11, 20, 30, 34) and reports 50% breakdown with ≈ 95%
//! efficiency, versus the S-stage's ≈ 29%.

use ndarray::Array2;
use robust_rs::prelude::*;

/// `starsCYG` with an intercept column prepended: `X` is `47 × 2`
/// (`[1, log.Te]`), `y` is `log.light`.
fn stars_with_intercept() -> (Array2<f64>, ndarray::Array1<f64>) {
    let (x_raw, y) = robust_rs::datasets::stars_cyg();
    let n = x_raw.nrows();
    let mut x = Array2::ones((n, 2));
    x.column_mut(1).assign(&x_raw.column(0));
    (x, y)
}

#[test]
fn ols_is_dragged_to_a_negative_slope() {
    let (x, y) = stars_with_intercept();
    let ols = MEstimator::new(LeastSquares, Mad::default())
        .fit(&x, &y)
        .unwrap();
    // The four giant stars flip the sign: OLS "concludes" hotter ⇒ dimmer.
    assert!(
        ols.coefficients()[1] < 0.0,
        "OLS slope should be negative, got {}",
        ols.coefficients()[1]
    );
}

#[test]
fn mm_recovers_positive_slope_and_rejects_the_giants() {
    let (x, y) = stars_with_intercept();
    let mm = MMEstimator::default().fit(&x, &y).unwrap();

    // Physically correct main-sequence slope: hotter ⇒ brighter.
    assert!(
        mm.coefficients()[1] > 0.0,
        "MM slope should be positive, got {}",
        mm.coefficients()[1]
    );

    // The four giant stars (1-indexed cases 11, 20, 30, 34) are outliers in the
    // H–R diagram; a redescending M-step drives their weights to exactly zero.
    for case in [11usize, 20, 30, 34] {
        let w = mm.weights[case - 1];
        assert!(
            w < 1e-6,
            "giant star case {case} should be rejected, weight = {w}"
        );
    }

    // Main-sequence stars keep substantial weight.
    let kept: f64 = [1usize, 2, 3, 5]
        .iter()
        .map(|&c| mm.weights[c - 1])
        .sum::<f64>()
        / 4.0;
    assert!(
        kept > 0.5,
        "main-sequence stars should keep weight, mean = {kept}"
    );
}

#[test]
fn mm_reports_high_breakdown_and_high_efficiency() {
    let (x, y) = stars_with_intercept();
    let mm = MMEstimator::default().fit(&x, &y).unwrap();
    assert!(
        (mm.breakdown_point() - 0.5).abs() < 1e-9,
        "MM breakdown should be 0.5, got {}",
        mm.breakdown_point()
    );
    assert!(
        mm.gaussian_efficiency() > 0.9,
        "MM efficiency should be ≈0.95, got {}",
        mm.gaussian_efficiency()
    );
}

#[test]
fn s_stage_is_robust_but_inefficient() {
    // The S-estimate is the counterpoint: same 50% breakdown, but its reported
    // efficiency is low (~0.29), which is why MM adds a step.
    let (x, y) = stars_with_intercept();
    let s = SEstimator::default().fit(&x, &y).unwrap();
    assert!(s.coefficients()[1] > 0.0, "S slope should be positive");
    assert!((s.breakdown_point() - 0.5).abs() < 1e-9);
    assert!(
        s.gaussian_efficiency() < 0.4,
        "S efficiency should be low, got {}",
        s.gaussian_efficiency()
    );
    // MM buys efficiency on top of the same breakdown.
    let mm = MMEstimator::default().fit(&x, &y).unwrap();
    assert!(mm.gaussian_efficiency() > s.gaussian_efficiency());
}

#[test]
fn fits_are_reproducible_from_the_seed() {
    let (x, y) = stars_with_intercept();
    let a = MMEstimator::default().seed(123).fit(&x, &y).unwrap();
    let b = MMEstimator::default().seed(123).fit(&x, &y).unwrap();
    assert_eq!(
        a.coefficients, b.coefficients,
        "same seed must give bit-identical coefficients"
    );
}
