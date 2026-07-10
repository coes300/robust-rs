//! Golden cross-check of robust-rs' **M-estimation spine** against Python
//! `statsmodels` RLM on the stack-loss data. Regenerate the fixture with
//! `scripts/gen_golden_py.py` (see `docs/validation.md`).
//!
//! statsmodels centers the residual MAD at **0** while `robust_rs::scale::Mad`
//! centers it at the **median**, so the two disagree on the scale *value*: a
//! documented convention, not a bug. To compare the IRLS + Huber-ψ + weighted
//! least squares spine like-for-like, this test fixes robust-rs' scale at
//! statsmodels' value (from the fixture); the coefficients then agree to ~1e-6
//! (the residual is convergence tolerance). The scale estimator itself is
//! validated separately by `robust-rs-core`'s `scale_consistency` tests.

use robust_rs::error::RobustError;
use robust_rs::prelude::*;
use robust_rs::types::Scale;

/// A scale estimator pinned to a fixed value, so the IRLS spine can be compared
/// independent of the MAD-centering convention.
struct FixedScale(f64);
impl ScaleEstimator for FixedScale {
    fn scale(&self, _residuals: &[f64]) -> Result<Scale, RobustError> {
        Scale::new(self.0)
    }
}

#[test]
fn huber_m_spine_matches_statsmodels_rlm_on_stackloss() {
    let json = std::fs::read_to_string("tests/fixtures/rlm_stackloss.json")
        .expect("fixture present; regenerate with scripts/gen_golden_py.py");
    let reference: serde_json::Value = serde_json::from_str(&json).unwrap();
    let coefs: Vec<f64> = reference["coefficients"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let scale = reference["scale_value"].as_f64().unwrap();

    // stackloss with an intercept column: X is 21 × 4.
    let (x, y) = robust_rs::datasets::stackloss();
    let n = x.nrows();
    let mut design = ndarray::Array2::ones((n, 4));
    for j in 0..3 {
        design.column_mut(j + 1).assign(&x.column(j));
    }

    let fit = MEstimator::new(Huber::default(), FixedScale(scale))
        .fit(&design, &y)
        .expect("fit");

    assert_eq!(fit.coefficients().len(), coefs.len());
    for (i, &reference_coef) in coefs.iter().enumerate() {
        approx::assert_relative_eq!(fit.coefficients()[i], reference_coef, epsilon = 1e-6);
    }
}
