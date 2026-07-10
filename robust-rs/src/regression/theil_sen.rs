//! The Theil–Sen estimator (Theil 1950; Sen 1968): the median of pairwise
//! slopes.
//!
//! For simple (single-predictor) linear regression `y ≈ a + b·x`, the slope is
//! the median of `(yⱼ − yᵢ)/(xⱼ − xᵢ)` over all pairs `i < j` and the intercept
//! is `median(yᵢ − b·xᵢ)`. Pairs with `xᵢ = xⱼ` give an undefined slope and are
//! **skipped**, so the slope is the median over the `n(n−1)/2` pairs *minus the
//! x-ties*. High breakdown (`1 − 1/√2 ≈ 0.293`) with no tuning constant.
//!
//! Like [`crate::regression::LtsFit`] this is a bespoke result type, not an
//! M-estimator. Theil–Sen has a known Gaussian efficiency, but it is not
//! `ρ`-derivable, so it is not reported through the `ρ`-based
//! [`crate::estimator::RobustEstimator`] API; only the breakdown point (a settled
//! constant) is exposed.

use crate::util::median;
use robust_rs_core::error::RobustError;

/// Theil–Sen asymptotic breakdown point, `1 − 1/√2 ≈ 0.293`.
pub const THEIL_SEN_BREAKDOWN: f64 = 1.0 - std::f64::consts::FRAC_1_SQRT_2;

/// A fitted Theil–Sen simple-regression line `y ≈ intercept + slope·x`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TheilSenFit {
    /// The median-of-pairwise-slopes estimate.
    pub slope: f64,
    /// The intercept, `median(yᵢ − slope·xᵢ)`.
    pub intercept: f64,
}

impl TheilSenFit {
    /// The estimated slope.
    pub fn slope(&self) -> f64 {
        self.slope
    }
    /// The estimated intercept.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
    /// The predicted response at `x`.
    pub fn predict(&self, x: f64) -> f64 {
        self.intercept + self.slope * x
    }
    /// The asymptotic breakdown point (`THEIL_SEN_BREAKDOWN` = `1 − 1/√2`).
    pub fn breakdown_point(&self) -> f64 {
        THEIL_SEN_BREAKDOWN
    }
}

/// Fit a Theil–Sen line to paired `(x, y)` data.
///
/// Returns [`RobustError::DimensionMismatch`] if the lengths differ,
/// [`RobustError::InsufficientData`] for fewer than two points and
/// [`RobustError::SingularDesign`] if every `xᵢ` is equal (the slope is then
/// unidentifiable; every pair is an x-tie).
pub fn theil_sen(x: &[f64], y: &[f64]) -> Result<TheilSenFit, RobustError> {
    let n = x.len();
    if y.len() != n {
        return Err(RobustError::DimensionMismatch {
            expected: n,
            got: y.len(),
        });
    }
    if n < 2 {
        return Err(RobustError::InsufficientData { needed: 2, got: n });
    }

    // Pairwise slopes, skipping x-ties (an undefined slope). Comparing `dx` to
    // the exact literal 0.0 is the intended test, a true tie, not a tolerance.
    let mut slopes = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[j] - x[i];
            if dx != 0.0 {
                slopes.push((y[j] - y[i]) / dx);
            }
        }
    }
    if slopes.is_empty() {
        return Err(RobustError::SingularDesign); // all x tied
    }
    let slope = median(&mut slopes);

    let mut offsets: Vec<f64> = x.iter().zip(y).map(|(&xi, &yi)| yi - slope * xi).collect();
    let intercept = median(&mut offsets);

    Ok(TheilSenFit { slope, intercept })
}
