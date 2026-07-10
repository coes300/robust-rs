//! The Hodges–Lehmann location estimator (Hodges & Lehmann 1963): the median of
//! the Walsh averages `(xᵢ + xⱼ)/2` over all pairs `i ≤ j`.
//!
//! The self-pairs `i = j` (which contribute the raw `xᵢ`) are **included**, so
//! there are `n(n+1)/2` Walsh averages, the standard HL definition. Its Gaussian
//! efficiency is the Wilcoxon asymptotic relative efficiency `3/π ≈ 0.955` and
//! its breakdown point is `1 − 1/√2 ≈ 0.293`.
//!
//! Like the crate's other non-M-estimators this is a bespoke type with no
//! `ρ`-based theory surface, but its efficiency *is* a settled constant, so
//! [`HodgesLehmannFit::gaussian_efficiency`] reports it directly (never via a
//! stored `ρ`).

use crate::util::median;
use robust_rs_core::error::RobustError;

/// A Hodges–Lehmann location estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HodgesLehmannFit {
    /// The estimate: the median of the Walsh averages.
    pub estimate: f64,
}

impl HodgesLehmannFit {
    /// The location estimate.
    pub fn estimate(&self) -> f64 {
        self.estimate
    }
    /// Gaussian efficiency `3/π ≈ 0.955`, the Wilcoxon asymptotic relative
    /// efficiency, a known constant (not `ρ`-derived).
    pub fn gaussian_efficiency(&self) -> f64 {
        3.0 / std::f64::consts::PI
    }
    /// Asymptotic breakdown point `1 − 1/√2 ≈ 0.293`.
    pub fn breakdown_point(&self) -> f64 {
        1.0 - std::f64::consts::FRAC_1_SQRT_2
    }
}

/// Compute the Hodges–Lehmann location of `data`.
///
/// Returns [`RobustError::InsufficientData`] for empty input.
pub fn hodges_lehmann(data: &[f64]) -> Result<HodgesLehmannFit, RobustError> {
    let n = data.len();
    if n == 0 {
        return Err(RobustError::InsufficientData { needed: 1, got: 0 });
    }

    // Walsh averages (xᵢ + xⱼ)/2 for i ≤ j; the diagonal i = j contributes xᵢ.
    let mut walsh = Vec::with_capacity(n * (n + 1) / 2);
    for i in 0..n {
        for &xj in &data[i..] {
            walsh.push(0.5 * (data[i] + xj));
        }
    }
    Ok(HodgesLehmannFit {
        estimate: median(&mut walsh),
    })
}
