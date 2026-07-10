//! Robust Mahalanobis distances and the multivariate outlier map, over *any*
//! robust location/scatter pair `(μ̂, Σ̂)`, plus the classical (non-robust)
//! mean/covariance baseline they are meant to replace.
//!
//! A classical Mahalanobis distance uses the sample mean and covariance, both of
//! which have breakdown point `0`: a few outliers inflate the covariance and
//! shrink their own distances (the *masking* effect), so they hide from the very
//! statistic meant to reveal them. Feeding a robust, `χ²`-calibrated `(μ̂, Σ̂)`
//! (from [`Mcd`], [`Ogk`] or [`MScatter`]) into the same formula fixes this: the
//! distances of the clean majority stay small and the outliers stand out.
//! (Tyler's estimator identifies only *shape*, so its distances are not
//! `χ²`-calibrated for this cutoff; see [`TylerFit`](crate::multivariate::TylerFit).)
//!
//! [`Mcd`]: crate::multivariate::Mcd
//! [`Ogk`]: crate::multivariate::Ogk
//! [`MScatter`]: crate::multivariate::MScatter

use ndarray::{Array1, Array2};
use robust_rs_core::error::RobustError;

use super::chi2::chi2_quantile;
use super::linalg::mean_covariance;

/// Robust Mahalanobis distances `dᵢ = √((xᵢ − loc)ᵀ Σ⁻¹ (xᵢ − loc))` for every
/// row of `x`, given a location `loc` (`p`-vector) and a symmetric positive-
/// definite scatter `Σ` (`p × p`). Errors on a dimension mismatch or a
/// non-positive-definite scatter.
pub fn mahalanobis_distances(
    x: &Array2<f64>,
    loc: &Array1<f64>,
    scatter: &Array2<f64>,
) -> Result<Array1<f64>, RobustError> {
    let p = x.ncols();
    if loc.len() != p {
        return Err(RobustError::DimensionMismatch {
            expected: p,
            got: loc.len(),
        });
    }
    if scatter.nrows() != p || scatter.ncols() != p {
        return Err(RobustError::DimensionMismatch {
            expected: p,
            got: scatter.nrows(),
        });
    }
    super::distances_from(x, loc, scatter)
}

/// The classical (non-robust) mean and unbiased sample covariance of `x`, the
/// breakdown-`0` baseline the robust estimators replace. Handy for
/// side-by-side comparison and for a maximum-likelihood starting point.
pub fn classical_covariance(x: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    mean_covariance(x)
}

/// The distance cutoff `√χ²_{p, quantile}` for flagging outliers on `p`
/// variables (e.g. `quantile = 0.975`).
pub fn distance_cutoff(p: usize, quantile: f64) -> f64 {
    chi2_quantile(quantile, p as f64).sqrt()
}

/// Flag distances exceeding `distance_cutoff(p, quantile)`. Under a `p`-variate
/// Gaussian the squared robust distances are approximately `χ²_p`, so a
/// `quantile` of `0.975` flags ≈ 2.5% of clean observations.
pub fn outlier_flags(distances: &Array1<f64>, p: usize, quantile: f64) -> Vec<bool> {
    let cut = distance_cutoff(p, quantile);
    distances.iter().map(|&d| d > cut).collect()
}
