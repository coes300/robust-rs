//! Shared MCD / OGK finishing corrections: the Gaussian **consistency factor**
//! and the one-step **hard-rejection reweighting** (RMCD / reweighted OGK).
//!
//! Both a raw MCD subset covariance and a raw OGK covariance are consistent for
//! `Σ` only up to a scalar that depends on the retained proportion `α`. The
//! consistency factor `c(α, p) = α / F_{χ²_{p+2}}(χ²_{p,α})` (Croux &
//! Haesbroeck 1999) supplies that scalar. The reweighting step then recomputes a
//! classical mean/covariance on the observations whose robust distance falls
//! within the `χ²_{p,q}` cutoff (recovering efficiency while keeping the
//! breakdown point) and applies the consistency factor at the *observed*
//! retention fraction.

use ndarray::{Array1, Array2, Axis};
use robust_rs_core::error::RobustError;

use super::chi2::{chi2_cdf, chi2_quantile};
use super::distances_from;
use super::linalg::mean_covariance;
use crate::util::median;

/// The MCD consistency factor `c(α, p) = α / F_{χ²_{p+2}}(χ²_{p,α})`, scaling a
/// raw covariance estimated from a proportion `α` of the data to be
/// Fisher-consistent for `Σ` at the Gaussian. Tends to `1` as `α → 1`.
pub(crate) fn consistency_factor(alpha: f64, p: f64) -> f64 {
    if alpha >= 1.0 {
        return 1.0;
    }
    let q = chi2_quantile(alpha, p);
    let denom = chi2_cdf(q, p + 2.0);
    if denom > 0.0 {
        alpha / denom
    } else {
        1.0
    }
}

/// The result of a one-step reweighting.
pub(crate) struct Reweighted {
    /// Reweighted location (mean of the retained observations).
    pub location: Array1<f64>,
    /// Reweighted, consistency-corrected covariance.
    pub scatter: Array2<f64>,
    /// Robust Mahalanobis distances w.r.t. the reweighted estimate.
    pub distances: Array1<f64>,
    /// `1.0` for retained observations, `0.0` for rejected ones.
    pub weights: Array1<f64>,
}

/// One-step hard-rejection reweighting: recompute the classical mean/covariance
/// on the observations whose squared robust distance (against the raw estimate)
/// is within a `χ²` cutoff, then apply the consistency factor at the observed
/// retention fraction.
///
/// The cutoff on squared distances is either **fixed** at `χ²_{p, quantile}`
/// (the RMCD rule, `adjusted = false`) or **median-adjusted** to
/// `med(d²) · χ²_{p, quantile} / χ²_{p, 0.5}` (the Maronna–Zamar OGK rule,
/// `adjusted = true`). The adjusted form self-calibrates to the raw estimate's
/// scale, so it does not silently revert to the classical covariance when the
/// raw scatter is only mildly inflated.
///
/// Returns `Ok(None)` when fewer than `p + 1` observations survive the cutoff
/// (so no covariance can be formed); the caller then falls back to the raw
/// estimate rather than fail.
pub(crate) fn hard_reweight(
    x: &Array2<f64>,
    raw_location: &Array1<f64>,
    raw_scatter: &Array2<f64>,
    quantile: f64,
    adjusted: bool,
) -> Result<Option<Reweighted>, RobustError> {
    let (n, p) = x.dim();
    let raw_dist = distances_from(x, raw_location, raw_scatter)?;

    // Threshold on squared distances.
    let cut2 = if adjusted {
        let mut d2: Vec<f64> = raw_dist.iter().map(|&d| d * d).collect();
        let med = median(&mut d2);
        med * chi2_quantile(quantile, p as f64) / chi2_quantile(0.5, p as f64)
    } else {
        chi2_quantile(quantile, p as f64)
    };

    let mut weights = Array1::<f64>::zeros(n);
    let mut keep: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        if raw_dist[i] * raw_dist[i] <= cut2 {
            weights[i] = 1.0;
            keep.push(i);
        }
    }
    if keep.len() < p + 1 {
        return Ok(None);
    }

    let xs = x.select(Axis(0), &keep);
    let (location, cov_raw) = mean_covariance(&xs);
    let alpha = keep.len() as f64 / n as f64;
    let scatter = &cov_raw * consistency_factor(alpha, p as f64);
    let distances = distances_from(x, &location, &scatter)?;

    Ok(Some(Reweighted {
        location,
        scatter,
        distances,
        weights,
    }))
}
