//! Multivariate robust statistics: robust estimates of multivariate location
//! and scatter and the outlier detection built on them.
//!
//! The organizing object here is a **location–scatter pair** `(μ̂, Σ̂)` (a
//! robust centre and a robust covariance) from which a robust Mahalanobis
//! distance `dᵢ = √((xᵢ − μ̂)ᵀ Σ̂⁻¹ (xᵢ − μ̂))` flags multivariate outliers, just
//! as robust residuals flag them in regression. Four estimators produce such a
//! pair, trading efficiency against breakdown and equivariance:
//!
//! - [`Mcd`]: the Minimum Covariance Determinant (Rousseeuw 1985) via FAST-MCD
//!   (Rousseeuw & Van Driessen 1999): 50%-breakdown, affine-equivariant, the
//!   multivariate analogue of LTS.
//! - [`Ogk`]: the Orthogonalized Gnanadesikan–Kettenring estimator
//!   (Maronna & Zamar 2002): a fast, deterministic, positive-definite pairwise
//!   estimator (orthogonally, not fully affine, equivariant).
//! - [`MScatter`]: a monotone M-estimator of location and scatter
//!   (Maronna 1976): the direct multivariate analogue of the regression
//!   M-estimator, reusing a [`robust_rs_core::rho::RhoFunction`] weight.
//! - [`Tyler`]: Tyler's (1987) distribution-free M-estimator of *shape*,
//!   normalized to unit determinant.
//!
//! [`mahalanobis`] exposes the distance/outlier map over *any* `(μ̂, Σ̂)` pair,
//! together with the classical (non-robust) mean/covariance baseline.

mod chi2;
mod correction;
mod linalg;

mod m_scatter;
pub mod mahalanobis;
mod mcd;
mod ogk;
mod tyler;

pub use self::m_scatter::MScatter;
pub use self::mcd::{Mcd, McdFit};
pub use self::ogk::Ogk;
pub use self::tyler::{Tyler, TylerFit};

use ndarray::{Array1, Array2};
use robust_rs_core::error::RobustError;

use self::chi2::chi2_quantile;
use self::linalg::{mahalanobis_sq, spd_inverse_logdet};

/// A fitted robust location–scatter estimate.
///
/// The shared result type of [`MScatter`] and [`Ogk`]. FAST-MCD carries extra
/// structure (a raw estimate and the retained subset), so it returns its own
/// [`McdFit`], but that type implements [`RobustScatter`] too, so all three are
/// interchangeable behind the trait and the Mahalanobis/outlier map is written
/// once against `(μ̂, Σ̂)`. [`Tyler`] does *not* use this type: it returns a
/// bespoke [`TylerFit`] implementing no shared trait, because its shape-only
/// distances are not `χ²`-calibrated, so this type's Gaussian outlier map would
/// misapply to them.
#[derive(Debug, Clone)]
pub struct ScatterFit {
    /// Robust location estimate `μ̂` (a `p`-vector).
    pub location: Array1<f64>,
    /// Robust scatter (covariance) estimate `Σ̂` (`p × p`, symmetric positive
    /// definite).
    pub scatter: Array2<f64>,
    /// Robust Mahalanobis distances `dᵢ` of the rows the fit was computed on.
    pub distances: Array1<f64>,
    /// Final per-observation weights (their meaning is estimator-specific: a
    /// hard `0/1` acceptance for a reweighting estimator, a smooth `w(dᵢ)` for
    /// an M-estimator).
    pub weights: Array1<f64>,
}

/// Quantities every fitted robust covariance estimator can report, the
/// multivariate counterpart of [`crate::estimator::RobustEstimator`].
pub trait RobustScatter {
    /// Robust location estimate `μ̂`.
    fn location(&self) -> &Array1<f64>;
    /// Robust scatter estimate `Σ̂`.
    fn scatter(&self) -> &Array2<f64>;
    /// Robust Mahalanobis distances of the fitted rows.
    fn distances(&self) -> &Array1<f64>;

    /// The distance cutoff `√χ²_{p, quantile}` (e.g. `quantile = 0.975`) against
    /// which robust Mahalanobis distances are compared to flag outliers.
    fn distance_cutoff(&self, quantile: f64) -> f64 {
        let p = self.scatter().nrows() as f64;
        chi2_quantile(quantile, p).sqrt()
    }

    /// Flag each fitted observation as an outlier when its robust distance
    /// exceeds `distance_cutoff(quantile)`. Under a `p`-variate Gaussian the
    /// squared distances are `χ²_p`, so `quantile = 0.975` flags ≈ 2.5% of clean
    /// data.
    fn outliers(&self, quantile: f64) -> Vec<bool> {
        let cut = self.distance_cutoff(quantile);
        self.distances().iter().map(|&d| d > cut).collect()
    }
}

impl RobustScatter for ScatterFit {
    fn location(&self) -> &Array1<f64> {
        &self.location
    }
    fn scatter(&self) -> &Array2<f64> {
        &self.scatter
    }
    fn distances(&self) -> &Array1<f64> {
        &self.distances
    }
}

/// Robust Mahalanobis distances `dᵢ = √((xᵢ − loc)ᵀ scatter⁻¹ (xᵢ − loc))` for
/// every row of `x`. Shared by the estimators to populate [`ScatterFit`] and
/// re-exported publicly as [`mahalanobis::mahalanobis_distances`]. Errors if the
/// scatter is not positive definite.
pub(crate) fn distances_from(
    x: &Array2<f64>,
    loc: &Array1<f64>,
    scatter: &Array2<f64>,
) -> Result<Array1<f64>, RobustError> {
    let (inv, _logdet) = spd_inverse_logdet(scatter)?;
    Ok(mahalanobis_sq(x, loc, &inv).mapv(f64::sqrt))
}
