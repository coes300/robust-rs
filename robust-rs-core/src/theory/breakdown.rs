//! Breakdown point of an estimator.

use crate::rho::RhoFunction;

/// Asymptotic breakdown point of a **regression M-estimator** using this loss.
///
/// This is `0`: a single observation with unbounded leverage can carry the fit
/// arbitrarily far, no matter how `ρ` bounds the *residual*. Bounding influence
/// in residual space does not bound it in the design. The high-breakdown
/// estimators (S, MM, LTS) attain up to `0.5` and report their design breakdown
/// instead.
pub fn breakdown_point(rho: &dyn RhoFunction) -> f64 {
    let _ = rho;
    0.0
}
