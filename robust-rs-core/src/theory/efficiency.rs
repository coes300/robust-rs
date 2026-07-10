//! Efficiency relative to the Gaussian MLE.

use super::variance::asymptotic_variance;
use crate::rho::RhoFunction;

/// Asymptotic efficiency at the Gaussian: `(E[ψ'])² / E[ψ²]`; the reciprocal of
/// the asymptotic variance, since the location MLE attains variance 1.
pub fn gaussian_efficiency(rho: &dyn RhoFunction, quad_points: usize) -> f64 {
    1.0 / asymptotic_variance(rho, quad_points)
}
