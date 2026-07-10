//! The influence function of an M-estimator of location.

use super::variance::expect_psi_prime;
use crate::rho::RhoFunction;

/// Return the influence function `x ↦ ψ(x) / E_Φ[ψ']` as a closure.
pub fn influence_function<'a>(
    rho: &'a dyn RhoFunction,
    quad_points: usize,
) -> impl Fn(f64) -> f64 + 'a {
    let c = expect_psi_prime(rho, quad_points);
    move |x: f64| rho.psi(x) / c
}
