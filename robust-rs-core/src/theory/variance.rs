//! Asymptotic variance `V(ψ,F) = E[ψ²] / (E[ψ'])²` at the standard normal.

use super::quadrature::gauss_hermite;
use crate::rho::RhoFunction;

/// `E_Φ[ψ(X)²]` via Gauss–Hermite quadrature with `quad_points` nodes.
pub fn expect_psi_squared(rho: &dyn RhoFunction, quad_points: usize) -> f64 {
    let (nodes, weights) = gauss_hermite(quad_points);
    nodes
        .iter()
        .zip(&weights)
        .map(|(&x, &w)| {
            let psi = rho.psi(x);
            w * psi * psi
        })
        .sum()
}

/// `E_Φ[ψ'(X)]` via Gauss–Hermite quadrature with `quad_points` nodes.
///
/// Evaluated through Stein's identity `E_Φ[ψ'(X)] = E_Φ[X·ψ(X)]` rather than by
/// integrating `psi_prime` directly. Both target the same number, but the `X·ψ`
/// integrand is continuous for clipped/kinked ψ (Huber, biweight) and remains
/// correct when ψ' drops mass in the classical sense (L1: ψ'≡0 a.e., yet the
/// denominator that belongs here is E[|X|]=√(2/π)). Integrating `psi_prime`
/// directly is trustworthy only when ψ' is continuous.
pub fn expect_psi_prime(rho: &dyn RhoFunction, quad_points: usize) -> f64 {
    let (nodes, weights) = gauss_hermite(quad_points);
    nodes
        .iter()
        .zip(&weights)
        .map(|(&x, &w)| w * x * rho.psi(x))
        .sum()
}

/// Asymptotic variance `E[ψ²] / (E[ψ'])²`.
pub fn asymptotic_variance(rho: &dyn RhoFunction, quad_points: usize) -> f64 {
    // One quadrature rule, both expectations.
    let (nodes, weights) = gauss_hermite(quad_points);
    let mut e_psi2 = 0.0;
    let mut e_psip = 0.0; // E[ψ'] via Stein: Σ wᵢ xᵢ ψ(xᵢ)
    for (&x, &w) in nodes.iter().zip(&weights) {
        let psi = rho.psi(x);
        e_psi2 += w * psi * psi;
        e_psip += w * x * psi;
    }
    e_psi2 / (e_psip * e_psip)
}
