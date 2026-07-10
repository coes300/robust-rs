//! The `LeastSquares` (L2) loss, giving the mean / ordinary least squares.

use crate::rho::RhoFunction;

/// `ρ(r) = r²/2`, `ψ(r) = r`. The efficient, zero-breakdown baseline.
#[derive(Debug, Clone, Copy, Default)]
pub struct LeastSquares;

impl RhoFunction for LeastSquares {
    fn rho(&self, r: f64) -> f64 {
        r.powi(2) / 2.0
    }
    fn psi(&self, r: f64) -> f64 {
        r
    }
    fn weight(&self, _r: f64) -> f64 {
        1.0
    }
    fn psi_prime(&self, _r: f64) -> f64 {
        1.0
    }
    fn tuning(&self) -> f64 {
        f64::NAN
    }
    fn is_redescending(&self) -> bool {
        false
    }
    fn rho_sup(&self) -> Option<f64> {
        None
    }
}
