//! The `L1` (absolute-value) loss, giving the median / L1 regression.

use crate::rho::RhoFunction;

/// `ρ(r) = |r|`, `ψ(r) = sign(r)`. Bounded influence, 50% breakdown (location).
#[derive(Debug, Clone, Copy, Default)]
pub struct L1;

impl RhoFunction for L1 {
    fn rho(&self, r: f64) -> f64 {
        r.abs()
    }
    fn psi(&self, r: f64) -> f64 {
        if r == 0.0 {
            0.0
        } else {
            r.signum()
        }
    }
    fn weight(&self, r: f64) -> f64 {
        if r == 0.0 {
            // The true limit ψ(r)/r = 1/|r| is +∞ at r = 0 (unlike the smooth
            // losses, whose limit is the finite ψ'(0) = 1). We cap it at the same
            // `weight(0) = 1` convention: finite and IRLS-stable, so a residual
            // sitting exactly on the fit is kept, not dropped. See docs/conventions.md.
            1.0
        } else {
            1.0 / r.abs()
        }
    }
    fn psi_prime(&self, _r: f64) -> f64 {
        0.0
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
