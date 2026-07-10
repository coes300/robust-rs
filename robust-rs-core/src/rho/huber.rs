//! Huber's loss: quadratic near zero, linear in the tails. Monotone `ψ` ⇒ convex.

use crate::error::RobustError;
use crate::rho::RhoFunction;

/// Huber's loss with tuning `k` (default `1.345`, ≈ 95% Gaussian efficiency).
/// `ρ(r) = r²/2` for `|r| ≤ k`, else `k|r| − k²/2`.
#[derive(Debug, Clone, Copy)]
pub struct Huber {
    k: f64,
}

impl Huber {
    /// Create a Huber loss with the given positive tuning constant.
    pub fn new(k: f64) -> Result<Self, RobustError> {
        if k.is_finite() && k > 0.0 {
            Ok(Self { k })
        } else {
            Err(RobustError::InvalidTuning { value: k })
        }
    }
}

impl Default for Huber {
    /// `k = 1.345`.
    fn default() -> Self {
        Self { k: 1.345 }
    }
}

impl RhoFunction for Huber {
    fn rho(&self, r: f64) -> f64 {
        if r.abs() <= self.k {
            r.powi(2) / 2.0
        } else {
            self.k * r.abs() - self.k.powi(2) / 2.0
        }
    }
    fn psi(&self, r: f64) -> f64 {
        r.clamp(-self.k, self.k)
    }
    fn weight(&self, r: f64) -> f64 {
        if r == 0.0 {
            1.0
        } else {
            (self.k / r.abs()).min(1.0)
        }
    }
    fn psi_prime(&self, r: f64) -> f64 {
        if r.abs() <= self.k {
            1.0
        } else {
            0.0
        }
    }
    fn tuning(&self) -> f64 {
        self.k
    }
    fn is_redescending(&self) -> bool {
        false
    }
    fn rho_sup(&self) -> Option<f64> {
        None
    }
}
