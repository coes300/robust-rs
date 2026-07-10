//! Welsch's (Leclerc's) loss: a smooth, bounded redescender with Gaussian-shaped
//! weights.

use crate::error::RobustError;
use crate::rho::RhoFunction;

/// Welsch's loss with tuning `c` (default `2.9846`, ≈ 95% Gaussian efficiency).
/// `ρ(r) = (c²/2)·[1 − exp(−(r/c)²)]`, `ψ(r) = r·exp(−(r/c)²)`,
/// `w(r) = exp(−(r/c)²)`.
///
/// Smoothly redescending (`ψ → 0`) with a bounded `ρ`, so `rho_sup = Some(c²/2)`
/// and it can seed an S-scale. The exponential weights fall off faster than the
/// biweight's, never reaching exactly zero at finite `r`.
#[derive(Debug, Clone, Copy)]
pub struct Welsch {
    c: f64,
}

impl Welsch {
    /// Create a Welsch loss with the given positive tuning constant.
    pub fn new(c: f64) -> Result<Self, RobustError> {
        if c.is_finite() && c > 0.0 {
            Ok(Self { c })
        } else {
            Err(RobustError::InvalidTuning { value: c })
        }
    }
}

impl Default for Welsch {
    /// `c = 2.9846`.
    fn default() -> Self {
        Self { c: 2.9846 }
    }
}

impl RhoFunction for Welsch {
    fn rho(&self, r: f64) -> f64 {
        (self.c * self.c / 2.0) * (1.0 - (-(r / self.c).powi(2)).exp())
    }
    fn psi(&self, r: f64) -> f64 {
        r * (-(r / self.c).powi(2)).exp()
    }
    fn weight(&self, r: f64) -> f64 {
        (-(r / self.c).powi(2)).exp()
    }
    fn psi_prime(&self, r: f64) -> f64 {
        let u2 = (r / self.c).powi(2);
        (1.0 - 2.0 * u2) * (-u2).exp()
    }
    fn tuning(&self) -> f64 {
        self.c
    }
    fn is_redescending(&self) -> bool {
        true
    }
    fn rho_sup(&self) -> Option<f64> {
        Some(self.c * self.c / 2.0)
    }
}
