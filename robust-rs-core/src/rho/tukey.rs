//! Tukey's biweight (bisquare): a redescending loss. `ρ` bounded ⇒ non-convex.

use crate::error::RobustError;
use crate::rho::RhoFunction;

/// Tukey's biweight with tuning `c` (default `4.685`, ≈ 95% efficiency as an
/// M-step; use `c ≈ 1.547` with `δ = 0.5` for a 50%-breakdown S-scale).
#[derive(Debug, Clone, Copy)]
pub struct TukeyBiweight {
    c: f64,
}

impl TukeyBiweight {
    /// Create a biweight loss with the given positive tuning constant.
    pub fn new(c: f64) -> Result<Self, RobustError> {
        if c.is_finite() && c > 0.0 {
            Ok(Self { c })
        } else {
            Err(RobustError::InvalidTuning { value: c })
        }
    }
}

impl Default for TukeyBiweight {
    /// `c = 4.685`.
    fn default() -> Self {
        Self { c: 4.685 }
    }
}

impl RhoFunction for TukeyBiweight {
    fn rho(&self, r: f64) -> f64 {
        let c2_6 = self.c * self.c / 6.0;
        if r.abs() <= self.c {
            let u2 = (r / self.c).powi(2);
            c2_6 * (1.0 - (1.0 - u2).powi(3))
        } else {
            c2_6
        }
    }
    fn psi(&self, r: f64) -> f64 {
        if r.abs() <= self.c {
            let u2 = (r / self.c).powi(2);
            r * (1.0 - u2).powi(2)
        } else {
            0.0
        }
    }
    fn weight(&self, r: f64) -> f64 {
        if r.abs() <= self.c {
            let u2 = (r / self.c).powi(2);
            (1.0 - u2).powi(2)
        } else {
            0.0
        }
    }
    fn psi_prime(&self, r: f64) -> f64 {
        if r.abs() <= self.c {
            let u2 = (r / self.c).powi(2);
            (1.0 - u2) * (1.0 - 5.0 * u2)
        } else {
            0.0
        }
    }
    fn tuning(&self) -> f64 {
        self.c
    }
    fn is_redescending(&self) -> bool {
        true
    }
    fn rho_sup(&self) -> Option<f64> {
        Some(self.c * self.c / 6.0)
    }
}
