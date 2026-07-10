//! The Cauchy (Lorentzian) loss: a soft redescender whose influence never quite
//! returns to zero.

use crate::error::RobustError;
use crate::rho::RhoFunction;

/// The Cauchy (a.k.a. Lorentzian) loss with tuning `c` (default `2.3849`,
/// ≈ 95% Gaussian efficiency).
/// `ρ(r) = (c²/2)·ln(1 + (r/c)²)`, `ψ(r) = r / (1 + (r/c)²)`.
///
/// `ψ` is non-monotone (it peaks at `r = c` then decays toward zero) so `ρ` is
/// non-convex and the estimator needs a good starting point (`is_redescending`
/// is `true`). But because `ρ` is *unbounded* the weights never reach exactly
/// zero, so no observation is ever fully rejected: a *soft* redescender. Hence
/// `rho_sup` is `None` and it is unsuitable as an S-scale loss.
#[derive(Debug, Clone, Copy)]
pub struct Cauchy {
    c: f64,
}

impl Cauchy {
    /// Create a Cauchy loss with the given positive tuning constant.
    pub fn new(c: f64) -> Result<Self, RobustError> {
        if c.is_finite() && c > 0.0 {
            Ok(Self { c })
        } else {
            Err(RobustError::InvalidTuning { value: c })
        }
    }
}

impl Default for Cauchy {
    /// `c = 2.3849`.
    fn default() -> Self {
        Self { c: 2.3849 }
    }
}

impl RhoFunction for Cauchy {
    fn rho(&self, r: f64) -> f64 {
        (self.c * self.c / 2.0) * (1.0 + (r / self.c).powi(2)).ln()
    }
    fn psi(&self, r: f64) -> f64 {
        r / (1.0 + (r / self.c).powi(2))
    }
    fn weight(&self, r: f64) -> f64 {
        1.0 / (1.0 + (r / self.c).powi(2))
    }
    fn psi_prime(&self, r: f64) -> f64 {
        let u2 = (r / self.c).powi(2);
        (1.0 - u2) / (1.0 + u2).powi(2)
    }
    fn tuning(&self) -> f64 {
        self.c
    }
    fn is_redescending(&self) -> bool {
        true
    }
    fn rho_sup(&self) -> Option<f64> {
        None
    }
}
