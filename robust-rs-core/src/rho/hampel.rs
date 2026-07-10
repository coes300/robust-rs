//! Hampel's three-part redescending loss.

use crate::error::RobustError;
use crate::rho::RhoFunction;

/// Hampel's piecewise-linear-`ψ` redescender, parameterized by three positive
/// break-points `a ≤ b < c` (default `(2.0, 4.0, 8.0)`, per Ripley's `MASS`).
///
/// For `r ≥ 0` (odd-extended to `r < 0`):
/// `ψ(r) = r` on `[0, a]`, `= a` on `[a, b]`, descends linearly to
/// `a·(c − r)/(c − b)` on `[b, c]` and `= 0` beyond `c`. Unlike the biweight it
/// keeps *full* influence on small residuals (`ψ(r) = r` near 0) before the
/// hard cutoff, trading a little efficiency for a gentler descent.
/// `rho_sup = (a/2)(b + c − a)`.
///
/// The `RhoFunction` trait exposes a single scalar `tuning()`; for Hampel that
/// is `a`. Use [`Hampel::constants`] for all three.
#[derive(Debug, Clone, Copy)]
pub struct Hampel {
    a: f64,
    b: f64,
    c: f64,
}

impl Hampel {
    /// Create a Hampel loss with break-points `0 < a ≤ b < c` (all finite).
    pub fn new(a: f64, b: f64, c: f64) -> Result<Self, RobustError> {
        if a.is_finite() && b.is_finite() && c.is_finite() && 0.0 < a && a <= b && b < c {
            Ok(Self { a, b, c })
        } else {
            Err(RobustError::InvalidTuning { value: a })
        }
    }

    /// The three break-points `(a, b, c)`.
    pub fn constants(&self) -> (f64, f64, f64) {
        (self.a, self.b, self.c)
    }
}

impl Default for Hampel {
    /// `(a, b, c) = (2.0, 4.0, 8.0)`.
    fn default() -> Self {
        Self {
            a: 2.0,
            b: 4.0,
            c: 8.0,
        }
    }
}

impl RhoFunction for Hampel {
    fn rho(&self, r: f64) -> f64 {
        let (a, b, c) = (self.a, self.b, self.c);
        let x = r.abs();
        if x <= a {
            x * x / 2.0
        } else if x <= b {
            a * x - a * a / 2.0
        } else if x <= c {
            // ρ(b) + ∫_b^x a(c − t)/(c − b) dt
            let rho_b = a * b - a * a / 2.0;
            rho_b + a / (c - b) * (c * (x - b) - (x * x - b * b) / 2.0)
        } else {
            0.5 * a * (b + c - a) // ρ(∞) = ρ(c)
        }
    }
    fn psi(&self, r: f64) -> f64 {
        let (a, b, c) = (self.a, self.b, self.c);
        let x = r.abs();
        let mag = if x <= a {
            x
        } else if x <= b {
            a
        } else if x <= c {
            a * (c - x) / (c - b)
        } else {
            0.0
        };
        mag * r.signum()
    }
    fn weight(&self, r: f64) -> f64 {
        let (a, b, c) = (self.a, self.b, self.c);
        let x = r.abs();
        // `x <= a` covers `x == 0` (a > 0 ⇒ weight 1); division only on x > a.
        if x <= a {
            1.0
        } else if x <= b {
            a / x
        } else if x <= c {
            a * (c - x) / ((c - b) * x)
        } else {
            0.0
        }
    }
    fn psi_prime(&self, r: f64) -> f64 {
        let (a, b, c) = (self.a, self.b, self.c);
        let x = r.abs();
        if x < a {
            1.0
        } else if x < b {
            0.0
        } else if x < c {
            -a / (c - b)
        } else {
            0.0
        }
    }
    fn tuning(&self) -> f64 {
        self.a
    }
    fn is_redescending(&self) -> bool {
        true
    }
    fn rho_sup(&self) -> Option<f64> {
        Some(0.5 * self.a * (self.b + self.c - self.a))
    }
}
