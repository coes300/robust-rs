//! Andrews' sine loss: a smooth, *hard* redescender (`ψ` is exactly zero past `cπ`).

use crate::error::RobustError;
use crate::rho::RhoFunction;
use std::f64::consts::PI;

/// Andrews' sine-wave loss with tuning `c` (default `1.339`, ≈ 95% Gaussian
/// efficiency).
/// `ψ(r) = c·sin(r/c)` for `|r| ≤ cπ` and `0` beyond; `ρ(r) = c²·(1 − cos(r/c))`
/// for `|r| ≤ cπ`, else `2c²`.
///
/// Parameterized with the leading `c` so that `ψ'(0) = 1` and `w(0) = 1`,
/// matching the crate's other losses (the weight reduces to the sinc
/// `sin(r/c)/(r/c)`). Like the biweight it redescends all the way to zero at a
/// finite point, so `rho_sup = Some(2c²)`.
#[derive(Debug, Clone, Copy)]
pub struct Andrews {
    c: f64,
}

impl Andrews {
    /// Create an Andrews loss with the given positive tuning constant.
    pub fn new(c: f64) -> Result<Self, RobustError> {
        if c.is_finite() && c > 0.0 {
            Ok(Self { c })
        } else {
            Err(RobustError::InvalidTuning { value: c })
        }
    }
}

impl Default for Andrews {
    /// `c = 1.339`.
    fn default() -> Self {
        Self { c: 1.339 }
    }
}

impl RhoFunction for Andrews {
    fn rho(&self, r: f64) -> f64 {
        let c2 = self.c * self.c;
        if r.abs() <= self.c * PI {
            c2 * (1.0 - (r / self.c).cos())
        } else {
            2.0 * c2
        }
    }
    fn psi(&self, r: f64) -> f64 {
        if r.abs() <= self.c * PI {
            self.c * (r / self.c).sin()
        } else {
            0.0
        }
    }
    fn weight(&self, r: f64) -> f64 {
        if r == 0.0 {
            1.0 // limit of sin(r/c)/(r/c)
        } else if r.abs() <= self.c * PI {
            self.c * (r / self.c).sin() / r
        } else {
            0.0
        }
    }
    fn psi_prime(&self, r: f64) -> f64 {
        if r.abs() <= self.c * PI {
            (r / self.c).cos()
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
        Some(2.0 * self.c * self.c)
    }
}
