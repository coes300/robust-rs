//! The S-estimator of scale.

use crate::error::RobustError;
use crate::rho::RhoFunction;
use crate::scale::ScaleEstimator;
use crate::solver::Control;
use crate::theory::gauss_hermite;
use crate::types::Scale;

/// The M-estimate of scale ("S-scale"): the `s > 0` solving
/// `(1/n) Σ ρ(rᵢ/s) = δ`.
///
/// With a bounded, redescending `ρ` and `δ = ρ.rho_sup()/2` this is the
/// 50%-breakdown scale that S- and MM-regression minimize; with `δ = E_Φ[ρ]` it
/// is Fisher-consistent for σ at the Gaussian (for the biweight the S-step
/// tuning `c ≈ 1.547` makes the two coincide). Generic over the loss, but the
/// loss **must** have bounded `ρ` (`rho_sup().is_some()`): an unbounded `ρ`
/// gives zero breakdown (a single `rᵢ → ∞` drives `s` without limit) so
/// `new` rejects it with `RobustError::UnboundedLoss`.
///
/// Solved by the standard fixed-point iteration
/// `s ← s·√(mean_i ρ(rᵢ/s) / δ)` from a MAD start; `g(s) = (1/n)Σρ(rᵢ/s)` is
/// monotone decreasing, so the root is unique.
#[derive(Debug, Clone, Copy)]
pub struct SScale<R: RhoFunction> {
    rho: R,
    delta: f64,
    control: Control,
}

impl<R: RhoFunction> SScale<R> {
    /// Create an S-scale for loss `rho` with consistency target `0 < delta < sup ρ`.
    ///
    /// Rejects an unbounded loss (`rho_sup() == None`) with
    /// `RobustError::UnboundedLoss`, since it cannot define a high-breakdown
    /// scale and a `delta` outside `(0, sup ρ)` (for which the scale equation
    /// has no root) with `RobustError::InvalidTuning`.
    pub fn new(rho: R, delta: f64) -> Result<Self, RobustError> {
        // Boundedness is the S-scale's defining precondition, not an
        // optimization: with unbounded ρ the equation still has a unique root,
        // but that root has zero breakdown, the opposite of the point.
        let sup = rho.rho_sup().ok_or(RobustError::UnboundedLoss)?;
        if delta.is_finite() && delta > 0.0 && delta < sup {
            Ok(Self {
                rho,
                delta,
                control: Control::default(),
            })
        } else {
            Err(RobustError::InvalidTuning { value: delta })
        }
    }

    /// Create a Fisher-consistent S-scale by setting `δ = E_Φ[ρ]`, evaluated by
    /// Gauss–Hermite quadrature with `quad_points` nodes.
    pub fn fisher_consistent(rho: R, quad_points: usize) -> Result<Self, RobustError> {
        let (nodes, weights) = gauss_hermite(quad_points);
        let delta = nodes
            .iter()
            .zip(&weights)
            .map(|(&x, &w)| w * rho.rho(x))
            .sum();
        Self::new(rho, delta)
    }

    /// Override the default convergence control.
    pub fn with_control(mut self, control: Control) -> Self {
        self.control = control;
        self
    }

    /// The consistency target `δ`.
    pub fn delta(&self) -> f64 {
        self.delta
    }
}

impl<R: RhoFunction> ScaleEstimator for SScale<R> {
    fn scale(&self, residuals: &[f64]) -> Result<Scale, RobustError> {
        let n = residuals.len();
        if n == 0 {
            return Err(RobustError::InsufficientData { needed: 1, got: 0 });
        }

        // MAD start (about the median); any positive scale seeds the fixed point.
        let mut buf = residuals.to_vec();
        let med = median(&mut buf);
        for (b, &r) in buf.iter_mut().zip(residuals) {
            *b = (r - med).abs();
        }
        let mut s = 1.482_602_218_505_602 * median(&mut buf);
        if !(s.is_finite() && s > 0.0) {
            return Err(RobustError::DegenerateScale); // ≥ half the residuals tied
        }

        // Fixed point on (1/n) Σ ρ(rᵢ/s) = δ:  s ← s·√(mean ρ(rᵢ/s) / δ).
        for _ in 0..self.control.max_iter {
            let mean_rho = residuals.iter().map(|&r| self.rho.rho(r / s)).sum::<f64>() / n as f64;
            let s_next = s * (mean_rho / self.delta).sqrt();
            if !(s_next.is_finite() && s_next > 0.0) {
                return Err(RobustError::DegenerateScale);
            }
            if (s_next - s).abs() <= self.control.tol * s {
                return Scale::new(s_next);
            }
            s = s_next;
        }
        Err(RobustError::NonConvergence {
            iters: self.control.max_iter,
        })
    }
}

/// Median via in-place total-order sort.
fn median(v: &mut [f64]) -> f64 {
    v.sort_unstable_by(f64::total_cmp);
    let n = v.len();
    let mid = n / 2;
    if n % 2 == 1 {
        v[mid]
    } else {
        0.5 * (v[mid - 1] + v[mid])
    }
}
