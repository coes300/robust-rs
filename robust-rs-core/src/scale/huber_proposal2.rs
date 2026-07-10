//! Huber's "Proposal 2" simultaneous location–scale estimate.

use crate::error::RobustError;
use crate::scale::ScaleEstimator;
use crate::types::Scale;

/// Huber's Proposal 2 scale: solve `(1/n) Σ ψ((rᵢ−μ)/s)² = β` (with
/// `β = E_Φ[ψ²]`) jointly with the location, by fixed-point iteration.
#[derive(Debug, Clone, Copy)]
pub struct HuberProposal2 {
    /// Tuning constant `k` for the ψ used in the scale equation.
    pub k: f64,
}

impl Default for HuberProposal2 {
    fn default() -> Self {
        Self { k: 1.345 }
    }
}

impl ScaleEstimator for HuberProposal2 {
    fn scale(&self, residuals: &[f64]) -> Result<Scale, RobustError> {
        let n = residuals.len();
        if n == 0 {
            return Err(RobustError::InsufficientData { needed: 1, got: 0 });
        }
        let k = self.k;
        if !(k.is_finite() && k > 0.0) {
            return Err(RobustError::InvalidTuning { value: k }); // field is public/unvalidated
        }

        // Consistency constant β = E_Φ[ψ_k(Z)²], in CLOSED FORM. Do not compute
        // this by Gauss–Hermite quadrature: ψ² has a kink at ±k, so GH converges
        // at ~O(1/n) and oscillates (~0.3% error even at 128 nodes), which would
        // bias every scale it produces. The closed form is exact.
        let phi_k = (-0.5 * k * k).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let cdf_k = normal_cdf(k);
        let beta = (2.0 * cdf_k - 1.0 - 2.0 * k * phi_k) + 2.0 * k * k * (1.0 - cdf_k);

        // Robust start: median and MAD.
        let mut buf = residuals.to_vec();
        let mut mu = median(&mut buf);
        for (b, &r) in buf.iter_mut().zip(residuals) {
            *b = (r - mu).abs();
        }
        let mut s = 1.482_602_218_505_602 * median(&mut buf);
        if !(s.is_finite() && s > 0.0) {
            return Err(RobustError::DegenerateScale); // ≥ half the residuals tied
        }

        // No `Control` is threaded through the ScaleEstimator trait, so mirror
        // the crate's default stopping rule.
        const TOL: f64 = 1e-8;
        const MAX_ITER: usize = 100;

        // Fixed point on the joint location–scale equations:
        //   winsorize each rᵢ to [μ−ks, μ+ks];  μ ← mean(winsorized);
        //   s ← sqrt( Σ (winsorized − μ)² / (n·β) ).
        for _ in 0..MAX_ITER {
            let (lo, hi) = (mu - k * s, mu + k * s);

            let mut sum = 0.0_f64;
            for &r in residuals {
                sum += r.clamp(lo, hi);
            }
            let mu1 = sum / n as f64;

            let mut ss = 0.0_f64;
            for &r in residuals {
                let d = r.clamp(lo, hi) - mu1;
                ss += d * d;
            }
            let s1 = (ss / (n as f64 * beta)).sqrt();

            if !(s1.is_finite() && s1 > 0.0) {
                return Err(RobustError::DegenerateScale);
            }
            if (mu - mu1).abs() < TOL * s && (s - s1).abs() < TOL * s {
                return Scale::new(s1);
            }
            mu = mu1;
            s = s1;
        }
        Err(RobustError::NonConvergence { iters: MAX_ITER })
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

/// Standard normal CDF via `libm::erf` (full double precision), keeping the core
/// crate's special-function surface dependency-light without `statrs`.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
}
