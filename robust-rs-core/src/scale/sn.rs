//! Rousseeuw & Croux's `Sn` robust scale.

use crate::error::RobustError;
use crate::scale::ScaleEstimator;
use crate::types::Scale;

/// `Sn = c · dₙ · med_i { med_j |rᵢ − rⱼ| }`: a low-median over points of the
/// per-point high-median of the pairwise absolute differences. The constant
/// `c = 1.1926` gives consistency for σ at the Gaussian; `dₙ` is a finite-sample
/// correction (Croux & Rousseeuw 1992).
///
/// Like `Qn` it needs no location reference and has a 50% breakdown point, with
/// ≈ 58% Gaussian efficiency; it is often preferred for its simpler influence
/// function. Evaluated in `O(n²)` time / `O(n)` space here (the `O(n log n)`
/// algorithm is a possible follow-up).
#[derive(Debug, Clone, Copy)]
pub struct Sn {
    /// Asymptotic consistency constant (default `1.1926`).
    pub consistency: f64,
    /// Apply the finite-sample correction `dₙ` (default `true`).
    pub finite_sample_correction: bool,
}

impl Default for Sn {
    fn default() -> Self {
        Self {
            consistency: 1.1926,
            finite_sample_correction: true,
        }
    }
}

impl ScaleEstimator for Sn {
    fn scale(&self, residuals: &[f64]) -> Result<Scale, RobustError> {
        let n = residuals.len();
        if n < 2 {
            return Err(RobustError::InsufficientData { needed: 2, got: n });
        }

        let mut inner = vec![0.0_f64; n];
        let mut a = vec![0.0_f64; n];
        for (i, &ri) in residuals.iter().enumerate() {
            for (j, &rj) in residuals.iter().enumerate() {
                inner[j] = (ri - rj).abs();
            }
            // high median of the n differences (includes the 0 at j = i).
            let (_, hm, _) = inner.select_nth_unstable_by(n / 2, f64::total_cmp);
            a[i] = *hm;
        }
        // low median of the per-point high-medians.
        let (_, lm, _) = a.select_nth_unstable_by((n - 1) / 2, f64::total_cmp);
        let med = *lm;

        let dn = if self.finite_sample_correction {
            sn_correction(n)
        } else {
            1.0
        };
        let s = self.consistency * dn * med;
        if !s.is_finite() || s <= 0.0 {
            return Err(RobustError::DegenerateScale);
        }
        Scale::new(s)
    }
}

/// Finite-sample correction `dₙ` for `Sn` (Croux & Rousseeuw 1992): tabulated
/// for `n ≤ 9`, else `1` (odd) or `n/(n−0.9)` (even); `→ 1` as `n → ∞`.
fn sn_correction(n: usize) -> f64 {
    const SMALL: [f64; 8] = [0.743, 1.851, 0.954, 1.351, 0.993, 1.198, 1.005, 1.131];
    if n <= 9 {
        SMALL[n - 2]
    } else if n % 2 == 1 {
        1.0
    } else {
        n as f64 / (n as f64 - 0.9)
    }
}
