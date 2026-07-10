//! Rousseeuw & Croux's `Qn` robust scale.

use crate::error::RobustError;
use crate::scale::ScaleEstimator;
use crate::types::Scale;

/// `Qn = c · dₙ · {|rᵢ − rⱼ| : i < j}₍ₖ₎`, the `k`-th order statistic of the
/// pairwise absolute differences with `k = C(⌊n/2⌋+1, 2)` (≈ the ¼-quantile).
/// The constant `c = 2.2219 = 1/(√2·Φ⁻¹(⅝))` gives consistency for σ at the
/// Gaussian; `dₙ` is a finite-sample correction (Croux & Rousseeuw 1992).
///
/// Uses no location step and stays smooth in the data, with ≈ 82% Gaussian
/// efficiency and a 50% breakdown point (versus the MAD's ≈ 37%).
///
/// Evaluated directly over all pairs in `O(n²)` time / `O(n²)` space; Rousseeuw
/// & Croux's `O(n log n)` algorithm is a possible follow-up (the datasets here
/// are small).
#[derive(Debug, Clone, Copy)]
pub struct Qn {
    /// Asymptotic consistency constant (default `2.2219`).
    pub consistency: f64,
    /// Apply the finite-sample correction `dₙ` (default `true`).
    pub finite_sample_correction: bool,
}

impl Default for Qn {
    fn default() -> Self {
        Self {
            consistency: 2.2219,
            finite_sample_correction: true,
        }
    }
}

impl ScaleEstimator for Qn {
    fn scale(&self, residuals: &[f64]) -> Result<Scale, RobustError> {
        let n = residuals.len();
        if n < 2 {
            return Err(RobustError::InsufficientData { needed: 2, got: n });
        }

        let mut diffs = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                diffs.push((residuals[i] - residuals[j]).abs());
            }
        }

        // k-th smallest (1-indexed), k = C(h, 2) with h = ⌊n/2⌋ + 1.
        let h = n / 2 + 1;
        let k = h * (h - 1) / 2;
        let (_, kth, _) = diffs.select_nth_unstable_by(k - 1, f64::total_cmp);
        let kth = *kth;

        let dn = if self.finite_sample_correction {
            qn_correction(n)
        } else {
            1.0
        };
        let s = self.consistency * dn * kth;
        if !s.is_finite() || s <= 0.0 {
            return Err(RobustError::DegenerateScale);
        }
        Scale::new(s)
    }
}

/// Finite-sample correction `dₙ` for `Qn` (Croux & Rousseeuw 1992): tabulated
/// for `n ≤ 9`, else `n/(n+3.8)` (even) or `n/(n+1.4)` (odd); `→ 1` as `n → ∞`.
fn qn_correction(n: usize) -> f64 {
    const SMALL: [f64; 8] = [0.399, 0.994, 0.512, 0.844, 0.611, 0.857, 0.669, 0.872];
    if n <= 9 {
        SMALL[n - 2]
    } else if n % 2 == 0 {
        n as f64 / (n as f64 + 3.8)
    } else {
        n as f64 / (n as f64 + 1.4)
    }
}
