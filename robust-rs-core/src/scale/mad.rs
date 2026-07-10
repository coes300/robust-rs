//! Median absolute deviation scale.

use crate::error::RobustError;
use crate::scale::ScaleEstimator;
use crate::types::Scale;

/// MAD: `s = 1.4826 · median(|rᵢ − median(r)|)`. The constant `1.4826 = 1/Φ⁻¹(¾)`
/// makes it consistent for σ at the Gaussian model.
#[derive(Debug, Clone, Copy)]
pub struct Mad {
    /// Consistency constant (default `1.4826`).
    pub consistency: f64,
}

impl Default for Mad {
    fn default() -> Self {
        Self {
            consistency: 1.482_602_218_505_602,
        }
    }
}

impl ScaleEstimator for Mad {
    fn scale(&self, residuals: &[f64]) -> Result<Scale, RobustError> {
        if residuals.is_empty() {
            return Err(RobustError::InsufficientData { needed: 1, got: 0 });
        }

        // Copy so the caller's slice is untouched; we sort in place twice.
        let mut buf = residuals.to_vec();

        let center = median(&mut buf);
        for x in buf.iter_mut() {
            *x = (*x - center).abs(); // reuse buf; order no longer matters
        }
        let mad = median(&mut buf);

        let s = mad * self.consistency;
        // Broadened from the literal "if 0": also reject NaN/±inf. A majority of
        // non-finite residuals reaches the median and yields a non-finite scale
        // that `== 0.0` would miss, leaking out as Ok(Scale(NaN)).
        if !s.is_finite() || s <= 0.0 {
            return Err(RobustError::DegenerateScale);
        }
        Scale::new(s)
    }
}

/// Median via in-place total-order sort. `total_cmp` never panics on NaN and
/// for even `n` we average the two central order statistics, which is the
/// definition the `1.4826` constant is derived for.
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
