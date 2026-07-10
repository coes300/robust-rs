//! Trimmed and Winsorized means.

use robust_rs_core::error::RobustError;

/// Validate `alpha ∈ [0, 0.5)` and `data` non-empty; return the data sorted
/// ascending with the per-tail trim count `g = ⌊alpha·n⌋`.
fn sort_and_trim_count(data: &[f64], alpha: f64) -> Result<(Vec<f64>, usize), RobustError> {
    let n = data.len();
    if n == 0 {
        return Err(RobustError::InsufficientData { needed: 1, got: 0 });
    }
    if !(0.0..0.5).contains(&alpha) {
        // also rejects NaN and +∞; neither lies in [0, 0.5)
        return Err(RobustError::InvalidTuning { value: alpha });
    }
    let g = (alpha * n as f64).floor() as usize; // per tail; 2g < n since alpha < 0.5
    let mut buf = data.to_vec();
    buf.sort_unstable_by(f64::total_cmp);
    Ok((buf, g))
}

/// The `alpha`-trimmed mean (drop the lowest and highest `alpha` fraction).
pub fn trimmed_mean(data: &[f64], alpha: f64) -> Result<f64, RobustError> {
    let (buf, g) = sort_and_trim_count(data, alpha)?;
    let n = buf.len();
    let kept = &buf[g..n - g]; // n − 2g ≥ 1
    Ok(kept.iter().sum::<f64>() / kept.len() as f64)
}

/// The `alpha`-Winsorized mean (clamp the tails instead of dropping them).
pub fn winsorized_mean(data: &[f64], alpha: f64) -> Result<f64, RobustError> {
    let (buf, g) = sort_and_trim_count(data, alpha)?;
    let n = buf.len();
    let lo = buf[g]; // smallest retained value
    let hi = buf[n - 1 - g]; // largest retained value
                             // g low tails clamped up to `lo`, g high tails clamped down to `hi`,
                             // the n − 2g middle values kept as-is; average all n.
    let middle: f64 = buf[g..n - g].iter().sum();
    Ok((g as f64 * lo + middle + g as f64 * hi) / n as f64)
}
