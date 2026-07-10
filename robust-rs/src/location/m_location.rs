//! M-estimation of location by IRLS (scalar; no linear algebra needed).

use robust_rs_core::error::RobustError;
use robust_rs_core::rho::RhoFunction;
use robust_rs_core::scale::ScaleEstimator;
use robust_rs_core::solver::Control;
use robust_rs_core::types::Scale;

/// A fitted location estimate.
#[derive(Debug, Clone, Copy)]
pub struct LocationFit {
    /// The location estimate `θ̂`.
    pub estimate: f64,
    /// The scale used to standardize residuals.
    pub scale: Scale,
    /// Iterations performed.
    pub iters: usize,
}

/// M-estimate of location: iterate `θ ← Σ wᵢ xᵢ / Σ wᵢ` with
/// `wᵢ = ρ.weight((xᵢ − θ)/s)` until convergence.
pub fn m_location(
    data: &[f64],
    rho: &dyn RhoFunction,
    scale: &dyn ScaleEstimator,
    ctrl: &Control,
) -> Result<LocationFit, RobustError> {
    if data.is_empty() {
        return Err(RobustError::InsufficientData { needed: 1, got: 0 });
    }

    // Scale computed once (about the data's median) and held fixed.
    // scale.scale already returns DegenerateScale if it collapses to zero.
    let scale_est = scale.scale(data)?;
    let s: f64 = scale_est.get();

    let mut buf = data.to_vec();
    let mut theta = median(&mut buf); // init at the median

    for i in 1..=ctrl.max_iter {
        // wᵢ = ρ.weight((xᵢ − θ)/s);  θ ← Σ wᵢ xᵢ / Σ wᵢ
        let (mut sw, mut swx) = (0.0_f64, 0.0_f64);
        for &x in data {
            let w = rho.weight((x - theta) / s);
            sw += w;
            swx += w * x;
        }
        let next = swx / sw;
        if !next.is_finite() {
            // 0/0 (a redescending ρ rejected every point) or non-finite data.
            return Err(RobustError::SingularDesign);
        }
        // tol read as relative-to-scale: θ can legitimately be ≈0, which makes a
        // relative-to-θ test degenerate, whereas s is a natural non-zero yardstick.
        if (next - theta).abs() <= ctrl.tol * s {
            return Ok(LocationFit {
                estimate: next,
                scale: scale_est,
                iters: i,
            });
        }
        theta = next;
    }
    Err(RobustError::NonConvergence {
        iters: ctrl.max_iter,
    })
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
