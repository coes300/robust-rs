//! The `RobustEstimator` trait (the sampling-theory result surface) and the
//! shared regression fit type.

use faer::linalg::solvers::DenseSolveCore;
use ndarray::{Array1, Array2};
use robust_rs_core::rho::RhoFunction;
use robust_rs_core::types::Scale;

const QUAD: usize = 128;

/// A fitted robust regression, carrying enough state to report the theory.
pub struct RegressionFit {
    /// Estimated coefficients.
    pub coefficients: Array1<f64>,
    /// Estimated residual scale.
    pub scale: Scale,
    /// Residuals `y − Xβ̂`.
    pub residuals: Array1<f64>,
    /// Final IRLS weights.
    pub weights: Array1<f64>,
    /// The loss used, retained so influence/efficiency can be reported. This is
    /// the estimator's **final-stage** loss (e.g. the M-step biweight for MM),
    /// so the reported efficiency and covariance describe the delivered fit.
    pub rho: Box<dyn RhoFunction>,
    /// Asymptotic breakdown point ε*, carried as data set by the producing
    /// estimator, *not* derived from `rho`, since ρ alone does not determine an
    /// estimator's breakdown. Plain M-regression sets `0.0` (leverage); S/MM set
    /// `min(δ/ρ_sup, 1 − δ/ρ_sup)`.
    pub breakdown_point: f64,
}

/// Quantities every fitted robust estimator can report.
pub trait RobustEstimator {
    /// Estimated coefficients.
    fn coefficients(&self) -> &Array1<f64>;
    /// Estimated residual scale.
    fn scale(&self) -> Scale;
    /// The influence function `x ↦ ψ(x)/E[ψ']`.
    fn influence_function(&self) -> Box<dyn Fn(f64) -> f64 + '_>;
    /// Asymptotic variance `E[ψ²]/(E[ψ'])²`.
    fn asymptotic_variance(&self) -> f64;
    /// Efficiency relative to the Gaussian MLE.
    fn gaussian_efficiency(&self) -> f64;
    /// Approximate coefficient covariance `ŝ²·V·(XᵀX)⁻¹`.
    fn coef_covariance(&self, x: &Array2<f64>) -> Array2<f64>;
    /// Asymptotic breakdown point.
    fn breakdown_point(&self) -> f64;
}

impl RobustEstimator for RegressionFit {
    fn coefficients(&self) -> &Array1<f64> {
        &self.coefficients
    }
    fn scale(&self) -> Scale {
        self.scale
    }
    fn influence_function(&self) -> Box<dyn Fn(f64) -> f64 + '_> {
        Box::new(robust_rs_core::theory::influence_function(&*self.rho, QUAD))
    }
    fn asymptotic_variance(&self) -> f64 {
        robust_rs_core::theory::asymptotic_variance(&*self.rho, QUAD)
    }
    fn gaussian_efficiency(&self) -> f64 {
        robust_rs_core::theory::gaussian_efficiency(&*self.rho, QUAD)
    }
    fn coef_covariance(&self, x: &Array2<f64>) -> Array2<f64> {
        let p = x.ncols();
        let s = self.scale.get();
        let factor = s * s * self.asymptotic_variance();

        // (XᵀX)⁻¹ via faer LU. XᵀX is SPD and full-rank after a successful fit.
        let xtx = x.t().dot(x);
        let a = faer::Mat::from_fn(p, p, |i, j| xtx[[i, j]]);
        let inv = a.partial_piv_lu().inverse();

        Array2::from_shape_fn((p, p), |(i, j)| inv[(i, j)] * factor)
    }
    fn breakdown_point(&self) -> f64 {
        self.breakdown_point
    }
}
