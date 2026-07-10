//! Regression M-estimation by iteratively reweighted least squares.

use ndarray::{Array1, Array2};
use robust_rs_core::error::RobustError;
use robust_rs_core::rho::RhoFunction;
use robust_rs_core::scale::ScaleEstimator;
use robust_rs_core::solver::Control;

use crate::estimator::RegressionFit;

/// A configured regression M-estimator: a loss, a scale estimator and solver
/// controls. Call [`MEstimator::fit`] to fit it to data.
pub struct MEstimator<R, S> {
    /// The robust loss.
    pub rho: R,
    /// The scale estimator used to standardize residuals.
    pub scale: S,
    /// Convergence controls.
    pub control: Control,
}

impl<R, S> MEstimator<R, S>
where
    R: RhoFunction + Clone + 'static,
    S: ScaleEstimator,
{
    /// Create an M-estimator with default solver controls.
    pub fn new(rho: R, scale: S) -> Self {
        Self {
            rho,
            scale,
            control: Control::default(),
        }
    }

    /// Fit by IRLS: standardize residuals by the scale, form the weights
    /// `wᵢ = ρ.weight(rᵢ/s)`, solve the weighted least squares, repeat.
    pub fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<RegressionFit, RobustError> {
        use crate::wls::weighted_least_squares;

        let n = x.nrows();

        // Initial fit: OLS (unit weights). Dim / rank errors propagate from WLS.
        let ones = Array1::from_elem(n, 1.0);
        let mut beta = weighted_least_squares(x, y, &ones)?;
        let mut resid = y - &x.dot(&beta);

        // Initial scale from the OLS residuals, held fixed through the loop.
        let scale_est = self.scale.scale(resid.as_slice().expect("contiguous"))?;
        let s: f64 = scale_est.get();

        let mut converged = false;
        for _ in 0..self.control.max_iter {
            let w = resid.mapv(|r| self.rho.weight(r / s));
            let next = weighted_least_squares(x, y, &w)?;

            // relative L2 change in coefficients (matches Control::tol's semantics)
            let diff = &next - &beta;
            let rel = diff.dot(&diff).sqrt() / (next.dot(&next).sqrt() + 1e-12);

            beta = next;
            resid = y - &x.dot(&beta);
            if rel <= self.control.tol {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(RobustError::NonConvergence {
                iters: self.control.max_iter,
            });
        }

        // Weights evaluated at the converged residuals (≈ the last solve's weights).
        let weights = resid.mapv(|r| self.rho.weight(r / s));

        Ok(RegressionFit {
            coefficients: beta,
            scale: scale_est,
            residuals: resid,
            weights,
            rho: Box::new(self.rho.clone()), // R: Clone + 'static → Box<dyn RhoFunction>
            breakdown_point: 0.0, // plain regression M-estimation: leverage ⇒ 0 breakdown
        })
    }
}
