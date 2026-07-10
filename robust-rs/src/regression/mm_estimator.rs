//! MM-estimation of regression (Yohai 1987): a high-breakdown, high-efficiency
//! estimator (R's `lmrob` default).
//!
//! Two stages: a 50%-breakdown [`SEstimator`] supplies a robust initial fit
//! `β̃` and a consistent residual scale `ŝ`; then a redescending M-step
//! (biweight `c = 4.685`) runs IRLS from `β̃` with the scale **held fixed at
//! `ŝ`**, climbing to the nearest local optimum. The result keeps the
//! S-estimate's 50% breakdown while recovering ≈ 95% Gaussian efficiency.
//!
//! Because the fit stores its *final-stage* loss (the M-step biweight), the
//! theory surface reports the MM efficiency/covariance automatically; the
//! breakdown point is inherited from the S-stage rather than derived from the
//! loss.

use ndarray::{Array1, Array2};
use rand::Rng;
use robust_rs_core::error::RobustError;
use robust_rs_core::rho::{RhoFunction, TukeyBiweight};
use robust_rs_core::solver::Control;

use crate::estimator::RegressionFit;
use crate::regression::SEstimator;
use crate::wls::weighted_least_squares;

/// A configured regression MM-estimator: an S-stage (initial fit + scale) and a
/// redescending M-step loss. Reproducibility (seed / `fit_with_rng`) is
/// inherited entirely from the wrapped [`SEstimator`].
#[derive(Debug, Clone, Copy)]
pub struct MMEstimator<RS = TukeyBiweight, RM = TukeyBiweight> {
    /// The high-breakdown S-stage (default biweight `c = 1.547`).
    s: SEstimator<RS>,
    /// The high-efficiency M-step loss (default biweight `c = 4.685`).
    m_rho: RM,
    /// Convergence control for the M-step (the S-stage keeps its own).
    control: Control,
}

impl Default for MMEstimator<TukeyBiweight, TukeyBiweight> {
    /// The standard MM-estimator: 50%-breakdown biweight S-scale (`c = 1.547`)
    /// followed by a 95%-efficient biweight M-step (`c = 4.685`).
    fn default() -> Self {
        Self {
            s: SEstimator::default(),
            m_rho: TukeyBiweight::default(),
            control: Control::default(),
        }
    }
}

impl<RS, RM> MMEstimator<RS, RM>
where
    RS: RhoFunction + Clone + 'static,
    RM: RhoFunction + Clone + 'static,
{
    /// Create an MM-estimator from an explicit S-stage loss and M-step loss.
    pub fn new(s_rho: RS, m_rho: RM) -> Self {
        Self {
            s: SEstimator::new(s_rho),
            m_rho,
            control: Control::default(),
        }
    }

    /// Set the master RNG seed of the S-stage.
    pub fn seed(mut self, seed: u64) -> Self {
        self.s = self.s.seed(seed);
        self
    }

    /// Set the number of random elemental subsets used by the S-stage.
    pub fn n_subsamples(mut self, n: usize) -> Self {
        self.s = self.s.n_subsamples(n);
        self
    }

    /// Override the M-step convergence control.
    pub fn control(mut self, control: Control) -> Self {
        self.control = control;
        self
    }

    /// Fit reproducibly from the S-stage's configured seed.
    pub fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<RegressionFit, RobustError> {
        let s_fit = self.s.fit(x, y)?;
        self.m_step(x, y, s_fit)
    }

    /// Fit drawing the S-stage master seed from a caller-supplied generator.
    pub fn fit_with_rng<G: Rng>(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        rng: &mut G,
    ) -> Result<RegressionFit, RobustError> {
        let s_fit = self.s.fit_with_rng(x, y, rng)?;
        self.m_step(x, y, s_fit)
    }

    /// The redescending M-step: IRLS from the S-estimate with the scale fixed at
    /// the S-scale.
    fn m_step(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        s_fit: RegressionFit,
    ) -> Result<RegressionFit, RobustError> {
        let scale = s_fit.scale; // Scale: Copy
        let breakdown = s_fit.breakdown_point; // inherit the S-stage's breakdown
        let s = scale.get();
        let mut beta = s_fit.coefficients; // start at β̃
        let mut resid = y - &x.dot(&beta);

        let mut converged = false;
        for _ in 0..self.control.max_iter {
            let w = resid.mapv(|r| self.m_rho.weight(r / s));
            let beta_next = weighted_least_squares(x, y, &w)?;
            let diff = &beta_next - &beta;
            let rel = diff.dot(&diff).sqrt() / (beta_next.dot(&beta_next).sqrt() + 1e-12);
            beta = beta_next;
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

        let weights = resid.mapv(|r| self.m_rho.weight(r / s));
        Ok(RegressionFit {
            coefficients: beta,
            scale,
            residuals: resid,
            weights,
            rho: Box::new(self.m_rho.clone()), // final-stage loss ⇒ ≈95% efficiency
            breakdown_point: breakdown,        // 0.5 from the S-stage
        })
    }
}
