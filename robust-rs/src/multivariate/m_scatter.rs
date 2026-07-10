//! A monotone M-estimator of multivariate location and scatter (Maronna 1976):
//! the direct multivariate analogue of the regression M-estimator, reusing a
//! [`RhoFunction`] weight.
//!
//! Where regression M-estimation reweights each observation by `ρ.weight(rᵢ/s)`
//! on its scalar residual, scatter M-estimation reweights by `ρ.weight(dᵢ)` on
//! its **Mahalanobis distance** `dᵢ = √((xᵢ − μ)ᵀ Σ⁻¹ (xᵢ − μ))`, then solves the
//! coupled fixed point
//!
//! ```text
//! μ  = Σᵢ wᵢ xᵢ / Σᵢ wᵢ,        Σ ∝ Σᵢ wᵢ (xᵢ − μ)(xᵢ − μ)ᵀ / Σᵢ wᵢ,
//! wᵢ = ρ.weight(dᵢ / median_j dⱼ),
//! ```
//! by iteratively reweighted covariance. The weight argument is the distance
//! divided by its sample median, so the *weights* are scale-free (affine
//! equivariant); the overall size of `Σ` is then pinned separately by matching
//! the median squared distance to `χ²_{p, 0.5}`, which makes the estimator
//! Fisher-consistent for `Σ` at the Gaussian.
//!
//! With the default monotone [`Huber`] weight the objective is convex, so (like
//! regression M-estimation) it converges from any reasonable start and is
//! unique. Its breakdown point is only about `1/(p + 1)`, though (Maronna 1976):
//! a *single* well-placed high-leverage outlier can carry a monotone scatter
//! M-estimate, exactly as in the regression case. For a high-breakdown scatter,
//! use [`crate::multivariate::Mcd`]. A redescending `ρ` would need a
//! high-breakdown start (MCD/OGK), the multivariate echo of why S seeds MM.

use ndarray::{Array1, Array2};
use robust_rs_core::error::RobustError;
use robust_rs_core::rho::{Huber, RhoFunction};
use robust_rs_core::scale::{Mad, ScaleEstimator};
use robust_rs_core::solver::Control;

use super::chi2::chi2_quantile;
use super::linalg::{center, mahalanobis_sq, spd_inverse_logdet};
use super::{distances_from, ScatterFit};
use crate::util::median;

/// A configured monotone M-estimator of location and scatter, generic over the
/// [`RhoFunction`] weight (default [`Huber`], `k = 1.345`).
#[derive(Debug, Clone, Copy)]
pub struct MScatter<R = Huber> {
    /// The loss whose IRLS weight `ρ.weight(·)` downweights distant points.
    rho: R,
    /// Convergence control for the reweighting iteration.
    control: Control,
}

impl Default for MScatter<Huber> {
    fn default() -> Self {
        Self::new(Huber::default())
    }
}

impl<R: RhoFunction> MScatter<R> {
    /// Create an M-scatter estimator with the given loss and default controls.
    pub fn new(rho: R) -> Self {
        Self {
            rho,
            control: Control::default(),
        }
    }

    /// Override the convergence control.
    pub fn control(mut self, control: Control) -> Self {
        self.control = control;
        self
    }

    /// Fit the estimator to the `n × p` data matrix `x`.
    pub fn fit(&self, x: &Array2<f64>) -> Result<ScatterFit, RobustError> {
        let (n, p) = x.dim();
        if p == 0 {
            return Err(RobustError::SingularDesign);
        }
        if n < p + 1 {
            return Err(RobustError::InsufficientData {
                needed: p + 1,
                got: n,
            });
        }
        let chi_med = chi2_quantile(0.5, p as f64); // E[d²] median target at N(μ,Σ)

        // Robust start: coordinatewise median location, diagonal MAD² scatter.
        let mut mu = Array1::from_shape_fn(p, |j| median(&mut x.column(j).to_vec()));
        let mut sigma = {
            let mad = Mad::default();
            let mut d = Array2::<f64>::eye(p);
            for j in 0..p {
                let s = mad.scale(&x.column(j).to_vec())?.get();
                d[[j, j]] = s * s;
            }
            d
        };

        let mut weights = Array1::<f64>::ones(n);
        let mut converged = false;
        for _ in 0..self.control.max_iter {
            let (inv, _logdet) = spd_inverse_logdet(&sigma)?;
            let d2 = mahalanobis_sq(x, &mu, &inv);
            let dist = d2.mapv(f64::sqrt);

            // Scale-free weight argument: distance over its sample median.
            let med = {
                let mut buf = dist.to_vec();
                median(&mut buf)
            };
            if med <= 0.0 || !med.is_finite() {
                return Err(RobustError::DegenerateScale);
            }
            weights = dist.mapv(|di| self.rho.weight(di / med));
            let wsum: f64 = weights.sum();
            if wsum <= 0.0 || !wsum.is_finite() {
                return Err(RobustError::DegenerateScale);
            }

            // Weighted location.
            let mut mu_next = Array1::<f64>::zeros(p);
            for i in 0..n {
                let wi = weights[i];
                for j in 0..p {
                    mu_next[j] += wi * x[[i, j]];
                }
            }
            mu_next.mapv_inplace(|v| v / wsum);

            // Weighted covariance about the new location.
            let centered = center(x, &mu_next);
            let mut cov = Array2::<f64>::zeros((p, p));
            for i in 0..n {
                let wi = weights[i];
                for a in 0..p {
                    let ca = centered[[i, a]];
                    for b in 0..p {
                        cov[[a, b]] += wi * ca * centered[[i, b]];
                    }
                }
            }
            cov.mapv_inplace(|v| v / wsum);

            // Pin the scale: rescale so median squared distance = χ²_{p,0.5}.
            let (cov_inv, _ld) = spd_inverse_logdet(&cov)?;
            let d2c = mahalanobis_sq(x, &mu_next, &cov_inv);
            let m = {
                let mut buf = d2c.to_vec();
                median(&mut buf)
            };
            if m > 0.0 {
                cov.mapv_inplace(|v| v * m / chi_med);
            }

            // Convergence on the combined (μ, Σ) change.
            let dmu: f64 = (&mu_next - &mu).iter().map(|v| v * v).sum();
            let dsig: f64 = (&cov - &sigma).iter().map(|v| v * v).sum();
            let base: f64 = mu.iter().map(|v| v * v).sum::<f64>()
                + sigma.iter().map(|v| v * v).sum::<f64>()
                + 1e-30;
            mu = mu_next;
            sigma = cov;
            if ((dmu + dsig) / base).sqrt() <= self.control.tol {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(RobustError::NonConvergence {
                iters: self.control.max_iter,
            });
        }

        let distances = distances_from(x, &mu, &sigma)?;
        Ok(ScatterFit {
            location: mu,
            scatter: sigma,
            distances,
            weights,
        })
    }
}
