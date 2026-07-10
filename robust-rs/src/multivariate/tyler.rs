//! Tyler's M-estimator of shape (Tyler 1987): a distribution-free scatter
//! estimator for elliptically-symmetric data, normalized to unit determinant.
//!
//! Tyler's estimator depends on the data only through the *directions*
//! `rᵢ / ‖rᵢ‖` of the centered observations, so it is the "most robust"
//! distribution-free estimator of shape: it is consistent for the shape matrix
//! of *any* elliptical distribution without knowing the radial density (no
//! Gaussianity assumed). It solves the fixed-point equation
//!
//! ```text
//! V ∝ Σᵢ (rᵢ rᵢᵀ) / (rᵢᵀ V⁻¹ rᵢ),      rᵢ = xᵢ − μ,
//! ```
//!
//! normalized to `det V = 1`, by the iteration `V ← normalize(Σ …)`. Because the
//! iterate is renormalized to unit determinant every step, the scalar in front
//! of the sum is immaterial; only the *shape* is identified, so the returned
//! [`TylerFit::shape`] carries no scale (`det = 1`). Mahalanobis distances
//! from it still rank observations correctly for outlier detection.
//!
//! Location is not estimated jointly; a robust centre must be supplied. The
//! default is the coordinatewise median.

use ndarray::{Array1, Array2};
use robust_rs_core::error::RobustError;
use robust_rs_core::solver::Control;

use super::distances_from;
use super::linalg::{center, spd_inverse_logdet, spd_logdet};
use crate::util::median;

/// A configured Tyler shape estimator.
#[derive(Debug, Clone, Default)]
pub struct Tyler {
    /// Location `μ`; `None` = the coordinatewise median.
    location: Option<Array1<f64>>,
    /// Convergence control for the fixed-point iteration.
    control: Control,
}

impl Tyler {
    /// A Tyler estimator centering at the coordinatewise median.
    pub fn new() -> Self {
        Self::default()
    }

    /// Center at a supplied location instead of the coordinatewise median.
    pub fn location(mut self, loc: Array1<f64>) -> Self {
        self.location = Some(loc);
        self
    }

    /// Override the fixed-point convergence control.
    pub fn control(mut self, control: Control) -> Self {
        self.control = control;
        self
    }

    /// Fit the shape estimator to the `n × p` data matrix `x`.
    pub fn fit(&self, x: &Array2<f64>) -> Result<TylerFit, RobustError> {
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

        let loc = match &self.location {
            Some(l) => {
                if l.len() != p {
                    return Err(RobustError::DimensionMismatch {
                        expected: p,
                        got: l.len(),
                    });
                }
                l.clone()
            }
            None => coordinatewise_median(x),
        };
        let r = center(x, &loc); // residuals rᵢ = xᵢ − μ

        // Iterate V ← normalize_to_unit_det( Σᵢ rᵢrᵢᵀ / (rᵢᵀ V⁻¹ rᵢ) ).
        let mut v = Array2::<f64>::eye(p);
        let tiny = 1e-12;
        let mut converged = false;
        for _ in 0..self.control.max_iter {
            let (v_inv, _logdet) = spd_inverse_logdet(&v)?;

            let mut s = Array2::<f64>::zeros((p, p));
            let mut used = 0usize;
            for i in 0..n {
                let ri = r.row(i);
                let vinv_ri = v_inv.dot(&ri);
                let d = ri.dot(&vinv_ri); // rᵢᵀ V⁻¹ rᵢ
                if d <= tiny {
                    continue; // residual at (essentially) the centre: skip 0/0 term
                }
                used += 1;
                for a in 0..p {
                    let ra = ri[a];
                    for b in 0..p {
                        s[[a, b]] += ra * ri[b] / d;
                    }
                }
            }
            if used <= p {
                // Too few non-degenerate directions to identify a full-rank shape.
                return Err(RobustError::InsufficientData {
                    needed: p + 1,
                    got: used,
                });
            }

            // Normalize to unit determinant (the leading scalar cancels here).
            let logdet = spd_logdet(&s)?;
            let scale = (logdet / p as f64).exp(); // det(S)^{1/p}
            let v_next = s.mapv(|x| x / scale);

            let num: f64 = (&v_next - &v).iter().map(|x| x * x).sum();
            let den: f64 = v.iter().map(|x| x * x).sum::<f64>().max(1e-30);
            v = v_next;
            if (num / den).sqrt() <= self.control.tol {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(RobustError::NonConvergence {
                iters: self.control.max_iter,
            });
        }

        let distances = distances_from(x, &loc, &v)?;
        Ok(TylerFit {
            location: loc,
            shape: v,
            distances,
        })
    }
}

/// A fitted Tyler shape estimate.
///
/// Tyler estimates **shape only**: [`shape`](Self::shape) is normalized to unit
/// determinant and carries no scale. A consequence is that its Mahalanobis
/// distances are correct for *ranking* observations but are **not**
/// `χ²_p`-calibrated; for `N(0, Σ)` data their *squared* distances are the
/// Gaussian `χ²_p` distances times the unidentified factor `det(Σ)^{1/p}` (so the
/// distances themselves scale by `det(Σ)^{1/(2p)}`) and for non-Gaussian
/// elliptical data the `χ²` shape does not apply at all. `TylerFit` therefore
/// does **not** implement [`RobustScatter`](crate::multivariate::RobustScatter)
/// and offers no default Gaussian outlier map, mirroring how the regression
/// [`LtsFit`](crate::regression::LtsFit) declines the ρ-derived efficiency it
/// cannot report. Rank with [`distances`](Self::distances), or opt in explicitly
/// via [`outliers_assuming_chi2_radial`](Self::outliers_assuming_chi2_radial).
#[derive(Debug, Clone)]
pub struct TylerFit {
    location: Array1<f64>,
    shape: Array2<f64>,
    distances: Array1<f64>,
}

impl TylerFit {
    /// The robust centre `μ̂` (the supplied location, or the coordinatewise median).
    pub fn location(&self) -> &Array1<f64> {
        &self.location
    }

    /// The unit-determinant shape matrix `V̂` (`det V̂ = 1`; no scale identified).
    pub fn shape(&self) -> &Array2<f64> {
        &self.shape
    }

    /// Robust Mahalanobis distances from the shape. Correct for **ranking**
    /// observations, but not `χ²_p`-calibrated (the scale is unidentified).
    pub fn distances(&self) -> &Array1<f64> {
        &self.distances
    }

    /// Flag outliers by comparing the Tyler distances to the Gaussian cutoff
    /// `√χ²_{p, quantile}`.
    ///
    /// As the name says, this assumes a `χ²`-radial model (e.g. Gaussian) (the
    /// assumption Tyler's shape estimate exists to avoid) and disregards the
    /// unidentified squared-distance factor `det(Σ)^{1/p}`, so the cutoff is
    /// meaningful only under that model. It is an opt-in for the Gaussian case;
    /// for distribution-free use, threshold [`distances`](Self::distances)
    /// directly. An *empirical* quantile flags a **fixed proportion** of points
    /// (≈ `1 − quantile`) whether or not any are outliers, unlike this absolute
    /// `χ²` cutoff, which flags ≈ none of clean Gaussian data.
    pub fn outliers_assuming_chi2_radial(&self, quantile: f64) -> Vec<bool> {
        let p = self.shape.nrows() as f64;
        let cut = super::chi2::chi2_quantile(quantile, p).sqrt();
        self.distances.iter().map(|&d| d > cut).collect()
    }
}

/// Coordinatewise median: the median of each column of `x`.
fn coordinatewise_median(x: &Array2<f64>) -> Array1<f64> {
    let p = x.ncols();
    Array1::from_shape_fn(p, |j| median(&mut x.column(j).to_vec()))
}
