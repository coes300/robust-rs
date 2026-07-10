//! The Orthogonalized Gnanadesikan–Kettenring (OGK) robust covariance estimator
//! (Maronna & Zamar 2002).
//!
//! Gnanadesikan & Kettenring (1972) build a robust covariance one *pair* of
//! variables at a time from the polarization identity, using a robust scale `σ`:
//! `cov(a, b) = ¼[σ(a + b)² − σ(a − b)²]`. Assembled entrywise this is fast and
//! highly robust, but the resulting matrix need not be positive definite. OGK
//! fixes that: scale the variables to unit `σ`, form the GK "correlation"
//! matrix, rotate into its eigenbasis (where the variables are nearly
//! uncorrelated), estimate a robust location and scale along each new axis and
//! transform back. Because the reconstructed covariance is `E · diag(γ²) · Eᵀ`
//! (conjugated by the diagonal scales), it is **positive definite by
//! construction**. The rotate–estimate–reconstruct pass is iterated (twice by
//! default) and a one-step reweighting recovers efficiency.
//!
//! OGK is deterministic (no random starts) and `O(n p² log n + p³)`, so it is a
//! cheap, positive-definite alternative to FAST-MCD and a good warm start for
//! it. It is equivariant under coordinatewise scaling and permutation and
//! (only) *approximately* affine equivariant, which is the price of avoiding the
//! combinatorial min-determinant search.

use ndarray::{Array1, Array2};
use robust_rs_core::error::RobustError;
use robust_rs_core::scale::{Qn, ScaleEstimator};

use super::correction::hard_reweight;
use super::linalg::symmetric_eigen;
use super::{distances_from, ScatterFit};
use crate::util::median;

/// A configured OGK estimator, generic over the robust univariate scale (the
/// default [`Qn`] is Maronna & Zamar's recommendation; the location functional
/// paired with it is the median).
#[derive(Debug, Clone, Copy)]
pub struct Ogk<S = Qn> {
    /// Robust univariate scale used both to standardize variables and inside the
    /// pairwise GK covariance.
    scale: S,
    /// Number of orthogonalization iterations (Maronna & Zamar recommend `2`).
    n_iter: usize,
    /// Whether to apply the one-step reweighting.
    reweight: bool,
    /// Reweighting cutoff quantile for the median-adjusted cutoff (Maronna &
    /// Zamar use `0.9`).
    reweight_quantile: f64,
}

impl Default for Ogk<Qn> {
    fn default() -> Self {
        Self::new(Qn::default())
    }
}

impl<S: ScaleEstimator> Ogk<S> {
    /// Create an OGK estimator with the given robust scale, two orthogonalization
    /// iterations and reweighting on.
    pub fn new(scale: S) -> Self {
        Self {
            scale,
            n_iter: 2,
            reweight: true,
            reweight_quantile: 0.9,
        }
    }

    /// Set the number of orthogonalization iterations (default `2`).
    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Enable or disable the one-step reweighting (default `true`).
    pub fn reweight(mut self, on: bool) -> Self {
        self.reweight = on;
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

        let (raw_location, raw_scatter) = self.raw(x)?;

        let reweighted = if self.reweight {
            // OGK uses the median-adjusted cutoff (Maronna & Zamar), which
            // self-calibrates to the raw scatter's scale.
            hard_reweight(x, &raw_location, &raw_scatter, self.reweight_quantile, true)?
        } else {
            None
        };

        let (location, scatter, distances, weights) = match reweighted {
            Some(rw) => (rw.location, rw.scatter, rw.distances, rw.weights),
            None => {
                let d = distances_from(x, &raw_location, &raw_scatter)?;
                (raw_location, raw_scatter, d, Array1::ones(n))
            }
        };

        Ok(ScatterFit {
            location,
            scatter,
            distances,
            weights,
        })
    }

    /// The raw (pre-reweighting) OGK location and covariance.
    ///
    /// Each iteration works on the current representation `W` (initially `X`):
    /// standardize columns to unit robust scale, form the pairwise GK matrix,
    /// take its eigenvectors `E`, project `Z = (W D⁻¹) E` and record the linear
    /// map `B = D E` that sends `Z`-coordinates back to `W`-coordinates. The maps
    /// compose into `M = B₁ B₂ … B_k`, so the final robust marginal location `ν`
    /// and scales `γ` (estimated on the last `Z`) transform back as
    /// `μ = M ν`, `Σ = M diag(γ²) Mᵀ`, positive definite since `diag(γ²) ≻ 0`.
    fn raw(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), RobustError> {
        let (n, p) = x.dim();
        let mut w = x.clone();
        let mut m = Array2::<f64>::eye(p);

        for _ in 0..self.n_iter.max(1) {
            // Robust column scales; standardize to Y = W · D⁻¹.
            let mut sigma = Array1::<f64>::zeros(p);
            for j in 0..p {
                sigma[j] = self.scale.scale(&w.column(j).to_vec())?.get();
            }
            let y = Array2::from_shape_fn((n, p), |(i, j)| w[[i, j]] / sigma[j]);

            // Pairwise GK "correlation" matrix (unit diagonal, since Y has unit
            // robust scale): U_jk = ¼[σ(Y_j + Y_k)² − σ(Y_j − Y_k)²].
            let mut u = Array2::<f64>::eye(p);
            for j in 0..p {
                for k in (j + 1)..p {
                    let sum: Vec<f64> = (0..n).map(|i| y[[i, j]] + y[[i, k]]).collect();
                    let dif: Vec<f64> = (0..n).map(|i| y[[i, j]] - y[[i, k]]).collect();
                    let sp = self.scale.scale(&sum)?.get();
                    let sm = self.scale.scale(&dif)?.get();
                    let ujk = 0.25 * (sp * sp - sm * sm);
                    u[[j, k]] = ujk;
                    u[[k, j]] = ujk;
                }
            }

            let (_vals, e) = symmetric_eigen(&u)?;
            let z = y.dot(&e); // Z = Y E
            let b = Array2::from_shape_fn((p, p), |(i, j)| sigma[i] * e[[i, j]]); // B = D E
            m = m.dot(&b);
            w = z;
        }

        // Robust marginal location/scale in the final coordinates.
        let mut nu = Array1::<f64>::zeros(p);
        let mut gamma2 = Array1::<f64>::zeros(p);
        for l in 0..p {
            let g = self.scale.scale(&w.column(l).to_vec())?.get();
            gamma2[l] = g * g;
            nu[l] = median(&mut w.column(l).to_vec());
        }

        let mu = m.dot(&nu);
        // Σ = M diag(γ²) Mᵀ, formed as (M scaled columnwise by γ²) · Mᵀ.
        let mg = Array2::from_shape_fn((p, p), |(i, j)| m[[i, j]] * gamma2[j]);
        let sigma = mg.dot(&m.t());
        Ok((mu, sigma))
    }
}
