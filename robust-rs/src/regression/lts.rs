//! Least Trimmed Squares (Rousseeuw 1984) via FAST-LTS (Rousseeuw & Van
//! Driessen 2006).
//!
//! LTS minimizes the sum of the `h` smallest squared residuals, a
//! high-breakdown fit that simply *ignores* the `n − h` largest residuals. It
//! reuses the shared elemental-resampling search in [`super::subsample`];
//! the only LTS-specific pieces are its objective (Σ of the `h` smallest
//! squared residuals) and its C-step (refit OLS on the `h` smallest-residual
//! observations). Both derive residuals from a single `residuals(β)` binding,
//! so the objective and the step can never select on differently-computed
//! residuals.
//!
//! **Convergence.** The shared concentration loop stops on a relative change in
//! the objective below `tol`. FAST-LTS's textbook criterion is instead "the
//! retained `h`-subset stops changing"; the two coincide here, because at a fixed
//! point the objective is *exactly* constant, so a below-`tol` change means the
//! subset has (bar sub-`tol` ties) stabilized. LTS is used as a high-breakdown
//! initializer, for which an ε-neighborhood of the fixed point suffices.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use robust_rs_core::error::RobustError;
use robust_rs_core::scale::{Mad, ScaleEstimator};
use robust_rs_core::solver::Control;
use robust_rs_core::types::Scale;

use super::subsample::{fast_resample, SearchConfig};
use crate::wls::weighted_least_squares;

/// Default master seed, so `fit` is reproducible without any configuration.
const DEFAULT_SEED: u64 = 0x0175_5EED;

/// A configured Least Trimmed Squares estimator.
///
/// Like [`crate::regression::SEstimator`], it is reproducible by default (a
/// fixed-seed [`rand_chacha::ChaCha8Rng`] sub-stream per subsample); configure
/// with [`Lts::seed`] / [`Lts::fit_with_rng`].
#[derive(Debug, Clone, Copy)]
pub struct Lts {
    /// Coverage as a fraction of `n`; `None` = the max-breakdown default
    /// `⌊(n + p + 1)/2⌋`.
    coverage: Option<f64>,
    /// Number of random elemental subsets to draw.
    n_subsamples: usize,
    /// Master RNG seed.
    seed: u64,
    /// Convergence control for the concentration steps.
    control: Control,
}

impl Default for Lts {
    /// Max-breakdown coverage, 500 subsamples.
    fn default() -> Self {
        Self {
            coverage: None,
            n_subsamples: 500,
            seed: DEFAULT_SEED,
            control: Control::default(),
        }
    }
}

impl Lts {
    /// A max-breakdown LTS with default search settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Retain a `fraction ∈ (0, 1]` of the observations instead of the
    /// max-breakdown default; the coverage is `h = ⌊fraction · n⌋` (clamped to
    /// `[p, n]`). Larger `fraction` trades breakdown for efficiency.
    pub fn coverage(mut self, fraction: f64) -> Self {
        self.coverage = Some(fraction);
        self
    }

    /// Set the number of random elemental subsets (default `500`).
    pub fn n_subsamples(mut self, n: usize) -> Self {
        self.n_subsamples = n;
        self
    }

    /// Set the master RNG seed (default is a fixed internal constant).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Override the concentration-step convergence control.
    pub fn control(mut self, control: Control) -> Self {
        self.control = control;
        self
    }

    /// Fit reproducibly from the configured seed.
    pub fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<LtsFit, RobustError> {
        self.fit_from_seed(x, y, self.seed)
    }

    /// Fit drawing the master seed from a caller-supplied generator.
    pub fn fit_with_rng<G: Rng>(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        rng: &mut G,
    ) -> Result<LtsFit, RobustError> {
        self.fit_from_seed(x, y, rng.random::<u64>())
    }

    fn fit_from_seed(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        master_seed: u64,
    ) -> Result<LtsFit, RobustError> {
        let (n, p) = x.dim();
        // Validate here too (not only in the resampler) because `h` needs sane
        // `n`/`p` before the clamp below.
        if p == 0 {
            return Err(RobustError::SingularDesign);
        }
        if y.len() != n {
            return Err(RobustError::DimensionMismatch {
                expected: n,
                got: y.len(),
            });
        }
        if n < p {
            return Err(RobustError::InsufficientData { needed: p, got: n });
        }

        let h = match self.coverage {
            Some(fraction) => {
                if !(fraction.is_finite() && fraction > 0.0 && fraction <= 1.0) {
                    return Err(RobustError::InvalidTuning { value: fraction });
                }
                ((fraction * n as f64).floor() as usize).clamp(p, n)
            }
            None => (n + p).div_ceil(2).clamp(p, n), // ⌊(n+p+1)/2⌋ = max breakdown
        };

        // Single residual definition shared by both closures ⇒ the objective and
        // the C-step provably select on the same residuals.
        let residuals = |beta: &Array1<f64>| -> Array1<f64> { y - &x.dot(beta) };
        let ones_h = Array1::ones(h);

        // Objective: Σ of the h smallest squared residuals.
        let score = |beta: &Array1<f64>| -> Result<f64, RobustError> {
            let r = residuals(beta);
            let idx = h_smallest_abs(&r, h);
            Ok(idx.iter().map(|&i| r[i] * r[i]).sum())
        };
        // One C-step: refit OLS on the h smallest-residual observations.
        let cstep = |beta: &Array1<f64>| -> Result<Array1<f64>, RobustError> {
            let r = residuals(beta);
            let idx = h_smallest_abs(&r, h);
            let xs = x.select(Axis(0), &idx);
            let ys = y.select(Axis(0), &idx);
            weighted_least_squares(&xs, &ys, &ones_h)
        };

        let cfg = SearchConfig {
            n_subsamples: self.n_subsamples,
            control: self.control,
            ..SearchConfig::default()
        };
        let (coefficients, objective) = fast_resample(x, y, master_seed, &cfg, score, cstep)?;

        let resid = y - &x.dot(&coefficients);
        let mut subset = h_smallest_abs(&resid, h);
        subset.sort_unstable(); // ascending, for readable auditing of the trim
                                // A MAD of the residuals is a consistent robust scale for the fit; the
                                // classical trimmed-RMS-with-consistency-factor is a later refinement.
        let scale = Mad::default().scale(resid.as_slice().expect("contiguous residuals"))?;
        let breakdown_point = (n - h + 1) as f64 / n as f64;

        Ok(LtsFit {
            coefficients,
            scale,
            residuals: resid,
            subset,
            objective,
            coverage: h,
            breakdown_point,
        })
    }
}

/// A fitted Least Trimmed Squares regression.
///
/// LTS is a high-breakdown *initializer*, not an M-estimator, so (unlike
/// [`crate::estimator::RegressionFit`]) it does **not** implement the ρ-based
/// [`crate::estimator::RobustEstimator`] surface. LTS has a well-defined influence
/// function and Gaussian efficiency (it is √n-consistent and asymptotically
/// normal, Rousseeuw & Leroy 1987; Víšek), but these follow from the asymptotics
/// of the trimmed objective, **not** from `ψ(r)/E[ψ']` and `1/V(ψ)` (a hard 0/1
/// trim has no smooth ψ) so they are not reported here. It exposes coefficients,
/// the retained `h`-subset (so the trimming is auditable), a robust residual scale
/// (MAD) and its coverage-implied breakdown point.
#[derive(Debug, Clone)]
pub struct LtsFit {
    /// Estimated coefficients.
    pub coefficients: Array1<f64>,
    /// Robust residual scale (MAD of the residuals).
    pub scale: Scale,
    /// Residuals `y − Xβ̂` for all `n` observations.
    pub residuals: Array1<f64>,
    /// Retained subset: the indices of the `h` smallest-residual observations
    /// (ascending).
    pub subset: Vec<usize>,
    /// The LTS objective at the fit: Σ of the `h` smallest squared residuals.
    pub objective: f64,
    /// Coverage `h`: the number of observations retained.
    pub coverage: usize,
    /// Breakdown point `(n − h + 1)/n`.
    pub breakdown_point: f64,
}

impl LtsFit {
    /// Estimated coefficients.
    pub fn coefficients(&self) -> &Array1<f64> {
        &self.coefficients
    }
    /// Robust residual scale.
    pub fn scale(&self) -> Scale {
        self.scale
    }
    /// The retained `h`-subset (ascending indices).
    pub fn subset(&self) -> &[usize] {
        &self.subset
    }
    /// Breakdown point.
    pub fn breakdown_point(&self) -> f64 {
        self.breakdown_point
    }
}

/// Indices of the `h` observations with the smallest absolute residual (=
/// smallest squared residual). `O(n log n)`.
fn h_smallest_abs(resid: &Array1<f64>, h: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..resid.len()).collect();
    idx.sort_by(|&a, &b| resid[a].abs().total_cmp(&resid[b].abs()));
    idx.truncate(h);
    idx
}
