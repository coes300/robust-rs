//! The Minimum Covariance Determinant estimator (Rousseeuw 1985) via FAST-MCD
//! (Rousseeuw & Van Driessen 1999).
//!
//! MCD is the multivariate analogue of LTS: among all `h`-subsets of the `n`
//! observations it seeks the one whose sample covariance has the **smallest
//! determinant** and reports that subset's mean and (scaled) covariance. It is
//! affine equivariant and (at the max-breakdown coverage `h ≈ ⌊(n+p+1)/2⌋`)
//! has a 50% breakdown point.
//!
//! The exact search is combinatorial, so FAST-MCD approximates it: draw many
//! random elemental starts, refine each with **C-steps** and keep the
//! min-determinant result. A C-step is the multivariate concentration step
//! (given `(μ, Σ)`, take the `h` observations with the smallest Mahalanobis
//! distance and recompute `(μ, Σ)` on them) and Rousseeuw & Van Driessen prove
//! it never increases `det Σ`, so the objective decreases monotonically to a
//! local optimum. This mirrors exactly the FAST-LTS C-step in
//! [`crate::regression::Lts`], with `det Σ` in place of the trimmed sum of
//! squares.
//!
//! Two corrections turn the raw min-determinant subset into a usable covariance
//! (Croux & Haesbroeck 1999; Pison et al. 2002):
//!
//! 1. a **consistency factor** `c(α, p) = α / F_{χ²_{p+2}}(χ²_{p,α})`, `α = h/n`,
//!    scaling the raw covariance to be Fisher-consistent for `Σ` at the Gaussian;
//! 2. a one-step **reweighting** (recompute mean/covariance on the points whose
//!    (consistency-corrected) robust distance is within the `χ²_{p,0.975}` cutoff,
//!    with its own consistency factor) which recovers efficiency while keeping
//!    the breakdown point. This *reweighted MCD* (RMCD) is the primary estimate,
//!    matching what R's `robustbase::covMcd` returns by default.
//!
//! The finite-sample multiplier that R additionally applies is a documented
//! deferral; the asymptotic consistency
//! factor above is implemented and validated by the Gaussian-consistency test.

use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use rand::Rng;
use robust_rs_core::error::RobustError;
use robust_rs_core::solver::Control;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::correction::{consistency_factor, hard_reweight};
use super::linalg::{mahalanobis_sq, mean_covariance, spd_inverse_logdet, spd_logdet};
use super::{distances_from, RobustScatter};
use crate::util::substream;

/// Default master seed, so `fit` is reproducible without configuration.
const DEFAULT_SEED: u64 = 0x11CD_5EED;
/// C-steps applied to every random start before ranking (cheap pre-refinement).
const INITIAL_CSTEPS: usize = 2;
/// How many of the best pre-refined starts are concentrated to convergence.
const N_KEEP: usize = 10;

/// A configured FAST-MCD estimator.
///
/// Reproducible by default (a fixed-seed [`rand_chacha::ChaCha8Rng`] sub-stream
/// per random start, so results are thread-count invariant even with the
/// `rayon` feature on); configure with [`Mcd::seed`] / [`Mcd::fit_with_rng`].
#[derive(Debug, Clone, Copy)]
pub struct Mcd {
    /// Coverage as a fraction of `n`; `None` = the max-breakdown default
    /// `⌊(n + p + 1)/2⌋`.
    coverage: Option<f64>,
    /// Number of random elemental starts.
    n_subsamples: usize,
    /// Whether to apply the one-step reweighting (RMCD). Off ⇒ report the raw
    /// (consistency-corrected) MCD as the primary estimate.
    reweight: bool,
    /// Reweighting cutoff quantile (default `0.975`).
    reweight_quantile: f64,
    /// Master RNG seed.
    seed: u64,
    /// Convergence control for the concentration loop.
    control: Control,
}

impl Default for Mcd {
    fn default() -> Self {
        Self {
            coverage: None,
            n_subsamples: 500,
            reweight: true,
            reweight_quantile: 0.975,
            seed: DEFAULT_SEED,
            control: Control::default(),
        }
    }
}

impl Mcd {
    /// A max-breakdown MCD with default search settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Retain a `fraction ∈ (0.5, 1]` of the observations instead of the
    /// max-breakdown default; the coverage is `h = ⌊fraction · n⌋` (clamped to
    /// `[p + 1, n]`). Larger `fraction` trades breakdown for efficiency.
    pub fn coverage(mut self, fraction: f64) -> Self {
        self.coverage = Some(fraction);
        self
    }

    /// Set the number of random elemental starts (default `500`).
    pub fn n_subsamples(mut self, n: usize) -> Self {
        self.n_subsamples = n;
        self
    }

    /// Enable or disable the one-step reweighting (default `true`).
    pub fn reweight(mut self, on: bool) -> Self {
        self.reweight = on;
        self
    }

    /// Set the master RNG seed (default is a fixed internal constant).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Override the concentration-loop convergence control.
    pub fn control(mut self, control: Control) -> Self {
        self.control = control;
        self
    }

    /// Fit reproducibly from the configured seed.
    pub fn fit(&self, x: &Array2<f64>) -> Result<McdFit, RobustError> {
        self.fit_from_seed(x, self.seed)
    }

    /// Fit drawing the master seed from a caller-supplied generator.
    pub fn fit_with_rng<G: Rng>(
        &self,
        x: &Array2<f64>,
        rng: &mut G,
    ) -> Result<McdFit, RobustError> {
        self.fit_from_seed(x, rng.random::<u64>())
    }

    fn fit_from_seed(&self, x: &Array2<f64>, master_seed: u64) -> Result<McdFit, RobustError> {
        let (n, p) = x.dim();
        if p == 0 {
            return Err(RobustError::SingularDesign);
        }
        // A non-degenerate covariance needs at least p + 1 points in the subset,
        // hence at least p + 2 observations for the default h.
        if n < p + 2 {
            return Err(RobustError::InsufficientData {
                needed: p + 2,
                got: n,
            });
        }

        let h = match self.coverage {
            Some(fraction) => {
                if !(fraction.is_finite() && fraction > 0.5 && fraction <= 1.0) {
                    return Err(RobustError::InvalidTuning { value: fraction });
                }
                ((fraction * n as f64).floor() as usize).clamp(p + 1, n)
            }
            None => (n + p).div_ceil(2), // ⌊(n+p+1)/2⌋ = max breakdown; ≥ p+1 since n ≥ p+2
        };

        // --- Stage 1: many cheap starts, pre-refined by a couple of C-steps. ---
        let eval_start = |i: u64| -> Option<Candidate> {
            let mut rng = substream(master_seed, i);
            let (mu0, cov0) = elemental_start(x, p, &mut rng)?;
            let mut state = State { mu: mu0, cov: cov0 };
            for _ in 0..INITIAL_CSTEPS {
                state = c_step(x, &state, h).ok()?;
            }
            let logdet = spd_logdet(&state.cov).ok()?;
            Some(Candidate { logdet, state })
        };

        #[cfg(feature = "rayon")]
        let mut candidates: Vec<Candidate> = (0..self.n_subsamples as u64)
            .into_par_iter()
            .filter_map(eval_start)
            .collect();
        #[cfg(not(feature = "rayon"))]
        let mut candidates: Vec<Candidate> = (0..self.n_subsamples as u64)
            .filter_map(eval_start)
            .collect();

        if candidates.is_empty() {
            return Err(RobustError::SubsampleFailure);
        }

        // --- Stage 2: fully concentrate the best few; keep the global minimum. ---
        candidates.sort_by(|a, b| a.logdet.total_cmp(&b.logdet));
        candidates.truncate(N_KEEP);

        let mut best: Option<(f64, State, Vec<usize>)> = None;
        for cand in candidates {
            if let Ok((state, subset, logdet)) =
                concentrate(x, cand.state, h, self.control.max_iter)
            {
                if best.as_ref().map_or(true, |(bl, _, _)| logdet < *bl) {
                    best = Some((logdet, state, subset));
                }
            }
        }
        let (objective, raw_state, mut support) = best.ok_or(RobustError::SubsampleFailure)?;
        support.sort_unstable();

        // --- Corrections. Raw MCD: subset mean/cov × consistency factor. ---
        let alpha = h as f64 / n as f64;
        let c_raw = consistency_factor(alpha, p as f64);
        let raw_location = raw_state.mu.clone();
        let raw_scatter = &raw_state.cov * c_raw;

        let breakdown_point = (n - h + 1) as f64 / n as f64;

        // Primary estimate: the one-step reweighted MCD (RMCD) if requested and
        // enough points survive the cutoff, else the raw estimate.
        let reweighted = if self.reweight {
            hard_reweight(
                x,
                &raw_location,
                &raw_scatter,
                self.reweight_quantile,
                false,
            )?
        } else {
            None
        };

        let (location, scatter, distances, weights) = match reweighted {
            Some(rw) => (rw.location, rw.scatter, rw.distances, rw.weights),
            None => {
                let d = distances_from(x, &raw_location, &raw_scatter)?;
                (
                    raw_location.clone(),
                    raw_scatter.clone(),
                    d,
                    Array1::ones(n),
                )
            }
        };

        Ok(McdFit {
            location,
            scatter,
            distances,
            weights,
            raw_location,
            raw_scatter,
            support,
            coverage: h,
            objective,
            breakdown_point,
        })
    }
}

/// A running location/scatter state during concentration.
#[derive(Clone)]
struct State {
    mu: Array1<f64>,
    cov: Array2<f64>,
}

/// A pre-refined random start, ranked by log-determinant.
struct Candidate {
    logdet: f64,
    state: State,
}

/// Draw an elemental start: a random `(p + 1)`-subset, extended one point at a
/// time (from the same shuffle) until its covariance is non-singular. Returns
/// `None` only if the *whole* data set is rank-deficient.
fn elemental_start<G: Rng>(
    x: &Array2<f64>,
    p: usize,
    rng: &mut G,
) -> Option<(Array1<f64>, Array2<f64>)> {
    let n = x.nrows();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(rng);
    let mut k = p + 1;
    loop {
        let sub = &idx[..k];
        let xs = x.select(Axis(0), sub);
        let (mu, cov) = mean_covariance(&xs);
        if spd_logdet(&cov).is_ok() {
            return Some((mu, cov));
        }
        k += 1;
        if k > n {
            return None; // degenerate: data lies on a lower-dimensional subspace
        }
    }
}

/// One C-step: with the current `(μ, Σ)`, keep the `h` smallest-Mahalanobis
/// observations and recompute `(μ, Σ)` on them.
fn c_step(x: &Array2<f64>, state: &State, h: usize) -> Result<State, RobustError> {
    let (inv, _logdet) = spd_inverse_logdet(&state.cov)?;
    let d2 = mahalanobis_sq(x, &state.mu, &inv);
    let subset = h_smallest(&d2, h);
    let xs = x.select(Axis(0), &subset);
    let (mu, cov) = mean_covariance(&xs);
    Ok(State { mu, cov })
}

/// Concentrate to convergence: apply C-steps until the retained `h`-subset stops
/// changing (the textbook FAST-MCD criterion, exact, since at a fixed point the
/// subset and hence `det Σ` are constant) or the iteration cap is hit. Returns
/// the converged state, its (ascending) subset and its log-determinant.
fn concentrate(
    x: &Array2<f64>,
    start: State,
    h: usize,
    max_steps: usize,
) -> Result<(State, Vec<usize>, f64), RobustError> {
    let mut state = start;
    let mut prev: Option<Vec<usize>> = None;
    for _ in 0..max_steps.max(1) {
        // Subset selected by the *current* state, then the state it induces.
        let (inv, _ld) = spd_inverse_logdet(&state.cov)?;
        let d2 = mahalanobis_sq(x, &state.mu, &inv);
        let subset = h_smallest(&d2, h);
        let xs = x.select(Axis(0), &subset);
        let (mu, cov) = mean_covariance(&xs);
        let logdet = spd_logdet(&cov)?;
        state = State { mu, cov };
        if prev.as_ref() == Some(&subset) {
            return Ok((state, subset, logdet));
        }
        prev = Some(subset);
    }
    // Cap hit: report the last state and its subset/objective.
    let (inv, _ld) = spd_inverse_logdet(&state.cov)?;
    let d2 = mahalanobis_sq(x, &state.mu, &inv);
    let subset = h_smallest(&d2, h);
    let logdet = spd_logdet(&state.cov)?;
    Ok((state, subset, logdet))
}

/// Indices of the `h` observations with the smallest values, returned in
/// ascending index order (so two calls that select the same *set* compare equal
/// regardless of ties in value). `O(n log n)`.
fn h_smallest(v: &Array1<f64>, h: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    idx.truncate(h);
    idx.sort_unstable();
    idx
}

/// A fitted Minimum Covariance Determinant estimate.
///
/// Like [`crate::regression::LtsFit`], MCD carries more than the shared
/// [`ScatterFit`](super::ScatterFit) triple, so it is its own type, but it
/// *does* implement [`RobustScatter`], reporting the reweighted (RMCD)
/// location/scatter through it. The raw min-determinant estimate and the
/// retained subset remain available as fields for auditing.
#[derive(Debug, Clone)]
pub struct McdFit {
    /// Reweighted (RMCD) location: the primary estimate.
    pub location: Array1<f64>,
    /// Reweighted (RMCD) covariance: the primary estimate.
    pub scatter: Array2<f64>,
    /// Robust Mahalanobis distances w.r.t. the primary `(location, scatter)`.
    pub distances: Array1<f64>,
    /// Reweighting weights: `1.0` for observations retained by the `χ²` cutoff,
    /// `0.0` for those rejected (all `1.0` when reweighting is disabled).
    pub weights: Array1<f64>,
    /// Raw (consistency-corrected) MCD location: the best `h`-subset mean.
    pub raw_location: Array1<f64>,
    /// Raw (consistency-corrected) MCD covariance: the min-determinant subset
    /// covariance times the consistency factor.
    pub raw_scatter: Array2<f64>,
    /// The retained min-determinant `h`-subset (ascending indices).
    pub support: Vec<usize>,
    /// Coverage `h`.
    pub coverage: usize,
    /// The MCD objective at the fit: `log det` of the raw (uncorrected) subset
    /// covariance.
    pub objective: f64,
    /// Breakdown point `(n − h + 1)/n`.
    pub breakdown_point: f64,
}

impl McdFit {
    /// Reweighted (RMCD) location.
    pub fn location(&self) -> &Array1<f64> {
        &self.location
    }
    /// Reweighted (RMCD) covariance.
    pub fn scatter(&self) -> &Array2<f64> {
        &self.scatter
    }
    /// The raw min-determinant `h`-subset (ascending indices).
    pub fn support(&self) -> &[usize] {
        &self.support
    }
    /// Breakdown point.
    pub fn breakdown_point(&self) -> f64 {
        self.breakdown_point
    }
}

impl RobustScatter for McdFit {
    fn location(&self) -> &Array1<f64> {
        &self.location
    }
    fn scatter(&self) -> &Array2<f64> {
        &self.scatter
    }
    fn distances(&self) -> &Array1<f64> {
        &self.distances
    }
}
