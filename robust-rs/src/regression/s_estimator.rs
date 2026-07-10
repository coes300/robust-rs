//! S-estimation of regression by FAST-S (Salibián-Barrera & Yohai 2006): a
//! high-breakdown estimate that minimizes a robust M-scale of the residuals.
//!
//! The elemental-resampling search itself lives in [`super::subsample`] and is
//! shared with FAST-LTS; this module supplies only the two S-specific pieces:
//! the objective (the S-scale of the residuals) and the C-step (reweight by
//! `ρ.weight(rᵢ/s)` at the current S-scale, then re-solve). With the S-step
//! biweight (`c ≈ 1.547`, `δ = ρ_sup/2`) this attains a 50% breakdown point,
//! but only ≈ 29% Gaussian efficiency, which is why MM adds an M-step on top
//! (see [`crate::regression::MMEstimator`]).
//!
//! Two speedups are deferred: the SB&Y *partial-scale rejection* trick (abort a
//! candidate's S-scale solve early once `Σ ρ(rᵢ/s) ≥ n·δ`, since it then cannot
//! beat the incumbent) and subset-level `rayon` parallelism (the per-subsample
//! sub-streams already make the result order-independent, so it parallelizes
//! without changing the answer).

use ndarray::{Array1, Array2};
use rand::Rng;
use robust_rs_core::error::RobustError;
use robust_rs_core::rho::{RhoFunction, TukeyBiweight};
use robust_rs_core::scale::{SScale, ScaleEstimator};
use robust_rs_core::solver::Control;
use robust_rs_core::types::Scale;

use super::subsample::{fast_resample, SearchConfig};
use crate::estimator::RegressionFit;
use crate::wls::weighted_least_squares;

/// Default master seed, so `fit` is reproducible without any configuration.
const DEFAULT_SEED: u64 = 0xFA57_5EED;

/// A configured regression S-estimator.
///
/// Reproducible by default (a fixed-seed [`rand_chacha::ChaCha8Rng`] sub-stream
/// per subsample); set the seed with [`SEstimator::seed`] or supply your own
/// generator via [`SEstimator::fit_with_rng`].
#[derive(Debug, Clone, Copy)]
pub struct SEstimator<R = TukeyBiweight> {
    /// The (bounded, redescending) S-step loss.
    rho: R,
    /// Consistency/breakdown target `δ` for the S-scale (`ρ_sup/2` ⇒ 50%).
    delta: f64,
    /// Number of random elemental subsets to draw.
    n_subsamples: usize,
    /// Master RNG seed.
    seed: u64,
    /// Convergence control for the concentration steps.
    control: Control,
}

impl Default for SEstimator<TukeyBiweight> {
    /// The standard 50%-breakdown S-estimator: biweight `c = 1.547`,
    /// `δ = ρ_sup/2`, 500 subsamples.
    fn default() -> Self {
        Self::new(TukeyBiweight::new(1.547).expect("1.547 is a valid tuning"))
    }
}

impl<R: RhoFunction + Clone + 'static> SEstimator<R> {
    /// Create an S-estimator for the given bounded loss, with `δ = ρ_sup/2` (the
    /// 50%-breakdown target) and default search settings. Losses without a
    /// bounded `ρ` (`rho_sup() == None`) are rejected at [`SEstimator::fit`]
    /// time by the S-scale.
    pub fn new(rho: R) -> Self {
        let delta = rho.rho_sup().map_or(f64::NAN, |sup| 0.5 * sup);
        Self {
            rho,
            delta,
            n_subsamples: 500,
            seed: DEFAULT_SEED,
            control: Control::default(),
        }
    }

    /// Override the S-scale target `δ` (default `ρ_sup/2`).
    pub fn delta(mut self, delta: f64) -> Self {
        self.delta = delta;
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
    pub fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<RegressionFit, RobustError> {
        self.fit_from_seed(x, y, self.seed)
    }

    /// Fit drawing the master seed from a caller-supplied generator. The search
    /// itself still runs through the internal seeded sub-streams, so a given
    /// master seed reproduces bit-for-bit regardless of thread count.
    pub fn fit_with_rng<G: Rng>(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        rng: &mut G,
    ) -> Result<RegressionFit, RobustError> {
        self.fit_from_seed(x, y, rng.random::<u64>())
    }

    fn fit_from_seed(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        master_seed: u64,
    ) -> Result<RegressionFit, RobustError> {
        // The S-scale enforces the bounded-ρ precondition; build it once.
        // (Dimension checks live in the shared resampler.)
        let sscale = SScale::new(self.rho.clone(), self.delta)?.with_control(self.control);
        let rho = &self.rho;

        // Everything S-specific is these two closures; the resampler is generic.
        // Objective: the S-scale of the residuals.
        let score = |beta: &Array1<f64>| -> Result<f64, RobustError> {
            let resid = y - &x.dot(beta);
            Ok(sscale
                .scale(resid.as_slice().expect("contiguous residuals"))?
                .get())
        };
        // One C-step: reweight by ρ.weight(rᵢ/s) at the current S-scale, re-solve.
        let cstep = |beta: &Array1<f64>| -> Result<Array1<f64>, RobustError> {
            let resid = y - &x.dot(beta);
            let s = sscale
                .scale(resid.as_slice().expect("contiguous residuals"))?
                .get();
            let w = resid.mapv(|r| rho.weight(r / s));
            weighted_least_squares(x, y, &w)
        };

        // S accepts the default (500, 5, 2) schedule, overriding only what its
        // own builders expose.
        let cfg = SearchConfig {
            n_subsamples: self.n_subsamples,
            control: self.control,
            ..SearchConfig::default()
        };
        let (coefficients, best_scale) = fast_resample(x, y, master_seed, &cfg, score, cstep)?;

        let scale = Scale::new(best_scale)?;
        let s = scale.get();
        let residuals = y - &x.dot(&coefficients);
        let weights = residuals.mapv(|r| self.rho.weight(r / s));
        Ok(RegressionFit {
            coefficients,
            scale,
            residuals,
            weights,
            rho: Box::new(self.rho.clone()),
            breakdown_point: breakdown_from_delta(self.delta, self.rho.rho_sup()),
        })
    }
}

/// Breakdown point implied by an S-scale target: `min(δ/sup, 1 − δ/sup)`
/// (`= 0.5` at the standard `δ = sup/2`). `0.0` if the loss is unbounded.
fn breakdown_from_delta(delta: f64, rho_sup: Option<f64>) -> f64 {
    match rho_sup {
        Some(sup) if sup > 0.0 => {
            let ratio = delta / sup;
            ratio.min(1.0 - ratio)
        }
        _ => 0.0,
    }
}
