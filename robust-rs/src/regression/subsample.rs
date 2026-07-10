//! Shared FAST-resampling scaffold for high-breakdown regression estimators
//! (S and LTS).
//!
//! This owns the parts common to FAST-S and FAST-LTS: drawing random *elemental*
//! `p`-subsets, deriving an independent, thread-count-invariant RNG sub-stream per
//! attempt, exact-fitting each subset, skipping degenerate (collinear) draws, the
//! keep-the-best finalists bookkeeping and the concentration ("C-step") control
//! flow.
//!
//! What *differs* between estimators enters only as closures: `score(&β) → f64`,
//! the objective the search minimizes and `cstep(&β) → β'`, one concentration
//! step. The helper is agnostic to scale, ρ and the trimming fraction `h`.

use ndarray::{Array1, Array2, Axis};
use robust_rs_core::error::RobustError;
use robust_rs_core::solver::Control;

use crate::util::substream;
use crate::wls::weighted_least_squares;

/// The FAST-search schedule, shared by the S- and LTS-estimators. Its defaults
/// `(500, 5, 2)` are the standard "many cheap starts, then fully concentrate a
/// few finalists" schedule; it is a caller-supplied value rather than helper
/// constants so an estimator can differ without touching the shared search.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SearchConfig {
    /// Number of random elemental subsets to draw.
    pub n_subsamples: usize,
    /// How many of the lowest-scoring candidates are fully concentrated.
    pub n_keep: usize,
    /// C-steps applied to every subsample before ranking (cheap pre-refinement).
    pub initial_csteps: usize,
    /// Convergence control for the concentration loop.
    pub control: Control,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            n_subsamples: 500,
            n_keep: 5,
            initial_csteps: 2,
            control: Control::default(),
        }
    }
}

/// Run a FAST elemental-resampling search: draw `cfg.n_subsamples` random
/// `p`-point exact fits, pre-refine each with a few C-steps, then fully
/// concentrate the best `cfg.n_keep`; return the coefficients with the smallest
/// `score` and that score.
///
/// `score` is the objective to **minimize**; `cstep` performs one concentration
/// step. A collinear elemental subset (singular exact fit) is skipped and
/// redrawn; if no valid subset is found within a bounded number of attempts the
/// search returns [`RobustError::SubsampleFailure`]. Reproducibility comes from
/// `master_seed`: attempt `i` always draws the same subset (via an independent
/// sub-stream keyed by `i`), so neither iteration order nor thread count can
/// change the result.
pub(crate) fn fast_resample<S, C>(
    x: &Array2<f64>,
    y: &Array1<f64>,
    master_seed: u64,
    cfg: &SearchConfig,
    score: S,
    cstep: C,
) -> Result<(Array1<f64>, f64), RobustError>
where
    S: Fn(&Array1<f64>) -> Result<f64, RobustError>,
    C: Fn(&Array1<f64>) -> Result<Array1<f64>, RobustError>,
{
    let (n, p) = x.dim();
    if y.len() != n {
        return Err(RobustError::DimensionMismatch {
            expected: n,
            got: y.len(),
        });
    }
    if p == 0 {
        return Err(RobustError::SingularDesign);
    }
    if n < p {
        return Err(RobustError::InsufficientData { needed: p, got: n });
    }
    let ones = Array1::ones(p);

    // Draw subsamples, keeping every candidate's post-pre-refinement score. A
    // singular exact fit means a collinear elemental subset; skip and redraw
    // rather than propagate; only genuine exhaustion is an error.
    let mut candidates: Vec<(f64, Array1<f64>)> = Vec::new();
    let max_attempts = cfg.n_subsamples.saturating_mul(20).max(50);
    let mut valid = 0usize;
    let mut attempt = 0u64;
    while valid < cfg.n_subsamples && (attempt as usize) < max_attempts {
        let mut sub = substream(master_seed, attempt);
        attempt += 1;

        let idx = rand::seq::index::sample(&mut sub, n, p).into_vec();
        let xs = x.select(Axis(0), &idx);
        let ys = y.select(Axis(0), &idx);
        let beta0 = match weighted_least_squares(&xs, &ys, &ones) {
            Ok(b) => b,
            Err(_) => continue, // collinear elemental subset ⇒ redraw
        };
        valid += 1;

        if let Ok((beta, sc)) =
            concentrate(beta0, &score, &cstep, cfg.initial_csteps, cfg.control.tol)
        {
            candidates.push((sc, beta));
        }
    }
    if candidates.is_empty() {
        return Err(RobustError::SubsampleFailure);
    }

    // Fully concentrate the most promising handful; keep the global best.
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    candidates.truncate(cfg.n_keep);

    let mut best_score = f64::INFINITY;
    let mut best_beta: Option<Array1<f64>> = None;
    for (_, beta0) in candidates {
        if let Ok((beta, sc)) =
            concentrate(beta0, &score, &cstep, cfg.control.max_iter, cfg.control.tol)
        {
            if sc < best_score {
                best_score = sc;
                best_beta = Some(beta);
            }
        }
    }
    let beta = best_beta.ok_or(RobustError::SubsampleFailure)?;
    Ok((beta, best_score))
}

/// Concentration loop: repeatedly apply `cstep`, stopping after `max_steps` or
/// when the relative change in `score` falls below `tol`. The convergence test
/// is evaluated on *every* step including the first (against the pre-step
/// score), so the stopping point is independent of how the loop is structured.
/// A `cstep` or `score` failure (e.g. a reweighting that turns the design
/// singular) ends the loop at the best state reached so far.
fn concentrate<S, C>(
    beta0: Array1<f64>,
    score: &S,
    cstep: &C,
    max_steps: usize,
    tol: f64,
) -> Result<(Array1<f64>, f64), RobustError>
where
    S: Fn(&Array1<f64>) -> Result<f64, RobustError>,
    C: Fn(&Array1<f64>) -> Result<Array1<f64>, RobustError>,
{
    let mut beta = beta0;
    let mut sc = score(&beta)?;
    for _ in 0..max_steps {
        let beta_next = match cstep(&beta) {
            Ok(b) => b,
            Err(_) => break,
        };
        let sc_next = match score(&beta_next) {
            Ok(s) => s,
            Err(_) => break,
        };
        let converged = (sc - sc_next).abs() <= tol * sc;
        beta = beta_next;
        sc = sc_next;
        if converged {
            break;
        }
    }
    Ok((beta, sc))
}
