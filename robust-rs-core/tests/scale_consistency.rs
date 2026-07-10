//! Robust scale estimators: Fisher-consistency for σ at the Gaussian (large n),
//! finite-sample-correction golden checks (small n) and S-scale preconditions.

use approx::assert_relative_eq;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use robust_rs_core::error::RobustError;
use robust_rs_core::rho::{Cauchy, Huber, TukeyBiweight};
use robust_rs_core::scale::{Mad, Qn, SScale, ScaleEstimator, Sn};

/// A reproducible `N(0, σ)` sample.
fn gaussian_sample(sigma: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, sigma).unwrap();
    (0..n).map(|_| normal.sample(&mut rng)).collect()
}

#[test]
fn mad_is_consistent_for_sigma_at_the_gaussian() {
    let data = gaussian_sample(2.5, 50_000, 42);
    let s = Mad::default().scale(&data).unwrap();
    assert_relative_eq!(s.get(), 2.5, epsilon = 0.1);
}

#[test]
fn sscale_is_consistent_for_sigma_at_the_gaussian() {
    let data = gaussian_sample(2.5, 50_000, 42);
    // S-step biweight tuning c ≈ 1.547; `fisher_consistent` sets δ = E_Φ[ρ]
    // (which for this c equals ρ_sup/2, the 50%-breakdown target).
    let sscale = SScale::fisher_consistent(TukeyBiweight::new(1.547).unwrap(), 128).unwrap();
    let s = sscale.scale(&data).unwrap();
    assert_relative_eq!(s.get(), 2.5, epsilon = 0.1);
}

#[test]
fn qn_is_consistent_for_sigma_at_the_gaussian() {
    // O(n²) pairwise differences ⇒ a moderate sample keeps the table small.
    let data = gaussian_sample(2.5, 2_000, 42);
    let s = Qn::default().scale(&data).unwrap();
    assert_relative_eq!(s.get(), 2.5, epsilon = 0.15);
}

#[test]
fn sn_is_consistent_for_sigma_at_the_gaussian() {
    let data = gaussian_sample(2.5, 2_000, 42);
    let s = Sn::default().scale(&data).unwrap();
    assert_relative_eq!(s.get(), 2.5, epsilon = 0.15);
}

// --- Small-n finite-sample-correction golden checks --------------------------
//
// The large-n tests above are blind to a broken dₙ/cₙ: at n = 2000 the
// correction is ~1/n from 1, well inside their tolerance, so an off-by-one in
// the table index or the wrong even/odd branch would pass unnoticed, yet the
// small-n case is the entire point of the correction. These pin hand-derived
// values on x = 1..=8 (n = 8, the `n ≤ 9` *table* branch), computed directly
// from the Croux & Rousseeuw (1992) definitions. (R/robustbase is unavailable
// in this environment; the derivations are spelled out so a reviewer can check
// the constants independently of the implementation.) Each test also pins the
// *uncorrected* value, so the correction is proven to fire rather than no-op.

#[test]
fn qn_small_sample_correction_golden() {
    let x: Vec<f64> = (1..=8).map(|i| i as f64).collect();
    // 28 pairwise |diffs|; value d occurs (8−d) times for d = 1..7.
    // k = C(⌊8/2⌋+1, 2) = C(5,2) = 10 ⇒ 10th smallest = 2. Uncorrected = 2.2219·2.
    let uncorrected = Qn {
        finite_sample_correction: false,
        ..Qn::default()
    };
    assert_relative_eq!(
        uncorrected.scale(&x).unwrap().get(),
        2.2219 * 2.0,
        epsilon = 1e-9
    );
    // d₈ = 0.669 (table) ⇒ 2.2219 · 0.669 · 2 = 2.972902.
    assert_relative_eq!(
        Qn::default().scale(&x).unwrap().get(),
        2.972_902,
        epsilon = 1e-4
    );
}

#[test]
fn sn_small_sample_correction_golden() {
    let x: Vec<f64> = (1..=8).map(|i| i as f64).collect();
    // Per-point high-medians a = {4,3,2,2,2,2,3,4}; low-median = 2.
    // Uncorrected = 1.1926·2.
    let uncorrected = Sn {
        finite_sample_correction: false,
        ..Sn::default()
    };
    assert_relative_eq!(
        uncorrected.scale(&x).unwrap().get(),
        1.1926 * 2.0,
        epsilon = 1e-9
    );
    // c₈ = 1.005 (table) ⇒ 1.1926 · 1.005 · 2 = 2.397126.
    assert_relative_eq!(
        Sn::default().scale(&x).unwrap().get(),
        2.397_126,
        epsilon = 1e-4
    );
}

// --- S-scale precondition: a bounded loss is required ------------------------

#[test]
fn sscale_rejects_unbounded_loss() {
    // Huber's ρ is unbounded ⇒ zero-breakdown "scale"; must fail loudly.
    assert_eq!(
        SScale::new(Huber::default(), 0.5).unwrap_err(),
        RobustError::UnboundedLoss
    );
    // Cauchy redescends but keeps ρ unbounded (rho_sup = None) ⇒ also rejected.
    assert_eq!(
        SScale::new(Cauchy::default(), 0.1).unwrap_err(),
        RobustError::UnboundedLoss
    );
    // A bounded redescender is accepted.
    assert!(SScale::new(TukeyBiweight::new(1.547).unwrap(), 0.19).is_ok());
}
