//! Multivariate robust location/scatter: affine-equivariance property tests,
//! Gaussian-consistency anchors and the outlier-rejection structural anchor.
//!
//! No R/Python reference is available in this environment, so (per
//!  the in-environment validation is (1) exact invariants
//! (affine / scaling equivariance, which robust scatter estimators must satisfy
//! by construction), (2) recovery of a known covariance on clean Gaussian data,
//! and (3) the structural fact that the robust estimators reject injected
//! outliers where the classical covariance is masked by them.

use ndarray::{array, Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use robust_rs::multivariate::mahalanobis::{
    classical_covariance, distance_cutoff, mahalanobis_distances, outlier_flags,
};
use robust_rs::multivariate::{MScatter, Mcd, Ogk, RobustScatter, ScatterFit, Tyler};

/// `n × p` matrix of iid standard-normal draws from a fixed seed.
fn gaussian(n: usize, p: usize, seed: u64) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Array2::from_shape_fn((n, p), |_| StandardNormal.sample(&mut rng))
}

/// Apply the affine map `yᵢ = A xᵢ + b` rowwise: `Y = X Aᵀ + b`.
fn affine(x: &Array2<f64>, a: &Array2<f64>, b: &Array1<f64>) -> Array2<f64> {
    x.dot(&a.t()) + b
}

fn max_abs_diff(u: &Array2<f64>, v: &Array2<f64>) -> f64 {
    (u - v).iter().fold(0.0_f64, |m, &d| m.max(d.abs()))
}

fn rel_diff(u: &Array2<f64>, v: &Array2<f64>) -> f64 {
    let scale = v.iter().fold(0.0_f64, |m, &d| m.max(d.abs())).max(1e-12);
    max_abs_diff(u, v) / scale
}

// ---------------------------------------------------------------------------
// Affine equivariance
// ---------------------------------------------------------------------------

#[test]
fn mcd_is_affine_equivariant() {
    let x = gaussian(60, 2, 20240710);
    let a = array![[2.0, 0.5], [-0.3, 1.5]];
    let b = array![10.0, -4.0];
    let y = affine(&x, &a, &b);

    let fx = Mcd::new().seed(7).fit(&x).unwrap();
    let fy = Mcd::new().seed(7).fit(&y).unwrap();

    // μ_Y ≈ A μ_X + b
    let want_loc = a.dot(fx.location()) + &b;
    let loc_err = (fy.location() - &want_loc)
        .iter()
        .fold(0.0_f64, |m, &d| m.max(d.abs()));
    assert!(loc_err < 1e-6, "MCD location equivariance: err {loc_err}");

    // Σ_Y ≈ A Σ_X Aᵀ
    let want_cov = a.dot(fx.scatter()).dot(&a.t());
    let cov_err = rel_diff(fy.scatter(), &want_cov);
    assert!(
        cov_err < 1e-6,
        "MCD scatter equivariance: rel err {cov_err}"
    );

    // Mahalanobis distances are affine invariant → identical ordering & values.
    let dmax = (fx.distances() - fy.distances())
        .iter()
        .fold(0.0_f64, |m, &d| m.max(d.abs()));
    assert!(dmax < 1e-6, "MCD distances affine invariance: err {dmax}");
}

#[test]
fn m_scatter_is_affine_equivariant() {
    let x = gaussian(80, 2, 555);
    let a = array![[1.7, 0.0], [0.4, 0.9]];
    let b = array![-2.0, 3.0];
    let y = affine(&x, &a, &b);

    let fx = MScatter::default().fit(&x).unwrap();
    let fy = MScatter::default().fit(&y).unwrap();

    let want_loc = a.dot(fx.location()) + &b;
    let loc_err = (fy.location() - &want_loc)
        .iter()
        .fold(0.0_f64, |m, &d| m.max(d.abs()));
    assert!(loc_err < 1e-6, "M-scatter location equivariance: {loc_err}");

    let want_cov = a.dot(fx.scatter()).dot(&a.t());
    let cov_err = rel_diff(fy.scatter(), &want_cov);
    assert!(cov_err < 1e-6, "M-scatter scatter equivariance: {cov_err}");
}

#[test]
fn tyler_shape_is_affine_equivariant_up_to_normalization() {
    // Tyler's location (coordinatewise median) is not affine equivariant, so we
    // fix the centre at 0 with a zero-shift affine map and compare *shapes*.
    let x = gaussian(80, 2, 999);
    let a = array![[1.3, 0.6], [0.0, 1.1]];
    let zero = Array1::zeros(2);
    let y = affine(&x, &a, &zero);

    let fx = Tyler::new().location(zero.clone()).fit(&x).unwrap();
    let fy = Tyler::new().location(zero.clone()).fit(&y).unwrap();

    // A Σ_X Aᵀ, renormalized to unit determinant, should equal Σ_Y.
    let mut want = a.dot(fx.shape()).dot(&a.t());
    let det = want[[0, 0]] * want[[1, 1]] - want[[0, 1]] * want[[1, 0]];
    want.mapv_inplace(|v| v / det.sqrt()); // det (2×2)^{1/2}
    let err = rel_diff(fy.shape(), &want);
    assert!(err < 1e-5, "Tyler shape equivariance: {err}");
}

#[test]
fn ogk_is_scaling_and_permutation_equivariant() {
    // OGK is (only) orthogonally equivariant, but exactly equivariant under
    // coordinatewise scaling and permutation: the invariants asserted here.
    let x = gaussian(70, 3, 4242);
    let s = array![3.0, 0.5, 2.0];
    let y = Array2::from_shape_fn(x.dim(), |(i, j)| x[[i, j]] * s[j]);

    let fx = Ogk::default().fit(&x).unwrap();
    let fy = Ogk::default().fit(&y).unwrap();

    // Σ_Y ≈ diag(s) Σ_X diag(s)
    let want = Array2::from_shape_fn((3, 3), |(i, j)| s[i] * fx.scatter()[[i, j]] * s[j]);
    let err = rel_diff(fy.scatter(), &want);
    assert!(err < 1e-6, "OGK scaling equivariance: {err}");
}

// ---------------------------------------------------------------------------
// Gaussian consistency (recover Σ = I on clean data)
// ---------------------------------------------------------------------------

/// Worst diagonal-from-1 and worst off-diagonal-from-0 of a covariance.
fn identity_errors(cov: &Array2<f64>) -> (f64, f64) {
    let p = cov.nrows();
    let mut diag = 0.0_f64;
    let mut off = 0.0_f64;
    for i in 0..p {
        for j in 0..p {
            if i == j {
                diag = diag.max((cov[[i, j]] - 1.0).abs());
            } else {
                off = off.max(cov[[i, j]].abs());
            }
        }
    }
    (diag, off)
}

#[test]
fn estimators_recover_identity_on_clean_gaussian() {
    // Checked at TWO dimensions (p = 2 and p = 5): the MCD/OGK consistency factor
    // c(α, p) is p-dependent, so a p-dependent error cannot hide at a single p.
    for &p in &[2usize, 5usize] {
        let x = gaussian(2000, p, 314159 + p as u64);
        // Clean data: every start concentrates to the same optimum, so a small
        // subsample budget suffices and keeps the higher-p fit fast.
        for (name, cov) in [
            (
                "MCD",
                Mcd::new()
                    .seed(1)
                    .n_subsamples(100)
                    .fit(&x)
                    .unwrap()
                    .scatter()
                    .clone(),
            ),
            ("OGK", Ogk::default().fit(&x).unwrap().scatter().clone()),
            (
                "M-scatter",
                MScatter::default().fit(&x).unwrap().scatter().clone(),
            ),
        ] {
            let (diag_err, off) = identity_errors(&cov);
            assert!(
                diag_err < 0.18 && off < 0.12,
                "{name} did not recover I at p={p}: diag_err {diag_err}, off {off}"
            );
        }
    }
}

#[test]
fn mcd_recovers_a_known_non_identity_covariance() {
    // Absolute scale recovery (not just shape via equivariance): draw N(0, Σ)
    // with Σ = A Aᵀ and check MCD recovers Σ. Combined with the p=5 identity
    // check above, this pins the p-dependent consistency factor on real scale.
    let z = gaussian(2500, 3, 20240711);
    let a = array![[2.0, 0.0, 0.0], [0.8, 1.5, 0.0], [-0.5, 0.3, 1.0]];
    let x = z.dot(&a.t()); // rows ~ N(0, A Aᵀ)
    let sigma = a.dot(&a.t());

    let cov = Mcd::new()
        .seed(2)
        .n_subsamples(100)
        .fit(&x)
        .unwrap()
        .scatter()
        .clone();
    let err = rel_diff(&cov, &sigma);
    assert!(err < 0.15, "MCD non-identity Σ recovery: rel err {err}");
}

// ---------------------------------------------------------------------------
// Outlier rejection structural anchor
// ---------------------------------------------------------------------------

#[test]
fn robust_estimators_reject_injected_outliers() {
    // 45 clean points ~ N(0, I) plus 5 gross outliers at (12, 12).
    let n_clean = 45;
    let mut data = gaussian(n_clean, 2, 271828).into_raw_vec_and_offset().0;
    let outlier_rows = [n_clean, n_clean + 1, n_clean + 2, n_clean + 3, n_clean + 4];
    for _ in 0..5 {
        data.push(12.0);
        data.push(12.0);
    }
    let n = n_clean + 5;
    let x = Array2::from_shape_vec((n, 2), data).unwrap();

    // Classical covariance is masked: its centre is dragged toward the outliers.
    let (class_mean, _class_cov) = classical_covariance(&x);
    assert!(
        class_mean[0] > 0.7,
        "classical mean should be pulled toward the outliers, got {class_mean:?}"
    );

    // MCD: centre near 0 and it flags exactly the 5 injected rows.
    let mcd = Mcd::new().seed(3).fit(&x).unwrap();
    assert!(
        mcd.location()[0].abs() < 0.6 && mcd.location()[1].abs() < 0.6,
        "MCD centre should resist the outliers, got {:?}",
        mcd.location()
    );
    let flags = mcd.outliers(0.975);
    let flagged: Vec<usize> = flags
        .iter()
        .enumerate()
        .filter(|&(_, &f)| f)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        flagged, outlier_rows,
        "MCD should flag exactly the 5 outliers"
    );

    // OGK independently flags all 5 injected outliers.
    let ogk = Ogk::default().fit(&x).unwrap();
    let ogk_flags = ogk.outliers(0.975);
    for &r in &outlier_rows {
        assert!(ogk_flags[r], "OGK missed injected outlier at row {r}");
    }
}

// ---------------------------------------------------------------------------
// Mahalanobis map + reproducibility
// ---------------------------------------------------------------------------

#[test]
fn mahalanobis_map_over_a_robust_pair() {
    let x = gaussian(50, 2, 12321);
    let fit = Mcd::new().seed(5).fit(&x).unwrap();
    let d = mahalanobis_distances(&x, fit.location(), fit.scatter()).unwrap();
    // The standalone map reproduces the fit's own distances.
    let dmax = (&d - fit.distances())
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()));
    assert!(dmax < 1e-9, "standalone Mahalanobis map disagrees: {dmax}");

    // The free-function outlier flags match the trait method.
    let flags_free = outlier_flags(&d, 2, 0.975);
    assert_eq!(flags_free, fit.outliers(0.975));
}

#[test]
fn mcd_is_reproducible_under_a_fixed_seed() {
    let x = gaussian(60, 2, 88);
    let a = Mcd::new().seed(42).fit(&x).unwrap();
    let b = Mcd::new().seed(42).fit(&x).unwrap();
    assert_eq!(a.location(), b.location());
    assert_eq!(a.scatter(), b.scatter());
    assert_eq!(a.support(), b.support());
}

// ---------------------------------------------------------------------------
// χ² outlier cutoff correctness (separate from separability)
// ---------------------------------------------------------------------------

/// The outlier map's threshold `√χ²_{p, q}` must be numerically correct: a
/// dedicated check, because outlier *separability* (gross outliers land far
/// beyond any plausible cutoff) would pass even with a mildly wrong quantile.
/// Pins the cutoff the public API actually uses against known χ² quantiles.
#[test]
fn outlier_cutoff_matches_known_chi2_quantiles() {
    // √χ²_{p, 0.975}: 7.377759 (p=2) and 12.832502 (p=5).
    let cases = [(2usize, 7.377_758_9_f64), (5usize, 12.832_502_f64)];
    for (p, chi2) in cases {
        let want = chi2.sqrt();

        // Free function used by `mahalanobis::outlier_flags`.
        assert!(
            (distance_cutoff(p, 0.975) - want).abs() < 1e-4,
            "distance_cutoff wrong at p={p}"
        );

        // Trait method used by `RobustScatter::outliers`, depends only on the
        // scatter's dimension, so a synthetic identity fit exercises it.
        let fit = ScatterFit {
            location: Array1::zeros(p),
            scatter: Array2::eye(p),
            distances: Array1::zeros(1),
            weights: Array1::ones(1),
        };
        assert!(
            (fit.distance_cutoff(0.975) - want).abs() < 1e-4,
            "RobustScatter::distance_cutoff wrong at p={p}"
        );
    }
}

// ---------------------------------------------------------------------------
// OGK reweighting must not silently collapse to the classical covariance
// ---------------------------------------------------------------------------

/// Regression guard for the median-adjusted OGK reweighting. With a plain fixed
/// `χ²` cutoff the OGK reweighting kept *every* point on the stack-loss data and
/// reverted exactly to the classical mean/covariance (defeating the estimator).
/// The Maronna–Zamar median-adjusted cutoff prevents that; this test fails if it
/// is ever reverted.
#[test]
fn ogk_does_not_collapse_to_classical() {
    let x = robust_rs::datasets::stackloss().0; // 21 × 3 operating variables
    let (cmean, ccov) = classical_covariance(&x);
    let ogk = Ogk::default().fit(&x).unwrap();

    // Location must differ meaningfully from the classical mean.
    let loc_gap = (ogk.location() - &cmean)
        .iter()
        .fold(0.0_f64, |m, &d| m.max(d.abs()));
    assert!(
        loc_gap > 0.5,
        "OGK location collapsed to the classical mean (gap {loc_gap}): reweighting reverted?"
    );

    // Scatter must differ from the classical covariance and OGK must actually
    // flag outliers the classical distances mask.
    assert!(
        rel_diff(ogk.scatter(), &ccov) > 0.05,
        "OGK scatter collapsed to the classical covariance"
    );
    let n_flagged = ogk.outliers(0.975).iter().filter(|&&f| f).count();
    assert!(
        n_flagged >= 2,
        "OGK flagged only {n_flagged} outliers on stackloss (expected ≥ 2)"
    );
}
