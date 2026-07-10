//! Golden cross-check of FAST-MCD against R `robustbase::covMcd` on the
//! stack-loss operating variables: the formal v0.3 exit gate.
//!
//! These are **`#[ignore]`d and pending**: R was unavailable in the environment
//! this crate was built in, so the fixture doesn't exist yet. The harness is in
//! place; generate the fixture where R lives and run the ignored tests:
//!
//! ```text
//! Rscript scripts/gen_golden_covmcd.R
//! cargo test --workspace -- --ignored
//! ```
//!
//! Tolerances and why they're provisional: (1) `covMcd`'s subsampling RNG
//! differs from robust-rs' ChaCha8 sub-streams, so estimates from independent
//! random starts are not bit-identical; (2) `covMcd` applies a small-sample
//! correction factor that robust-rs currently defers (only the asymptotic
//! consistency factor is applied), so the covariances can differ by a scalar of
//! a few percent at this `n`. The centre is checked to a tight absolute
//! tolerance; the covariance is checked *scale-free* (as a correlation matrix),
//! which is invariant to the deferred multiplier. Calibrate the epsilons on the
//! first real run. Until then, the in-environment validation is the affine-
//! equivariance and Gaussian-consistency anchors in `tests/multivariate.rs`
//!

use ndarray::Array2;
use robust_rs::multivariate::Mcd;

fn fixture() -> serde_json::Value {
    let json = std::fs::read_to_string("tests/fixtures/covmcd_stackloss.json").expect(
        "covMcd fixture missing; run `Rscript scripts/gen_golden_covmcd.R` (needs R + robustbase)",
    );
    serde_json::from_str(&json).unwrap()
}

fn stackloss_x() -> Array2<f64> {
    robust_rs::datasets::stackloss().0
}

/// Turn a flat row-major `p·p` vector into a correlation matrix (scale-free).
fn correlation(cov_flat: &[f64], p: usize) -> Vec<f64> {
    let at = |i: usize, j: usize| cov_flat[i * p + j];
    let sd: Vec<f64> = (0..p).map(|i| at(i, i).sqrt()).collect();
    let mut r = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            r[i * p + j] = at(i, j) / (sd[i] * sd[j]);
        }
    }
    r
}

#[test]
#[ignore = "pending robustbase fixture; see scripts/gen_golden_covmcd.R and docs/validation.md"]
fn mcd_matches_covmcd_on_stackloss() {
    let reference = fixture();
    let refm = &reference["stackloss_mcd"];
    let p = refm["p"].as_u64().unwrap() as usize;

    let ref_center: Vec<f64> = refm["center"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let ref_cov: Vec<f64> = refm["cov"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let x = stackloss_x();
    let fit = Mcd::new().seed(1).fit(&x).unwrap();

    // Centre: tight absolute tolerance (provisional).
    for (j, &r) in ref_center.iter().enumerate() {
        approx::assert_abs_diff_eq!(fit.location()[j], r, epsilon = 1.0);
    }

    // Covariance: compare as correlation matrices (invariant to the deferred
    // small-sample scale factor).
    let got_cov: Vec<f64> = fit.scatter().iter().copied().collect();
    let got_corr = correlation(&got_cov, p);
    let ref_corr = correlation(&ref_cov, p);
    for (g, r) in got_corr.iter().zip(ref_corr.iter()) {
        approx::assert_abs_diff_eq!(g, r, epsilon = 0.1);
    }
}
