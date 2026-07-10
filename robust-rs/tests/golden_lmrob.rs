//! Golden cross-check of the high-breakdown estimators (MM, LTS) against R
//! `robustbase` (`lmrob`, `ltsReg`) on `starsCYG`: the formal v0.2 exit gate.
//!
//! These are **`#[ignore]`d and pending**: R was unavailable in the environment
//! this crate was built in, so the fixture doesn't exist yet. The harness is in
//! place; generate the fixture where R lives and run the ignored tests:
//!
//! ```text
//! Rscript scripts/gen_golden_r.R
//! cargo test --workspace -- --ignored
//! ```
//!
//! Tolerance and why it's provisional: `robustbase`'s subsampling RNG differs
//! from robust-rs' ChaCha8 sub-streams, so even when both converge to the same
//! optimum the point estimates are not bit-identical. The checks below assert
//! *closeness* (same optimum, different random starts), not equality. The
//! epsilons are a starting point to be **calibrated on the first real run**; a
//! failure most likely means a tuning/convergence misalignment or a genuinely
//! different local optimum to investigate, not necessarily a bug. Until then
//! the in-environment validation is the internal anchors
//! plus the statsmodels M-spine golden (`golden_rlm`).

use ndarray::{Array1, Array2};
use robust_rs::prelude::*;

fn stars_with_intercept() -> (Array2<f64>, Array1<f64>) {
    let (x_raw, y) = robust_rs::datasets::stars_cyg();
    let n = x_raw.nrows();
    let mut x = Array2::ones((n, 2));
    x.column_mut(1).assign(&x_raw.column(0));
    (x, y)
}

fn fixture() -> serde_json::Value {
    let json = std::fs::read_to_string("tests/fixtures/lmrob_starsCYG.json").expect(
        "robustbase fixture missing; run `Rscript scripts/gen_golden_r.R` (needs R + robustbase)",
    );
    serde_json::from_str(&json).unwrap()
}

#[test]
#[ignore = "pending robustbase fixture"]
fn mm_matches_lmrob_on_starscyg() {
    let reference = fixture();
    let ref_coef = reference["starsCYG_mm"]["coefficients"].as_array().unwrap();
    let (ref_intercept, ref_slope) = (ref_coef[0].as_f64().unwrap(), ref_coef[1].as_f64().unwrap());

    let (x, y) = stars_with_intercept();
    let mm = MMEstimator::default().fit(&x, &y).unwrap();

    // Provisional tolerances: calibrate on the first real run.
    approx::assert_abs_diff_eq!(mm.coefficients()[1], ref_slope, epsilon = 0.15);
    approx::assert_abs_diff_eq!(mm.coefficients()[0], ref_intercept, epsilon = 1.0);
}

#[test]
#[ignore = "pending robustbase fixture"]
fn lts_matches_ltsreg_on_starscyg() {
    let reference = fixture();
    let ref_coef = reference["starsCYG_lts"]["coefficients"]
        .as_array()
        .unwrap();
    let ref_slope = ref_coef[1].as_f64().unwrap();

    let (x, y) = stars_with_intercept();
    let lts = Lts::default().fit(&x, &y).unwrap();

    // Provisional tolerance: calibrate on the first real run.
    approx::assert_abs_diff_eq!(lts.coefficients()[1], ref_slope, epsilon = 0.25);
}
