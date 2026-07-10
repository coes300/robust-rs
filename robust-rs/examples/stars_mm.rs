//! Hertzsprung–Russell stars (CYG OB1): four giant stars are high-*leverage*
//! outliers. OLS is pulled toward them and a Huber M-estimator resists them
//! only partially, because a plain M-estimator has ~0 breakdown against
//! leverage; both return a physically wrong *negative* main-sequence slope. The
//! MM-estimator (R's `lmrob` default: 50% breakdown, ≈95% efficiency) recovers
//! the positive slope and drives the four giants to near-zero weight.
//! Run with `cargo run --example stars_mm`.

use ndarray::Array2;
use robust_rs::prelude::*;

fn main() {
    let (x_raw, y) = robust_rs::datasets::stars_cyg(); // 47 × 1 (log.Te)
    let n = x_raw.nrows();
    let mut x = Array2::ones((n, 2)); // prepend an intercept column
    x.column_mut(1).assign(&x_raw.column(0));

    let ols = MEstimator::new(LeastSquares, Mad::default())
        .fit(&x, &y)
        .unwrap();
    let huber = MEstimator::new(Huber::default(), Mad::default())
        .fit(&x, &y)
        .unwrap();
    let mm = MMEstimator::default().fit(&x, &y).unwrap(); // reproducible (fixed default seed)

    for (name, fit) in [("OLS", &ols), ("Huber M", &huber), ("MM", &mm)] {
        println!(
            "{name:8}: intercept={:7.3}  slope={:+.3}",
            fit.coefficients()[0],
            fit.coefficients()[1]
        );
    }

    println!(
        "\nOLS and Huber slope < 0 (dragged by the giants); MM slope > 0: {}",
        mm.coefficients()[1] > 0.0
    );

    println!("\nMM weights (the four giant stars are driven to ≈0):");
    let mut flagged = 0usize;
    for i in 0..n {
        if mm.weights[i] < 0.01 {
            println!("  obs {:2}: weight = {:.4}", i + 1, mm.weights[i]);
            flagged += 1;
        }
    }
    println!(
        "  {flagged} observations rejected; MM breakdown point = {:.2}",
        mm.breakdown_point()
    );
}
