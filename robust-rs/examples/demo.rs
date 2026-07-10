//! End-to-end demo: OLS is dragged by gross y-outliers; a Huber M-estimator
//! recovers the true line, down-weights the outliers and reports its own
//! efficiency, coefficient standard errors and (bounded) influence function.
//! Run with `cargo run --example demo`.

use ndarray::{Array1, Array2};
use robust_rs::prelude::*;

fn main() {
    // y = 2 + 3x, then corrupt three responses with gross outliers.
    let n = 20usize;
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut y: Array1<f64> = xs.iter().map(|&xi| 2.0 + 3.0 * xi).collect();
    y[5] = 100.0;
    y[12] = -50.0;
    y[18] = 120.0;

    let mut x: Array2<f64> = Array2::ones((n, 2)); // [1, x]
    for i in 0..n {
        x[[i, 1]] = xs[i];
    }

    let ols = MEstimator::new(LeastSquares, Mad::default())
        .fit(&x, &y)
        .unwrap();
    let huber = MEstimator::new(Huber::default(), Mad::default())
        .fit(&x, &y)
        .unwrap();

    println!("true coefficients   : intercept=2.000  slope=3.000");
    println!(
        "OLS coefficients    : intercept={:.3}  slope={:.3}   (dragged by outliers)",
        ols.coefficients()[0],
        ols.coefficients()[1]
    );
    println!(
        "Huber M coefficients: intercept={:.3}  slope={:.3}",
        huber.coefficients()[0],
        huber.coefficients()[1]
    );

    println!("\nHuber weights (≈0 ⇒ treated as an outlier):");
    for &i in &[5usize, 12, 18] {
        println!("  obs {i:2}: weight = {:.4}", huber.weights[i]);
    }
    let clean_min = (0..n)
        .filter(|i| ![5usize, 12, 18].contains(i))
        .map(|i| huber.weights[i])
        .fold(f64::INFINITY, f64::min);
    println!("  clean observations: min weight = {clean_min:.4}");

    println!("\nGaussian efficiency : {:.4}", huber.gaussian_efficiency());
    println!("asymptotic variance : {:.4}", huber.asymptotic_variance());
    let cov = huber.coef_covariance(&x);
    println!(
        "coef std errors     : intercept={:.4}  slope={:.4}",
        cov[[0, 0]].sqrt(),
        cov[[1, 1]].sqrt()
    );

    let psi = huber.influence_function();
    println!("\ninfluence function IF(r) at standardized residual r (bounded):");
    for r in [0.5, 2.0, 10.0, 100.0] {
        println!("  IF({r:6.1}) = {:.4}", psi(r));
    }
}
