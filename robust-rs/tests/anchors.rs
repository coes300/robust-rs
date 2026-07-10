//! Exactly-known anchors: the estimators must reproduce their classical limits.

use approx::assert_relative_eq;
use ndarray::{array, Array1, Array2};
use robust_rs::prelude::*;

/// Ordinary least squares via the normal equations, computed independently so
/// the test does not depend on the crate under test.
fn ols_reference(x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);
    solve_dense(&xtx, &xty)
}

/// Tiny Gauss–Jordan solver for the small, well-conditioned systems in this test.
fn solve_dense(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut m = a.clone();
    let mut x = b.clone();
    for i in 0..n {
        let piv = m[[i, i]];
        for j in i..n {
            m[[i, j]] /= piv;
        }
        x[i] /= piv;
        for k in 0..n {
            if k != i {
                let f = m[[k, i]];
                for j in i..n {
                    m[[k, j]] -= f * m[[i, j]];
                }
                x[k] -= f * x[i];
            }
        }
    }
    x
}

#[test]
fn least_squares_m_estimator_reproduces_ols() {
    // A small design with an intercept column.
    let x: Array2<f64> = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]];
    let y: Array1<f64> = array![1.1, 1.9, 3.2, 3.9, 5.1];

    let expected = ols_reference(&x, &y);
    let fit = MEstimator::new(LeastSquares, Mad::default())
        .fit(&x, &y)
        .expect("fit");
    for (a, b) in fit.coefficients.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-6);
    }
}

#[test]
fn huber_location_recovers_central_value() {
    // With no outliers, a Huber location sits at the centre.
    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let fit = m_location(
        &data,
        &Huber::default(),
        &Mad::default(),
        &Control::default(),
    )
    .expect("fit");
    assert_relative_eq!(fit.estimate, 3.0, epsilon = 1e-6);
}
