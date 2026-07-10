//! Weighted least squares via a QR factorization (the numerical core of IRLS).

use ndarray::{Array1, Array2};
use robust_rs_core::error::RobustError;

use faer::prelude::*; // Mat, col_piv_qr, Solve/SolveLstsq, (i, j) indexing

/// Solve `argmin_β Σ wᵢ (yᵢ − xᵢ·β)²` for non-negative weights `w`, via a QR
/// factorization of `√W · X` (never by forming `XᵀWX`), using `faer` internally.
pub fn weighted_least_squares(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
) -> Result<Array1<f64>, RobustError> {
    let (n, p) = x.dim();

    if y.len() != n {
        return Err(RobustError::DimensionMismatch {
            expected: n,
            got: y.len(),
        });
    }
    if w.len() != n {
        return Err(RobustError::DimensionMismatch {
            expected: n,
            got: w.len(),
        });
    }
    if p == 0 {
        return Err(RobustError::SingularDesign);
    }
    if n < p {
        // fewer observations than parameters ⇒ rank < p
        return Err(RobustError::InsufficientData { needed: p, got: n });
    }
    if let Some(&bad) = w.iter().find(|&&wi| !wi.is_finite() || wi < 0.0) {
        return Err(RobustError::InvalidWeight { value: bad });
    }

    // B = √W·X and c = √W·y, built straight into faer's column-major storage.
    // wᵢ == 0 zeroes row i, i.e. drops that observation.
    let sqrt_w: Vec<f64> = w.iter().map(|&wi| wi.sqrt()).collect();
    let b = Mat::from_fn(n, p, |i, j| sqrt_w[i] * x[[i, j]]);
    let c = Mat::from_fn(n, 1, |i, _| sqrt_w[i] * y[i]);

    // Rank-revealing QR:  B·Pᵀ = Q·R.
    let qr = b.col_piv_qr();

    // faer's ColPivQr has no `.rank()`, so read deficiency off R's diagonal.
    let r = qr.R();
    let diag_max = (0..p).map(|k| r[(k, k)].abs()).fold(0.0_f64, f64::max);
    let tol = diag_max * (n.max(p) as f64) * f64::EPSILON;
    if !(0..p).all(|k| r[(k, k)].abs() > tol) {
        return Err(RobustError::SingularDesign);
    }

    // LS solve (applies the pivot internally). For an overdetermined rhs (n×1),
    // the solution lands in the top p rows of the returned matrix.
    let sol = qr.solve_lstsq(&c);

    Ok(Array1::from_iter((0..p).map(|k| sol[(k, 0)])))
}
