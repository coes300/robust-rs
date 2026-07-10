//! Small dense linear-algebra helpers for the multivariate estimators, bridging
//! `ndarray` storage to `faer` factorizations.
//!
//! Everything here operates on `p × p` scatter matrices with `p` (the number of
//! variables) small, so the emphasis is legibility over micro-optimization. The
//! numerically delicate operations (the SPD inverse and log-determinant used in
//! every Mahalanobis evaluation and MCD determinant comparison) go through a
//! Cholesky factorization (`faer` `Llt`), which both detects a non-positive-
//! definite (rank-deficient) scatter cleanly and yields a stable
//! `log det = 2 Σ ln Lᵢᵢ` without forming a determinant that could under/overflow.

use faer::linalg::solvers::DenseSolveCore;
use faer::{Mat, Side};
use ndarray::{Array1, Array2, Axis};
use robust_rs_core::error::RobustError;

/// Copy an `ndarray` matrix into `faer` column-major storage.
pub(crate) fn to_faer(a: &Array2<f64>) -> Mat<f64> {
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[[i, j]])
}

/// Copy a `faer` matrix back into an `ndarray` array.
pub(crate) fn from_faer(m: &Mat<f64>) -> Array2<f64> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)])
}

/// Column means of the `n × p` data matrix `x` (the `p`-vector of variable
/// means). Panics only on empty input, which every caller rules out first.
pub(crate) fn mean(x: &Array2<f64>) -> Array1<f64> {
    x.mean_axis(Axis(0)).expect("non-empty data")
}

/// Rows of `x` centered by subtracting the `p`-vector `loc`.
pub(crate) fn center(x: &Array2<f64>, loc: &Array1<f64>) -> Array2<f64> {
    let (n, p) = x.dim();
    Array2::from_shape_fn((n, p), |(i, j)| x[[i, j]] - loc[j])
}

/// Sample mean and (unbiased, `m − 1` denominator) covariance of the data `x`.
/// With a single row the covariance is zero.
pub(crate) fn mean_covariance(x: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let (m, _p) = x.dim();
    let mu = mean(x);
    let centered = center(x, &mu);
    let denom = if m > 1 { (m - 1) as f64 } else { 1.0 };
    let cov = centered.t().dot(&centered) / denom;
    (mu, cov)
}

/// The SPD inverse of `a` and its log-determinant, via a Cholesky factorization.
/// Returns [`RobustError::SingularDesign`] if `a` is not (numerically) positive
/// definite, the signal, in FAST-MCD, that a subset spans a lower-dimensional
/// affine subspace (an exact fit).
pub(crate) fn spd_inverse_logdet(a: &Array2<f64>) -> Result<(Array2<f64>, f64), RobustError> {
    let p = a.nrows();
    let llt = to_faer(a)
        .llt(Side::Lower)
        .map_err(|_| RobustError::SingularDesign)?;
    let l = llt.L();
    let mut logdet = 0.0;
    for i in 0..p {
        logdet += l[(i, i)].ln();
    }
    logdet *= 2.0;
    if !logdet.is_finite() {
        return Err(RobustError::SingularDesign);
    }
    Ok((from_faer(&llt.inverse()), logdet))
}

/// The log-determinant of an SPD matrix `a` (Cholesky diagonal). Errors as in
/// [`spd_inverse_logdet`].
pub(crate) fn spd_logdet(a: &Array2<f64>) -> Result<f64, RobustError> {
    let p = a.nrows();
    let llt = to_faer(a)
        .llt(Side::Lower)
        .map_err(|_| RobustError::SingularDesign)?;
    let l = llt.L();
    let mut logdet = 0.0;
    for i in 0..p {
        logdet += l[(i, i)].ln();
    }
    logdet *= 2.0;
    if logdet.is_finite() {
        Ok(logdet)
    } else {
        Err(RobustError::SingularDesign)
    }
}

/// Squared Mahalanobis distances `dᵢ² = (xᵢ − loc)ᵀ Σ⁻¹ (xᵢ − loc)` for every
/// row of `x`, given the precomputed inverse scatter `inv = Σ⁻¹`.
pub(crate) fn mahalanobis_sq(x: &Array2<f64>, loc: &Array1<f64>, inv: &Array2<f64>) -> Array1<f64> {
    // Row-wise dot of `centered` and `centered · Σ⁻¹` gives the quadratic form.
    let centered = center(x, loc); // (n, p)
    let tmp = centered.dot(inv); // (n, p) · (p, p) = (n, p)
    (&centered * &tmp).sum_axis(Axis(1))
}

/// Eigendecomposition of a symmetric matrix `a`, returning `(values, vectors)`
/// with the eigenvectors as the columns of `vectors` (as `faer` returns them).
/// Errors if the decomposition fails to converge.
pub(crate) fn symmetric_eigen(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), RobustError> {
    let p = a.nrows();
    let eig = to_faer(a)
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| RobustError::SingularDesign)?;
    let s = eig.S().column_vector();
    let u = eig.U();
    let values = Array1::from_shape_fn(p, |i| s[i]);
    let vectors = Array2::from_shape_fn((p, p), |(i, j)| u[(i, j)]);
    Ok((values, vectors))
}
