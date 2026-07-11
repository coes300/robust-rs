//! The classic robust-statistics reference datasets, as numpy arrays.

use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use robust_rs::datasets as ds;

/// Brownlee's stack-loss data: 21 observations. Returns ``(X, y)`` with ``X``
/// shaped ``21 × 3`` (air flow, water temperature, acid concentration) and ``y``
/// the stack loss.
#[pyfunction]
pub fn stackloss(py: Python<'_>) -> (Bound<'_, PyArray2<f64>>, Bound<'_, PyArray1<f64>>) {
    let (x, y) = ds::stackloss();
    (x.to_pyarray(py), y.to_pyarray(py))
}

/// Hertzsprung–Russell diagram of star cluster CYG OB1: 47 stars. Returns
/// ``(X, y)`` with ``X`` shaped ``47 × 1`` (``log.Te``) and ``y`` (``log.light``).
/// Cases 11, 20, 30, 34 are giant-star outliers.
#[pyfunction]
pub fn stars_cyg(py: Python<'_>) -> (Bound<'_, PyArray2<f64>>, Bound<'_, PyArray1<f64>>) {
    let (x, y) = ds::stars_cyg();
    (x.to_pyarray(py), y.to_pyarray(py))
}
