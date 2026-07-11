//! Robust univariate location: M-estimation, trimmed / Winsorized means and the
//! Hodges–Lehmann estimator.

use pyo3::prelude::*;
use robust_rs::location as loc;
use robust_rs_core::rho::Huber;
use robust_rs_core::scale::Mad;
use robust_rs_core::solver::Control;

use crate::dispatch::{AnyLoss, AnyScale};
use crate::error::IntoPyResult;
use crate::loss::coerce_loss;
use crate::scale::coerce_scale;

/// A fitted M-estimate of location.
#[pyclass(module = "robustat_py", frozen)]
pub struct LocationFit {
    inner: loc::LocationFit,
}

#[pymethods]
impl LocationFit {
    /// The location estimate ``θ̂``.
    #[getter]
    fn estimate(&self) -> f64 {
        self.inner.estimate
    }
    /// The robust scale used to standardise residuals.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale.get()
    }
    /// IRLS iterations performed.
    #[getter]
    fn iters(&self) -> usize {
        self.inner.iters
    }
    fn __repr__(&self) -> String {
        format!(
            "LocationFit(estimate={}, scale={}, iters={})",
            self.inner.estimate,
            self.inner.scale.get(),
            self.inner.iters
        )
    }
}

/// A Hodges–Lehmann location estimate (median of the Walsh averages).
#[pyclass(module = "robustat_py", frozen)]
pub struct HodgesLehmannFit {
    inner: loc::HodgesLehmannFit,
}

#[pymethods]
impl HodgesLehmannFit {
    /// The location estimate.
    #[getter]
    fn estimate(&self) -> f64 {
        self.inner.estimate
    }
    /// Gaussian efficiency ``3/π ≈ 0.955`` (the Wilcoxon ARE).
    fn gaussian_efficiency(&self) -> f64 {
        self.inner.gaussian_efficiency()
    }
    /// Asymptotic breakdown point ``1 − 1/√2 ≈ 0.293``.
    fn breakdown_point(&self) -> f64 {
        self.inner.breakdown_point()
    }
    fn __repr__(&self) -> String {
        format!("HodgesLehmannFit(estimate={})", self.inner.estimate)
    }
}

/// M-estimate of location by IRLS: iterate ``θ ← Σ wᵢxᵢ / Σ wᵢ`` with
/// ``wᵢ = loss.weight((xᵢ − θ)/s)`` until convergence.
///
/// ``loss`` defaults to Huber (k=1.345), ``scale`` to the MAD.
#[pyfunction]
#[pyo3(signature = (data, loss=None, scale=None, *, tol=1e-8, max_iter=100))]
pub fn m_location(
    data: Vec<f64>,
    loss: Option<&Bound<'_, PyAny>>,
    scale: Option<&Bound<'_, PyAny>>,
    tol: f64,
    max_iter: usize,
) -> PyResult<LocationFit> {
    let rho = coerce_loss(loss, AnyLoss::Huber(Huber::default()))?;
    let scl = coerce_scale(scale, AnyScale::Mad(Mad::default()))?;
    let ctrl = Control { tol, max_iter };
    let inner = loc::m_location(&data, &rho, &scl, &ctrl).into_py_result()?;
    Ok(LocationFit { inner })
}

/// The ``alpha``-trimmed mean (drop the lowest and highest ``alpha`` fraction);
/// ``alpha ∈ [0, 0.5)``.
#[pyfunction]
pub fn trimmed_mean(data: Vec<f64>, alpha: f64) -> PyResult<f64> {
    loc::trimmed_mean(&data, alpha).into_py_result()
}

/// The ``alpha``-Winsorized mean (clamp the tails instead of dropping them);
/// ``alpha ∈ [0, 0.5)``.
#[pyfunction]
pub fn winsorized_mean(data: Vec<f64>, alpha: f64) -> PyResult<f64> {
    loc::winsorized_mean(&data, alpha).into_py_result()
}

/// The Hodges–Lehmann location estimate: the median of the Walsh averages
/// ``(xᵢ + xⱼ)/2`` over ``i ≤ j``.
#[pyfunction]
pub fn hodges_lehmann(data: Vec<f64>) -> PyResult<HodgesLehmannFit> {
    let inner = loc::hodges_lehmann(&data).into_py_result()?;
    Ok(HodgesLehmannFit { inner })
}
