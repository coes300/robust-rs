//! Multivariate robust statistics: robust location–scatter (MCD, OGK, M-scatter,
//! Tyler) and the Mahalanobis / outlier map built on them.

use ndarray::Array1;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use robust_rs::multivariate::{self as mv, mahalanobis, RobustScatter};
use robust_rs_core::solver::Control;

use crate::dispatch::{AnyLoss, AnyScale};
use crate::error::IntoPyResult;
use crate::loss::coerce_loss;
use crate::scale::coerce_scale;

/// A fitted robust location–scatter pair (shared by :class:`Ogk` and
/// :class:`MScatter`).
#[pyclass(name = "ScatterFit", module = "robust_py", frozen)]
pub struct ScatterFit {
    inner: mv::ScatterFit,
}

#[pymethods]
impl ScatterFit {
    /// Robust location estimate ``μ̂`` (a ``p``-vector).
    #[getter]
    fn location<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.location.to_pyarray(py)
    }
    /// Robust scatter (covariance) estimate ``Σ̂`` (``p × p``, SPD).
    #[getter]
    fn scatter<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.scatter.to_pyarray(py)
    }
    /// Robust Mahalanobis distances of the fitted rows.
    #[getter]
    fn distances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.distances.to_pyarray(py)
    }
    /// Final per-observation weights.
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.weights.to_pyarray(py)
    }
    /// The distance cutoff ``√χ²_{p, quantile}``.
    fn distance_cutoff(&self, quantile: f64) -> f64 {
        self.inner.distance_cutoff(quantile)
    }
    /// Flag each fitted row as an outlier (distance above the cutoff).
    fn outliers(&self, quantile: f64) -> Vec<bool> {
        self.inner.outliers(quantile)
    }
}

/// A fitted Minimum Covariance Determinant estimate.
#[pyclass(name = "McdFit", module = "robust_py", frozen)]
pub struct McdFit {
    inner: mv::McdFit,
}

#[pymethods]
impl McdFit {
    /// Reweighted (RMCD) location: the primary estimate.
    #[getter]
    fn location<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.location.to_pyarray(py)
    }
    /// Reweighted (RMCD) covariance: the primary estimate.
    #[getter]
    fn scatter<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.scatter.to_pyarray(py)
    }
    /// Robust Mahalanobis distances w.r.t. the primary ``(location, scatter)``.
    #[getter]
    fn distances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.distances.to_pyarray(py)
    }
    /// Reweighting weights (``1`` retained, ``0`` rejected by the χ² cutoff).
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.weights.to_pyarray(py)
    }
    /// Raw (consistency-corrected) MCD location: the best ``h``-subset mean.
    #[getter]
    fn raw_location<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.raw_location.to_pyarray(py)
    }
    /// Raw (consistency-corrected) MCD covariance.
    #[getter]
    fn raw_scatter<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.raw_scatter.to_pyarray(py)
    }
    /// The retained min-determinant ``h``-subset (ascending indices).
    #[getter]
    fn support(&self) -> Vec<usize> {
        self.inner.support.clone()
    }
    /// Coverage ``h``.
    #[getter]
    fn coverage(&self) -> usize {
        self.inner.coverage
    }
    /// The MCD objective: ``log det`` of the raw subset covariance.
    #[getter]
    fn objective(&self) -> f64 {
        self.inner.objective
    }
    /// Breakdown point ``(n − h + 1)/n``.
    #[getter]
    fn breakdown_point(&self) -> f64 {
        self.inner.breakdown_point
    }
    /// The distance cutoff ``√χ²_{p, quantile}``.
    fn distance_cutoff(&self, quantile: f64) -> f64 {
        self.inner.distance_cutoff(quantile)
    }
    /// Flag each fitted row as an outlier (distance above the cutoff).
    fn outliers(&self, quantile: f64) -> Vec<bool> {
        self.inner.outliers(quantile)
    }
}

/// A fitted Tyler shape estimate (shape only; distances rank but are not
/// χ²-calibrated).
#[pyclass(name = "TylerFit", module = "robust_py", frozen)]
pub struct TylerFit {
    inner: mv::TylerFit,
}

#[pymethods]
impl TylerFit {
    /// The robust centre ``μ̂``.
    #[getter]
    fn location<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.location().to_pyarray(py)
    }
    /// The unit-determinant shape matrix ``V̂`` (``det V̂ = 1``).
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.shape().to_pyarray(py)
    }
    /// Robust distances from the shape (good for ranking, not χ²-calibrated).
    #[getter]
    fn distances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.distances().to_pyarray(py)
    }
    /// Flag outliers assuming a χ²-radial (e.g. Gaussian) model, an opt-in that
    /// disregards the unidentified scale; for distribution-free use, threshold
    /// :attr:`distances` directly.
    fn outliers_assuming_chi2_radial(&self, quantile: f64) -> Vec<bool> {
        self.inner.outliers_assuming_chi2_radial(quantile)
    }
}

/// FAST-MCD: affine-equivariant, 50%-breakdown robust covariance, reweighted
/// (RMCD) by default. Reproducible by default; set ``seed`` for another run.
#[pyclass(name = "Mcd", module = "robust_py", frozen)]
pub struct Mcd {
    coverage: Option<f64>,
    n_subsamples: usize,
    reweight: bool,
    seed: Option<u64>,
    control: Control,
}

#[pymethods]
impl Mcd {
    #[new]
    #[pyo3(signature = (*, coverage=None, n_subsamples=500, reweight=true, seed=None, tol=1e-8, max_iter=100))]
    fn new(
        coverage: Option<f64>,
        n_subsamples: usize,
        reweight: bool,
        seed: Option<u64>,
        tol: f64,
        max_iter: usize,
    ) -> Self {
        Self {
            coverage,
            n_subsamples,
            reweight,
            seed,
            control: Control { tol, max_iter },
        }
    }

    /// Fit to the ``n × p`` data matrix ``X``.
    fn fit(&self, x: PyReadonlyArray2<'_, f64>) -> PyResult<McdFit> {
        let x = x.as_array().to_owned();
        let mut est = mv::Mcd::new()
            .n_subsamples(self.n_subsamples)
            .reweight(self.reweight)
            .control(self.control);
        if let Some(c) = self.coverage {
            est = est.coverage(c);
        }
        if let Some(s) = self.seed {
            est = est.seed(s);
        }
        Ok(McdFit {
            inner: est.fit(&x).into_py_result()?,
        })
    }
}

/// OGK: a fast, deterministic, positive-definite robust covariance. ``scale``
/// defaults to ``Qn``.
#[pyclass(name = "Ogk", module = "robust_py", frozen)]
pub struct Ogk {
    scale: AnyScale,
    n_iter: usize,
    reweight: bool,
}

#[pymethods]
impl Ogk {
    #[new]
    #[pyo3(signature = (scale=None, *, n_iter=2, reweight=true))]
    fn new(scale: Option<&Bound<'_, PyAny>>, n_iter: usize, reweight: bool) -> PyResult<Self> {
        Ok(Self {
            scale: coerce_scale(scale, AnyScale::Qn(robust_rs_core::scale::Qn::default()))?,
            n_iter,
            reweight,
        })
    }

    /// Fit to the ``n × p`` data matrix ``X``.
    fn fit(&self, x: PyReadonlyArray2<'_, f64>) -> PyResult<ScatterFit> {
        let x = x.as_array().to_owned();
        let est = mv::Ogk::new(self.scale)
            .n_iter(self.n_iter)
            .reweight(self.reweight);
        Ok(ScatterFit {
            inner: est.fit(&x).into_py_result()?,
        })
    }
}

/// A monotone M-estimator of location and scatter (low breakdown ≈ ``1/(p+1)``;
/// the multivariate analogue of regression M-estimation). ``loss`` defaults to
/// Huber.
#[pyclass(name = "MScatter", module = "robust_py", frozen)]
pub struct MScatter {
    loss: AnyLoss,
    control: Control,
}

#[pymethods]
impl MScatter {
    #[new]
    #[pyo3(signature = (loss=None, *, tol=1e-8, max_iter=100))]
    fn new(loss: Option<&Bound<'_, PyAny>>, tol: f64, max_iter: usize) -> PyResult<Self> {
        Ok(Self {
            loss: coerce_loss(loss, AnyLoss::Huber(robust_rs_core::rho::Huber::default()))?,
            control: Control { tol, max_iter },
        })
    }

    /// Fit to the ``n × p`` data matrix ``X``.
    fn fit(&self, x: PyReadonlyArray2<'_, f64>) -> PyResult<ScatterFit> {
        let x = x.as_array().to_owned();
        let est = mv::MScatter::new(self.loss).control(self.control);
        Ok(ScatterFit {
            inner: est.fit(&x).into_py_result()?,
        })
    }
}

/// Tyler's distribution-free M-estimator of *shape* (unit determinant). A robust
/// centre may be supplied via ``location``; the default is the coordinatewise
/// median.
#[pyclass(name = "Tyler", module = "robust_py", frozen)]
pub struct Tyler {
    location: Option<Vec<f64>>,
    control: Control,
}

#[pymethods]
impl Tyler {
    #[new]
    #[pyo3(signature = (*, location=None, tol=1e-8, max_iter=100))]
    fn new(location: Option<Vec<f64>>, tol: f64, max_iter: usize) -> Self {
        Self {
            location,
            control: Control { tol, max_iter },
        }
    }

    /// Fit to the ``n × p`` data matrix ``X``.
    fn fit(&self, x: PyReadonlyArray2<'_, f64>) -> PyResult<TylerFit> {
        let x = x.as_array().to_owned();
        let mut est = mv::Tyler::new().control(self.control);
        if let Some(loc) = &self.location {
            est = est.location(Array1::from_vec(loc.clone()));
        }
        Ok(TylerFit {
            inner: est.fit(&x).into_py_result()?,
        })
    }
}

// --- Mahalanobis / outlier map over any (μ̂, Σ̂) pair. ---

/// Robust Mahalanobis distances of every row of ``X`` given a location ``loc``
/// (``p``-vector) and an SPD scatter ``scatter`` (``p × p``).
#[pyfunction]
pub fn mahalanobis_distances<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    loc: Vec<f64>,
    scatter: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_array().to_owned();
    let loc = Array1::from_vec(loc);
    let scatter = scatter.as_array().to_owned();
    let d = mahalanobis::mahalanobis_distances(&x, &loc, &scatter).into_py_result()?;
    Ok(d.to_pyarray(py))
}

/// The classical (non-robust) mean and unbiased sample covariance of ``X``,
/// returned as ``(mean, covariance)``.
#[pyfunction]
pub fn classical_covariance<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>) {
    let x = x.as_array().to_owned();
    let (mean, cov) = mahalanobis::classical_covariance(&x);
    (mean.to_pyarray(py), cov.to_pyarray(py))
}

/// The distance cutoff ``√χ²_{p, quantile}`` for flagging outliers on ``p``
/// variables.
#[pyfunction]
pub fn distance_cutoff(p: usize, quantile: f64) -> f64 {
    mahalanobis::distance_cutoff(p, quantile)
}

/// Flag ``distances`` exceeding ``distance_cutoff(p, quantile)``.
#[pyfunction]
pub fn outlier_flags(distances: Vec<f64>, p: usize, quantile: f64) -> Vec<bool> {
    let d = Array1::from_vec(distances);
    mahalanobis::outlier_flags(&d, p, quantile)
}
