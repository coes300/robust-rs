//! Robust linear regression: M-, S-, MM- and LTS-estimators, plus Theil–Sen.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use robust_rs::estimator::RobustEstimator;
use robust_rs::regression as reg;
use robust_rs_core::rho::TukeyBiweight;
use robust_rs_core::solver::Control;

use crate::dispatch::{AnyLoss, AnyScale};
use crate::error::IntoPyResult;
use crate::loss::coerce_loss;
use crate::scale::coerce_scale;

/// The S-stage / MM S-stage default biweight tuning (50% breakdown).
fn s_default_loss() -> AnyLoss {
    AnyLoss::Tukey(TukeyBiweight::new(1.547).expect("1.547 is valid"))
}
/// Huber default for the plain M-estimator.
fn huber_default() -> AnyLoss {
    AnyLoss::Huber(robust_rs_core::rho::Huber::default())
}
/// MAD default scale.
fn mad_default() -> AnyScale {
    AnyScale::Mad(robust_rs_core::scale::Mad::default())
}

/// A fitted robust regression that can report its sampling theory.
///
/// Returned by :class:`MEstimator`, :class:`SEstimator` and :class:`MMEstimator`.
///
/// .. note::
///    This object wraps a non-``Send`` loss (``Box<dyn RhoFunction>``), so it is
///    **tied to the thread that created it** and cannot be shared across threads
///    or sent to another process (e.g. ``multiprocessing``); doing so raises.
///    Read its arrays out first if you need to move results across threads. The
///    other fit types (:class:`LtsFit`, scatter fits, …) have no such limit.
#[pyclass(name = "RegressionFit", module = "robustat_py", unsendable)]
pub struct RegressionFit {
    inner: robust_rs::estimator::RegressionFit,
}

#[pymethods]
impl RegressionFit {
    /// Estimated coefficients ``β̂`` (a ``p``-vector).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.coefficients.to_pyarray(py)
    }
    /// Estimated residual scale ``ŝ``.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale.get()
    }
    /// Residuals ``y − Xβ̂``.
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.residuals.to_pyarray(py)
    }
    /// Final IRLS weights (near 0 for down-weighted outliers).
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.weights.to_pyarray(py)
    }
    /// Asymptotic breakdown point (0 for a plain M-estimator; 0.5 for S/MM).
    #[getter]
    fn breakdown_point(&self) -> f64 {
        self.inner.breakdown_point
    }

    /// Efficiency relative to the Gaussian MLE (from the final-stage loss).
    fn gaussian_efficiency(&self) -> f64 {
        self.inner.gaussian_efficiency()
    }
    /// Asymptotic variance ``E[ψ²]/(E[ψ'])²``.
    fn asymptotic_variance(&self) -> f64 {
        self.inner.asymptotic_variance()
    }
    /// The influence function ``x ↦ ψ(x)/E[ψ']`` evaluated at ``x``.
    fn influence(&self, x: f64) -> f64 {
        self.inner.influence_function()(x)
    }
    /// Approximate coefficient covariance ``ŝ²·V·(XᵀX)⁻¹`` for the design ``X``.
    fn coef_covariance<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let x = x.as_array().to_owned();
        self.inner.coef_covariance(&x).to_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "RegressionFit(coefficients={:?}, scale={}, breakdown_point={})",
            self.inner.coefficients.as_slice().unwrap_or(&[]),
            self.inner.scale.get(),
            self.inner.breakdown_point
        )
    }
}

/// A fitted Least Trimmed Squares regression.
#[pyclass(name = "LtsFit", module = "robustat_py", frozen)]
pub struct LtsFit {
    inner: reg::LtsFit,
}

#[pymethods]
impl LtsFit {
    /// Estimated coefficients.
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.coefficients.to_pyarray(py)
    }
    /// Robust residual scale (MAD of the residuals).
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale.get()
    }
    /// Residuals ``y − Xβ̂`` for all ``n`` observations.
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.residuals.to_pyarray(py)
    }
    /// The retained ``h``-subset: indices of the smallest-residual observations.
    #[getter]
    fn subset(&self) -> Vec<usize> {
        self.inner.subset.clone()
    }
    /// The LTS objective: Σ of the ``h`` smallest squared residuals.
    #[getter]
    fn objective(&self) -> f64 {
        self.inner.objective
    }
    /// Coverage ``h``: the number of observations retained.
    #[getter]
    fn coverage(&self) -> usize {
        self.inner.coverage
    }
    /// Breakdown point ``(n − h + 1)/n``.
    #[getter]
    fn breakdown_point(&self) -> f64 {
        self.inner.breakdown_point
    }
    /// Not reported: LTS's Gaussian efficiency is **not** ρ-derivable (a hard
    /// 0/1 trim has no smooth ψ), unlike an M/S/MM :class:`RegressionFit`. Raises
    /// ``NotImplementedError``.
    fn gaussian_efficiency(&self) -> PyResult<f64> {
        Err(PyNotImplementedError::new_err(
            "LTS efficiency is not ρ-derivable (its hard trim has no smooth ψ); it \
             is deliberately not reported. Use MMEstimator for a fit that reports \
             efficiency, or read this fit's coefficients/subset directly.",
        ))
    }
    fn __repr__(&self) -> String {
        format!(
            "LtsFit(coefficients={:?}, coverage={}, breakdown_point={})",
            self.inner.coefficients.as_slice().unwrap_or(&[]),
            self.inner.coverage,
            self.inner.breakdown_point
        )
    }
}

/// A fitted Theil–Sen simple-regression line ``y ≈ intercept + slope·x``.
#[pyclass(name = "TheilSenFit", module = "robustat_py", frozen)]
pub struct TheilSenFit {
    inner: reg::TheilSenFit,
}

#[pymethods]
impl TheilSenFit {
    /// The median-of-pairwise-slopes estimate.
    #[getter]
    fn slope(&self) -> f64 {
        self.inner.slope
    }
    /// The intercept, ``median(yᵢ − slope·xᵢ)``.
    #[getter]
    fn intercept(&self) -> f64 {
        self.inner.intercept
    }
    /// The predicted response at ``x``.
    fn predict(&self, x: f64) -> f64 {
        self.inner.predict(x)
    }
    /// Asymptotic breakdown point ``1 − 1/√2 ≈ 0.293``.
    fn breakdown_point(&self) -> f64 {
        self.inner.breakdown_point()
    }
    /// Not reported: Theil–Sen's Gaussian efficiency is a known constant but is
    /// **not** ρ-derivable, so only :meth:`breakdown_point` is exposed. Raises
    /// ``NotImplementedError``.
    fn gaussian_efficiency(&self) -> PyResult<f64> {
        Err(PyNotImplementedError::new_err(
            "Theil–Sen's Gaussian efficiency is a known constant but is not \
             ρ-derivable, so it is not reported here; only breakdown_point() is.",
        ))
    }
    fn __repr__(&self) -> String {
        format!(
            "TheilSenFit(slope={}, intercept={})",
            self.inner.slope, self.inner.intercept
        )
    }
}

/// Regression M-estimator (IRLS). Convex/unique for a monotone loss, but has
/// **0 breakdown** against leverage; reach for :class:`MMEstimator` when
/// outliers may also sit in ``X``.
///
/// ``loss`` defaults to Huber (k=1.345), ``scale`` to the MAD.
#[pyclass(name = "MEstimator", module = "robustat_py", frozen)]
pub struct MEstimator {
    loss: AnyLoss,
    scale: AnyScale,
    control: Control,
}

#[pymethods]
impl MEstimator {
    #[new]
    #[pyo3(signature = (loss=None, scale=None, *, tol=1e-8, max_iter=100))]
    fn new(
        loss: Option<&Bound<'_, PyAny>>,
        scale: Option<&Bound<'_, PyAny>>,
        tol: f64,
        max_iter: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            loss: coerce_loss(loss, huber_default())?,
            scale: coerce_scale(scale, mad_default())?,
            control: Control { tol, max_iter },
        })
    }

    /// Fit to design matrix ``X`` (``n × p``) and response ``y`` (``n``).
    fn fit(
        &self,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<RegressionFit> {
        let x = x.as_array().to_owned();
        let y = y.as_array().to_owned();
        let est = reg::MEstimator {
            rho: self.loss,
            scale: self.scale,
            control: self.control,
        };
        Ok(RegressionFit {
            inner: est.fit(&x, &y).into_py_result()?,
        })
    }
}

/// S-estimator (FAST-S): 50% breakdown by minimising a robust M-scale of the
/// residuals. Lower efficiency than MM (which builds on it). Reproducible by
/// default; set ``seed`` for a different reproducible run.
#[pyclass(name = "SEstimator", module = "robustat_py", frozen)]
pub struct SEstimator {
    loss: AnyLoss,
    delta: Option<f64>,
    n_subsamples: usize,
    seed: Option<u64>,
    control: Control,
}

#[pymethods]
impl SEstimator {
    #[new]
    #[pyo3(signature = (loss=None, *, delta=None, n_subsamples=500, seed=None, tol=1e-8, max_iter=100))]
    fn new(
        loss: Option<&Bound<'_, PyAny>>,
        delta: Option<f64>,
        n_subsamples: usize,
        seed: Option<u64>,
        tol: f64,
        max_iter: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            loss: coerce_loss(loss, s_default_loss())?,
            delta,
            n_subsamples,
            seed,
            control: Control { tol, max_iter },
        })
    }

    /// Fit to design matrix ``X`` (``n × p``) and response ``y`` (``n``).
    fn fit(
        &self,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<RegressionFit> {
        let x = x.as_array().to_owned();
        let y = y.as_array().to_owned();
        let mut est = reg::SEstimator::new(self.loss)
            .n_subsamples(self.n_subsamples)
            .control(self.control);
        if let Some(d) = self.delta {
            est = est.delta(d);
        }
        if let Some(s) = self.seed {
            est = est.seed(s);
        }
        Ok(RegressionFit {
            inner: est.fit(&x, &y).into_py_result()?,
        })
    }
}

/// MM-estimator (R's ``lmrob`` default): an S-stage init + a redescending
/// fixed-scale M-step, giving 50% breakdown **and** ≈ 95% efficiency. Start here
/// for regression when contamination may sit in ``X`` as well as ``y``.
#[pyclass(name = "MMEstimator", module = "robustat_py", frozen)]
pub struct MMEstimator {
    s_loss: AnyLoss,
    m_loss: AnyLoss,
    n_subsamples: usize,
    seed: Option<u64>,
    control: Control,
}

#[pymethods]
impl MMEstimator {
    #[new]
    #[pyo3(signature = (s_loss=None, m_loss=None, *, n_subsamples=500, seed=None, tol=1e-8, max_iter=100))]
    fn new(
        s_loss: Option<&Bound<'_, PyAny>>,
        m_loss: Option<&Bound<'_, PyAny>>,
        n_subsamples: usize,
        seed: Option<u64>,
        tol: f64,
        max_iter: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            s_loss: coerce_loss(s_loss, s_default_loss())?,
            m_loss: coerce_loss(m_loss, AnyLoss::Tukey(TukeyBiweight::default()))?,
            n_subsamples,
            seed,
            control: Control { tol, max_iter },
        })
    }

    /// Fit to design matrix ``X`` (``n × p``) and response ``y`` (``n``).
    fn fit(
        &self,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<RegressionFit> {
        let x = x.as_array().to_owned();
        let y = y.as_array().to_owned();
        let mut est = reg::MMEstimator::new(self.s_loss, self.m_loss)
            .n_subsamples(self.n_subsamples)
            .control(self.control);
        if let Some(s) = self.seed {
            est = est.seed(s);
        }
        Ok(RegressionFit {
            inner: est.fit(&x, &y).into_py_result()?,
        })
    }
}

/// Least Trimmed Squares (FAST-LTS): a high-breakdown initializer that minimises
/// the sum of the ``h`` smallest squared residuals.
#[pyclass(name = "Lts", module = "robustat_py", frozen)]
pub struct Lts {
    coverage: Option<f64>,
    n_subsamples: usize,
    seed: Option<u64>,
    control: Control,
}

#[pymethods]
impl Lts {
    #[new]
    #[pyo3(signature = (*, coverage=None, n_subsamples=500, seed=None, tol=1e-8, max_iter=100))]
    fn new(
        coverage: Option<f64>,
        n_subsamples: usize,
        seed: Option<u64>,
        tol: f64,
        max_iter: usize,
    ) -> Self {
        Self {
            coverage,
            n_subsamples,
            seed,
            control: Control { tol, max_iter },
        }
    }

    /// Fit to design matrix ``X`` (``n × p``) and response ``y`` (``n``).
    fn fit(&self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<LtsFit> {
        let x = x.as_array().to_owned();
        let y = y.as_array().to_owned();
        let mut est = reg::Lts::new()
            .n_subsamples(self.n_subsamples)
            .control(self.control);
        if let Some(c) = self.coverage {
            est = est.coverage(c);
        }
        if let Some(s) = self.seed {
            est = est.seed(s);
        }
        Ok(LtsFit {
            inner: est.fit(&x, &y).into_py_result()?,
        })
    }
}

/// Fit a Theil–Sen line (median of pairwise slopes) to paired ``(x, y)`` data.
#[pyfunction]
pub fn theil_sen(x: Vec<f64>, y: Vec<f64>) -> PyResult<TheilSenFit> {
    Ok(TheilSenFit {
        inner: reg::theil_sen(&x, &y).into_py_result()?,
    })
}
