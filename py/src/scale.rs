//! The `ScaleEstimator` Python class (robust scale of a sample) and its
//! factory functions.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use robust_rs_core::scale::{HuberProposal2, Mad, Qn, ScaleEstimator as ScaleTrait, Sn};

use crate::dispatch::AnyScale;
use crate::error::IntoPyResult;

/// A robust scale estimator: given a sample, return a Gaussian-consistent
/// estimate of its spread `s` (used to standardise residuals as `r/s`).
///
/// Construct one with a factory: :func:`mad`, :func:`qn`, :func:`sn`,
/// :func:`huber_proposal2` (or the aliases ``Mad``, ``Qn``, ``Sn`` …).
#[pyclass(name = "ScaleEstimator", module = "robust_py", frozen)]
#[derive(Clone)]
pub struct PyScale {
    pub(crate) inner: AnyScale,
    repr: String,
}

impl PyScale {
    fn build(inner: AnyScale, repr: String) -> Self {
        Self { inner, repr }
    }
}

#[pymethods]
impl PyScale {
    /// The robust scale of ``data`` (raises ``RobustError`` on degenerate input).
    fn scale(&self, data: Vec<f64>) -> PyResult<f64> {
        Ok(self.inner.scale(&data).into_py_result()?.get())
    }

    fn __repr__(&self) -> String {
        self.repr.clone()
    }
}

/// `None` → `default`; a `ScaleEstimator` → its scale; a `str` → the same-named
/// default scale.
pub(crate) fn coerce_scale(
    obj: Option<&Bound<'_, PyAny>>,
    default: AnyScale,
) -> PyResult<AnyScale> {
    let Some(o) = obj else { return Ok(default) };
    if let Ok(s) = o.downcast::<PyScale>() {
        return Ok(s.borrow().inner);
    }
    if let Ok(name) = o.extract::<String>() {
        return scale_from_name(&name);
    }
    Err(PyTypeError::new_err(
        "expected a ScaleEstimator, a scale-name string, or None",
    ))
}

fn scale_from_name(s: &str) -> PyResult<AnyScale> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "mad" => AnyScale::Mad(Mad::default()),
        "qn" => AnyScale::Qn(Qn::default()),
        "sn" => AnyScale::Sn(Sn::default()),
        "huber" | "huber_proposal2" | "proposal2" => {
            AnyScale::HuberProposal2(HuberProposal2::default())
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown scale name {other:?}"
            )))
        }
    })
}

// --- Factory functions (aliased to CamelCase names in the Python package). ---

/// Median absolute deviation, ``s = consistency · median|rᵢ − median r|`` (the
/// default ``consistency = 1.4826`` makes it consistent for σ at the Gaussian).
#[pyfunction]
#[pyo3(signature = (consistency=1.482_602_218_505_602))]
pub fn mad(consistency: f64) -> PyScale {
    PyScale::build(
        AnyScale::Mad(Mad { consistency }),
        format!("Mad(consistency={consistency})"),
    )
}

/// Rousseeuw & Croux's ``Qn`` scale (≈ 82% efficiency, 50% breakdown).
#[pyfunction]
#[pyo3(signature = (consistency=2.2219, finite_sample_correction=true))]
pub fn qn(consistency: f64, finite_sample_correction: bool) -> PyScale {
    PyScale::build(
        AnyScale::Qn(Qn {
            consistency,
            finite_sample_correction,
        }),
        format!(
            "Qn(consistency={consistency}, finite_sample_correction={finite_sample_correction})"
        ),
    )
}

/// Rousseeuw & Croux's ``Sn`` scale (≈ 58% efficiency, 50% breakdown).
#[pyfunction]
#[pyo3(signature = (consistency=1.1926, finite_sample_correction=true))]
pub fn sn(consistency: f64, finite_sample_correction: bool) -> PyScale {
    PyScale::build(
        AnyScale::Sn(Sn {
            consistency,
            finite_sample_correction,
        }),
        format!(
            "Sn(consistency={consistency}, finite_sample_correction={finite_sample_correction})"
        ),
    )
}

/// Huber's "Proposal 2" simultaneous location–scale estimate with tuning ``k``
/// (default ``1.345``).
#[pyfunction]
#[pyo3(signature = (k=1.345))]
pub fn huber_proposal2(k: f64) -> PyScale {
    PyScale::build(
        AnyScale::HuberProposal2(HuberProposal2 { k }),
        format!("HuberProposal2(k={k})"),
    )
}
