//! The `Loss` (ρ-function) Python class, its factory functions and the
//! loss-derived sampling theory.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use robust_rs_core::rho::{
    Andrews, Cauchy, Hampel, Huber, LeastSquares, RhoFunction, TukeyBiweight, Welsch, L1,
};
use robust_rs_core::theory;

use crate::dispatch::AnyLoss;
use crate::error::IntoPyResult;

/// Gauss–Hermite nodes used for the theory integrals (matches the Rust crate).
const QUAD: usize = 128;

/// A robust loss function `ρ`, together with the quantities derived from it: its
/// score `ψ = ρ'`, IRLS weight `w(r) = ψ(r)/r`, and the influence-function /
/// efficiency / variance theory.
///
/// Construct one with a factory: :func:`huber`, :func:`tukey`, :func:`l1`, … (or
/// their capitalised aliases ``Huber``, ``Tukey`` …).
#[pyclass(name = "Loss", module = "robustat_py", frozen)]
#[derive(Clone)]
pub struct Loss {
    pub(crate) inner: AnyLoss,
    name: &'static str,
    repr: String,
}

impl Loss {
    fn build(inner: AnyLoss, name: &'static str, repr: String) -> Self {
        Self { inner, name, repr }
    }
}

#[pymethods]
impl Loss {
    /// ``ρ(r)``: the loss at residual ``r``.
    fn rho(&self, r: f64) -> f64 {
        self.inner.rho(r)
    }
    /// ``ψ(r) = ρ'(r)``: the score, i.e. the (unstandardised) influence shape.
    fn psi(&self, r: f64) -> f64 {
        self.inner.psi(r)
    }
    /// ``w(r) = ψ(r)/r``: the IRLS weight (``w(0) = 1`` by convention).
    fn weight(&self, r: f64) -> f64 {
        self.inner.weight(r)
    }
    /// ``ψ'(r) = ρ''(r)``: the curvature of the loss.
    fn psi_prime(&self, r: f64) -> f64 {
        self.inner.psi_prime(r)
    }

    /// The tuning constant (``nan`` for losses without one, e.g. L1/least squares).
    #[getter]
    fn tuning(&self) -> f64 {
        self.inner.tuning()
    }
    /// Whether ``ψ`` redescends to zero (⇒ ``ρ`` non-convex ⇒ needs a good start).
    #[getter]
    fn is_redescending(&self) -> bool {
        self.inner.is_redescending()
    }
    /// ``sup ρ = ρ(∞)``: a float for bounded (redescending) losses, else ``None``.
    #[getter]
    fn rho_sup(&self) -> Option<f64> {
        self.inner.rho_sup()
    }
    /// The loss family name, e.g. ``"Huber"``.
    #[getter]
    fn name(&self) -> &str {
        self.name
    }

    /// Asymptotic efficiency at the Gaussian, ``(E[ψ'])² / E[ψ²]``.
    fn gaussian_efficiency(&self) -> f64 {
        theory::gaussian_efficiency(&self.inner, QUAD)
    }
    /// Asymptotic variance ``E[ψ²] / (E[ψ'])²``.
    fn asymptotic_variance(&self) -> f64 {
        theory::asymptotic_variance(&self.inner, QUAD)
    }
    /// The influence function ``x ↦ ψ(x)/E[ψ']`` evaluated at ``x`` (bounded for
    /// every loss here, so an extreme residual cannot blow the estimate up).
    fn influence(&self, x: f64) -> f64 {
        theory::influence_function(&self.inner, QUAD)(x)
    }

    fn __repr__(&self) -> String {
        self.repr.clone()
    }
}

/// `None` → `default`; a `Loss` → its loss; a `str` → the same-named default-tuned
/// loss. Used by estimators so a loss can be passed as an object or a name.
pub(crate) fn coerce_loss(obj: Option<&Bound<'_, PyAny>>, default: AnyLoss) -> PyResult<AnyLoss> {
    let Some(o) = obj else { return Ok(default) };
    if let Ok(loss) = o.downcast::<Loss>() {
        return Ok(loss.borrow().inner);
    }
    if let Ok(s) = o.extract::<String>() {
        return loss_from_name(&s);
    }
    Err(PyTypeError::new_err(
        "expected a Loss, a loss-name string, or None",
    ))
}

/// A default-tuned loss looked up by (case-insensitive) name.
fn loss_from_name(s: &str) -> PyResult<AnyLoss> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "huber" => AnyLoss::Huber(Huber::default()),
        "tukey" | "biweight" | "bisquare" | "tukeybiweight" => {
            AnyLoss::Tukey(TukeyBiweight::default())
        }
        "leastsquares" | "least_squares" | "l2" | "ols" => AnyLoss::LeastSquares(LeastSquares),
        "l1" | "lad" | "median" => AnyLoss::L1(L1),
        "cauchy" => AnyLoss::Cauchy(Cauchy::default()),
        "welsch" => AnyLoss::Welsch(Welsch::default()),
        "andrews" => AnyLoss::Andrews(Andrews::default()),
        "hampel" => AnyLoss::Hampel(Hampel::default()),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown loss name {other:?}"
            )))
        }
    })
}

// --- Factory functions (aliased to CamelCase names in the Python package). ---

/// The least-squares (L2) loss ``ρ(r) = r²/2``: the efficient, zero-breakdown
/// baseline (gives the mean / OLS).
#[pyfunction]
pub fn least_squares() -> Loss {
    Loss::build(
        AnyLoss::LeastSquares(LeastSquares),
        "LeastSquares",
        "LeastSquares()".to_string(),
    )
}

/// The L1 (absolute-value) loss ``ρ(r) = |r|``: gives the median / L1 regression.
#[pyfunction]
pub fn l1() -> Loss {
    Loss::build(AnyLoss::L1(L1), "L1", "L1()".to_string())
}

/// Huber's loss (monotone, convex): quadratic near 0, linear in the tails.
/// ``k`` defaults to ``1.345`` (≈ 95% Gaussian efficiency).
#[pyfunction]
#[pyo3(signature = (k=1.345))]
pub fn huber(k: f64) -> PyResult<Loss> {
    let h = Huber::new(k).into_py_result()?;
    Ok(Loss::build(
        AnyLoss::Huber(h),
        "Huber",
        format!("Huber(k={k})"),
    ))
}

/// Tukey's biweight (redescending). ``c`` defaults to ``4.685`` (≈ 95%
/// efficiency as an M-step; use ``c ≈ 1.547`` for a 50%-breakdown S-scale).
#[pyfunction]
#[pyo3(signature = (c=4.685))]
pub fn tukey(c: f64) -> PyResult<Loss> {
    let t = TukeyBiweight::new(c).into_py_result()?;
    Ok(Loss::build(
        AnyLoss::Tukey(t),
        "TukeyBiweight",
        format!("TukeyBiweight(c={c})"),
    ))
}

/// The Cauchy (Lorentzian) soft redescender. ``c`` defaults to ``2.3849``.
#[pyfunction]
#[pyo3(signature = (c=2.3849))]
pub fn cauchy(c: f64) -> PyResult<Loss> {
    let l = Cauchy::new(c).into_py_result()?;
    Ok(Loss::build(
        AnyLoss::Cauchy(l),
        "Cauchy",
        format!("Cauchy(c={c})"),
    ))
}

/// Welsch's (Leclerc's) smooth bounded redescender. ``c`` defaults to ``2.9846``.
#[pyfunction]
#[pyo3(signature = (c=2.9846))]
pub fn welsch(c: f64) -> PyResult<Loss> {
    let w = Welsch::new(c).into_py_result()?;
    Ok(Loss::build(
        AnyLoss::Welsch(w),
        "Welsch",
        format!("Welsch(c={c})"),
    ))
}

/// Andrews' sine-wave redescender (hard cutoff at ``cπ``). ``c`` defaults to ``1.339``.
#[pyfunction]
#[pyo3(signature = (c=1.339))]
pub fn andrews(c: f64) -> PyResult<Loss> {
    let a = Andrews::new(c).into_py_result()?;
    Ok(Loss::build(
        AnyLoss::Andrews(a),
        "Andrews",
        format!("Andrews(c={c})"),
    ))
}

/// Hampel's three-part linear redescender with break-points ``0 < a ≤ b < c``
/// (defaults ``(2, 4, 8)``).
#[pyfunction]
#[pyo3(signature = (a=2.0, b=4.0, c=8.0))]
pub fn hampel(a: f64, b: f64, c: f64) -> PyResult<Loss> {
    let h = Hampel::new(a, b, c).into_py_result()?;
    Ok(Loss::build(
        AnyLoss::Hampel(h),
        "Hampel",
        format!("Hampel(a={a}, b={b}, c={c})"),
    ))
}

// --- Loss-derived theory as free functions (also methods on `Loss`). ---

/// Asymptotic Gaussian efficiency of an M-estimator using this `loss`.
#[pyfunction]
pub fn gaussian_efficiency(loss: &Loss) -> f64 {
    loss.gaussian_efficiency()
}

/// Asymptotic variance ``E[ψ²]/(E[ψ'])²`` of this `loss`.
#[pyfunction]
pub fn asymptotic_variance(loss: &Loss) -> f64 {
    loss.asymptotic_variance()
}

/// Breakdown point of a plain regression M-estimator using this `loss`: always
/// ``0.0`` (a single high-leverage point can carry the fit). The high-breakdown
/// estimators report their own design breakdown instead.
#[pyfunction]
pub fn breakdown_point(loss: &Loss) -> f64 {
    theory::breakdown_point(&loss.inner)
}
