//! Python bindings for [`robust-rs`](https://docs.rs/robust-rs): robust
//! statistics (M/S/MM regression, robust scale, MCD/OGK/Tyler covariance and
//! the influence-function theory).
//!
//! This crate is the compiled extension `pyrobust._pyrobust`; the pure-Python
//! `pyrobust` package re-exports its contents (and adds the CamelCase loss/scale
//! aliases). See `py/README.md` for the user-facing API.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod datasets;
mod dispatch;
mod error;
mod location;
mod loss;
mod multivariate;
mod regression;
mod scale;

/// The compiled extension module. Registered classes and functions are surfaced
/// to users through the `pyrobust` Python package.
#[pymodule]
fn _pyrobust(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("RobustError", py.get_type::<error::RobustError>())?;

    // Core value types.
    m.add_class::<loss::Loss>()?;
    m.add_class::<scale::PyScale>()?;

    // Result types.
    m.add_class::<location::LocationFit>()?;
    m.add_class::<location::HodgesLehmannFit>()?;
    m.add_class::<regression::RegressionFit>()?;
    m.add_class::<regression::LtsFit>()?;
    m.add_class::<regression::TheilSenFit>()?;
    m.add_class::<multivariate::ScatterFit>()?;
    m.add_class::<multivariate::McdFit>()?;
    m.add_class::<multivariate::TylerFit>()?;

    // Estimators (class-based, with `.fit`).
    m.add_class::<regression::MEstimator>()?;
    m.add_class::<regression::SEstimator>()?;
    m.add_class::<regression::MMEstimator>()?;
    m.add_class::<regression::Lts>()?;
    m.add_class::<multivariate::Mcd>()?;
    m.add_class::<multivariate::Ogk>()?;
    m.add_class::<multivariate::MScatter>()?;
    m.add_class::<multivariate::Tyler>()?;

    // Loss factories + loss-derived theory.
    m.add_function(wrap_pyfunction!(loss::least_squares, m)?)?;
    m.add_function(wrap_pyfunction!(loss::l1, m)?)?;
    m.add_function(wrap_pyfunction!(loss::huber, m)?)?;
    m.add_function(wrap_pyfunction!(loss::tukey, m)?)?;
    m.add_function(wrap_pyfunction!(loss::cauchy, m)?)?;
    m.add_function(wrap_pyfunction!(loss::welsch, m)?)?;
    m.add_function(wrap_pyfunction!(loss::andrews, m)?)?;
    m.add_function(wrap_pyfunction!(loss::hampel, m)?)?;
    m.add_function(wrap_pyfunction!(loss::gaussian_efficiency, m)?)?;
    m.add_function(wrap_pyfunction!(loss::asymptotic_variance, m)?)?;
    m.add_function(wrap_pyfunction!(loss::breakdown_point, m)?)?;

    // Scale factories.
    m.add_function(wrap_pyfunction!(scale::mad, m)?)?;
    m.add_function(wrap_pyfunction!(scale::qn, m)?)?;
    m.add_function(wrap_pyfunction!(scale::sn, m)?)?;
    m.add_function(wrap_pyfunction!(scale::huber_proposal2, m)?)?;

    // Location.
    m.add_function(wrap_pyfunction!(location::m_location, m)?)?;
    m.add_function(wrap_pyfunction!(location::trimmed_mean, m)?)?;
    m.add_function(wrap_pyfunction!(location::winsorized_mean, m)?)?;
    m.add_function(wrap_pyfunction!(location::hodges_lehmann, m)?)?;

    // Regression.
    m.add_function(wrap_pyfunction!(regression::theil_sen, m)?)?;

    // Multivariate Mahalanobis / outlier map.
    m.add_function(wrap_pyfunction!(multivariate::mahalanobis_distances, m)?)?;
    m.add_function(wrap_pyfunction!(multivariate::classical_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(multivariate::distance_cutoff, m)?)?;
    m.add_function(wrap_pyfunction!(multivariate::outlier_flags, m)?)?;

    // `datasets` submodule. Registered in sys.modules so `import
    // pyrobust.datasets` works even against the bare compiled extension.
    let ds = PyModule::new(py, "datasets")?;
    ds.add_function(wrap_pyfunction!(datasets::stackloss, &ds)?)?;
    ds.add_function(wrap_pyfunction!(datasets::stars_cyg, &ds)?)?;
    m.add_submodule(&ds)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pyrobust.datasets", &ds)?;

    Ok(())
}
