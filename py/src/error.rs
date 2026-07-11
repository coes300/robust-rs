//! Mapping the crate's [`RobustError`](robust_rs_core::error::RobustError) onto a
//! Python exception.

use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use robust_rs_core::error::RobustError as CoreError;

create_exception!(
    robust_py,
    RobustError,
    PyValueError,
    "Raised by robust estimation routines on invalid input or a failed fit."
);

/// Convert a core [`CoreError`] into the Python `RobustError` exception,
/// preserving its message.
pub fn to_py_err(e: CoreError) -> PyErr {
    RobustError::new_err(e.to_string())
}

/// `result.into_py_result()?` on any `Result<T, RobustError>`.
pub trait IntoPyResult<T> {
    /// Map the error arm onto the Python `RobustError` exception.
    fn into_py_result(self) -> PyResult<T>;
}

impl<T> IntoPyResult<T> for Result<T, CoreError> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(to_py_err)
    }
}
