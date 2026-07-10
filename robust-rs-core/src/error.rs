//! Error type for the `robust-rs-core` crate.

use thiserror::Error;

/// Errors returned by robust estimation routines.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RobustError {
    /// An iterative solver failed to converge within the iteration cap.
    #[error("solver did not converge within {iters} iterations")]
    NonConvergence {
        /// Iterations performed before giving up.
        iters: usize,
    },
    /// The (weighted) design matrix is singular or rank-deficient.
    #[error("design matrix is singular or rank-deficient")]
    SingularDesign,
    /// The robust scale collapsed to zero (e.g. more than half the data tied).
    #[error("scale estimate is zero or non-finite (degenerate data)")]
    DegenerateScale,
    /// Not enough observations for the requested estimator.
    #[error("insufficient data: needed {needed}, got {got}")]
    InsufficientData {
        /// Minimum number of observations required.
        needed: usize,
        /// Number of observations supplied.
        got: usize,
    },
    /// A tuning constant was outside its valid range.
    #[error("invalid tuning constant: {value}")]
    InvalidTuning {
        /// The offending value.
        value: f64,
    },
    /// Subsampling (FAST-MCD / FAST-LTS) failed to find a valid subset.
    #[error("subsampling failed to produce a valid subset")]
    SubsampleFailure,
    /// Input arrays have inconsistent lengths.
    #[error("dimension mismatch: expected length {expected}, got {got}")]
    DimensionMismatch {
        /// The length that was required.
        expected: usize,
        /// The length that was supplied.
        got: usize,
    },
    /// A weight was negative or non-finite.
    #[error("invalid weight: {value} (weights must be finite and non-negative)")]
    InvalidWeight {
        /// The offending weight.
        value: f64,
    },
    /// A scale that requires a bounded loss (the S-scale) was given a loss with
    /// unbounded `ρ` (`rho_sup() == None`), which cannot define a high-breakdown
    /// scale.
    #[error("loss has unbounded ρ (rho_sup is None); it cannot define an S-scale")]
    UnboundedLoss,
}
