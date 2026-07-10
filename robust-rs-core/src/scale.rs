//! The `ScaleEstimator` trait and standard robust scale estimates.

mod huber_proposal2;
mod mad;
mod qn;
mod s_scale;
mod sn;

pub use self::huber_proposal2::HuberProposal2;
pub use self::mad::Mad;
pub use self::qn::Qn;
pub use self::s_scale::SScale;
pub use self::sn::Sn;

use crate::error::RobustError;
use crate::types::Scale;

/// Estimates the scale `s` used to standardize residuals as `r / s`. Without a
/// scale estimate the tuning constants are meaningless.
pub trait ScaleEstimator {
    /// A robust scale of the given residuals.
    fn scale(&self, residuals: &[f64]) -> Result<Scale, RobustError>;
}
