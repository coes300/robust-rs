//! Robust linear regression.

mod lts;
mod m_estimator;
mod mm_estimator;
mod s_estimator;
mod subsample; // shared FAST-{S,LTS} elemental-resampling scaffold
mod theil_sen;

pub use self::lts::{Lts, LtsFit};
pub use self::m_estimator::MEstimator;
pub use self::mm_estimator::MMEstimator;
pub use self::s_estimator::SEstimator;
pub use self::theil_sen::{theil_sen, TheilSenFit};
