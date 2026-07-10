//! Common imports: `use robust_rs::prelude::*;`.

pub use crate::estimator::{RegressionFit, RobustEstimator};
pub use crate::location::{hodges_lehmann, m_location, HodgesLehmannFit};
pub use crate::multivariate::{
    mahalanobis, MScatter, Mcd, McdFit, Ogk, RobustScatter, ScatterFit, Tyler, TylerFit,
};
pub use crate::regression::{
    theil_sen, Lts, LtsFit, MEstimator, MMEstimator, SEstimator, TheilSenFit,
};
pub use robust_rs_core::rho::{Huber, LeastSquares, RhoFunction, TukeyBiweight, L1};
pub use robust_rs_core::scale::{Mad, ScaleEstimator};
pub use robust_rs_core::solver::Control;
pub use robust_rs_core::types::Scale;
