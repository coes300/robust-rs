//! Enum dispatch bridging the crate's generic, trait-based Rust API to the
//! dynamic Python surface.
//!
//! The Rust estimators are generic over `R: RhoFunction` / `S: ScaleEstimator`,
//! resolved at compile time. Python needs a single concrete type per role, so
//! [`AnyLoss`] and [`AnyScale`] are `enum`s over the built-in losses/scales that
//! re-implement the traits by delegation. Both are `Copy + 'static + Send + Sync`
//! (every variant is plain `f64` data), so they satisfy the estimators' bounds
//! (`RhoFunction + Clone + 'static`) and can live inside `Send`/`Sync` pyclasses.

use robust_rs_core::error::RobustError;
use robust_rs_core::rho::{
    Andrews, Cauchy, Hampel, Huber, LeastSquares, RhoFunction, TukeyBiweight, Welsch, L1,
};
use robust_rs_core::scale::{HuberProposal2, Mad, Qn, ScaleEstimator, Sn};
use robust_rs_core::types::Scale;

/// One of the crate's built-in [`RhoFunction`] losses.
#[derive(Debug, Clone, Copy)]
pub enum AnyLoss {
    /// [`LeastSquares`].
    LeastSquares(LeastSquares),
    /// [`L1`].
    L1(L1),
    /// [`Huber`].
    Huber(Huber),
    /// [`TukeyBiweight`].
    Tukey(TukeyBiweight),
    /// [`Cauchy`].
    Cauchy(Cauchy),
    /// [`Welsch`].
    Welsch(Welsch),
    /// [`Andrews`].
    Andrews(Andrews),
    /// [`Hampel`].
    Hampel(Hampel),
}

impl AnyLoss {
    fn as_dyn(&self) -> &dyn RhoFunction {
        match self {
            AnyLoss::LeastSquares(x) => x,
            AnyLoss::L1(x) => x,
            AnyLoss::Huber(x) => x,
            AnyLoss::Tukey(x) => x,
            AnyLoss::Cauchy(x) => x,
            AnyLoss::Welsch(x) => x,
            AnyLoss::Andrews(x) => x,
            AnyLoss::Hampel(x) => x,
        }
    }
}

impl RhoFunction for AnyLoss {
    fn rho(&self, r: f64) -> f64 {
        self.as_dyn().rho(r)
    }
    fn psi(&self, r: f64) -> f64 {
        self.as_dyn().psi(r)
    }
    fn weight(&self, r: f64) -> f64 {
        self.as_dyn().weight(r)
    }
    fn psi_prime(&self, r: f64) -> f64 {
        self.as_dyn().psi_prime(r)
    }
    fn tuning(&self) -> f64 {
        self.as_dyn().tuning()
    }
    fn is_redescending(&self) -> bool {
        self.as_dyn().is_redescending()
    }
    fn rho_sup(&self) -> Option<f64> {
        self.as_dyn().rho_sup()
    }
}

/// One of the crate's built-in [`ScaleEstimator`] robust scales.
#[derive(Debug, Clone, Copy)]
pub enum AnyScale {
    /// [`Mad`].
    Mad(Mad),
    /// [`Qn`].
    Qn(Qn),
    /// [`Sn`].
    Sn(Sn),
    /// [`HuberProposal2`].
    HuberProposal2(HuberProposal2),
}

impl AnyScale {
    fn as_dyn(&self) -> &dyn ScaleEstimator {
        match self {
            AnyScale::Mad(x) => x,
            AnyScale::Qn(x) => x,
            AnyScale::Sn(x) => x,
            AnyScale::HuberProposal2(x) => x,
        }
    }
}

impl ScaleEstimator for AnyScale {
    fn scale(&self, residuals: &[f64]) -> Result<Scale, RobustError> {
        self.as_dyn().scale(residuals)
    }
}
