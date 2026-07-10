//! Shared spine of the [`robust-rs`] project: the M-estimator loss trait, robust
//! scale estimation and the influence-function / efficiency / breakdown-point
//! theory. Dependency-light (no linear algebra; only `libm`, `num-traits`,
//! `thiserror`), so it builds for `wasm32` and can be depended on for just the
//! losses and theory. The estimators live in the [`robust-rs`] crate.
//!
//! # Modules
//!
//! - [`rho`]: the organizing [`RhoFunction`](rho::RhoFunction) trait (`ρ`, `ψ`,
//!   `weight`, `ψ'`, `tuning`, `is_redescending`, `rho_sup`) and its
//!   implementations: [`LeastSquares`](rho::LeastSquares), [`L1`](rho::L1),
//!   [`Huber`](rho::Huber), [`TukeyBiweight`](rho::TukeyBiweight),
//!   [`Cauchy`](rho::Cauchy), [`Welsch`](rho::Welsch), [`Andrews`](rho::Andrews),
//!   [`Hampel`](rho::Hampel).
//! - [`scale`]: the [`ScaleEstimator`](scale::ScaleEstimator) trait and the robust
//!   scales [`Mad`](scale::Mad), [`HuberProposal2`](scale::HuberProposal2),
//!   [`SScale`](scale::SScale), [`Qn`](scale::Qn), [`Sn`](scale::Sn).
//! - [`theory`][]: [`influence_function`](theory::influence_function),
//!   [`asymptotic_variance`](theory::asymptotic_variance),
//!   [`gaussian_efficiency`](theory::gaussian_efficiency),
//!   [`breakdown_point`](theory::breakdown_point) and the
//!   [`gauss_hermite`](theory::gauss_hermite) quadrature underneath them.
//! - [`solver`]: the [`Control`](solver::Control) stopping rule shared by the
//!   iterative estimators.
//! - [`types`]: the [`Scale`](types::Scale), [`TuningConstant`](types::TuningConstant),
//!   [`RawResidual`](types::RawResidual) / [`ScaledResidual`](types::ScaledResidual)
//!   newtypes.
//! - [`error`]: the crate-wide [`RobustError`](error::RobustError).
//!
//! ```
//! use robust_rs_core::rho::{Huber, RhoFunction};
//! use robust_rs_core::theory::gaussian_efficiency;
//!
//! let huber = Huber::default();          // k = 1.345
//! assert_eq!(huber.psi(10.0), 1.345);    // clipped score: bounded influence
//! assert!((gaussian_efficiency(&huber, 128) - 0.95).abs() < 0.01);
//! ```
//!
//! [`robust-rs`]: https://docs.rs/robust-rs
#![deny(missing_docs)]

pub mod error;
pub mod rho;
pub mod scale;
pub mod solver;
pub mod theory;
pub mod types;
