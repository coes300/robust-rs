//! Iterative solver utilities shared across estimators.
//!
//! The *regression* IRLS driver lives in the `robust-rs` crate (it needs linear
//! algebra). This module holds the scale-free machinery: the [`Control`]
//! convergence controller shared by the location and scale M-estimators (each of
//! which runs its own bespoke fixed-point iteration).

mod control;

pub use self::control::Control;
