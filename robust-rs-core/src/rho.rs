//! The `RhoFunction` trait: the organizing abstraction of the crate and its
//! standard implementations.
//!
//! Every robust M-estimator is defined by a loss `ρ`; its score `ψ = ρ'` is the
//! influence-function shape and the IRLS weight is `w(r) = ψ(r)/r`. Adding a
//! new estimator is, in the common case, implementing this one trait, not
//! writing a new algorithm.

mod andrews;
mod cauchy;
mod hampel;
mod huber;
mod l1;
mod least_squares;
mod tukey;
mod welsch;

pub use self::andrews::Andrews;
pub use self::cauchy::Cauchy;
pub use self::hampel::Hampel;
pub use self::huber::Huber;
pub use self::l1::L1;
pub use self::least_squares::LeastSquares;
pub use self::tukey::TukeyBiweight;
pub use self::welsch::Welsch;

/// A robust loss function `ρ` and the quantities derived from it.
pub trait RhoFunction {
    /// The loss `ρ(r)`.
    fn rho(&self, r: f64) -> f64;
    /// The score `ψ(r) = ρ'(r)`: the (unstandardized) influence-function shape.
    fn psi(&self, r: f64) -> f64;
    /// The IRLS weight `w(r) = ψ(r)/r`, with the removable singularity at
    /// `r = 0` resolved by the limit `ψ'(0)`.
    fn weight(&self, r: f64) -> f64;
    /// The derivative `ψ'(r) = ρ''(r)`: the curvature of the loss and the shape
    /// of the `E[ψ']` term in the asymptotic variance. (That `E[ψ']` is computed
    /// via Stein's identity `E[X·ψ(X)]`, not by integrating this, so kinked
    /// scores stay exact; `psi_prime` is exposed for callers and completeness.)
    fn psi_prime(&self, r: f64) -> f64;
    /// The tuning constant controlling the efficiency/robustness trade-off
    /// (`f64::NAN` for losses without one, e.g. L1/least squares).
    fn tuning(&self) -> f64;

    /// Whether `ψ` redescends to zero (⇒ `ρ` non-convex ⇒ needs a good start).
    fn is_redescending(&self) -> bool;
    /// `sup ρ = ρ(∞)`: `Some(_)` for bounded (redescending) losses, `None` for
    /// unbounded ones (Huber, least squares). Consumed by the S-scale.
    fn rho_sup(&self) -> Option<f64>;
}
