//! Sampling-theory computations: influence function, asymptotic variance,
//! Gaussian efficiency and breakdown point.

mod breakdown;
mod efficiency;
mod influence;
mod quadrature;
mod variance;

pub use self::breakdown::breakdown_point;
pub use self::efficiency::gaussian_efficiency;
pub use self::influence::influence_function;
pub use self::quadrature::gauss_hermite;
pub use self::variance::{asymptotic_variance, expect_psi_prime, expect_psi_squared};
