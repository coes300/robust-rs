//! Robust estimators of univariate location.

mod hodges_lehmann;
mod m_location;
mod trimmed;

pub use self::hodges_lehmann::{hodges_lehmann, HodgesLehmannFit};
pub use self::m_location::{m_location, LocationFit};
pub use self::trimmed::{trimmed_mean, winsorized_mean};
