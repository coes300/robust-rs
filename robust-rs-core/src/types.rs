//! Strongly-typed wrappers that make illegal states unrepresentable and prevent
//! conflating a raw residual with a scaled one.

use crate::error::RobustError;

/// A strictly positive, finite scale estimate `s`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Scale(f64);

impl Scale {
    /// Construct a scale, rejecting non-positive or non-finite values.
    pub fn new(value: f64) -> Result<Self, RobustError> {
        if value.is_finite() && value > 0.0 {
            Ok(Self(value))
        } else {
            Err(RobustError::DegenerateScale)
        }
    }
    /// The underlying value.
    pub fn get(self) -> f64 {
        self.0
    }
}

/// A tuning constant (e.g. Huber `k`, Tukey `c`), required strictly positive.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TuningConstant(f64);

impl TuningConstant {
    /// Construct a tuning constant, rejecting non-positive or non-finite values.
    pub fn new(value: f64) -> Result<Self, RobustError> {
        if value.is_finite() && value > 0.0 {
            Ok(Self(value))
        } else {
            Err(RobustError::InvalidTuning { value })
        }
    }
    /// The underlying value.
    pub fn get(self) -> f64 {
        self.0
    }
}

/// A residual `y − ŷ` that has NOT yet been divided by the scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawResidual(pub f64);

/// A residual that has been standardized as `r / s`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaledResidual(pub f64);

impl RawResidual {
    /// Standardize by dividing by a positive scale.
    pub fn scale_by(self, s: Scale) -> ScaledResidual {
        ScaledResidual(self.0 / s.get())
    }
}
