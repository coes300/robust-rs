//! Equivariance properties of location M-estimators.

use approx::assert_relative_eq;
use proptest::prelude::*;
use robust_rs::prelude::*;

proptest! {
    #[test]
    fn location_is_translation_equivariant(
        data in prop::collection::vec(-100.0f64..100.0, 5..50),
        shift in -50.0f64..50.0,
    ) {
        let base = m_location(&data, &Huber::default(), &Mad::default(), &Control::default());
        let shifted_data: Vec<f64> = data.iter().map(|x| x + shift).collect();
        let shifted = m_location(&shifted_data, &Huber::default(), &Mad::default(), &Control::default());
        if let (Ok(b), Ok(s)) = (base, shifted) {
            assert_relative_eq!(s.estimate, b.estimate + shift, epsilon = 1e-6);
        }
    }

    #[test]
    fn location_is_scale_equivariant(
        data in prop::collection::vec(-100.0f64..100.0, 5..50),
        c in 0.1f64..10.0,
    ) {
        let base = m_location(&data, &Huber::default(), &Mad::default(), &Control::default());
        let scaled_data: Vec<f64> = data.iter().map(|x| x * c).collect();
        let scaled = m_location(&scaled_data, &Huber::default(), &Mad::default(), &Control::default());
        if let (Ok(b), Ok(s)) = (base, scaled) {
            assert_relative_eq!(s.estimate, b.estimate * c, epsilon = 1e-6);
        }
    }
}
