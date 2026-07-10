//! Efficiency targets at the Gaussian model. These encode the design spec:
//! the tuned M-step losses (Huber@1.345, Tukey@4.685, Cauchy@2.3849,
//! Welsch@2.9846 andrews@1.339) are ≈ 95% efficient; least squares is 100%;
//! Hampel@(2,4,8) keeps near-full efficiency by leaving small residuals alone.

use approx::assert_relative_eq;
use robust_rs_core::rho::{Andrews, Cauchy, Hampel, Huber, LeastSquares, TukeyBiweight, Welsch};
use robust_rs_core::theory::gaussian_efficiency;

const QUAD: usize = 128;

#[test]
fn least_squares_is_fully_efficient() {
    assert_relative_eq!(
        gaussian_efficiency(&LeastSquares, QUAD),
        1.0,
        epsilon = 1e-6
    );
}

#[test]
fn huber_default_is_about_95_percent_efficient() {
    assert_relative_eq!(
        gaussian_efficiency(&Huber::default(), QUAD),
        0.95,
        epsilon = 1e-2
    );
}

#[test]
fn tukey_default_is_about_95_percent_efficient() {
    assert_relative_eq!(
        gaussian_efficiency(&TukeyBiweight::default(), QUAD),
        0.95,
        epsilon = 1e-2
    );
}

#[test]
fn cauchy_default_is_about_95_percent_efficient() {
    assert_relative_eq!(
        gaussian_efficiency(&Cauchy::default(), QUAD),
        0.95,
        epsilon = 1e-2
    );
}

#[test]
fn welsch_default_is_about_95_percent_efficient() {
    assert_relative_eq!(
        gaussian_efficiency(&Welsch::default(), QUAD),
        0.95,
        epsilon = 1e-2
    );
}

#[test]
fn andrews_default_is_about_95_percent_efficient() {
    assert_relative_eq!(
        gaussian_efficiency(&Andrews::default(), QUAD),
        0.95,
        epsilon = 1e-2
    );
}

#[test]
fn hampel_default_is_highly_efficient() {
    // (2,4,8) is ρ = r²/2 out to |r| = 2, so it is close to fully efficient.
    let eff = gaussian_efficiency(&Hampel::default(), QUAD);
    assert!(
        (0.98..1.0).contains(&eff),
        "Hampel(2,4,8) efficiency = {eff}"
    );
}
