//! Analytic invariants every RhoFunction must satisfy.

use approx::assert_relative_eq;
use robust_rs_core::rho::{Andrews, Cauchy, Hampel, Huber, RhoFunction, TukeyBiweight, Welsch};

/// ψ should equal the numerical derivative of ρ.
fn check_psi_is_rho_derivative(rho: &dyn RhoFunction, points: &[f64]) {
    let h = 1e-6;
    for &r in points {
        let numeric = (rho.rho(r + h) - rho.rho(r - h)) / (2.0 * h);
        assert_relative_eq!(rho.psi(r), numeric, epsilon = 1e-4);
    }
}

/// weight(r) * r should equal psi(r) away from the origin.
///
/// This invariant is deliberately blind at `r = 0`: `weight(0)·0 == 0 == psi(0)`
/// holds for *any* finite `weight(0)`, so it cannot discriminate the value chosen
/// there (the removable-singularity limit for the smooth losses, or the finite
/// convention for L1; see `docs/conventions.md`). The origin is excluded here.
fn check_weight_matches_psi(rho: &dyn RhoFunction, points: &[f64]) {
    for &r in points {
        if r.abs() > 1e-6 {
            assert_relative_eq!(rho.weight(r) * r, rho.psi(r), epsilon = 1e-9);
        }
    }
}

#[test]
fn huber_psi_is_derivative_of_rho() {
    let points = [-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0];
    check_psi_is_rho_derivative(&Huber::default(), &points);
}

#[test]
fn tukey_psi_is_derivative_of_rho() {
    let points = [-6.0, -3.0, -1.0, 1.0, 3.0, 6.0];
    check_psi_is_rho_derivative(&TukeyBiweight::default(), &points);
}

// The new v0.2 losses satisfy the same analytic invariants. Test points avoid
// the piecewise break-points (Andrews at ±cπ ≈ ±4.21, Hampel at ±2/±4/±8)
// where a central difference would straddle a kink and misestimate ψ.
#[test]
fn cauchy_psi_is_derivative_of_rho() {
    let points = [-6.0, -3.0, -1.0, -0.5, 0.5, 1.0, 3.0, 6.0];
    check_psi_is_rho_derivative(&Cauchy::default(), &points);
}

#[test]
fn welsch_psi_is_derivative_of_rho() {
    let points = [-6.0, -3.0, -1.0, -0.5, 0.5, 1.0, 3.0, 6.0];
    check_psi_is_rho_derivative(&Welsch::default(), &points);
}

#[test]
fn andrews_psi_is_derivative_of_rho() {
    let points = [-6.0, -3.0, -1.0, -0.5, 0.5, 1.0, 3.0, 6.0];
    check_psi_is_rho_derivative(&Andrews::default(), &points);
}

#[test]
fn hampel_psi_is_derivative_of_rho() {
    let points = [-9.0, -6.0, -3.0, -1.0, 1.0, 3.0, 6.0, 9.0];
    check_psi_is_rho_derivative(&Hampel::default(), &points);
}

#[test]
fn huber_weight_matches_psi() {
    let points = [-5.0, -2.0, -1.0, 1.0, 2.0, 5.0];
    check_weight_matches_psi(&Huber::default(), &points);
}

#[test]
fn new_losses_weight_matches_psi() {
    let points = [-9.0, -6.0, -3.0, -1.0, 1.0, 3.0, 6.0, 9.0];
    check_weight_matches_psi(&Cauchy::default(), &points);
    check_weight_matches_psi(&Welsch::default(), &points);
    check_weight_matches_psi(&Andrews::default(), &points);
    check_weight_matches_psi(&Hampel::default(), &points);
}

#[test]
fn tukey_is_redescending_huber_is_not() {
    assert!(TukeyBiweight::default().is_redescending());
    assert!(!Huber::default().is_redescending());
}

#[test]
fn new_redescenders_are_flagged() {
    assert!(Cauchy::default().is_redescending());
    assert!(Welsch::default().is_redescending());
    assert!(Andrews::default().is_redescending());
    assert!(Hampel::default().is_redescending());
}

#[test]
fn tukey_rho_is_bounded_huber_is_not() {
    assert!(TukeyBiweight::default().rho_sup().is_some());
    assert!(Huber::default().rho_sup().is_none());
}

#[test]
fn bounded_losses_report_rho_sup_but_soft_cauchy_does_not() {
    // Hard redescenders bound ρ and expose sup ρ (consumed by the S-scale); the
    // soft Cauchy redescends yet keeps ρ unbounded, so it reports `None`.
    assert!(Welsch::default().rho_sup().is_some());
    assert!(Andrews::default().rho_sup().is_some());
    assert!(Hampel::default().rho_sup().is_some());
    assert!(Cauchy::default().rho_sup().is_none());
}
