//! Chi-square distribution helpers (CDF and quantile) via the regularized
//! incomplete gamma function.
//!
//! Robust multivariate outlier detection thresholds a squared Mahalanobis
//! distance at a chi-square quantile `χ²_{p,q}` (a point is flagged when
//! `d² > χ²_{p,0.975}`) and the FAST-MCD consistency correction needs both the
//! quantile and the CDF at a *shifted* degrees of freedom `p + 2`
//! (Croux & Haesbroeck 1999).
//!
//! These are implemented directly on `libm::lgamma`, consistent with the
//! workspace's dependency-light special-function policy (no `statrs`): each
//! special function is either `libm`-based or self-implemented and pinned against
//! known values.
//!
//! `χ²_k` is `Gamma(shape = k/2, scale = 2)`, so its CDF is the regularized
//! lower incomplete gamma `P(k/2, x/2)`. `P` is evaluated by the standard
//! series / continued-fraction split (Press et al., *Numerical Recipes* §6.2):
//! the series for `x < a + 1`, Lentz's continued fraction for `Q = 1 − P`
//! otherwise. The quantile inverts the (strictly increasing) CDF by safeguarded
//! Newton iteration with a bisection fallback, so it can never leave the bracket.

/// The `χ²_k` CDF at `x`: `P(k/2, x/2)`.
pub(crate) fn chi2_cdf(x: f64, k: f64) -> f64 {
    reg_lower_gamma(0.5 * k, 0.5 * x)
}

/// The `q`-quantile of `χ²_k` for `0 < q < 1`: the `x` with `chi2_cdf(x, k) = q`.
pub(crate) fn chi2_quantile(q: f64, k: f64) -> f64 {
    debug_assert!(q > 0.0 && q < 1.0 && k > 0.0);

    // Bracket the root. F is strictly increasing, so doubling `hi` until the CDF
    // clears `q` always yields a valid [lo, hi] bracket.
    let mut lo = 0.0_f64;
    let mut hi = k.max(1.0);
    while chi2_cdf(hi, k) < q && hi < 1e12 {
        hi *= 2.0;
    }

    let mut x = 0.5 * (lo + hi);
    for _ in 0..200 {
        let f = chi2_cdf(x, k) - q;
        if f > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        // Newton step using the pdf; fall back to the bisection midpoint if it
        // would step outside the current bracket.
        let pdf = chi2_pdf(x, k);
        let x_newton = x - f / pdf;
        x = if pdf > 0.0 && x_newton > lo && x_newton < hi {
            x_newton
        } else {
            0.5 * (lo + hi)
        };
        if hi - lo <= 1e-12 * x.max(1.0) {
            break;
        }
    }
    x
}

/// The `χ²_k` pdf at `x`, used only to accelerate the quantile's Newton step:
/// `f(x) = x^{a-1} e^{-x/2} / (2^a Γ(a))`, `a = k/2`.
fn chi2_pdf(x: f64, k: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let a = 0.5 * k;
    (-0.5 * x + (a - 1.0) * x.ln() - a * std::f64::consts::LN_2 - libm::lgamma(a)).exp()
}

/// Regularized lower incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`, `a > 0`,
/// `x ≥ 0`.
fn reg_lower_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cf(a, x)
    }
}

/// Series expansion for `P(a, x)`, which converges quickly for `x < a + 1`:
/// `P = e^{-x + a ln x − ln Γ(a)} · Σ_{n≥0} xⁿ / (a(a+1)…(a+n))`.
fn gamma_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut term = 1.0 / a;
    let mut sum = term;
    for _ in 0..1000 {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum * (-x + a * x.ln() - libm::lgamma(a)).exp()
}

/// Lentz's continued fraction for `Q(a, x) = 1 − P(a, x)`, used for `x ≥ a + 1`.
fn gamma_cf(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..1000 {
        let i = i as f64;
        let an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h * (-x + a * x.ln() - libm::lgamma(a)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Quantiles from standard χ² tables / R `qchisq`; CDF is the inverse check.
    #[test]
    fn quantiles_match_tables() {
        let cases = [
            (0.5, 1.0, 0.454_936_42),
            (0.975, 1.0, 5.023_886_2),
            (0.95, 2.0, 5.991_464_5),
            (0.975, 2.0, 7.377_758_9),
            (0.975, 3.0, 9.348_403_6),
            (0.975, 5.0, 12.832_502),
            (0.99, 10.0, 23.209_251),
        ];
        for (q, k, want) in cases {
            let got = chi2_quantile(q, k);
            assert!(
                (got - want).abs() < 1e-4,
                "χ²_{k} quantile at {q}: got {got}, want {want}"
            );
        }
    }

    // Direct CDF values from R `pchisq`, chosen to exercise BOTH branches of the
    // incomplete gamma: the series (x/2 < k/2 + 1) and Lentz's continued fraction
    // (x/2 ≥ k/2 + 1). This validates `reg_lower_gamma`, not merely `lgamma`.
    #[test]
    fn cdf_matches_known_values() {
        let cases = [
            // (x, k, pchisq(x, k))            branch (a = k/2, arg = x/2)
            (1.0, 1.0, 0.682_689_492_1),  // series      (0.5 < 1.5)
            (0.5, 2.0, 0.221_199_216_9),  // series      (0.25 < 2.0)
            (2.0, 2.0, 0.632_120_558_8),  // series      (1.0 < 2.0)
            (3.0, 3.0, 0.608_374_823_9),  // series      (1.5 < 2.5)
            (5.0, 1.0, 0.974_652_681_3),  // continued f (2.5 ≥ 1.5)
            (10.0, 2.0, 0.993_262_053_0), // continued f (5.0 ≥ 2.0)
            (20.0, 5.0, 0.998_750_269_4), // continued f (10.0 ≥ 3.5)
        ];
        for (x, k, want) in cases {
            let got = chi2_cdf(x, k);
            assert!(
                (got - want).abs() < 1e-9,
                "χ²_{k} CDF at {x}: got {got}, want {want}"
            );
        }
    }

    // Cross-check against the χ²₂ closed form 1 − e^{−x/2} across both branches.
    #[test]
    fn cdf_matches_chi2_2_closed_form() {
        for &x in &[0.1_f64, 1.0, 2.0, 3.9, 4.1, 8.0, 25.0] {
            let want = 1.0 - (-0.5 * x).exp();
            assert!((chi2_cdf(x, 2.0) - want).abs() < 1e-12, "χ²₂ at {x}");
        }
    }

    #[test]
    fn cdf_inverts_quantile() {
        for &k in &[1.0, 2.0, 3.5, 7.0, 20.0] {
            for &q in &[0.05, 0.5, 0.9, 0.975, 0.999] {
                let x = chi2_quantile(q, k);
                assert!((chi2_cdf(x, k) - q).abs() < 1e-9, "roundtrip k={k} q={q}");
            }
        }
    }

    #[test]
    fn cdf_endpoints() {
        assert_eq!(chi2_cdf(0.0, 3.0), 0.0);
        assert!(chi2_cdf(-1.0, 3.0) == 0.0);
        assert!(chi2_cdf(1e6, 3.0) > 0.999_999);
    }
}
