"""The tests that actually exercise the FFI boundary and cross-check the numbers.

Two kinds of check that ``cargo`` cannot see:

1. **Cross-language equivalence.** The binding must be a transparent pass-through:
   the same estimator on the same data must give the same answer whether called
   from Rust or from Python. The ``RUST_*`` constants below were produced by an
   independent Rust program (``robust-rs`` used directly, not through the
   binding); Python must reproduce them to ~1e-12. Divergence would mean the
   numpy↔ndarray marshalling or the enum dispatch is silently wrong.

2. **Cross-reference** against established implementations (scikit-learn,
   statsmodels) now that they are callable in the same process.
"""

import numpy as np
import pytest

import robustat_py as rp

# --- Reference values from an independent Rust run (robust-rs, not the binding).
#     M-Huber and MM on starsCYG (X = [1, log.Te]); MCD location on stackloss.
RUST_M_HUBER = (6.85823392836914092, -0.42626477881288038)
RUST_MM_SEED1 = (-5.12205020556999635, 2.28763618282871395)
RUST_MM_SEED1_EFF = 0.94999730616566225
RUST_MCD_SEED1_LOC = (59.5, 20.83333333333333215, 87.33333333333332860)
RUST_HUBER_EFF = 0.95023092846305857
RUST_TUKEY_EFF = 0.94999730616566225


def _stars():
    x_raw, y = rp.datasets.stars_cyg()
    return np.column_stack([np.ones(len(y)), x_raw[:, 0]]), y


class TestCrossLanguageEquivalence:
    def test_m_huber_matches_rust(self):
        x, y = _stars()
        beta = rp.MEstimator(rp.Huber(), rp.Mad()).fit(x, y).coefficients
        np.testing.assert_allclose(beta, RUST_M_HUBER, rtol=0, atol=1e-12)

    def test_mm_matches_rust_bitwise(self):
        # MM is randomized but reproducible from its seed; Python calls the same
        # seeded Rust code, so it must agree to full f64 precision.
        x, y = _stars()
        fit = rp.MMEstimator(seed=1).fit(x, y)
        np.testing.assert_allclose(fit.coefficients, RUST_MM_SEED1, rtol=0, atol=1e-12)
        assert fit.gaussian_efficiency() == pytest.approx(RUST_MM_SEED1_EFF, abs=1e-12)

    def test_mcd_matches_rust(self):
        xs, _ = rp.datasets.stackloss()
        loc = rp.Mcd(seed=1).fit(xs).location
        np.testing.assert_allclose(loc, RUST_MCD_SEED1_LOC, rtol=0, atol=1e-9)

    def test_loss_theory_matches_rust(self):
        # Exact theory values prove AnyLoss delegates psi/psi_prime/rho_sup, not
        # just rho/psi/weight: a stubbed derivative would move these numbers.
        assert rp.Huber().gaussian_efficiency() == pytest.approx(RUST_HUBER_EFF, abs=1e-12)
        assert rp.Tukey().gaussian_efficiency() == pytest.approx(RUST_TUKEY_EFF, abs=1e-12)
        # avar is the reciprocal of efficiency, a second independent path.
        assert rp.Huber().asymptotic_variance() == pytest.approx(1.0 / RUST_HUBER_EFF, abs=1e-12)


class TestSklearnCrossReference:
    """Cross-check MCD against scikit-learn's MinCovDet (an established RMCD)."""

    def test_mcd_flags_agree_with_sklearn(self):
        MinCovDet = pytest.importorskip("sklearn.covariance").MinCovDet
        chi2 = pytest.importorskip("scipy.stats").chi2

        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(size=(40, 2)), np.full((5, 2), 8.0)])
        injected = set(range(40, 45))

        pr_flags = set(np.where(rp.Mcd(seed=1).fit(x).outliers(0.975))[0].tolist())
        sk = MinCovDet(random_state=0).fit(x)
        sk_flags = set(np.where(sk.mahalanobis(x) > chi2.ppf(0.975, 2))[0].tolist())

        # Both must catch every injected outlier, and agree with each other.
        assert injected <= pr_flags
        assert injected <= sk_flags
        assert pr_flags == sk_flags

    def test_mcd_location_close_to_sklearn(self):
        MinCovDet = pytest.importorskip("sklearn.covariance").MinCovDet
        xs, _ = rp.datasets.stackloss()
        pr_loc = rp.Mcd(seed=1).fit(xs).location
        sk_loc = MinCovDet(random_state=0).fit(xs).location_
        # Different corrections/subset search ⇒ close, not identical.
        np.testing.assert_allclose(pr_loc, sk_loc, rtol=0.10)


class TestStatsmodelsCrossReference:
    """Cross-check Huber M-regression against statsmodels' RLM."""

    def test_huber_matches_rlm(self):
        sm = pytest.importorskip("statsmodels.api")
        x, y = _stars()
        pr_beta = rp.MEstimator(rp.Huber(1.345), rp.Mad()).fit(x, y).coefficients
        rlm = sm.RLM(y, x, M=sm.robust.norms.HuberT(t=1.345)).fit()
        np.testing.assert_allclose(pr_beta, rlm.params, rtol=0.02)
