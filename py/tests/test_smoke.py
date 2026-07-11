"""Smoke tests mirroring the robust-rs doctests/examples.

Run after building the extension into the current environment::

    maturin develop -m py/Cargo.toml --features extension-module
    pytest py/tests
"""

import numpy as np
import pytest

import pyrobust as pr


def test_version_and_exports():
    assert isinstance(pr.__version__, str)
    assert issubclass(pr.RobustError, ValueError)


# --- Losses and theory ---------------------------------------------------------


def test_huber_loss_and_theory():
    h = pr.Huber()  # k = 1.345
    assert h.tuning == pytest.approx(1.345)
    assert h.is_redescending is False
    assert h.rho_sup is None
    assert h.psi(10.0) == pytest.approx(1.345)  # clipped score ⇒ bounded influence
    assert h.gaussian_efficiency() == pytest.approx(0.95, abs=0.01)
    assert np.isfinite(h.influence(1e6))  # bounded


def test_redescending_losses_reject_far_points():
    for loss in (pr.Tukey(), pr.Welsch(), pr.Andrews(), pr.Hampel()):
        assert loss.is_redescending is True
        assert loss.rho_sup is not None
        assert loss.gaussian_efficiency() == pytest.approx(0.95, abs=0.05)
    # Tukey weight is exactly 0 past its cutoff.
    assert pr.Tukey(4.685).weight(100.0) == 0.0


def test_loss_by_name():
    fit = pr.MEstimator(loss="huber", scale="mad")
    assert isinstance(fit, pr.MEstimator)
    # An unknown name is a plain ValueError (a bad lookup); an invalid *tuning*
    # is a RobustError (a ValueError subclass). Both are caught here.
    with pytest.raises(ValueError):
        pr.MEstimator(loss="nope")
    with pytest.raises(pr.RobustError):
        pr.Huber(-1.0)


# --- Scale ---------------------------------------------------------------------


def test_scales():
    data = [2.1, 2.3, 1.9, 2.0, 2.2, 47.0]
    assert pr.Mad().scale(data) > 0
    assert pr.Qn().scale(data) > 0
    assert pr.Sn().scale(data) > 0


# --- Location ------------------------------------------------------------------


def test_m_location_is_robust():
    data = [2.1, 2.3, 1.9, 2.0, 2.2, 47.0]  # one gross outlier (mean ~ 9.58)
    fit = pr.m_location(data)
    assert fit.estimate < 3.0  # sits with the bulk near ~2.16
    assert fit.iters >= 1


def test_trimmed_winsorized_hodges_lehmann():
    data = [2.1, 2.3, 1.9, 2.0, 2.2, 47.0]
    assert pr.trimmed_mean(data, 0.2) < 3.0
    assert pr.winsorized_mean(data, 0.2) < 5.0
    hl = pr.hodges_lehmann(data)
    assert hl.gaussian_efficiency() == pytest.approx(3 / np.pi, abs=1e-9)
    assert 0.29 < hl.breakdown_point() < 0.30


def test_empty_input_raises():
    with pytest.raises(pr.RobustError):
        pr.m_location([])


# --- Regression ----------------------------------------------------------------


def _line_with_outlier():
    x = np.column_stack([np.ones(10), np.arange(1, 11, dtype=float)])
    y = np.array([3.2, 4.8, 7.3, 9.1, 40.0, 12.7, 15.2, 17.1, 18.8, 21.3])
    return x, y


def test_m_estimator_downweights_outlier():
    x, y = _line_with_outlier()
    fit = pr.MEstimator(pr.Huber(), pr.Mad()).fit(x, y)
    assert abs(fit.coefficients[1] - 2.0) < 0.5  # slope ~ 2, not dragged
    assert fit.weights[4] < 0.1  # the outlier is down-weighted
    assert fit.breakdown_point == 0.0
    cov = fit.coef_covariance(x)
    assert cov.shape == (2, 2)


def test_theil_sen():
    x = list(np.arange(1, 11, dtype=float))
    y = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 100.0]
    ts = pr.theil_sen(x, y)
    assert ts.slope == pytest.approx(2.0, abs=0.1)
    assert ts.predict(0.0) == pytest.approx(ts.intercept)


def test_mm_recovers_stars_slope():
    x_raw, y = pr.datasets.stars_cyg()
    x = np.column_stack([np.ones(len(y)), x_raw[:, 0]])
    mm = pr.MMEstimator(seed=1).fit(x, y)
    assert mm.coefficients[1] > 0.0  # main-sequence slope recovered
    assert mm.breakdown_point == pytest.approx(0.5, abs=1e-9)
    assert mm.gaussian_efficiency() == pytest.approx(0.95, abs=0.05)


def test_mm_is_reproducible():
    x_raw, y = pr.datasets.stars_cyg()
    x = np.column_stack([np.ones(len(y)), x_raw[:, 0]])
    a = pr.MMEstimator(seed=7).fit(x, y).coefficients
    b = pr.MMEstimator(seed=7).fit(x, y).coefficients
    np.testing.assert_allclose(a, b)


def test_s_estimator_and_lts():
    x_raw, y = pr.datasets.stars_cyg()
    x = np.column_stack([np.ones(len(y)), x_raw[:, 0]])
    s = pr.SEstimator(seed=1).fit(x, y)
    assert s.breakdown_point == pytest.approx(0.5, abs=1e-9)
    lts = pr.Lts(seed=1).fit(x, y)
    assert lts.coverage <= len(y)
    assert 0.0 < lts.breakdown_point <= 0.5
    assert len(lts.subset) == lts.coverage


# --- Multivariate --------------------------------------------------------------


def test_mcd_flags_outliers():
    x, _ = pr.datasets.stackloss()
    mcd = pr.Mcd(seed=1).fit(x)
    flags = mcd.outliers(0.975)
    assert len(flags) == x.shape[0]
    assert mcd.breakdown_point > 0.0
    assert mcd.location.shape == (x.shape[1],)
    assert mcd.scatter.shape == (x.shape[1], x.shape[1])


def test_ogk_mscatter_tyler():
    x, _ = pr.datasets.stackloss()
    for fit in (pr.Ogk().fit(x), pr.MScatter().fit(x)):
        assert fit.scatter.shape == (x.shape[1], x.shape[1])
        assert len(fit.outliers(0.975)) == x.shape[0]
    tyler = pr.Tyler().fit(x)
    assert tyler.shape.shape == (x.shape[1], x.shape[1])
    # unit determinant
    assert np.linalg.det(tyler.shape) == pytest.approx(1.0, abs=1e-6)
    assert len(tyler.outliers_assuming_chi2_radial(0.975)) == x.shape[0]


def test_mahalanobis_helpers():
    x, _ = pr.datasets.stackloss()
    mean, cov = pr.classical_covariance(x)
    d = pr.mahalanobis_distances(x, mean, cov)
    assert d.shape == (x.shape[0],)
    p = x.shape[1]
    cut = pr.distance_cutoff(p, 0.975)
    flags = pr.outlier_flags(d, p, 0.975)
    assert cut > 0
    assert len(flags) == x.shape[0]


# --- Declined values must survive the boundary ----------------------------------
# The crate does not report ρ-derived efficiency for estimators with no smooth ψ.
# The binding must not fabricate one: it raises NotImplementedError with an
# explanation rather than returning a number.


def test_lts_efficiency_declined_not_fabricated():
    x_raw, y = pr.datasets.stars_cyg()
    x = np.column_stack([np.ones(len(y)), x_raw[:, 0]])
    lts = pr.Lts(seed=1).fit(x, y)
    with pytest.raises(NotImplementedError) as e:
        lts.gaussian_efficiency()
    assert "ρ-derivable" in str(e.value) or "rho-derivable" in str(e.value).replace("ρ", "rho")


def test_theil_sen_efficiency_declined():
    ts = pr.theil_sen([1.0, 2, 3, 4], [2.0, 4, 6, 8])
    with pytest.raises(NotImplementedError):
        ts.gaussian_efficiency()


def test_tyler_has_no_generic_outliers():
    # TylerFit's distances are shape-only (not χ²-calibrated), so it must expose
    # the assumption in the method name and NOT a bare .outliers() that silently
    # applies the Gaussian cutoff (the exact bug fixed structurally in Rust).
    tyler = pr.Tyler().fit(pr.datasets.stackloss()[0])
    assert not hasattr(tyler, "outliers")
    assert hasattr(tyler, "outliers_assuming_chi2_radial")


def test_regressionfit_is_unsendable_but_scatterfits_are_not():
    # RegressionFit wraps a non-Send loss ⇒ tied to its creating thread; the
    # Send+Sync fit types have no such limit. Verify the contract at runtime.
    import threading

    x_raw, y = pr.datasets.stars_cyg()
    x = np.column_stack([np.ones(len(y)), x_raw[:, 0]])
    reg = pr.MEstimator(pr.Huber(), pr.Mad()).fit(x, y)
    mcd = pr.Mcd(seed=1).fit(pr.datasets.stackloss()[0])

    def touch(obj, attr, box):
        try:
            getattr(obj, attr)
            box.append(None)
        except BaseException as e:  # PanicException is a BaseException
            box.append(e)

    box_reg, box_mcd = [], []
    for obj, attr, box in ((reg, "coefficients", box_reg), (mcd, "location", box_mcd)):
        t = threading.Thread(target=touch, args=(obj, attr, box))
        t.start()
        t.join()

    assert box_reg[0] is not None  # off-thread access raised
    assert "unsendable" in str(box_reg[0])
    assert box_mcd[0] is None  # McdFit is fine across threads


def test_anyloss_delegates_full_trait():
    # Exact efficiency values (not just "a number") prove psi/psi_prime/rho_sup
    # all delegate through AnyLoss.
    assert pr.Huber().gaussian_efficiency() == pytest.approx(0.95, abs=0.01)
    assert pr.Tukey().gaussian_efficiency() == pytest.approx(0.95, abs=0.01)
    assert pr.Huber().rho_sup is None  # unbounded ρ
    assert pr.Tukey().rho_sup is not None  # bounded ρ (redescends)
    assert pr.Huber().psi_prime(0.0) == pytest.approx(1.0)
