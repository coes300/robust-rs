"""Robust statistics for Python.

``pyrobust`` wraps the Rust crate `robust-rs
<https://docs.rs/robust-rs>`_: M/S/MM regression, robust scale, robust
multivariate location–scatter (MCD, OGK, Tyler) and the influence-function
sampling theory, all with a numpy interface.

The API is class-based for the configurable estimators (build one, then call
``.fit``) and functional for the parameter-free ones::

    import numpy as np
    import pyrobust as pr

    # High-breakdown, high-efficiency regression (R's lmrob default).
    X, y = pr.datasets.stars_cyg()
    X = np.column_stack([np.ones(len(y)), X[:, 0]])   # prepend intercept
    fit = pr.MMEstimator(seed=1).fit(X, y)
    fit.coefficients            # -> np.ndarray, main-sequence slope recovered
    fit.gaussian_efficiency()   # ~0.95

    # Robust covariance + multivariate outlier flags.
    Xs, _ = pr.datasets.stackloss()
    mcd = pr.Mcd(seed=1).fit(Xs)
    mcd.outliers(0.975)

Losses and scales are created with factory callables that double as their
familiar names: ``pr.Huber(1.345)``, ``pr.Tukey(4.685)``, ``pr.Mad()``,
``pr.Qn()`` … Every estimator argument that takes a loss or scale also accepts a
lowercase name string, e.g. ``pr.MEstimator(loss="huber", scale="mad")``.
"""

from __future__ import annotations

import sys as _sys

from ._pyrobust import (  # noqa: F401  (re-exported)
    __version__,
    # exception
    RobustError,
    # value types
    Loss,
    ScaleEstimator,
    # result types
    LocationFit,
    HodgesLehmannFit,
    RegressionFit,
    LtsFit,
    TheilSenFit,
    ScatterFit,
    McdFit,
    TylerFit,
    # regression estimators
    MEstimator,
    SEstimator,
    MMEstimator,
    Lts,
    # multivariate estimators
    Mcd,
    Ogk,
    MScatter,
    Tyler,
    # loss factories
    least_squares,
    l1,
    huber,
    tukey,
    cauchy,
    welsch,
    andrews,
    hampel,
    # loss-derived theory
    gaussian_efficiency,
    asymptotic_variance,
    breakdown_point,
    # scale factories
    mad,
    qn,
    sn,
    huber_proposal2,
    # location
    m_location,
    trimmed_mean,
    winsorized_mean,
    hodges_lehmann,
    # regression
    theil_sen,
    # multivariate Mahalanobis / outlier map
    mahalanobis_distances,
    classical_covariance,
    distance_cutoff,
    outlier_flags,
    # submodule
    datasets,
)

# `import pyrobust.datasets` and `pr.datasets.stackloss()` both work.
_sys.modules.setdefault("pyrobust.datasets", datasets)

# CamelCase aliases: `pr.Huber(1.345)` reads like a class while calling the
# underlying factory. Each returns a `Loss` / `ScaleEstimator`.
LeastSquares = least_squares
L1 = l1
Huber = huber
Tukey = tukey
TukeyBiweight = tukey
Bisquare = tukey
Cauchy = cauchy
Welsch = welsch
Andrews = andrews
Hampel = hampel

Mad = mad
Qn = qn
Sn = sn
HuberProposal2 = huber_proposal2

__all__ = [
    "__version__",
    "RobustError",
    "Loss",
    "ScaleEstimator",
    "LocationFit",
    "HodgesLehmannFit",
    "RegressionFit",
    "LtsFit",
    "TheilSenFit",
    "ScatterFit",
    "McdFit",
    "TylerFit",
    "MEstimator",
    "SEstimator",
    "MMEstimator",
    "Lts",
    "Mcd",
    "Ogk",
    "MScatter",
    "Tyler",
    # loss factories + aliases
    "least_squares",
    "l1",
    "huber",
    "tukey",
    "cauchy",
    "welsch",
    "andrews",
    "hampel",
    "LeastSquares",
    "L1",
    "Huber",
    "Tukey",
    "TukeyBiweight",
    "Bisquare",
    "Cauchy",
    "Welsch",
    "Andrews",
    "Hampel",
    # theory
    "gaussian_efficiency",
    "asymptotic_variance",
    "breakdown_point",
    # scale factories + aliases
    "mad",
    "qn",
    "sn",
    "huber_proposal2",
    "Mad",
    "Qn",
    "Sn",
    "HuberProposal2",
    # location
    "m_location",
    "trimmed_mean",
    "winsorized_mean",
    "hodges_lehmann",
    # regression
    "theil_sen",
    # multivariate
    "mahalanobis_distances",
    "classical_covariance",
    "distance_cutoff",
    "outlier_flags",
    "datasets",
]
