//! Robust statistics for Rust, built around a single abstraction: the M-estimator
//! loss `ρ`, its score `ψ = ρ'` and the IRLS weight `w(r) = ψ(r)/r`. Almost every
//! robust method (a robust mean, a median, a Huber or bisquare regression, a
//! high-breakdown S/MM fit, a robust covariance) is an M-estimator differing only
//! in that loss, so implementing the [`RhoFunction`](crate::rho::RhoFunction) trait
//! is enough to add an estimator. On that spine the crate layers location,
//! regression and multivariate estimators, plus a result surface that reports the
//! sampling theory derived from each loss: influence function, asymptotic variance,
//! Gaussian efficiency, coefficient covariance and breakdown point.
//!
//! The workspace is split at the linear-algebra boundary. [`robust_rs_core`] owns
//! the losses, robust scale, solver and theory and depends on nothing heavier than
//! `libm` (so it builds for `wasm32`); this crate adds the estimators, which need
//! `ndarray` + `faer`. The core is re-exported here, so `use robust_rs::prelude::*;`
//! is all most callers need.
//!
//! # Choosing an estimator
//!
//! If you know the shape of your problem but not the name of the method:
//!
//! | Your problem | Reach for | Notes |
//! |---|---|---|
//! | A robust average of one variable | [`m_location`](crate::location::m_location) | or [`trimmed_mean`](crate::location::trimmed_mean) / [`hodges_lehmann`](crate::location::hodges_lehmann) |
//! | Regression, outliers only in `y` | [`MEstimator`](crate::regression::MEstimator) + [`Huber`](crate::rho::Huber) | fast, convex, unique; **0 breakdown** against leverage |
//! | Regression, outliers also in `X` (leverage) | [`MMEstimator`](crate::regression::MMEstimator) **(start here)** | R's `lmrob`; 50% breakdown **and** ≈95% efficiency |
//! | Regression, want a bare high-breakdown fit | [`SEstimator`](crate::regression::SEstimator) / [`Lts`](crate::regression::Lts) | 50% breakdown, lower efficiency (what MM builds on) |
//! | Simple one-predictor regression, no tuning | [`theil_sen`](crate::regression::theil_sen) | median of pairwise slopes |
//! | Robust covariance / multivariate outliers | [`Mcd`](crate::multivariate::Mcd) **(start here)** / [`Ogk`](crate::multivariate::Ogk) | MCD is affine-equivariant; OGK is deterministic |
//! | Just a robust spread of some numbers | [`Mad`](crate::scale::Mad) | or [`Qn`](crate::scale::Qn) / [`Sn`](crate::scale::Sn); Gaussian-consistent |
//!
//! Rule of thumb: for regression reach for `MMEstimator::default()` unless you
//! know the contamination is purely vertical (then `MEstimator` is cheaper); for
//! covariance and multivariate outlier flagging reach for `Mcd::new()`.
//!
//! # Robust location
//!
//! A robust location estimate is unmoved by a gross outlier that wrecks the mean:
//!
//! ```
//! use robust_rs::prelude::*;
//!
//! let data = [2.1, 2.3, 1.9, 2.0, 2.2, 47.0]; // one gross outlier (mean ≈ 9.58)
//! let fit = m_location(&data, &Huber::default(), &Mad::default(), &Control::default())?;
//! assert!(fit.estimate < 3.0); // sits with the bulk near ≈ 2.16, not the mean
//! # Ok::<(), robust_rs::error::RobustError>(())
//! ```
//!
//! # Robust regression
//!
//! An outlier in the response drags ordinary least squares off; a Huber
//! M-estimator resists it and drives the offending observation's weight toward 0:
//!
//! ```
//! use ndarray::array;
//! use robust_rs::prelude::*;
//!
//! // y ≈ 1 + 2x, with one gross outlier at x = 5.
//! let x = array![
//!     [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0],
//!     [1.0, 6.0], [1.0, 7.0], [1.0, 8.0], [1.0, 9.0], [1.0, 10.0],
//! ];
//! let y = array![3.2, 4.8, 7.3, 9.1, 40.0, 12.7, 15.2, 17.1, 18.8, 21.3];
//!
//! let fit = MEstimator::new(Huber::default(), Mad::default()).fit(&x, &y)?;
//! assert!((fit.coefficients()[1] - 2.0).abs() < 0.5); // slope ≈ 2, not dragged
//! assert!(fit.weights[4] < 0.1);                        // the outlier is down-weighted
//! # Ok::<(), robust_rs::error::RobustError>(())
//! ```
//!
//! # Sampling theory
//!
//! Every [`RegressionFit`](crate::estimator::RegressionFit) implements
//! [`RobustEstimator`](crate::estimator::RobustEstimator), reporting the sampling
//! theory derived from its loss:
//!
//! ```
//! # use ndarray::array;
//! # use robust_rs::prelude::*;
//! # let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]];
//! # let y = array![1.1, 1.9, 3.2, 3.9, 5.1];
//! let fit = MEstimator::new(Huber::default(), Mad::default()).fit(&x, &y)?;
//!
//! let _eff = fit.gaussian_efficiency();   // ≈ 0.95, asymptotic efficiency at the Gaussian
//! let _var = fit.asymptotic_variance();   // V(ψ) = E[ψ²] / (E[ψ'])²
//! let _cov = fit.coef_covariance(&x);     // ŝ²·V·(XᵀX)⁻¹, coefficient covariance
//! let inf = fit.influence_function();     // x ↦ ψ(x) / E[ψ'], a bounded closure
//! assert!(inf(1e6).is_finite());          // bounded influence: an extreme residual can't blow up
//! # Ok::<(), robust_rs::error::RobustError>(())
//! ```
//!
//! # High-breakdown regression
//!
//! A plain M-estimator bounds the influence of a large *residual* but has ~0
//! breakdown against *leverage*. The high-breakdown estimators escape that; MM
//! (R's `lmrob` default) pairs 50% breakdown with ≈ 95% efficiency. On the
//! `starsCYG` H–R diagram four giant stars flip the OLS slope negative; MM recovers
//! the physical positive slope and rejects them:
//!
//! ```
//! use ndarray::Array2;
//! use robust_rs::prelude::*;
//!
//! let (x_raw, y) = robust_rs::datasets::stars_cyg();
//! let mut x = Array2::ones((x_raw.nrows(), 2)); // prepend an intercept column
//! x.column_mut(1).assign(&x_raw.column(0));
//!
//! let mm = MMEstimator::default().fit(&x, &y)?;    // reproducible (fixed default seed)
//! assert!(mm.coefficients()[1] > 0.0);             // main-sequence slope recovered
//! assert!((mm.breakdown_point() - 0.5).abs() < 1e-9);
//! # Ok::<(), robust_rs::error::RobustError>(())
//! ```
//!
//! Randomized estimators ([`SEstimator`](crate::regression::SEstimator),
//! [`MMEstimator`](crate::regression::MMEstimator), [`Lts`](crate::regression::Lts),
//! [`Mcd`](crate::multivariate::Mcd)) are reproducible by default and expose a
//! `.seed(u64)` builder and a `fit_with_rng` escape hatch; their per-subsample RNG
//! sub-streams make results thread-count invariant.
//!
//! # Multivariate location and scatter
//!
//! The multivariate analogue of a robust residual is a robust **Mahalanobis
//! distance** built from a robust location–scatter pair `(μ̂, Σ̂)`. Feeding a
//! robust pair into the distance/outlier map defeats the *masking* that lets
//! outliers hide from the classical covariance:
//!
//! ```
//! use robust_rs::prelude::*;
//!
//! let (x, _y) = robust_rs::datasets::stackloss();      // 21 × 3 operating variables
//! let mcd = Mcd::new().seed(1).fit(&x)?;                // FAST-MCD, affine equivariant
//! let flags = mcd.outliers(0.975);                     // χ²_{p,0.975} cutoff
//! assert_eq!(flags.len(), x.nrows());
//! # Ok::<(), robust_rs::error::RobustError>(())
//! ```
//!
//! # What is implemented
//!
//! ## Losses ([`rho`]): implement [`RhoFunction`](crate::rho::RhoFunction)
//!
//! | Loss | Default tuning | ψ | Gaussian efficiency | Bounded ρ | Redescending |
//! |---|---|---|---|---|---|
//! | [`LeastSquares`](crate::rho::LeastSquares) | – | monotone | 1.000 | no | no |
//! | [`L1`](crate::rho::L1) | – | monotone | (median) | no | no |
//! | [`Huber`](crate::rho::Huber) | `k = 1.345` | monotone (clipped) | ≈ 0.95 | no | no |
//! | [`TukeyBiweight`](crate::rho::TukeyBiweight) | `c = 4.685` | redescends to 0 | ≈ 0.95 | `c²/6` | yes |
//! | [`Cauchy`](crate::rho::Cauchy) | `c = 2.3849` | soft redescend | ≈ 0.95 | **no** (soft) | yes |
//! | [`Welsch`](crate::rho::Welsch) | `c = 2.9846` | redescends | ≈ 0.95 | `c²/2` | yes |
//! | [`Andrews`](crate::rho::Andrews) | `c = 1.339` | sine, hard cutoff | ≈ 0.95 | `2c²` | yes |
//! | [`Hampel`](crate::rho::Hampel) | `(2, 4, 8)` | 3-part linear | ≈ 0.99 | `(a/2)(b+c−a)` | yes |
//!
//! The trait exposes `rho`, `psi`, `weight`, `psi_prime`, `tuning`,
//! `is_redescending` and `rho_sup`. See [`docs/conventions.md`] for the
//! `ψ'(0) = 1 ⇒ weight(0) = 1` convention every loss follows.
//!
//! ## Robust scale ([`scale`]): implement [`ScaleEstimator`](crate::scale::ScaleEstimator)
//!
//! | Estimator | Formula | Consistency | Efficiency | Breakdown |
//! |---|---|---|---|---|
//! | [`Mad`](crate::scale::Mad) | `1.4826 · med\|rᵢ − med r\|` | `1/Φ⁻¹(¾)` | ≈ 0.37 | 0.5 |
//! | [`HuberProposal2`](crate::scale::HuberProposal2) | joint `(μ, s)`, `Σψ² = β` | closed-form `β` | – | – |
//! | [`SScale`](crate::scale::SScale) | `s` solving `(1/n)Σρ(rᵢ/s) = δ` | `δ = E_Φ[ρ]` | (loss) | up to 0.5 |
//! | [`Qn`](crate::scale::Qn) | `¼`-quantile of pairwise `\|rᵢ − rⱼ\|` | `2.2219` | ≈ 0.82 | 0.5 |
//! | [`Sn`](crate::scale::Sn) | `med_i med_j \|rᵢ − rⱼ\|` | `1.1926` | ≈ 0.58 | 0.5 |
//!
//! ## Location ([`location`])
//!
//! - [`m_location`](crate::location::m_location): M-estimate of location by IRLS
//!   (returns [`LocationFit`](crate::location::LocationFit)).
//! - [`trimmed_mean`](crate::location::trimmed_mean) /
//!   [`winsorized_mean`](crate::location::winsorized_mean): `α`-trimmed / Winsorized means.
//! - [`hodges_lehmann`](crate::location::hodges_lehmann): median of the Walsh
//!   averages (returns [`HodgesLehmannFit`](crate::location::HodgesLehmannFit)).
//!
//! ## Regression ([`regression`])
//!
//! - [`MEstimator`](crate::regression::MEstimator): M-regression by IRLS
//!   (convex/unique for monotone losses; **0 breakdown** against leverage).
//! - [`SEstimator`](crate::regression::SEstimator): FAST-S, 50% breakdown by
//!   minimizing an S-scale of the residuals.
//! - [`MMEstimator`](crate::regression::MMEstimator): S-init + fixed-scale
//!   redescending M-step ⇒ 50% breakdown **and** ≈ 95% efficiency.
//! - [`Lts`](crate::regression::Lts): FAST-LTS with a coverage knob (returns
//!   [`LtsFit`](crate::regression::LtsFit)).
//! - [`theil_sen`](crate::regression::theil_sen): median of pairwise slopes for
//!   simple regression (returns [`TheilSenFit`](crate::regression::TheilSenFit)).
//! - [`weighted_least_squares`](crate::wls::weighted_least_squares): the WLS core
//!   (rank-revealing QR on `√W·X`, never the normal equations).
//! - M/S/MM fits are [`RegressionFit`](crate::estimator::RegressionFit)s
//!   implementing [`RobustEstimator`](crate::estimator::RobustEstimator).
//!
//! ## Multivariate ([`multivariate`])
//!
//! [`Mcd`](crate::multivariate::Mcd), [`Ogk`](crate::multivariate::Ogk) and
//! [`MScatter`](crate::multivariate::MScatter) produce a Gaussian-calibrated
//! location–scatter pair and implement [`RobustScatter`](crate::multivariate::RobustScatter)
//! (the χ²-calibrated distance/outlier map); [`Tyler`](crate::multivariate::Tyler)
//! identifies *shape* only, so it returns a bespoke
//! [`TylerFit`](crate::multivariate::TylerFit) that does **not** implement that
//! trait (its distances aren't χ²-calibrated).
//!
//! - [`Mcd`](crate::multivariate::Mcd): FAST-MCD, affine equivariant, 50%
//!   breakdown, reweighted (RMCD) by default (returns
//!   [`McdFit`](crate::multivariate::McdFit)).
//! - [`Ogk`](crate::multivariate::Ogk): deterministic, positive-definite
//!   orthogonalized Gnanadesikan–Kettenring estimator.
//! - [`MScatter`](crate::multivariate::MScatter): monotone M-estimator of
//!   location/scatter (low breakdown; the multivariate analogue of M-regression).
//! - [`Tyler`](crate::multivariate::Tyler): distribution-free M-estimator of
//!   *shape* (unit determinant); returns [`TylerFit`](crate::multivariate::TylerFit).
//! - [`mahalanobis`](crate::multivariate::mahalanobis): the robust distance /
//!   outlier map over any Gaussian-calibrated `(μ̂, Σ̂)` pair, plus the classical
//!   baseline.
//!
//! ## Theory ([`theory`])
//!
//! - [`influence_function`](crate::theory::influence_function),
//!   [`asymptotic_variance`](crate::theory::asymptotic_variance),
//!   [`gaussian_efficiency`](crate::theory::gaussian_efficiency),
//!   [`breakdown_point`](crate::theory::breakdown_point).
//! - [`gauss_hermite`](crate::theory::gauss_hermite) quadrature (Golub–Welsch) and
//!   the expectation helpers [`expect_psi_squared`](crate::theory::expect_psi_squared) /
//!   [`expect_psi_prime`](crate::theory::expect_psi_prime) (the latter via Stein's
//!   identity, so kinked scores and L1 are exact).
//!
//! ## Types and errors
//!
//! Newtypes make illegal states unrepresentable:
//! [`Scale`](crate::types::Scale), [`TuningConstant`](crate::types::TuningConstant),
//! and the [`RawResidual`](crate::types::RawResidual) /
//! [`ScaledResidual`](crate::types::ScaledResidual) pair; all fallible operations
//! return [`RobustError`](crate::error::RobustError).
//!
//! # Cargo features
//!
//! - `rayon`: parallelize the FAST-MCD random starts (thread-count invariant).
//!
//! # Datasets
//!
//! [`datasets::stackloss`] and [`datasets::stars_cyg`] vendor the classic
//! `robustbase` reference data (predictors only; prepend your own intercept
//! column).
//!
//! [`docs/conventions.md`]: https://github.com/coes300/robust-rs/blob/master/docs/conventions.md
#![deny(missing_docs)]

pub mod datasets;
pub mod estimator;
pub mod location;
pub mod multivariate;
pub mod regression;
pub mod wls;

mod util; // small internal numeric helpers (shared median)

pub mod prelude;

#[doc(inline)]
pub use robust_rs_core::{error, rho, scale, theory, types};
