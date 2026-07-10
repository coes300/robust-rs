# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/); this crate
follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-07-10

Initial release.

### Losses (`RhoFunction`)
- `LeastSquares`, `L1`, `Huber` (k = 1.345), `TukeyBiweight` (c = 4.685),
  `Cauchy` (2.3849), `Welsch` (2.9846) and `Andrews` (1.339) — each ≈ 95%
  Gaussian-efficient — plus `Hampel` (2, 4, 8), ≈ 99%. `Cauchy` is a *soft*
  redescender (bounded ψ, unbounded ρ, so `rho_sup() == None`).

### Robust scale (`ScaleEstimator`)
- `Mad`, `HuberProposal2` (closed-form β), `SScale` (the M-scale of residuals,
  generic over ρ; requires a bounded ρ, else `RobustError::UnboundedLoss`) and
  `Qn` (2.2219) / `Sn` (1.1926) with the Croux–Rousseeuw finite-sample
  corrections. Fisher-consistent for σ at the Gaussian.

### Location
- `m_location` (IRLS), `trimmed_mean` / `winsorized_mean` and `hodges_lehmann`
  (median of the Walsh averages).

### Regression
- `MEstimator` (M-regression by IRLS; ~0 breakdown against leverage),
  `SEstimator` (FAST-S, 50% breakdown), `MMEstimator` (S-init + fixed-scale
  redescending M-step ⇒ 50% breakdown **and** ≈ 95% efficiency; R's `lmrob`
  default), `Lts` (FAST-LTS with a coverage knob) and `theil_sen`.

### Multivariate location/scatter
- `Mcd` (FAST-MCD: affine-equivariant, 50% breakdown, reweighted RMCD by
  default), `Ogk` (deterministic, positive-definite), `MScatter` (monotone
  M-estimator) and `Tyler` (distribution-free shape). `ScatterFit` / `McdFit`
  implement `RobustScatter` (a χ²-calibrated Mahalanobis distance / outlier
  map); `mahalanobis` exposes that map over any `(μ̂, Σ̂)` pair, plus the
  classical baseline.

### Sampling theory (`theory`)
- `influence_function`, `asymptotic_variance`, `gaussian_efficiency` and
  `breakdown_point`, reported off each M-estimator fit through the
  `RobustEstimator` trait. Expectations use in-crate Gauss–Hermite quadrature
  (Golub–Welsch), with `E[ψ']` via Stein's identity so kinked scores and L1 stay
  exact.

### Reproducibility
- The randomized estimators (`SEstimator`, `MMEstimator`, `Lts`, `Mcd`) use a
  version-stable `ChaCha8Rng`, reproducible by default with a `.seed(u64)`
  builder and a `fit_with_rng` escape hatch. Per-subsample sub-streams make
  results thread-count invariant, so the optional `rayon` feature does not change
  the answer.

### Workspace
- Split at the linear-algebra boundary: `robust-rs-core` (losses, scale, theory;
  `libm` / `num-traits` / `thiserror` only, `wasm32`-friendly) and `robust-rs`
  (the estimators, on `ndarray` + `faer`). Embedded `stackloss` / `starsCYG`
  reference datasets.

### Known limitations
- MCD applies only the asymptotic consistency factor; R `covMcd`'s finite-sample
  multiplier (Pison et al. 2002) is deferred.
- `Qn` / `Sn` are computed in `O(n²)`; the Croux–Rousseeuw `O(n log n)`
  algorithms are a follow-up.
- FAST-S defers the Salibián-Barrera–Yohai partial-scale-rejection speedup.
- `Tyler` identifies shape only; its distances are not χ²-calibrated (the
  Gaussian outlier map is gated behind `TylerFit::outliers_assuming_chi2_radial`).
- The `lapack` feature (swap `faer` for `ndarray-linalg`) and the `serde` feature
  (serialize fitted models) are reserved.

[0.1.0]: https://github.com/coes300/robust-rs/releases/tag/v0.1.0
