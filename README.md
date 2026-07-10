# robust-rs

[![Crates.io](https://img.shields.io/crates/v/robust-rs.svg)](https://crates.io/crates/robust-rs)
[![Documentation](https://docs.rs/robust-rs/badge.svg)](https://docs.rs/robust-rs)
[![CI](https://github.com/coes300/robust-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/coes300/robust-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

Robust statistics for Rust, built around one abstraction: the M-estimator. Almost
every robust method is a loss `ρ` with a score `ψ = ρ'` and an IRLS weight `ψ(r)/r`.
Implement the `RhoFunction` trait and you get location, scale and regression
estimation, plus the sampling theory (influence function, efficiency, covariance)
reported off each fit.

## Features

- **Losses** behind one `RhoFunction` trait: Huber, Tukey biweight, Cauchy, Welsch,
  Andrews, Hampel, L1 and least squares.
- **Robust scale**: MAD, Huber Proposal 2, the S-scale, Qn and Sn, Fisher-consistent
  for σ at the Gaussian.
- **Robust regression**: M, S, MM (the `lmrob` equivalent), LTS (FAST-LTS) and
  Theil–Sen. Randomized fits are reproducible by default via seeded,
  thread-count-invariant RNG sub-streams.
- **Multivariate location/scatter**: FAST-MCD, OGK and monotone M-scatter, sharing a
  robust-Mahalanobis / χ² outlier map, plus Tyler's distribution-free shape estimator.
- **Sampling theory off each fit**: influence function, asymptotic variance, Gaussian
  efficiency and coefficient covariance, besides point estimates.
- Weighted least squares via rank-revealing QR (never the normal equations);
  Gauss–Hermite quadrature (Golub–Welsch) with `E[ψ']` via Stein's identity, so kinked
  scores and L1 stay exact.
- `robust-rs-core` is dependency-light and `wasm32`-friendly (no linear algebra).

## Choosing an estimator

If you know the shape of your problem but not the name of the method:

| Your problem | Reach for | Notes |
|---|---|---|
| A robust average of one variable | `m_location` | or `trimmed_mean` / `hodges_lehmann` |
| Regression, outliers only in `y` | `MEstimator` + `Huber` | fast, convex, unique; **0 breakdown** against leverage |
| Regression, outliers also in `X` (leverage) | `MMEstimator` **(start here)** | R's `lmrob`; 50% breakdown **and** ≈95% efficiency |
| Regression, want a bare high-breakdown fit | `SEstimator` / `Lts` | 50% breakdown, lower efficiency (what MM builds on) |
| Simple one-predictor regression, no tuning | `theil_sen` | median of pairwise slopes |
| Robust covariance / multivariate outliers | `Mcd` **(start here)** / `Ogk` | MCD is affine-equivariant; OGK is deterministic |
| Just a robust spread of some numbers | `Mad` | or `Qn` / `Sn`; Gaussian-consistent |

Rule of thumb: for regression reach for `MMEstimator::default()` unless you know the
contamination is purely vertical (then `MEstimator` is cheaper); for covariance and
multivariate outlier flagging reach for `Mcd::new()`.

## Installation

```toml
[dependencies]
robust-rs = "0.1"
```

## Quick start

A robust location estimate is unmoved by a gross outlier that wrecks the mean:

```rust
use robust_rs::prelude::*;

let data = [2.1, 2.3, 1.9, 2.0, 2.2, 47.0];   // one gross outlier
let fit = m_location(&data, &Huber::default(), &Mad::default(), &Control::default())?;
println!("{:.2}", fit.estimate);              // 2.16   (the mean is 9.58)
```

## Usage

### Robust regression

An outlier in the response drags ordinary least squares off the trend; a Huber
M-estimator resists it and down-weights the offending observation toward zero.

```rust
use ndarray::array;
use robust_rs::prelude::*;

// y ≈ 1 + 2x, with one gross outlier at x = 5.
let x = array![
    [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0],
    [1.0, 6.0], [1.0, 7.0], [1.0, 8.0], [1.0, 9.0], [1.0, 10.0],
];
let y = array![3.2, 4.8, 7.3, 9.1, 40.0, 12.7, 15.2, 17.1, 18.8, 21.3];

let fit = MEstimator::new(Huber::default(), Mad::default()).fit(&x, &y)?;
let beta = fit.coefficients();

println!("intercept = {:.2}, slope = {:.2}", beta[0], beta[1]); // 1.17, 1.99  (OLS: 4.92, 1.82)
println!("outlier weight = {:.3}", fit.weights[4]);             // 0.026
```

### Sampling theory

A fitted estimator reports the sampling theory derived from its loss:

```rust
let fit = MEstimator::new(Huber::default(), Mad::default()).fit(&x, &y)?;

fit.gaussian_efficiency();   // 0.950, asymptotic efficiency at the Gaussian
fit.asymptotic_variance();   // V(ψ) = E[ψ²] / (E[ψ'])²
fit.coef_covariance(&x);     // s²·V·(XᵀX)⁻¹  (coefficient covariance)
fit.influence_function();    // x ↦ ψ(x) / E[ψ'], a bounded closure
```

## Examples

```
cargo run --example demo           # OLS dragged by vertical outliers; Huber recovers the line
cargo run --example stars_mm       # the classic starsCYG high-leverage example
cargo run --example mcd_outliers   # multivariate on Brownlee's stackloss data
```

## Workspace

- **`robust-rs-core`**: losses, robust scale and the influence/variance/efficiency
  theory. No linear-algebra dependency; usable on its own and on `wasm32`.
- **`robust-rs`**: the location, regression and multivariate estimators, built on
  the core.

## Status

Working and tested; see `CHANGELOG.md` for per-stage detail.

- **v0.1**: `RhoFunction` losses, MAD scale, location and regression M-estimation,
  and the sampling-theory API. high-breakdown regression (S, MM, LTS/FAST-LTS, Theil–Sen), the full loss
  set (Cauchy, Welsch andrews, Hampel), extra scales (Qn, Sn, S-scale, Huber
  Proposal 2), trimmed/Winsorized means and Hodges–Lehmann and seeded RNG for the
  randomized fits. Multivariate location/scatter (FAST-MCD, OGK, M-scatter, Tyler) and the
  robust-Mahalanobis / χ² outlier map.

A plain M-estimator has ~0 breakdown against leverage: it bounds the influence of a
large *residual*, but a high-leverage outlier can still carry the fit. On the
`starsCYG` data (the `stars_mm` example) both OLS and Huber return a negative slope
through a physically positive relationship, because four giant stars sit at high
leverage with small residuals; `MMEstimator` recovers the positive slope and drives the
four giants to zero weight.

**Planned (v0.2+):** robust PCA, cellwise-outlier tooling, an optional LAPACK backend,
and a `pyrobust` PyO3 binding.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.