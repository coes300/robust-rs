# robust-rs-core

[![Crates.io](https://img.shields.io/crates/v/robust-rs-core.svg)](https://crates.io/crates/robust-rs-core)
[![Documentation](https://docs.rs/robust-rs-core/badge.svg)](https://docs.rs/robust-rs-core)

Core abstractions for [`robust-rs`]: the `RhoFunction` loss trait (Huber, Tukey
biweight, Cauchy, Welsch andrews, Hampel, L1, least squares), robust scale
estimators (MAD, Huber Proposal 2, S-scale, Qn, Sn) and the influence-function /
asymptotic-variance / Gaussian-efficiency / breakdown-point theory built on
Gauss–Hermite quadrature.

Dependency-light (only `libm`, `num-traits`, `thiserror`) and `wasm32`-friendly.
it carries **no** linear-algebra dependency, so it can be depended on for just the
losses and theory.

**Most users want the [`robust-rs`] crate instead**, which layers the location,
regression (M/S/MM/LTS/Theil–Sen) and multivariate (MCD/OGK/M-scatter/Tyler)
estimators on top of this core.

```rust
use robust_rs_core::rho::{Huber, RhoFunction};
use robust_rs_core::theory::gaussian_efficiency;

let huber = Huber::default();          // k = 1.345
assert_eq!(huber.psi(10.0), 1.345);    // clipped score: bounded influence
assert!((gaussian_efficiency(&huber, 128) - 0.95).abs() < 0.01);
```

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.

[`robust-rs`]: https://crates.io/crates/robust-rs
