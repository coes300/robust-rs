# robustat-py

Robust statistics for Python: Rust-powered bindings to
[`robust-rs`](https://docs.rs/robust-rs). M/S/MM regression, robust scale,
robust multivariate location–scatter (MCD, OGK, Tyler) and the
influence-function sampling theory, all with a numpy interface.

The numerics run in compiled Rust (via [PyO3](https://pyo3.rs) and
[rust-numpy](https://github.com/PyO3/rust-numpy)); randomized estimators
(S/MM/LTS/MCD) are reproducible by default and expose a `seed`.

## Install

```console
pip install robustat-py
```

### From source (this repo)

Requires a Rust toolchain and [maturin](https://www.maturin.rs):

```console
pip install maturin
maturin develop -m py/Cargo.toml --features extension-module   # into the active venv
# or build a wheel:
maturin build  -m py/Cargo.toml --release
```

## Quick start

```python
import numpy as np
import robustat_py as rp

# --- Robust regression -------------------------------------------------------
# starsCYG: four giant stars flip the OLS slope negative; MM recovers the
# physical positive slope and rejects them (R's lmrob default).
x_raw, y = rp.datasets.stars_cyg()
X = np.column_stack([np.ones(len(y)), x_raw[:, 0]])   # prepend an intercept

mm = rp.MMEstimator(seed=1).fit(X, y)     # 50% breakdown AND ~95% efficiency
mm.coefficients                           # -> np.ndarray; slope > 0
mm.gaussian_efficiency()                  # ~0.95
mm.breakdown_point                        # 0.5

# Outliers only in y? A plain M-estimator is cheaper (but 0 breakdown vs leverage):
fit = rp.MEstimator(rp.Huber(), rp.Mad()).fit(X, y)
fit.weights                               # near 0 for down-weighted rows

# --- Robust covariance / multivariate outliers -------------------------------
Xs, _ = rp.datasets.stackloss()
mcd = rp.Mcd(seed=1).fit(Xs)              # FAST-MCD, affine equivariant
mcd.outliers(0.975)                       # bool per row, χ² cutoff
mcd.location, mcd.scatter

# --- Robust location & scale -------------------------------------------------
data = [2.1, 2.3, 1.9, 2.0, 2.2, 47.0]    # one gross outlier
rp.m_location(data).estimate              # ~2.16, not the mean (~9.58)
rp.Mad().scale(data)
rp.hodges_lehmann(data).estimate
```

## API at a glance

**Estimators (build, then `.fit`)**

| Class | `.fit` returns | Notes |
|---|---|---|
| `MEstimator(loss, scale)` | `RegressionFit` | IRLS; fast/convex; **0 breakdown** vs leverage |
| `SEstimator(loss, seed=…)` | `RegressionFit` | FAST-S; 50% breakdown |
| `MMEstimator(seed=…)` | `RegressionFit` | **start here**; 50% breakdown + ~95% efficiency |
| `Lts(seed=…)` | `LtsFit` | FAST-LTS; coverage knob |
| `Mcd(seed=…)` | `McdFit` | **start here** for covariance; RMCD by default |
| `Ogk(scale)` | `ScatterFit` | deterministic, positive-definite |
| `MScatter(loss)` | `ScatterFit` | monotone M-scatter (low breakdown) |
| `Tyler()` | `TylerFit` | distribution-free *shape* (unit det) |

**Functions**: `m_location`, `trimmed_mean`, `winsorized_mean`, `hodges_lehmann`,
`theil_sen`, `mahalanobis_distances`, `classical_covariance`, `distance_cutoff`,
`outlier_flags`, `gaussian_efficiency`, `asymptotic_variance`, `breakdown_point`.

**Losses** (redescending ones marked *): `Huber(k=1.345)`, `Tukey(c=4.685)` *,
`Cauchy` *, `Welsch` *, `Andrews` *, `Hampel(a,b,c)` *, `LeastSquares`, `L1`.
Each exposes `rho/psi/weight/psi_prime`, `tuning`, `is_redescending`, `rho_sup`
and `gaussian_efficiency()/asymptotic_variance()/influence()`.

**Scales**: `Mad()`, `Qn()`, `Sn()`, `HuberProposal2()`. Every `loss`/`scale`
argument also accepts a name string (`"huber"`, `"tukey"`, `"mad"`, `"qn"`, …).

**Datasets**: `robustat_py.datasets.stackloss()`, `robustat_py.datasets.stars_cyg()`
(each returns `(X, y)` with predictors only; prepend your own intercept column).

Errors surface as `robustat_py.RobustError` (a `ValueError` subclass).

## License

MIT OR Apache-2.0, matching `robust-rs`.
