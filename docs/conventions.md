# Conventions

Every consistency constant, default tuning and numerical convention the crate
follows.

## Residual standardization

Losses act on the **standardized** residual `u = r / s`. `weight(r) = ψ(r)/r`
uses the removable-singularity limit `ψ'(0)` at `r = 0`; the crate's convention
is `ψ'(0) = 1` for every smooth loss (Huber, L2, Tukey, Cauchy, Welsch andrews,
Hampel), so `weight(0) = 1`. New losses are parameterized to preserve this
(e.g. Andrews uses `ψ(r) = c·sin(r/c)`, not `sin(r/c)`, so that `ψ'(0) = 1`).

`L1` also reports `weight(0) = 1`, but as a **chosen finite cap, not a limit**:
its true `ψ(r)/r = 1/|r|` diverges to `+∞` at the origin (there is no `ψ'(0)`;
`ψ = sign` has a kink). Capping it at `1` keeps the weight finite and IRLS
stable, so a residual lying exactly on the fit is kept rather than dropped;
`ψ(0) = 0` is unchanged, so `weight(0)·0 = ψ(0)` still holds. It is a convention,
not a limit: the invariant `weight(r)·r = ψ(r)` is blind at `r = 0` (`w·0 = 0` for
any finite `w`), so it neither confirms nor constrains the value chosen there.

## Loss functions: default tuning and Gaussian efficiency

The redescender tunings below are the standard "95% efficient M-step" values;
verified against the `theory` module in `tests/theory_efficiency.rs`.

| Loss            | Default tuning       | Gaussian efficiency | Bounded ρ (`rho_sup`) | Redescending |
|-----------------|----------------------|---------------------|-----------------------|--------------|
| `LeastSquares`  | –                    | 1.0000              | no                    | no           |
| `L1`            | –                    | (median)            | no                    | no           |
| `Huber`         | `k = 1.345`          | ≈ 0.950             | no                    | no           |
| `TukeyBiweight` | `c = 4.685`          | ≈ 0.950             | `c²/6`                | yes          |
| `Cauchy`        | `c = 2.3849`         | ≈ 0.950             | **no** (soft)         | yes          |
| `Welsch`        | `c = 2.9846`         | ≈ 0.950             | `c²/2`                | yes          |
| `Andrews`       | `c = 1.339`          | ≈ 0.950             | `2c²`                 | yes          |
| `Hampel`        | `(a,b,c) = (2,4,8)`  | ≈ 0.990             | `(a/2)(b+c−a)`        | yes          |

- **S-step biweight:** `c ≈ 1.547` (with `δ = 0.5`) gives a 50%-breakdown
  S-scale; at this `c`, `E_Φ[ρ] = ρ_sup/2`, so it is simultaneously consistent
  and 50%-breakdown. `TukeyBiweight::default()` (`c = 4.685`) is the M-step.
- **Cauchy** is a *soft* redescender: `ψ → 0` but `ρ` is unbounded, so
  `rho_sup() = None` and it cannot seed an S-scale.
- **Hampel** carries three break-points `a ≤ b < c`; the trait's scalar
  `tuning()` returns `a` and `Hampel::constants()` returns `(a, b, c)`.

## Scale estimators

| Estimator        | Formula / constant                                              | Consistency constant | Efficiency | Breakdown |
|------------------|-----------------------------------------------------------------|----------------------|------------|-----------|
| `Mad`            | `1.4826 · median\|rᵢ − median r\|`, `1.4826 = 1/Φ⁻¹(¾)`         | `1.482602218505602`  | ≈ 0.37     | 0.5       |
| `HuberProposal2` | joint `(μ,s)`; `(1/n)Σψ((rᵢ−μ)/s)² = β`, `β = E_Φ[ψ²]`          | β in **closed form** | –          | –         |
| `SScale`         | `s` solving `(1/n)Σ ρ(rᵢ/s) = δ`                                | `δ = E_Φ[ρ]`         | (loss)     | up to 0.5 |
| `Qn`             | `c·dₙ·{\|rᵢ−rⱼ\|; i<j}₍ₖ₎`, `k = C(⌊n/2⌋+1, 2)`                 | `2.2219 = 1/(√2·Φ⁻¹(⅝))` | ≈ 0.82 | 0.5       |
| `Sn`             | `c·dₙ·med_i med_j \|rᵢ−rⱼ\|`                                    | `1.1926`             | ≈ 0.58     | 0.5       |

- **Huber Proposal 2 `β`** is computed in closed form, *not* by Gauss–Hermite:
  `ψ²` has a kink at `±k` that makes GH converge at only `O(1/n)` (~0.3% error
  at 128 nodes), which would bias every scale. See `scale/huber_proposal2.rs`.
- **`Qn`/`Sn` finite-sample corrections `dₙ`** (Croux & Rousseeuw 1992):
  tabulated for `n ≤ 9`; for larger `n`, `Qn` uses `n/(n+3.8)` (even) or
  `n/(n+1.4)` (odd) and `Sn` uses `n/(n−0.9)` (even) or `1` (odd). Toggle with
  the `finite_sample_correction` field. Both are currently `O(n²)`; the
  `O(n log n)` algorithms are a possible follow-up.
- **`SScale`** requires a **bounded** loss (`rho_sup().is_some()`; an unbounded ρ
  yields a zero-breakdown "scale", so `SScale::new` rejects it with
  `RobustError::UnboundedLoss`; this cleanly excludes the soft Cauchy). It is
  solved by the fixed-point iteration `s ← s·√(mean ρ(rᵢ/s)/δ)` from a MAD start;
  `g(s)` is monotone decreasing so the root is unique.

## Multivariate location/scatter (`multivariate` module)

Every estimator produces a **location–scatter pair** `(μ̂, Σ̂)`; robust
Mahalanobis distances `dᵢ = √((xᵢ − μ̂)ᵀ Σ̂⁻¹ (xᵢ − μ̂))` and the outlier map are
built on it (`ScatterFit` / `McdFit` implement the `RobustScatter` trait).
`Tyler` is the exception: it identifies *shape* only, so it returns a bespoke
`TylerFit` whose distances are **not** `χ²`-calibrated; it implements no shared
trait and has no default χ² outlier map, only the explicitly-named opt-in
`TylerFit::outliers_assuming_chi2_radial`.

| Estimator  | Kind                              | Equivariance          | Breakdown              | Consistency device |
|------------|-----------------------------------|-----------------------|------------------------|--------------------|
| `Mcd`      | min-covariance-determinant subset | **affine** (exact)    | `(n − h + 1)/n` (≈0.5) | consistency factor + RMCD reweighting |
| `Ogk`      | pairwise GK, orthogonalized       | scaling + permutation | moderate               | median-adjusted reweighting + consistency factor |
| `MScatter` | monotone M-estimator (Maronna)    | **affine** (exact)    | ≈ `1/(p+1)` (low)      | median-calibrated scale |
| `Tyler`    | distribution-free M of *shape*    | affine (up to scale)  | ≈ `1/p`                | unit determinant (no scale) |

- **Outlier cutoff.** A point is flagged when `dᵢ > √χ²_{p, q}` (default
  `q = 0.975`): under a `p`-variate Gaussian the squared robust distances are
  `χ²_p`. `RobustScatter::outliers(q)` and `mahalanobis::outlier_flags` share this
  convention.
- **χ² CDF / quantile** are computed from the regularized incomplete gamma
  `P(k/2, x/2)` via `libm::lgamma` (series + Lentz continued fraction; quantile by
  safeguarded Newton), the same dependency-light choice as the `erf`-based normal
  CDF, *not* `statrs`. Pinned against `qchisq` tables in `multivariate/chi2.rs`.
- **MCD.** Coverage defaults to `h = ⌊(n + p + 1)/2⌋` (max breakdown); `.coverage(f)`
  trades breakdown for efficiency. The **C-step** (recompute `(μ,Σ)` on the `h`
  smallest-Mahalanobis points) provably never increases `det Σ`, the exact
  multivariate analogue of the FAST-LTS C-step. The raw min-determinant covariance
  is scaled by the **consistency factor** `c(α, p) = α / F_{χ²_{p+2}}(χ²_{p,α})`,
  `α = h/n` (Croux & Haesbroeck 1999); the default estimate is the **reweighted**
  MCD (RMCD), recomputed on the points within the `χ²_{p,0.975}` cutoff. The
  additional small-sample multiplier R's `covMcd` applies (Pison et al. 2002) is a
  documented **deferral**; only the asymptotic factor is applied.
- **OGK** reweights with the **median-adjusted** cutoff `med(d²)·χ²_{p,0.9}/χ²_{p,0.5}`
  (Maronna & Zamar 2002), which self-calibrates to the raw scale rather than
  reverting to the classical covariance; two orthogonalization iterations by
  default; robust scale defaults to `Qn`.
- **Reproducibility.** `Mcd` threads the same version-stable `ChaCha8Rng`
  sub-stream scheme as the regression S/LTS estimators (reproducible by default,
  `.seed(u64)` / `fit_with_rng`, thread-count invariant so `rayon` over the random
  starts does not change the answer).

## Theory (`theory` module)

- Expectations against `Φ` use **Gauss–Hermite quadrature** (Golub–Welsch on the
  probabilists' Hermite recurrence); default `QUAD = 128` nodes.
- `E_Φ[ψ']` is evaluated via **Stein's identity** `E_Φ[ψ'(X)] = E_Φ[X·ψ(X)]`,
  which stays correct for clipped/kinked `ψ` where integrating `psi_prime`
  directly would not (e.g. L1, where `ψ' ≡ 0` a.e. yet the right denominator is
  `E[|X|] = √(2/π)`).
- Asymptotic variance `V = E[ψ²]/(E[ψ'])²`; Gaussian efficiency `= 1/V`.
- The standard normal CDF (used for the Proposal-2 `β`) is a dependency-light
  `erf` via `libm`, not `statrs`.

## Reproducibility (randomized estimators, v0.2 regression)

S / MM / LTS thread a **version-stable `ChaCha8Rng`**, reproducible by default,
with a `.seed(u64)` builder and a `fit_with_rng(&mut impl Rng)` escape hatch;
parallel subsamples draw independent sub-streams derived from the master seed so
thread count never changes the result.
