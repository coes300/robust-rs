# Theory notes

The influence-function, efficiency and breakdown derivations behind the
`theory` module and the estimators, with the choices that were made and why.

## `E_Φ[ψ']` via Stein's identity

The asymptotic variance is `V(ψ, Φ) = E_Φ[ψ²] / (E_Φ[ψ'])²` and the Gaussian
efficiency is `1/V`. The denominator `E_Φ[ψ']` is **not** evaluated by
integrating `psi_prime` against `Φ`. Instead it uses Stein's identity

```
E_Φ[ψ'(X)] = E_Φ[X · ψ(X)].
```

Both target the same number, but the `X·ψ` integrand is continuous even where
`ψ` is clipped or kinked (Huber at ±k, the biweight at ±c), so Gauss–Hermite
converges fast; and it remains correct when `ψ'` loses mass in the classical
sense. The sharp case is **L1**: `ψ(r) = sign(r)`, so `ψ' ≡ 0` almost everywhere;
integrating `psi_prime` would give `0` and a divide-by-zero, yet the
denominator that belongs there is `E_Φ[|X|] = √(2/π)`, which `E[X·sign(X)]`
delivers exactly. (See `robust-rs-core/src/theory/variance.rs`.)

## Closed-form `β` for Huber's Proposal 2 and why not quadrature

The Proposal-2 scale solves `(1/n) Σ ψ_k((rᵢ−μ)/s)² = β` with the consistency
constant `β = E_Φ[ψ_k(Z)²]`. That `β` is computed in **closed form**, not by
Gauss–Hermite. The reason is a genuine numerical trap: `ψ_k² ` has a kink at
`±k`, so GH quadrature converges only at `O(1/n)` and oscillates (roughly 0.3 %
error even at 128 nodes) which would bias *every* scale the estimator produces.
With `Φ`, `φ` the standard normal CDF/pdf,

```
β = (2Φ(k) − 1 − 2k·φ(k)) + 2k²·(1 − Φ(k)),
```

which is exact. (See `robust-rs-core/src/scale/huber_proposal2.rs`; the normal
CDF is a dependency-light `erf` via `libm`, not `statrs`.)

## The S-scale and the `δ = ρ_sup/2` coincidence

The S-scale is the `s > 0` solving `(1/n) Σ ρ(rᵢ/s) = δ`. Two choices of `δ`
matter and for the biweight they *coincide*:

- **Consistency:** `δ = E_Φ[ρ]` makes `s` Fisher-consistent for σ at the
  Gaussian.
- **Breakdown:** the S-estimator's breakdown point is `min(δ/ρ_sup, 1 − δ/ρ_sup)`,
  maximised at **0.5** when `δ = ρ_sup/2`.

For Tukey's biweight the S-step tuning `c ≈ 1.5476` is (essentially) the special
constant at which `E_Φ[ρ] = ρ_sup/2`, so a single `δ = ρ_sup/2` buys **both**
50 % breakdown *and* (to sub-0.1 %) consistency. The crate exposes both:
`SScale::new(ρ, δ)` and `SScale::fisher_consistent(ρ, quad)`. An unbounded ρ has
no `ρ_sup`, hence no meaningful `δ`, hence zero breakdown, which is why
`SScale::new` rejects it (`RobustError::UnboundedLoss`).

## Why S needs MM: the efficiency purchase

A 50 %-breakdown S-estimate is **inefficient**; the biweight at `c ≈ 1.547` is
only ≈ 29 % Gaussian-efficient (verified: 0.286 on `starsCYG`). This is Yohai's
(1987) premise. MM buys the efficiency back without spending breakdown:

1. **S-step:** a high-breakdown S-estimate gives `(β̃, ŝ)` (50 % breakdown).
2. **M-step:** *fix* the scale at `ŝ` and run IRLS with a high-efficiency
   redescender (biweight `c = 4.685`) **starting from `β̃`**, climbing to the
   nearest local optimum.

The result keeps the S-estimate's 50 % breakdown while reaching ≈ 95 %
efficiency. Fixing the scale is essential; it is what makes the M-step a pure
efficiency refinement rather than a second, competing scale estimate. Because the
fit stores its *final-stage* loss (the M-step biweight), the theory-forward API
then reports the MM efficiency and covariance automatically.

## Breakdown as carried data, not a ρ property

`ρ` alone does not determine an estimator's breakdown (a redescending ρ used as a
plain regression M-estimator still has 0 breakdown; a single high-leverage point
carries the fit). So each fit *carries* its breakdown:

- Regression M-estimation: **0** (leverage).
- S / MM: `min(δ/ρ_sup, 1 − δ/ρ_sup)` = 0.5 at the default `δ`; MM inherits the
  S-stage's value.
- LTS: `(n − h + 1)/n` from the coverage `h`.
- Theil–Sen and Hodges–Lehmann: `1 − 1/√2 ≈ 0.293`.

## Non-M-estimators have efficiency, just not a ρ-derivable one

LTS, Theil–Sen and Hodges–Lehmann are √n-consistent and asymptotically normal, so
they *do* have well-defined influence functions and Gaussian efficiencies, but
those come from the asymptotics of their (trimmed / rank) objectives, not from
`ψ(r)/E[ψ']` and `1/V(ψ)`. The crate therefore keeps them off the ρ-based
`RobustEstimator` trait rather than fabricate a value through it. Where the
number is a settled constant it is reported directly: **Hodges–Lehmann** exposes
the Wilcoxon ARE `3/π ≈ 0.955`. LTS's efficiency is low (single digits, below
S's ≈ 29 %), which is, again, exactly why MM refines it and Theil–Sen's is
omitted here rather than quoted from memory.

## Multivariate: the C-step, the consistency factor and equivariance

The MCD C-step is the multivariate C-step of LTS. LTS refits OLS on the `h`
smallest-residual points; MCD recomputes `(μ, Σ)` on the `h` smallest-Mahalanobis
points. Rousseeuw & Van Driessen (1999) prove the C-step never increases `det Σ`,
the same monotone-descent guarantee that makes FAST-LTS's trimmed sum of
squares non-increasing, so both are correct concentration steps and both plug
into the identical "many random elemental starts → pre-refine → concentrate the
best few → keep the global optimum" search. In this crate they even share the RNG
sub-stream helper (`util::substream`); only the objective (`det Σ` vs the trimmed
`Σ r²`) and the concentration step differ.

The consistency factor is there because the covariance of the `h` innermost points of a
Gaussian is biased *downward* (it is a truncated covariance). The factor
`c(α, p) = α / F_{χ²_{p+2}}(χ²_{p,α})`, `α = h/n` (Croux & Haesbroeck 1999),
is exactly `1 / E[ d² · 1{d² ≤ χ²_{p,α}} ] · α · p`-style truncation correction
that rescales it to be Fisher-consistent for `Σ` at the Gaussian; it tends to `1`
as `α → 1`. It needs both the χ² **quantile** (the truncation radius `χ²_{p,α}`)
and the χ² **CDF at `p + 2`** degrees of freedom (the truncated second moment),
which is why the crate carries a small χ² implementation. R's `covMcd` multiplies
in an additional *finite-sample* factor (Pison et al. 2002) fitted by simulation;
that is deferred here; matching a wrong
finite-sample constant would be worse than shipping the asymptotic one.

Breakdown is carried as data in the multivariate world too. As with regression
(above), `ρ`/the estimator kind alone doesn't fix breakdown, so it is carried:
MCD reports `(n − h + 1)/n` (≈ 0.5 at the default `h`), while the monotone scatter
M-estimator has breakdown only `≈ 1/(p + 1)`; a *single* high-leverage outlier
can carry it, the exact multivariate echo of why a plain regression M-estimate has
0 breakdown and needs MM. That is why MCD (not M-scatter) is the high-breakdown
scatter estimator and why M-scatter's docs point users to MCD for high breakdown.

Tyler is distribution-free; the others are Gaussian-calibrated. Tyler's
estimator depends on the data only through the directions `rᵢ/‖rᵢ‖`, so it is
consistent for the shape of *any* elliptical distribution without a consistency
factor, but it identifies only shape (unit determinant), no scale. MCD, OGK and
M-scatter instead calibrate their scale to the Gaussian (consistency factor /
median calibration), so they estimate a full covariance but assume the bulk is
approximately Gaussian for that calibration to be exact.

Equivariance is the multivariate invariant and it is graded. MCD is exactly
affine equivariant (the C-step selects on affine-invariant Mahalanobis distances
and refits by affine-equivariant sample moments) and so is the M-estimator;
`tests/multivariate.rs` checks `T(XAᵀ + b) = A T(X) Aᵀ` to `< 1e-6`. OGK is the
deliberate exception: the pairwise GK construction is only *orthogonally* (and
coordinatewise-scaling / permutation) equivariant, the price paid for a fast,
positive-definite, subsampling-free estimate, so its test asserts only the
scaling/permutation invariants it actually satisfies.

## Quadrature

Expectations against `Φ` use Gauss–Hermite quadrature built by the Golub–Welsch
algorithm on the probabilists' Hermite recurrence (symmetric tridiagonal Jacobi
matrix; implicit-shift QL for its eigenpairs), rescaled so the weights sum to 1.
The default is 128 nodes (`QUAD` / the `quad_points` arguments). Closed forms are
substituted where a kink would otherwise slow convergence (see Proposal 2 above).
