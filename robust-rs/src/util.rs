//! Small internal numeric helpers shared across estimators.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Derive an independent [`ChaCha8Rng`] for `index` from a master seed via a
/// splitmix64 mix, so each draw depends only on `(master_seed, index)`, never on
/// iteration order or thread count. Shared by the FAST-resampling regression
/// search ([`crate::regression`]) and the FAST-MCD multivariate search
/// ([`crate::multivariate`]), so a parallel (rayon) fan-out over starts produces
/// the same result as a serial one.
pub(crate) fn substream(master_seed: u64, index: u64) -> ChaCha8Rng {
    let mut z = master_seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ChaCha8Rng::seed_from_u64(z)
}

/// Median of a slice via an in-place total-order sort: for even length, the
/// average of the two middle order statistics, the convention the crate's
/// consistency constants (e.g. the MAD's `1.4826`) are derived for. `total_cmp`
/// never panics on NaN.
///
/// Panics on an empty slice; every caller guards non-emptiness first.
pub(crate) fn median(v: &mut [f64]) -> f64 {
    v.sort_unstable_by(f64::total_cmp);
    let n = v.len();
    let mid = n / 2;
    if n % 2 == 1 {
        v[mid]
    } else {
        0.5 * (v[mid - 1] + v[mid])
    }
}
