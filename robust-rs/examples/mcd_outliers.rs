//! Multivariate outlier detection on Brownlee's stack-loss data.
//!
//! The three operating variables (air flow, water temperature, acid
//! concentration) are treated as a 3-variate sample. The classical mean and
//! covariance are dragged by the well-known influential runs, masking them; a
//! robust MCD location/scatter gives each observation a robust Mahalanobis
//! distance and the `χ²_{3, 0.975}` cutoff flags the outliers that the classical
//! distances hide. OGK reaches the same verdict deterministically and far
//! cheaper.
//!
//! Run with: `cargo run -p robust-rs --example mcd_outliers`

use robust_rs::datasets::stackloss;
use robust_rs::multivariate::mahalanobis::{classical_covariance, mahalanobis_distances};
use robust_rs::multivariate::{Mcd, Ogk, RobustScatter};

fn main() {
    // Use the three predictor columns as a multivariate sample.
    let (x, _y) = stackloss();
    let (n, p) = x.dim();
    println!("stack-loss operating variables: {n} observations, {p} variables\n");

    // Classical (non-robust) mean/covariance and its Mahalanobis distances.
    let (cmean, ccov) = classical_covariance(&x);
    let cdist = mahalanobis_distances(&x, &cmean, &ccov).unwrap();

    // Robust MCD (reweighted) and OGK.
    let mcd = Mcd::new().seed(1).fit(&x).unwrap();
    let ogk = Ogk::default().fit(&x).unwrap();

    println!("classical mean : {:.3}", cmean);
    println!("MCD location   : {:.3}", mcd.location());
    println!("OGK location   : {:.3}\n", ogk.location());

    let cut = mcd.distance_cutoff(0.975);
    println!("χ²_{{{p},0.975}} distance cutoff = {cut:.3}\n");

    println!(" obs | classical d | MCD d  | OGK d  | MCD flag");
    println!("-----+-------------+--------+--------+---------");
    let mcd_flags = mcd.outliers(0.975);
    for i in 0..n {
        println!(
            " {:>3} | {:>11.2} | {:>6.2} | {:>6.2} |   {}",
            i + 1,
            cdist[i],
            mcd.distances()[i],
            ogk.distances()[i],
            if mcd_flags[i] { "OUT" } else { "." }
        );
    }

    let flagged: Vec<usize> = mcd_flags
        .iter()
        .enumerate()
        .filter(|&(_, &f)| f)
        .map(|(i, _)| i + 1)
        .collect();
    println!("\nMCD flags observations (1-based): {flagged:?}");
    println!("MCD breakdown point: {:.3}", mcd.breakdown_point());
}
