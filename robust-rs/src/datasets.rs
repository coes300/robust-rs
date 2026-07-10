//! Classic robust-statistics datasets, embedded for tests and examples.
//!
//! Values are sourced from R's `robustbase`; see `datasets/data/*.csv`.

use ndarray::{Array1, Array2};

/// Brownlee's stack-loss data: 21 observations, predictors (air flow, water
/// temperature, acid concentration) and the response (stack loss). Returns
/// `(X, y)` with `X` shaped `21 × 3`.
pub fn stackloss() -> (Array2<f64>, Array1<f64>) {
    parse_csv(include_str!("datasets/data/stackloss.csv"), 3)
}

/// Hertzsprung–Russell diagram of star cluster CYG OB1: 47 stars, predictor
/// `log.Te`, response `log.light`. Cases 11, 20, 30, 34 are giant-star
/// outliers. Returns `(X, y)` with `X` shaped `47 × 1`.
pub fn stars_cyg() -> (Array2<f64>, Array1<f64>) {
    parse_csv(include_str!("datasets/data/starsCYG.csv"), 1)
}

/// Parse an embedded, header-carrying CSV into a design matrix of the first
/// `n_features` columns and a response vector from the final column. Panics on
/// malformed data, which for a compiled-in dataset is a build-time bug.
fn parse_csv(text: &str, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let rows: Vec<Vec<f64>> = text
        .lines()
        .skip(1)
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| {
            l.split(',')
                .map(|f| {
                    f.trim()
                        .parse::<f64>()
                        .expect("dataset field is not numeric")
                })
                .collect()
        })
        .collect();

    let n = rows.len();
    let ncol = n_features + 1;
    assert!(
        rows.iter().all(|r| r.len() == ncol),
        "dataset row width mismatch: expected {ncol} columns"
    );

    let mut x = Array2::zeros((n, n_features));
    let mut y = Array1::zeros(n);
    for (i, row) in rows.iter().enumerate() {
        for j in 0..n_features {
            x[[i, j]] = row[j];
        }
        y[i] = row[n_features];
    }
    (x, y)
}
