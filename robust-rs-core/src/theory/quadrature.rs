//! Gauss–Hermite quadrature for expectations against the standard normal.

/// Nodes and weights approximating `∫ f(x) φ(x) dx` with `n` points, where `φ`
/// is the standard normal density (probabilists' convention).
pub fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 1, "gauss_hermite requires n >= 1");

    // Golub–Welsch on the monic *probabilists'* Hermite recurrence
    //   He_{k+1}(x) = x He_k(x) - k He_{k-1}(x),
    // whose Jacobi matrix is symmetric tridiagonal with
    //   diagonal   a_k = 0,
    //   off-diag   b_k = sqrt(k)   (k = 1..n-1).
    // Nodes are its eigenvalues; the weight at node i is mu_0 * v_i[0]^2 where
    // v_i is the unit eigenvector and mu_0 = ∫ e^{-x^2/2} dx = sqrt(2π). That
    // sqrt(2π) cancels the 1/sqrt(2π) in φ, so the returned weights sum to 1.

    // d: diagonal (-> eigenvalues). e[i] is the off-diagonal coupling d[i] and
    // d[i+1]; e[n-1] is a zero sentinel.
    let mut d = vec![0.0_f64; n];
    let mut e = vec![0.0_f64; n];
    for i in 1..n {
        e[i - 1] = (i as f64).sqrt();
    }
    // First row of the accumulated eigenvector matrix (starts as identity).
    let mut z = vec![0.0_f64; n];
    z[0] = 1.0;

    // Implicit-shift QL for a symmetric tridiagonal matrix
    // (port of EISPACK imtql2 / Numerical Recipes `tqli`, first row of Z only).
    for l in 0..n {
        let mut iter = 0usize;
        loop {
            // Find the first negligible subdiagonal at or below l.
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() + dd == dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break; // eigenvalue l isolated
            }
            iter += 1;
            assert!(iter <= 50, "gauss_hermite: QL failed to converge");

            // Wilkinson-style shift from the leading 2x2.
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let mut s = 1.0_f64;
            let mut c = 1.0_f64;
            let mut p = 0.0_f64;

            let mut cancelled = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    cancelled = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                let zf = z[i + 1];
                z[i + 1] = s * z[i] + c * zf;
                z[i] = c * z[i] - s * zf;
            }
            if cancelled {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }

    // nodes = eigenvalues, weights = (first eigenvector component)^2, sorted.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap());
    let nodes = idx.iter().map(|&i| d[i]).collect();
    let weights = idx.iter().map(|&i| z[i] * z[i]).collect();
    (nodes, weights)
}
