//! Convergence control for iterative solvers.

/// Stopping rule for iterative solvers.
#[derive(Debug, Clone, Copy)]
pub struct Control {
    /// Relative tolerance on the parameter change between iterations.
    pub tol: f64,
    /// Maximum number of iterations before returning `NonConvergence`.
    pub max_iter: usize,
}

impl Default for Control {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            max_iter: 100,
        }
    }
}
