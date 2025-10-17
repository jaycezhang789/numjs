/// Canonical error codes emitted by the Rust core. The codes are designed to be
/// stable across backends so that higher layers can provide predictable error
/// handling.
pub mod codes {
    /// Shapes, strides, or other size-related constraints were violated.
    pub const SHAPE_MISMATCH: &str = "E_SHAPE_MISMATCH";
    /// Numeric invariants were broken (overflow, NaN/Inf in a context that forbids them).
    pub const NUMERIC_ISSUE: &str = "E_NUMERIC_ISSUE";
    /// Cholesky factorisation failed because the matrix is not symmetric positive definite.
    pub const CHOLESKY_NOT_SPD: &str = "E_CHOLESKY_NOT_SPD";
}

/// Helper that formats a code/message pair into a single string. The format is
/// intentionally simple so that foreign-language bindings can either show the
/// message directly or split on the first colon to recover the code.
pub fn format(code: &str, message: impl AsRef<str>) -> String {
    format!("{code}: {}", message.as_ref())
}

/// Convenience function for shape mismatch style errors.
pub fn shape_mismatch(message: impl AsRef<str>) -> String {
    format(codes::SHAPE_MISMATCH, message)
}

/// Convenience function for numeric issues such as overflow/NaN/Inf.
pub fn numeric_issue(message: impl AsRef<str>) -> String {
    format(codes::NUMERIC_ISSUE, message)
}

/// Convenience function for non-SPD Cholesky failures.
pub fn cholesky_not_spd(message: impl AsRef<str>) -> String {
    format(codes::CHOLESKY_NOT_SPD, message)
}
